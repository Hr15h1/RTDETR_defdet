from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo



__all__ = ['SqueezeNet', 'squeezenet1_1', 'SqueezeNetBackbone']


model_urls = {
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth',
}


class Fire(nn.Module):
    """
    The Fire module, which is the building block of SqueezeNet.
    """
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        # Squeeze layer is a 1x1 convolution
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        # Expand layer has a mix of 1x1 and 3x3 convolutions
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        # The output of the expand layer is the concatenation of the 1x1 and 3x3 convolutions
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):
    """
    SqueezeNet base model. This version is adapted to return intermediate
    feature maps for object detection, similar to the DLA backbone.
    """
    def __init__(self, version='1_1', out_indices=(0, 1, 2)):
        super(SqueezeNet, self).__init__()
        self.out_indices = out_indices
        
        if version == '1_1':
            # SqueezeNet v1.1 architecture
            self.features = nn.ModuleList([
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=0), # 0
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # 2
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64), # 4
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # 5
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128), # 7
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # 8
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192), # 10
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256), # 12
            ])
            # These are the layers from which we will extract features for the neck
            self.feature_extraction_layers = [4, 7, 12]
        else:
            raise ValueError("Unsupported SqueezeNet version {}. Only '1_1' is supported.".format(version))

    def forward(self, x):
        outputs = []
        # We iterate through the feature modules and save the output of the desired layers
        for i, module in enumerate(self.features):
            x = module(x)
            if i in self.feature_extraction_layers:
                # Check if the current layer index is one of the requested output stages
                if len(outputs) in self.out_indices:
                    outputs.append(x)
        return outputs

    def load_pretrained_model(self, model_name='squeezenet1_1'):
        if model_name in model_urls:
            state_dict = model_zoo.load_url(model_urls[model_name])
            # The pretrained model has a 'features' and 'classifier' key. We only need 'features'.
            # We also need to adapt the keys to match我们的 ModuleList structure.
            # PyTorch's SqueezeNet saves features as a Sequential module, so keys are like 'features.0.weight'
            # We need to map them to our ModuleList, which has keys like '0.weight'
            model_dict = self.state_dict()
            for k in state_dict:
                if k.startswith('features.'):
                    new_k = k[len('features.'):]
                    if new_k in model_dict:
                        model_dict[new_k] = state_dict[k]
            self.load_state_dict(model_dict)
            print(f"Loaded pretrained weights for {model_name}")
        else:
            print(f"No pretrained weights found for {model_name}")


def squeezenet1_1(pretrained=True, **kwargs):
    """
    Instantiates a SqueezeNet v1.1 model.
    """
    model = SqueezeNet(version='1_1', **kwargs)
    if pretrained:
        model.load_pretrained_model('squeezenet1_1')
    return model



class SqueezeNetBackbone(nn.Module):
    """
    Wrapper for the SqueezeNet backbone to be used in RT-DETR.
    This class is registered and can be configured in the main model's YAML file.
    """
    def __init__(
        self,
        model_name='squeezenet1_1',
        pretrained=True,
        return_indices=[0, 1, 2],
    ):
        super(SqueezeNetBackbone, self).__init__()
        # The SqueezeNet model is instantiated with the desired output feature maps
        self.model = squeezenet1_1(pretrained=pretrained, out_indices=return_indices)
        
        # The number of channels for the returned feature maps from SqueezeNet v1.1
        # These correspond to the outputs of Fire modules 4, 7, and 12
        self._out_channels = [128, 256, 512]
        self.return_indices = return_indices

    @property
    def channels(self):
        """
        Returns the number of channels for the feature maps that will be passed to the neck.
        """
        return [self._out_channels[i] for i in self.return_indices]

    def forward(self, x):
        """
        Runs the forward pass and returns the selected multi-scale feature maps.
        """
        return self.model(x)
