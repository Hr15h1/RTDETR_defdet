import torch
import torch.nn as nn
from ultralytics import RTDETR
from squeezenet_backbone import SqueezeNetBackbone
from ultralytics.nn.modules.head import RTDETRDecoder

from ultralytics.utils.torch_utils import get_num_params


class SmallAIFI(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels  # default: identity mapping

        self.input_proj = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, 1, bias=False) for c in in_channels
        ])
        self.input_bn = nn.ModuleList([nn.BatchNorm2d(hidden_dim) for _ in in_channels])
        self.act = nn.ReLU(inplace=True)

        # lightweight attention-like fusion
        self.fuse_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False) for _ in in_channels
        ])
        self.fuse_bns = nn.ModuleList([nn.BatchNorm2d(hidden_dim) for _ in in_channels])

        # project each branch back to expected head channels
        self.output_proj = nn.ModuleList([
            nn.Conv2d(hidden_dim, out_c, 1, bias=False) for out_c in out_channels
        ])
        self.output_bn = nn.ModuleList([
            nn.BatchNorm2d(out_c) for out_c in out_channels
        ])

    def forward(self, feats):
        outs = []
        for i, x in enumerate(feats):
            x = self.input_proj[i](x)
            x = self.input_bn[i](x)
            x = self.act(x)
            x = self.fuse_bns[i](self.fuse_convs[i](x))
            x = self.act(x)
            x = self.output_bn[i](self.output_proj[i](x))
            outs.append(x)
        return outs


# --- Clean RTDETR wrapper that ignores Ultralytics graph internals ---
class RTDETRCustom(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading base RT-DETR model...")
        base = RTDETR("rtdetr-l.yaml")

        print("Replacing backbone with SqueezeNet...")
        self.backbone = SqueezeNetBackbone(pretrained=True, return_indices=[0, 1, 2])
        new_in_channels = self.backbone.channels

        print("Adding AIFI neck...")
        self.neck = SmallAIFI(in_channels=new_in_channels, hidden_dim=256, out_channels=new_in_channels)


        print("Configuring RTDETRDecoder head...")
        self.head = base.model.model[-1]  # reuse pretrained decoder
        hidden_dim = self.head.hidden_dim
        self.head.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, hidden_dim, 1, bias=False),
                          nn.BatchNorm2d(hidden_dim))
            for c in new_in_channels
        ])
        self.head.in_channels = new_in_channels

        print("✅ Custom RTDETR model initialized successfully.")
        print(f"Model parameters: {get_num_params(self)}")


    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)

