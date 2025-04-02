import torch
import torch.nn as nn
from torchvision.models import resnet34
from torchvision.models._utils import IntermediateLayerGetter

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetResNet34(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        backbone = resnet34(pretrained=True)

        # Extract layers for skip connections
        self.encoder = IntermediateLayerGetter(
            backbone,
            return_layers={
                "relu": "layer0",     # After initial conv + bn + relu
                "layer1": "layer1",   # Down 1
                "layer2": "layer2",   # Down 2
                "layer3": "layer3",   # Down 3
                "layer4": "layer4",   # Down 4
            }
        )

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 32)

        # Final decoder (no skip connection)
        self.decoder0 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        enc = self.encoder(x)

        x0 = enc["layer0"]  # 64 x H/2
        x1 = enc["layer1"]  # 64 x H/4
        x2 = enc["layer2"]  # 128 x H/8
        x3 = enc["layer3"]  # 256 x H/16
        x4 = enc["layer4"]  # 512 x H/32

        center = self.center(x4)
        d4 = self.decoder4(center, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, x0)
        d0 = self.decoder0(d1)

        return self.final_conv(d0)

