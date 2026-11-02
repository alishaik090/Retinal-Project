"""The original vessel-segmentation architectures used by saved checkpoints."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class VesselUNet(nn.Module):
    """Unchanged compact U-Net retained for compatible historical checkpoints."""

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        channels = base_channels
        self.enc1, self.enc2 = ConvBlock(3, channels), ConvBlock(channels, channels * 2)
        self.enc3, self.enc4 = ConvBlock(channels * 2, channels * 4), ConvBlock(channels * 4, channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(channels * 8, channels * 16)
        self.up4, self.dec4 = nn.ConvTranspose2d(channels * 16, channels * 8, 2, 2), ConvBlock(channels * 16, channels * 8)
        self.up3, self.dec3 = nn.ConvTranspose2d(channels * 8, channels * 4, 2, 2), ConvBlock(channels * 8, channels * 4)
        self.up2, self.dec2 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, 2), ConvBlock(channels * 4, channels * 2)
        self.up1, self.dec1 = nn.ConvTranspose2d(channels * 2, channels, 2, 2), ConvBlock(channels * 2, channels)
        self.out = nn.Conv2d(channels, 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(value)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bottleneck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class ResNet18UNet(nn.Module):
    """Unchanged ResNet-18 U-Net decoder used by the evaluated checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        encoder = resnet18(weights=None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 64, 2, 2), ConvBlock(128, 64)
        self.up0, self.out = nn.ConvTranspose2d(64, 32, 2, 2), nn.Conv2d(32, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = (value - self.mean) / self.std
        stem = self.stem(value)
        e1 = self.layer1(self.maxpool(stem))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), stem], dim=1))
        return self.out(self.up0(d1))


def build_segmenter(architecture: str, base_channels: int = 24) -> nn.Module:
    if architecture == "unet":
        return VesselUNet(base_channels)
    if architecture == "resnet18_unet":
        return ResNet18UNet()
    raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
