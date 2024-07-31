# edsr.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return out

class EDSR(nn.Module):
    def __init__(self, num_channels=3, num_blocks=8, num_features=32, scale_factor=2):
        super(EDSR, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * scale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.residual_blocks(x)
        x = self.conv2(x)
        x += residual
        x = self.upsample(x)
        return x
