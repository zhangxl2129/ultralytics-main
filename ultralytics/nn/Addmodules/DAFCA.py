import torch
import torch.nn as nn
import torch.nn.functional as F


class DAFCA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(DAFCA, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # 1. 通道注意力机制
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 2. 空间注意力机制
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Sigmoid()
        )

        # 3. 坐标注意力机制
        self.coord_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_weights = self.channel_att(x)
        x = x * channel_weights

        # 空间注意力
        spatial_weights = self.spatial_att(x)
        x = x * spatial_weights

        # 坐标注意力
        coord_weights = self.coord_att(x)
        x = x * coord_weights

        return x
