import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        batch_size, channel, _, _ = x.size()
        y = nn.functional.adaptive_avg_pool2d(x, 1).view(batch_size, channel)
        y = nn.functional.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channel, 1, 1)
        return x * y.expand_as(x)

class SEConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SEConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return self.se(x)
