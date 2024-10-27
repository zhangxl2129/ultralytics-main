from pathlib import Path
from ultralytics import YOLO  # 从 ultralytics 引入 YOLO
import torch
import torch.nn as nn

__all__ = ['SPDConv']


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class SPDConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        c1 = c1 * 4
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.act(self.conv(x))



class self_net(YOLO):
    """自定义 YOLO 模型类，可以在这里添加额外的修改或功能。"""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """初始化自定义 YOLO 模型，并替换 backbone 和数据增强参数。"""
        super().__init__(model=model, task=task, verbose=verbose)

        # 定义自定义 backbone 结构
        new_backbone = [
            [-1, 1, 'Conv', [64, 3, 2]]  # 0-P1/2
            [-1, 1, 'SPDConv', [128]]  # 1-P2/4
            [-1, 2, 'C3k2', [256, False, 0.25]]
            [-1, 1, 'SPDConv', [256]]  # 3-P3/8
            [-1, 2, 'C3k2', [512, False, 0.25]]
            [-1, 1, 'SPDConv', [512]]  # 5-P4/16
            [-1, 2, 'C3k2', [512, True]]
            [-1, 1, 'SPDConv', [1024]]  # 7-P5/32
            [-1, 2, 'C3k2', [1024, True]]
            [-1, 1, 'SPPF', [1024, 5]]  # 9
            [-1, 2, 'C2PSA', [1024]]  # 10
        ]

        # 应用新的 backbone 配置
        self.model.model.backbone = new_backbone

        # 数据增强参数
        self.model.train.augment = {
            "degrees": 5.0,
            "translate": 0.1,
            "shear": 1.0,
            "scale": [0.5, 1.5],
            "mosaic": 1.0,
            "random_perspective": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "cutout": 0.1
        }

        # 超参数设置
        self.model.train.hyperparameters = {
            "cache": 'disk',
            "imgsz": 512,
            "epochs": 300,
            "batch": 32,
            "close_mosaic": 64,
            "workers": 16,
            "patience": 30,
            "optimizer": 'SGD',
            "lr0": 0.01,  # 初始学习率
            "lrf": 0.2,    # 最终学习率为初始学习率的 20%
            "momentum": 0.937,  # 动量设置
            "weight_decay": 5e-4  # 权重衰减
        }

# 使用自定义模型
model = self_net("yolo11n.pt")

