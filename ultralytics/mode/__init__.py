import torch.nn as nn
from ultralytics import YOLO

class SPDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SPDConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class CustomYOLO(YOLO):
    """自定义 YOLO 模型类，可以在这里添加额外的修改或功能。"""

    def __init__(self, model="yolov10n.pt", task=None, verbose=False):
        """初始化自定义 YOLO 模型，并替换 backbone 和数据增强参数。"""
        super().__init__(model=model, task=task, verbose=verbose)

        # 定义自定义 backbone 结构，使用 SPDConv
        self.model.model.backbone = self.create_custom_backbone()

        # 打印修改后的主干网络结构
        print("Modified Backbone Structure:")
        for layer in self.model.model.backbone:
            print(layer)

        # 打印完整模型结构
        print("Current Model Structure After Modification:")
        print(self.model.model)

    def create_custom_backbone(self):
        """创建自定义 backbone 结构并返回。"""
        return [
            [-1, 1, SPDConv, [64, 3, 2]],  # 使用 SPDConv
            [-1, 1, SPDConv, [128, 3, 2]],  # 使用 SPDConv
            [-1, 1, SPDConv, [256, 1, 1]],  # 新增 SPDConv 层
            [-1, 3, 'C2f', [128, True]],
            [-1, 1, SPDConv, [256, 3, 2]],  # 使用 SPDConv
            [-1, 6, 'C2f', [256, True]],
            [-1, 1, 'SCDown', [512, 3, 2]],  # 5-P4/16
            [-1, 6, 'C2f', [512, True]],
            [-1, 1, 'SCDown', [1024, 3, 2]],  # 7-P5/32
            [-1, 3, 'C3', [1024, True]],
            [-1, 1, 'SPPF', [1024, 5]],  # 9
            [-1, 1, 'PSA', [1024]],  # 10
        ]

augment_params = {
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

train_params = {
    "cache": 'disk',
    "imgsz": 512,
    "epochs": 1,
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
#当你调用 `train` 时，可以传递这些参数
#model.train(data='path/to/dataset', augment=augment_params, **train_params)



# from pathlib import Path
# from ultralytics import YOLO  # 从 ultralytics 引入 YOLO
#
# class CustomYOLO(YOLO):
#     """自定义 YOLO 模型类，可以在这里添加额外的修改或功能。"""
#
#     def __init__(self, model="yolov10n.pt", task=None, verbose=False):
#         """初始化自定义 YOLO 模型，并替换 backbone。"""
#         super().__init__(model=model, task=task, verbose=verbose)
#
#         # 定义自定义 backbone 结构
#         new_backbone = [
#             [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
#             [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
#             [-1, 1, 'Conv', [256, 1, 1]],  # 新增 Conv 层，输出 256 通道
#             [-1, 3, 'C2f', [128, True]],
#             [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
#             [-1, 6, 'C2f', [256, True]],
#             [-1, 1, 'SCDown', [512, 3, 2]],  # 5-P4/16
#             [-1, 6, 'C2f', [512, True]],
#             [-1, 1, 'SCDown', [1024, 3, 2]],  # 7-P5/32
#             [-1, 3, 'C3', [1024, True]],
#             [-1, 1, 'SPPF', [1024, 5]],  # 9
#             [-1, 1, 'PSA', [1024]],  # 10
#         ]
#
#         # 应用新的 backbone 配置
#         self.model.model.backbone = new_backbone
#
# # 使用自定义模型
# model = CustomYOLO('yolov10n.pt')
#
# # 训练时定义数据增强参数和超参数
# augment_params = {
#     "degrees": 5.0,
#     "translate": 0.1,
#     "shear": 1.0,
#     "scale": [0.5, 1.5],
#     "mosaic": 1.0,
#     "random_perspective": 0.5,
#     "hsv_h": 0.015,
#     "hsv_s": 0.7,
#     "hsv_v": 0.4,
#     "cutout": 0.1
# }
#
# train_params = {
#     "cache": 'disk',
#     "imgsz": 512,
#     "epochs": 300,
#     "batch": 32,
#     "close_mosaic": 64,
#     "workers": 16,
#     "patience": 30,
#     "optimizer": 'SGD',
#     "lr0": 0.01,  # 初始学习率
#     "lrf": 0.2,    # 最终学习率为初始学习率的 20%
#     "momentum": 0.937,  # 动量设置
#     "weight_decay": 5e-4  # 权重衰减
# }
#
# #当你调用 `train` 时，可以传递这些参数
# model.train(data='path/to/dataset', augment=augment_params, **train_params)
#
