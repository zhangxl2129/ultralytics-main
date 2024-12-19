import os
import shutil

import torch

from ultralytics import YOLO


class YOLOTrain:
    """
    YOLO训练类：封装YOLO训练逻辑
    """
    def __init__(self, dataset_path, model_save_path):
        self.dataset_path = os.path.abspath(dataset_path)

        self.model_save_path = os.path.abspath(model_save_path)
        self.data_yaml_path = os.path.join(self.dataset_path, "data.yaml")

        # 确定YOLO配置文件和设备
        self.model_config = "ultralytics/cfg/models/11/yolo11-WTConv-SPDConv.yaml"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def validate_paths(self):
        """
        验证数据路径和配置文件
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        if not os.path.exists(self.data_yaml_path):
            raise FileNotFoundError(f"data.yaml not found in {self.dataset_path}")

    def train(self):
        """
        执行YOLO训练并返回最佳模型路径
        """
        self.validate_paths()

        print(f"Starting training with data: {self.data_yaml_path}")
        print(f"Using device: {self.device}")

        # 初始化模型
        model = YOLO(self.model_config)
        model.train(
            #data='C:\\Users\Administrator\PycharmProjects\\flaskProject2\data\\train\\NEU-DTC\data.yaml',
            data=self.data_yaml_path,
            imgsz=512,
            epochs=200,
            batch=32,
            optimizer="SGD",
            lr0=0.01,
            lrf=0.2,
            weight_decay=5e-4,
            device=self.device,
            pretrained="yolov8s.pt"  # 加载预训练权重
        )

        # 保存最佳模型
        best_model_src = "runs/detect/train/weights/best.pt"
        best_model_dst = os.path.join(self.model_save_path, "best.pt")
        os.makedirs(self.model_save_path, exist_ok=True)

        if os.path.exists(best_model_src):
            shutil.copy(best_model_src, best_model_dst)
            print(f"Best model saved to: {best_model_dst}")
            return best_model_dst
        else:
            raise FileNotFoundError("Best model file not found after training.")