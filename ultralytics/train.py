import warnings
import torch
from ultralytics import YOLO

warnings.filterwarnings('ignore')
if __name__ == '__main__':
    # 指定显卡设备（如 0 表示使用第一块显卡）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = YOLO('cfg/models/11/yolo11.yaml')  # 指定YOLO模型对象，并加载指定配置文件中的模型配置
    #model.load('yolov8s.pt')  # 加载预训练的权重文件'yolov8s.pt'，加速训练并提升模型性能
    model.train(data='C:\\Users\Administrator\PycharmProjects\\flaskProject2\data\\train\\NEU-DTC\data.yaml',  # 指定训练数据集的配置文件路径，这个.yaml文件包含了数据集的路径和类别信息
                cache='disk',
                imgsz=512,
                epochs=1000,
                batch=32,
                close_mosaic=64,
                workers=16,
                #patience=30,
                optimizer='SGD',
                lr0=0.01,  # 初始学习率
                lrf=0.2,  # 最终学习率为初始学习率的 20%
                momentum=0.937,  # 动量设置
                weight_decay=5e-4,  # 权重衰减
                #pretrained='runs/detect/train37/weights/best.pt' , # 加载之前的最佳模型权重
                )
