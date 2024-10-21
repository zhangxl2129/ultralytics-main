import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 加载最优的模型权重
    best_model = 'runs/detect/train/weights/best.pt'  # 修改为你保存模型的位置
    model = YOLO(best_model)

    # 进行验证，使用真实标签
    val_results = model.val(
        data='E:/Projects/YOLOV10/NEU-Seg/data.yaml',  # 指定数据集配置文件
        save=True,  # 保存检测结果
        conf=0.25,  # 置信度阈值
        imgsz=512,  # 图像大小
        iou=0.5,  # IoU 阈值，默认为0.5，可以调整
        batch=16,  # 验证批次大小
        device='0',  # 使用GPU
        task='val'  # 验证任务
    )

    # 输出验证结果，包括精度（P）、召回率（R）、mAP 等指标
    # 打印验证结果的属性
    print(f"Precision: {val_results.metrics.precision}")
    print(f"Recall: {val_results.metrics.recall}")
    print(f"mAP@50: {val_results.metrics.map50}")
    print(f"mAP@50-95: {val_results.metrics.map50_95}")

