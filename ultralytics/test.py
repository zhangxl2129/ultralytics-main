import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 加载最优的模型权重
    best_model = 'runs/detect/train8/weights/best.pt'  # 修改为你保存模型的位置
    model = YOLO(best_model)

    # 进行推理（检测）
    results = model.predict(source='E:/Projects/YOLOV8/NEU-DET/valid/images',  # 测试图像路径
                            save=True,  # 保存检测结果
                            conf=0.25,  # 置信度阈值
                            imgsz=640)  # 测试时使用的图像大小

    # 输出检测结果
    for result in results:
        print(result)

