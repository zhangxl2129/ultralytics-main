import cv2
import numpy as np
import os

# 分割掩码路径和保存YOLO格式标签的路径
mask_dir = 'E:\Projects\YOLOV10\DataB\Lab'  # 分割掩码目录
output_dir = 'E:\Projects\YOLOV10\DataB\labels'  # 保存YOLO标签的目录
image_dir = 'E:\Projects\YOLOV10\DataB\Img'  # 原图目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 处理每个分割掩码
for mask_filename in os.listdir(mask_dir):
    if mask_filename.endswith('.png'):  # 假设掩码文件是png格式
        mask_path = os.path.join(mask_dir, mask_filename)
        image_path = os.path.join(image_dir, mask_filename)  # 假设图像与掩码同名

        # 读取分割掩码和原图
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path)

        # 获取图像尺寸
        height, width = mask.shape

        # 打开标签文件，YOLO格式为：class_id center_x center_y width height
        label_filename = os.path.splitext(mask_filename)[0] + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        with open(label_path, 'w') as f:
            # 逐类处理（假设类别是0-3，背景为0）
            for class_id in range(1, 4):
                # 创建掩码，提取每个类的区域
                class_mask = np.where(mask == class_id, 255, 0).astype(np.uint8)

                # 使用cv2.findContours找到物体轮廓
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    # 生成最小外接矩形（bounding box）
                    x, y, w, h = cv2.boundingRect(contour)

                    # 计算YOLO格式的中心点坐标和宽高（相对图像尺寸的归一化坐标）
                    center_x = (x + w / 2) / width
                    center_y = (y + h / 2) / height
                    bbox_width = w / width
                    bbox_height = h / height

                    # 写入到txt文件，格式：class_id center_x center_y width height
                    f.write(f'{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n')

        print(f'Saved {label_path}')

print('完成边界框生成！')
