import cv2
import numpy as np
import os

# 定义忽略的背景颜色（黑色）
ignore_color = [0, 0, 0]

# 设置容差值，允许颜色匹配有轻微变化
tolerance = 30  # 可以根据具体图像调整这个值

# 定义缺陷框颜色映射，每类缺陷对应一种颜色
color_mapping = {
    0: (255, 0, 0),  # 红色框
    1: (0, 255, 0),  # 绿色框
    2: (0, 0, 255),  # 蓝色框
    3: (255, 255, 0),  # 黄色框
    4: (255, 0, 255),  # 紫色框
    5: (0, 255, 255)  # 青色框
}


def color_in_range(color, lower, upper):
    """ 检查颜色是否在给定范围内 """
    return all(lower[i] <= color[i] <= upper[i] for i in range(3))


def convert_to_yolo_format(image_path, output_txt_path, corresponding_jpeg_path, output_jpeg_path, color_to_class):
    # 读取PNG标记图和对应的JPEG图像
    image = cv2.imread(image_path)
    original_image = cv2.imread(corresponding_jpeg_path)

    # 由于图像尺寸固定为200x200，直接设置宽高
    height, width = 200, 200

    # 打开txt文件
    with open(output_txt_path, 'w') as f:
        # 获取图像中出现的所有颜色，并排除黑色背景
        unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)

        for color in unique_colors:
            if np.all(color == ignore_color):
                continue  # 忽略黑色背景

            # 检查该颜色是否已经被分类
            color_key = tuple(color)
            if color_key in color_to_class:
                class_id = color_to_class[color_key]
            else:
                # 分配新类别ID
                class_id = len(color_to_class)
                color_to_class[color_key] = class_id  # 记录新颜色及其类别ID

            # 创建掩膜，以分离当前颜色
            lower_bound = np.array([max(c - tolerance, 0) for c in color])
            upper_bound = np.array([min(c + tolerance, 255) for c in color])
            mask = cv2.inRange(image, lower_bound, upper_bound)

            # 找到所有连通区域
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # 获取每个连通区域的最小外接矩形
                x, y, w, h = cv2.boundingRect(contour)

                if w > 0 and h > 0:  # 确保框有大小
                    # 计算YOLO格式的中心坐标、宽度和高度（归一化）
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_normalized = w / width
                    h_normalized = h / height

                    # 写入txt文件，格式为: class_id x_center y_center width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_normalized:.6f} {h_normalized:.6f}\n")

                    # 在原图上绘制边界框
                    box_color = color_mapping[class_id % len(color_mapping)]  # 选择对应的框颜色
                    cv2.rectangle(original_image, (x, y), (x + w, y + h), box_color, 2)
                    # 在边框上标记类别
                    cv2.putText(original_image, f"{class_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 1)

    # 保存绘制了标记框的JPEG图像
    cv2.imwrite(output_jpeg_path, original_image)


# 文件夹路径
input_dir = r"E:\Projects\trainData\SegmentationClass"
output_dir = r"E:\Projects\trainData\YOLOLabels"
jpeg_dir = r"E:\Projects\trainData\JPEGImages"
output_jpeg_dir = r"E:\Projects\trainData\JPEGImagesWithBoxes"

# 创建YOLO标签输出文件夹（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建标记框的JPEG图像输出文件夹
if not os.path.exists(output_jpeg_dir):
    os.makedirs(output_jpeg_dir)

# 存储颜色到类别ID的映射
color_to_class = {}

# 批量处理所有png图片
for image_file in os.listdir(input_dir):
    if image_file.endswith(".png"):
        image_path = os.path.join(input_dir, image_file)
        output_txt_path = os.path.join(output_dir, image_file.replace(".png", ".txt"))
        corresponding_jpeg_path = os.path.join(jpeg_dir, image_file.replace(".png", ".jpg"))
        output_jpeg_path = os.path.join(output_jpeg_dir, image_file.replace(".png", ".jpg"))

        # 检查对应的JPEG文件是否存在
        if os.path.exists(corresponding_jpeg_path):
            convert_to_yolo_format(image_path, output_txt_path, corresponding_jpeg_path, output_jpeg_path,
                                   color_to_class)
        else:
            print(f"JPEG文件 {corresponding_jpeg_path} 不存在，跳过。")

print("标签转换和绘制完成！")
