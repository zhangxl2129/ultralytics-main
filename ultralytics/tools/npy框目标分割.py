import numpy as np
import os
import cv2

def secondary_segmentation(region):
    if region.size == 0:
        return region  # 如果区域为空，直接返回
    # 在单通道灰度图上进行处理
    _, thresh = cv2.threshold(region, 50, 255, cv2.THRESH_BINARY)  # 根据需要调整阈值
    return thresh

def segment_defects(npy_folder, image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(npy_folder):
        if file.endswith('.npy'):
            npy_path = os.path.join(npy_folder, file)
            image_path = os.path.join(image_folder, file.replace('.npy', '.jpg'))  # 假设原图为 JPG 格式

            print("Loading image from:", image_path)

            if not os.path.isfile(image_path):
                print(f"Error: File does not exist at {image_path}")
                continue

            masks = np.load(npy_path)
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

            if original_image is None:
                print(f"Error: Could not read image at {image_path}")
                continue

            segmented_image = np.zeros_like(original_image)

            # 对每个框进行二次分割
            for class_id in np.unique(masks):
                if class_id > 0:  # 忽略背景
                    mask = (masks == class_id)
                    # 提取框内区域
                    region = original_image[mask]

                    # 确保提取的区域是有效的
                    if region.size > 0:
                        # 二次分割
                        processed_region = secondary_segmentation(region)

                        # 创建与原图相同形状的空白区域
                        full_mask = np.zeros_like(original_image)
                        full_mask_mask = np.zeros_like(original_image)  # 创建用于处理后的区域的掩膜
                        full_mask_mask[mask] = 1  # 填充原掩膜的位置

                        # 处理后的区域需要转换为与全掩膜相同的形状
                        if processed_region.ndim == 1:
                            processed_region = processed_region[:, np.newaxis]  # 增加一个维度

                        # 将处理结果放回对应区域
                        full_mask[mask] = processed_region.flatten()  # 使用 flatten 确保维度匹配

                        # 将处理后的区域放回分割图像
                        segmented_image = cv2.bitwise_or(segmented_image, full_mask)

            output_path = os.path.join(output_folder, file.replace('.npy', '_segmented.png'))
            cv2.imwrite(output_path, segmented_image)

npy_folder = 'E:\Projects\YOLOV10\DataB_Pre\predictions'
image_folder = 'E:\Projects\YOLOV10\DataB_Pre\\test\images'
output_folder = 'E:\Projects\YOLOV10\DataB_Pre\predictionsIMG'

segment_defects(npy_folder, image_folder, output_folder)
