import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_npy_mask(npy_file_path):
    """
    Load the npy mask file.

    Args:
        npy_file_path (str): Path to the npy file.

    Returns:
        np.ndarray: The mask array.
    """
    return np.load(npy_file_path)


def apply_mask_to_image(image, mask, class_colors):
    """
    Apply the defect mask to the original image.

    Args:
        image (np.ndarray): The original image.
        mask (np.ndarray): The defect mask array (200x200).
        class_colors (dict): A dictionary mapping class ids to colors.

    Returns:
        np.ndarray: Image with the mask applied.
    """
    # 确保掩码和图像大小一致，如果图像不是200x200，则需要调整
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 创建一个彩色的掩码图像
    mask_color = np.zeros_like(image)

    # 根据不同的类别进行着色
    for class_id, color in class_colors.items():
        mask_class = (mask_resized == class_id)
        mask_color[mask_class] = color

    # 将掩码与原图合并 (设置透明度 alpha)
    alpha = 0.5  # 透明度因子
    combined_img = cv2.addWeighted(image, 1, mask_color, alpha, 0)

    return combined_img


if __name__ == "__main__":
    # 图片和掩码文件夹路径
    image_folder = "E:\Projects\YOLOV10\DataB_Pre\\test\images"  # 替换为你的图片文件夹路径
    npy_mask_folder = "E:\Projects\YOLOV10\DataB_Pre\predictions"  # 替换为对应的掩码文件夹路径
    result_folder = "E:\Projects\YOLOV10\DataB_Pre\predictionsIMG"  # 保存结果的文件夹
    os.makedirs(result_folder, exist_ok=True)

    # 获取所有图片文件和掩码文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]  # 假设图像为 .jpg 格式
    npy_files = [f for f in os.listdir(npy_mask_folder) if f.endswith('.npy')]

    # 定义类别颜色映射
    class_colors = {
        0: [0, 0, 0],  # 背景 - 黑色
        1: [0, 255, 0],  # 类别 1 - 绿色
        2: [0, 0, 255],  # 类别 2 - 红色
        3: [255, 0, 0]  # 类别 3 - 蓝色
    }

    # 遍历每张图片和对应的掩码
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        npy_mask_path = os.path.join(npy_mask_folder, image_file.replace('.jpg', '.npy'))  # 假设文件名相同

        if os.path.exists(npy_mask_path):
            # 加载原图像
            image = cv2.imread(image_path)

            # 加载掩码
            mask = load_npy_mask(npy_mask_path)

            # 将掩码应用到原图上
            result_image = apply_mask_to_image(image, mask, class_colors)

            # 保存结果
            result_image_path = os.path.join(result_folder, image_file.replace('.jpg', '_with_mask.jpg'))
            cv2.imwrite(result_image_path, result_image)

            print(f"Saved result image with mask: {result_image_path}")
        else:
            print(f"No mask found for {image_file}")

    # 显示示例结果（可选）
    example_image_path = os.path.join(result_folder, image_files[0].replace('.jpg', '_with_mask.jpg'))
    if os.path.exists(example_image_path):
        example_image = cv2.imread(example_image_path)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))
        plt.title("Image with Defect Mask")

        plt.show()
