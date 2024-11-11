import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_png_to_npy(mask_dir, npy_dir):
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    # 获取文件夹中的所有 PNG 文件
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

    # 遍历文件夹中的所有掩码图
    for mask_file in tqdm(mask_files):
        mask_path = os.path.join(mask_dir, mask_file)

        # 读取掩码图并转换为 NumPy 数组
        mask_img = Image.open(mask_path)
        mask_array = np.array(mask_img)

        # 检查图像大小是否为200x200且值在0-3之间
        if mask_array.shape != (200, 200):
            print(f"Warning: {mask_file} is not of size 200x200, skipping.")
            continue
        if np.max(mask_array) > 3 or np.min(mask_array) < 0:
            print(f"Warning: {mask_file} contains values outside 0-3 range, skipping.")
            continue

        # 保存为 .npy 文件
        npy_file = mask_file.replace('.png', '.npy')
        npy_path = os.path.join(npy_dir, npy_file)
        np.save(npy_path, mask_array)


# 设置输入输出路径
mask_dir = 'E:\Projects\YOLOV10\DataB\Lab'  # 掩码图文件夹路径（路径1）
npy_dir = 'E:\Projects\YOLOV10\DataB\Label'  # 保存 .npy 文件的文件夹路径（路径2）

# 执行转换
convert_png_to_npy(mask_dir, npy_dir)
