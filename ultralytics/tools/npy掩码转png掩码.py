import numpy as np
from PIL import Image
import os


def npy_to_png(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有npy文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.npy'):
            # 加载npy文件
            np_array = np.load(os.path.join(input_folder, filename))

            # 确保数组是200x200并且值在0-3之间
            if np_array.shape == (200, 200) and np.array_equal(np.unique(np_array), [0, 1, 2, 3]):
                # 转换为PIL图像
                img = Image.fromarray(np_array.astype(np.uint8))

                # 保存为png文件
                output_filename = filename.replace('.npy', '.png')
                img.save(os.path.join(output_folder, output_filename))
                print(f"Converted {filename} to {output_filename}")
            else:
                print(f"Skipping {filename}: not valid array shape or values.")


# 使用示例
input_folder = 'path/to/your/npy_files'  # 替换为你的npy文件夹路径
output_folder = 'path/to/output/png_files'  # 替换为输出png文件夹路径
npy_to_png(input_folder, output_folder)
