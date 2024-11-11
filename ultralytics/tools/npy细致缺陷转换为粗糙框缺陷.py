import os
import numpy as np
from scipy.ndimage import label


def find_bounding_boxes(mask, defect_value):
    # 找到特定缺陷类别的连通区域
    labeled_mask, num_features = label(mask == defect_value)

    bounding_boxes = []

    for i in range(1, num_features + 1):
        # 获取连通区域的坐标
        defect_indices = np.argwhere(labeled_mask == i)

        if defect_indices.size == 0:
            continue

        # 获取最小外接矩形的坐标
        y_min, x_min = defect_indices.min(axis=0)
        y_max, x_max = defect_indices.max(axis=0)

        bounding_boxes.append((y_min, y_max, x_min, x_max))

    return bounding_boxes


def process_npy_file(input_file, output_folder):
    # 加载npy文件
    mask = np.load(input_file)

    # 创建一个新的mask，作为输出
    new_mask = np.zeros_like(mask)

    # 处理缺陷类别 1, 2, 3
    for defect_value in [1, 2, 3]:
        bounding_boxes = find_bounding_boxes(mask, defect_value)

        for box in bounding_boxes:
            y_min, y_max, x_min, x_max = box

            # 将外接框内的所有值设置为当前的缺陷值
            new_mask[y_min:y_max + 1, x_min:x_max + 1] = defect_value

    # 获取输出文件名
    output_file = os.path.join(output_folder, os.path.basename(input_file))

    # 保存新的npy文件
    np.save(output_file, new_mask)


def process_all_npy_files(input_folder, output_folder):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有npy文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.npy'):
            input_file = os.path.join(input_folder, file_name)
            process_npy_file(input_file, output_folder)


input_folder = 'E:\Projects\\UNet\\NEU-Seg\\test\predict'  # 输入npy文件夹
output_folder = 'E:\Projects\\UNet\\NEU-Seg\\test\predictA'  # 输出npy文件夹

process_all_npy_files(input_folder, output_folder)
