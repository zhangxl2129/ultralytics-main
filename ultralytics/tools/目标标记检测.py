import glob
import os
import cv2
import numpy as np


def check_labels(txt_labels, images_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    txt_files = glob.glob(txt_labels + "/*.txt")
    for txt_file in txt_files:
        filename = os.path.splitext(os.path.basename(txt_file))[0]
        pic_path = os.path.join(images_dir, filename + ".jpg")

        if not os.path.exists(pic_path):
            print(f"Image not found: {pic_path}")
            continue

        img = cv2.imread(pic_path)
        height, width, _ = img.shape

        with open(txt_file) as file_handle:
            cnt_info = file_handle.readlines()
            new_cnt_info = [line_str.replace("\n", "").split(" ") for line_str in cnt_info]

        color_map = {"0": (0, 255, 255)}
        for new_info in new_cnt_info:
            if len(new_info) < 3:  # 确保有至少一个类和两个坐标
                continue

            s = []
            for i in range(1, len(new_info), 2):
                if new_info[i] and new_info[i + 1]:  # 检查是否为空
                    try:
                        b = [float(new_info[i]), float(new_info[i + 1])]
                        s.append([int(b[0] * width), int(b[1] * height)])
                    except ValueError:
                        print(f"无法转换为浮点数: {new_info[i]}, {new_info[i + 1]}")

            if s:  # 只有在s不为空时绘制多边形
                cv2.polylines(img, [np.array(s, np.int32)], True, color_map.get(new_info[0]))

        output_path = os.path.join(output_dir, filename + "_marked.jpg")
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")


# 数据集路径
images_dir = "E:\\Projects\\trainingr\\true_images_training"
labels_dir = "E:\\Projects\\trainingr\\true_cocoTxt_training"
output_dir = "E:\\Projects\\trainingr\\true_CT"

check_labels(labels_dir, images_dir, output_dir)
