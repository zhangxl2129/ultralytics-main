import os
import shutil

def rename_and_copy_files(source_dir, dest_dir, new_name_format):
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 获取源文件夹中的所有文件
    files = os.listdir(source_dir)

    for index, filename in enumerate(files):
        # 生成新的文件名
        new_name = new_name_format.format(index + 1)  # 从1开始编号
        file_extension = os.path.splitext(filename)[1]  # 获取文件扩展名
        new_file_path = os.path.join(dest_dir, new_name + file_extension)

        # 源文件路径
        source_file_path = os.path.join(source_dir, filename)

        # 复制文件到目标文件夹并重命名
        shutil.copy(source_file_path, new_file_path)
        print(f"复制并重命名: {source_file_path} -> {new_file_path}")

# 示例参数
source_dir = "E:\Projects\YOLOV10\DataB_Pre\predictions"  # 源文件夹
dest_dir = "E:\Projects\YOLOV10\DataB_Pre\predictionsC"  # 目标文件夹
new_name_format = "c_test_predictions_{:06d}"  # 新文件名格式，{:03d}表示三位数字格式

rename_and_copy_files(source_dir, dest_dir, new_name_format)
