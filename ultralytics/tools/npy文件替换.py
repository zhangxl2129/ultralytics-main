import os
import shutil


def copy_npy_files(jpg_folder, npy_folder, target_folder):
    # 检查目标文件夹是否存在，不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历jpg文件夹中的所有jpg文件
    for file_name in os.listdir(jpg_folder):
        if file_name.endswith('_with_mask.jpg'):
            # 提取jpg文件名前的数字部分
            base_name = file_name.split('_with_mask.jpg')[0]

            # 对应的npy文件名
            npy_file_name = base_name + '.npy'
            npy_file_path = os.path.join(npy_folder, npy_file_name)
            target_file_path = os.path.join(target_folder, npy_file_name)

            # 检查npy文件是否存在
            if os.path.exists(npy_file_path):
                # 复制并替换到目标文件夹
                shutil.copy2(npy_file_path, target_file_path)
                print(f'Copied: {npy_file_name} to {target_folder}')
            else:
                # 抛出错误并终止程序
                raise FileNotFoundError(f'Error: {npy_file_name} not found in {npy_folder}. Operation terminated.')


# 使用示例
jpg_folder = 'E:\\Projects\\CpIOU\\Test_Img\Box\\TH'  # jpg图片所在的文件夹路径
npy_folder = 'E:\\Projects\CpIOU\\test_baseline_predictions'  # npy文件所在的文件夹路径
target_folder = 'E:\\Projects\\CpIOU\\test_ground_truths'  # 目标文件夹路径

copy_npy_files(jpg_folder, npy_folder, target_folder)
