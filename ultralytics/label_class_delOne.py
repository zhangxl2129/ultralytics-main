import os

label_dir = 'E:\Projects\YOLOV10\\NEU-Seg\\test\labels'

for label_file in os.listdir(label_dir):
    file_path = os.path.join(label_dir, label_file)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 修改类别索引
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        parts[0] = str(int(parts[0]) - 1)  # 将类别索引减1
        new_lines.append(' '.join(parts) + '\n')

    # 保存修改后的标签文件
    with open(file_path, 'w') as file:
        file.writelines(new_lines)

print('所有标签文件已成功修改！')
