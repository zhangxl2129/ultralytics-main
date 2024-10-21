import os

# 定义文件夹路径
folder_path = r"E:\我的比赛\第六届校园人工智能大赛\test_json"

# 定义期望的文件编号范围
expected_files = {f"{i:06d}.json" for i in range(1, 841)}

# 获取文件夹中的所有文件名
actual_files = set(os.listdir(folder_path))

# 找到缺少的文件
missing_files = expected_files - actual_files

# 输出缺少的文件
if missing_files:
    print(f"缺少的文件: {sorted(missing_files)}")
else:
    print("没有缺少的文件。")
