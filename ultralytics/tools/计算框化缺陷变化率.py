import numpy as np
import os

def count_defects(folder_path):
    total_counts = {1: 0, 2: 0, 3: 0}
    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            data = np.load(os.path.join(folder_path, file))
            for class_id in total_counts.keys():
                total_counts[class_id] += np.sum(data == class_id)
    return total_counts

folder_a = 'E:\\Projects\\CpIOU\\baseline_predictions'
folder_b = 'E:\\Projects\\CpIOU\\test_baseline_predictions'

counts_a = count_defects(folder_a)
counts_b = count_defects(folder_b)

# 计算变化率
change_rates = {}
for class_id in range(1, 4):
    if counts_a[class_id] > 0:
        change_rates[class_id] = (counts_b[class_id] - counts_a[class_id]) / counts_a[class_id]
    else:
        change_rates[class_id] = None  # 如果没有原始数据，变化率无法计算

# 计算整体平均变化率
overall_change_rate = np.mean([rate for rate in change_rates.values() if rate is not None])

print("1类缺陷变化率:", change_rates[1])
print("2类缺陷变化率:", change_rates[2])
print("3类缺陷变化率:", change_rates[3])
print("整体平均变化率:", overall_change_rate)


