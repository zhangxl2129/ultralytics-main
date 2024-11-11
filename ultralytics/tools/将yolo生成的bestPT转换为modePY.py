import torch

# 加载训练好的模型
model = torch.load(r'E:\Projects\YOLOV10\ultralytics-main\ultralytics\runs\detect\train18\weights\best.pt')

# 打印模型内容以检查其结构
print(type(model))  # 检查加载的内容类型
print(model.keys())  # 如果是字典，查看其键

# 如果 model 是一个字典，直接保存
if isinstance(model, dict):
    torch.save(model, r'E:\Projects\YOLOV10\ultralytics-main\ultralytics\runs\detect\train18\weights\mode.pth')
else:
    # 否则，假设它是模型对象
    torch.save(model.state_dict(), r'E:\Projects\YOLOV10\ultralytics-main\ultralytics\runs\detect\train18\weights\mode.pth')
