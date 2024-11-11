import numpy as np
np.set_printoptions(threshold=np.inf)
masks = np.load("E:\Projects\\trainingr\\true_yanma_test_npy\\000810.npy")
masks2 = np.load("E:\Projects\CpIOU\A\\000759.npy")
masks3 = np.load("E:/Projects/CpIOU/test_ground_truths/000810.npy")
print(masks2)

from PIL import Image
import numpy as np

# 读取 .png 文件
image = Image.open("E:\\我的比赛\\第六届校园人工智能大赛\\第2题 钢材表面缺陷检测与分割\\赛题2-赛题说明-参赛细则-数据集-计分办法\\NEU_Seg-main\\annotations\\training\\004390.png")

# 将图像转换为 NumPy 数组
masks1 = np.array(image)

# 打印掩码的值
#print(masks1)
