# 基于YOLO11+SPD-Conv+WTConv+H-RAMi 的钢材表面缺陷检测

YOLO11+SPD-Conv作为参加钢材表面缺陷检测的解决方案，旨在通过深度学习技术实现准确高效的钢材表面缺陷检测。

## 最新日志
**2024-12-26** 引入H-RAMi注意力机制，识别准确率和效率进一步得到提升。
            
        数据集：NUE-DET 
             mAP    YOLO11+SPD-Conv+WTConv    YOLO11+SPD-Conv+WTConv+H-RAMi
                            0.777                          0.83  
<img width="849" alt="548a35b983017845e85bd22897e1fa5" src="https://github.com/user-attachments/assets/9e2eed8e-9fad-4e3d-a7de-227506a94642" />

     其它常见数据集表现：
**2024-12-19** 更新：本次更新涉及将本项目转换为微服务，方便用户体验本模型。  
由于本次更新偏于应用服务，故此新建仓库进行分享。  
用户可以通过以下链接进行访问：[GitHub 仓库](https://github.com/zhangxl2129/FlaskYOLO/tree/master)

## 部署实现

### 方式 1：使用 pip 安装
1. 安装 `ultralytics YOLO` 库（源程序）
    ```bash
    pip install ultralytics
    ```

2. 通过以下命令获取作者创新的模块 `SPDConv`、`WTConv`
    ```bash
    pip install xl-yolo-pkg
    ```

#### SPDConv
这是用于 YOLO 模型的自定义 SPDConv 模块，旨在提高特定任务的性能。SPD-Conv 通过替换步幅卷积和池化层，保持了细粒度信息，提高了特征学习效率，特别适用于低分辨率或小目标任务。

#### WTConv
这是用于 YOLO 模型的自定义 WTConv 模块，引入小波卷积，旨在扩大卷积的感受野并有效捕捉图像中的低频信息，对于多尺度问题和小目标问题有显著效果。

#### H-RAMi
H-RAMi应用注意力机制来动态计算每个输入元素的重要性。模型会根据每个元素的重要性分配不同的权重，从而专注于最关键的信息。

### 方式 2：下载源代码
1. 下载数据集  
   通过百度网盘分享的文件：[trainingr.zip](https://pan.baidu.com/s/1gA3fHWneXgnpWwKnvG75NA?pwd=o0f3) 提取码：`o0f3`

2. 将程序代码下载至本地，并存储于数据集同级目录下。建议路径为英文路径，防止意外报错。

3. 使用 `environment.yml` 文件一键安装程序运行环境。

4. 运行 `train.py` 进行模型训练，运行 `test.py` 进行模型测试。

### 方式 3：下载可执行程序
此方式便捷实现，优点是下载即用，无需配置操作，缺点是占用大量空间，不够灵活。  
通过百度网盘分享的文件：[AI 程序](https://pan.baidu.com/s/1_esFwSwl6A7SkOhrACHnfg?pwd=xnyr) 提取码：`xnyr`

---

## 竞赛官网
[钢材表面缺陷检测与分割竞赛](http://bdc.saikr.com/vse/50185)

![image](https://github.com/user-attachments/assets/070da04c-448e-49da-a65a-9edc9ceaae2c)

## 分析
国赛的突破点在于数据分析。通过对 B 和 C 数据集的深入分析，发现它们应用了多种数据增强方法，包括对角线拼接、左右拼接、上下拼接、椒盐噪声、左上角四分之一添加不规则矩形、亮暗度变化等。C 榜单中的 mIoU 值普遍高于 A 榜，主要是因为数据泄露——B 和 C 数据集中的原始图像出现在 A 数据集的训练和测试集中。

为了解决这一问题，我们采用了基于 A 数据集增强后的图像来测试模型的泛化能力，确保模型能够应对不同场景。

### 项目亮点
1. **在线数据增强，贴近目标分布**  
   我们通过在线数据增强，使训练数据尽可能接近 C 数据集的分布，从而增强模型的泛化能力。

2. **合理的评估方式，快速迭代优化**  
   使用基于 A 数据集的在线增强数据来评估模型的泛化性，实现了高效的训练和快速优化。

---

## 项目结构
- `yolo11-WTConv-SPDConv.yaml`：改造后的 YOLO11-WTConv-SPDConv 网络模型，路径：`ultralytics/cfg/models/11/`
- `yolo11-SPDConv.yaml`：改造后的 YOLO11-SPDConv 网络模型，路径：`ultralytics/cfg/models/11/`
- `NEU-Seg-DataB.yaml`：训练、验证、测试数据集配置文件，路径：`ultralytics/cfg/datasets/NEU-Seg-DataB.yaml`
- `train.py`：训练配置路径，可直接运行或手动自定义参数配置。
- `val.py`：验证训练出的模型性能和检测效果。
- `test.py`：测试训练好的模型并根据测试集生成检测结果，结果以 `.npy` 格式文件存储于 `ultralytics/runs/detect` 路径。
- `README.md`：项目文档（即本文件），描述项目的结构、使用方法和相关细节。
- `best.pt`：经过训练后的最优模型参数，路径：`ultralytics/runs/detect/train40/weights/best.pt`

---

## 对标经典 UNet 网络模型的性能提升
通过网络结构的改进，我们使用全球 AI 大赛提供的性能测试工具对比了本网络模型与传统 UNet 网络模型的性能：

| 模型      | Class1 IoU   | Class2 IoU   | Class3 IoU   | mIoU        | FPS    | Model Parameters |
|-----------|--------------|--------------|--------------|-------------|--------|------------------|
| **UNet**  | 0.6644       | 0.8391       | 0.7363       | 0.7466      | 66.33  | 17,266,436       |
| **Ours**  | 0.7469       | 0.9002       | 0.8445       | 0.8302      | 83.42  | 2,212,404        |

可以看出，改进后的 YOLOv11 模型在三类缺陷上的准确率分别提升了：  
- Class1 IoU: +0.0825  
- Class2 IoU: +0.0611  
- Class3 IoU: +0.1081  

同时，模型的参数量相比于 UNet 从 17,266,436 减少到 2,212,404，实现了显著的轻量化。

---

## 数据集和性能对比
模型在不同数据集和配置下的性能对比如下：

### YOLO11 + SPD-Conv 性能对比
| 模型       | Class1 IoU   | Class2 IoU   | Class3 IoU   | mIoU        |
|------------|--------------|--------------|--------------|-------------|
| **YOLOv11** | 0.7469       | 0.9002       | 0.8445       | 0.8302      |
| **YOLOv10** | 0.7538       | 0.8683       | 0.8332       | 0.8200      |
| **UNet**    | 0.6644       | 0.8391       | 0.7363       | 0.7466      |

---

## YOLO11-SPDConv 技术文档
YOLO11+SPD-Conv 作为参加钢材表面缺陷检测与分割竞赛的解决方案，旨在通过深度学习技术实现高效的钢材表面缺陷检测和分割。

### 竞赛成绩
- **全球 AI 人工智能大赛四川赛区第4名**（2024-11-04）
- **第六届全球校园人工智能算法精英大赛 研究生组 国家三等奖**（2024-11-14）

### 项目结构概览
1. **绪论**
   - 研究背景与意义：钢材表面缺陷的自动化检测和分割是提升钢材质量、降低生产成本的关键技术。
   - 研究任务：基于YOLOv11设计并实现高效准确的钢材表面缺陷检测模型。

2. **算法改进**
   - **SPD-Conv** 模块：替代传统的步长卷积和池化层，通过空间到深度转换卷积增强特征提取能力。
   - **数据增强**：旋转、平移、剪切变换等技术增强数据集，提升模型的鲁棒性和泛化能力。

### 图像增强策略
- **旋转增强**、**平移增强**、**Mosaic增强**等增强方法提高模型的识别能力。
  
yolo11-SPDConv钢材表面缺陷检测技术文档 
===
YOLO11+SPD-Con作为参加钢材表面缺陷检测与分割竞赛的解决方案，旨在通过深度学习技术实现高效的钢材表面缺陷检测和分割。

-------
   第1章 绪论  
   1.1 研究背景与意义  
   　　随着工业化进程的加速，钢材作为基础材料在建筑、交通、机械制造等多个领域中扮演者至关重要的角色。钢材的质量直接影响到产品的安全性和可靠性，因此对钢材表面缺陷的检测显得尤为重要。传统的人工检测方法效率低下，受限于检测人员的经验和主观判断，常常导致漏检和误检的问题。此外，随着钢铁生产规模的不断扩大，人工检测难以满足大规模生产的需求。因此，开发高效、准确的自动化检测系统成为了行业内亟待解决的课题。
   　　近年来，深度学习技术在图像处理领域取得了显著进展，其强大的特征学习能力使其在缺陷检测和分割任务中展现出良好的应用前景。通过构建基于深度学习的钢材表面缺陷检测与分割模型，可以实现对缺陷区域的精确定位和分类，提高检测的准确性和效率。这不仅能有效减少人工干预，降低生产成本，还能在一定程度上提升产品的整体质量。
   　　因此，采用自动化、智能化的检测技术成为了行业发展的必然趋势，基于深度学习的自动化检测技术成为提升钢材表面缺陷检测效率和准确性的重要手段。这项研究不仅具有重要的学术价值，也为实际工业应用提供了新的思路和方法。通过提升钢材表面缺陷的检测效率和准确性，能够有效改善生产流程，降低资源浪费，为钢铁企业的可持续发展做出贡献。
   
   1.2 研究任务  
   　　本项目使用赛会主办方指定的钢材表面缺陷检测数据集基于YOLOv11设计并实现一种钢材缺陷检测模型，该模型能够高效准确地对钢材表面的缺陷进行像素级分割，实现精确的缺陷识别。  
   　　本项目主要由数据预处理、模型选择、损失函数设计、优化策略、后处理技术以及模型评估共同构成，确保能够开发出一种既准确又高效的缺陷检测与分割模型。  
   　　
     
   第2章 基于yolov11网络的算法改进  
   　　YOLOv11是一种针对目标检测任务的改进型深度学习模型。其核心目标是在保持高精度的同时，进一步提升推理速度和效率。本项目对于YOLOv11的改进主要体现在以下几个方面：  
   2.1 SPD-Conv空间深度转换卷积改进yolov11  
   　　这是一个基于YOLOv11架构改进的目标检测模型，在YOLOv11中利用SPD-Conv替换传统的步长卷积和池化层，增强了特征提取能力，提高了物体检测的精度。其精炼的设计和优化的训练流程提升了处理速度，在准确性和性能之间实现了良好的平衡。使用更少的参数，使其在保证准确性的情况下具备更高的计算效率。  
   (1) SPD-Conv空间深度转换卷积  
   　　SPD-Conv模块由一个空间-深度（Space-to-Depth, SPD）层和一个非步幅卷积层组成。SPD层将输入特征图按照指定比例下采样，将其空间信息重新排列到通道维度上，以此减少空间分辨率而不丢失细节信息。随后，非步幅卷积层（即步幅为1的卷积）进一步处理这些特征，提取判别性信息，从而实现特征的压缩和增强，同时避免了传统步幅卷积和池化层带来的信息丢失问题。  
   ![image](https://github.com/user-attachments/assets/bc8378b6-10ad-42b2-8807-c9b7229a1452)  
   图1 SPD-Conv模型图  
   
   　　SPD-Conv（空间到深度卷积）的基本原理是用于改进传统卷积神经网络（CNN）中对小物体和低分辨率图像处理的性能。它主要通过以下几个关键步骤实现：  
   　　1. 替换步长卷积和池化层：SPD-Conv设计用来替代传统CNN架构中的步长卷积层和池化层。步长卷积和池化层在处理低分辨率图像或小物体时会导致细粒度信息的丢失。  
   　　2. 空间到深度（SPD）层：SPD层的作用是降采样特征图的通道维度，将特征图的空间维度转换成深度维度，通过增加通道数来保留更多信息。这种方式可以避免传统方法中的信息丢失。  
   　　3.非步长卷积层：在SPD层之后，SPD-Conv使用一个非步长（即步长为1）的卷积层。保持了空间维度，减少了通道数。这种替代方法避免了信息的丢失，并允许网络捕获更精细的特征，从而提高了在复杂任务上的性能。  
   　　空间到深度（SPD）层是SPD-Conv中的一个关键组件，其作用是将输入特征图的空间块（像素块）重新排列进入深度（通道）维度，以此来增加通道数，同时减少空间分辨率，但不丢失信息。通过这种方式，这一转换允许CNN捕捉和保留在处理小物体和低分辨率图像时经常丢失的精细信息。SPD层后面紧跟的是非步长卷积层，它进一步处理重新排列后的特征图，确保有效特征的提取和使用。通过这种方法，SPD-Conv能够在特征提取阶段保留更丰富的信息，从而提高模型对于小物体和低分辨率图像的识别性能  
   　　非步长卷积层采用的是步长为1的卷积操作，意味着在卷积过程中，滤波器（或称为卷积核）会在输入特征图上逐像素移动，没有跳过任何像素。这样可以确保在特征图的每个位置都能应用卷积核，最大程度地保留信息，并生成丰富的特征表示。非步长卷积层是紧随空间到深度（SPD）层的一个重要组成部分。在SPD层将输入特征图的空间信息重新映射到深度（通道）维度后，非步长卷积层（即步长为1的卷积层）被用来处理这些重新排列的特征图。由于步长为1，这个卷积层不会导致任何进一步的空间分辨率降低，这允许网络在不损失细节的情况下减少特征图的通道数。这种方法有助于改善特征的表征，特别是在处理小物体或低分辨率图像时，这些场景在传统CNN结构中往往会丢失重要信息。  
   
   (2) SPD-Conv模块在YOLOv11的改进  
   　　如图2所示，SPD-Conv模块基于YOLOv11的改进被分为三个主要部分：  
   　　1.主干网络（Backbone）：这是特征提取的核心部分，每个SPD和Conv层的组合都替换了原始YOLOv11中的步长卷积层。  
   　　2.颈部（Neck）：这部分用于进一步处理特征图，以获得不同尺度的特征，从而提高检测不同大小物体的能力。它也包含SPD和Conv层的组合，以优化特征提取。  
   　　3.头部（Head）：这是决策部分，用于物体检测任务，包括定位和分类。头部保持了YOLO原始架构的设计。  
   ![image](https://github.com/user-attachments/assets/d1f9ba2c-c07e-4511-88c6-a175f52c87f4)  
   图 2 YOLOv11改进后模型图  
   
   　　YOLOv11利用SPD-Conv模块改进Backbone和Neck架构，增强了特征提取能力，提高了物体检测的精度。其精炼的设计和优化的训练流程提升了处理速度，在准确性和性能之间实现了良好的平衡。YOLOv11在数据集上达到了更高的均值平均精度（mAP），使用更少的参数，使其在不妥协准确性的情况下具备更高的计算效率。此外，YOLOv11具有出色的环境适应性，可部署于多种环境中，支持物体检测、实例分割、图像分类、姿态估计和定向物体检测（OBB）等多种计算机视觉任务。  
   　　
   2.2数据增强策略的改进  
   　　为了提高YOLOv10模型的鲁棒性和泛化能力，改进后的数据增强策略通过模拟多种真实世界的场景变化，增加了图像的多样性，具体实现如下：  
   　　由于数据集的缺陷样本数量有限，实施有效的数据增强策略可以显著提高模型的泛化能力。本项目采用了多种数据增强技术来丰富数据集，并提高模型对钢材表面缺陷检测的鲁棒性。  
   (1) 旋转增强  
   　　对图像应用了不同程度的旋转增强，如图3所示，旋转角度最多可达5度。这模拟了在实际生产环境中可能遇到的不同视角的钢材表面，有助于模型学习到不同方向的缺陷特征。  
   ![image](https://github.com/user-attachments/assets/ad6d608b-dda8-4d52-bf59-4b67623ffec8)  
   图 3 旋转增强  
   
   (2) 平移增强  
   　　通过平移增强，随机移动图像中的对象，如图4所示。平移范围最多可达图像尺寸的20%。这种增强方式模拟了缺陷在钢材表面不同位置出现的情况，增强了模型对缺陷位置变化的适应能力。  
   ![image](https://github.com/user-attachments/assets/b5e2e6e6-d4ce-4339-b3c6-b0ef5712bba9)
   图 4平移增强  
   
   (3) 剪切变换  
   　　剪切变换通过在图像上施加剪切力来模拟不同的视角效果，如图5所示，在数据增强中加入了最多1.0度的剪切变换。即使在视角变化的情况下，也有助于模型更好地理解缺陷的形状和大小。  
   
   (4) 缩放增强  
   　　对图像进行了缩放增强，如图6所示，缩放比例范围从50%到150%。这种增强方式使得模型能够适应不同尺寸的缺陷，提高了模型对缺陷大小变化的识别能力。  
   ![image](https://github.com/user-attachments/assets/3f328317-13f7-41a7-add1-5355b3e6beef)  
   图 5剪切变换  
   
   ![image](https://github.com/user-attachments/assets/8b9f2f55-40c6-44c3-998d-04083984c1a9)  
   图 6缩放增强  
   
   （5）Mosaic增强  
   　　Mosaic增强是一种有效的数据增强技术，通过将四张不同的图像拼接在一起形成一个新的图像，我们以1.0的比例使用mosaic增强，如图7所示。进一步丰富了训练集的多样性，有助于模型学习到更多样化的缺陷特征，并且提高了模型对新场景的泛化能力。  
   　　
   
   ![image](https://github.com/user-attachments/assets/669e5b97-37b6-49bc-a80e-76bfb96c84f7)  
   图 7 Mosaic增强  
   　　（6）随机透视变换  
   　　通过模拟不同的视角和深度效果来增强图像，以0.5的比例应用这种变换。如图8所示，通过改变图像的透视角度使得模型在面对不同视角和深度的缺陷时，仍能保持较高的识别准确率。  
   ![image](https://github.com/user-attachments/assets/fa2c7aab-021e-4978-9e5c-43c87e6df3fc)
   图 8 随机透视变换  
   
   　（7）HSV色彩空间增强  
   　　在HSV色彩空间中，对色调(H)、饱和度(S)和明度(V)分别进行了增强，如图9所示，变化范围分别为0.015、0.7和0.4，通过改变这些参数，模型能够适应光照和颜色变化对图像的影响，使得模型能够在不同的光照条件下稳定地识别缺陷。  
   ![image](https://github.com/user-attachments/assets/a2097fba-175c-4d39-80e9-10b5e7eee6be)
   图 9 HSV色彩空间增强  
   （8）遮盖增强  
   　　通过在图像的一定区域内随机遮盖一部分像素来模拟图像损坏的情况，以0.1的比例使用这种增强，如图10所示，在图像中随机遮盖部分区域，迫使模型在不完全依赖图像局部信息的情况下进行缺陷识别，从而提升模型的鲁棒性，这有助于模型学习到在图像不完整时如何进行缺陷检测。  
   ![image](https://github.com/user-attachments/assets/eaa4a1b7-1881-4121-bb12-3633803097bd)
   图 10 遮盖增强  
   　　采用这些先进的数据增强技术，显著提升了模型针对钢材表面缺陷检测的泛化性和稳健性。这些技术不仅增强了模型在训练数据上的表现，还为模型在实际工业环境中的运用打下了牢固的基础。通过实施这些增强措施，我们得以更加精确地模拟实际生产过程中可能出现的多样化情况，进而确保了模型在现实世界应用中的高度有效性和稳定性。  
   
   2.3超参数调优  
   　　对YOLOv10的超参数进行了调优，目的是为了提升检测精度、提高模型稳定性和泛化能力，同时优化训练效率并满足计算资源的限制。通过合理设置学习率、批量大小、训练轮数等参数，模型能够更好地学习数据特征，增强对钢材表面缺陷的识别精度。选择合适的优化器、动量和权重衰减等超参数，确保模型在训练过程中的稳定性和快速收敛。同时，数据增强和其他参数设置提升了模型的泛化能力，减少过拟合风险。优化数据加载和输入尺寸等策略也提高了训练效率，确保在有限资源下实现良好性能。  

WTConv 技术文档 
===
![image](https://github.com/user-attachments/assets/10649b1e-65e9-47a3-a8ad-fc831c6b8343)
利用小波变换（WT）提出了WTConv层，这是一种新的CNN层，能够在不大幅增加参数的情况下显著增加感受野。WTConv层通过在小波域中进行卷积操作，实现了对输入数据的多频率响应，这使得网络能够更好地捕捉低频信息，从而提高了对形状的敏感性，并增强了网络的鲁棒性

通过小波变换将输入图像分解为不同的频率成分，并在每个频率层上进行小尺寸卷积，最后通过逆小波变换将结果重新组合，从而实现对图像的多尺度分析

H-RAMi 技术文档 
一种基于层次结构的循环注意力机制，通常用于处理时序数据和复杂的输入特征。它结合了层次结构和注意力机制，旨在通过递归的方式加强模型对重要信息的关注，并且通过层次化的处理方式提高模型的效率和性能。通过对图像内容的层次化关注，提高模型在复杂视觉任务中的表现。

