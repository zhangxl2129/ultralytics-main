import io
import os
import cv2
import glob
import timeit
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from ultralytics.mode.EDRNet import EDRNet
from ultralytics.mode.data_loader import RescaleT, ToTensorLab, SalObjDataset


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


# def save_output(image_name, pred, d_dir):
#     predict = pred.squeeze()
#     predict_np = predict.cpu().data.numpy()
#     im = Image.fromarray((predict_np * 255).astype(np.uint8)).convert('RGB')
#     image = io.imread(image_name)
#     imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
#     img_name = os.path.basename(image_name)
#     imidx = img_name.split(".")[0]
#     imo.save(os.path.join(d_dir, f"{imidx}.png"))
def save_output(image_name, pred, d_dir):
    # 处理预测结果，避免无效值
    predict = pred.clamp(0, 1)  # 确保值在0到1之间
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    # 使用PIL读取图像
    image = Image.open(image_name).convert('RGB')
    imo = Image.fromarray((predict_np * 255).astype(np.uint8)).convert('RGB')
    imo = imo.resize((image.size[0], image.size[1]), resample=Image.BILINEAR)

    img_name = os.path.basename(image_name).split(".")[0]
    imo.save(os.path.join(d_dir, img_name + '.png'))

# 其他部分保持不变


# def edr_segment(image, model):
#     input_tensor = transforms.Compose([
#         RescaleT(256),
#         ToTensorLab(flag=0)
#     ])(image)  # 应用您定义的预处理
#
#     with torch.no_grad():
#         s_out, _, _, _, _, _, _ = model(input_tensor.unsqueeze(0).cuda())
#         pred = s_out[:, 0, :, :]
#         return normPRED(pred)
def edr_segment(image, model):
    # 将图像转换为PIL格式
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整大小
        transforms.ToTensor(),            # 转换为Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正则化
    ])

    input_tensor = preprocess(pil_image).unsqueeze(0).cuda()  # 增加一个维度并转移到GPU

    with torch.no_grad():
        s_out, _, _, _, _, _, _ = model(input_tensor)
        pred = s_out[:, 0, :, :]  # 选择第一个通道
        return normPRED(pred)




def predict_and_segment(image_folder, save_dir, model_det, model_seg):
    os.makedirs(save_dir, exist_ok=True)
    img_name_list = glob.glob(os.path.join(image_folder, '*.jpg'))

    for image_path in img_name_list:
        print(f"Processing: {image_path}")
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # 使用YOLO进行目标检测
        detection_results = model_det.predict(source=image_path, save=False)

        if detection_results and len(detection_results) > 0:
            det_result = detection_results[0]
            if det_result.boxes is not None:
                for box in det_result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0].item())
                    cropped_image = image[y1:y2, x1:x2]

                    # 使用EDR-Net进行分割
                    seg_mask = edr_segment(cropped_image, model_seg)

                    # 保存结果
                    save_output(image_path, seg_mask, save_dir)


if __name__ == "__main__":
    image_folder = "E:\\Projects\\trainingr\\test_decoded_images"
    save_dir = "runs\\predictions\\"

    detection_model = YOLO("runs/detect/train15/weights/best.pt")
    segmentation_model = EDRNet(in_channels=3)  # 根据您定义的模型构造实例
    segmentation_model.load_state_dict(
        torch.load("runs/segment/EDRNet_epoch_600_trnloss_nan_priloss_0.614606.pth", map_location='cpu'))
    segmentation_model.cuda()
    segmentation_model.eval()

    start = timeit.default_timer()
    predict_and_segment(image_folder, save_dir, detection_model, segmentation_model)
    end = timeit.default_timer()
    print(f"Processing time: {end - start:.2f} seconds")
