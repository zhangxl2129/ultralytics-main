import os
import warnings
import time
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

warnings.filterwarnings('ignore')


def convert_boxes_to_npy(boxes, img_size=(200, 200)):
    """
    Convert bounding boxes to a numpy array of shape (200, 200).

    Args:
        boxes: Array of bounding boxes in format (n, 5) where each box is [x_center, y_center, width, height, class].
        img_size: Size of the output array.

    Returns:
        np.ndarray: Array of shape (200, 200) with uint8 type.
    """
    # Create an empty image with background class
    mask = np.zeros(img_size, dtype=np.uint8)

    # Fill the mask with class values based on bounding boxes
    for box in boxes:
        x_center, y_center, width, height, class_id = box
        x1 = int(max(0, x_center - width / 2))
        y1 = int(max(0, y_center - height / 2))
        x2 = int(min(img_size[1] - 1, x_center + width / 2))
        y2 = int(min(img_size[0] - 1, y_center + height / 2))

        # Fill the mask with the class value
        mask[y1:y2, x1:x2] = class_id  # Use class_id for the mask

    return mask


if __name__ == '__main__':
    # Load the best model weights
    best_model = 'runs/detect/train16/weights/best.pt'  # Change to your model path
    model = YOLO(best_model)

    # Create a directory to save predictions
    predict_dir = 'E:/Projects/YOLOV10/predict'
    os.makedirs(predict_dir, exist_ok=True)

    # Validation dataset image path
    val_dataset = 'E:/Projects/YOLOV10/NEU-Seg-New/test/images'
    image_files = [os.path.join(val_dataset, img) for img in os.listdir(val_dataset) if img.endswith('.jpg')]

    # Save predictions directory
    save_dir = 'E:/Projects/YOLOV10/NEU-Seg-New/test/predictions/'  # Change to your desired save path
    os.makedirs(save_dir, exist_ok=True)

    total_time = 0  # For calculating FPS
    for img_path in image_files:
        start_time = time.time()  # Start timing

        # Predict using YOLO
        results = model.predict(source=img_path, save=False)  # Do not save predicted images

        end_time = time.time()  # End timing
        total_time += (end_time - start_time)

        # Get predicted boxes and save as .npy file
        if results and len(results) > 0:
            result = results[0]  # Get the first result
            boxes = result.boxes  # Get predicted bounding boxes
            boxes_xywh = boxes.xywh.cpu().numpy()  # Convert to numpy array

            # Extract class IDs using cls attribute
            class_ids = boxes.cls.cpu().numpy()  # Use the cls attribute to get class IDs

            # Ensure class_ids are integers
            class_ids = class_ids.astype(np.int32)

            # Add one to class_ids to ensure classes start from 1 (background remains 0)
            #class_ids += 1

            # Check if we have only 0s and 1s
            print("Unique class ids in this image:", np.unique(class_ids))

            boxes_with_class = np.hstack((boxes_xywh[:, :4], class_ids.reshape(-1, 1)))

            # Convert each box to a shape of (200, 200) array
            mask = convert_boxes_to_npy(boxes_with_class, img_size=(200, 200))
            npy_file_path = os.path.join(save_dir, os.path.basename(img_path).replace('.jpg', '.npy'))

            np.save(npy_file_path, mask)  # Save as numpy format
            print(f"Prediction saved for image: {img_path} as {npy_file_path}.")

    # Calculate FPS
    total_images = len(image_files)
    fps = total_images / total_time
    print(f"FPS (Frames per second): {fps}")

    # Calculate the number of model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.model.parameters())
    param_count = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Model parameter count: {param_count}")

# import os
# import warnings
# import time
# import numpy as np
# import torch
# from ultralytics import YOLO
# from PIL import Image
# import cv2  # 用于图像处理和显示
# from segment_anything import SamPredictor
#
# warnings.filterwarnings('ignore')
#
#
# def convert_boxes_to_npy(masks, img_size=(200, 200)):
#     """
#     Convert segmentation masks to a numpy array of shape (200, 200).
#
#     Args:
#         masks: Boolean mask for the defect.
#         img_size: Size of the output array.
#
#     Returns:
#         np.ndarray: Array of shape (200, 200) with uint8 type.
#     """
#     # Resize masks to match output size
#     mask_resized = cv2.resize(masks.astype(np.uint8), img_size, interpolation=cv2.INTER_NEAREST)
#
#     return mask_resized
#
#
# def add_mask(image, mask, color=(0, 255, 0), alpha=0.5):
#     """在图像上叠加透明的分割掩码"""
#     overlay = image.copy()
#     image[mask] = cv2.addWeighted(overlay[mask], alpha, np.array(color, dtype=np.uint8), 1 - alpha, 0)
#     return image
#
#
# def show_box(image, box, color=(255, 0, 0), thickness=2):
#     """在图像上绘制矩形框"""
#     x1, y1, x2, y2 = map(int, box)
#     return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
#
#
# if __name__ == '__main__':
#     # 加载YOLO模型
#     best_model = 'runs/detect/train8/weights/best.pt'  # YOLOv10权重文件路径
#     model = YOLO(best_model)
#
#     # 加载MobileSAM模型
#     checkpoint = torch.load('MobileSAM-master/weights/mobile_sam.pt', map_location=torch.device('cpu'))
#     mobile_sam = setup_model()
#     mobile_sam.load_state_dict(checkpoint, strict=True)
#     device = "cpu"
#     mobile_sam.to(device=device)
#     mobile_sam.eval()
#     predictor = SamPredictor(mobile_sam)
#
#     # 创建保存预测的文件夹
#     predict_dir = 'E:/Projects/YOLOV10/predict'
#     os.makedirs(predict_dir, exist_ok=True)
#
#     # 验证集图片路径
#     val_dataset = 'E:/Projects/YOLOV10/NEU-Seg/test/images'
#     image_files = [os.path.join(val_dataset, img) for img in os.listdir(val_dataset) if img.endswith('.jpg')]
#
#     # 保存预测结果的文件夹
#     save_dir = 'E:/Projects/YOLOV10/NEU-Seg/test/predictions/'
#     os.makedirs(save_dir, exist_ok=True)
#
#     total_time = 0  # 计算FPS
#     for img_path in image_files:
#         start_time = time.time()  # 开始计时
#
#         # 使用YOLO进行预测
#         results = model.predict(source=img_path, save=False)  # 不保存预测图片
#
#         end_time = time.time()  # 结束计时
#         total_time += (end_time - start_time)
#
#         # 读取图片
#         image = cv2.imread(img_path)
#
#         # 获取YOLO预测的检测框和类ID
#         if results and len(results) > 0:
#             result = results[0]  # 获取第一个结果
#             boxes = result.boxes  # 获取检测框
#             boxes_xywh = boxes.xywh.cpu().numpy()  # 转为numpy数组
#
#             # 获取类别ID
#             class_ids = boxes.cls.cpu().numpy().astype(np.int32)
#
#             # 遍历每个检测框，使用MobileSAM分割目标
#             for i, box in enumerate(boxes_xywh):
#                 x_center, y_center, width, height = box[:4]
#                 x1 = int(max(0, x_center - width / 2))
#                 y1 = int(max(0, y_center - height / 2))
#                 x2 = int(min(image.shape[1] - 1, x_center + width / 2))
#                 y2 = int(min(image.shape[0] - 1, y_center + height / 2))
#
#                 input_box = np.array([x1, y1, x2, y2])
#
#                 # 使用MobileSAM进行分割
#                 predictor.set_image(image)
#                 masks, _, _ = predictor.predict(
#                     point_coords=None,
#                     point_labels=None,
#                     box=input_box[None, :],
#                     multimask_output=False,
#                 )
#
#                 # 显示分割掩码和检测框
#                 image_with_mask = add_mask(image.copy(), masks[0])
#                 image_with_box = show_box(image_with_mask, input_box)
#
#                 # 保存分割结果图片
#                 result_img_path = os.path.join(save_dir, os.path.basename(img_path).replace('.jpg', f'_seg_{i}.jpg'))
#                 cv2.imwrite(result_img_path, image_with_box)
#                 print(f"Segmentation saved for image: {img_path} as {result_img_path}.")
#
#                 # 将分割掩码保存为.npy文件
#                 mask_resized = convert_boxes_to_npy(masks[0], img_size=(200, 200))
#                 npy_file_path = os.path.join(save_dir, os.path.basename(img_path).replace('.jpg', f'_seg_{i}.npy'))
#                 np.save(npy_file_path, mask_resized)
#                 print(f"Segmentation mask saved for image: {img_path} as {npy_file_path}.")
#
#     # 计算FPS
#     total_images = len(image_files)
#     fps = total_images / total_time
#     print(f"FPS (Frames per second): {fps}")


# import os
# import warnings
# import time
# import numpy as np
# import torch
# from ultralytics import YOLO
# from PIL import Image
#
# warnings.filterwarnings('ignore')
#
#
# def convert_boxes_to_npy(boxes, img_size=(200, 200)):
#     """
#     Convert bounding boxes to a numpy array of shape (200, 200).
#
#     Args:
#         boxes: Array of bounding boxes in format (n, 5) where each box is [x_center, y_center, width, height, class].
#         img_size: Size of the output array.
#
#     Returns:
#         np.ndarray: Array of shape (200, 200) with uint8 type.
#     """
#     # Create an empty image with background class
#     mask = np.zeros(img_size, dtype=np.uint8)
#
#     # Fill the mask with class values based on bounding boxes
#     for box in boxes:
#         x_center, y_center, width, height, class_id = box
#         x1 = int(max(0, x_center - width / 2))
#         y1 = int(max(0, y_center - height / 2))
#         x2 = int(min(img_size[1] - 1, x_center + width / 2))
#         y2 = int(min(img_size[0] - 1, y_center + height / 2))
#
#         # Fill the mask with the class value
#         mask[y1:y2, x1:x2] = class_id  # Use class_id for the mask
#
#     return mask
#
#
# if __name__ == '__main__':
#     # Load the best model weights
#     best_model = 'runs/detect/train8/weights/best.pt'  # Change to your model path
#     model = YOLO(best_model)
#
#     # Create a directory to save predictions
#     predict_dir = 'E:/Projects/YOLOV10/predict'
#     os.makedirs(predict_dir, exist_ok=True)
#
#     # Validation dataset image path
#     val_dataset = 'E:/Projects/YOLOV10/NEU-Seg/test/images'
#     image_files = [os.path.join(val_dataset, img) for img in os.listdir(val_dataset) if img.endswith('.jpg')]
#
#     # Save predictions directory
#     save_dir = 'E:/Projects/YOLOV10/NEU-Seg/test/predictions/'  # Change to your desired save path
#     os.makedirs(save_dir, exist_ok=True)
#
#     total_time = 0  # For calculating FPS
#     for img_path in image_files:
#         start_time = time.time()  # Start timing
#
#         # Predict using YOLO
#         results = model.predict(source=img_path, save=False)  # Do not save predicted images
#
#         end_time = time.time()  # End timing
#         total_time += (end_time - start_time)
#
#         # Get predicted boxes and save as .npy file
#         if results and len(results) > 0:
#             result = results[0]  # Get the first result
#             boxes = result.boxes  # Get predicted bounding boxes
#             boxes_xywh = boxes.xywh.cpu().numpy()  # Convert to numpy array
#
#             # Extract class IDs using cls attribute
#             class_ids = boxes.cls.cpu().numpy()  # Use the cls attribute to get class IDs
#
#             # Ensure class_ids are integers
#             class_ids = class_ids.astype(np.int32)
#
#             # Add one to class_ids to ensure classes start from 1 (background remains 0)
#             class_ids += 1
#
#             # Check if we have only 0s and 1s
#             print("Unique class ids in this image:", np.unique(class_ids))
#
#             boxes_with_class = np.hstack((boxes_xywh[:, :4], class_ids.reshape(-1, 1)))
#
#             # Convert each box to a shape of (200, 200) array
#             mask = convert_boxes_to_npy(boxes_with_class, img_size=(200, 200))
#             npy_file_path = os.path.join(save_dir, os.path.basename(img_path).replace('.jpg', '.npy'))
#
#             np.save(npy_file_path, mask)  # Save as numpy format
#             print(f"Prediction saved for image: {img_path} as {npy_file_path}.")
#
#     # Calculate FPS
#     total_images = len(image_files)
#     fps = total_images / total_time
#     print(f"FPS (Frames per second): {fps}")
#
#     # Calculate the number of model parameters
#     model_parameters = filter(lambda p: p.requires_grad, model.model.parameters())
#     param_count = sum([np.prod(p.size()) for p in model_parameters])
#     print(f"Model parameter count: {param_count}")
#
