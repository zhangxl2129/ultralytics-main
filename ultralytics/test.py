import os
import warnings
import time
import numpy as np
import torch
from ultralytics import YOLO
from mode import CustomYOLO

warnings.filterwarnings('ignore')

def convert_boxes_to_npy(boxes, img_size=(200, 200)):
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
    best_model = 'runs/detect/train2/weights/best.pt'  # Change to your model path 18
    model = CustomYOLO(best_model)

    # Save predictions directory
    save_dir = 'E:\Projects\YOLOV10\DataB/predictions/'  # Change to your desired save path
    os.makedirs(save_dir, exist_ok=True)

    # Validation dataset image path
    val_dataset = 'E:\Projects\YOLOV10\DataB\\test\images'
    image_files = [os.path.join(val_dataset, img) for img in os.listdir(val_dataset) if img.endswith('.jpg')]

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

            boxes_with_class = np.hstack((boxes_xywh[:, :4], class_ids.reshape(-1, 1)))

            # Convert each box to a shape of (200, 200) array
            mask = convert_boxes_to_npy(boxes_with_class, img_size=(200, 200))
            npy_file_path = os.path.join(save_dir, os.path.basename(img_path).replace('.jpg', '.npy'))
            np.save(npy_file_path, mask)  # Save as numpy format

    # Calculate and print FPS
    fps = len(image_files) / total_time if total_time > 0 else 0
    print(f'FPS: {fps:.2f}')

