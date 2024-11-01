import os
import cv2
import numpy as np
from ultralytics import YOLO

def apply_mask_to_image(image, mask, class_id, output_array, bbox):
    """
    Apply the defect mask to the original image based on the bounding box area
    and update the output array with defect types.

    Args:
        image (np.ndarray): The original image.
        mask (np.ndarray): The defect mask array (bounding box area).
        class_id (int): The class ID for the bounding box.
        bbox (tuple): Bounding box (x1, y1, x2, y2) where the mask applies.
        output_array (np.ndarray): The array to store defect types.

    Returns:
        np.ndarray: Image with the mask applied in the bounding box area.
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Resize mask to match the bounding box size
    mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    # Update output array with defect types
    output_array[y1:y2, x1:x2][mask_resized == 1] = class_id  # Set class ID for defect

    return output_array

def predict_and_segment(image_path, model_seg, save_dir):
    """
    Perform segmentation on the image and save the output mask array.

    Args:
        image_path (str): Path to the image.
        model_seg: Segmentation model (YOLO-seg).
        save_dir (str): Directory to save the output images.
    """
    # Load the original image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Initialize an output array of zeros with shape (200, 200)
    output_array = np.zeros((200, 200), dtype=np.uint8)

    # Perform segmentation on the image
    seg_results = model_seg.predict(source=image_path, save=False)

    if seg_results and len(seg_results) > 0:
        seg_result = seg_results[0]

        # If the segmentation model outputs a mask
        if seg_result.masks is not None:
            masks = seg_result.masks.data.cpu().numpy()  # Get segmentation masks

            # For each mask, update the output array
            for class_id in range(1, len(masks) + 1):  # Assuming class_id starts from 1
                mask = masks[class_id - 1]  # Get the corresponding mask
                output_array = apply_mask_to_image(image, mask, class_id, output_array, (0, 0, width, height))

    # Save the output array as .npy file
    output_npy_path = os.path.join(save_dir, os.path.basename(image_path).replace('.jpg', '.npy'))
    np.save(output_npy_path, output_array)
    print(f"Saved output mask array: {output_npy_path}")

def detect_and_update_mask(image_path, model_det, output_array, class_colors, save_dir):
    """
    Detect defects in the image and update the output array based on detected bounding boxes.

    Args:
        image_path (str): Path to the image.
        model_det: YOLO detection model.
        output_array (np.ndarray): The array with segmented defects.
        class_colors (dict): Colors to apply to segmented classes.
        save_dir (str): Directory to save the output images.
    """
    # Load the original image
    image = cv2.imread(image_path)

    # Use the detection model to predict objects
    detection_results = model_det.predict(source=image_path, save=False)

    if detection_results and len(detection_results) > 0:
        det_result = detection_results[0]

        if det_result.boxes is not None:
            # Iterate through each detected box
            for box in det_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].item())  # Class of the detection box
                conf = box.conf[0].item()      # Confidence

                print(f"Detected box: ({x1}, {y1}, {x2}, {y2}), class: {cls}, conf: {conf}")

                # Update output array based on bounding box
                output_array[int(y1):int(y2), int(x1):int(x2)][output_array[int(y1):int(y2), int(x1):int(x2)] == 1] = cls  # Update pixels in the bounding box

                # Draw the bounding box on the image
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), class_colors[cls], 2)
                cv2.putText(image, f"Class {cls}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[cls], 2)

        else:
            print(f"No boxes detected in {image_path}")
    else:
        print(f"No detection results for {image_path}")

    # Save the result image
    result_image_path = os.path.join(save_dir, os.path.basename(image_path).replace('.jpg', '_detected.jpg'))
    cv2.imwrite(result_image_path, image)
    print(f"Saved result image with detection: {result_image_path}")

if __name__ == "__main__":
    # Image folder path
    image_folder = "E:\\Projects\\trainingr\\test_decoded_images"  # Replace with your image folder path
    save_dir = "../runs/predictions\\"  # Folder to save results
    os.makedirs(save_dir, exist_ok=True)

    # Load segmentation and detection models
    segmentation_model = YOLO("../runs/segment/train41/weights/best.pt")  # Replace with your YOLO segmentation model path
    detection_model = YOLO("../runs/detect/train16/weights/best.pt")  # Replace with your YOLO detection model path

    # Define color mapping for classes
    class_colors = {
        0: [0, 0, 0],   # Background - Black
        1: [0, 255, 0], # Class 1 - Green
        2: [0, 0, 255], # Class 2 - Red
        3: [255, 0, 0]  # Class 3 - Blue
    }

    # Process each image
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        if image_file.endswith('.jpg'):
            predict_and_segment(image_path, segmentation_model, save_dir)
            output_array = np.load(os.path.join(save_dir, image_file.replace('.jpg', '.npy')))
            detect_and_update_mask(image_path, detection_model, output_array, class_colors, save_dir)
