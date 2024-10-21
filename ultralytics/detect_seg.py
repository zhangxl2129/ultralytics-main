import os
import cv2
import numpy as np
from ultralytics import YOLO

def apply_mask_to_image(image, mask, class_id, class_colors, bbox, output_array):
    """
    Apply the defect mask to the original image based on the bounding box area
    and update the output array with defect types.

    Args:
        image (np.ndarray): The original image.
        mask (np.ndarray): The defect mask array (bounding box area).
        class_id (int): The class ID for the bounding box.
        class_colors (dict): A dictionary mapping class ids to colors.
        bbox (tuple): Bounding box (x1, y1, x2, y2) where the mask applies.
        output_array (np.ndarray): The array to store defect types.

    Returns:
        np.ndarray: Image with the mask applied in the bounding box area.
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Resize mask to match the bounding box size
    mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    # Create a color mask image
    mask_color = np.zeros_like(image[y1:y2, x1:x2])

    # Fill mask with the color corresponding to the class ID
    mask_color[mask_resized == 1] = class_colors[class_id]  # Assuming binary mask (1 for defect)

    # Update output array with defect types
    output_array[y1:y2, x1:x2][mask_resized == 1] = class_id  # Set class ID for defect

    # Blend the mask with the original image
    alpha = 0.5  # Transparency factor
    combined_img = cv2.addWeighted(image[y1:y2, x1:x2], 1, mask_color, alpha, 0)

    # Apply the blended image back to the original image
    image[y1:y2, x1:x2] = combined_img

    return image
def predict_and_segment(image_path, model_det, model_seg, class_colors, save_dir):
    """
    Perform object detection and segmentation on the detected bounding boxes
    and save the output mask array.

    Args:
        image_path (str): Path to the image.
        model_det: YOLO detection model.
        model_seg: Segmentation model (YOLO-seg or other).
        class_colors (dict): Colors to apply to segmented classes.
        save_dir (str): Directory to save the output images.
    """
    # Load the original image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Initialize an output array of zeros with shape (200, 200)
    output_array = np.zeros((200, 200), dtype=np.uint8)

    # Use the detection model to predict objects
    detection_results = model_det.predict(source=image_path, save=False)

    if detection_results and len(detection_results) > 0:
        det_result = detection_results[0]  # Get detection results

        if det_result.boxes is not None:
            # Iterate through each detected box
            for box in det_result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].item())  # Class of the detection box
                #cls = int(box.cls[0].item())+1  # Class of the detection box
                conf = box.conf[0].item()      # Confidence

                print(f"Detected box: ({x1}, {y1}, {x2}, {y2}), class: {cls}, conf: {conf}")

                # Perform segmentation on the area of the detection box
                cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
                seg_results = model_seg.predict(source=cropped_image, save=False)

                if seg_results and len(seg_results) > 0:
                    seg_result = seg_results[0]

                    # If the segmentation model outputs a mask
                    if seg_result.masks is not None:
                        mask = seg_result.masks.data.cpu().numpy()[0]  # Get the first segmentation mask

                        # Apply the mask to the image and ensure class consistency
                        image = apply_mask_to_image(image, mask, cls, class_colors, (x1, y1, x2, y2), output_array)

                        # Draw the bounding box on the image
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), class_colors[cls], 2)
                        cv2.putText(image, f"Class {cls}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[cls], 2)
                    else:
                        # If no mask found, consider the entire bounding box as the defect area
                        print(f"No mask found for bounding box. Treating the entire box as defect for class {cls}.")
                        mask = np.ones((int(y2 - y1), int(x2 - x1)), dtype=np.uint8)  # Create a full mask for the bounding box
                        image = apply_mask_to_image(image, mask, cls, class_colors, (x1, y1, x2, y2), output_array)

                        # Draw the bounding box on the image
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), class_colors[cls], 2)
                        cv2.putText(image, f"Class {cls}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[cls], 2)

                else:
                    print(f"No segmentation results for box: ({x1}, {y1}, {x2}, {y2}), treating as defect.")

        else:
            print(f"No boxes detected in {image_path}")
    else:
        print(f"No detection results for {image_path}")

    # Save the result image
    result_image_path = os.path.join(save_dir, os.path.basename(image_path).replace('.jpg', '_segmented.jpg'))
    cv2.imwrite(result_image_path, image)
    print(f"Saved result image with segmentation: {result_image_path}")

    # Save the output array as .npy file
    output_npy_path = os.path.join(save_dir, os.path.basename(image_path).replace('.jpg', '.npy'))
    np.save(output_npy_path, output_array)
    print(f"Saved output mask array: {output_npy_path}")


if __name__ == "__main__":
    # Image folder path
    image_folder = "E:\\Projects\\trainingr\\test_decoded_images"  # Replace with your image folder path
    save_dir = "runs\\predictions\\"  # Folder to save results
    os.makedirs(save_dir, exist_ok=True)

    # Load detection and segmentation models
    detection_model = YOLO("runs/detect/train15/weights/best.pt")  # Replace with your YOLO detection model path
    segmentation_model = YOLO("runs/segment/train41/weights/best.pt")  # Replace with your YOLO segmentation model path

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
            predict_and_segment(image_path, detection_model, segmentation_model, class_colors, save_dir)
