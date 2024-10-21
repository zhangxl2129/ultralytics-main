import numpy as np
import cv2
import json
import os
import glob
from tqdm import tqdm
import base64


def decode_image_data(image_data):
    image_data = base64.b64decode(image_data)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def create_mask(img_shape, bbox_list):
    """
    Create a mask for each defect type defined in the bounding box list
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    for bbox in bbox_list:
        x, y, w, h, label = bbox
        # Ensure the box is within image boundaries
        x1 = int(max(0, x))
        y1 = int(max(0, y))
        x2 = int(min(img_shape[1], x1 + w))
        y2 = int(min(img_shape[0], y1 + h))
        mask[y1:y2, x1:x2] = label  # Fill the rectangle with the defect label

    return mask


def convert_labelme_json_to_npy(json_path, out_npy_path, decoded_img_path):
    json_list = glob.glob(json_path + '/*.json')

    for json_file in tqdm(json_list):
        with open(json_file, "r") as f_json:
            try:
                json_data = json.loads(f_json.read())
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file {json_file}: {e}")
                continue

        if 'imageData' in json_data:
            img = decode_image_data(json_data['imageData'])
        else:
            print(f"No imageData found in {json_file}, skipping.")
            continue

        img = cv2.resize(img, (200, 200))
        infos = json_data.get('shapes', [])

        if not infos:
            continue

        base_name = os.path.basename(json_file).replace('.json', '')
        image_name = base_name + '.jpg'
        image_path = os.path.join(decoded_img_path, image_name)

        cv2.imwrite(image_path, img)

        bbox_list = []
        for label in infos:
            points = label['points']
            if len(points) < 3:  # Must have at least 3 points for a polygon
                continue

            # Map label from JSON to defect types
            defect_label = int(label['label'])  # Get defect type from label

            segmentation = [(int(p[0]), int(p[1])) for p in points]

            # Calculate bounding box
            x_min = min(segmentation, key=lambda p: p[0])[0]
            x_max = max(segmentation, key=lambda p: p[0])[0]
            y_min = min(segmentation, key=lambda p: p[1])[1]
            y_max = max(segmentation, key=lambda p: p[1])[1]
            w = x_max - x_min
            h = y_max - y_min

            bbox_list.append([x_min, y_min, w, h, defect_label])

        # Create mask array
        mask = create_mask((200, 200), bbox_list)

        npy_name = base_name + '.npy'
        npy_path = os.path.join(out_npy_path, npy_name)
        np.save(npy_path, mask)


if __name__ == "__main__":
    json_path = r'E:\Projects\trainingr\test_jsons'
    out_npy_path = r'E:\Projects\trainingr\test_labels_npy'
    decoded_img_path = r'E:\Projects\trainingr\test_decoded_images'

    if not os.path.exists(out_npy_path):
        os.makedirs(out_npy_path)

    if not os.path.exists(decoded_img_path):
        os.makedirs(decoded_img_path)

    convert_labelme_json_to_npy(json_path, out_npy_path, decoded_img_path)
