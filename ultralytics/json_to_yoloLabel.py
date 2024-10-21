import json
import os
import glob
from tqdm import tqdm
import cv2
import base64
import numpy as np

def convert_poly_to_rect(coordinateList):
    """
    将多边形坐标转换为矩形框
    """
    X = [int(coordinateList[2 * i]) for i in range(int(len(coordinateList) / 2))]
    Y = [int(coordinateList[2 * i + 1]) for i in range(int(len(coordinateList) / 2))]

    Xmax = max(X)
    Xmin = min(X)
    Ymax = max(Y)
    Ymin = min(Y)

    flag = False
    # 如果生成的框大小不合理（宽或高为0），则忽略该框
    if (Xmax - Xmin) == 0 or (Ymax - Ymin) == 0:
        flag = True

    return [Xmin, Ymin, Xmax - Xmin, Ymax - Ymin], flag

def draw_bbox_on_image(image, bbox_list, output_img_path):
    """
    在图片上绘制检测框并保存
    """
    img = np.array(image)
    if img is None:
        print(f"Failed to load image.")
        return

    for bbox in bbox_list:
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        # 仅绘制合理的检测框，避免超出图片边界的情况
        if w > 0 and h > 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 保存标注后的图片
    cv2.imwrite(output_img_path, img)

def decode_image_data(image_data):
    """
    将 base64 编码的 imageData 字符串解码为 OpenCV 图像格式
    """
    image_data = base64.b64decode(image_data)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def convert_labelme_json_to_txt(json_path, out_txt_path, out_img_path, decoded_img_path):
    """
    将LabelMe格式的JSON文件转换为YOLO格式的TXT标签，并绘制检测框，同时将base64图像数据转换为jpg格式
    """
    json_list = glob.glob(json_path + '/*.json')

    for json_file in tqdm(json_list):
        with open(json_file, "r") as f_json:
            try:
                json_data = json.loads(f_json.read())
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file {json_file}: {e}")
                continue

        # 如果json包含imageData字段，将其转换为图片
        if 'imageData' in json_data:
            img = decode_image_data(json_data['imageData'])
        else:
            print(f"No imageData found in {json_file}, skipping.")
            continue

        # 从JSON文件中获取图像的宽度和高度
        img_w = json_data.get('imageWidth', img.shape[1])
        img_h = json_data.get('imageHeight', img.shape[0])

        # 使用json文件名作为生成的图片名和txt文件名
        base_name = os.path.basename(json_file).replace('.json', '')
        image_name = base_name + '.jpg'
        image_path = os.path.join(decoded_img_path, image_name)

        # 保存从imageData转换出的jpg图像
        cv2.imwrite(image_path, img)

        # 创建对应的TXT文件
        txt_name = base_name + '.txt'
        txt_path = os.path.join(out_txt_path, txt_name)

        bbox_list = []

        with open(txt_path, 'w') as f:
            for label in json_data.get('shapes', []):
                points = label['points']
                if len(points) < 2:
                    continue

                # 若为矩形，补全四个角点
                if len(points) == 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                segmentation = []
                for p in points:
                    segmentation.append(int(p[0]))
                    segmentation.append(int(p[1]))

                # 转换为YOLO格式
                bbox, flag = convert_poly_to_rect(segmentation)
                if flag:
                    continue

                x1, y1, w, h = bbox

                # 计算中心点和宽高的比例，按原始图像尺寸计算
                x_center = x1 + w / 2
                y_center = y1 + h / 2
                norm_x = x_center / img_w
                norm_y = y_center / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                obj_cls = label['label']
                line = [obj_cls, norm_x, norm_y, norm_w, norm_h]
                line = ' '.join([str(ll) for ll in line]) + '\n'
                f.write(line)

                # 收集标注框以便绘制
                bbox_list.append([x1, y1, w, h])

        # 在图片上绘制检测框
        output_img_path = os.path.join(out_img_path, image_name)
        draw_bbox_on_image(img, bbox_list, output_img_path)

if __name__ == "__main__":
    json_path = r'E:\Projects\trainingr\train_Jsons'  # JSON文件夹路径
    out_txt_path = r'E:\Projects\trainingr\train_labels'  # 输出标签文件夹路径
    out_img_path = r'E:\Projects\trainingr\train_labeled_images'  # 输出带标注的图片文件夹路径
    decoded_img_path = r'E:\Projects\trainingr\train_decoded_images'  # 从json imageData转换后的原始图片路径

    if not os.path.exists(out_txt_path):
        os.makedirs(out_txt_path)

    if not os.path.exists(out_img_path):
        os.makedirs(out_img_path)

    if not os.path.exists(decoded_img_path):
        os.makedirs(decoded_img_path)

    convert_labelme_json_to_txt(json_path, out_txt_path, out_img_path, decoded_img_path)


# import json
# import os
# import glob
# from tqdm import tqdm
# import cv2
# import base64
# import numpy as np
#
# def convert_poly_to_rect(coordinateList):
#     """
#     将多边形坐标转换为矩形框
#     """
#     X = [int(coordinateList[2 * i]) for i in range(int(len(coordinateList) / 2))]
#     Y = [int(coordinateList[2 * i + 1]) for i in range(int(len(coordinateList) / 2))]
#
#     Xmax = max(X)
#     Xmin = min(X)
#     Ymax = max(Y)
#     Ymin = min(Y)
#
#     flag = False
#     # 如果生成的框大小不合理（宽或高为0），则忽略该框
#     if (Xmax - Xmin) == 0 or (Ymax - Ymin) == 0:
#         flag = True
#
#     return [Xmin, Ymin, Xmax - Xmin, Ymax - Ymin], flag
#
# def draw_bbox_on_image(image, bbox_list, output_img_path):
#     """
#     在图片上绘制检测框并保存
#     """
#     img = np.array(image)
#     if img is None:
#         print(f"Failed to load image.")
#         return
#
#     for bbox in bbox_list:
#         x1, y1, w, h = bbox
#         x2 = x1 + w
#         y2 = y1 + h
#         # 仅绘制合理的检测框，避免超出图片边界的情况
#         if w > 0 and h > 0:
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     # 保存标注后的图片
#     cv2.imwrite(output_img_path, img)
#
# def decode_image_data(image_data):
#     """
#     将 base64 编码的 imageData 字符串解码为 OpenCV 图像格式
#     """
#     image_data = base64.b64decode(image_data)
#     nparr = np.frombuffer(image_data, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return img
#
# def convert_labelme_json_to_txt(json_path, out_txt_path, out_img_path, decoded_img_path):
#     """
#     将LabelMe格式的JSON文件转换为YOLO格式的TXT标签，并绘制检测框，同时将base64图像数据转换为jpg格式
#     """
#     json_list = glob.glob(json_path + '/*.json')
#
#     for json_file in tqdm(json_list):
#         with open(json_file, "r") as f_json:
#             try:
#                 json_data = json.loads(f_json.read())
#             except json.JSONDecodeError as e:
#                 print(f"Error reading JSON file {json_file}: {e}")
#                 continue
#
#         # 如果json包含imageData字段，将其转换为图片
#         if 'imageData' in json_data:
#             img = decode_image_data(json_data['imageData'])
#         else:
#             print(f"No imageData found in {json_file}, skipping.")
#             continue
#
#         # 从JSON文件中获取图像的宽度和高度，默认为200x200
#         img_w = json_data.get('imageWidth', img.shape[1])
#         img_h = json_data.get('imageHeight', img.shape[0])
#
#         # 计算放大比例
#         scale_w = 400 / img_w
#         scale_h = 400 / img_h
#
#         # 将图片放大到400x400分辨率
#         img = cv2.resize(img, (400, 400))
#
#         infos = json_data.get('shapes', [])
#         if len(infos) == 0:
#             continue
#
#         # 使用json文件名作为生成的图片名和txt文件名
#         base_name = os.path.basename(json_file).replace('.json', '')
#         image_name = base_name + '.jpg'
#         image_path = os.path.join(decoded_img_path, image_name)
#
#         # 保存从imageData转换出的jpg图像
#         cv2.imwrite(image_path, img)
#
#         # 创建对应的TXT文件
#         txt_name = base_name + '.txt'
#         txt_path = os.path.join(out_txt_path, txt_name)
#
#         bbox_list = []
#
#         with open(txt_path, 'w') as f:
#             for label in infos:
#                 points = label['points']
#                 if len(points) < 2:
#                     continue
#
#                 # 若为矩形，补全四个角点
#                 if len(points) == 2:
#                     x1, y1 = points[0]
#                     x2, y2 = points[1]
#                     points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
#
#                 segmentation = []
#                 for p in points:
#                     segmentation.append(int(p[0]))
#                     segmentation.append(int(p[1]))
#
#                 # 转换为YOLO格式
#                 bbox, flag = convert_poly_to_rect(segmentation)
#                 if flag:
#                     continue
#
#                 x1, y1, w, h = bbox
#                 # 放大后的坐标
#                 x1 = int(x1 * scale_w)
#                 y1 = int(y1 * scale_h)
#                 w = int(w * scale_w)
#                 h = int(h * scale_h)
#
#                 # 计算中心点和宽高的比例，按400x400计算
#                 x_center = x1 + w / 2
#                 y_center = y1 + h / 2
#                 norm_x = x_center / 400
#                 norm_y = y_center / 400
#                 norm_w = w / 400
#                 norm_h = h / 400
#
#                 obj_cls = label['label']
#                 line = [obj_cls, norm_x, norm_y, norm_w, norm_h]
#                 line = ' '.join([str(ll) for ll in line]) + '\n'
#                 f.write(line)
#
#                 # 收集标注框以便绘制
#                 bbox_list.append([x1, y1, w, h])
#
#         # 在图片上绘制检测框
#         output_img_path = os.path.join(out_img_path, image_name)
#         draw_bbox_on_image(img, bbox_list, output_img_path)
#
# if __name__ == "__main__":
#     json_path = r'E:\Projects\trainingr\train_Jsons'  # JSON文件夹路径
#     out_txt_path = r'E:\Projects\trainingr\train_labels'  # 输出标签文件夹路径
#     out_img_path = r'E:\Projects\trainingr\train_labeled_images'  # 输出带标注的图片文件夹路径
#     decoded_img_path = r'E:\Projects\trainingr\train_decoded_images'  # 从json imageData转换后的原始图片路径
#
#     if not os.path.exists(out_txt_path):
#         os.makedirs(out_txt_path)
#
#     if not os.path.exists(out_img_path):
#         os.makedirs(out_img_path)
#
#     if not os.path.exists(decoded_img_path):
#         os.makedirs(decoded_img_path)
#
#     convert_labelme_json_to_txt(json_path, out_txt_path, out_img_path, decoded_img_path)
