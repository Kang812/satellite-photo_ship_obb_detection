import json
import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm

def airbus_ship_convert_yolo_format(image_path, image_format, split, save_dir):
    
    json_path = image_path[:image_path.rfind(image_format)] + "json"
    filename = image_path.split("/")[-1]

    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    with open(json_path, "r") as f:
        annotation = json.load(f)
    
    if not os.path.exists(os.path.join(save_dir, 'images', split)):
        os.makedirs(os.path.join(save_dir, 'images', split))
    
    if not os.path.exists(os.path.join(save_dir, 'labels', split)):
        os.makedirs(os.path.join(save_dir, 'labels', split))

    shapes = annotation['shapes']
    with open(os.path.join(save_dir, 'labels', split, filename[:filename.rfind(image_format)] + 'txt'), "a") as f:
        for shape in shapes:
            points = shape['points']
            if points != []:
                coords = list(np.ravel(np.array(points)))
                normalized_coords = [coords[i] / width if i % 2 == 0 else coords[i] / height for i in range(len(coords))]
                yolo_format = f"0 {normalized_coords[0]} {normalized_coords[1]} {normalized_coords[2]} {normalized_coords[3]} {normalized_coords[4]} {normalized_coords[5]} {normalized_coords[6]} {normalized_coords[7]}"
            else:
                yolo_format = ""
            f.write(yolo_format + "\n")
    cv2.imwrite(os.path.join(save_dir, 'images', split, filename), img)

if __name__ == '__main__':
    # train
    print("Train:")
    train_img_paths = glob("/workspace/oriented_dota/oriented_object_detection/original_data/airbus_ship/train/*.jpg")
    print("데이터 수 :", len(train_img_paths))
    image_format = 'jpg'
    save_dir = '/workspace/oriented_dota/oriented_object_detection/dataset/'
    for i in tqdm(range(len(train_img_paths))):
        airbus_ship_convert_yolo_format(train_img_paths[i], image_format, 'train', save_dir)

    print("val:")
    val_img_paths = glob("/workspace/oriented_dota/oriented_object_detection/original_data/airbus_ship/valid/*.jpg")
    print("데이터 수 :", len(val_img_paths))
    image_format = 'jpg'
    save_dir = '/workspace/oriented_dota/oriented_object_detection/dataset/'
    for i in tqdm(range(len(val_img_paths))):
        airbus_ship_convert_yolo_format(val_img_paths[i], image_format, 'val', save_dir)

    print("test:")
    test_img_paths = glob("/workspace/oriented_dota/oriented_object_detection/original_data/airbus_ship/test/*.jpg")
    print("데이터 수 :", len(test_img_paths))
    image_format = 'jpg'
    save_dir = '/workspace/oriented_dota/oriented_object_detection/dataset/'
    for i in tqdm(range(len(test_img_paths))):
        airbus_ship_convert_yolo_format(test_img_paths[i], image_format, 'test', save_dir)
