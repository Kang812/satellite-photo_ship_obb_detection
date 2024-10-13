
import numpy as np
import base64
import os
import cv2
import json
from glob import glob
from ultralytics import YOLO
from tqdm import tqdm

def auto_labeling_labelme_format(model, image_path, save_dir):
    labelme_dict = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": "0ae49bc36_jpg.rf.ae54a3c6973a26192734e383bdf941ee.jpg",
        "imageData": "",
        "imageHeight": 640,
        "imageWidth": 640}
    
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    results = model(img)

    shapes = []
    for result in results:
        polygon = result.obb.xyxyxyxy.detach().cpu().numpy() 
        conf = result.obb.conf.detach().cpu().numpy()
        for p , conf in zip(polygon, conf):
            if conf > 0.45:
                point_dict = { "label": "ship", "points": [], "group_id": None, "description": "", "shape_type": "polygon", "flags": {}, "mask": None}
                polygon_all = list(np.ravel(p))
                polygon_x = [polygon_all[i] for i in range(len(polygon_all)) if i % 2 == 0]
                polygon_y = [polygon_all[i] for i in range(len(polygon_all)) if i % 2 == 1]
                point = [[float(x), float(y)] for x, y in zip(polygon_x, polygon_y)]
                point_dict['points'] = point
                shapes.append(point_dict)
    
    imageData = base64.b64encode(open(image_path, "rb").read()).decode('utf-8')
    labelme_dict['shapes'] = shapes
    labelme_dict['imageHeight'] = height
    labelme_dict['imageWidth'] = width
    labelme_dict['imagePath'] = image_path.split("/")[-1]
    labelme_dict['imageData'] = imageData

    with open(os.path.join(save_dir, image_path.split("/")[-1].replace(".jpg", "") + ".json"), "w") as json_file:
        json.dump(labelme_dict, json_file)

if __name__ == '__main__':
    model_path = "/workspace/oriented_dota/runs/obb/train2/weights/best.pt"
    model = YOLO(model_path)
    
    image_paths = glob("/workspace/oriented_dota/oriented_object_detection/dataset/images/test/*.jpg")
    save_dir = '/workspace/oriented_dota/oriented_object_detection/original_data/airbus_ship/test_json/'
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        auto_labeling_labelme_format(model, image_path, save_dir)

