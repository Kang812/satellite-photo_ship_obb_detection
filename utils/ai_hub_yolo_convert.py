import cv2
import os
import json
from tqdm import tqdm
from glob import glob
from pycocotools.coco import COCO
import matplotlib.pyplot  as plt

def aihub_convert_yolo_format(json_paths, data_dir, split, save_dir):
    for i in tqdm(range(len(json_paths))):
        json_path = json_paths[i]
        
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        features = annotations['features']
        
        for feature in features:
            if 'ship' in feature['properties']['type_name']:
                name, ext = feature['properties']['image_id'].split(".")

                if not os.path.exists(os.path.join(save_dir, 'images', split)):
                    os.makedirs(os.path.join(save_dir, 'images', split))
                
                if not os.path.exists(os.path.join(save_dir, 'labels', split)):
                    os.makedirs(os.path.join(save_dir, 'labels', split))

                image_path = os.path.join(data_dir, name + "." + ext)
                img = cv2.imread(image_path)
                cv2.imwrite(os.path.join(save_dir, 'images', split, name + '.jpg'), img)
                height, width = img.shape[:2]

                coords = [float(i) for i in feature['properties']['object_imcoords'].split(", ")]
                normalized_coords = [coords[i] / width if i % 2 == 0 else coords[i] / height for i in range(len(coords))]
                yolo_format = f"0 {normalized_coords[0]} {normalized_coords[1]} {normalized_coords[2]} {normalized_coords[3]} {normalized_coords[4]} {normalized_coords[5]} {normalized_coords[6]} {normalized_coords[7]}"

                with open(os.path.join(save_dir, 'labels', split, name + '.txt'), "a") as f:
                    f.write(yolo_format + "\n")




if __name__ == '__main__':
    #train
    print("Train:")
    json_paths = glob("/workspace/oriented_dota/oriented_object_detection/original_data/train/labels/*.json")
    data_dir = '/workspace/oriented_dota/oriented_object_detection/original_data/train/images/'
    split = 'train'
    save_dir = '/workspace/oriented_dota/oriented_object_detection/dataset/'
    aihub_convert_yolo_format(json_paths, data_dir, split, save_dir)

    #val
    print("Valid:")
    json_paths = glob("/workspace/oriented_dota/oriented_object_detection/original_data/val/labels/*.json")
    data_dir = '/workspace/oriented_dota/oriented_object_detection/original_data/val/images/'
    split = 'val'
    save_dir = '/workspace/oriented_dota/oriented_object_detection/dataset/'
    aihub_convert_yolo_format(json_paths, data_dir, split, save_dir)
