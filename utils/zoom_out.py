import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from shutil import copyfile
import os

def zoom_out(image_path, scale_factor, data):
    # 이미지 읽기
    image = cv2.imread(image_path)
    
    if image is None:
        print("이미지를 불러올 수 없습니다.")
        return

    # 이미지 크기 가져오기
    coordinates = []
    zoomed_out_coordinates = []
    original_height, original_width = image.shape[:2]
    for d in data:
        d = d.split(" ")
        d = d[1:]
        d = [float(d[i]) * original_width if i % 2 == 0 else float(d[i]) * original_height for i in range(len(d))]
        p_x = [d[i] for i in range(len(d)) if i % 2 == 0]
        p_y = [d[i] for i in range(len(d)) if i % 2 == 1]
        d = [(x, y) for x, y in zip(p_x, p_y)]
        coordinates.append(d)
    
    # 줌 아웃할 새로운 크기 계산
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # 크기 조정
    zoomed_out_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # 패딩 추가하여 원본 크기 유지
    pad_top = (original_height - new_height) // 2
    pad_bottom = original_height - new_height - pad_top
    pad_left = (original_width - new_width) // 2
    pad_right = original_width - new_width - pad_left

    padded_image = cv2.copyMakeBorder(zoomed_out_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 좌표 변환
    for coordinate in coordinates:
        zoomed_out_coordinates.append([
            (int(point[0] * scale_factor) + pad_left, int(point[1] * scale_factor) + pad_top)
            for point in coordinate
        ])

    return zoomed_out_coordinates, padded_image

def zoom_out_aug_apply(image_paths, split, scale_factors, save_dir):
    if not os.path.exists(os.path.join(save_dir, 'images', split)):
        os.makedirs(os.path.join(save_dir, 'images', split))
    
    if not os.path.exists(os.path.join(save_dir, 'labels', split)):
        os.makedirs(os.path.join(save_dir, 'labels', split))
    
    for i in tqdm(range(len(image_paths))):
        image_path = image_paths[i]
        file_name = image_path.split("/")[-1]
        name = file_name[:image_path.rfind("jpg")]
        format_index = image_path.rfind("jpg")
        txt_path = image_path.replace("images", "labels")[:format_index] + "txt"
        
        copyfile(image_path, os.path.join(save_dir, 'images', split, file_name))
        copyfile(txt_path, os.path.join(save_dir, 'labels', split, txt_path.split("/")[-1]))

        with open(txt_path, "r") as f:
            data= f.readlines()
        
        for scale_factor in scale_factors:
            zoomed_out_coordinates, zoomed_out_image = zoom_out(image_path, scale_factor, data)
            height, width = zoomed_out_image.shape[:2]
            cv2.imwrite(os.path.join(save_dir, 'images', split, name + f"_{scale_factor}"+ '.jpg'), zoomed_out_image)
            
            for coords in zoomed_out_coordinates:
                coords = np.ravel(coords)
                normalized_coords = [coords[i] / width if i % 2 == 0 else coords[i] / height for i in range(len(coords))]
                yolo_format = f"0 {normalized_coords[0]} {normalized_coords[1]} {normalized_coords[2]} {normalized_coords[3]} {normalized_coords[4]} {normalized_coords[5]} {normalized_coords[6]} {normalized_coords[7]}"

                with open(os.path.join(save_dir, 'labels', split, name + f"_{scale_factor}"+ '.txt'), "a") as f:
                    f.write(yolo_format + "\n")

if __name__ == "__main__":
    image_paths = glob("/workspace/oriented_dota/oriented_object_detection/dataset2/images/train/*.jpg")
    scale_factors = [0.4, 0.5]
    save_dir = "/workspace/oriented_dota/oriented_object_detection/dataset3/"
    split = 'train'

    zoom_out_aug_apply(image_paths, split, scale_factors, save_dir)
