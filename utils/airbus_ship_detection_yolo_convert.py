import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

def rle_decode(mask_rle, shape=(768, 768)) -> np.array:
    """
    decode run-length encoded segmentation mask
    Assumed all images aRe 768x768 (and ThereforE have the saMe shape)
    """
    
    # if no segmentation mask (nan) return matrix of zeros
    if not mask_rle or pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    # RLE sequence str split to and map to int
    s = list(map(int, mask_rle.split()))

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # indices: 2k - starts, 2k+1 lengths
    starts, lengths = s[0::2], s[1::2]
    for start, length in zip(starts, lengths):
        img[start:start + length] = 1

    return img.reshape(shape).T

def get_obb_coord(img_mask):
    obb_boxes = []
    cnts, hierarchy = cv2.findContours((255*img_mask).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    
    for cnt in cnts:
        rect = cv2.minAreaRect(cnts[0])
        
        if rect[1][1]>rect[1][0]:
            angle = 90-rect[2]
        else:
            angle = -rect[2]

        box = cv2.boxPoints(rect)
        obb_boxes.append(list(np.ravel(box)))
    return obb_boxes

def convert_yolo_format(train_object, data_dir, image_id, split, save_dir):

    img_rle_seqs = train_object.loc[train_object['ImageId'] == image_id]['EncodedPixels']
    img = cv2.imread(os.path.join(data_dir, image_id))
    height, width = img.shape[:2]
    img_mask = np.zeros((height, width), dtype=np.uint8)
    
    for rle in img_rle_seqs:
        img_mask += rle_decode(rle)
    
    obb_boxes = get_obb_coord(img_mask)
    
    if not os.path.exists(os.path.join(save_dir, 'images', split)):
        os.makedirs(os.path.join(save_dir, 'images', split))
    
    if not os.path.exists(os.path.join(save_dir, 'labels', split)):
        os.makedirs(os.path.join(save_dir, 'labels', split))

    cv2.imwrite(os.path.join(save_dir, 'images', split, image_id), img)

    with open(os.path.join(save_dir, 'labels', split, image_id.split(".")[0] + ".txt"), 'a') as f:
        for coord in obb_boxes:
            normalized_coords = [coord[i] / width if i % 2 == 0 else coord[i] / height for i in range(len(coord))]
            yolo_format = f"0 {normalized_coords[0]} {normalized_coords[1]} {normalized_coords[2]} {normalized_coords[3]} {normalized_coords[4]} {normalized_coords[5]} {normalized_coords[6]} {normalized_coords[7]}"
            f.write(yolo_format + "\n")
    

if __name__ == '__main__':
    train_df = pd.read_csv("/workspace/oriented_dota/oriented_object_detection/airbus-ship_detection/train_ship_segmentations_v2.csv")
    train_df.head()

    train_object = train_df[train_df['EncodedPixels'].isna() == False].reset_index(drop=True)
    train_background = train_df[train_df['EncodedPixels'].isna() == True].reset_index(drop=True)
    
    data_dir = '/workspace/oriented_dota/oriented_object_detection/airbus-ship_detection/train_v2/'
    split = 'train'
    image_ids = list(set((train_object['ImageId'].to_list())))
    save_dir = '/workspace/oriented_dota/oriented_object_detection/dataset3'
    
    # train
    train_ids = image_ids[:int(len(image_ids) * 0.8)]
    val_ids = image_ids[int(len(image_ids) * 0.8):]
    print("Train")
    train_ids = train_ids[1733:][1307:]
    
    for i in tqdm(range(len(train_ids))):
        image_id = train_ids[i]
        img = cv2.imread(os.path.join(data_dir, image_id))
        if type(img) == type(np.array([12])):
            convert_yolo_format(train_object, data_dir, image_id, split, save_dir)
    
    save_dir = '/workspace/oriented_dota/oriented_object_detection/dataset3'
    split = 'val'
    print("val")
    
    for i in tqdm(range(len(val_ids))):
        image_id = val_ids[i]
        img = cv2.imread(os.path.join(data_dir, image_id))
        if type(img) == type(np.array([12])):
            convert_yolo_format(train_object, data_dir, image_id, split, save_dir)



    