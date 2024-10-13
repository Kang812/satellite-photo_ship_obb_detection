import random
from tqdm import tqdm
from shutil import copyfile
import pandas as pd
import cv2
import numpy as np
import os

train_df = pd.read_csv("/workspace/oriented_dota/oriented_object_detection/airbus-ship_detection/train_ship_segmentations_v2.csv")
train_background = train_df[train_df['EncodedPixels'].isna() == True].reset_index(drop=True)
imageIds = random.choices(train_background['ImageId'].to_list(), k = 2000)
data_dir = "/workspace/oriented_dota/oriented_object_detection/airbus-ship_detection/train_v2"
save_dir = '/workspace/oriented_dota/oriented_object_detection/dataset3/'

for i in tqdm(range(len(imageIds))):
    image_id = imageIds[i]
    image_path = os.path.join("/workspace/oriented_dota/oriented_object_detection/airbus-ship_detection/train_v2" , image_id)
    img = cv2.imread(image_path)
    if type(img) == type(np.array([12])):
        copyfile(image_path, os.path.join(save_dir, 'images', 'train', image_id))
        with open(os.path.join(save_dir, 'labels', 'train', image_id.replace("jpg", 'txt')), 'w') as f:
            f.write("")