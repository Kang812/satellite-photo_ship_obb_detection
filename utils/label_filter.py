# object가 없는 데이터를 걸러내기 위한 코드
import os
from tqdm import tqdm
from shutil import copyfile
from glob import glob

def label_filter(txt_path, move_path, dataset):

    if not os.path.exists(os.path.join(move_path, 'images', dataset)):
        os.makedirs(os.path.join(move_path, 'images', dataset))
    
    if not os.path.exists(os.path.join(move_path, 'labels', dataset)):
        os.makedirs(os.path.join(move_path, 'labels', dataset))
    
    with open(txt_path, 'r') as f:
        data = f.readlines()
    
    if data != []:
        image_path = txt_path.replace("labels", 'images')[:txt_path.rfind('txt')] + 'jpg'
        copyfile(txt_path, os.path.join(move_path, 'labels', dataset, txt_path.split("/")[-1]))
        copyfile(image_path, os.path.join(move_path, 'images', dataset, image_path.split("/")[-1]))

if __name__=='__main__':
    move_path = '/workspace/oriented_dota/oriented_object_detection/dataset2/'

    print("Train")
    label_txt_paths = glob("/workspace/oriented_dota/oriented_object_detection/dataset/labels/train/*.txt")
    dataset = 'train'
    for i in tqdm(range(len(label_txt_paths))):
        label_filter(label_txt_paths[i], move_path, dataset)
    
    print("Val")
    label_txt_paths = glob("/workspace/oriented_dota/oriented_object_detection/dataset/labels/val/*.txt")
    dataset = 'val'
    for i in tqdm(range(len(label_txt_paths))):
        label_filter(label_txt_paths[i], move_path, dataset)
    

    print("Test")
    label_txt_paths = glob("/workspace/oriented_dota/oriented_object_detection/dataset/labels/test/*.txt")
    dataset = 'test'
    for i in tqdm(range(len(label_txt_paths))):
        label_filter(label_txt_paths[i], move_path, dataset)