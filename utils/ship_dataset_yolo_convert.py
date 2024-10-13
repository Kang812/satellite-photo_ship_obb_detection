import os
import cv2
from pycocotools.coco import COCO

def ship_convert_yolo_format(coco, data_dir, split, save_dir):
    ImgIds = coco.getImgIds()
    ShipIds = coco.getCatIds(catNms=['Ship'])

    for i in range(len(ImgIds)):
        annIds = coco.getAnnIds(imgIds = ImgIds[i], catIds = ShipIds)
        anns = coco.loadAnns(annIds)
        imgInfo = coco.loadImgs(ImgIds[i])[0]
        file_name = imgInfo['file_name']
        
        image_path = os.path.join(data_dir, file_name)
        img = cv2.imread(image_path)
        cv2.imwrite(os.path.join(save_dir, 'images', split, file_name.split(".")[0] + ".jpg"), img)

        width = imgInfo['width']
        height = imgInfo['height']
        with open(os.path.join(save_dir, 'labels', split, file_name.split(".")[0] + ".txt"), 'a') as f:
            for ann in anns:
                coord = ann['segmentation'][0]
                normalized_coords = [coord[i] / width if i % 2 == 0 else coord[i] / height for i in range(len(coord))]
                yolo_format = f"0 {normalized_coords[0]} {normalized_coords[1]} {normalized_coords[2]} {normalized_coords[3]} {normalized_coords[4]} {normalized_coords[5]} {normalized_coords[6]} {normalized_coords[7]}"
                f.write(yolo_format + "\n")


if __name__ == '__main__':
    # train
    coco = COCO("/workspace/oriented_dota/oriented_object_detection/original_data/ShipRSImageNet_V1/COCO_Format/ShipRSImageNet_bbox_train_level_0.json")
    data_dir = '/workspace/oriented_dota/oriented_object_detection/original_data/ShipRSImageNet_V1/VOC_Format/JPEGImages/'
    save_dir = '/workspace/oriented_dota/oriented_object_detection/dataset/'
    split = 'train'

    ship_convert_yolo_format(coco, data_dir, split, save_dir)

    # val
    coco = COCO("/workspace/oriented_dota/oriented_object_detection/original_data/ShipRSImageNet_V1/COCO_Format/ShipRSImageNet_bbox_val_level_0.json")
    data_dir = '/workspace/oriented_dota/oriented_object_detection/original_data/ShipRSImageNet_V1/VOC_Format/JPEGImages/'
    save_dir = '/workspace/oriented_dota/oriented_object_detection/dataset/'
    split = 'val'
    ship_convert_yolo_format(coco, data_dir, split, save_dir)

    # test
    coco = COCO("/workspace/oriented_dota/oriented_object_detection/original_data/ShipRSImageNet_V1/COCO_Format/ShipRSImageNet_bbox_test_level_0.json")
    data_dir = '/workspace/oriented_dota/oriented_object_detection/original_data/ShipRSImageNet_V1/VOC_Format/JPEGImages/'
    save_dir = '/workspace/oriented_dota/oriented_object_detection/dataset/'
    split = 'test'

    ship_convert_yolo_format(coco, data_dir, split, save_dir)