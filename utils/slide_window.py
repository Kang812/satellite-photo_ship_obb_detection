import glob
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
import cv2
import time
import torch
from torch.utils.data import DataLoader, Dataset
from shapely.geometry import Polygon
from ultralytics.utils.ops import nms_rotated

def clear_gpu_memory():
    torch.cuda.empty_cache()

class SlidingWindowDataset(Dataset):
    def __init__(self, img, stride, overlap):
        self.img = img
        self.stride = stride
        self.overlap = overlap
        self.height, self.width = img.shape[:2]
        self.windows = []
        
        # 슬라이딩 윈도우의 좌표를 미리 계산합니다.
        for w in range(0, self.width + 1, stride - overlap):
            for h in range(0, self.height + 1, stride - overlap):
                start_w = w
                end_w = w + stride
                
                if end_w > self.width:
                    end_w = self.width
                    start_w = end_w - stride

                start_h = h
                end_h = h + stride
                
                if end_h > self.height:
                    end_h = self.height
                    start_h = end_h - stride

                self.windows.append((start_w, end_w, start_h, end_h))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start_w, end_w, start_h, end_h = self.windows[idx]
        crop = self.img[start_h:end_h, start_w:end_w, :]
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        crop_tensor = torch.from_numpy(crop_bgr).permute(2, 0, 1).float() / 255.0  # (H, W, C) -> (C, H, W)
        return crop_tensor, torch.tensor(self.windows[idx], dtype=torch.float32)

def slide_window_prediction(model, img, stride, overlap, thres=0.4, batch_size=16):
    """
    수정된 슬라이딩 윈도우 예측 함수로, PyTorch DataLoader를 사용하여 배치로 예측합니다.
    매개변수:
        model: 예측에 사용되는 YOLO 모델.
        img: 입력 이미지 (numpy 배열).
        stride: 슬라이딩 윈도우의 크기.
        overlap: 각 윈도우의 겹치는 픽셀 수.
        thres: 예측에 대한 신뢰도 임계값.
        batch_size: DataLoader의 배치 크기.
    반환값:
        예측 및 점수 리스트.
    """
    dataset = SlidingWindowDataset(img, stride, overlap)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # num_workers=0으로 설정하여 병렬 처리 오류 방지

    predictions = []
    scores = []

    model = model.to('cuda')  # 모델을 GPU로 이동
    start_time = time.time()
    for batch in dataloader:
        batch_images, batch_windows = batch
        batch_images = batch_images.to('cuda')  # GPU로 이동
        batch_results = model(batch_images)  # 모델에 배치 입력

        # 결과 처리 (예시 - 모델 출력 형태에 따라 수정 필요)
        for window, results in zip(batch_windows, batch_results):
            start_w, end_w, start_h, end_h = window.tolist()
            window_predictions = []
            window_scores = []

            for result in results:
                conf = result.obb.conf.detach().cpu().numpy()
                prediction = result.obb.xywhr.detach().cpu().numpy()
                for p, score in zip(prediction, conf):
                    if score > thres:
                        center_x = p[0] + start_w
                        center_y = p[1] + start_h
                        w_p = p[2]
                        h_p = p[3]
                        rotation = np.rad2deg(p[4])
                        window_predictions.append([center_x, center_y, w_p, h_p, rotation])
                        window_scores.append(float(score))

            predictions.extend(window_predictions)
            scores.extend(window_scores)
        # GPU 메모리 캐시 정리
        clear_gpu_memory()

    end_time = time.time()

    print(f"Total processing time: {end_time - start_time:.2f} seconds")

    return predictions, scores

def merge_results(model, img, file_name, stride, overlap, thres=0.4, iou_threshold=0.5, batch_size = 8):
    predictions, scores = slide_window_prediction(model, img, stride, overlap, thres=thres, batch_size= batch_size)
    keep_boxes_index = nms_rotated(torch.tensor(predictions), torch.tensor(scores), threshold=iou_threshold)

    if isinstance(keep_boxes_index, np.ndarray):
        keep_boxes = [predictions[i] for i in list(keep_boxes_index)]
    else:
        keep_boxes = [predictions[i] for i in list(keep_boxes_index.numpy())]

    image_names =[]
    cxs = []
    cys = []
    widths = []
    heights = []
    angles = []
    
    for bbox in keep_boxes:
        image_names.append(file_name)
        cxs.append(bbox[0])
        cys.append(bbox[1])
        widths.append(bbox[2])
        heights.append(bbox[3])
        angles.append(bbox[4])

    df = pd.DataFrame({
        "image_name":image_names,
        "cx":cxs,
        "cy":cys,
        "width":widths,
        "height":heights,
        "angle":angles})
    
    return df


if __name__ == '__main__':
    test_img_path = "/workspace/oriented_dota/oriented_object_detection/sample_image/"

    df = pd.DataFrame()
    stride = 256
    overlap = 128
    model = YOLO("/workspace/oriented_dota/runs/obb/train3/weights/best.pt")  # load a custom mode
    
    path = "/workspace/oriented_dota/oriented_object_detection/sample_image/competition_sample.png"
    with Image.open(path) as image:
        img = np.array(image, dtype=np.uint8)

    file_name = path.split("/")[-1]
    tmp_df = merge_results(model, img, file_name, stride, overlap, thres=0.6, iou_threshold=0.3, batch_size = 32)
    df = pd.concat([tmp_df, df])
    df.to_csv("/workspace/oriented_dota/utils/submission3.csv", index=False)  # CSV로 저장