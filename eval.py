import argparse
import yaml
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='yolo_obb_eval')
parser.add_argument('--model_path', type = str, default="/workspace/oriented_dota/runs/obb/train/weights/best.pt")
parser.add_argument('--data_yaml_path', type = str, default="/workspace/oriented_dota/DOTA.yaml")
parser.add_argument('--split', type = str, default="test")

if __name__ == '__main__':
    args = parser.parse_args()
    model = YOLO(args.model_path)
    metrics = model.val(data=args.data_yaml_path, split = args.split)  # no arguments needed, dataset and settings remembered

    print()
    print("eval results:")
    print("map50-95(B):", np.round(metrics.box.map, 4))  # map50-95(B)
    print("map50(B):",np.round(metrics.box.map50, 4))  # map50(B)
    print("map75(B):", np.round(metrics.box.map75, 4))  # map75(B)

