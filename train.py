import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='yolo_obb_Training')
parser.add_argument('--model_name', type = str, default="yolo11l-obb")
parser.add_argument('--dataset_yaml', type = str, default="/workspace/oriented_dota/DOTA.yaml")
parser.add_argument('--epochs', type = int, default=100)
parser.add_argument('--batch', type = int, default=8)
parser.add_argument('--imgsz', type = int, default=640)
parser.add_argument('--workers', type = int, default=4)
parser.add_argument('--close_mosaic', type=int, default=10) #
parser.add_argument('--device', type = str, default='0,1')
parser.add_argument('--seed', type = int, default=0)
parser.add_argument('--scale', type = float, default=0.3)
parser.add_argument('--single_cls', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    model_name = f"{args.model_name}.yaml"
    model = YOLO(model_name)
    device = args.device.split(",")
    
    if type(device) == type([0,1]):
        device = [int(d) for d in device]
    else:
        device = int(device)

    results = model.train(data=args.dataset_yaml,
                          epochs=args.epochs,
                          batch = args.batch,
                          imgsz=args.imgsz,
                          workers = args.workers,
                          close_mosaic = args.close_mosaic,
                          device=device,
                          seed = args.seed,
                          scale = args.scale,
                          single_cls = args.single_cls,)