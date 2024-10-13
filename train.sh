#best
#python train.py --model_name yolo11l-obb \
#    --dataset_yaml /workspace/oriented_dota/DOTA2.yaml \
#    --epochs 100 \
#    --batch 8 \
#    --imgsz 640 \
#    --workers 4 \
#    --close_mosaic 10 \
#    --device 0,1 \
#    --seed 0 \
#    --single_cls

python train.py --model_name yolo11l-obb \
    --dataset_yaml /workspace/oriented_dota/DOTA3.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --workers 4 \
    --close_mosaic 10 \
    --device 0,1 \
    --seed 0 \
    --scale 0.3 \
    --single_cls