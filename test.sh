#for float:
python val.py --data data/anytrek_forklift.yaml --float_model runs/train/yolov5s-forklift-float32/weights/best.pt --weights runs/train/yolov5s-forklift-float32/weights/best.pt --imgs 640 --device 0 --batch-size 100 --model_only

#for 8bit:
# python val.py --data data/anytrek_forklift.yaml --float_model runs/train/yolov5s-forklift-float32/weights/best.pt --weights ./checkpoint/8W8F/weights/best.pt --imgs 640 --device 0 --batch-size 100 --bit 8

#for 4bit:
# python val.py --data data/anytrek_forklift.yaml --float_model runs/train/yolov5s-forklift-float32/weights/best.pt --weights ./checkpoint/4W4F/weights/best.pt --imgs 640 --device 0 --batch-size 100 --bit 4
