# magik_quantize.py bit=8

python magik_quantize.py \
  --cfg models/yolov5s.yaml \
  --weights ./runs/train/yolov5s-forklift-float32/weights/best.pt \
  --bit 8
