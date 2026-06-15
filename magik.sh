# magik_quantize.py bit=8

python magik_quantize.py \
  --cfg models/yolov5s.yaml \
  --weights ./checkpoint/32W32F/weights/best-tmp.pt \
  --bit 8
