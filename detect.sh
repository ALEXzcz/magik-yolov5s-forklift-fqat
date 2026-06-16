#for detect single img
# float
python detect.py \
  --source data/anytrek_detect/images \
  --data data/anytrek_forklift.yaml \
  --float_model checkpoint/32W32F/weights/best-tmp.pt \
  --weights checkpoint/32W32F/weights/best-tmp.pt \
  --imgs 640 \
  --device 0 \
  --conf-thres 0.6 \
  --iou-thres 0.4 \
  --view-img \
  --save-txt \
  --save-conf \
  --model_only

# 8bit
# python detect.py \
#   --source data/anytrek_detect/images \
#   --data data/anytrek_forklift.yaml \
#   --float_model checkpoint/32W32F/weights/best-tmp.pt \
#   --weights ./checkpoint/8W8F/weights/best-tmp.pt \
#   --imgs 640 \
#   --device 0 \
#   --view-img \
#   --bit 8

# 4bit
# python detect.py \
#   --source data/anytrek_detect/images \
#   --data data/anytrek_forklift.yaml \
#   --float_model checkpoint/32W32F/weights/best-tmp.pt \
#   --weights ./checkpoint/4W4F/weights/best-tmp.pt \
#   --imgs 640 \
#   --device 0 \
#   --view-img \
#   --bit 4

#for onnx:
# 8bit
# python detect.py \
#   --source data/anytrek_detect/images \
#   --data data/anytrek_forklift.yaml \
#   --float_model checkpoint/32W32F/weights/best-tmp.pt \
#   --weights ./checkpoint/8W8F/weights/best-tmp.pt \
#   --imgs 640 \
#   --device 0 \
#   --view-img \
#   --onnx \
#   --bit 8

# 4bit
# python detect.py \
#   --source data/anytrek_detect/images \
#   --data data/anytrek_forklift.yaml \
#   --float_model checkpoint/32W32F/weights/best-tmp.pt \
#   --weights ./checkpoint/4W4F/weights/best-tmp.pt \
#   --imgs 640 \
#   --device 0 \
#   --view-img \
#   --onnx \
#   --bit 4
