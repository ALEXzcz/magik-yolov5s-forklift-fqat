set -e

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info

#float32:
python3 train.py \
    --data data/anytrek_forklift.yaml \
    --cfg models/yolov5s.yaml \
    --weights '' \
    --batch-size 16 \
    --hyp data/hyp.scratch-float.yaml \
    --project ./runs/train/yolov5s-forklift-float32 \
    --epochs 300 \
    --device 0 \
    --bit 32

LATEST_BEST="$(find ./runs/train -type f -path '*/weights/best.pt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
cp "$LATEST_BEST" ./checkpoint/32W32F/weights/best-tmp.pt

###4bit coco-person:
# python3 train.py \
#     --data data/coco-person.yaml \
#     --cfg models/yolov5s.yaml \
#     --weights 'checkpoint/32W32F/weights/best.pt' \
#     --batch-size 16 \
#     --hyp data/hyp.scratch.yaml \
#     --project ./runs/train/yolov5s-person-4bit \
#     --epochs 300 \
#     --device 0 \
#     --optimizer Adam \
#     --bit 4

# 4bit anytrek_2310:
# python3 train.py \
#     --data data/anytrek_2310.yaml \
#     --cfg models/yolov5s.yaml \
#     --weights checkpoint/32W32F/weights/best.pt \
#     --batch-size 16 \
#     --hyp data/hyp.scratch.yaml \
#     --project ./runs/train/yolov5s-anytrek-4bit \
#     --epochs 5 \
#     --device 0 \
#     --optimizer Adam \
#     --bit 4

# LATEST_BEST="$(find ./runs/train -type f -path '*/weights/best.pt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
# cp "$LATEST_BEST" ./checkpoint/4W4F/weights/best-tmp.pt
