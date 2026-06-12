#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import yaml

from utils.metrics import box_iou


@dataclass
class LabelBox:
    cls: int
    xywh: List[float]
    conf: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GT labels with detect.py predicted labels.")
    parser.add_argument(
        "--gt-dir",
        default="data/anytrek_detect_labels",
        help="Ground-truth YOLO label directory.",
    )
    parser.add_argument(
        "--pred-dir",
        default=None,
        help="Predicted YOLO label directory. Default: latest runs/detect/exp*/labels.",
    )
    parser.add_argument(
        "--data",
        default="data/anytrek_forklift.yaml",
        help="Dataset yaml for class names.",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.5,
        help="IoU threshold used to match GT and prediction.",
    )
    parser.add_argument(
        "--show-matches",
        action="store_true",
        help="Print matched box pairs for each image.",
    )
    return parser.parse_args()


def find_latest_pred_dir() -> Path:
    detect_root = Path("runs/detect")
    label_dirs = [p for p in detect_root.glob("exp*/labels") if p.is_dir()]
    if not label_dirs:
        raise FileNotFoundError("No predicted label directory found under runs/detect/exp*/labels")
    return max(label_dirs, key=lambda p: p.stat().st_mtime)


def load_names(data_yaml: Path) -> Dict[int, str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        return {idx: name for idx, name in enumerate(names)}
    return {int(idx): name for idx, name in names.items()}


def load_label_file(path: Path) -> List[LabelBox]:
    if not path.exists():
        return []

    boxes: List[LabelBox] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) not in (5, 6):
                raise ValueError(f"{path}:{line_no} has {len(parts)} columns, expected 5 or 6")
            cls = int(float(parts[0]))
            xywh = [float(x) for x in parts[1:5]]
            conf = float(parts[5]) if len(parts) == 6 else None
            boxes.append(LabelBox(cls=cls, xywh=xywh, conf=conf))
    return boxes


def xywh_to_xyxy(boxes: Sequence[LabelBox]) -> torch.Tensor:
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    tensor = torch.tensor([box.xywh for box in boxes], dtype=torch.float32)
    x, y, w, h = tensor[:, 0], tensor[:, 1], tensor[:, 2], tensor[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=1)


def match_boxes(
    gt_boxes: Sequence[LabelBox], pred_boxes: Sequence[LabelBox], iou_thres: float
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    if not gt_boxes or not pred_boxes:
        return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

    gt_xyxy = xywh_to_xyxy(gt_boxes)
    pred_xyxy = xywh_to_xyxy(pred_boxes)
    ious = box_iou(gt_xyxy, pred_xyxy)

    candidates: List[Tuple[float, int, int]] = []
    for gt_idx, gt in enumerate(gt_boxes):
        for pred_idx, pred in enumerate(pred_boxes):
            if gt.cls != pred.cls:
                continue
            iou = float(ious[gt_idx, pred_idx])
            if iou >= iou_thres:
                candidates.append((iou, gt_idx, pred_idx))

    candidates.sort(reverse=True)
    matches: List[Tuple[int, int, float]] = []
    used_gt = set()
    used_pred = set()
    for iou, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))

    missed_gt = [idx for idx in range(len(gt_boxes)) if idx not in used_gt]
    extra_pred = [idx for idx in range(len(pred_boxes)) if idx not in used_pred]
    return matches, missed_gt, extra_pred


def format_box(box: LabelBox, names: Dict[int, str]) -> str:
    cls_name = names.get(box.cls, str(box.cls))
    conf_text = f", conf={box.conf:.3f}" if box.conf is not None else ""
    x, y, w, h = box.xywh
    return f"{cls_name} [{x:.3f}, {y:.3f}, {w:.3f}, {h:.3f}{conf_text}]"


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir) if args.pred_dir else find_latest_pred_dir()
    data_yaml = Path(args.data)
    names = load_names(data_yaml)

    gt_files = sorted(gt_dir.glob("*.txt"))
    if not gt_files:
        raise FileNotFoundError(f"No GT labels found in {gt_dir}")

    print(f"GT labels:   {gt_dir}")
    print(f"Pred labels: {pred_dir}")
    print(f"IoU thres:   {args.iou_thres}")
    print()

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fn = 0
    total_fp = 0

    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        gt_boxes = load_label_file(gt_path)
        pred_boxes = load_label_file(pred_path)
        matches, missed_gt, extra_pred = match_boxes(gt_boxes, pred_boxes, args.iou_thres)

        tp = len(matches)
        fn = len(missed_gt)
        fp = len(extra_pred)

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        total_tp += tp
        total_fn += fn
        total_fp += fp

        status = "OK" if fn == 0 and fp == 0 else "CHECK"
        print(
            f"{gt_path.stem}: {status} | gt={len(gt_boxes)} pred={len(pred_boxes)} "
            f"tp={tp} fn={fn} fp={fp}"
        )

        if args.show_matches and matches:
            for gt_idx, pred_idx, iou in matches:
                print(
                    f"  match  iou={iou:.3f} | gt={format_box(gt_boxes[gt_idx], names)} "
                    f"| pred={format_box(pred_boxes[pred_idx], names)}"
                )

        for idx in missed_gt:
            print(f"  miss   gt={format_box(gt_boxes[idx], names)}")
        for idx in extra_pred:
            print(f"  extra  pred={format_box(pred_boxes[idx], names)}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print()
    print("Summary")
    print(f"  images={len(gt_files)}")
    print(f"  gt_boxes={total_gt}")
    print(f"  pred_boxes={total_pred}")
    print(f"  tp={total_tp}")
    print(f"  fn={total_fn}")
    print(f"  fp={total_fp}")
    print(f"  precision={precision:.4f}")
    print(f"  recall={recall:.4f}")
    print(f"  f1={f1:.4f}")


if __name__ == "__main__":
    main()
