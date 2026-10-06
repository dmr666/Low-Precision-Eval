# -*- coding: utf-8 -*-
"""由 YOLO 格式数据生成 COCO GT json(AP_s/m/l 分档指标需要 COCOeval)。

为什么需要:ultralytics 的 val 只给逐类 AP,不给按目标尺寸分档;分档要 pycocotools。
GT 的 category_id 基数是关键坑(见 skill yolo-coco-eval 坑 1):
  ultralytics save_json 写出的 predictions.json 里的 category_id **未必**等于类索引,
  所以本脚本支持 --id_base 0|1,eval 脚本会自动读 predictions 的 id 集合来选。
用法:
  python make_coco_gt.py --dataset gc10 --split val --id_base 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

import ptq_common as C


def load_data_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split(data: dict, yaml_path: Path, split: str):
    """返回 (images_dir, labels_dir)。支持 train/val 为目录名或 .txt 列表。

    labels_dir 只作为兜底;每张图的标签路径由 label_for() 逐图推导
    (数据集的 images/labels 是同级的兄弟目录,不能用 img_dir.parent 推)。
    """
    root = Path(data.get("path") or yaml_path.parent)
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    entry = data.get(split)
    if entry is None:
        raise SystemExit(f"{yaml_path} 里没有 {split} 字段")
    if isinstance(entry, str) and entry.endswith(".txt"):
        lst = (root / entry) if not Path(entry).is_absolute() else Path(entry)
        lines = [ln.strip() for ln in lst.read_text(encoding="utf-8").splitlines() if ln.strip()]
        img_dir = Path(lines[0]).parent
        return img_dir, _labels_sibling(img_dir)
    img_dir = (root / entry).resolve()
    return img_dir, _labels_sibling(img_dir)


def _labels_sibling(img_dir: Path) -> Path:
    """若图片目录下还有一层 images/,标签在同级 labels/;否则退回 img_dir 内。"""
    if img_dir.name == "images":
        return img_dir.parent / "labels"
    cand = img_dir / "labels"
    return cand if cand.exists() else img_dir


def label_for(img: Path, lbl_dir: Path) -> Path:
    """逐图推导标签路径,按优先级尝试三种常见布局。"""
    same = img.with_suffix(".txt")                        # 图与标签同目录
    if same.exists():
        return same
    parts = list(img.parts)
    if "images" in parts:                                 # .../images/x.jpg -> .../labels/x.txt
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        swapped = Path(*parts).with_suffix(".txt")
        if swapped.exists():
            return swapped
        return swapped
    return lbl_dir / f"{img.stem}.txt"


def build(dataset: str, split: str, id_base: int, out_json: Path,
          image_id_mode: str = "stem", area_space: str = None) -> Path:
    """image_id_mode: 'stem' = 用图片文件名(不含扩展名)当 image_id —— ultralytics 对
    YOLO 格式数据集 的 save_json 就是这么写的,必须与之一致,否则 COCOeval 报
    'Results do not correspond to current coco set';'index' = 顺序整数(仅备用)。

    area_space: 'eq640' = 面积折算到 640 输入空间(与论文三口径一致,'raw' = 原始像素面积),
    决定 COCOeval 的 APs/APm/APl 分档含义。"""
    meta = C.DATASETS[dataset]
    area_space = area_space or C.AREA_SPACE
    data = load_data_yaml(meta["yaml"])
    names = data.get("names") or meta["names"]
    if isinstance(names, dict):                      # yaml 里可能是 {0: 'a', ...}
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    img_dir, lbl_dir = resolve_split(data, meta["yaml"], split)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts)
    if not images:
        raise SystemExit(f"没找到图片:{img_dir}")

    coco = {
        "info": {"description": f"{dataset} {split} (converted from YOLO format)"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + id_base, "name": n, "supercategory": "defect"}
            for i, n in enumerate(names)
        ],
    }
    aid = 1
    n_boxes = 0
    for i, img in enumerate(images, start=1):
        img_id = img.stem if image_id_mode == "stem" else i
        with Image.open(img) as im:
            w, h = im.size
        sc = C.EQ_IMGSZ / max(w, h)          # 折算到 640 输入的缩放因子
        coco["images"].append({"id": img_id, "file_name": img.name,
                               "width": w, "height": h})
        lbl = label_for(img, lbl_dir)
        if not lbl.exists():
            continue
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            ww, hh = bw * w, bh * h
            if ww <= 0 or hh <= 0:
                continue
            area_raw = ww * hh
            area_val = round(area_raw * sc * sc, 2) if area_space == "eq640" else round(area_raw, 2)
            coco["annotations"].append({
                "id": aid, "image_id": img_id, "category_id": cls + id_base,
                "bbox": [round(x, 2), round(y, 2), round(ww, 2), round(hh, 2)],
                "area": area_val, "iscrowd": 0,
                "area_raw": round(area_raw, 2),
            })
            aid += 1
            n_boxes += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco), encoding="utf-8")
    print(f"[gt] {dataset}/{split}: {len(images)} 图 / {n_boxes} 框 / "
          f"{len(names)} 类 / id_base={id_base} -> {out_json}")
    return out_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--id_base", type=int, default=1, choices=[0, 1])
    ap.add_argument("--image_id", default="stem", choices=["stem", "index"],
                    help="image_id 约定:stem = 文件名(与 ultralytics save_json 一致)")
    ap.add_argument("--area_space", default=None, choices=["eq640", "raw"],
                    help="面积分档口径:eq640(默认,与论文三一致)/ raw(原始像素)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    space = a.area_space or C.AREA_SPACE
    out = Path(a.out) if a.out else (
        C.OUT / f"gt_{a.dataset}_{a.split}_base{a.id_base}_{a.image_id}_{space}.json")
    build(a.dataset, a.split, a.id_base, out, a.image_id, space)


if __name__ == "__main__":
    main()
