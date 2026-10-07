# -*- coding: utf-8 -*-
"""逐类指标(ultralytics 口径)+ AP_s/m/l 分档(COCOeval 口径)+ 与论文三 FP32 对表。

口径纪律(来自 skill yolo-coco-eval / coco-detection-evaluation):
- predictions.json 的 category_id 基数不确定 -> 先从预测里读 id 集合,再决定 GT 的 id_base;
- GT 不做类别重映射、bbox 不做 letterbox 反变换(ultralytics 已 rescale);
- COCOeval.stats 必须 evaluate+accumulate+summarize 之后才非空;
- 单 IoU 阈值时 summarize() 会崩,AP50 分档手动从 eval['precision'] 取(maxDets 索引 2 = 100)。

用法:
  python ptq_eval.py --all --imgsz 640 --split val
  python ptq_eval.py --dataset gc10 --scale n --seed 42 --precision fp32
"""
from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path

import numpy as np

import ptq_common as C


# ------------------------------------------------------------ 工具
def append_row(path: Path, row: dict) -> None:
    """同键(数据集/尺度/种子/精度)覆盖写,避免 --force 重跑产生重复行。"""
    key = ("dataset", "scale", "seed", "precision")
    rows, fields = [], list(row)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows = [r for r in rows if tuple(r.get(k) for k in key) != tuple(str(row[k]) for k in key)]
        if rows:
            fields = list(dict.fromkeys(list(rows[0]) + fields))
    rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def already_done(dataset: str, scale: str, seed: int, precision: str,
                 path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 4 and parts[0] == dataset and parts[1] == scale \
                and parts[2] == str(seed) and parts[3] == precision:
            return True
    return False


def detect_id_base(pred_json: Path) -> int:
    """从 predictions.json 读 category_id 集合,判断 GT 的类别 id_base。"""
    d = json.loads(pred_json.read_text(encoding="utf-8"))
    ids = {int(p["category_id"]) for p in d}
    return 0 if 0 in ids else 1


def detect_image_id_mode(pred_json: Path) -> str:
    """pred 的 image_id 是文件名(stem)还是整数 -> 决定 GT 的 image_id 约定。"""
    d = json.loads(pred_json.read_text(encoding="utf-8"))
    if not d:
        return "stem"
    v = d[0]["image_id"]
    return "stem" if isinstance(v, str) else "index"


def ensure_gt(dataset: str, split: str, id_base: int, mode: str = "stem") -> Path:
    from make_coco_gt import build
    p = C.OUT / f"gt_{dataset}_{split}_base{id_base}_{mode}_{C.AREA_SPACE}.json"
    if p.exists():
        return p
    return build(dataset, split, id_base, p, mode, C.AREA_SPACE)


def read_row(path: Path, dataset: str, scale: str, seed: int, precision: str):
    """读 results.csv 里已有的那一行(供 --coco-only 复用精度数字)。"""
    if not path.exists():
        return None
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if (r.get("dataset") == dataset and r.get("scale") == scale
                and str(r.get("seed")) == str(seed) and r.get("precision") == precision):
            return r
    return None


def coco_by_size(gt_json: Path, pred_json: Path) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(gt_json))
    pred = gt.loadRes(str(pred_json))

    ev = COCOeval(gt, pred, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats  # AP, AP50, AP75, APs, APm, APl, AR..., 需 summarize 后才有值
    out = {"AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
           "APs": float(s[3]), "APm": float(s[4]), "APl": float(s[5])}

    # 单 IoU=0.5 的分档 AP 必须手动取(单阈值下 summarize 崩溃)
    ev2 = COCOeval(gt, pred, "bbox")
    ev2.params.iouThrs = [0.5]
    ev2.evaluate(); ev2.accumulate()
    prec = ev2.eval["precision"]  # (1, 101, K, A=4, M=3)

    def ap_area(aidx: int) -> float:
        p = prec[0, :, :, aidx, 2]      # maxDets 索引 2 == 100
        p = p[p != -1]
        return float(p.mean()) if p.size else 0.0

    out["AP50_s"] = ap_area(1)
    out["AP50_m"] = ap_area(2)
    out["AP50_l"] = ap_area(3)
    return out


def per_class_from_result(r, names: list) -> dict:
    """从 ultralytics 的 val 结果里取逐类 AP(防御式:不同版本属性名不同)。"""
    box = r.box
    out = {"names": names}
    maps = getattr(box, "maps", None)              # per-class mAP50-95
    ap50 = getattr(box, "ap50", None)              # per-class AP50(部分版本有)
    idx = getattr(box, "ap_class_index", None)
    if maps is not None:
        out["map50_95_per_class"] = [None if float(v) != float(v) else round(float(v), 5)
                                     for v in np.asarray(maps)]
    if ap50 is not None:
        out["map50_per_class"] = [round(float(v), 5) for v in np.asarray(ap50)]
    if idx is not None:
        out["class_index"] = [int(v) for v in np.asarray(idx)]
    return out


# ------------------------------------------------------------ 单个评测
def one(dataset: str, scale: str, seed: int, precision: str, imgsz: int,
        split: str, force: bool, coco_only: bool = False) -> None:
    onnx = C.onnx_path(dataset, scale, seed, precision)
    # 分区隔离:test 评测必须写自己的 CSV 与自己的 run 目录,
    # 否则 test 行会混进 val 聚合(results.csv 的跳过判断也不含 split),
    # 并且会覆盖 val 的 predictions.json / 逐类转储。
    suffix = "" if split == "val" else f"_{split}"
    results_csv = C.OUT / (f"results{suffix}.csv")
    tag = f"{dataset}_{scale}_{precision}_seed{seed}{suffix}"
    pred_json = C.RUNS / tag / "predictions.json"

    if coco_only:
        # 复用已存的 predictions.json,只重算 COCOeval 分档(改分档口径后不必重跑 val;
        # int8-dynamic 的 val 一次要 ~18 min)
        if not pred_json.exists():
            print(f"  [skip] 没有历史 predictions.json:{pred_json}")
            return
        row = read_row(results_csv, dataset, scale, seed, precision)
        if row is None:
            print(f"  [skip] results.csv 里没有 {tag} 的历史行")
            return
        try:
            id_base = detect_id_base(pred_json)
            mode = detect_image_id_mode(pred_json)
            gt_json = ensure_gt(dataset, split, id_base, mode)
            d = coco_by_size(gt_json, pred_json)
            row = dict(row)
            row.update({k: round(v, 5) for k, v in d.items()})
            row["gt_id_base"] = id_base
            append_row(results_csv, row)
            print(f"  [coco-only] {tag}: AP50={d['AP50']:.4f} APs={d['APs']:.4f} "
                  f"APm={d['APm']:.4f} APl={d['APl']:.4f} AP50_s={d['AP50_s']:.4f}")
        except Exception as e:
            print(f"  [ERR] coco-only {tag}: {type(e).__name__}: {e}")
            with open(C.OUT / "errors.log", "a", encoding="utf-8") as f:
                f.write(f"coco-only {tag}\n{traceback.format_exc()}\n")
        return

    if not onnx.exists():
        print(f"  [skip] 没有 {onnx.name}")
        return
    if not force and already_done(dataset, scale, seed, precision, results_csv):
        print(f"  [skip] 已完成 {dataset} {scale} seed{seed} {precision}")
        return

    from ultralytics import YOLO
    print(f"\n=== eval {tag} ===")
    m = YOLO(str(onnx))
    r = m.val(data=str(C.DATASETS[dataset]["yaml"]), split=split, imgsz=imgsz,
              save_json=True, conf=0.001, iou=0.7, project=str(C.RUNS),
              name=tag, exist_ok=True, verbose=False, plots=False)

    mAP50 = float(r.box.map50)
    mAP5095 = float(r.box.map)
    row = {"dataset": dataset, "scale": scale, "seed": seed,
           "precision": precision, "split": split, "imgsz": imgsz,
           "mAP50": round(mAP50, 5), "mAP50_95": round(mAP5095, 5),
           "save_dir": str(r.save_dir)}

    # 逐类
    pc = per_class_from_result(r, C.DATASETS[dataset]["names"])
    pcdir = C.OUT / "perclass"
    pcdir.mkdir(parents=True, exist_ok=True)
    (pcdir / f"{tag}.json").write_text(json.dumps(pc, ensure_ascii=False, indent=1),
                                       encoding="utf-8")

    # 分档(COCOeval)
    pred_json = Path(r.save_dir) / "predictions.json"
    if pred_json.exists():
        try:
            id_base = detect_id_base(pred_json)
            mode = detect_image_id_mode(pred_json)
            gt_json = ensure_gt(dataset, split, id_base, mode)
            d = coco_by_size(gt_json, pred_json)
            row.update({k: round(v, 5) for k, v in d.items()})
            row["gt_id_base"] = id_base
            print(f"  ultralytics mAP50={mAP50:.4f} mAP50-95={mAP5095:.4f} | "
                  f"COCOeval AP50={d['AP50']:.4f} APs={d['APs']:.4f} "
                  f"AP50_s={d['AP50_s']:.4f}")
        except Exception as e:
            print(f"  [WARN] COCOeval 失败:{type(e).__name__}: {e}")
            with open(C.OUT / "errors.log", "a", encoding="utf-8") as f:
                f.write(f"cocoeval {tag}\n{traceback.format_exc()}\n")
    else:
        print("  [WARN] 没有 predictions.json")

    # 与论文三 FP32 对表(只对 fp32 做,且只在能找到参考时)
    if precision == "fp32":
        refs = C.find_paper3_reference(dataset, scale)
        if refs:
            print(f"  论文三参考文件 {len(refs)} 个,例:{refs[0].name}")
            row["ref_files"] = len(refs)
        else:
            print("  [note] 未自动定位论文三参考(results.csv);FP32 对表请手动核对")

    append_row(results_csv, row)
    print(f"  -> 追加 {results_csv.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", choices=list(C.DATASETS))
    ap.add_argument("--scale", choices=C.SCALES)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--precision", choices=C.PRECISIONS)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="val")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--coco-only", action="store_true",
                    help="只重算 COCOeval 分档(复用 runs/<tag>/predictions.json,不重跑 val)")
    a = ap.parse_args()

    if a.all:
        combos = [(d, s, sd, p) for (d, s, sd) in C.cells() for p in C.PRECISIONS]
    else:
        combos = [(a.dataset, a.scale, a.seed, a.precision)]
        if any(x is None for x in combos[0]):
            raise SystemExit("需要 --all 或同时给出 --dataset/--scale/--seed/--precision")
    for d, s, sd, p in combos:
        try:
            one(d, s, sd, p, a.imgsz, a.split, a.force, a.coco_only)
        except Exception as e:
            print(f"  [ERR] {d} {s} seed{sd} {p}: {type(e).__name__}: {e}")
            with open(C.OUT / "errors.log", "a", encoding="utf-8") as f:
                f.write(f"eval {d} {s} {sd} {p}\n{traceback.format_exc()}\n")
    print(f"\n评测阶段结束,结果:{C.OUT / 'results.csv'}(test 分区写 results_test.csv)")


if __name__ == "__main__":
    main()
