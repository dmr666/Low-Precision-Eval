# -*- coding: utf-8 -*-
"""ONNX Runtime CPU 延迟 / 内存 / 体积测量(单图、批 1,线边口径)。

纪律:
- 用**真实 val 图**(不是随机噪声):随机输入会低估延迟(激活稀疏性差异),审稿人也会问;
- 前 warmup 次不计;记录 median / mean / p95 / SD / min;
- 记录模型文件体积;峰值内存用 psutil(有则记,无则跳过并标注)。

用法:
  python ptq_latency.py --all --imgsz 640 --n 100 --threads 4
  python ptq_latency.py --dataset gc10 --scale n --seed 42 --precision int8dyn
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import numpy as np

import ptq_common as C


def imread_unicode(path: Path):
    """cv2.imread 在 Windows 的非 ASCII 路径上会失败 -> 用 fromfile + imdecode。"""
    import cv2
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        raise SystemExit(f"读不到图(空文件?):{path}")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def letterbox_bgr(path: Path, imgsz: int):
    """复刻 ultralytics 的 letterbox:等比缩放 + pad 到 32 的倍数(灰 114)+ **/255 归一化**。

    注意:必须除以 255。漏掉它会让输入是 0–255 而不是 0–1 —— 延迟虽仍可测,
    但用于**校准**时激活定标会错 255 倍,静态量化模型直接崩(mAP 归零,实测踩过)。
    返回 float32 HWC(RGB),取值范围 0–1。
    """
    import cv2
    im = imread_unicode(path)
    if im is None:
        raise SystemExit(f"读不到图:{path}")
    h0, w0 = im.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (imgsz - nw) // 2, (imgsz - nh) // 2
    top, bottom = dh, imgsz - nh - dh
    left, right = dw, imgsz - nw - dw
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=(114, 114, 114))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im.astype(np.float32) / 255.0


def first_val_image(dataset: str) -> Path:
    import yaml
    meta = C.DATASETS[dataset]
    d = yaml.safe_load(meta["yaml"].read_text(encoding="utf-8"))
    root = Path(d.get("path") or meta["yaml"].parent)
    if not root.is_absolute():
        root = (meta["yaml"].parent / root).resolve()
    entry = d.get("val")
    if isinstance(entry, str) and entry.endswith(".txt"):
        lst = root / entry
        first = lst.read_text(encoding="utf-8").splitlines()[0].strip()
        return Path(first) if Path(first).is_absolute() else (root / first).resolve()
    vdir = (root / entry).resolve()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    imgs = sorted(p for p in vdir.rglob("*") if p.suffix.lower() in exts)
    if not imgs:
        raise SystemExit(f"val 目录没图:{vdir}")
    return imgs[0]


def one(dataset: str, scale: str, seed: int, precision: str, imgsz: int,
        n: int, threads: int, warmup: int) -> None:
    onnx = C.onnx_path(dataset, scale, seed, precision)
    if not onnx.exists():
        print(f"  [skip] 没有 {onnx.name}")
        return
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    is_half = "float16" in inp.type
    x = letterbox_bgr(first_val_image(dataset), imgsz)
    x = x.transpose(2, 0, 1)[None].astype(np.float16 if is_half else np.float32)

    for _ in range(warmup):
        sess.run(None, {inp.name: x})
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        outs = sess.run(None, {inp.name: x})
        for o in outs:                       # 物化输出,避免惰性计算
            np.asarray(o).sum()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    med = statistics.median(times)
    p95 = times[int(0.95 * (len(times) - 1))]
    mb = onnx.stat().st_size / 1e6

    rss = None
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1e6
    except Exception:
        pass

    # 会话标记:同一段不间断、独占 CPU 的测量会话必须共享同一个 id,否则延迟表不可比。
    # 由调用方(remeasure_latency_all.sh)一次性 export PTQ_SESSION;缺省为空 = 来历不明,不得直接引用。
    row = {"session": os.environ.get("PTQ_SESSION", ""),
           "dataset": dataset, "scale": scale, "seed": seed, "precision": precision,
           "imgsz": imgsz, "threads": threads, "n": n,
           "lat_median_ms": round(med, 3), "lat_mean_ms": round(statistics.fmean(times), 3),
           "lat_p95_ms": round(p95, 3),
           "lat_sd_ms": round(statistics.pstdev(times), 3),
           "lat_min_ms": round(times[0], 3),
           "fps": round(1000.0 / med, 2), "size_mb": round(mb, 3),
           "sess_rss_mb": round(rss, 1) if rss else "",
           "input_type": inp.type, "file": onnx.name}
    print(f"  {dataset} {scale} seed{seed} {precision:10s} "
          f"median {med:7.2f} ms  p95 {p95:7.2f}  {mb:6.2f} MB")
    p = C.OUT / "latency.csv"
    key = ("dataset", "scale", "seed", "precision")
    old, fields = [], list(row)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            old = list(csv.DictReader(f))
        old = [r for r in old if tuple(r.get(k) for k in key) != tuple(str(row[k]) for k in key)]
        if old:
            fields = list(dict.fromkeys(list(old[0]) + fields))
    old.append(row)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in old:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", choices=list(C.DATASETS))
    ap.add_argument("--scale", choices=C.SCALES)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--precision", choices=C.PRECISIONS)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()

    if a.all:
        combos = [(d, s, sd, p) for (d, s, sd) in C.cells() for p in C.PRECISIONS]
    else:
        combos = [(a.dataset, a.scale, a.seed, a.precision)]
        if any(x is None for x in combos[0]):
            raise SystemExit("需要 --all 或同时给出 --dataset/--scale/--seed/--precision")
    print(f"CPU 线程 {a.threads},每配置 {a.n} 次(warmup {a.warmup})")
    for d, s, sd, p in combos:
        try:
            one(d, s, sd, p, a.imgsz, a.n, a.threads, a.warmup)
        except Exception as e:
            print(f"  [ERR] {d} {s} seed{sd} {p}: {type(e).__name__}: {e}")
    print(f"\n延迟测量结束:{C.OUT / 'latency.csv'}")


if __name__ == "__main__":
    main()
