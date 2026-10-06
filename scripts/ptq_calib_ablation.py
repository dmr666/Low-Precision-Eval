# -*- coding: utf-8 -*-
"""校准集消融:静态 INT8 的损失是"校准数据不好"造成的,还是"静态量化本身"造成的?

与 ultralytics 8.4.115 的实现对齐(ultralytics/utils/export/onnx.py::onnx_int8_quantize):
  exclude = 所有非 Conv/Gemm/MatMul 的节点;quantize_static(onnx, out, reader, nodes_to_exclude=exclude)
  其余参数全部用 onnxruntime 默认(与 ultralytics 一致)。
三种校准来源(同一套量化参数,只换 reader):
  val      : 该数据集 val split 全部图  —— ultralytics 默认;本矩阵的 int8static 即此(作为自检基准)
  train200 : 从 train split 等间隔取 200 张(同分布、样本更多、且非评测集)
  random20 : 20 个随机均匀 NCHW 张量(完全不含真实数据)

对每个 (dataset, scale, seed) 的 **已有 FP32 ONNX** 生成三个静态模型 → 评测 + 测延迟。
零训练;输出 out/calib_ablation.csv。

用法: python scripts/ptq_calib_ablation.py            # 默认 gc10 s 42 与 neu s 42
      python scripts/ptq_calib_ablation.py --configs gc10:n:42
"""
from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import ptq_common as C
from ptq_latency import letterbox_bgr

CALIB_DIR = C.PROJ / "calib"
CALIB_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 数据侧
def split_images(dataset: str, split: str) -> list:
    """按 yaml 解析出该 split 的图片列表(排序,保证可复现)。"""
    import yaml
    meta = C.DATASETS[dataset]
    d = yaml.safe_load(meta["yaml"].read_text(encoding="utf-8"))
    root = Path(d.get("path") or meta["yaml"].parent)
    if not root.is_absolute():
        root = (meta["yaml"].parent / root).resolve()
    entry = d.get(split)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if isinstance(entry, str) and entry.endswith(".txt"):
        lst = (root / entry) if not Path(entry).is_absolute() else Path(entry)
        files = [l.strip() for l in lst.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [Path(f) if Path(f).is_absolute() else (root / f).resolve() for f in files]
    d0 = (root / entry).resolve()
    return sorted(p for p in d0.rglob("*") if p.suffix.lower() in exts)


def pick_even(files: list, k: int) -> list:
    if len(files) <= k:
        return files
    step = len(files) / k
    return [files[int(i * step)] for i in range(k)]


class FilesReader:
    """真实图片校准 reader:letterbox + **/255 归一化** → NCHW float32(0–1)。

    与 ultralytics 的 transform_fn 口径一致(其预处理器会把 uint8 转成 0–1 的 float);
    漏掉 /255 会让校准激活定标错 255 倍,静态模型直接崩(实测 mAP 归零)。
    """

    def __init__(self, files: list, imgsz: int, input_name: str = "images"):
        self.samples = []
        for f in files:
            im = letterbox_bgr(f, imgsz).transpose(2, 0, 1)[None].astype(np.float32)
            self.samples.append({input_name: im})
        self.it = iter(self.samples)

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter(self.samples)


class RandomReader:
    """随机数据校准 reader(对照:不含任何真实数据)。"""

    def __init__(self, n: int, imgsz: int, input_name: str = "images", seed: int = 0):
        rs = np.random.RandomState(seed)
        self.samples = [{input_name: rs.rand(1, 3, imgsz, imgsz).astype(np.float32)}
                        for _ in range(n)]
        self.it = iter(self.samples)

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter(self.samples)


# ---------------------------------------------------------------- 量化
def quantize_like_ultralytics(fp32: Path, dest: Path, reader) -> Path:
    """复刻 ultralytics 的 onnx_int8_quantize(全部默认参数 + 只量化 Conv/Gemm/MatMul)。

    在 ASCII 临时目录里做 —— ORT 在非 ASCII 路径上会写/读 '<name>-inferred.onnx' 而失败。
    """
    import onnx
    from onnxruntime.quantization import quantize_static

    tmp = Path(tempfile.gettempdir()) / "ptq_calib"
    tmp.mkdir(parents=True, exist_ok=True)
    src, out = tmp / "in_fp32.onnx", tmp / "out_int8.onnx"
    shutil.copyfile(fp32, src)
    src2 = tmp / "in_fp32_excl.onnx"
    shutil.copyfile(fp32, src2)

    graph = onnx.load(str(src2)).graph
    exclude = [n.name for n in graph.node if n.op_type not in {"Conv", "Gemm", "MatMul"}]
    quantize_static(str(src), str(out), reader, nodes_to_exclude=exclude)
    shutil.copyfile(out, dest)
    return dest


# ---------------------------------------------------------------- 评测
def evaluate(onnx: Path, dataset: str, imgsz: int) -> dict:
    from ultralytics import YOLO
    r = YOLO(str(onnx)).val(data=str(C.DATASETS[dataset]["yaml"]), split="val", imgsz=imgsz,
                            verbose=False, plots=False)
    return {"mAP50": round(float(r.box.map50), 5), "mAP50_95": round(float(r.box.map), 5)}


def latency(onnx: Path, dataset: str, imgsz: int, n: int = 30, warmup: int = 5,
            threads: int = 4) -> float:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = letterbox_bgr(split_images(dataset, "val")[0], imgsz).transpose(2, 0, 1)[None].astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {inp.name: x})
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        outs = sess.run(None, {inp.name: x})
        for o in outs:
            np.asarray(o).sum()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return round(statistics.median(ts), 3)


# ---------------------------------------------------------------- 主流程
def run(dataset: str, scale: str, seed: int, imgsz: int, sources: list) -> None:
    fp32 = C.onnx_path(dataset, scale, seed, "fp32")
    if not fp32.exists():
        print(f"  [skip] 缺 FP32 ONNX:{fp32.name}")
        return
    val_files = split_images(dataset, "val")
    train_files = split_images(dataset, "train")
    t200 = pick_even(train_files, 200)
    print(f"\n=== {dataset} {scale} seed{seed} === val {len(val_files)} 图 / train {len(train_files)} 图")
    print(f"  train200 抽出的文件已写入 calib/{dataset}_train200.txt(等间隔,可复现)")
    (CALIB_DIR / f"{dataset}_train200.txt").write_text(
        "\n".join(str(p) for p in t200), encoding="utf-8")

    p = C.OUT / "calib_ablation.csv"
    for src in sources:
        if src == "val":
            reader = FilesReader(val_files, imgsz)
        elif src == "train200":
            reader = FilesReader(t200, imgsz)
        elif src == "random20":
            reader = RandomReader(20, imgsz, seed=seed)
        else:
            print(f"  [skip] 未知来源 {src}")
            continue
        dest = CALIB_DIR / f"{dataset}_{scale}_seed{seed}_int8static_{src}.onnx"
        print(f"  -- 校准来源 {src:9s} -> {dest.name}")
        try:
            quantize_like_ultralytics(fp32, dest, reader)
            ev = evaluate(dest, dataset, imgsz)
            lat = latency(dest, dataset, imgsz)
            mb = dest.stat().st_size / 1e6
            print(f"     mAP50={ev['mAP50']:.5f}  mAP50-95={ev['mAP50_95']:.5f}  "
                  f"latency={lat:.1f} ms  size={mb:.2f} MB")
            new = not p.exists()
            with open(p, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["dataset", "scale", "seed", "calib",
                                                  "mAP50", "mAP50_95", "lat_median_ms",
                                                  "size_mb", "file"])
                if new:
                    w.writeheader()
                w.writerow({"dataset": dataset, "scale": scale, "seed": seed, "calib": src,
                            "mAP50": ev["mAP50"], "mAP50_95": ev["mAP50_95"],
                            "lat_median_ms": lat, "size_mb": round(mb, 3), "file": dest.name})
        except Exception as e:
            print(f"     [ERR] {type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="gc10:s:42,neu:s:42",
                    help="逗号分隔的 dataset:scale:seed")
    ap.add_argument("--sources", default="val,train200,random20")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    combos = []
    for c in a.configs.split(","):
        d, s, sd = c.strip().split(":")
        combos.append((d, s, int(sd)))
    sources = [s.strip() for s in a.sources.split(",") if s.strip()]
    for d, s, sd in combos:
        run(d, s, sd, a.imgsz, sources)
    print(f"\n结果:{C.OUT / 'calib_ablation.csv'}")


if __name__ == "__main__":
    main()
