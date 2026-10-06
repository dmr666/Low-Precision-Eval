# -*- coding: utf-8 -*-
"""导出四种精度(零训练):fp32 / fp16 / int8-dynamic / int8-static。

- fp32/fp16/int8-static 由 ultralytics export 完成(int8-static 用数据集校准);
- int8-dynamic 由 onnxruntime.quantization.quantize_dynamic 在 fp32 ONNX 上完成(无需校准数据)。
幂等:已存在的变体默认跳过(--force 覆盖)。每次导出把体积写入 out/export_sizes.csv。

用法:
  python ptq_export.py --all --variants fp32,fp16,int8dyn,int8static --imgsz 640
  python ptq_export.py --dataset gc10 --scale n --seed 42 --variants fp32
"""
from __future__ import annotations

import argparse
import csv
import shutil
import traceback
from pathlib import Path

import ptq_common as C


def log_csv(row: dict) -> None:
    p = C.OUT / "export_sizes.csv"
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def stage_export(w: Path, imgsz: int, half: bool = False, int8: bool = False,
                 data_yaml: Path | None = None) -> Path:
    """在纯 ASCII 临时目录里做 ultralytics 导出,再把产物拷回。

    原因:onnxruntime 量化(ultralytics int8 导出内部会调它)会写/读
    '<model>-inferred.onnx',在非 ASCII 路径(本项目目录含中文)上会
    FileNotFoundError —— 与动态量化同一个坑。
    """
    import shutil
    import tempfile

    from ultralytics import YOLO

    stage = Path(tempfile.gettempdir()) / "ptq_export_stage"
    stage.mkdir(parents=True, exist_ok=True)
    w_local = stage / "model.pt"
    shutil.copyfile(w, w_local)
    m = YOLO(str(w_local))
    kw = dict(format="onnx", imgsz=imgsz, simplify=True, opset=17,
              dynamic=False, device="cpu")
    if half:
        kw["half"] = True
    if int8:
        kw["int8"] = True
        kw["data"] = str(data_yaml)
    return Path(m.export(**kw))


def export_fp32_or_fp16(w: Path, imgsz: int, half: bool) -> Path:
    return stage_export(w, imgsz, half=half)


def export_int8_static(w: Path, imgsz: int, data_yaml: Path) -> Path:
    return stage_export(w, imgsz, int8=True, data_yaml=data_yaml)


def make_int8_dynamic(fp32: Path, dest: Path) -> Path:
    """int8 动态量化。

    两个坑都绕开:
      1) onnxruntime 1.30 的 quantize_dynamic(model_input, model_output) 两个参数都必填;
      2) ORT 会在输入同目录写 '<name>-inferred.onnx',在非 ASCII 路径上会 FileNotFoundError
         -> 全程在 ASCII 临时目录里做,最后再把结果拷回项目目录。
    """
    import shutil
    import tempfile
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tmp = Path(tempfile.gettempdir()) / "ptq_quant"
    tmp.mkdir(parents=True, exist_ok=True)
    src, out = tmp / "in_fp32.onnx", tmp / "out_int8.onnx"
    shutil.copyfile(fp32, src)
    quantize_dynamic(model_input=str(src), model_output=str(out),
                     weight_type=QuantType.QInt8)
    shutil.copyfile(out, dest)
    return dest


def one(dataset: str, scale: str, seed: int, variants: list, imgsz: int,
        force: bool) -> None:
    meta = C.DATASETS[dataset]
    w = C.weight_path(dataset, scale, seed)
    if not w.exists():
        print(f"  [skip] 权重不存在:{w.name}")
        return
    print(f"\n=== {dataset} {scale} seed{seed} ===  权重 {w.stat().st_size / 1e6:.1f} MB")
    fp32_done = None
    for v in variants:
        dest = C.onnx_path(dataset, scale, seed, v)
        if dest.exists() and not force:
            print(f"  [skip] {v:10s} 已存在 {dest.name}")
            if v == "fp32":
                fp32_done = dest
            continue
        try:
            if v == "fp32":
                src = export_fp32_or_fp16(w, imgsz, half=False)
                shutil.copyfile(src, dest)
                fp32_done = dest
            elif v == "fp16":
                src = export_fp32_or_fp16(w, imgsz, half=True)
                shutil.copyfile(src, dest)
            elif v == "int8static":
                src = export_int8_static(w, imgsz, meta["yaml"])
                shutil.copyfile(src, dest)
            elif v == "int8dyn":
                if fp32_done is None:
                    fp32_done = C.onnx_path(dataset, scale, seed, "fp32")
                if not fp32_done.exists():
                    src = export_fp32_or_fp16(w, imgsz, half=False)
                    shutil.copyfile(src, fp32_done)
                make_int8_dynamic(fp32_done, dest)
            else:
                print(f"  [skip] 未知变体 {v}")
                continue
            mb = dest.stat().st_size / 1e6
            print(f"  [ok]   {v:10s} {mb:7.2f} MB  {dest.name}")
            log_csv({"dataset": dataset, "scale": scale, "seed": seed,
                     "precision": v, "size_mb": round(mb, 3),
                     "imgsz": imgsz, "file": dest.name})
        except Exception as e:
            print(f"  [ERR]  {v}: {type(e).__name__}: {e}")
            with open(C.OUT / "errors.log", "a", encoding="utf-8") as f:
                f.write(f"export {dataset} {scale} seed{seed} {v}\n{traceback.format_exc()}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", choices=list(C.DATASETS))
    ap.add_argument("--scale", choices=C.SCALES)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--variants", default="fp32,fp16,int8dyn,int8static")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    variants = [s.strip() for s in a.variants.split(",") if s.strip()]

    if a.all:
        combos = C.cells()
    else:
        combos = [(a.dataset, a.scale, a.seed)]
    for d, s, sd in combos:
        if d is None or s is None or sd is None:
            raise SystemExit("需要 --all 或同时给出 --dataset/--scale/--seed")
        one(d, s, sd, variants, a.imgsz, a.force)
    print(f"\n导出完成。ONNX 目录:{C.ONNX}")


if __name__ == "__main__":
    main()
