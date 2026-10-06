# -*- coding: utf-8 -*-
"""预检:确认数据 / 权重 / 依赖 / 目录就绪,缺什么直接说清楚。

用法: python preflight.py
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import ptq_common as C


def check_env() -> list:
    problems = []
    mods = ["torch", "ultralytics", "onnx", "onnxruntime", "pycocotools",
            "numpy", "PIL", "yaml"]
    print("=== 依赖 ===")
    for m in mods:
        try:
            mod = importlib.import_module(m)
            print(f"  ok   {m:14s} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"  MISS {m:14s} {type(e).__name__}")
            problems.append(f"缺依赖:{m}")
    try:
        import onnxruntime as ort
        print(f"  ok   providers     {ort.get_available_providers()}")
        if "CPUExecutionProvider" not in ort.get_available_providers():
            problems.append("onnxruntime 没有 CPUExecutionProvider")
    except Exception:
        pass
    return problems


def check_data() -> list:
    problems = []
    print("\n=== 数据集 ===")
    for key, meta in C.DATASETS.items():
        y = meta["yaml"]
        if not y.exists():
            print(f"  MISS {key:5s} {y}")
            problems.append(f"缺数据 yaml:{y}")
            continue
        txt = y.read_text(encoding="utf-8")
        n_img = 0
        try:
            import yaml as _y
            d = _y.safe_load(txt)
            root = Path(d.get("path") or y.parent)
            if not root.is_absolute():
                root = (y.parent / root).resolve()
            for split in ("train", "val", "test"):
                e = d.get(split)
                if not e or str(e).endswith(".txt"):
                    continue
                p = (root / e).resolve()
                c = sum(1 for f in p.rglob("*")
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
                n_img += c
                print(f"  ok   {key:5s} {split:5s} {c:5d} 图  {p}")
            print(f"       {key:5s} yaml={y}")
            if n_img == 0:
                problems.append(f"{key}: 一个图片都没数到,检查 yaml 的 path/train/val")
        except Exception as e:
            print(f"  WARN {key:5s} 解析 yaml 失败:{e}")
            problems.append(f"{key}: 数据 yaml 解析失败")
    return problems


def check_weights() -> list:
    print("\n=== 权重 ===")
    problems = []
    found, missing = 0, []
    for ds, sc, sd in C.cells():
        p = C.weight_path(ds, sc, sd)
        if p.exists():
            found += 1
        else:
            missing.append(f"{ds}_{sc}_seed{sd}")
    print(f"  已就绪 {found} / {len(C.cells())}")
    if missing:
        print("  缺失(先跑 extract_weights.py):")
        for m in missing[:20]:
            print(f"    {m}")
    tars = sorted((C.PAPER3 / "_results" / "weights_20261002").glob("*.tar")) \
        if (C.PAPER3 / "_results" / "weights_20261002").exists() else []
    print(f"  tar 备份:{[t.name for t in tars] or '未找到'}")
    if found == 0 and tars:
        problems.append("权重全缺:需要从 tar 解出(extract_weights.py)")
    return problems


def check_out() -> list:
    print("\n=== 输出目录 ===")
    for d in (C.OUT, C.WEIGHTS, C.ONNX, C.RUNS):
        print(f"  {'ok  ' if d.exists() else 'MISS'} {d}")
    for name in ("results.csv", "latency.csv", "stats.csv"):
        p = C.OUT / name
        if p.exists():
            n = len(p.read_text(encoding="utf-8").splitlines()) - 1
            print(f"  {name:14s} 已有 {n} 行(追加模式,可续跑)")
    return []


def main() -> None:
    print(f"PROJ = {C.PROJ}")
    print(f"PAPER3 = {C.PAPER3}\n")
    problems = check_env() + check_data() + check_weights() + check_out()
    print("\n=== 结论 ===")
    if problems:
        for p in problems:
            print(f"  ! {p}")
        sys.exit(1)
    print("  全部就绪,可以开跑。")


if __name__ == "__main__":
    main()
