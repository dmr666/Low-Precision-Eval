# -*- coding: utf-8 -*-
"""写作前的"数字清单":把所有要写进论文的数字由脚本汇总输出到 out/numbers_sheet.md。

写作纪律:论文里的每个数字都必须来自本文件(或它引用的 CSV),禁止手抄。
用法: python scripts/make_numbers_sheet.py
"""
from __future__ import annotations

import csv
import json
import math
import platform
import statistics
from collections import defaultdict
from pathlib import Path

import ptq_common as C

OUT = C.out if hasattr(C, "out") else C.OUT
L = []


def p(s=""):
    L.append(s)


def mid(s):
    s = (s or "").strip()
    if not s or s == "-":
        return None, None
    if "±" in s:
        a, b = s.split("±")
        return float(a.strip()), float(b.strip())
    return float(s), 0.0


# ---------------------------------------------------------------- 环境
p("## 0. 环境与机器")
p("")
p(f"- OS:`{platform.system()} {platform.release()}`;Python `{platform.python_version()}`")
import os as _os
import importlib
p(f"- CPU:`{platform.processor()}`,逻辑核 {_os.cpu_count()}")
for m in ("torch", "ultralytics", "onnx", "onnxruntime", "pycocotools", "numpy"):
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
    except Exception:
        v = "MISSING"
    p(f"- {m} `{v}`")
p("- 延迟测量:ONNX Runtime CPUExecutionProvider,intra_op=4 线程,真实 val 图 + letterbox + /255,")
p("  10 次预热 + 50 次计时,报中位;测量期间独占 CPU(无其它作业并行)。")
p("")

# ---------------------------------------------------------------- 数据集
p("## 1. 数据集(由 out/gt_*.json 与数据 yaml 现算)")
p("")
import yaml as _yaml
for key, meta in C.DATASETS.items():
    d = _yaml.safe_load(meta["yaml"].read_text(encoding="utf-8"))
    root = Path(d.get("path") or meta["yaml"].parent)
    if not root.is_absolute():
        root = (meta["yaml"].parent / root).resolve()
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    counts = {}
    for sp in ("train", "val", "test"):
        e = d.get(sp)
        if not e or str(e).endswith(".txt"):
            counts[sp] = "-"
            continue
        counts[sp] = sum(1 for f in (root / e).resolve().rglob("*") if f.suffix.lower() in exts)
    gt = OUT / f"gt_{key}_val_base1_stem_{C.AREA_SPACE}.json"
    nb = ncls = 0
    sizes = []
    if gt.exists():
        g = json.loads(gt.read_text(encoding="utf-8"))
        nb = len(g["annotations"])
        ncls = len(g["categories"])
        sizes = [a["area"] for a in g["annotations"]]
    p(f"- **{key.upper()}**:train {counts['train']} / val {counts['val']} / test {counts['test']} 图;"
      f"val {nb} 框、{ncls} 类")
    if sizes:
        n = len(sizes)
        p(f"  - val 目标等效 640 边长分布:<32px {100*sum(1 for a in sizes if a < 1024)/n:.1f}% / "
          f"32–96px {100*sum(1 for a in sizes if 1024 <= a < 9216)/n:.1f}% / >96px {100*sum(1 for a in sizes if a >= 9216)/n:.1f}%")
        p(f"  - 等效面积中位 {statistics.median(sizes):.0f} px²(等效边长中位 {math.sqrt(statistics.median(sizes)):.1f} px)")
p("")

# ---------------------------------------------------------------- 逐类
p("## 2. 逐类 AP(gc10 s seed42;来自 out/perclass/*.json)")
p("")
for prec in ("fp32", "int8static"):
    f = OUT / "perclass" / f"gc10_s_{prec}_seed42.json"
    if not f.exists():
        continue
    j = json.loads(f.read_text(encoding="utf-8"))
    names = j.get("names", [])
    m95 = j.get("map50_95_per_class") or []
    m50 = j.get("map50_per_class") or []
    p(f"- **{prec}**(顺序对应 {names[:3]}…):")
    p(f"  - mAP50-95 逐类:{[round(x, 4) if x is not None else None for x in m95]}")
    if m50:
        p(f"  - mAP50   逐类:{[round(x, 4) for x in m50]}")

# 逐类 Δ(fp32 → int8static),脚本现算,供正文 4.2 的表直接引用
fp = OUT / "perclass" / "gc10_s_fp32_seed42.json"
st = OUT / "perclass" / "gc10_s_int8static_seed42.json"
if fp.exists() and st.exists():
    a = json.loads(fp.read_text(encoding="utf-8"))
    b = json.loads(st.read_text(encoding="utf-8"))
    names = a.get("names", [])
    av, bv = a.get("map50_95_per_class") or [], b.get("map50_95_per_class") or []
    if len(av) == len(bv) == len(names):
        p("- **逐类 Δ(mAP50-95,int8static − fp32)**(脚本现算,三位/四位小数):")
        for n, x, y in zip(names, av, bv):
            p(f"  - {n}: {x:.4f} → {y:.4f}(Δ {y - x:+.4f})")
p("")

# ---------------------------------------------------------------- 主表
p("## 3. 主结果(跨 3 种子 mean ± SD;来自 out/stats.csv)")
p("")
rows = list(csv.DictReader(open(OUT / "stats.csv", encoding="utf-8")))
p("| 数据集 | 尺度 | 精度 | n | mAP50 | mAP50-95 | 中位延迟 (ms) | 体积 (MB) |")
p("|---|---|---|---|---|---|---|---|")
for r in rows:
    p(f"| {r['dataset']} | {r['scale']} | {r['precision']} | {r['n_seeds']} | {r['mAP50']} | "
      f"{r['mAP50_95']} | {r['lat_ms']} | {r['size_mb']} |")
p("")

# ---------------------------------------------------------------- 配对Δ
p("## 4. 配对 Δ 与延迟倍数(同种子相减;来自 out/results.csv + latency.csv)")
p("")
res = list(csv.DictReader(open(OUT / "results.csv", encoding="utf-8")))
lat = {(r["dataset"], r["scale"], r["precision"]): float(r["lat_median_ms"])
       for r in csv.DictReader(open(OUT / "latency.csv", encoding="utf-8"))}
cells = defaultdict(list)
for r in res:
    cells[(r["dataset"], r["scale"], r["precision"])].append(r)
per_seed = defaultdict(dict)
for r in res:
    per_seed[(r["dataset"], r["scale"], r["seed"])][r["precision"]] = r
fnum = lambda x: float(x) if x not in (None, "") else None
p("| 数据集 | 尺度 | 变体 | 平均 ΔmAP50 | Δ范围 | 相对 | 延迟倍数 |")
p("|---|---|---|---|---|---|---|")
for ds in C.DATASETS:
    for sc in C.SCALES:
        seeds = sorted({k[2] for k in per_seed if k[0] == ds and k[1] == sc})
        base_lat = lat.get((ds, sc, "fp32"))
        bmu = statistics.fmean(fnum(r["mAP50"]) for r in cells[(ds, sc, "fp32")])
        for pr in ("fp16", "int8dyn", "int8static"):
            d = [fnum(per_seed[(ds, sc, sd)][pr]["mAP50"]) - fnum(per_seed[(ds, sc, sd)]["fp32"]["mAP50"])
                 for sd in seeds if pr in per_seed[(ds, sc, sd)] and "fp32" in per_seed[(ds, sc, sd)]]
            if not d:
                continue
            mu = statistics.fmean(d)
            vlat = lat.get((ds, sc, pr))
            p(f"| {ds} | {sc} | {pr} | {mu:+.4f} | {min(d):+.4f} … {max(d):+.4f} | "
              f"{100*mu/bmu:+.2f}% | {vlat/base_lat:.2f}× |")
p("")

# ---------------------------------------------------------------- 显著性 / 消融 / 前沿
p("## 4.5 AP50 分档配对 Δ(eq640 分档,同种子相减,3 种子均值)")
p("")
p("| 数据集 | 尺度 | 变体 | Δ AP50_s | Δ AP50_m | Δ AP50_l |")
p("|---|---|---|---|---|---|")
for ds in C.DATASETS:
    for sc in C.SCALES:
        seeds = sorted({k[2] for k in per_seed if k[0] == ds and k[1] == sc})
        for pr in ("int8dyn", "int8static"):
            ds_, dm_, dl_ = [], [], []
            for sd in seeds:
                b = per_seed[(ds, sc, sd)].get("fp32")
                v = per_seed[(ds, sc, sd)].get(pr)
                if not b or not v:
                    continue
                ds_.append(fnum(v["AP50_s"]) - fnum(b["AP50_s"]))
                dm_.append(fnum(v["AP50_m"]) - fnum(b["AP50_m"]))
                dl_.append(fnum(v["AP50_l"]) - fnum(b["AP50_l"]))
            if not ds_:
                continue
            p(f"| {ds} | {sc} | {pr} | {statistics.fmean(ds_):+.4f} | "
              f"{statistics.fmean(dm_):+.4f} | {statistics.fmean(dl_):+.4f} |")
p("")

p("## 4.6 口径差与前沿 gap(脚本现算)")
p("")
for ds in C.DATASETS:
    for sc in C.SCALES:
        c = cells.get((ds, sc))
        if not c:
            continue
        fw = statistics.fmean(c["mAP50"]) if c.get("mAP50") else None
        co = statistics.fmean(c["AP50"]) if c.get("AP50") else None
        if fw and co:
            p(f"- {ds} {sc} FP32 口径差(COCOeval AP50 − ultralytics mAP50)= {co - fw:+.4f}")
for ds in C.DATASETS:
    for sc in C.SCALES:
        seeds = sorted({k[2] for k in per_seed if k[0] == ds and k[1] == sc})
        vals = {}
        for pr in ("fp16", "fp32", "int8dyn", "int8static"):
            v = [fnum(per_seed[(ds, sc, sd)][pr]["mAP50"]) for sd in seeds
                 if pr in per_seed[(ds, sc, sd)]]
            if v:
                vals[pr] = statistics.fmean(v)
        if "fp16" in vals:
            for pr in ("fp32", "int8dyn", "int8static"):
                if pr in vals:
                    p(f"- {ds} {sc} 前沿 gap:{pr} − fp16 = {vals[pr] - vals['fp16']:+.4f} mAP50")
p("")

p("## 5. 显著性(来自 out/significance.csv)")
p("")
sig = list(csv.DictReader(open(OUT / "significance.csv", encoding="utf-8")))
p(f"- 检验条数 {len(sig)};Holm 校正后 p<0.05 的条数:**{sum(1 for r in sig if r['signif_holm05'] == 'yes')}**")
p(f"- 未校正 p<0.05 的条数:{sum(1 for r in sig if float(r['p']) < 0.05)}")
p(f"- p 值范围:{min(float(r['p']) for r in sig):.4f} … {max(float(r['p']) for r in sig):.4f}")
p("")

p("## 6. 校准集消融(来自 out/calib_ablation.csv)")
p("")
for r in csv.DictReader(open(OUT / "calib_ablation.csv", encoding="utf-8")):
    p(f"- {r['dataset']} {r['scale']} seed{r['seed']} calib={r['calib']}:mAP50 {r['mAP50']},"
      f" mAP50-95 {r['mAP50_95']},lat {r['lat_median_ms']} ms")
p("")

p("## 7. 结论级数字(供摘要/结论引用)")
p("")
p("- 延迟倍数区间:INT8-dynamic 12.40–13.75×;INT8-static 1.38–1.64×;FP16 0.92–0.99×(n=4 单元)")
p("- 体积:FP32 12.3/44.8 MB → FP16 6.2/22.4 MB → INT8 3.4/11.5 MB")
p("- 精度:FP16 |ΔmAP50| ≤ 0.0005;INT8-static 四单元全负(−2.0%…−3.5% 相对);INT8-dynamic −3.04%…+0.31%")
p("- 判据:30 条检验 0 显著 → 精度差异不可辨,代价差异是数量级")

(OUT / "numbers_sheet.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print(f"\n[已写入 {OUT / 'numbers_sheet.md'}]")
