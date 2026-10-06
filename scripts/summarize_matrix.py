# -*- coding: utf-8 -*-
"""汇总全矩阵:跨种子 mean±SD、**配对** Δ vs FP32(同种子相减)、门禁核对、成本对照。

用法: python scripts/summarize_matrix.py
所有数字来自 out/results.csv 与 out/latency.csv,禁止手抄。
"""
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import ptq_common as C

RES = C.OUT / "results.csv"
LAT = C.OUT / "latency.csv"

# 论文三的 FP32 参考(val mAP50)
# 注:0.6423 是论文三 **s 尺度 5 种子** 的扩展值,不能用于 nano,故 nano 不设参考。
REF = {
    ("gc10", "s"): {"mAP50": 0.6390, "sd": 0.0121},
    ("gc10", "n"): None,
    ("neu", "s"): None,
    ("neu", "n"): None,
}


def load(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def msd(v: list) -> tuple:
    v = [x for x in v if x is not None]
    if not v:
        return (None, None, 0)
    if len(v) == 1:
        return (v[0], 0.0, 1)
    return (statistics.fmean(v), statistics.stdev(v), len(v))


def main() -> None:
    res = load(RES)
    lat = {(r["dataset"], r["scale"], r["seed"], r["precision"]): r for r in load(LAT)}
    print(f"results.csv {len(res)} 行 | latency.csv {len(lat)} 行")
    print(f"缺失评测:{48 - len(res)} 行;缺失延迟:{48 - len(lat)} 行")

    # 收集
    cells = defaultdict(dict)          # (ds, sc, prec) -> metric -> [values]
    per_seed = defaultdict(dict)       # (ds, sc, seed) -> prec -> row
    for r in res:
        k = (r["dataset"], r["scale"], r["precision"])
        for m in ("mAP50", "mAP50_95", "AP50", "APs", "APm", "APl", "AP50_s", "AP50_m", "AP50_l"):
            v = f(r.get(m))
            if v is not None:
                cells[k].setdefault(m, []).append(v)
        lk = (r["dataset"], r["scale"], r["seed"], r["precision"])
        if lk in lat:
            cells[k].setdefault("lat", []).append(f(lat[lk]["lat_median_ms"]))
            cells[k].setdefault("size", []).append(f(lat[lk]["size_mb"]))
        per_seed[(r["dataset"], r["scale"], r["seed"])][r["precision"]] = r

    # ---- 表 1:精度 mean±SD
    print("\n## 表 1 · 精度(跨种子 mean ± SD,ultralytics 口径)")
    print("| 数据集 | 尺度 | 精度 | mAP50 | mAP50-95 | AP50(COCO) | AP50_s | AP50_m | AP50_l |")
    print("|---|---|---|---|---|---|---|---|---|")
    order = []
    for ds in C.DATASETS:
        for sc in C.SCALES:
            for pr in C.PRECISIONS:
                order.append((ds, sc, pr))
    for ds, sc, pr in order:
        c = cells.get((ds, sc, pr))
        if not c:
            continue
        def cell(m):
            mu, sd, n = msd(c.get(m, []))
            return "-" if mu is None else (f"{mu:.4f} ± {sd:.4f}" if n > 1 else f"{mu:.4f}")
        print(f"| {ds} | {sc} | {pr} | {cell('mAP50')} | {cell('mAP50_95')} | {cell('AP50')} | "
              f"{cell('AP50_s')} | {cell('AP50_m')} | {cell('AP50_l')} |")

    # ---- 表 2:延迟/体积
    print("\n## 表 2 · 部署代价(跨种子)")
    print("| 数据集 | 尺度 | 精度 | 中位延迟 (ms) | 体积 (MB) |")
    print("|---|---|---|---|---|")
    for ds, sc, pr in order:
        c = cells.get((ds, sc, pr))
        if not c:
            continue
        def cell(m):
            mu, sd, n = msd(c.get(m, []))
            return "-" if mu is None else (f"{mu:.1f} ± {sd:.1f}" if n > 1 else f"{mu:.1f}")
        print(f"| {ds} | {sc} | {pr} | {cell('lat')} | {cell('size')} |")

    # ---- 表 3:配对 Δ vs FP32(同种子相减,消除种子噪声)
    print("\n## 表 3 · 配对 Δ(mAP50)相对 FP32 —— 同种子相减")
    print("| 数据集 | 尺度 | 变体 | 种子数 | 平均 Δ | Δ范围 | 相对 FP32 | 延迟倍数 |")
    print("|---|---|---|---|---|---|---|---|")
    for ds in C.DATASETS:
        for sc in C.SCALES:
            seeds = sorted({k[2] for k in per_seed if k[0] == ds and k[1] == sc})
            base_lat = msd(cells[(ds, sc, "fp32")].get("lat", []))[0] if cells.get((ds, sc, "fp32")) else None
            for pr in C.PRECISIONS:
                if pr == "fp32":
                    continue
                ds_, lo, hi = [], None, None
                for sd_ in seeds:
                    b = per_seed[(ds, sc, sd_)].get("fp32")
                    v = per_seed[(ds, sc, sd_)].get(pr)
                    if not b or not v:
                        continue
                    d = f(v["mAP50"]) - f(b["mAP50"])
                    ds_.append(d)
                    lo = d if lo is None else min(lo, d)
                    hi = d if hi is None else max(hi, d)
                if not ds_:
                    continue
                mu = statistics.fmean(ds_)
                rel = None
                bmu = msd(cells[(ds, sc, "fp32")]["mAP50"])[0]
                if bmu:
                    rel = 100.0 * mu / bmu
                vlat = msd(cells[(ds, sc, pr)].get("lat", []))[0]
                ratio = (vlat / base_lat) if (vlat and base_lat) else None
                print(f"| {ds} | {sc} | {pr} | {len(ds_)} | {mu:+.4f} | {lo:+.4f} … {hi:+.4f} | "
                      f"{rel:+.2f}% | {ratio:.2f}x |")

    # ---- 门禁:FP32 与论文三对表
    print("\n## 门禁 · FP32 与论文三对表(val mAP50)")
    for (ds, sc), ref in REF.items():
        c = cells.get((ds, sc, "fp32"), {})
        mu, sd, n = msd(c.get("mAP50", []))
        if mu is None:
            print(f"- {ds} {sc}: 本矩阵缺 FP32 行")
            continue
        if ref is None:
            print(f"- {ds} {sc}: 本矩阵 FP32 = {mu:.4f} ± {sd:.4f}(n={n}),论文三无对应参考,记录备查")
            continue
        d = mu - ref["mAP50"]
        verdict = "✓ 在容差内" if abs(d) <= C.TOL else ("~ 略超容差" if abs(d) <= 2 * C.TOL else "✗ 超差,需查口径")
        print(f"- {ds} {sc}: 本矩阵 {mu:.4f} ± {sd:.4f}(n={n}) vs 论文三 {ref['mAP50']:.4f} ± {ref['sd']:.4f} "
              f"→ Δ = {d:+.4f} {verdict}")

    # ---- 小目标敏感度(配对)
    print("\n## 小目标敏感度(配对 Δ AP50_s 与 Δ AP50_l,eq640 口径)")
    print("| 数据集 | 尺度 | 变体 | Δ AP50_s | Δ AP50_l | 小/大倍数 |")
    print("|---|---|---|---|---|---|")
    for ds in C.DATASETS:
        for sc in C.SCALES:
            seeds = sorted({k[2] for k in per_seed if k[0] == ds and k[1] == sc})
            for pr in ("int8dyn", "int8static"):
                ds_s, ds_l = [], []
                for sd_ in seeds:
                    b = per_seed[(ds, sc, sd_)].get("fp32")
                    v = per_seed[(ds, sc, sd_)].get(pr)
                    if not b or not v or f(b.get("AP50_s")) is None or f(v.get("AP50_s")) is None:
                        continue
                    ds_s.append(f(v["AP50_s"]) - f(b["AP50_s"]))
                    ds_l.append(f(v["AP50_l"]) - f(b["AP50_l"]))
                if not ds_s:
                    continue
                ms, ml = statistics.fmean(ds_s), statistics.fmean(ds_l)
                ratio = (abs(ms) / abs(ml)) if ml else float("nan")
                print(f"| {ds} | {sc} | {pr} | {ms:+.4f} | {ml:+.4f} | {ratio:.1f}× |")


if __name__ == "__main__":
    main()
