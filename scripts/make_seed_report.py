# -*- coding: utf-8 -*-
"""生成补种子报告 out/seed_extension_report.md —— 全部数字现场从 out/*.csv 算,禁止手抄。

用法:python scripts/make_seed_report.py
补种子(C1/C2)后随时可重跑;报告随数据自动更新。
"""
from __future__ import annotations

import collections
import csv
import statistics
from pathlib import Path

import ptq_common as C

OUT = C.OUT
L: list[str] = []


def f(x):
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return None


def p(s: str = "") -> None:
    L.append(s)


def main() -> None:
    res = list(csv.DictReader(open(OUT / "results.csv", encoding="utf-8-sig")))
    lat = list(csv.DictReader(open(OUT / "latency.csv", encoding="utf-8-sig")))
    sig = list(csv.DictReader(open(OUT / "significance.csv", encoding="utf-8-sig")))

    p("# 论文四 · 补种子(seed 13 / 99)报告")
    p()
    p("> 本文件由 `scripts/make_seed_report.py` 从 `out/results.csv`、`out/latency.csv`、"
      "`out/significance.csv` 现场生成。数字不得手工修改。")
    p()
    counts = collections.Counter((r["dataset"], r["scale"]) for r in res if r["precision"] == "fp32")
    p(f"评测行数 {len(res)};延迟行数 {len(lat)};每个格子的种子数:"
      + "、".join(f"{k[0]}-{k[1]}={v}" for k, v in sorted(counts.items())))
    p()

    # ---- 表 1:各格子 mAP50
    p("## 1. mAP50(跨种子 mean ± SD)")
    p()
    p("| dataset | scale | precision | n seeds | mAP50 | SD |")
    p("|---|---|---|---|---|---|")
    g = collections.defaultdict(list)
    for r in res:
        g[(r["dataset"], r["scale"], r["precision"])].append(f(r["mAP50"]))
    for k in sorted(g):
        v = [x for x in g[k] if x is not None]
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        p(f"| {k[0]} | {k[1]} | {k[2]} | {len(v)} | {statistics.fmean(v):.4f} | {sd:.4f} |")
    p()

    # ---- 表 2:配对 Δ
    p("## 2. 配对 ΔmAP50(vs 同种子 FP32)")
    p()
    p("| dataset | scale | variant | n seeds | Δ mean | per-seed min | per-seed max | 相对变化 |")
    p("|---|---|---|---|---|---|---|---|")
    for ds in C.DATASETS:
        for sc in C.SCALES:
            seeds = sorted({r["seed"] for r in res if r["dataset"] == ds and r["scale"] == sc})
            for pr in ("fp16", "int8dyn", "int8static"):
                d, base_m = [], []
                for sd in seeds:
                    b = [r for r in res if r["dataset"] == ds and r["scale"] == sc
                         and r["precision"] == "fp32" and r["seed"] == sd]
                    x = [r for r in res if r["dataset"] == ds and r["scale"] == sc
                         and r["precision"] == pr and r["seed"] == sd]
                    if b and x:
                        d.append(f(x[0]["mAP50"]) - f(b[0]["mAP50"]))
                        base_m.append(f(b[0]["mAP50"]))
                if not d:
                    continue
                rel = statistics.fmean(d) / statistics.fmean(base_m) * 100 if statistics.fmean(base_m) else 0
                p(f"| {ds} | {sc} | {pr} | {len(d)} | {statistics.fmean(d):+.4f} | {min(d):+.4f} | "
                  f"{max(d):+.4f} | {rel:+.2f}% |")
    p()

    # ---- 表 3:延迟与体积
    p("## 3. 延迟与体积(来自 latency.csv)")
    p()
    p("| dataset | scale | precision | latency (ms) | size (MB) | vs FP32 延迟比 |")
    p("|---|---|---|---|---|---|")
    for ds in C.DATASETS:
        for sc in C.SCALES:
            base = [f(r["lat_median_ms"]) for r in lat if r["dataset"] == ds and r["scale"] == sc
                    and r["precision"] == "fp32"]
            bm = statistics.fmean(base) if base else 0
            for pr in C.PRECISIONS:
                rows = [r for r in lat if r["dataset"] == ds and r["scale"] == sc and r["precision"] == pr]
                if not rows:
                    continue
                ms = statistics.fmean([f(r["lat_median_ms"]) for r in rows
                                       if f(r["lat_median_ms"]) is not None])
                mb = statistics.fmean([f(r["size_mb"]) for r in rows if f(r["size_mb"]) is not None])
                ratio = f"{ms / bm:.2f}x" if bm and ms else "?"
                p(f"| {ds} | {sc} | {pr} | {ms:.1f} | {mb:.1f} | {ratio} |")
    p()

    # ---- 表 4:显著性
    p("## 4. 显著性(Welch t + Holm,全部比较)")
    p()
    n_yes = sum(1 for s in sig if s.get("signif_holm05") == "yes")
    unc = [s for s in sig if f(s["p"]) is not None and f(s["p"]) < 0.05]
    p(f"- 比较总数 **{len(sig)}**;Holm 校正后显著 **{n_yes}** 个;未校正 p<0.05 有 **{len(unc)}** 个")
    p()
    if unc:
        p("| dataset | scale | variant | metric | Δ(mean) | p | p(Holm) |")
        p("|---|---|---|---|---|---|---|")
        for s in unc:
            p(f"| {s['dataset']} | {s['scale']} | {s['variant']} | {s['metric']} | "
              f"{s['mean_diff']} | {s['p']} | {s['p_holm']} |")
        p()
    p("## 5. 结论(自动判断,供写作参考)")
    p()
    fp16_max = 0.0
    for ds in C.DATASETS:
        for sc in C.SCALES:
            seeds = sorted({r["seed"] for r in res if r["dataset"] == ds and r["scale"] == sc})
            d = []
            for sd in seeds:
                b = [r for r in res if r["dataset"] == ds and r["scale"] == sc
                     and r["precision"] == "fp32" and r["seed"] == sd]
                x = [r for r in res if r["dataset"] == ds and r["scale"] == sc
                     and r["precision"] == "fp16" and r["seed"] == sd]
                if b and x:
                    d.append(abs(f(x[0]["mAP50"]) - f(b[0]["mAP50"])))
            if d:
                fp16_max = max(fp16_max, max(d))
    p(f"- FP16 与 FP32 的最大绝对 ΔmAP50(所有格子、所有种子)= **{fp16_max:.4f}**"
      + ("(仍属免费)" if fp16_max <= 0.002 else "(注意:超过 0.002)"))
    p(f"- 精度差异能否被种子噪声解释:Holm 后显著 {n_yes} 个 → "
      + ("**仍无法解析**(与三种子结论一致)" if n_yes == 0 else "**有比较可解析**"))
    static_rel = []
    for ds in C.DATASETS:
        for sc in C.SCALES:
            seeds = sorted({r["seed"] for r in res if r["dataset"] == ds and r["scale"] == sc})
            d, bm = [], []
            for sd in seeds:
                b = [r for r in res if r["dataset"] == ds and r["scale"] == sc
                     and r["precision"] == "fp32" and r["seed"] == sd]
                x = [r for r in res if r["dataset"] == ds and r["scale"] == sc
                     and r["precision"] == "int8static" and r["seed"] == sd]
                if b and x:
                    d.append(f(x[0]["mAP50"]) - f(b[0]["mAP50"]))
                    bm.append(f(b[0]["mAP50"]))
            if d:
                static_rel.append(statistics.fmean(d) / statistics.fmean(bm) * 100)
    if static_rel:
        p(f"- static INT8 的相对 mAP50 代价区间:**{min(static_rel):+.2f}% … {max(static_rel):+.2f}%**")
    p()
    (OUT / "seed_extension_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"报告已生成:{OUT / 'seed_extension_report.md'}")
    print(f"  行数 {len(L)};Holm 显著 {n_yes} 个;未校正 p<0.05 {len(unc)} 个")


if __name__ == "__main__":
    main()
