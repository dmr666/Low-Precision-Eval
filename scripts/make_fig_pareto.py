# -*- coding: utf-8 -*-
"""论文图 1:精度–延迟–体积 帕累托图(数据全部来自 out/stats.csv)。

四个面板:gc10 n / gc10 s / neu n / neu s。
x = ONNX Runtime CPU 中位延迟(对数轴),y = val mAP50,点面积 ∝ 模型体积,虚线 = 非支配前沿(Pareto)。
同时导出 PNG(300 dpi)与 PDF(矢量,投稿用)。

用法: python scripts/make_fig_pareto.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ptq_common as C

FIGDIR = C.PROJ / "figs"
LABEL = {"fp32": "FP32", "fp16": "FP16", "int8dyn": "INT8-dynamic", "int8static": "INT8-static"}
COLOR = {"fp32": "#1f3864", "fp16": "#2e7d32", "int8dyn": "#c62828", "int8static": "#ef6c00"}
MARK = {"fp32": "o", "fp16": "s", "int8dyn": "^", "int8static": "D"}


def load_rows():
    p = C.OUT / "stats.csv"
    if not p.exists():
        raise SystemExit(f"缺 {p};先跑 ptq_stats.py")
    rows = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        def mid(s):
            """'0.6352 ± 0.0117' -> (mean, sd);单一值 -> (v, 0)。"""
            s = (s or "").strip()
            if not s or s == "-":
                return None, None
            if "±" in s:
                a, b = s.split("±")
                return float(a.strip()), float(b.strip())
            return float(s), 0.0
        m50, m50sd = mid(r.get("mAP50"))
        lat, latsd = mid(r.get("lat_ms"))
        size, _ = mid(r.get("size_mb"))
        if m50 is None or lat is None:
            continue
        rows.append(dict(dataset=r["dataset"], scale=r["scale"], precision=r["precision"],
                         mAP50=m50, mAP50_sd=m50sd or 0.0,
                         lat=lat, lat_sd=latsd or 0.0, size=size or 1.0))
    return rows


def pareto(points):
    """非支配点:(延迟更小 且 精度更高)不被任何点支配。"""
    out = []
    for p in points:
        dominated = any((q["lat"] <= p["lat"] and q["mAP50"] >= p["mAP50"]
                         and (q["lat"] < p["lat"] or q["mAP50"] > p["mAP50"]))
                        for q in points if q is not p)
        if not dominated:
            out.append(p)
    return sorted(out, key=lambda x: x["lat"])


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), dpi=150)
    order = [("gc10", "n"), ("gc10", "s"), ("neu", "n"), ("neu", "s")]
    for ax, (ds, sc) in zip(axes.ravel(), order):
        pts = [r for r in rows if r["dataset"] == ds and r["scale"] == sc]
        if not pts:
            continue
        # 体积映射到散点面积
        sizes = [r["size"] for r in pts]
        smin, smax = min(sizes), max(sizes)
        for r in pts:
            scale = 0.6 + 0.4 * ((r["size"] - smin) / (smax - smin)) if smax > smin else 1.0
            ax.scatter(r["lat"], r["mAP50"], s=180 * scale,
                       color=COLOR.get(r["precision"], "0.4"),
                       marker=MARK.get(r["precision"], "o"),
                       edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(LABEL.get(r["precision"], r["precision"]),
                        (r["lat"], r["mAP50"]), textcoords="offset points",
                        xytext=(7, -3), fontsize=7.5, color=COLOR.get(r["precision"], "0.3"))
        front = pareto(pts)
        if len(front) > 1:
            ax.plot([p["lat"] for p in front], [p["mAP50"] for p in front],
                    ls="--", lw=0.9, color="0.5", zorder=2,
                    label="Pareto front")
        ax.set_xscale("log")
        ax.set_xlabel("CPU latency, median (ms, ONNX Runtime, 4 threads, 640)", fontsize=8)
        ax.set_ylabel("val mAP50", fontsize=8)
        ax.set_title(f"{ds.upper()} · YOLOv8{sc}", fontsize=9.5)
        ax.grid(alpha=0.25, which="both", ls=":")
        ax.tick_params(labelsize=7.5)
    fig.suptitle("Accuracy–latency–size trade-offs of low-precision deployment "
                 "(steel surface defect detection, 3 seeds)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png, pdf = FIGDIR / "fig_pareto.png", FIGDIR / "fig_pareto.pdf"
    fig.savefig(png); fig.savefig(pdf)
    print(f"图已生成:{png}\n         {pdf}")
    # 文字版前沿,便于写正文
    for ds, sc in order:
        pts = [r for r in rows if r["dataset"] == ds and r["scale"] == sc]
        front = pareto(pts)
        print(f"  {ds} {sc} 前沿:" +
              " → ".join(f"{LABEL.get(p['precision'])} ({p['lat']:.0f} ms, {p['mAP50']:.4f})"
                         for p in front))


if __name__ == "__main__":
    main()
