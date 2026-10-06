# -*- coding: utf-8 -*-
"""论文图 1:精度–延迟–体积 帕累托图(数据全部来自 out/stats.csv)。

四个面板:gc10 n / gc10 s / neu n / neu s。
x = ONNX Runtime CPU 中位延迟(对数轴),y = val mAP50,点面积 ∝ 模型体积,虚线 = 非支配前沿(Pareto)。
同时导出 PNG(600 dpi 备选)与 PDF(矢量,投稿主用)。

用法: python scripts/make_fig_pareto.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

import ptq_common as C

FIGDIR = C.PROJ / "figs"
LABEL = {"fp32": "FP32", "fp16": "FP16", "int8dyn": "INT8-dynamic", "int8static": "INT8-static"}
COLOR = {"fp32": "#1f3864", "fp16": "#2e7d32", "int8dyn": "#c62828", "int8static": "#ef6c00"}
MARK = {"fp32": "o", "fp16": "s", "int8dyn": "^", "int8static": "D"}
# 显式刻度:每格只有 ~1.5 in 宽,只放 3 个刻度(100/500/2000),否则数字必然互相压叠;
# 数据点最远到 ~3200 ms,超出刻度也无妨(matplotlib 会自动扩展显示范围)
XTICKS = [100, 500, 2000]


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
    # 最终以 174 mm(整页宽)排版,figsize 就取 6.8 in,缩放比≈1.0 → 图内字号即最终字号(Springer 要求 8–12 pt)
    # 单栏排版:栏宽 85 mm ≈ 3.35 in,figsize 取 3.4 in → 缩放≈1.0,图内字号即最终字号
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.0), dpi=150)
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
            ax.scatter([], [], color=COLOR.get(r["precision"], "0.4"),
                       marker=MARK.get(r["precision"], "o"), edgecolor="white",
                       linewidth=0.8, label=LABEL.get(r["precision"], r["precision"]))
        front = pareto(pts)
        if len(front) > 1:
            ax.plot([p["lat"] for p in front], [p["mAP50"] for p in front],
                    ls="--", lw=0.9, color="0.5", zorder=2,
                    label="Pareto front")
        ax.set_xscale("log")
        ax.set_xticks(XTICKS)
        ax.set_xticklabels([f"{v:g}" for v in XTICKS], fontsize=7)
        ax.xaxis.set_minor_locator(FixedLocator([]))   # 关掉次刻度标签
        ax.set_xlabel("median latency (ms)", fontsize=8)   # 缩短:面板仅 1.5 in 宽,长标签会越过栏边
        ax.set_ylabel("val mAP50", fontsize=8)
        ax.set_title(f"{ds.upper()} · YOLOv8{sc}", fontsize=8.5)
        ax.grid(alpha=0.25, which="major", ls=":")
        if (ds, sc) == ("gc10", "n"):
            ax.legend(fontsize=6.5, frameon=False, loc="lower right", handletextpad=0.2)
        ax.tick_params(labelsize=7.5)
    # suptitle 去掉:同样的信息写在图题里,省一行整页宽的高度
    fig.tight_layout()
    png, pdf = FIGDIR / "fig_pareto.png", FIGDIR / "fig_pareto.pdf"
    # PNG 备选栅格图:按 Springer 组合图要求 ≥600 dpi(正式投稿用矢量 PDF)
    fig.savefig(png, dpi=600); fig.savefig(pdf)
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
