# -*- coding: utf-8 -*-
"""跨种子统计:mean ± SD、Welch t(对 FP32)+ Holm 校正,并生成论文表骨架。

所有数字都从 out/*.csv 现算 —— 论文里禁止手抄。
统计口径与论文三一致:多种子、逐类离散度、显著性;

用法: python ptq_stats.py
"""
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import ptq_common as C


def t_sf(t: float, df: float) -> float:
    """t 分布上尾概率;有 scipy 用 scipy,没有则正态近似(会在输出里标注)。"""
    try:
        from scipy import stats
        return float(stats.t.sf(t, df))
    except Exception:
        return 0.5 * math.erfc(t / math.sqrt(2))


def welch(a: list, b: list):
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return None
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = 2 * t_sf(abs(t), df)
    return {"t": t, "df": df, "p": p, "diff": ma - mb}


def holm(pvals: list) -> list:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def read_csv(p: Path) -> list:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except Exception:
        return None


def msd(vals: list) -> str:
    v = [x for x in vals if x is not None]
    if not v:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.4f}"
    return f"{statistics.fmean(v):.4f} ± {statistics.stdev(v):.4f}"


def main() -> None:
    res = read_csv(C.OUT / "results.csv")
    lat = read_csv(C.OUT / "latency.csv")
    if not res:
        raise SystemExit(f"没有结果文件:{C.OUT / 'results.csv'}")

    # 延迟查表
    latmap = {(r["dataset"], r["scale"], r["precision"], r["seed"]): fnum(r["lat_median_ms"])
              for r in lat}
    sizemap = {(r["dataset"], r["scale"], r["precision"], r["seed"]): fnum(r["size_mb"])
               for r in lat}

    g = defaultdict(lambda: defaultdict(list))
    for r in res:
        key = (r["dataset"], r["scale"], r["precision"])
        for m in ("mAP50", "mAP50_95", "APs", "APm", "APl", "AP50_s", "AP50_m", "AP50_l"):
            if m in r and r[m] not in ("", None):
                g[key][m].append(fnum(r[m]))
        latv = latmap.get((r["dataset"], r["scale"], r["precision"], r["seed"]))
        if latv is not None:
            g[key]["lat"].append(latv)
        sz = sizemap.get((r["dataset"], r["scale"], r["precision"], r["seed"]))
        if sz is not None:
            g[key]["size"].append(sz)

    # ---- 汇总表
    rows = []
    for (ds, sc, pr), d in sorted(g.items()):
        rows.append({
            "dataset": ds, "scale": sc, "precision": pr,
            "n_seeds": len(d.get("mAP50", [])),
            "mAP50": msd(d.get("mAP50", [])),
            "mAP50_95": msd(d.get("mAP50_95", [])),
            "APs": msd(d.get("APs", [])), "APm": msd(d.get("APm", [])),
            "APl": msd(d.get("APl", [])),
            "AP50_s": msd(d.get("AP50_s", [])),
            "lat_ms": msd(d.get("lat", [])),
            "size_mb": msd(d.get("size", [])),
        })
    out = C.OUT / "stats.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"汇总:{out}({len(rows)} 行)")

    # ---- 显著性:每个 (dataset, scale) 内,各变体 vs fp32
    sig = []
    for ds in sorted({k[0] for k in g}):
        for sc in sorted({k[1] for k in g if k[0] == ds}):
            base = g.get((ds, sc, "fp32"))
            if not base:
                continue
            fam, meta = [], []
            for pr in C.PRECISIONS:
                if pr == "fp32":
                    continue
                cur = g.get((ds, sc, pr))
                if not cur:
                    continue
                for m in ("mAP50", "mAP50_95", "APs"):
                    r = welch(cur.get(m, []), base.get(m, []))
                    if r:
                        fam.append(r["p"]); meta.append((pr, m, r))
            if not fam:
                continue
            adj = holm(fam)
            for (pr, m, r), padj in zip(meta, adj):
                sig.append({"dataset": ds, "scale": sc, "variant": pr, "metric": m,
                            "mean_diff": round(r["diff"], 5),
                            "t": round(r["t"], 3), "df": round(r["df"], 2),
                            "p": round(r["p"], 4), "p_holm": round(padj, 4),
                            "signif_holm05": "yes" if padj < 0.05 else "no"})
    sigp = C.OUT / "significance.csv"
    if sig:
        with open(sigp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sig[0]))
            w.writeheader(); w.writerows(sig)
        print(f"显著性:{sigp}({len(sig)} 行)")
    else:
        print("显著性:种子数不足(<2)或数据缺失,未生成")

    # ---- 论文表骨架(markdown,数字全部来自上面的 CSV)
    md = ["# 论文四 · 自动生成的表骨架(数字来自 out/*.csv,勿手抄)", ""]
    md.append("## Table 1 · 精度对比(跨种子 mean ± SD)")
    md.append("")
    md.append("| dataset | scale | precision | seeds | mAP50 | mAP50-95 | APs | APm | APl | latency (ms) | size (MB) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append("| {dataset} | {scale} | {precision} | {n_seeds} | {mAP50} | {mAP50_95} | "
                  "{APs} | {APm} | {APl} | {lat_ms} | {size_mb} |".format(**r))
    md.append("")
    if sig:
        md.append("## Table 2 · 相对 FP32 的差异与显著性(Welch t + Holm)")
        md.append("")
        md.append("| dataset | scale | variant | metric | Δ(mean) | t | df | p | p(Holm) | signif@0.05 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in sig:
            md.append("| {dataset} | {scale} | {variant} | {metric} | {mean_diff} | {t} | {df} | "
                      "{p} | {p_holm} | {signif_holm05} |".format(**s))
    md.append("")
    md.append("> 口径:逐类/mAP = ultralytics val;APs/APm/APl = pycocotools COCOeval。")
    md.append("> 本文件由 ptq_stats.py 生成;写作时引用 CSV,不复制粘贴手改。")
    (C.OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"表骨架:{C.OUT / 'tables.md'}")


if __name__ == "__main__":
    main()
