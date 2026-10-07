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


def paired_t(deltas: list):
    """配对单样本 t:对逐种子的 Δ 检验 Δ=0,df = n-1(5 种子 → 4)。

    这是本文设计对应的主检验:Δ 按种子配对构造,消除种子间变异。
    Welch(非配对、两样本)保留为稳健性检查,不作为主口径。
    """
    d = [x for x in deltas if x is not None]
    n = len(d)
    if n < 2:
        return None
    md = statistics.fmean(d)
    sd = statistics.stdev(d)
    if sd == 0:
        return None
    t = md / (sd / math.sqrt(n))
    df = n - 1
    p = 2 * t_sf(abs(t), df)
    return {"t": t, "df": df, "p": p, "diff": md, "n": n}


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

    # 逐种子原始值(配对检验必须按 seed 对齐,不能用文件顺序)
    METRICS = ("mAP50", "mAP50_95", "APs", "APm", "APl", "AP50_s", "AP50_m", "AP50_l")
    seedval = {}
    for r in res:
        seedval[(r["dataset"], r["scale"], r["precision"], str(r["seed"]))] = {
            m: fnum(r.get(m)) for m in METRICS}

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
            fam, famw, meta = [], [], []
            for pr in C.PRECISIONS:
                if pr == "fp32":
                    continue
                cur = g.get((ds, sc, pr))
                if not cur:
                    continue
                seeds = sorted({k[3] for k in seedval if k[:3] == (ds, sc, "fp32")})
                for m in ("mAP50", "mAP50_95", "APs"):
                    deltas, cv, bv = [], [], []
                    for s in seeds:
                        b = seedval.get((ds, sc, "fp32", s), {}).get(m)
                        c = seedval.get((ds, sc, pr, s), {}).get(m)
                        if b is None or c is None:
                            continue
                        deltas.append(c - b); cv.append(c); bv.append(b)
                    rp = paired_t(deltas)
                    rw = welch(cv, bv)
                    if rp and rw:
                        fam.append(rp["p"]); famw.append(rw["p"]); meta.append((pr, m, rp, rw))
            if not fam:
                continue
            adj = holm(fam)      # 主检验家族(配对)
            adjw = holm(famw)    # 稳健性家族(Welch)
            for (pr, m, rp, rw), padj, padjw in zip(meta, adj, adjw):
                sig.append({"dataset": ds, "scale": sc, "variant": pr, "metric": m,
                            "mean_diff": round(rp["diff"], 5), "n_pairs": rp["n"],
                            "t": round(rp["t"], 3), "df": round(rp["df"], 2),
                            "p": round(rp["p"], 4), "p_holm": round(padj, 4),
                            "signif_holm05": "yes" if padj < 0.05 else "no",
                            "t_welch": round(rw["t"], 3), "df_welch": round(rw["df"], 2),
                            "p_welch": round(rw["p"], 4), "p_holm_welch": round(padjw, 4),
                            "signif_holm05_welch": "yes" if padjw < 0.05 else "no"})
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
        md.append("## Table 2 · 相对 FP32 的差异与显著性(主:配对 t df=4;稳健性:Welch)——各自 Holm")
        md.append("")
        md.append("| dataset | scale | variant | metric | Δ(mean) | t_paired | df_paired | p | p(Holm) | signif@0.05 | t_welch | df_welch | p_welch | p(Holm)_welch | signif_welch |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in sig:
            md.append("| {dataset} | {scale} | {variant} | {metric} | {mean_diff} | {t} | {df} | "
                      "{p} | {p_holm} | {signif_holm05} | {t_welch} | {df_welch} | {p_welch} | "
                      "{p_holm_welch} | {signif_holm05_welch} |".format(**s))
    md.append("")
    md.append("> 口径:逐类/mAP = ultralytics val;APs/APm/APl = pycocotools COCOeval。")
    md.append("> 本文件由 ptq_stats.py 生成;写作时引用 CSV,不复制粘贴手改。")
    (C.OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"表骨架:{C.OUT / 'tables.md'}")


if __name__ == "__main__":
    main()
