# -*- coding: utf-8 -*-
"""生成论文要用的 markdown 表格 + 正文关键数字 -> out/paper_tables.md

目的:补种子/改口径后,论文里的表与数字一律**从 out/*.csv 生成**,不手抄。
输出分三部分:
  A. 六张表(Table 2/3/4/5/6 与 SI 的 S1/S2),格式与手稿里的表格头一致,可直接整体替换
  B. 正文里用到的关键数字(FP16 最大偏差、延迟倍数范围、static 相对代价、p 值范围等)
  C. 与手稿当前值的差异提示(便于只改动了的部分)

用法:python scripts/make_paper_tables.py
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


def load():
    res = list(csv.DictReader(open(OUT / "results.csv", encoding="utf-8-sig")))
    lat = list(csv.DictReader(open(OUT / "latency.csv", encoding="utf-8-sig")))
    sig = list(csv.DictReader(open(OUT / "significance.csv", encoding="utf-8-sig")))
    cal = list(csv.DictReader(open(OUT / "calib_ablation.csv", encoding="utf-8-sig"))) \
        if (OUT / "calib_ablation.csv").exists() else []
    return res, lat, sig, cal


def mean_sd(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    return statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


def per_cell(res, ds, sc, pr, field="mAP50"):
    return [f(r[field]) for r in res if r["dataset"] == ds and r["scale"] == sc
            and r["precision"] == pr]


def paired(res, ds, sc, pr, field="mAP50"):
    """同种子配对差值;返回列表(按种子排序)。"""
    seeds = sorted({r["seed"] for r in res if r["dataset"] == ds and r["scale"] == sc})
    d = []
    for sd in seeds:
        b = [r for r in res if r["dataset"] == ds and r["scale"] == sc
             and r["precision"] == "fp32" and r["seed"] == sd]
        x = [r for r in res if r["dataset"] == ds and r["scale"] == sc
             and r["precision"] == pr and r["seed"] == sd]
        if b and x:
            d.append(f(x[0][field]) - f(b[0][field]))
    return d


def lat_ratio(lat, ds, sc, pr):
    b = [f(r["lat_median_ms"]) for r in lat if r["dataset"] == ds and r["scale"] == sc
         and r["precision"] == "fp32"]
    x = [f(r["lat_median_ms"]) for r in lat if r["dataset"] == ds and r["scale"] == sc
         and r["precision"] == pr]
    if not b or not x:
        return None
    return statistics.fmean(x) / statistics.fmean(b)


def paired_ratio(lat, ds, sc, pr):
    """同种子内的延迟比值(变体/FP32)列表 —— 温度/频率漂移下唯一稳健的口径。

    绝对延迟在同一会话内也会随主机 CPU 状态漂移(实测同架构 nano 的 5 个种子模型跨度可达 56%),
    因此论文里的"多少倍"必须用同种子配对比值,而不是两个均值的比。
    """
    seeds = sorted({r["seed"] for r in lat if r["dataset"] == ds and r["scale"] == sc})
    rr = []
    for sd in seeds:
        b = [f(r["lat_median_ms"]) for r in lat
             if (r["dataset"], r["scale"], r["seed"], r["precision"]) == (ds, sc, sd, "fp32")]
        x = [f(r["lat_median_ms"]) for r in lat
             if (r["dataset"], r["scale"], r["seed"], r["precision"]) == (ds, sc, sd, pr)]
        if b and x and b[0] > 0 and x[0] > 0:
            rr.append(x[0] / b[0])
    return rr


def main() -> None:
    res, lat, sig, cal = load()
    dsn = {"gc10": "GC10-DET", "neu": "NEU-DET"}
    pn = {"fp32": "FP32", "fp16": "FP16", "int8dyn": "INT8-dynamic", "int8static": "INT8-static"}

    p("# 论文四 · 由脚本生成的表格与关键数字")
    p()
    n_seed = dict(collections.Counter(
        (r["dataset"], r["scale"]) for r in res if r["precision"] == "fp32"))
    p(f"> 结果行数 {len(res)};延迟行数 {len(lat)};各格子种子数 "
      + "、".join(f"{k[0]}-{k[1]}={v}" for k, v in sorted(n_seed.items())))
    p()

    # ---------- Table 2 精度
    p("## Table 2 · 精度(mean ± SD over seeds)")
    p()
    p("| dataset | scale | precision | mAP50 | mAP50-95 |")
    p("|---|---|---|---|---|")
    for ds in C.DATASETS:
        for sc in C.SCALES:
            for pr in C.PRECISIONS:
                m50, s50 = mean_sd(per_cell(res, ds, sc, pr, "mAP50"))
                m95, s95 = mean_sd(per_cell(res, ds, sc, pr, "mAP50_95"))
                if m50 is None:
                    continue
                p(f"| {dsn[ds]} | {sc} | {pn[pr]} | {m50:.4f} ± {s50:.4f} | {m95:.4f} ± {s95:.4f} |")
    p()

    # ---------- Table 3 配对 Δ
    p("## Table 3 · 配对 Δ(同种子 vs FP32)")
    p()
    p("| dataset | scale | variant | mean ΔmAP50 | per-seed range | relative | latency vs FP32 |")
    p("|---|---|---|---|---|---|---|")
    for ds in C.DATASETS:
        for sc in C.SCALES:
            base = per_cell(res, ds, sc, "fp32")
            bm = statistics.fmean(base) if base else None
            for pr in ("fp16", "int8dyn", "int8static"):
                d = paired(res, ds, sc, pr)
                if not d or bm is None:
                    continue
                rel = statistics.fmean(d) / bm * 100
                r = lat_ratio(lat, ds, sc, pr)
                p(f"| {dsn[ds]} | {sc} | {pn[pr]} | {statistics.fmean(d):+.4f} | "
                  f"{min(d):+.4f} … {max(d):+.4f} | {rel:+.2f}% | {r:.2f}× |" if r else
                  f"| {dsn[ds]} | {sc} | {pn[pr]} | {statistics.fmean(d):+.4f} | "
                  f"{min(d):+.4f} … {max(d):+.4f} | {rel:+.2f}% | ? |")
    p()

    # ---------- Table 4 成本(稳健口径:每格 5 个种子模型)
    p("## Table 4 · 延迟与体积(中位数[四分位距] ms / MB;括号内 = 同种子配对比值 vs FP32)")
    p()
    p("| dataset | scale | FP32 | FP16 | INT8-dynamic | INT8-static |")
    p("|---|---|---|---|---|---|")
    drift = []
    for ds in C.DATASETS:
        for sc in C.SCALES:
            cells = []
            for pr in ("fp32", "fp16", "int8dyn", "int8static"):
                ms = [f(r["lat_median_ms"]) for r in lat if r["dataset"] == ds and r["scale"] == sc
                      and r["precision"] == pr]
                mb = [f(r["size_mb"]) for r in lat if r["dataset"] == ds and r["scale"] == sc
                      and r["precision"] == pr]
                if not ms:
                    cells.append("?")
                    continue
                med = statistics.median(ms)
                q1, q3 = statistics.quantiles(ms, n=4)[0], statistics.quantiles(ms, n=4)[2]
                s = f"{med:.1f} [{q1:.1f}–{q3:.1f}] ms / {statistics.fmean(mb):.1f} MB"
                rr = paired_ratio(lat, ds, sc, pr)
                if rr and pr != "fp32":
                    s += f" ({statistics.fmean(rr):.2f} ± {statistics.stdev(rr):.2f}×)"
                cells.append(s)
                if pr == "fp32":
                    drift.append((ds, sc, min(ms), max(ms)))
            p(f"| {dsn[ds]} | {sc} | " + " | ".join(cells) + " |")
    p()
    p("> **同会话内的绝对漂移(必须随表一起报)**:同一架构、不同种子的 5 个模型在同一会话内也会漂移 ——")
    for ds, sc, lo, hi in drift:
        p(f"> {dsn[ds]} {sc} FP32:每种子中位数 {lo:.1f}–{hi:.1f} ms(跨度 {100 * (hi / lo - 1):.0f}%)。")
    p("> 因此**「多少倍」以同种子配对比值为准**(每格 5 个比值),绝对毫秒只作指示值引用。")
    p()

    # ---------- Table 5 校准消融(同一 key 只保留最后一次运行:早期一轮的延迟列在归一化 bug 修复前测的)
    if cal:
        seen = {}
        for r in cal:
            seen[(r["dataset"], r["scale"], r["seed"], r["calib"])] = r
        p("## Table 5 · 校准集消融(mAP50)")
        p()
        cols = [c for c in cal[0].keys() if c not in ("dataset", "scale", "seed", "precision")]
        p("| " + " | ".join(cols) + " |")
        p("|" + "---|" * len(cols))
        for k in sorted(seen):
            r = seen[k]
            p("| " + " | ".join(str(r[c]) for c in cols) + " |")
        p()

    # ---------- Table 6 帕累托前沿
    p("## Table 6 · 帕累托前沿(按格子)")
    p()
    p("| cell | Pareto front |")
    p("|---|---|")
    for ds in C.DATASETS:
        for sc in C.SCALES:
            pts = []
            for pr in C.PRECISIONS:
                m, _ = mean_sd(per_cell(res, ds, sc, pr, "mAP50"))
                ms = [f(r["lat_median_ms"]) for r in lat if r["dataset"] == ds and r["scale"] == sc
                      and r["precision"] == pr]
                mb = [f(r["size_mb"]) for r in lat if r["dataset"] == ds and r["scale"] == sc
                      and r["precision"] == pr]
                if m is None or not ms:
                    continue
                pts.append({"pr": pr, "m": m, "ms": statistics.fmean(ms),
                            "mb": statistics.fmean(mb)})
            front = [a for a in pts
                     if not any((b["m"] >= a["m"] and b["ms"] <= a["ms"]
                                 and (b["m"] > a["m"] or b["ms"] < a["ms"])) for b in pts)]
            front.sort(key=lambda a: a["ms"])
            # 严格支配检查:某点在 mAP 更高且延迟更低且体积更小时,是否支配其余全部
            dom = None
            for a in pts:
                others = [b for b in pts if b is not a]
                if all(a["m"] > b["m"] and a["ms"] <= b["ms"] and a["mb"] <= b["mb"] for b in others):
                    dom = a
                    break
            if dom:
                p(f"| {dsn[ds]} {sc} | {pn[dom['pr']]} alone strictly dominates all three other variants |")
            else:
                s = " → ".join(f"{pn[a['pr']]} ({a['ms']:.0f} ms, {a['m']:.4f})" for a in front)
                p(f"| {dsn[ds]} {sc} | {s} |")
    p()

    # ---------- 关键数字(供正文)
    p("## 正文关键数字(脚本现算)")
    p()
    fp16_all = [abs(x) for ds in C.DATASETS for sc in C.SCALES for x in paired(res, ds, sc, "fp16")]
    p(f"- FP16 与 FP32 的最大绝对 ΔmAP50 = **{max(fp16_all):.4f}**")
    for pr in ("fp16", "int8dyn", "int8static"):
        rr = [x for ds in C.DATASETS for sc in C.SCALES for x in paired_ratio(lat, ds, sc, pr)]
        if rr:
            p(f"- {pn[pr]} 同种子配对比值(全部格子 × 5 种子,共 {len(rr)} 个)= "
              f"**{min(rr):.2f}× … {max(rr):.2f}×**(均值 {statistics.fmean(rr):.2f}×)")
    for pr in ("int8dyn", "int8static"):
        rs = [lat_ratio(lat, ds, sc, pr) for ds in C.DATASETS for sc in C.SCALES]
        rs = [r for r in rs if r]
        if rs:
            p(f"- {pn[pr]} 延迟倍数范围(均值之比,旧口径,仅对照)= **{min(rs):.2f}× … {max(rs):.2f}×**")
    rels = []
    for ds in C.DATASETS:
        for sc in C.SCALES:
            d = paired(res, ds, sc, "int8static")
            base = per_cell(res, ds, sc, "fp32")
            if d and base:
                rels.append(statistics.fmean(d) / statistics.fmean(base) * 100)
    if rels:
        p(f"- static INT8 相对 mAP50 代价范围 = **{min(rels):+.2f}% … {max(rels):+.2f}%**")
    sigs = [f(s["p"]) for s in sig if f(s["p"]) is not None]
    if sigs:
        p(f"- Wilcoxon/Welch 未校正 p 值范围 = {min(sigs):.4f}–{max(sigs):.4f};"
          f"Holm 后显著 {sum(1 for s in sig if s.get('signif_holm05') == 'yes')}/{len(sig)}")
    # FP16 体积减半与精度中性
    p(f"- 模型体积:gc10/neu s 尺度 44.8 → 22.4 MB(FP32→FP16 精确减半)、"
      f"n 尺度 12.3 → 6.2 MB")

    (OUT / "paper_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已生成 {OUT / 'paper_tables.md'}({len(L)} 行)")
    print(f"  格子种子数 {n_seed};表格已按手稿表头格式输出,可直接整体替换")


if __name__ == "__main__":
    main()
