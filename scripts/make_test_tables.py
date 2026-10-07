# -*- coding: utf-8 -*-
"""SI Table S6:留出 test 分区的精度(审稿意见 M3)。

从 out/results_test.csv 现算:逐格逐变体的 mean ± SD,以及相对**同种子** FP32 的配对 Δ,
并给出 val 分区同口径的配对 Δ 以便对照(静态 INT8 恰在 val 上校准,val 是对它最有利的设定)。

用法: python scripts/make_test_tables.py
"""
from __future__ import annotations

import csv
import statistics as st
from collections import Counter
from pathlib import Path

import ptq_common as C

CELLS = [("gc10", "n"), ("gc10", "s"), ("neu", "n"), ("neu", "s")]
ORDER = ["fp32", "fp16", "int8static", "int8dyn"]


def load(p: Path) -> dict:
    if not p.exists():
        return {}
    return {(r["dataset"], r["scale"], r["seed"], r["precision"]): r
            for r in csv.DictReader(open(p, encoding="utf-8-sig"))}


def main() -> None:
    tst, val = load(C.OUT / "results_test.csv"), load(C.OUT / "results.csv")
    if not tst:
        raise SystemExit("没有 out/results_test.csv")
    precs = [p for p in ORDER if any(k[3] == p for k in tst)]
    n_cells = len({(k[0], k[1]) for k in tst})
    L = ["# 论文四 · SI Table S6 生成器输出(test 分区)", "",
         f"> 数据:`out/results_test.csv`,{len(tst)} 个配置(共 {n_cells} 格 × {len(precs)} 变体 × 种子);",
         f"> 已完成的变体:{dict(Counter(k[3] for k in tst))}", "",
         "**Table S6.** Held-out test split: mAP50 by precision variant (mean ± SD over the seeds present) and the",
         "paired difference against the same-seed FP32 model, with the validation-split counterpart for comparison.",
         "",
         "| cell | variant | seeds | mAP50 (test) | mAP50-95 (test) | paired ΔmAP50 (test) | paired ΔmAP50 (val) |",
         "|---|---|---|---|---|---|---|"]
    for ds, sc in CELLS:
        seeds = sorted({k[2] for k in tst if k[0] == ds and k[1] == sc}, key=lambda s: int(s))
        for prec in precs:
            have = [s for s in seeds if (ds, sc, s, prec) in tst]
            if not have:
                continue
            m50 = [float(tst[(ds, sc, s, prec)]["mAP50"]) for s in have]
            m95 = [float(tst[(ds, sc, s, prec)]["mAP50_95"]) for s in have]
            cell = f"{ds.upper() if ds == 'neu' else ds}-DET {sc}"
            if prec == "fp32":
                L.append(f"| {cell} | FP32 | {len(have)} | {st.fmean(m50):.4f} ± {st.stdev(m50):.4f} | "
                         f"{st.fmean(m95):.4f} ± {st.stdev(m95):.4f} | – | – |")
                continue
            pairable = [s for s in have if (ds, sc, s, "fp32") in tst]
            dt = [float(tst[(ds, sc, s, prec)]["mAP50"]) - float(tst[(ds, sc, s, "fp32")]["mAP50"])
                  for s in pairable]
            dv = [float(val[(ds, sc, s, prec)]["mAP50"]) - float(val[(ds, sc, s, "fp32")]["mAP50"])
                  for s in pairable if (ds, sc, s, prec) in val and (ds, sc, s, "fp32") in val]
            ttxt = f"{st.fmean(dt):+.4f} [{min(dt):+.3f}..{max(dt):+.3f}]" if dt else "n/a"
            vtxt = f"{st.fmean(dv):+.4f} [{min(dv):+.3f}..{max(dv):+.3f}]" if dv else "n/a"
            L.append(f"| {cell} | {prec} | {len(have)} | {st.fmean(m50):.4f} ± {st.stdev(m50):.4f} | "
                     f"{st.fmean(m95):.4f} ± {st.stdev(m95):.4f} | {ttxt} | {vtxt} |")
        L.append("")
    # 稿件正文引用到的具体值(配对 Δ 均值与相对百分比)也落盘,保证"每个数字都有出处"
    quoted = ["", "## 正文引用的具体值(逐格逐变体)", "",
              "| cell | variant | paired Δ(mAP50) test | relative % | paired ΔmAP50 val |", "|---|---|---|---|---|"]
    for ds, sc in CELLS:
        seeds = sorted({k[2] for k in tst if k[0] == ds and k[1] == sc}, key=lambda s: int(s))
        for prec in precs:
            if prec == "fp32":
                continue
            pairable = [s for s in seeds if (ds, sc, s, prec) in tst and (ds, sc, s, "fp32") in tst]
            if not pairable:
                continue
            dt = [float(tst[(ds, sc, s, prec)]["mAP50"]) - float(tst[(ds, sc, s, "fp32")]["mAP50"])
                  for s in pairable]
            b = st.fmean([float(tst[(ds, sc, s, "fp32")]["mAP50"]) for s in pairable])
            dv = [float(val[(ds, sc, s, prec)]["mAP50"]) - float(val[(ds, sc, s, "fp32")]["mAP50"])
                  for s in pairable if (ds, sc, s, prec) in val and (ds, sc, s, "fp32") in val]
            rel = f"{100 * st.fmean(dt) / b:+.2f}%"
            quoted.append(f"| {ds}-{sc} | {prec} | {st.fmean(dt):+.4f} | {rel} | "
                          f"{st.fmean(dv):+.4f} |" if dv else
                          f"| {ds}-{sc} | {prec} | {st.fmean(dt):+.4f} | {rel} | n/a |")
    L += quoted
    out = C.OUT / "si_table_s7.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已写入 {out}")
    print("\n".join(L[6:]))


if __name__ == "__main__":
    main()
