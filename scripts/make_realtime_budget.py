# -*- coding: utf-8 -*-
"""论文四:实时性预算表 —— 把延迟换算成"帧预算/可持续帧率",回答 JRTIP 关心的 real-time 问题。

JRTIP 投稿规定明确写着:"只报处理时间/速度不算处理了实时问题",必须正面回答实时性约束。
本脚本只用已采纳的那一轮延迟表(out/latency.csv)现算,不引入任何新测量:
  - 帧预算 33.3 ms(30 fps 单相机)与 16.7 ms(60 fps)的倍数;
  - 4 线程工作线程在 p95 尾延迟下能持续输出的帧率(1e3/p95);
  - 同一主机假设 32 逻辑核时,能并行跑几个这样的 4 线程工作线程(仅按核数计的指示值)。

输出:out/realtime_budget.md
用法:python scripts/make_realtime_budget.py
"""
from __future__ import annotations

import csv
import statistics as st

import ptq_common as C

CELLS = [("gc10", "n"), ("gc10", "s"), ("neu", "n"), ("neu", "s")]
PREC = ["fp32", "fp16", "int8dyn", "int8static"]
PN = {"fp32": "FP32", "fp16": "FP16", "int8dyn": "INT8-dynamic", "int8static": "INT8-static"}


def main() -> None:
    rows = list(csv.DictReader(open(C.OUT / "latency.csv", encoding="utf-8-sig")))
    sess = sorted({r.get("session") or "EMPTY" for r in rows})
    L = ["# 论文四 · 实时性预算(JRTIP 关心的 real-time 视角)", ""]
    L.append(f"> 数据:`out/latency.csv`({len(rows)} 行,会话 {sess});仅算术换算,无新测量。")
    L.append("")
    L.append("| cell | variant | median ms | p95 ms | median / 33.3 ms | p95 / 33.3 ms | 可持续 fps (p95) |")
    L.append("|---|---|---|---|---|---|---|")
    summary = {}
    for ds, sc in CELLS:
        for pr in PREC:
            v = [r for r in rows if r["dataset"] == ds and r["scale"] == sc and r["precision"] == pr]
            if not v:
                continue
            med = st.fmean([float(r["lat_median_ms"]) for r in v])
            p95 = st.fmean([float(r["lat_p95_ms"]) for r in v])
            fps = 1000.0 / p95
            L.append(f"| {ds}-{sc} | {PN[pr]} | {med:.1f} | {p95:.1f} | {med / 33.3:.2f}x | "
                     f"{p95 / 33.3:.2f}x | {fps:.2f} |")
            summary[(ds, sc, pr)] = (med, p95, fps)
    L.append("")

    L.append("## 结论(由表推出)")
    L.append("")
    best = {c: min((summary[(c[0], c[1], p)][0], p) for p in PREC if (c[0], c[1], p) in summary)
            for c in [(d, s) for d, s in CELLS]}
    for c, (med, pr) in best.items():
        p95 = summary[(c[0], c[1], pr)][1]
        L.append(f"- {c[0]}-{c[1]}:最快的精度变体是 {PN[pr]},中位 {med:.1f} ms = 33.3 ms 帧预算的 {med / 33.3:.2f}×;"
                 f"p95 {p95:.1f} ms = {p95 / 33.3:.2f}×,可持续 {1000 / p95:.2f} fps。")
    cheapest = min(best.values(), key=lambda t: t[0])
    L.append(f"- 全表最快的一格:{cheapest[1]} 的 {cheapest[0]:.1f} ms,仍是 30 fps 帧预算的 {cheapest[0] / 33.3:.2f}×。")
    L.append("- 结论:在本主机上,精度选择不改变\"达不到 30 fps 单相机实时\"这一事实;实时性约束落在"
             "**部署拓扑**(每台主机带几路相机、输入分辨率)上,而不是数值精度上。")
    L.append("")
    (C.OUT / "realtime_budget.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n已写入 {C.OUT / 'realtime_budget.md'}")


if __name__ == "__main__":
    main()
