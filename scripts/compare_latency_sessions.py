# -*- coding: utf-8 -*-
"""论文四:对比多轮延迟会话(判定"绝对毫秒是否已是可复现数字"),结果写进 out/latency_session_compare.md。

判定标准(事先写死,避免看着数字找理由):
1. 同一架构的 5 个种子模型,在同一轮会话内 FP32 中位数的相对跨度(max/min-1):
   - ≤10% → 该轮可视为热/频率稳态,绝对毫秒可用;
   - >10% → 该轮仍有漂移,绝对毫秒只能作指示值。
2. 两轮的"同种子配对比值"(变体/FP32)必须一致(差 ≤0.5×),否则两轮环境有实质差异,不得混用。
3. 干扰迹象:每格 p95/中位数 的最大值 —— 明显 >1.5 说明那一格测量期间有后台打扰,该轮不可用作"可复现数字"。

用法:python scripts/compare_latency_sessions.py
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

import ptq_common as C

SEEDS = ["42", "7", "2024", "13", "99"]
PREC = ["fp32", "fp16", "int8dyn", "int8static"]
CELLS = [("gc10", "n"), ("gc10", "s"), ("neu", "n"), ("neu", "s")]


def load(p: Path):
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    med = {}
    spike = {}
    for r in rows:
        k = (r["dataset"], r["scale"], r["seed"], r["precision"])
        med[k] = float(r["lat_median_ms"])
        if r.get("lat_p95_ms") and float(r["lat_median_ms"]) > 0:
            spike[k] = float(r["lat_p95_ms"]) / float(r["lat_median_ms"])
    sid = sorted({(r.get("session") or "EMPTY") for r in rows})
    return {"session": sid[0] if len(sid) == 1 else "/".join(sid), "n": len(rows),
            "med": med, "spike": spike}


def main() -> None:
    tabs: list[tuple[str, dict]] = []
    seen = set()
    for p in sorted(C.OUT.glob("latency_session_*.csv")) + [C.OUT / "latency.csv"]:
        if not p.exists():
            continue
        d = load(p)
        key = (d["session"], d["n"])
        if key in seen:
            continue
        seen.add(key)
        tabs.append((p.stem if p.name != "latency.csv" else f"current({d['session']})", d))

    L = ["# 论文四 · 延迟会话对比(判定采纳哪一轮 / 绝对毫秒是否可用)", ""]
    L.append("| 会话 | 文件 | 行数 | " + " | ".join(f"{d}-{s} FP32 跨度" for d, s in CELLS) + " | 最差格子 |")
    L.append("|---" * (len(CELLS) + 4) + "|")
    spread = {}
    for name, d in tabs:
        vals, worst = [], 0.0
        for ds, sc in CELLS:
            v = [d["med"][(ds, sc, sd, "fp32")] for sd in SEEDS if (ds, sc, sd, "fp32") in d["med"]]
            if len(v) < 2:
                vals.append("n/a")
                continue
            r = max(v) / min(v) - 1
            spread[(name, ds, sc)] = r
            worst = max(worst, r)
            vals.append(f"{min(v):.1f}-{max(v):.1f} ms ({r * 100:.0f}%)")
        L.append(f"| {name} | `{d['session']}` | {d['n']} | " + " | ".join(vals) + f" | {worst * 100:.0f}% |")
    L.append("")

    L.append("## 干扰迹象:每格 p95/中位数 的最大值(>1.5 说明该格测量期间有后台打扰)")
    L.append("")
    L.append("| 会话 | " + " | ".join(f"{d}-{s}" for d, s in CELLS) + " |")
    L.append("|---" * (len(CELLS) + 1) + "|")
    spike_max = {}
    for name, d in tabs:
        cells = []
        for ds, sc in CELLS:
            v = [x for k, x in d["spike"].items() if k[0] == ds and k[1] == sc]
            m = max(v) if v else float("nan")
            spike_max[(name, ds, sc)] = m
            cells.append(f"{m:.2f}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 同种子配对比值(变体/FP32,均值 ± SD over 5 seeds)")
    L.append("")
    L.append("| cell | variant | " + " | ".join(n for n, _ in tabs) + " |")
    L.append("|---" * (2 + len(tabs)) + "|")
    ratio = {}
    for ds, sc in CELLS:
        for pr in PREC[1:]:
            cells = []
            for name, d in tabs:
                rr = []
                for sd in SEEDS:
                    b = d["med"].get((ds, sc, sd, "fp32"))
                    x = d["med"].get((ds, sc, sd, pr))
                    if b and x:
                        rr.append(x / b)
                if rr:
                    ratio[(name, ds, sc, pr)] = st.fmean(rr)
                    cells.append(f"{st.fmean(rr):.2f} ± {st.stdev(rr):.2f}x")
                else:
                    cells.append("n/a")
            L.append(f"| {ds}-{sc} | {pr} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 判定")
    L.append("")
    for name, d in tabs:
        w = max((v for (n, ds, sc), v in spread.items() if n == name and v is not None), default=None)
        sp = max((v for (n, ds, sc), v in spike_max.items() if n == name), default=float("nan"))
        if w is None:
            L.append(f"- {name}:数据不足")
        else:
            L.append(f"- {name}:最差格子跨度 {w * 100:.0f}%;最差 p95/中位数 {sp:.2f}")
    if len(tabs) >= 2:
        names = [n for n, _ in tabs]
        a, b = names[0], names[-1]
        diffs = [abs(ratio[(a, ds, sc, pr)] - ratio[(b, ds, sc, pr)])
                 for ds, sc in CELLS for pr in PREC[1:]
                 if (a, ds, sc, pr) in ratio and (b, ds, sc, pr) in ratio]
        mx = max(diffs) if diffs else None
        L.append(f"- 两轮配对比值的最大差异:{mx:.2f}x" if mx is not None else "- 无法比较两轮")
        L.append("- 采纳规则:预热轮跨度 ≤10% 且两轮比值差 ≤0.5x 且无 p95/中位数 >1.5 的格子 → 采纳预热轮;"
                 "否则保留旧轮、双轮留档并在论文中披露。")
    (C.OUT / "latency_session_compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n已写入 {C.OUT / 'latency_session_compare.md'}")


if __name__ == "__main__":
    main()
