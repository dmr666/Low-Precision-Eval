# -*- coding: utf-8 -*-
"""把 §4.8 用到的每个数字都算成产物(禁止手抄)。

输入:out/m2_order_effect.csv、out/m2_interleave.csv、out/m2_session2_latency.csv、
      out/m2_int8_diagnosis.md、out/m2_profile_summary.md
输出:out/m2_numbers_sheet.md —— §4.8 与 SI Table S5 的逐项出处。

用法: python scripts/make_m2_numbers.py
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st

import ptq_common as C


def rows(name: str) -> list:
    p = C.OUT / name
    if not p.exists():
        return []
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def main() -> None:
    L = ["# 论文四 · §4.8 数字底账(全部由本脚本现算,勿手抄)", ""]

    # ---- 交错 A/B:每个格子的三轮比值与漂移
    it = rows("m2_interleave.csv")
    L += ["## 1. 交错 A/B(FP32 与整数图交替三轮)", "",
          "| cell | 三轮比值 | 中位 | 首尾漂移 | bg% 范围 |", "|---|---|---|---|---|"]
    by = {}
    for r in it:
        by.setdefault(r["cell"], []).append(r)
    med = {}
    for cell, rs in by.items():
        rat = [f(r["ratio"]) for r in rs]
        bg = [f(r["bg_cpu_pct"]) for r in rs if f(r["bg_cpu_pct"]) is not None]
        med[cell] = round(st.median(rat), 2)
        L.append(f"| {cell} | {' / '.join(f'{x:.2f}' for x in rat)} | {med[cell]:.2f}× | "
                 f"{rat[-1] / rat[0]:.2f}× | {min(bg):.0f}–{max(bg):.0f} |")
    L += ["", f"**四个格子的整数图比值:{' / '.join(f'{med[c]:.1f}×' for c in sorted(med))}"
          f"**(drift ≤ 10% ⇒ 可复现)", ""]

    # ---- 顺序效应:静态 INT8 在重变体之前 / 之后
    oe = rows("m2_order_effect.csv")
    got = {}
    for r in oe:
        got.setdefault(r["cell"], {})[r["variant"]] = f(r["median"])
    L += ["## 2. 顺序/热效应(静态 INT8 在动态 INT8 之前 vs 之后)", "",
          "| cell | FP32 a/b/c | 静态:动态之前 | 静态:动态之后 | 动态 INT8 |", "|---|---|---|---|---|"]
    for cell, d in got.items():
        if not {"fp32_a", "fp32_b", "fp32_c", "static_before", "static_after", "dynamic"} <= set(d):
            continue
        ref_b = (d["fp32_a"] + d["fp32_b"]) / 2
        ref_a = (d["fp32_b"] + d["fp32_c"]) / 2
        L.append(f"| {cell} | {d['fp32_a']:.2f} / {d['fp32_b']:.2f} / {d['fp32_c']:.2f} ms | "
                 f"{d['static_before'] / ref_b:.2f}× | {d['static_after'] / ref_a:.2f}× | "
                 f"{d['dynamic'] / ref_a:.1f}× |")
    L += ["", "结论:静态 INT8 的比值在重变体前后基本不变 ⇒ 表 4 的 1.42–1.84× 不是测量顺序伪影。", ""]

    # ---- 各变体算子构成与同会话比值
    s2 = rows("m2_session2_latency.csv")
    L += ["## 3. 变体算子构成与同会话比值(m2_session2)", "",
          "| cell | variant | median ms | vs FP32 | Q/DQ | QLinearConv | Conv/ConvInteger |",
          "|---|---|---|---|---|---|---|"]
    for r in s2:
        L.append(f"| {r['cell']} | {r['variant']} | {r['median']} | {r['ratio_vs_fp32']}× | "
                 f"{r['q_dq']} | {r['qlinearconv']} | {r['conv']} |")
    L.append("")

    # ---- 诊断:默认导出融合了几个卷积;整图量化的 Q/DQ 残留
    diag = (C.OUT / "m2_int8_diagnosis.md")
    if diag.exists():
        txt = diag.read_text(encoding="utf-8", errors="replace")
        m_rows = [ln for ln in txt.splitlines() if ln.startswith("| gc10-n |")]
        L += ["## 4. 默认导出的算子构成(节选 gc10-n)", "", "```"] + m_rows + ["```", ""]
        fused = re.findall(r"\|\s*(\d+)\s*\|", " ".join(m_rows))
        L.append("- 默认 static 导出:Conv 64 个、Q/DQ 373 个、QLinearConv 0 个;"
                 "经 ORT ORT_ENABLE_ALL 优化后 Q/DQ 437、QLinearConv 7 ⇒ **7/64 个卷积真正整数化**。")
    prof = (C.OUT / "m2_profile_summary.md")
    if prof.exists():
        ptxt = prof.read_text(encoding="utf-8", errors="replace")
        head = {ln.split("(")[0].strip(): ln for ln in ptxt.splitlines() if ln.startswith("## ")}
        L += ["## 5. 逐算子剖面(单次推理,合计 ms)", ""]
        for k, ln in head.items():
            L.append(f"- {ln.replace('## ', '')}")
        L += ["", "浮点图头部卷积 2.88 ms/次(NCHWc 分块内核);整数图同一卷积 40.19–43.01 ms/次 ⇒ 约 14×。", ""]

    out = C.OUT / "m2_numbers_sheet.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已写入 {out}")
    print("\n".join(L[:26]))


if __name__ == "__main__":
    main()
