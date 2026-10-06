# -*- coding: utf-8 -*-
"""核验:论文草稿里的每个数字都必须能在原始产物里找到(允许四舍五入)。

做法:取草稿中所有表格行(| 开头)与含数字的正文行,抽出浮点数;
与"证据库"(out/*.csv、out/numbers_sheet.md、_log_fix2.log 的前沿输出)里的浮点数做
同精度四舍五入比对;找不对的任何数字都打印出来(附所在行),由人判断是笔误还是漏进清单。

用法: python scripts/verify_draft_numbers.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import ptq_common as C

DRAFT = C.PROJ / "_paper" / "manuscript_draft_20261005.md"
NUM = re.compile(r"-?\d+\.\d+")

# 统一减号:草稿用 U+2212(−),产物里是 ASCII '-'
DRAFT_TEXT = DRAFT.read_text(encoding="utf-8").replace("\u2212", "-")

# ---- 证据库:所有原始产物 + 清单 + 图前沿日志
evidence_text = []
for p in [C.OUT / "numbers_sheet.md", C.OUT / "paper_tables.md", C.OUT / "si_tables.md",
          C.OUT / "tables.md", C.OUT / "seed_extension_report.md",
          C.PROJ / "_log_fix2.log", C.PROJ / "_log_latency_all.log",
          C.PROJ / "_log_seed_extension_n.log",
          C.PROJ / "_log_full_matrix.log", C.PROJ / "_log_calib_ablation.log"]:
    if p.exists():
        evidence_text.append(p.read_text(encoding="utf-8", errors="replace"))
for name in ("results.csv", "latency.csv", "stats.csv", "significance.csv",
             "calib_ablation.csv", "export_sizes.csv"):
    p = C.OUT / name
    if p.exists():
        evidence_text.append(p.read_text(encoding="utf-8", errors="replace"))
for p in (C.OUT / "perclass").glob("*.json"):
    evidence_text.append(p.read_text(encoding="utf-8", errors="replace"))
for p in (C.OUT / "gt_gc10_val_base1_stem_eq640.json",):
    if p.exists():
        evidence_text.append(p.read_text(encoding="utf-8", errors="replace"))
corpus = "\n".join(evidence_text).replace("\u2212", "-")

corpus_vals = [float(x) for x in NUM.findall(corpus)]


def found(v: float, decimals: int) -> tuple:
    """(是否找到, 是否仅绝对值匹配) —— 后者用于正文里符号在语境中的情况(如 "losing 1.98%")。"""
    exact = False
    absok = False
    for c in corpus_vals:
        if round(c, decimals) == v:
            exact = True
            break
        if round(abs(c), decimals) == abs(v):
            absok = True
    return exact, (absok and not exact)


def main() -> None:
    text = DRAFT_TEXT.splitlines()
    print(f"草稿 {len(text)} 行;证据库数值 {len(corpus_vals)} 个\n")
    bad, warn, checked = [], [], 0
    for i, line in enumerate(text, 1):
        # 只看表格行与正文行(跳过代码块内的命令行)
        if line.strip().startswith("```") or line.strip().startswith("#"):
            continue
        if "±" not in line and "|" not in line and "ms" not in line and "MB" not in line \
                and "×" not in line and "%" not in line and "mAP" not in line:
            continue
        for tok in NUM.findall(line):
            v = float(tok)
            decimals = len(tok.split(".")[1])
            checked += 1
            exact, abs_only = found(v, decimals)
            if not exact and not abs_only:
                bad.append((i, tok, line.strip()[:120]))
            elif abs_only:
                warn.append((i, tok))
    print(f"核验数字 {checked} 个;未在证据库中找到的 {len(bad)} 个;仅绝对值匹配 {len(warn)} 个(符号在语境里)")
    for i, tok, ctx in bad:
        print(f"  行 {i:4d}  {tok:>10s}   {ctx}")
    if warn:
        print("  仅绝对值匹配(需人工确认符号合理):",
              ", ".join(f"行{i}:{t}" for i, t in warn[:12]))
    if not bad:
        print("  (全部可溯源 ✓)")
    print()
    print("说明:本检查是'存在性'校验,不含因果 —— 它保证草稿里的数字都来自产物,"
          "但不保证用在了正确的语境(语境需人工读)。")


if __name__ == "__main__":
    main()
