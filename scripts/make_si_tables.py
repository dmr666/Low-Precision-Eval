# -*- coding: utf-8 -*-
"""论文四:由原始产物生成 Supplementary 表 S1 / S2 的数字(禁手抄)。

S1 = 配对 ΔAP50 按目标尺寸分档(eq-640 面积,5 种子),GC10-DET 与 NEU-DET 都给;
     并给出每个格子 AP50_small 的跨种子离散(用于说明 Δ 落在种子噪声内)。
S2 = GC10-DET 小尺度 seed 42 的逐类 mAP50-95(FP32 vs INT8-static)。

数据来源:out/results.csv(分档列 AP50_s/m/l,COCOeval eq640 口径)、out/perclass/*.json。
输出:out/si_tables.md
用法:python scripts/make_si_tables.py
"""
from __future__ import annotations

import csv
import json
import statistics as st

import ptq_common as C

SEEDS = ["42", "7", "2024", "13", "99"]
VARIANTS = ["fp16", "int8dyn", "int8static"]
BUCKETS = [("AP50_s", "small"), ("AP50_m", "medium"), ("AP50_l", "large")]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    rows = list(csv.DictReader(open(C.OUT / "results.csv", encoding="utf-8-sig")))
    A = {(r["dataset"], r["scale"], r["seed"], r["precision"]): r for r in rows}

    L: list[str] = []
    L.append("# 论文四 · Supplementary 表(脚本生成,勿手改)")
    L.append("")
    L.append(f"> 来源 `out/results.csv`({len(rows)} 行);分档口径 pycocotools COCOeval,面积折算到 640 输入空间(eq640)。")
    L.append("")

    # ---------- Table S1 ----------
    L.append("## Table S1 · 配对 ΔAP50 按尺寸分档(5 种子均值)")
    L.append("")
    L.append("| dataset | scale | variant | ΔAP50_small | ΔAP50_medium | ΔAP50_large |")
    L.append("|---|---|---|---|---|---|")
    detail: dict[tuple, list] = {}
    for ds in ("gc10", "neu"):
        for sc in ("n", "s"):
            for p in ("int8static", "int8dyn", "fp16"):
                cells, ok = [], True
                for col, name in BUCKETS:
                    d = []
                    for sd in SEEDS:
                        v = f(A[(ds, sc, sd, p)][col])
                        b = f(A[(ds, sc, sd, "fp32")][col])
                        if v is None or b is None or v < 0 or b < 0:
                            ok = False
                            break
                        d.append(v - b)
                    if not ok:
                        cells.append("n/a(该数据集无此档目标)")
                        continue
                    detail[(ds, sc, p, name)] = d
                    cells.append(f"{st.fmean(d):+.4f}")
                L.append(f"| {ds.upper()} | {sc} | {p} | " + " | ".join(cells) + " |")
    L.append("")

    # ---------- 分档 AP 的跨种子离散(说明 Δ 是否落在噪声内) ----------
    L.append("## Table S1b · 每格 AP50_small 的跨种子离散(SD)")
    L.append("")
    L.append("| dataset | scale | precision | mean AP50_small | SD across seeds |")
    L.append("|---|---|---|---|---|")
    disp = {}
    for ds in ("gc10", "neu"):
        for sc in ("n", "s"):
            for p in ("fp32", "fp16", "int8dyn", "int8static"):
                v = [f(A[(ds, sc, sd, p)]["AP50_s"]) for sd in SEEDS]
                if any(x is None or x < 0 for x in v):
                    L.append(f"| {ds.upper()} | {sc} | {p} | n/a | n/a |")
                    continue
                disp[(ds, sc, p)] = (st.fmean(v), st.stdev(v))
                L.append(f"| {ds.upper()} | {sc} | {p} | {st.fmean(v):.4f} | {st.stdev(v):.4f} |")
    L.append("")

    # ---------- 小目标损失 vs 大目标损失的比值 ----------
    L.append("## Table S1c · 小目标损失 / 大目标损失(仅 >1 表示小目标更吃亏;GC10-DET)")
    L.append("")
    L.append("| scale | variant | Δsmall | Δlarge | 比值 small/large |")
    L.append("|---|---|---|---|---|")
    for sc in ("n", "s"):
        for p in ("int8static", "int8dyn", "fp16"):
            k = (("gc10"), sc, p)
            ds_ = detail.get(k + ("small",))
            dl_ = detail.get(k + ("large",))
            if not ds_ or not dl_:
                continue
            a, b = st.fmean(ds_), st.fmean(dl_)
            r = f"{a / b:.2f}x" if b not in (0.0,) else "n/a"
            L.append(f"| {sc} | {p} | {a:+.4f} | {b:+.4f} | {r} |")
    L.append("")

    # ---------- NEU 的 medium vs large ----------
    L.append("## Table S1d · NEU-DET 的 medium vs large(无小目标,故只比 m/l)")
    L.append("")
    L.append("| scale | variant | Δmedium | Δlarge |")
    L.append("|---|---|---|---|")
    for sc in ("n", "s"):
        for p in ("int8static", "int8dyn"):
            dm = st.fmean(detail[("neu", sc, p, "medium")])
            dl = st.fmean(detail[("neu", sc, p, "large")])
            L.append(f"| {sc} | {p} | {dm:+.4f} | {dl:+.4f} |")
    L.append("")

    # ---------- Table S2:逐类(seed 42, gc10 s) ----------
    def perclass(prec: str) -> dict:
        p = C.OUT / "perclass" / f"gc10_s_{prec}_seed42.json"
        return json.loads(p.read_text(encoding="utf-8"))

    fp32, q8 = perclass("fp32"), perclass("int8static")
    names = fp32["names"]
    L.append("## Table S2 · GC10-DET s,seed 42 逐类 mAP50-95")
    L.append("")
    L.append("| class | " + " | ".join(names) + " |")
    L.append("|---" * (len(names) + 1) + "|")
    v1 = [f(x) for x in fp32["map50_95_per_class"]]
    v2 = [f(x) for x in q8["map50_95_per_class"]]
    L.append("| FP32 | " + " | ".join(f"{x:.4f}" for x in v1) + " |")
    L.append("| INT8-static | " + " | ".join(f"{x:.4f}" for x in v2) + " |")
    L.append("| Δ | " + " | ".join(f"{y - x:+.4f}" for x, y in zip(v1, v2)) + " |")
    L.append("")

    (C.OUT / "si_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n已写入 {C.OUT / 'si_tables.md'}")


if __name__ == "__main__":
    main()
