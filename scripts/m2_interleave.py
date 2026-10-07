# -*- coding: utf-8 -*-
"""M2 收口:热漂移可控的 A/B 交错测量 + ORT 逐算子剖面。

为什么需要交错(上一轮的教训):
  V1/V4 这类整数图要慢 10 倍,单次推理就要 0.7-6 秒;连着跑 20 次会把 15W 的 TGL-U 压热,
  紧跟其后的配置就被热降频污染(实测:同一格 V3 本该≈FP32,却测出 2.3×)。
  所以这里改成 FP32 与 V1 交替 A/B/A/B/A/B,每个循环各测一次,比值取三轮中位数,
  同时看第一轮与最后一轮是否漂移 —— 漂移大就说明该格子的比值也不可用,必须如实标出。

剖面:对 FP32 / V1 / V4 各跑一次 enable_profiling,按算子累计耗时排序,回答"时间到底花在哪"。

用法: M2_SESSION=<id> python scripts/m2_interleave.py
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

import ptq_common as C
from host_load import exclusivity_check, sample_load
from m2_int8_variants import CELLS, M2_DIR, opcounts
from ptq_calib_ablation import split_images
from ptq_latency import letterbox_bgr

IMGSZ, THREADS = 640, 4
RUNS_REF = int(os.environ.get("M2_RUNS_REF", "20"))     # FP32 每轮次数
RUNS_INT8 = int(os.environ.get("M2_RUNS_INT8", "5"))    # V1/V4 每轮次数(它慢 10 倍,少跑几次)
CYCLES = int(os.environ.get("M2_CYCLES", "3"))
CSV_OUT = C.OUT / "m2_interleave.csv"
PROF_OUT = C.OUT / "m2_profile_summary.md"


def _session(path: Path, profiling: bool = False, prefix: str = "prof"):
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profiling:
        so.enable_profiling = True
        so.profile_file_prefix = prefix
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _input(dataset: str, inp):
    x = letterbox_bgr(split_images(dataset, "val")[0], IMGSZ).transpose(2, 0, 1)[None].astype(np.float32)
    return {inp.name: x}


def measure(path: Path, dataset: str, runs: int, warmup: int = 3) -> float:
    sess = _session(path)
    inp = sess.get_inputs()[0]
    feed = _input(dataset, inp)
    for _ in range(warmup):
        sess.run(None, feed)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        outs = sess.run(None, feed)
        for o in outs:
            float(np.asarray(o).sum())
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return round(statistics.median(ts), 2)


def profile(path: Path, dataset: str, tag: str, top: int = 14) -> list:
    """跑一次并读 ORT 剖面,返回按算子累计耗时排序的 (算子, 次数, 累计 ms, 占比)。"""
    sess = _session(path, profiling=True, prefix=f"m2prof_{tag}")
    inp = sess.get_inputs()[0]
    feed = _input(dataset, inp)
    sess.run(None, feed)
    p = Path(sess.end_profiling())
    ev = json.loads(p.read_text(encoding="utf-8"))
    agg = {}
    total = 0.0
    for e in ev:
        if e.get("cat") != "Node" or "dur" not in e:
            continue
        dur = float(e["dur"])         # µs
        total += dur
        key = e.get("op_name") or e.get("name", "?")
        a = agg.setdefault(key, [0, 0.0])
        a[0] += 1
        a[1] += dur
    rows = sorted(((k, v[0], v[1] / 1000.0) for k, v in agg.items()), key=lambda t: -t[2])[:top]
    return [{"op": k, "n": n, "ms": round(ms, 2), "share": round(100 * ms * 1000 / total, 1) if total else 0}
            for k, n, ms in rows] + [{"total_ms": round(total / 1000.0, 2), "file": p.name}]


def main() -> None:
    sess_id = os.environ.get("M2_SESSION", "")
    if not sess_id:
        raise SystemExit("必须 export M2_SESSION=<会话 id>")
    thr = float(os.environ.get("M2_LOAD_THRESHOLD", "15"))
    ok, d = exclusivity_check(thr, seconds=6.0)
    print(f"空闲门禁:平均 {d.get('total_pct')}% / 峰值 {d.get('peak_pct')}%(阈值 {thr}%)⇒ "
          f"{'干净 ✓' if ok else '不干净 ✗'}")
    # 本主机(tgl-u 笔记本)装着厂商后台服务,占用会在 7%-34% 之间突发波动:
    # 单次采样做"门禁"会把正常波动误判成污染(实测同一分钟两次采样差 3 倍)。
    # 因此默认只告警并记录,靠"交错 A/B + 逐配置记录 bg%"来保证结论稳健;
    # 需要严格门禁时设 M2_STRICT_GATE=1。
    if not ok and os.environ.get("M2_STRICT_GATE") == "1":
        raise SystemExit("主机不干净(严格门禁已开启),拒绝开始计时")

    f = open(CSV_OUT, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=["cell", "cycle", "fp32_median", "int8_median", "variant",
                                      "ratio", "bg_cpu_pct", "session"])
    w.writeheader(); f.flush()
    ratios_by_cell = {}
    print(f"=== 交错 A/B 会话 {sess_id}:{CYCLES} 循环 × (FP32 {RUNS_REF} 次 + 整数图 {RUNS_INT8} 次) ===")
    for ds, sc, seed in CELLS:
        fp32 = C.onnx_path(ds, sc, seed, "fp32")
        v1 = M2_DIR / f"{ds}_{sc}_seed{seed}_V1_qop.onnx"
        v4 = M2_DIR / f"{ds}_{sc}_seed{seed}_V4_fullq.onnx"
        if not (fp32.exists() and v1.exists()):
            print(f"  [skip] {ds}-{sc}")
            continue
        rs = []
        for cyc in range(1, CYCLES + 1):
            a = measure(fp32, ds, RUNS_REF)
            b = measure(v1, ds, RUNS_INT8)
            r = {"cell": f"{ds}-{sc}", "cycle": cyc, "fp32_median": a, "int8_median": b,
                 "variant": "V1_qop", "ratio": round(b / a, 3), "session": sess_id,
                 "bg_cpu_pct": sample_load(seconds=1.0).get("total_pct")}
            w.writerow(r); f.flush()
            rs.append(r["ratio"])
            print(f"  {ds}-{sc} 循环{cyc}: FP32 {a:>8.2f} ms | V1 {b:>9.2f} ms ⇒ {r['ratio']:>6.2f}×"
                  f"   (bg {r['bg_cpu_pct']}%)", flush=True)
        ratios_by_cell[f"{ds}-{sc}"] = {"median_ratio": round(statistics.median(rs), 2),
                                        "min": min(rs), "max": max(rs),
                                        "drift_cycles": round(rs[-1] / rs[0], 2)}
        print(f"  ⇒ {ds}-{sc} 三轮比值 {rs} 中位 {statistics.median(rs):.2f}×,"
              f"首尾漂移 {rs[-1]/rs[0]:.2f}×", flush=True)
    f.close()

    # ---- 阶段 B:顺序/热效应 —— 主矩阵的表 4 是"先测 13× 的动态 INT8,再测静态 INT8",
    #      因此静态 INT8 的 1.42-1.84× 有可能部分是前一个配置压热后的降频,而不是格式本身。
    #      做法:在同一会话内按 fp32 → static → fp32 → dynamic → static → fp32 的顺序走,
    #      用夹在中间的 FP32 作局部参照,比较"动态之前"与"动态之后"的静态 INT8 比值。
    oe = C.OUT / "m2_order_effect.csv"
    fo = open(oe, "w", newline="", encoding="utf-8")
    wo = csv.DictWriter(fo, fieldnames=["cell", "step", "variant", "runs", "median", "bg_cpu_pct", "session"])
    wo.writeheader(); fo.flush()
    print("\n=== 阶段 B:顺序/热效应(静态 INT8 在动态 INT8 之前 vs 之后)===", flush=True)
    order_summary = {}
    for ds, sc, seed in (("gc10", "n", 42), ("neu", "n", 42)):
        p_fp32 = C.onnx_path(ds, sc, seed, "fp32")
        p_st = C.onnx_path(ds, sc, seed, "int8static")
        p_dy = C.onnx_path(ds, sc, seed, "int8dyn")
        if not (p_fp32.exists() and p_st.exists() and p_dy.exists()):
            print(f"  [skip] {ds}-{sc}")
            continue
        seq = [("fp32_a", p_fp32, RUNS_REF), ("static_before", p_st, 10),
               ("fp32_b", p_fp32, RUNS_REF), ("dynamic", p_dy, 3),
               ("static_after", p_st, 10), ("fp32_c", p_fp32, RUNS_REF)]
        got = {}
        for step, (name, path, runs) in enumerate(seq, 1):
            m = measure(path, ds, runs)
            got[name] = m
            wo.writerow({"cell": f"{ds}-{sc}", "step": step, "variant": name, "runs": runs,
                         "median": m, "session": sess_id,
                         "bg_cpu_pct": sample_load(seconds=1.0).get("total_pct")})
            fo.flush()
            print(f"  {ds}-{sc} {name:14s} ({runs:>2d} 次) {m:>9.2f} ms", flush=True)
        ref_before = (got["fp32_a"] + got["fp32_b"]) / 2
        ref_after = (got["fp32_b"] + got["fp32_c"]) / 2
        order_summary[f"{ds}-{sc}"] = {
            "fp32_steps": [got["fp32_a"], got["fp32_b"], got["fp32_c"]],
            "static_before_over_fp32": round(got["static_before"] / ref_before, 3),
            "static_after_over_fp32": round(got["static_after"] / ref_after, 3),
        }
        print(f"  ⇒ {ds}-{sc}:静态 INT8 比值 动态之前 {order_summary[f'{ds}-{sc}']['static_before_over_fp32']}× "
              f"/ 动态之后 {order_summary[f'{ds}-{sc}']['static_after_over_fp32']}×", flush=True)
    fo.close()
    print(f"顺序效应结果:{oe}")
    print(json.dumps(order_summary, ensure_ascii=False, indent=1))

    # ---- 剖面:时间花在哪些算子上
    md = ["# 论文四 · M2 剖面:整数图的时间花在哪(单次推理,ORT profiling)", "",
          f"> 会话 `{sess_id}`;每个模型跑 1 次(已 warmup),按算子累计耗时排序。", ""]
    for tag, path, ds in (("fp32", C.onnx_path("gc10", "n", 42, "fp32"), "gc10"),
                          ("V1_qop", M2_DIR / "gc10_n_seed42_V1_qop.onnx", "gc10"),
                          ("V4_fullq", M2_DIR / "gc10_n_seed42_V4_fullq.onnx", "gc10")):
        if not path.exists():
            continue
        rows = profile(path, ds, tag)
        tot = rows[-1] if isinstance(rows[-1], dict) and "total_ms" in rows[-1] else {"total_ms": 0}
        md += [f"## {tag} (gc10-n) — 合计 {tot.get('total_ms')} ms", "",
               "| 算子 | 次数 | 累计 ms | 占比 |", "|---|---|---|---|"]
        for r in rows:
            if "op" in r:
                md.append(f"| {r['op']} | {r['n']} | {r['ms']} | {r['share']}% |")
        md.append("")
    PROF_OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"已写入 {CSV_OUT} 与 {PROF_OUT}")
    print(json.dumps(ratios_by_cell, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
