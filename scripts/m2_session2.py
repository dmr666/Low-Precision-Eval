# -*- coding: utf-8 -*-
"""M2 诊断会话 2(有界、边测边落盘)。

上一轮的教训:
  - CSV 只在全部结束后写一次 ⇒ 中途中断(或我手动停)就把所有计时结果丢掉;
  - 原版 YOLOv8n 对照要联网下载权重,可能长时间挂住;
  - per-channel 与 per-tensor 的 QOperator 结果几乎一样(631 vs 642 ms),不必重复两次。
本脚本因此:逐行 append + flush(断电也只丢最后一行)、不联网、只测需要的变体。

每格测:FP32(同会话参照)、V1 QOperator per-tensor、V2 QOperator per-channel、
        V3 仅 MatMul/Gemm 动态、V4 整图量化(QOperator,不排除任何节点)。
V1-V3 复用上轮导出(onnx_m2/),V4 现场导出。

用法: M2_SESSION=<id> M2_RUNS=20 python scripts/m2_session2.py
"""
from __future__ import annotations

import collections
import csv
import os
import statistics
import time
from pathlib import Path

import numpy as np

import ptq_common as C
from host_load import exclusivity_check, sample_load
from m2_fully_quantized import quant_full
from m2_int8_variants import CELLS, M2_DIR, opcounts, quant_static, quant_dynamic_mm
from ptq_calib_ablation import FilesReader, split_images
from ptq_latency import letterbox_bgr

IMGSZ = 640
THREADS = 4
RUNS = int(os.environ.get("M2_RUNS", "20"))
WARMUP = 5
OUT_CSV = C.OUT / "m2_session2_latency.csv"
COLS = ["cell", "variant", "onnx", "q_dq", "qlinearconv", "conv", "median", "p95", "min", "max",
        "ratio_vs_fp32", "bg_cpu_pct", "runs", "session"]


def timed(onnx: Path, dataset: str) -> dict:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = letterbox_bgr(split_images(dataset, "val")[0], IMGSZ).transpose(2, 0, 1)[None].astype(np.float32)
    for _ in range(WARMUP):
        sess.run(None, {inp.name: x})
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        outs = sess.run(None, {inp.name: x})
        for o in outs:
            float(np.asarray(o).sum())
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    n = len(ts)
    return {"median": round(statistics.median(ts), 2),
            "p95": round(ts[min(n - 1, int(round(0.95 * (n - 1))))], 2),
            "min": round(ts[0], 2), "max": round(ts[-1], 2)}


def main() -> None:
    sess_id = os.environ.get("M2_SESSION", "")
    if not sess_id:
        raise SystemExit("必须 export M2_SESSION=<会话 id>")
    thr = float(os.environ.get("M2_LOAD_THRESHOLD", "15"))
    ok, d = exclusivity_check(thr, seconds=6.0)
    print(f"空闲门禁:平均 {d.get('total_pct')}% / 峰值 {d.get('peak_pct')}%(阈值 {thr}%)⇒ "
          f"{'干净 ✓' if ok else '不干净 ✗'}")
    for t in d.get("top", [])[:4]:
        print(f"    {t['cpu_pct_of_host']:>5.1f}%  {t['name']} (pid {t['pid']})")
    if not ok and os.environ.get("M2_ALLOW_BUSY") != "1":
        raise SystemExit("主机不干净,拒绝开始计时")

    f = open(OUT_CSV, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader(); f.flush()
    print(f"=== 会话 {sess_id},每格 {RUNS} 次,结果逐行写入 {OUT_CSV.name} ===")

    for ds, sc, seed in CELLS:
        fp32 = C.onnx_path(ds, sc, seed, "fp32")
        fp16 = C.onnx_path(ds, sc, seed, "fp16")
        if not fp32.exists():
            print(f"  [skip] {ds}-{sc}:缺 FP32")
            continue
        reader = FilesReader(split_images(ds, "val"), IMGSZ)
        plan = [
            ("fp32", fp32),
            ("fp16", fp16),
            ("V1_qop_pertensor", M2_DIR / f"{ds}_{sc}_seed{seed}_V1_qop.onnx"),
            ("V2_qop_perchannel", M2_DIR / f"{ds}_{sc}_seed{seed}_V2_qop_pc.onnx"),
            ("V3_dyn_MatMulGemm", M2_DIR / f"{ds}_{sc}_seed{seed}_V3_dynmm.onnx"),
            ("V4_fullquant", M2_DIR / f"{ds}_{sc}_seed{seed}_V4_fullq.onnx"),
        ]
        # 需要现场导出的:V4(其余应已在 onnx_m2/)
        if not plan[-1][1].exists():
            print(f"  [{ds}-{sc}] 导出 V4 整图量化 …", flush=True)
            try:
                quant_full(fp32, plan[-1][1], reader)
            except Exception as e:
                print(f"  [skip] V4 导出失败:{type(e).__name__}: {e}")
                plan = plan[:-1]
        base = None
        for name, path in plan:
            if not path.exists():
                print(f"  [skip] {name}:缺 {path.name}")
                continue
            h = opcounts(path)
            r = {"cell": f"{ds}-{sc}", "variant": name, "onnx": path.name,
                 "q_dq": h["QuantizeLinear"] + h["DequantizeLinear"],
                 "qlinearconv": h["QLinearConv"], "conv": h["Conv"] + h["ConvInteger"],
                 "runs": RUNS, "session": sess_id}
            r.update(timed(path, ds))
            if name == "fp32":
                base = r["median"]
            r["ratio_vs_fp32"] = round(r["median"] / base, 3) if base else ""
            r["bg_cpu_pct"] = sample_load(seconds=1.5).get("total_pct")
            w.writerow(r); f.flush()          # 逐行落盘:中断也只丢最后一行
            print(f"  {ds}-{sc} {name:20s} median {r['median']:>8.2f} ms  "
                  f"ratio {r['ratio_vs_fp32']}×  (Q/DQ {r['q_dq']}, QLinearConv {r['qlinearconv']}, "
                  f"Conv {r['conv']})", flush=True)
    f.close()
    print(f"\n完成。逐行结果:{OUT_CSV}({RUNS} 次/配置,会话 {sess_id})")


if __name__ == "__main__":
    main()
