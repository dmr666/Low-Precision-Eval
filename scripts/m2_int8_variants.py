# -*- coding: utf-8 -*-
"""M2 交叉验证:把 INT8 的"导出口径"换成若干已知正确的做法,再在同一次独占会话里测延迟。

背景(out/m2_int8_diagnosis.md 的诊断):
  - 框架默认导出的 static INT8 是 QDQ 格式,373 个 Q/DQ 节点;ORT 用 ORT_ENABLE_ALL 优化后
    Q/DQ 反而变成 437,只有 7/64 个 Conv 被融合成 QLinearConv → 绝大多数卷积仍在浮点,且两侧多出
    量化/反量化开销 ⇒ "静态 INT8 反而慢 1.42-1.84x" 很可能是导出口径伪影。
  - 框架默认导出的 dynamic INT8 把 64 个 Conv 全变成 ConvInteger(动态反量化权重),
    这是 ORT 动态量化在 CNN 上的已知病态用法 ⇒ "动态 INT8 慢 13-15x" 同理。

本脚本为 4 个格子 × seed42 构造三种"修好口径"的变体,并与同会话的 FP32/FP16 对照:
  V1 = 静态 INT8,QOperator 格式(整数算子 QLinearConv),per-tensor
  V2 = 静态 INT8,QOperator,per-channel
  V3 = 动态 INT8,但只量化 MatMul/Gemm(ORT 推荐用法;CNN 上应≈无变化)
另做 best-effort 对照:原版 YOLOv8n(无自定义模块)按框架默认 QDQ 路径导出,看融合是否同样失败。

严格口径:所有延迟来自**一段不间断、独占 CPU 的会话**,每行带 session id;
产出 out/m2_variants_latency.csv / out/m2_variants_report.md(仅供诊断,不替换主矩阵)。

用法: M2_SESSION=sess_xxx python scripts/m2_int8_variants.py
"""
from __future__ import annotations

import collections
import csv
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import ptq_common as C
from host_load import exclusivity_check, sample_load
from ptq_calib_ablation import FilesReader, pick_even, split_images
from ptq_latency import letterbox_bgr

CELLS = [("gc10", "n", 42), ("gc10", "s", 42), ("neu", "n", 42), ("neu", "s", 42)]
M2_DIR = C.PROJ / "onnx_m2"
M2_DIR.mkdir(parents=True, exist_ok=True)
IMGSZ, RUNS, WARMUP, THREADS = 640, 50, 10, 4


def opcounts(p: Path) -> collections.Counter:
    import onnx
    m = onnx.load(str(p), load_external_data=False)
    return collections.Counter(n.op_type for n in m.graph.node)


def quant_static(fp32: Path, dest: Path, reader, fmt: str, per_channel: bool) -> Path:
    """复刻框架的节点选择(只量化 Conv/Gemm/MatMul),只改量化格式。"""
    import onnx
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    if dest.exists() and dest.stat().st_size > 1000:
        print(f"    [cache] {dest.name}")          # 同参数可复用:量化对这些输入是确定性的
        return dest
    tmp = Path(tempfile.gettempdir()) / "m2q"
    tmp.mkdir(parents=True, exist_ok=True)
    src, out = tmp / "in.onnx", tmp / "out.onnx"
    shutil.copyfile(fp32, src)
    graph = onnx.load(str(src)).graph
    exclude = [n.name for n in graph.node if n.op_type not in {"Conv", "Gemm", "MatMul"}]
    if hasattr(reader, "rewind"):
        reader.rewind()          # reader 是一次性的:不复位第二次调用会 "No data is collected".
    quantize_static(str(src), str(out), reader, nodes_to_exclude=exclude,
                    quant_format=(QuantFormat.QOperator if fmt == "qop" else QuantFormat.QDQ),
                    per_channel=per_channel, weight_type=QuantType.QInt8)
    shutil.copyfile(out, dest)
    return dest


def quant_dynamic_mm(fp32: Path, dest: Path) -> Path:
    """动态量化,但只量化 MatMul/Gemm(ORT 文档推荐;对纯卷积网络应≈无损无益)。"""
    from onnxruntime.quantization import quantize_dynamic
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"    [cache] {dest.name}")
        return dest
    tmp = Path(tempfile.gettempdir()) / "m2q"
    tmp.mkdir(parents=True, exist_ok=True)
    src, out = tmp / "din.onnx", tmp / "dout.onnx"
    shutil.copyfile(fp32, src)
    quantize_dynamic(str(src), str(out), op_types_to_quantize=["MatMul", "Gemm"])
    shutil.copyfile(out, dest)
    return dest


def timed(onnx: Path, dataset: str, imgsz: int = IMGSZ) -> dict:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = letterbox_bgr(split_images(dataset, "val")[0], imgsz).transpose(2, 0, 1)[None].astype(np.float32)
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


def stock_control(sess_id: str, rows: list) -> None:
    """原版 YOLOv8n(baseline,无自定义模块)按框架默认 QDQ 路径导出,看融合是否同样失败。"""
    try:
        from ultralytics import YOLO
        pt = Path(tempfile.gettempdir()) / "yolov8n_stock.pt"
        if not pt.exists():
            import urllib.request
            urllib.request.urlretrieve(
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt", str(pt))
        m = YOLO(str(pt))
        fp32 = M2_DIR / "stock_yolov8n_fp32.onnx"
        m.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True)
        exported = Path(str(pt).replace(".pt", ".onnx"))
        shutil.copyfile(exported, fp32)
        print(f"  [stock] FP32 导出 OK:{fp32.name}")
        rows.append({"cell": "stock-yolov8n", "variant": "fp32", "onnx": fp32.name,
                     "q_dq": "", "qlinearconv": "", "conv": "", "session": sess_id,
                     "bg_cpu_pct": sample_load(seconds=1.5).get("total_pct"), **timed(fp32, "gc10")})
    except Exception as e:      # 网络/导出失败不阻塞其余诊断
        print(f"  [stock] 跳过:{type(e).__name__}: {e}")


def main() -> None:
    sess_id = os.environ.get("M2_SESSION", "")
    if not sess_id:
        raise SystemExit("必须 export M2_SESSION=<会话 id>:延迟只能来自一段独占会话")
    thr = float(os.environ.get("M2_LOAD_THRESHOLD", "15"))
    ok, d = exclusivity_check(thr, seconds=6.0)
    print(f"主机占用门禁:平均 {d.get('total_pct')}% / 峰值 {d.get('peak_pct')}%"
          f"({d.get('cpu_count')} 逻辑核,阈值 {thr}%)⇒ {'干净 ✓' if ok else '不干净 ✗'}")
    for t in d.get("top", [])[:4]:
        print(f"    {t['cpu_pct_of_host']:>5.1f}%  {t['name']} (pid {t['pid']})")
    if not ok and os.environ.get("M2_ALLOW_BUSY") != "1":
        raise SystemExit("主机不干净,拒绝开始计时;清理后台后重跑(或 M2_ALLOW_BUSY=1 强制,但数据不得对外引用)")
    print(f"=== M2 变体交叉验证,会话 {sess_id} ===")
    rows: list[dict] = []
    for ds, sc, seed in CELLS:
        fp32 = C.onnx_path(ds, sc, seed, "fp32")
        fp16 = C.onnx_path(ds, sc, seed, "fp16")
        if not fp32.exists():
            print(f"  [skip] {ds}-{sc}:缺 FP32")
            continue
        val_files = split_images(ds, "val")
        reader = FilesReader(val_files, IMGSZ)
        jobs: list[tuple[str, Path]] = [("fp32", fp32), ("fp16", fp16)]
        builders = [
            ("V1_static_QOperator_pertensor",
             lambda: quant_static(fp32, M2_DIR / f"{ds}_{sc}_seed{seed}_V1_qop.onnx", reader, "qop", False)),
            ("V2_static_QOperator_perchannel",
             lambda: quant_static(fp32, M2_DIR / f"{ds}_{sc}_seed{seed}_V2_qop_pc.onnx", reader, "qop", True)),
            ("V3_dynamic_MatMulGemm_only",
             lambda: quant_dynamic_mm(fp32, M2_DIR / f"{ds}_{sc}_seed{seed}_V3_dynmm.onnx")),
        ]
        for name, fn in builders:
            try:
                jobs.append((name, fn()))
            except Exception as e:      # 某个变体导出失败不阻塞其余(会在报告里注明)
                print(f"  [skip] {ds}-{sc} {name}: {type(e).__name__}: {e}")
        variants: list[tuple[str, Path, bool]] = [(n, p, False) for n, p in jobs]
        for name, path, _ in variants:
            h = opcounts(path)
            r = {"cell": f"{ds}-{sc}", "variant": name, "onnx": path.name,
                 "q_dq": h["QuantizeLinear"] + h["DequantizeLinear"],
                 "qlinearconv": h["QLinearConv"], "conv": h["Conv"] + h["ConvInteger"],
                 "session": sess_id}
            r.update(timed(path, ds))
            r["bg_cpu_pct"] = sample_load(seconds=1.5).get("total_pct")
            rows.append(r)
            print(f"  {ds}-{sc} {name:32s} median {r['median']:>8.2f} ms "
                  f"(Q/DQ {r['q_dq']}, QLinearConv {r['qlinearconv']}, Conv {r['conv']})")
    stock_control(sess_id, rows)

    # 同种子/同会话比值(相对该格子的 FP32)
    base = {(r["cell"], r["variant"]): r["median"] for r in rows if r["variant"] == "fp32"}
    for r in rows:
        b = base.get((r["cell"], "fp32"))
        r["ratio_vs_fp32"] = round(r["median"] / b, 3) if b else ""
    csvp = C.OUT / "m2_variants_latency.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "variant", "onnx", "q_dq", "qlinearconv",
                                          "conv", "median", "p95", "min", "max",
                                          "ratio_vs_fp32", "bg_cpu_pct", "session"])
        w.writeheader(); w.writerows(rows)
    print(f"\n已写入 {csvp}({len(rows)} 行,会话 {sess_id})")

    md = ["# 论文四 · M2 交叉验证:换成已知正确的导出口径后,INT8 还是慢吗?", "",
          f"> 会话 `{sess_id}`(一段不间断、独占 CPU);50 次单图推理,4 线程,640×640;",
          "> 与主矩阵同一测量协议。本表是**诊断**,不替换主矩阵。", "",
          "| cell | variant | median ms | p95 ms | vs FP32 | Q/DQ 节点 | QLinearConv | Conv/ConvInteger |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['cell']} | {r['variant']} | {r['median']} | {r['p95']} | "
                  f"{r['ratio_vs_fp32']}× | {r['q_dq']} | {r['qlinearconv']} | {r['conv']} |")
    (C.OUT / "m2_variants_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"已写入 {C.OUT / 'm2_variants_report.md'}")


if __name__ == "__main__":
    main()
