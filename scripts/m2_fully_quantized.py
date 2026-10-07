# -*- coding: utf-8 -*-
"""M2 追加诊断:把"部分量化(只量化 Conv/Gemm/MatMul)"换成"整图量化",看慢的到底是
   (a) 量化边界来回换格式的开销,还是 (b) ORT 在本 CPU 上的整数卷积内核本身不划算。

背景:上一轮已测出,即使把图正确融合成 64 个 QLinearConv(QOperator),
静态 INT8 仍比 FP32 慢约 12.7-12.9 倍;而那张图里还残留 59 个 QuantizeLinear / 64 个 DequantizeLinear
—— 因为框架的节点选择把 Concat/Sigmoid/Mul 等全留在浮点,每两个卷积之间就要 反量化→浮点→再量化 一次。

V4 = quantize_static(QOperator, **不排除任何节点**) → 整数算子之间不再需要边界转换。
    若 V4 明显变快 ⇒ 结论是"框架的部分量化策略"是代价来源;
    若 V4 仍慢 ⇒ 结论是"本 CPU 上 ORT 的整数卷积内核在这个模型族上不划算"。

只测延迟(不评精度):这组数字是机制诊断,主矩阵与结论仍以默认导出路径为准。
用法: M2_SESSION=<id> python scripts/m2_fully_quantized.py
"""
from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path

import ptq_common as C
from host_load import exclusivity_check, sample_load
from m2_int8_variants import CELLS, IMGSZ, M2_DIR, opcounts, timed
from ptq_calib_ablation import FilesReader, split_images


def quant_full(fp32: Path, dest: Path, reader, per_channel: bool = False) -> Path:
    """整图量化:不排除任何节点(能量化的都量化)。"""
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"    [cache] {dest.name}")
        return dest
    tmp = Path(tempfile.gettempdir()) / "m2f"
    tmp.mkdir(parents=True, exist_ok=True)
    src, out = tmp / "in.onnx", tmp / "out.onnx"
    shutil.copyfile(fp32, src)
    if hasattr(reader, "rewind"):
        reader.rewind()
    quantize_static(str(src), str(out), reader, quant_format=QuantFormat.QOperator,
                    per_channel=per_channel, weight_type=QuantType.QInt8)
    shutil.copyfile(out, dest)
    return dest


def main() -> None:
    sess_id = os.environ.get("M2_SESSION", "")
    if not sess_id:
        raise SystemExit("必须 export M2_SESSION=<会话 id>")
    thr = float(os.environ.get("M2_LOAD_THRESHOLD", "15"))
    ok, d = exclusivity_check(thr, seconds=6.0)
    print(f"主机占用门禁:平均 {d.get('total_pct')}% / 峰值 {d.get('peak_pct')}%(阈值 {thr}%)⇒ "
          f"{'干净 ✓' if ok else '不干净 ✗'}")
    if not ok and os.environ.get("M2_ALLOW_BUSY") != "1":
        raise SystemExit("主机不干净,拒绝开始计时")

    rows = []
    for ds, sc, seed in CELLS:
        fp32 = C.onnx_path(ds, sc, seed, "fp32")
        if not fp32.exists():
            print(f"  [skip] {ds}-{sc}:缺 FP32")
            continue
        reader = FilesReader(split_images(ds, "val"), IMGSZ)
        v4 = quant_full(fp32, M2_DIR / f"{ds}_{sc}_seed{seed}_V4_fullq.onnx", reader)
        h = opcounts(v4)
        r = {"cell": f"{ds}-{sc}", "variant": "V4_static_fullquant_QOperator", "onnx": v4.name,
             "q_dq": h["QuantizeLinear"] + h["DequantizeLinear"],
             "qlinearconv": h["QLinearConv"], "conv": h["Conv"] + h["ConvInteger"],
             "session": sess_id}
        r.update(timed(v4, ds))
        # 同会话 FP32 参照,保证比值可比
        r["fp32_median_same_session"] = timed(fp32, ds)["median"]
        r["ratio_vs_fp32"] = round(r["median"] / r["fp32_median_same_session"], 3)
        r["bg_cpu_pct"] = sample_load(seconds=1.5).get("total_pct")
        rows.append(r)
        print(f"  {ds}-{sc} V4 整图量化 median {r['median']:>8.2f} ms vs 同会话 FP32 "
              f"{r['fp32_median_same_session']:>7.2f} ms ⇒ {r['ratio_vs_fp32']}× "
              f"(Q/DQ {r['q_dq']}, QLinearConv {r['qlinearconv']}, Conv {r['conv']})")

    out = C.OUT / "m2_fullquant_latency.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "variant", "onnx", "q_dq", "qlinearconv", "conv",
                                          "median", "p95", "min", "max", "fp32_median_same_session",
                                          "ratio_vs_fp32", "bg_cpu_pct", "session"])
        w.writeheader(); w.writerows(rows)
    print(f"已写入 {out}({len(rows)} 行)")


if __name__ == "__main__":
    main()
