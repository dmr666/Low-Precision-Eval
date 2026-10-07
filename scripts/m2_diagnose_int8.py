# -*- coding: utf-8 -*-
"""M2 诊断(审稿意见:INT8 在支持 VNNI 的 CPU 上慢 13-15 倍,需排除导出口径伪影)。

产出 out/m2_int8_diagnosis.md,包含:
  1) 主机 ISA 证据(CPU 型号 + AVX-512/VNNI 是否存在,由 coreinfo 采集后落盘);
  2) 各精度 ONNX 的算子直方图(Q/DQ、QLinearConv、ConvInteger、DynamicQuantizeLinear 是否出现);
  3) 用 ORT 图优化器 dump optimized 图,再数一次 Q/DQ —— 用于判断 QDQ 是否被融合成整数算子;
  4) 会话配置(优化级别、线程数、providers)。

只做诊断,不测延迟(计时另开独占会话)。
用法: python scripts/m2_diagnose_int8.py
"""
from __future__ import annotations

import collections
import csv
import platform
import sys
from pathlib import Path

import onnx
import onnxruntime as ort

import ptq_common as C

CELLS = [("gc10", "n", "42"), ("gc10", "s", "42"), ("neu", "n", "42"), ("neu", "s", "42")]
PRECS = ["fp32", "fp16", "int8dyn", "int8static"]
QOPS = ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear",
        "QLinearConv", "ConvInteger", "MatMulInteger", "QGemm", "QLinearMatMul")


def onnx_path(ds: str, sc: str, seed: str, prec: str) -> Path:
    cands = list(C.ONNX.glob(f"{ds}_{sc}_*seed{seed}*{prec}.onnx")) or \
            list(C.ONNX.glob(f"{ds}_{sc}_seed{seed}_{prec}.onnx"))
    if not cands:
        raise SystemExit(f"找不到 {ds}/{sc}/seed{seed}/{prec} 的 ONNX:{C.ONNX}")
    return cands[0]


def op_hist(p: Path) -> collections.Counter:
    m = onnx.load(str(p), load_external_data=False)
    return collections.Counter(n.op_type for n in m.graph.node)


def optimized_hist(p: Path) -> tuple[collections.Counter, Path]:
    """让 ORT 把优化后的图落盘(默认优化级别 = ORT_ENABLE_ALL),再数算子。"""
    out = C.OUT / f"_opt_{p.stem}.onnx"
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = str(out)
    so.intra_op_num_threads = 4
    ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
    return op_hist(out), out


def main() -> None:
    L: list[str] = ["# 论文四 · M2 诊断:INT8 变体的导出口径(审稿意见 M2 的证据)",
                    "",
                    f"> 生成环境:{platform.platform()},python {sys.version.split()[0]},"
                    f" onnxruntime {ort.__version__}, onnx {onnx.__version__}",
                    "",
                    "## 1. 主机与指令集", ""]
    isa = None
    for cand in (C.PROJ / "_verify" / "coreinfo_i5-1155G7.txt",
                 C.OUT / "coreinfo_i5-1155G7.txt",
                 C.PROJ / "results" / "coreinfo_i5-1155G7.txt"):
        if cand.exists():
            isa = cand
            break
    if isa is not None:
        L += [f"coreinfo 采集(原始文件 `{isa.name}`):", "", "```"]
        for ln in isa.read_text(encoding="utf-8", errors="replace").splitlines():
            s = ln.strip()
            if s.startswith(("11th Gen Intel", "Intel64 Family 6", "AVX", "x86-64-v")):
                L.append(s)
        L += ["```", ""]
    else:
        L += ["未找到 coreinfo 采集文件 —— 需先落盘 ISA 证据。", ""]

    L += [f"- ORT providers: {ort.get_available_providers()}", ""]

    # 2) 各精度算子直方图
    L += ["## 2. 各精度导出的算子构成(4 个格子 × 4 个变体)", "",
          "| cell | variant | nodes | Conv | MatMul | Q/DQ | QLinearConv | ConvInteger | MatMulInteger | DynamicQuantizeLinear |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    raw: dict[tuple, collections.Counter] = {}
    for ds, sc, seed in CELLS:
        for prec in PRECS:
            p = onnx_path(ds, sc, seed, prec)
            h = op_hist(p)
            raw[(ds, sc, prec)] = h
            nqdq = h["QuantizeLinear"] + h["DequantizeLinear"]
            L.append(f"| {ds}-{sc} | {prec} | {sum(h.values())} | {h['Conv']} | {h['MatMul']} | {nqdq} | "
                     f"{h['QLinearConv']} | {h['ConvInteger']} | {h['MatMulInteger']} | {h['DynamicQuantizeLinear']} |")
    L.append("")

    # 3) ORT 优化后的图:QDQ 是否被融合
    L += ["## 3. ORT 优化后的图(QDQ 是否融合成整数算子)", "",
          "| cell | variant | 优化前 Q/DQ | 优化后 Q/DQ | 优化后 QLinearConv | 优化后 ConvInteger | 结论 |",
          "|---|---|---|---|---|---|---|"]
    for ds, sc, seed in CELLS:
        for prec in ("int8dyn", "int8static"):
            p = onnx_path(ds, sc, seed, prec)
            h0 = raw[(ds, sc, prec)]
            h1, _ = optimized_hist(p)
            before = h0["QuantizeLinear"] + h0["DequantizeLinear"]
            after = h1["QuantizeLinear"] + h1["DequantizeLinear"]
            fused = h1["QLinearConv"] + h1["ConvInteger"] + h1["MatMulInteger"]
            if fused and after < before:
                concl = "融合成功(整数内核可命中)"
            elif fused and after == 0:
                concl = "融合成功"
            elif after and not fused:
                concl = "**未融合**:Q/DQ 残留,每算子来回换格式"
            else:
                concl = "无可融合的量化算子"
            L.append(f"| {ds}-{sc} | {prec} | {before} | {after} | {h1['QLinearConv']} | "
                     f"{h1['ConvInteger']} | {concl} |")
    L.append("")

    # 4) 会话配置
    so = ort.SessionOptions()
    L += ["## 4. 会话配置", "",
          f"- 默认优化级别:ORT_ENABLE_ALL(脚本用 `optimized_model_filepath` 导出了优化图,可直接查看)",
          f"- intra_op_num_threads(测量口径):4",
          f"- 量化器:`onnxruntime.quantization.quantize_dynamic`(动态)/ 训练框架内置静态路径(见 3.2 节)",
          ""]
    # 写出
    out = C.OUT / "m2_int8_diagnosis.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"已写入 {out}")
    print("\n".join(L[:6]))


if __name__ == "__main__":
    main()
