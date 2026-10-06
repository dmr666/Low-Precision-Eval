# -*- coding: utf-8 -*-
"""论文四 · PTQ 受控评估:公共配置(路径 / 配置矩阵 / 口径常量)

设计约束(来自本项目的历史教训):
- 一切路径用绝对路径(Windows 正斜杠),数据 yaml 必须绝对路径,否则 ultralytics 报 FileNotFoundError。
- 屏幕上/论文里出现的数字**一律由脚本从 out/*.csv 现算**,禁止手抄。
- FP32 必须与论文三的 val 数字对齐(容差 TOL),超差不得解读后续精度对比。

用法:被其他脚本 import。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
PAPER3 = Path(os.environ.get("PAPER3_DIR", "upstream/paper3"))
PROJ = Path(os.environ.get("PTQ_DIR", ".")).resolve()
OUT = PROJ / "out"
WEIGHTS = PROJ / "weights"
ONNX = PROJ / "onnx"
RUNS = PROJ / "runs"          # ultralytics val 的 save_dir

for _d in (OUT, WEIGHTS, ONNX, RUNS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 数据集
# names 直接取自论文三的模型 yaml / 数据 yaml(不手抄类别名)
DATASETS = {
    "gc10": {
        "yaml": PAPER3 / "data" / "GC10-DET-yolo" / "gc10_det.yaml",
        "names": ["chongkong", "hanfeng", "yueyawan", "shuiban", "youban",
                  "siban", "yiwu", "yahen", "zhehen", "yaozhe"],
        "note": "钢材冷轧 10 类缺陷,11.4:1 实例不平衡,25% 目标 <32px",
    },
    "neu": {
        "yaml": PAPER3 / "data" / "NEU-DET-split" / "neu_det.yaml",
        "names": ["crazing", "inclusion", "patches", "pitted_surface",
                  "rolled-in_scale", "scratches"],
        "note": "热轧 6 类缺陷,第二基准",
    },
}

SCALES = ["n", "s"]
SEEDS = [42, 7, 2024]          # 原始三种子:所有 dataset×scale 格子都有
# 补充种子:论文三返修阶段已用**完全相同配方**训过 s 尺度,可零训练复用(配方已逐字段核对
# args.yaml:epochs/batch/imgsz/optimizer/lr0/close_mosaic/全部增强字段一致,仅 seed 不同)
SEEDS_EXTRA = {
    ("gc10", "s"): [13, 99],    # 论文三返修阶段已训,零成本复用
    ("neu", "s"): [13, 99],
    ("gc10", "n"): [13, 99],    # 2026-10-06 在 cloud GPU 用同配方补训(审计 47/0)
    ("neu", "n"): [13, 99],
}


def seeds_for(dataset: str, scale: str) -> list:
    """该格子实际可用的种子(权重不存在的不算,避免脚本报错)。"""
    extra = [s for s in SEEDS_EXTRA.get((dataset, scale), [])
             if weight_path(dataset, scale, s).exists()]
    return SEEDS + extra


def cells() -> list:
    """(dataset, scale, seed) 全表;供各脚本枚举用。"""
    return [(d, s, sd) for d in DATASETS for s in SCALES for sd in seeds_for(d, s)]


# 精度变体:fp32 = 参照;fp16/int8dyn/int8static 为被评估对象
PRECISIONS = ["fp32", "fp16", "int8dyn", "int8static"]

# 权重文件命名:<dataset>_<scale>_<tag>_seed<seed>.pt,tag 默认 base(不做处置的基线)
def weight_path(dataset: str, scale: str, seed: int, tag: str = "base") -> Path:
    return WEIGHTS / f"{dataset}_{scale}_{tag}_seed{seed}.pt"


def onnx_path(dataset: str, scale: str, seed: int, precision: str,
              tag: str = "base") -> Path:
    return ONNX / f"{dataset}_{scale}_{tag}_seed{seed}_{precision}.onnx"


# ---------------------------------------------------------------- 口径
# AP_s/m/l 的 COCO 面积分档(COCO 标准,像素^2)
AREA_RANGES = {
    "all": (0 ** 2, 1e10),
    "small": (0 ** 2, 32 ** 2),
    "medium": (32 ** 2, 96 ** 2),
    "large": (96 ** 2, 1e10),
}

# 与论文三 FP32 的对表容差:超过此值先查口径,不做解读
TOL = 0.010

# 面积分档口径:eq640 = 把 bbox 面积折算到 640 输入空间(与论文三 scale_profile.py 一致:
# s = 640/max(W,H),eq 面积 = w*h*s^2),这样 APs/APm/APl 的 1024/9216 阈值才对应
# "等效边长 <32px / 32–96px / >96px"。GC10-DET 图像 2048x1000,用原始像素分档会几乎全落进
# "large" 桶(实测 raw 口径下 APs 桶近乎为空,曾算出 AP50_s=1.0 的退化值)。
AREA_SPACE = "eq640"
EQ_IMGSZ = 640

# 论文三的公开仓(逐种子指标可复算 FP32 参考值)
PAPER3_REPO = PAPER3 / "_github" / "Steel-Defect-Eval"
PAPER3_METRICS_CANDIDATES = [
    PAPER3_REPO / "metrics",
    PAPER3 / "_results" / "server_20260930",
    PAPER3 / "_results",
]


def find_paper3_reference(dataset: str, scale: str, tag: str = "base"):
    """在论文三的产物里找 FP32 参考结果(逐 run 的 results.csv)。

    返回 [(path, dict)];找不到返回 [];由调用方决定是否作为硬门。
    只做"找文件",不做数值判断 —— 数值一律现算。
    """
    hits = []
    for root in PAPER3_METRICS_CANDIDATES:
        if not root.exists():
            continue
        for csv in root.rglob("results.csv"):
            name = str(csv).lower()
            if dataset in name and tag in name:
                # 尺度线索:n / s;论文三的 run 目录名里带尺度或 tag
                if f"_{scale}_" in name or f"{scale}_" in csv.parent.name.lower():
                    hits.append(csv)
    return hits
