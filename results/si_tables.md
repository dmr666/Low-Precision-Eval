# 论文四 · Supplementary 表(脚本生成,勿手改)

> 来源 `out/results.csv`(80 行);分档口径 pycocotools COCOeval,面积折算到 640 输入空间(eq640)。

## Table S1 · 配对 ΔAP50 按尺寸分档(5 种子均值)

| dataset | scale | variant | ΔAP50_small | ΔAP50_medium | ΔAP50_large |
|---|---|---|---|---|---|
| GC10 | n | int8static | -0.0139 | -0.0190 | -0.0133 |
| GC10 | n | int8dyn | +0.0214 | -0.0097 | -0.0083 |
| GC10 | n | fp16 | -0.0001 | -0.0003 | -0.0019 |
| GC10 | s | int8static | -0.0796 | -0.0114 | -0.0324 |
| GC10 | s | int8dyn | -0.0094 | -0.0013 | -0.0008 |
| GC10 | s | fp16 | +0.0003 | +0.0000 | -0.0001 |
| NEU | n | int8static | +0.0000 | -0.0107 | -0.0030 |
| NEU | n | int8dyn | +0.0000 | -0.0290 | -0.0067 |
| NEU | n | fp16 | +0.0000 | +0.0007 | +0.0004 |
| NEU | s | int8static | +0.0000 | -0.1260 | +0.0063 |
| NEU | s | int8dyn | +0.0000 | -0.0397 | -0.0020 |
| NEU | s | fp16 | +0.0000 | +0.0070 | +0.0000 |

## Table S1b · 每格 AP50_small 的跨种子离散(SD)

| dataset | scale | precision | mean AP50_small | SD across seeds |
|---|---|---|---|---|
| GC10 | n | fp32 | 0.6000 | 0.0877 |
| GC10 | n | fp16 | 0.5999 | 0.0876 |
| GC10 | n | int8dyn | 0.6213 | 0.1022 |
| GC10 | n | int8static | 0.5860 | 0.0874 |
| GC10 | s | fp32 | 0.6583 | 0.1133 |
| GC10 | s | fp16 | 0.6586 | 0.1132 |
| GC10 | s | int8dyn | 0.6489 | 0.0993 |
| GC10 | s | int8static | 0.5787 | 0.1229 |
| NEU | n | fp32 | 0.0000 | 0.0000 |
| NEU | n | fp16 | 0.0000 | 0.0000 |
| NEU | n | int8dyn | 0.0000 | 0.0000 |
| NEU | n | int8static | 0.0000 | 0.0000 |
| NEU | s | fp32 | 0.0000 | 0.0000 |
| NEU | s | fp16 | 0.0000 | 0.0000 |
| NEU | s | int8dyn | 0.0000 | 0.0000 |
| NEU | s | int8static | 0.0000 | 0.0000 |

## Table S1c · 小目标损失 / 大目标损失(仅 >1 表示小目标更吃亏;GC10-DET)

| scale | variant | Δsmall | Δlarge | 比值 small/large |
|---|---|---|---|---|
| n | int8static | -0.0139 | -0.0133 | 1.05x |
| n | int8dyn | +0.0214 | -0.0083 | -2.59x |
| n | fp16 | -0.0001 | -0.0019 | 0.05x |
| s | int8static | -0.0796 | -0.0324 | 2.46x |
| s | int8dyn | -0.0094 | -0.0008 | 11.66x |
| s | fp16 | +0.0003 | -0.0001 | -3.57x |

## Table S1d · NEU-DET 的 medium vs large(无小目标,故只比 m/l)

| scale | variant | Δmedium | Δlarge |
|---|---|---|---|
| n | int8static | -0.0107 | -0.0030 |
| n | int8dyn | -0.0290 | -0.0067 |
| s | int8static | -0.1260 | +0.0063 |
| s | int8dyn | -0.0397 | -0.0020 |

## Table S2 · GC10-DET s,seed 42 逐类 mAP50-95

| class | chongkong | hanfeng | yueyawan | shuiban | youban | siban | yiwu | yahen | zhehen | yaozhe |
|---|---|---|---|---|---|---|---|---|---|---|
| FP32 | 0.5816 | 0.4581 | 0.5717 | 0.3755 | 0.3398 | 0.2620 | 0.1030 | 0.0877 | 0.0540 | 0.2524 |
| INT8-static | 0.5660 | 0.4592 | 0.5536 | 0.3105 | 0.2995 | 0.2148 | 0.0946 | 0.0673 | 0.0741 | 0.2154 |
| Δ | -0.0156 | +0.0011 | -0.0181 | -0.0651 | -0.0404 | -0.0472 | -0.0084 | -0.0205 | +0.0201 | -0.0370 |

