# 论文四 · M2 剖面:整数图的时间花在哪(单次推理,ORT profiling)

> 会话 `m2h_20261007_175222`;每个模型跑 1 次(已 warmup),按算子累计耗时排序。

## fp32 (gc10-n) — 合计 108.79 ms

| 算子 | 次数 | 累计 ms | 占比 |
|---|---|---|---|
| /model.1/conv/Conv_output_0_nchwc_kernel_time | 1 | 10.27 | 9.4% |
| /model.18/m.0/cv1/conv/Conv_output_0_nchwc_kernel_time | 1 | 7.25 | 6.7% |
| /model.15/cv1/conv/Conv_output_0_nchwc_kernel_time | 1 | 5.88 | 5.4% |
| /model.22/dfl/Softmax_kernel_time | 1 | 5.33 | 4.9% |
| /model.2/cv1/conv/Conv_output_0_nchwc_kernel_time | 1 | 5.16 | 4.7% |
| /model.22/cv3.0/cv3.0.0/conv/Conv_output_0_nchwc_kernel_time | 1 | 2.88 | 2.7% |
| /model.22/cv2.0/cv2.0.0/conv/Conv_output_0_nchwc_kernel_time | 1 | 2.88 | 2.6% |
| /model.22/cv3.0/cv3.0.1/conv/Conv_output_0_nchwc_kernel_time | 1 | 2.88 | 2.6% |
| /model.22/cv2.0/cv2.0.1/conv/Conv_output_0_nchwc_kernel_time | 1 | 2.87 | 2.6% |
| /model.2/cv1/act/Mul/QuickGeluFusion/_kernel_time | 1 | 2.74 | 2.5% |
| /model.0/conv/Conv_output_0_nchwc_kernel_time | 1 | 2.03 | 1.9% |
| /model.15/cv1/act/Mul/QuickGeluFusion/_kernel_time | 1 | 1.85 | 1.7% |
| /model.18/m.0/cv2/conv/Conv_output_0_nchwc_kernel_time | 1 | 1.73 | 1.6% |
| /model.1/act/Mul/QuickGeluFusion/_kernel_time | 1 | 1.62 | 1.5% |

## V1_qop (gc10-n) — 合计 800.06 ms

| 算子 | 次数 | 累计 ms | 占比 |
|---|---|---|---|
| /model.22/cv3.0/cv3.0.0/conv/Conv_quant_token_235_kernel_time | 1 | 43.01 | 5.4% |
| /model.22/cv2.0/cv2.0.0/conv/Conv_quant_token_158_kernel_time | 1 | 41.54 | 5.2% |
| /model.22/cv2.0/cv2.0.1/conv/Conv_quant_token_162_kernel_time | 1 | 40.8 | 5.1% |
| /model.22/cv3.0/cv3.0.1/conv/Conv_quant_token_239_kernel_time | 1 | 40.19 | 5.0% |
| /model.7/conv/Conv_quant_token_98_kernel_time | 1 | 24.17 | 3.0% |
| /model.1/conv/Conv_quant_token_22_kernel_time | 1 | 24.07 | 3.0% |
| /model.3/conv/Conv_quant_token_42_kernel_time | 1 | 22.54 | 2.8% |
| /model.22/cv2.1/cv2.1.0/conv/Conv_quant_token_189_kernel_time | 1 | 21.95 | 2.7% |
| /model.22/cv3.1/cv3.1.0/conv/Conv_quant_token_246_kernel_time | 1 | 21.77 | 2.7% |
| /model.5/conv/Conv_quant_token_70_kernel_time | 1 | 20.48 | 2.6% |
| /model.0/conv/Conv_quant_token_18_kernel_time | 1 | 14.82 | 1.9% |
| /model.19/conv/Conv_quant_token_200_kernel_time | 1 | 14.05 | 1.8% |
| /model.12/cv1/conv/Conv_quant_token_126_kernel_time | 1 | 13.69 | 1.7% |
| /model.2/m.0/cv1/conv/Conv_quant_token_30_kernel_time | 1 | 13.64 | 1.7% |

## V4_fullq (gc10-n) — 合计 808.25 ms

| 算子 | 次数 | 累计 ms | 占比 |
|---|---|---|---|
| /model.22/cv2.0/cv2.0.0/conv/Conv_quant_token_203_kernel_time | 1 | 54.28 | 6.7% |
| /model.22/cv3.0/cv3.0.0/conv/Conv_quant_token_240_kernel_time | 1 | 46.66 | 5.8% |
| /model.22/cv3.0/cv3.0.1/conv/Conv_quant_token_244_kernel_time | 1 | 42.65 | 5.3% |
| /model.22/cv2.0/cv2.0.1/conv/Conv_quant_token_207_kernel_time | 1 | 41.32 | 5.1% |
| /model.7/conv/Conv_quant_token_81_kernel_time | 1 | 25.25 | 3.1% |
| /model.3/conv/Conv_quant_token_25_kernel_time | 1 | 21.85 | 2.7% |
| /model.5/conv/Conv_quant_token_53_kernel_time | 1 | 21.79 | 2.7% |
| /model.1/conv/Conv_quant_token_5_kernel_time | 1 | 20.95 | 2.6% |
| /model.22/cv2.1/cv2.1.0/conv/Conv_quant_token_192_kernel_time | 1 | 19.93 | 2.5% |
| /model.22/cv3.1/cv3.1.0/conv/Conv_quant_token_229_kernel_time | 1 | 17.47 | 2.2% |
| /model.12/cv1/conv/Conv_quant_token_109_kernel_time | 1 | 17.11 | 2.1% |
| /model.15/cv1/conv/Conv_quant_token_125_kernel_time | 1 | 15.43 | 1.9% |
| /model.21/m.0/cv1/conv/Conv_quant_token_169_kernel_time | 1 | 14.24 | 1.8% |
| /model.2/m.0/cv2/conv/Conv_quant_token_17_kernel_time | 1 | 13.31 | 1.6% |

