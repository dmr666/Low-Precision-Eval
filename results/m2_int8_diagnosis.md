# 论文四 · M2 诊断:INT8 变体的导出口径(审稿意见 M2 的证据)

> 生成环境:Windows-11-10.0.26200-SP0,python 3.12.13, onnxruntime 1.30.0, onnx 1.23.1

## 1. 主机与指令集

coreinfo 采集(原始文件 `_verify/coreinfo_i5-1155G7.txt`):

```
11th Gen Intel(R) Core(TM) i5-1155G7 @ 2.50GHz
Intel64 Family 6 Model 140 Stepping 2, GenuineIntel
AVX       	*	Supports AVX instruction extensions
AVX2      	*	Supports AVX2 instruction extensions
AVX-512-F 	*	Supports AVX-512 Foundation instructions
AVX-512-DQ	*	Supports AVX-512 double and quadword instructions
AVX-512-IFMA	*	Supports AVX-512 integer Fused multiply-add instructions
AVX-512-PF	-	Supports AVX-512 prefetch instructions
AVX-512-ER	-	Supports AVX-512 exponential and reciprocal instructions
AVX-512-CD	*	Supports AVX-512 conflict detection instructions
AVX-512-BW	*	Supports AVX-512 byte and word instructions
AVX-512-VL	*	Supports AVX-512 vector length instructions
AVX-512-VBMI	*	Supports AVX-512 Vector Byte Manipulation instructions
AVX-512-VBMI2	*	Supports AVX-512 Vector Byte Manipulation instructions 2
AVX-512-VNNI	*	Supports AVX-512 Vector Neural Network instructions
AVX-512-BITALG	*	Supports AVX-512 Bit Algorithms instructions
AVX-512-VPCNTDQ	*	Supports AVX-512 Vector Population Count Doubleword/Quadword
AVX-512-4VNNIW	-	Supports AVX-512 4-register Neural Network instructions
AVX-512-4FMAPS	-	Supports AVX-512 4-register Multiply Accumulation Single precision
AVX-512-VPISECT	*	Supports AVX-512 Vector Pair Intersection to a Pair of Mask Registers
AVX-512-FP16	-	Supports AVX-512 half-precision floating-point instructions
AVX-512-BF16	-	Supports AVX-512 BFloat16 instructions
AVX-VNNI  	-	Supports VEX-encoded VNNI instructions
AVX-IFMA  	-	Supports VEX-encoded Integer FMA instructions
x86-64-v1 	*	Baseline: CMOV, CX8, FPU, FXSR, MMX, SSE, SSE2
x86-64-v2 	*	Level 2: CX16, LAHF-SAHF, POPCNT, SSE3, SSE4.1, SSE4.2, SSSE3
x86-64-v3 	*	Level 3: AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, OSXSAVE
x86-64-v4 	*	Level 4: AVX-512-F, AVX-512-BW, AVX-512-CD, AVX-512-DQ, AVX-512-VL
```

- ORT providers: ['AzureExecutionProvider', 'CPUExecutionProvider']

## 2. 各精度导出的算子构成(4 个格子 × 4 个变体)

| cell | variant | nodes | Conv | MatMul | Q/DQ | QLinearConv | ConvInteger | MatMulInteger | DynamicQuantizeLinear |
|---|---|---|---|---|---|---|---|---|---|
| gc10-n | fp32 | 231 | 64 | 0 | 0 | 0 | 0 | 0 | 0 |
| gc10-n | fp16 | 233 | 64 | 0 | 0 | 0 | 0 | 0 | 0 |
| gc10-n | int8dyn | 608 | 0 | 0 | 0 | 0 | 64 | 0 | 59 |
| gc10-n | int8static | 604 | 64 | 0 | 373 | 0 | 0 | 0 | 0 |
| gc10-s | fp32 | 231 | 64 | 0 | 0 | 0 | 0 | 0 | 0 |
| gc10-s | fp16 | 233 | 64 | 0 | 0 | 0 | 0 | 0 | 0 |
| gc10-s | int8dyn | 608 | 0 | 0 | 0 | 0 | 64 | 0 | 59 |
| gc10-s | int8static | 604 | 64 | 0 | 373 | 0 | 0 | 0 | 0 |
| neu-n | fp32 | 231 | 64 | 0 | 0 | 0 | 0 | 0 | 0 |
| neu-n | fp16 | 233 | 64 | 0 | 0 | 0 | 0 | 0 | 0 |
| neu-n | int8dyn | 608 | 0 | 0 | 0 | 0 | 64 | 0 | 59 |
| neu-n | int8static | 604 | 64 | 0 | 373 | 0 | 0 | 0 | 0 |
| neu-s | fp32 | 231 | 64 | 0 | 0 | 0 | 0 | 0 | 0 |
| neu-s | fp16 | 233 | 64 | 0 | 0 | 0 | 0 | 0 | 0 |
| neu-s | int8dyn | 608 | 0 | 0 | 0 | 0 | 64 | 0 | 59 |
| neu-s | int8static | 604 | 64 | 0 | 373 | 0 | 0 | 0 | 0 |

## 3. ORT 优化后的图(QDQ 是否融合成整数算子)

| cell | variant | 优化前 Q/DQ | 优化后 Q/DQ | 优化后 QLinearConv | 优化后 ConvInteger | 结论 |
|---|---|---|---|---|---|---|
| gc10-n | int8dyn | 0 | 0 | 0 | 64 | 融合成功 |
| gc10-n | int8static | 373 | 437 | 7 | 0 | 无可融合的量化算子 |
| gc10-s | int8dyn | 0 | 0 | 0 | 64 | 融合成功 |
| gc10-s | int8static | 373 | 437 | 7 | 0 | 无可融合的量化算子 |
| neu-n | int8dyn | 0 | 0 | 0 | 64 | 融合成功 |
| neu-n | int8static | 373 | 437 | 7 | 0 | 无可融合的量化算子 |
| neu-s | int8dyn | 0 | 0 | 0 | 64 | 融合成功 |
| neu-s | int8static | 373 | 437 | 7 | 0 | 无可融合的量化算子 |

## 4. 会话配置

- 默认优化级别:ORT_ENABLE_ALL(脚本用 `optimized_model_filepath` 导出了优化图,可直接查看)
- intra_op_num_threads(测量口径):4
- 量化器:`onnxruntime.quantization.quantize_dynamic`(动态)/ 训练框架内置静态路径(见 3.2 节)

