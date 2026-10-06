# Low-Precision-Eval — artifacts for
# "What Does Low-Precision Deployment Actually Cost Small Industrial Detectors?"

Everything in this repository supports the measurements in the paper: twenty trained detectors are exported to four
ONNX precision variants (FP32 / FP16 / INT8-dynamic / INT8-static), evaluated for accuracy (overall, per class,
per size bucket) and measured for CPU latency and size. **No training is involved** — the study is a controlled
evaluation of already-trained models.

## Layout

```
scripts/    every script that produces every number in the paper
weights/    the twenty baseline checkpoints (minimal, metadata-free), 6.2 MB / 22.5 MB each
results/    raw result files (one row per evaluation / measurement)
  results.csv          80 rows: per-configuration accuracy (framework and COCOeval metrics)
  latency.csv          80 rows: median CPU latency over 50 runs, one uninterrupted session
                       (a `session` column records which; never mix rows from different sessions),
                       model size, threads, input type
  latency_session_*.csv  the two complete measurement sessions kept as evidence (both are 80 rows,
                       each with its own `session` id; they must never be mixed)
  realtime_budget.md   real-time framing: per-frame budget (33.3 / 16.7 ms), p95 tail and sustained
                       frames per second per four-thread worker, derived from latency.csv
  latency_session_compare.md  cross-session comparison: within-cell spread of the five seed models,
                       interference indicators (max p95/median) and the within-seed ratio agreement
  stats.csv            cross-seed means and standard deviations
  significance.csv     Welch t vs FP32 with Holm correction (30 comparisons)
  calib_ablation.csv   calibration-set ablation (validation split / 200 train images / random tensors)
  export_sizes.csv     ONNX file sizes per variant
  weights_manifest.csv provenance of every weight file (source archive + member path)
  tables.md            auto-generated paper table skeletons
  paper_tables.md      the manuscript's Tables 2-6, script-generated
  si_tables.md         the Supplementary tables S1/S2, script-generated
  seed_extension_report.md  report for the two extra seeds (accuracy, paired deltas, cost, significance)
  numbers_sheet.md     every number cited by the manuscript, script-generated
  perclass/            per-class AP dumps, one JSON per evaluation (80 files)
figs/       fig_pareto.png / .pdf — accuracy–latency–size trade-offs (Figure 1)
calib/      the exact 200-image training lists used by the calibration ablation
```

## Reproducing

```
export PAPER3_DIR=/path/to/paper3-protocol-dir   # datasets + baseline weights (see below)
python scripts/preflight.py
python scripts/ptq_export.py --all --variants fp32,fp16,int8dyn,int8static --imgsz 640
python scripts/ptq_eval.py   --all --imgsz 640 --split val
python scripts/ptq_latency.py --all --imgsz 640 --n 50 --threads 4
# recommended instead of the single line above: all 80 latency rows in ONE uninterrupted,
# CPU-exclusive session (~75 min on 8 logical cores); it stamps a session id per row and
# then re-runs every statistics/table/figure step below. Add PTQ_WARMUP=300 to spend five
# minutes on the same workload first (thermal steady state); keep both sessions and compare
# them with scripts/compare_latency_sessions.py before choosing which one to report
PTQ_WARMUP=300 bash scripts/remeasure_latency_all.sh
python scripts/ptq_stats.py
python scripts/ptq_calib_ablation.py --configs gc10:s:42,neu:s:42 --sources val,train200,random20
python scripts/make_fig_pareto.py
python scripts/make_numbers_sheet.py
python scripts/make_paper_tables.py
python scripts/make_seed_report.py
python scripts/make_si_tables.py
python scripts/verify_draft_numbers.py     # every number in the manuscript must be traceable
```

Datasets are public (GC10-DET: Kaggle `alex000kim/gc10det`; NEU-DET: distributed with its original publication).
The twenty baseline checkpoints are in `weights/` — the same models used in the paper, re-saved as minimal
checkpoints (`{"model": …}`) so that no training-time environment metadata (paths, arguments) is published with them.
They originate from the earlier controlled steel-defect study this work extends
(`dmr666/Steel-Defect-Eval`, tag `v1.1-revision`); `results/weights_manifest.csv` additionally records, for every
checkpoint, which archive and member path it was extracted from.

## Measurement conditions (needed to interpret the numbers)

- Windows 11, Python 3.12.13, `torch 2.14.0+cpu`, `ultralytics 8.4.115`, `onnx 1.23.1`, `onnxruntime 1.30.0`,
  `numpy 2.5.3`; CPU `Intel64 Family 6 Model 140`, 8 logical cores.
- Latency: ONNX Runtime CPU execution provider, 4 intra-op threads, one real validation image letterboxed to
  640 and normalised to [0, 1], 10 warm-up + 50 timed single-image inferences, median reported, measured with no
  other CPU-heavy job running.
- All 80 latency rows of the reported table were produced in **one uninterrupted session**, and each row carries a
  `session` identifier; rows from another session (or from an empty id) must not be mixed into the same table. Even
  inside one session the host's effective CPU state drifts: the five seed models of the *same* architecture spanned
  53.7-83.6 ms (GC10-DET, nano) and 66.9-70.0 ms (NEU-DET, nano).
- A **second, warmed-up session** (five minutes of the same inference workload before timing, `PTQ_WARMUP=300`) is
  kept in `results/latency_session_20261006_2119.csv`: it reproduced every within-seed ratio to within 1.4x but moved
  the per-cell medians by up to 22% and had somewhat larger timing spikes (max p95/median 1.87 vs 1.66). Absolute
  milliseconds on this class of host are therefore indicative rather than reproducible digits; ratios are computed
  **within seed** (variant / that seed's FP32) and absolute values are reported as median [IQR] over the five seed
  models. `results/latency_session_compare.md` records the comparison and the adoption rule used.
- **The sign of the CPU INT8 result is hardware-dependent.** Re-measure on your own target host before adopting
  INT8; the paper's latency table is a property of the (model, runtime, ISA) triple used here.
- Static INT8 is produced by the training framework's default path: only `Conv`/`Gemm`/`MatMul` nodes are quantized,
  ONNX Runtime defaults elsewhere, and the **validation split** is used for calibration. The calibration ablation in
  `results/calib_ablation.csv` shows that the choice of calibration data does not explain the accuracy cost.

## Notes for re-users

- Accuracy metrics are reported in two conventions: aggregate/per-class from the training framework, per-size
  (`AP50_s/m/l`) from `pycocotools`, with ground-truth areas expressed in the *640-input equivalent* scale
  (`s = 640 / max(W, H)`, equivalent side `sqrt(w·h)·s`) so that the 32 px / 96 px buckets are meaningful for
  2048 × 1000 images. The two conventions differ by 0.002–0.009 mAP50.
- All comparisons in the paper are **paired by seed** (variant minus FP32 for the same dataset × scale × seed).
- ONNX exports and prediction files are not stored here because they are large and fully regenerable from the
  checkpoints by `ptq_export.py` and `ptq_eval.py`; the per-class AP dumps and all metric files *are* stored.
