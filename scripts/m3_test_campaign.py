# -*- coding: utf-8 -*-
"""M3:held-out test 分区评测(审稿意见 M3 —— 精度不能只在 val 上、而静态 INT8 恰在 val 上校准)。

为什么不用 `ptq_eval.py --all --split test`:
  它按 fp32 → fp16 → int8dyn → int8static 的固定顺序走,而动态 INT8 在 test 上每格要 ~18 分钟
  (它比 FP32 慢 13 倍),于是最关键、最便宜的静态 INT8 会被排在 20×18 分钟之后。
  本脚本把顺序改成 fp32 → fp16 → int8static → int8dyn:先拿到能支撑结论的三个变体。

每行由 ptq_eval.one() 追加写 out/results_test.csv(含 split 列),中断也保留已完成的行;
test 的 run 目录与逐类转储都带 _test 后缀,不会覆盖 val 的任何产物。

用法: python scripts/m3_test_campaign.py [--imgsz 640]
"""
from __future__ import annotations

import argparse
import time
import traceback

import ptq_common as C
import ptq_eval

ORDER = ["fp32", "fp16", "int8static", "int8dyn"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    cells = C.cells()                       # [(dataset, scale, seed), ...]
    n = len(cells) * len(ORDER)
    print(f"=== M3 评测 {a.split} 分区:{len(cells)} 个 (dataset, scale, seed) × "
          f"{len(ORDER)} 个变体 = {n} 次 ===", flush=True)
    print(f"顺序:{' → '.join(ORDER)}(动态 INT8 垫底,因为它每格要 ~18 min)", flush=True)

    done = 0
    t_start = time.time()
    for prec in ORDER:
        for ds, sc, sd in cells:
            done += 1
            elapsed = (time.time() - t_start) / 60.0
            print(f"\n[{done}/{n}] {ds} {sc} seed{sd} {prec}  (已用 {elapsed:.0f} min)", flush=True)
            try:
                ptq_eval.one(ds, sc, sd, prec, a.imgsz, a.split, False, False)
            except Exception as e:
                print(f"  [ERR] {ds} {sc} seed{sd} {prec}: {type(e).__name__}: {e}", flush=True)
                with open(C.OUT / "errors.log", "a", encoding="utf-8") as f:
                    f.write(f"m3test {ds} {sc} {sd} {prec}\n{traceback.format_exc()}\n")
    print(f"\n完成:{(time.time() - t_start) / 60.0:.0f} min,结果 {C.OUT / 'results_test.csv'}", flush=True)


if __name__ == "__main__":
    main()
