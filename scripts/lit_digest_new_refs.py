# -*- coding: utf-8 -*-
r"""把 lit_search_new_refs.py 落盘的结果汇总成可读候选清单(按查询分组,按被引排序,标注类型)。

用法:python scripts/lit_digest_new_refs.py            # 每个查询只列前 8 条
      python scripts/lit_digest_new_refs.py --min 40   # 只看被引 ≥40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OUTD = PROJ / "_verify"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=0)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    for f in sorted(OUTD.glob("litnew_batch*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            hits = [h for h in rec.get("hits", [])
                    if (h.get("cited") or 0) >= a.min and h.get("doi")]
            hits.sort(key=lambda h: -(h.get("cited") or 0))
            print(f"\n=== [{f.name}] {rec['query']}  (命中 {rec.get('count')}) ===")
            for h in hits[: a.top]:
                print(f"  {h['cited']:>6}c {h['year']} {h['type'][:12]:12s} {h['title'][:88]}")
                print(f"         {h['first_author'][:24]:24s} | {str(h['venue'])[:52]} | {h['doi']}")


if __name__ == "__main__":
    main()
