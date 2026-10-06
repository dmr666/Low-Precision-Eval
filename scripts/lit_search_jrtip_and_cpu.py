# -*- coding: utf-8 -*-
r"""定向检索(第 4 批):JRTIP 刊内文献(ISSN 过滤,引期刊自家论文)+ CPU 推理/引擎测量 + 尾延迟。

用法:python scripts/lit_search_new_refs.py --batch 4   ← 本文件是 batch 4 的补充实现
输出:_verify/litnew_batch4.jsonl
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OUTD = PROJ / "_verify"
API = "https://api.openalex.org/works"
MAIL = "694169145@qq.com"
SEL = "title,publication_year,cited_by_count,doi,type,primary_location,authorships"

# JRTIP:ISSN 1861-8200(print)/ 1861-8219(electronic)
QUERIES = [
    # JRTIP 刊内:带主题过滤,按被引排序
    ("jrtip", 'locations.source.issn:1861-8219,title_and_abstract.search:real-time object detection',
     "cited_by_count:desc"),
    ("jrtip", 'locations.source.issn:1861-8219,title_and_abstract.search:quantization',
     "cited_by_count:desc"),
    ("jrtip", 'locations.source.issn:1861-8219,title_and_abstract.search:embedded deep learning inference',
     "cited_by_count:desc"),
    ("jrtip", 'locations.source.issn:1861-8219,title_and_abstract.search:defect detection',
     "cited_by_count:desc"),
    # CPU 推理性能评估
    ("", "deep learning inference performance evaluation CPU GPU edge devices",
     "cited_by_count:desc"),
    # 尾延迟 / 实时系统
    ("", "tail latency large scale services",
     "cited_by_count:desc"),
]


def fetch(url: str, tries: int = 2):
    for a in range(tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"    [retry {a}] {type(e).__name__}: {e}")
            time.sleep(5 if a == 0 else 15)
    return None


def main() -> None:
    OUTD.mkdir(exist_ok=True)
    out = OUTD / "litnew_batch4.jsonl"
    for tag, flt, sort in QUERIES:
        url = (f"{API}?filter={urllib.parse.quote(flt)}&sort={sort}"
               f"&per-page=15&select={SEL}&mailto={MAIL}")
        print(f"  query[{tag or 'open'}]: {flt[:80]}")
        d = fetch(url)
        rec = {"query": f"[{tag}] {flt}", "sort": sort, "hits": []}
        if not d:
            rec["error"] = "no_response"
            print("    !! 失败")
        else:
            for w in d.get("results", []):
                loc = (w.get("primary_location") or {}).get("source") or {}
                au = w.get("authorships") or []
                rec["hits"].append({
                    "title": (w.get("title") or "").strip(),
                    "year": w.get("publication_year"),
                    "cited": w.get("cited_by_count"),
                    "doi": w.get("doi"),
                    "type": w.get("type"),
                    "venue": loc.get("display_name"),
                    "first_author": (au[0].get("author") or {}).get("display_name", "") if au else "",
                    "n_authors": len(au),
                })
            rec["count"] = d.get("meta", {}).get("count")
            print(f"    命中 {rec['count']}")
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        time.sleep(4)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
