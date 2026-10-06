# -*- coding: utf-8 -*-
r"""论文四 §2 扩写用的文献检索(OpenAlex,免费无 key)。

纪律(来自 skill academic-literature-search):
- 用 `filter=title_and_abstract.search:`(不要用 ?search=,那是全文检索,噪声极大);
- 每个查询之间 sleep,失败最多重试 2 次(短固定退避),socket 超时 40 s;
- 每个查询一落地就追加写盘(JSONL),避免超时丢结果;
- 一次调用最多 6 个查询(batch 参数控制)。

用法:python scripts/lit_search_new_refs.py --batch 1
输出:_verify/litnew_batch<N>.jsonl(逐行 JSON:query + hits[])
"""
from __future__ import annotations

import argparse
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

BATCHES = {
    1: [
        ("real-time object detection industrial surface inspection",
         "cited_by_count:desc"),
        ("deep learning model quantization survey",
         "cited_by_count:desc"),
        ("small object detection survey",
         "cited_by_count:desc"),
        ("quantization aware training object detection",
         "cited_by_count:desc"),
        ("CNN inference latency CPU runtime benchmark",
         "cited_by_count:desc"),
        ("reproducibility variability random seeds deep learning evaluation",
         "cited_by_count:desc"),
    ],
    2: [
        ("edge deployment inference engine comparison ONNX Runtime TensorRT",
         "cited_by_count:desc"),
        ("post-training quantization calibration data",
         "cited_by_count:desc"),
        ("defect detection real-time production line deployment",
         "cited_by_count:desc"),
        ("INT8 quantized inference speedup CPU instruction set",
         "cited_by_count:desc"),
        ("real-time video analytics model compression edge",
         "cited_by_count:desc"),
        ("statistical significance testing machine learning comparison",
         "cited_by_count:desc"),
    ],
    3: [
        ("JRTIP_REALTIME", "cited_by_count:desc"),      # 期刊内检索:Journal of Real-Time Image Processing
        ("real-time image processing industrial inspection system",
         "cited_by_count:desc"),
        ("YOLO lightweight model deployment latency measurement",
         "cited_by_count:desc"),
        ("quantization accuracy degradation detection model",
         "cited_by_count:desc"),
        ("object detection scale distribution small instances benchmark",
         "cited_by_count:desc"),
        ("mixed precision inference accelerator quantization",
         "cited_by_count:desc"),
    ],
}


def fetch(url: str, tries: int = 2) -> dict | None:
    for a in range(tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"    [retry {a}] {type(e).__name__}: {e}")
            time.sleep(5 if a == 0 else 15)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    a = ap.parse_args()
    OUTD.mkdir(exist_ok=True)
    out = OUTD / f"litnew_batch{a.batch}.jsonl"
    for q, sort in BATCHES[a.batch]:
        if q == "JRTIP_REALTIME":
            # JRTIP 刊物内检索(引期刊自家论文对编辑友好);按被引排序看经典
            flt = ('primary_location.source.display_name.search:'
                   'Journal of Real-Time Image Processing')
            url = (f"{API}?filter={urllib.parse.quote(flt)}&sort={sort}"
                   f"&per-page=25&select={SEL}&mailto={MAIL}")
        else:
            flt = f"title_and_abstract.search:{q}"
            url = (f"{API}?filter={urllib.parse.quote(flt)}&sort={sort}"
                   f"&per-page=25&select={SEL}&mailto={MAIL}")
        print(f"  query: {q}")
        d = fetch(url)
        if not d:
            print("    !! 失败(记为空结果,不当成'没有这篇文献')")
            rec = {"query": q, "sort": sort, "hits": [], "error": "no_response"}
        else:
            hits = []
            for w in d.get("results", []):
                loc = (w.get("primary_location") or {}).get("source") or {}
                a1 = ""
                au = w.get("authorships") or []
                if au:
                    a1 = (au[0].get("author") or {}).get("display_name", "")
                hits.append({
                    "title": (w.get("title") or "").strip(),
                    "year": w.get("publication_year"),
                    "cited": w.get("cited_by_count"),
                    "doi": w.get("doi"),
                    "type": w.get("type"),
                    "venue": loc.get("display_name"),
                    "first_author": a1,
                    "n_authors": len(au),
                })
            rec = {"query": q, "sort": sort, "count": d.get("meta", {}).get("count"),
                   "hits": hits}
            print(f"    命中 {rec['count']} 条,取前 {len(hits)}")
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        time.sleep(4)
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
