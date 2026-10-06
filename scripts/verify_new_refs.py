# -*- coding: utf-8 -*-
r"""为新引文献做两阶段核验。

阶段 1(--stage find):按题名在 OpenAlex 里定位,打印前 3 个候选(题名/年份/venue/type/DOI/被引),
  供人工确认该条到底是哪一篇(避免"查询成功但命中另一篇"的经典陷阱)。
阶段 2(--stage verify --pins pins.json):对已确认的 (题名, DOI) 做**精确 DOI** 双通道核对
  (OpenAlex works/doi: + Crossref works/{doi}),逐字段比对题名/年份/venue/卷期页,写核验报告。

输出:_verify/ref_verify_new.txt(报告)、out/refs_new.json(机器可读)
用法:python scripts/verify_new_refs.py --stage find
      python scripts/verify_new_refs.py --stage verify
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
VERIFY = PROJ / "_verify"
OUT = PROJ / "out"
MAIL = "694169145@qq.com"

CANDIDATES = [
    # (本地代号, 检索用题名, 预期 venue/说明)
    ("slimneck", "Slim-neck by GSConv a lightweight-design for real-time detector architectures", "JRTIP 2024"),
    ("yolov4_defect", "A real-time and efficient surface defect detection method based on YOLOv4", "JRTIP 2023"),
    ("rtdetr_steel", "Steel surface defect detection based on the lightweight improved RT-DETR algorithm", "JRTIP 2024"),
    ("usv_edge", "Yolo-based power-efficient object detection on edge devices for USVs", "JRTIP 2025"),
    ("quant_survey", "A Comprehensive Survey on Model Quantization for Deep Neural Networks in Image Classification", "ACM TIST 2023"),
    ("gholami", "A Survey of Quantization Methods for Efficient Neural Network Inference", "book chapter 2021"),
    ("krishnamoorthi", "Quantizing deep convolutional networks for efficient inference A whitepaper", "arXiv 2018"),
    ("fqnet", "Fully Quantized Network for Object Detection", "CVPR 2019"),
    ("brecq", "BRECQ Pushing the Limit of Post-Training Quantization by Block Reconstruction", "ICLR 2021"),
    ("smallobj_survey", "Towards Large-Scale Small Object Detection Survey and Benchmarks", "TPAMI 2023"),
    ("variance_bench", "Accounting for Variance in Machine Learning Benchmarks", "MLSys 2021"),
    ("dror_signif", "The Hitchhikers Guide to Testing Statistical Significance in Natural Language Processing", "ACL 2018"),
    ("tail_at_scale", "The Tail at Scale", "CACM 2013"),
    ("mlperf", "MLPerf Inference Benchmark", "ISCA 2020"),
]


def get(url: str, tries: int = 2):
    for a in range(tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"    [retry {a}] {type(e).__name__}: {e}")
            time.sleep(5 if a == 0 else 15)
    return None


def brief(w: dict) -> dict:
    loc = (w.get("primary_location") or {}).get("source") or {}
    au = w.get("authorships") or []
    return {"title": (w.get("title") or "").strip(), "year": w.get("publication_year"),
            "cited": w.get("cited_by_count"), "doi": w.get("doi"), "type": w.get("type"),
            "venue": loc.get("display_name"),
            "first_author": (au[0].get("author") or {}).get("display_name", "") if au else "",
            "n_authors": len(au)}


def stage_find() -> None:
    for code, title, expect in CANDIDATES:
        url = ("https://api.openalex.org/works?filter="
               + urllib.parse.quote(f"title.search:{title}")
               + "&sort=cited_by_count:desc&per-page=3&select=title,publication_year,"
                 "cited_by_count,doi,type,primary_location,authorships&mailto=" + MAIL)
        print(f"\n### {code}  (预期 {expect})")
        d = get(url)
        if not d:
            print("    !! 检索失败")
            continue
        for w in (d.get("results") or [])[:3]:
            b = brief(w)
            print(f"  {b['cited']:>6}c {b['year']} {str(b['type'])[:14]:14s} {b['title'][:82]}")
            print(f"         {b['first_author'][:22]:22s} | {str(b['venue'])[:46]} | {b['doi']}")
        time.sleep(3)


def stage_verify() -> None:
    pins = json.loads((VERIFY / "ref_verify_pins.json").read_text(encoding="utf-8"))
    lines, machine = [], {}
    for code, pin in pins.items():
        title, doi = pin["title"], pin["doi"]
        oa = get(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}?mailto={MAIL}")
        time.sleep(2)
        cr = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
        time.sleep(2)
        rec = {"title_expected": title, "doi": doi}
        if oa:
            rec.update({"openalex": {k: brief(oa)[k] for k in
                                     ("title", "year", "venue", "type", "first_author", "cited")},
                        "openalex_biblio": oa.get("biblio"),
                        "openalex_type": oa.get("type"),
                        "authors_oa": [((a.get("author") or {}).get("display_name") or "")
                                       for a in (oa.get("authorships") or [])]})
        if cr and cr.get("message"):
            m = cr["message"]
            rec["crossref"] = {
                "title": (m.get("title") or [""])[0],
                "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                "venue": (m.get("container-title") or [""])[0],
                "volume": m.get("volume"), "issue": m.get("issue"), "page": m.get("page"),
                "type": m.get("type"),
                "first_author": ((m.get("author") or [{}])[0].get("family") or ""),
                "authors_cr": [(((a.get("family") or "") + " " + (a.get("given") or "")).strip())
                               for a in (m.get("author") or [])],
            }
        machine[code] = rec
        lines.append(f"\n## {code}\n- 预期:{title}\n- DOI:{doi}")
        for ch in ("openalex", "crossref"):
            if ch in rec:
                c = rec[ch]
                lines.append(f"- {ch}: {c.get('title')} | {c.get('year')} | {c.get('venue')} | "
                             f"{(c.get('volume') or '')}{(('(' + str(c['issue']) + ')') if c.get('issue') else '')}"
                             f"{(':' + c['page']) if c.get('page') else ''} | "
                             f"{c.get('first_author')} | type={c.get('type')}"
                             + (f" | cited={c['cited']}" if 'cited' in c else ""))
            else:
                lines.append(f"- {ch}: 查不到(注意:查不到 ≠ 文献是假的,换通道再试)")
    report = "# 论文四 · 新增参考文献核验(OpenAlex 精确 DOI + Crossref 双通道)\n" + "\n".join(lines) + "\n"
    (VERIFY / "ref_verify_new.txt").write_text(report, encoding="utf-8")
    (OUT / "refs_new.json").write_text(json.dumps(machine, ensure_ascii=False, indent=1), encoding="utf-8")
    print(report)
    print(f"报告:{VERIFY / 'ref_verify_new.txt'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["find", "verify"], required=True)
    a = ap.parse_args()
    stage_find() if a.stage == "find" else stage_verify()


if __name__ == "__main__":
    main()
