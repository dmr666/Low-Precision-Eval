# -*- coding: utf-8 -*-
r"""解出论文三的基线权重(精确 run 名匹配,不做模糊猜测),并留来源清单。

论文三的权重散在多个 tar / tar.gz 里:
  _results/weights_20261002/neu_main_weights_20261002.tar        (NEU 主矩阵,60 个 best.pt)
  _results/weights_20261002/revision_weights_20261002.tar        (NEU 返修 run)
  _results/weights_gc10_20260930/weights_gc10.tar.gz             (GC10)
  _results/neu_20260929/weights_neu.tar.gz / weights_gc10.tar.gz
  _results/neu_ext_20260930/weights_ext_neu.tar.gz

我们只需要 **基线**:  <dataset>_<scale>_base_seed<seed>.pt   (2 数据集 × 2 尺度 × 3 种子 = 12 个)
匹配用正则,默认只认 run 目录名恰为 `v_base_seed<seed>` 的 best.pt:
  NEU s 尺度  -> defect/runs/v_base_seed42/weights/best.pt
  NEU n 尺度  -> defect/runs_nano/v_base_seed42/weights/best.pt
  GC10 同理(其 run 根目录名以 tar 内路径为准)
其它 tag(处置组)用 --pattern 显式指定,避免误取同名的不同配置。

用法:
  python extract_weights.py --list
  python extract_weights.py --extract                          # 基线 12 个
  python extract_weights.py --extract --pattern "v_full_seed(\d+)" --tag full
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import tarfile
from pathlib import Path

import ptq_common as C

P3 = C.PAPER3
CANDIDATES = [
    P3 / "_results" / "weights_20261002" / "neu_main_weights_20261002.tar",
    P3 / "_results" / "weights_20261002" / "revision_weights_20261002.tar",
    P3 / "_results" / "weights_gc10_20260930" / "weights_gc10.tar.gz",
    P3 / "_results" / "neu_20260929" / "weights_neu.tar.gz",
    P3 / "_results" / "neu_20260929" / "weights_gc10.tar.gz",
    P3 / "_results" / "neu_ext_20260930" / "weights_ext_neu.tar.gz",
]


def archives() -> list:
    out = []
    for p in CANDIDATES:
        if p.exists():
            out.append(p)
        else:
            for alt in p.parent.glob(p.name.split(".")[0] + "*") if p.parent.exists() else []:
                if alt.is_file() and alt not in out:
                    out.append(alt)
    return out


def infer(member: str, tar_name: str):
    """返回 (dataset, scale, tag, seed) 或 None。"""
    low = member.lower()
    m = re.search(r"seed(\d+)", low)
    if not m:
        return None
    seed = int(m.group(1))
    tlow = tar_name.lower()
    if "gc10" in tlow or "gc10" in low:
        dataset = "gc10"
    elif "neu" in tlow or "defect" in low:
        dataset = "neu"
    else:
        return None
    scale = "n" if "runs_nano" in low else "s"
    tag = "base"
    for t in ("full_cpw0p5", "p2dcsawiou", "resample", "focal", "wiou", "dcsa",
              "full", "p2", "p1", "scratch"):
        if f"/v_{t}" in low or f"v_{t}_" in low:
            tag = t
            break
    return dataset, scale, tag, seed


def do_list(pattern: str) -> None:
    ars = archives()
    if not ars:
        print("没找到任何权重归档")
        return
    rx = re.compile(pattern)
    tot = 0
    for a in ars:
        print(f"\n=== {a.name} ({a.stat().st_size / 1e6:.1f} MB) ===")
        n = 0
        with tarfile.open(a) as tf:
            for mem in tf.getmembers():
                if not mem.name.endswith("best.pt"):
                    continue
                n += 1
                if rx.search(mem.name):
                    print(f"  [命中] {mem.name}  ({mem.size / 1e6:.1f} MB) "
                          f"-> {infer(mem.name, a.name)}")
        print(f"  共 {n} 个 best.pt")
        tot += n
    print(f"\n合计 {tot} 个 best.pt;匹配 '{pattern}' 的已用 [命中] 标出")


def do_extract(tags: set, pattern: str, tag_name: str) -> None:
    rx = re.compile(pattern)
    manifest = C.OUT / "weights_manifest.csv"
    new = not manifest.exists()
    got = 0
    for a in archives():
        with tarfile.open(a) as tf:
            for mem in tf.getmembers():
                if not mem.name.endswith("best.pt") or not rx.search(mem.name):
                    continue
                info = infer(mem.name, a.name)
                if not info:
                    print(f"  [skip] 认不出:{mem.name}")
                    continue
                dataset, scale, tag, seed = info
                if tag not in tags:
                    continue
                dest = C.weight_path(dataset, scale, seed, tag)
                if dest.exists() and dest.stat().st_size == mem.size:
                    continue
                with tf.extractfile(mem) as fsrc, open(dest, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                print(f"  [ok] {dest.name}  ({mem.size / 1e6:.1f} MB)  <- {a.name}:{mem.name}")
                with open(manifest, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["file", "dataset", "scale",
                                                      "tag", "seed", "source_tar",
                                                      "member", "bytes"])
                    if new:
                        w.writeheader(); new = False
                    w.writerow({"file": dest.name, "dataset": dataset, "scale": scale,
                                "tag": tag, "seed": seed, "source_tar": a.name,
                                "member": mem.name, "bytes": mem.size})
                got += 1
    print(f"\n解出 {got} 个 -> {C.WEIGHTS}\n来源清单:{manifest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--pattern", default=r"v_base_seed(\d+)/weights/best\.pt$")
    ap.add_argument("--tags", default="base")
    a = ap.parse_args()
    if a.list or not a.extract:
        do_list(a.pattern)
    if a.extract:
        do_extract({s.strip() for s in a.tags.split(",") if s.strip()},
                   a.pattern, a.tags)


if __name__ == "__main__":
    main()
