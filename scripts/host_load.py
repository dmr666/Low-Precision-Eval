# -*- coding: utf-8 -*-
"""主机占用检测:测量必须在"没有别的重活"的会话里做,以前全靠人自觉,现在落成可执行的检查。

用法(库):
    from host_load import sample_load, exclusivity_check
    ok, detail = exclusivity_check(threshold_pct=15.0)

用法(命令行,跑测量前先看一眼):
    python scripts/host_load.py            # 采样 5 秒并打印 top 占用
"""
from __future__ import annotations

import argparse
import statistics
import time

try:
    import psutil
except Exception:      # pragma: no cover
    psutil = None


def sample_load(seconds: float = 5.0, interval: float = 0.25, top: int = 8) -> dict:
    """采样窗口内的平均总 CPU 占用(%)与占用最高的进程。"""
    if psutil is None:
        return {"ok": False, "reason": "psutil 不可用", "total_pct": None, "top": []}
    # 第一次调用是相对进程启动的累计值,丢弃
    psutil.cpu_percent(interval=None)
    for p in psutil.process_iter():
        try:
            p.cpu_percent(None)
        except Exception:
            pass
    time.sleep(0.4)
    samples, agg = [], {}
    t_end = time.time() + seconds
    while time.time() < t_end:
        samples.append(psutil.cpu_percent(interval=interval))
        for p in psutil.process_iter(["pid", "name"]):
            try:
                c = p.cpu_percent(None)
            except Exception:
                continue
            if c:
                key = (p.info["name"], p.info["pid"])
                agg[key] = agg.get(key, 0.0) + c
    total_ncpu = psutil.cpu_count(logical=True) or 1
    tops = sorted(((v / max(1, len(samples)) / total_ncpu, k[1], k[0]) for k, v in agg.items()),
                  reverse=True)[:top]
    return {
        "ok": True,
        "total_pct": round(statistics.fmean(samples), 1) if samples else None,     # 均值
        "median_pct": round(statistics.median(samples), 1) if samples else None,   # 中位数(抗突发)
        "peak_pct": round(max(samples), 1) if samples else None,
        "cpu_count": total_ncpu,
        "top": [{"cpu_pct_of_host": round(v, 1), "pid": pid, "name": nm} for v, pid, nm in tops],
    }


def exclusivity_check(threshold_pct: float = 15.0, seconds: float = 5.0) -> tuple:
    """返回 (是否够干净, 详情)。threshold 指整机平均 CPU 占用上限。

    阈值 15% 的含义:4 核 8 线程的机器上,15% ≈ 1.2 个逻辑核被别的东西吃掉;
    超过它,计时会话就不能算"独占",按项目规则该次测量不得用于对外数字。
    """
    d = sample_load(seconds=seconds)
    if not d.get("ok"):
        return False, d
    # 用中位数判定:后台服务是突发型,单看均值会把正常波动误判成"不干净"
    base = d.get("median_pct") if d.get("median_pct") is not None else d.get("total_pct")
    ok = (base is not None) and (base <= threshold_pct)
    d["threshold_pct"] = threshold_pct
    d["clean"] = ok
    return ok, d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=15.0)
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args()
    ok, d = exclusivity_check(a.threshold, a.seconds)
    print(f"整机平均 CPU 占用 {d.get('total_pct')}%(峰值 {d.get('peak_pct')}%,"
          f"{d.get('cpu_count')} 逻辑核);阈值 {a.threshold}% ⇒ {'干净 ✓' if ok else '不干净 ✗'}")
    print("占用最高的进程:")
    for t in d.get("top", []):
        print(f"  {t['cpu_pct_of_host']:>5.1f}%  {t['name']} (pid {t['pid']})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
