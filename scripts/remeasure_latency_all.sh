#!/usr/bin/env bash
# 论文四 · 单会话延迟重测(全部 80 个配置)
#
# 为什么:补种子后延迟是分两次会话测的,发现同一架构的 nano 模型跨会话差近一倍
#   (gc10 n FP32:会话A 68–81 ms,会话B 155–159 ms),而比例稳定(11.8–14.8×)。
#   论文的绝对延迟必须来自**同一段不间断、独占 CPU 的会话**,否则表内不可比。
# 做法:延迟 CSV 不删除,按固定顺序把 80 个 key 全部重新测量覆盖(逐行写 session 列);把 4 格子 × 5 种子 × 4 精度全部重测,最后重算统计/图/清单/报告。
# 运行期间**不要在本机跑任何其它作业**(这正是本脚本要保证的前提)。
# 预计约 75 分钟(8 逻辑核,4 线程;int8-dynamic 占大头)。
set -u
cd "." || exit 1
VP="${PYTHON:-python3}"
LOG=_log_latency_all.log
: > "$LOG"

# 会话 id:80 个配置必须落在同一段不间断、独占 CPU 的会话里,否则延迟不可比。
# 半途断掉时表里会留下别的 id 或空值的行 —— 一眼可见,不必再靠"行数够 80"判断是否跑完。
export PTQ_SESSION="sess_$(date +%Y%m%d_%H%M%S)"


say() { echo "$@" | tee -a "$LOG"; }

say "=== 备份旧延迟表 ==="
cp out/latency.csv out/latency_prev_sessions.csv && say "  已备份为 out/latency_prev_sessions.csv(留证:会话差异)"

# ---- 可选预热(PTQ_WARMUP=秒数):先用**同一负载、同一线程数**把机器带到热/频率稳态,再开始计时。
# 动机:同一架构的模型在冷机/刚开机时能快 50%+,绝对毫秒只有在稳态下才可复现。
# 预热阶段写的行用 warmup_* 会话 id;正式全量重测会把 80 个 key 全部覆盖,不会留下残留行。
if [ "${PTQ_WARMUP:-0}" -gt 0 ]; then
  say "=== 预热 ${PTQ_WARMUP}s(真实负载循环:gc10 s 的 4 个精度 × 20 次,torch/ORT 同为 4 线程)==="
  export PTQ_SESSION="warmup_$(date +%Y%m%d_%H%M%S)"
  w0=$(date +%s)
  while [ $(( $(date +%s) - w0 )) -lt "$PTQ_WARMUP" ]; do
    for p in int8dyn fp32 fp16 int8static; do
      "$VP" scripts/ptq_latency.py --dataset gc10 --scale s --seed 42 \
        --precision "$p" --imgsz 640 --n 20 --threads 4 >> "$LOG" 2>&1
    done
    say "  预热第 $(( $(date +%s) - w0 ))s / ${PTQ_WARMUP}s"
  done
  say "  预热结束(实测 $(( $(date +%s) - w0 ))s);正式会话从此刻开始"
  export PTQ_SESSION="sess_$(date +%Y%m%d_%H%M%S)"
fi
say "会话 id = $PTQ_SESSION"

say "=== 单会话重测开始(80 个配置)==="
n=0
for ds in gc10 neu; do
  for sc in n s; do
    for sd in 42 7 2024 13 99; do
      for p in fp32 fp16 int8dyn int8static; do
        n=$((n+1))
        say "[$n/80] $ds $sc seed$sd $p"
        "$VP" scripts/ptq_latency.py --dataset "$ds" --scale "$sc" --seed "$sd" \
          --precision "$p" --imgsz 640 --n 50 --threads 4 >> "$LOG" 2>&1 \
          || say "!! latency FAILED $ds $sc seed$sd $p"
      done
    done
  done
done

say "=== 重算统计 / 图 / 数字清单 / 补种子报告 / 论文表格 ==="
for s in ptq_stats make_fig_pareto make_numbers_sheet make_seed_report make_paper_tables; do
  "$VP" "scripts/$s.py" >> "$LOG" 2>&1 && say "  $s ✓" || say "!! $s FAILED"
done

say "=== 单会话内一致性自检(FP32 逐种子)==="
"$VP" -c "
import csv, statistics
lat=list(csv.DictReader(open('out/latency.csv',encoding='utf-8-sig')))
for ds in ('gc10','neu'):
    for sc in ('n','s'):
        v=[float(r['lat_median_ms']) for r in lat if r['dataset']==ds and r['scale']==sc and r['precision']=='fp32']
        if v:
            print(f'  {ds} {sc} FP32: {min(v):.1f}-{max(v):.1f} ms (5 seeds, spread {(max(v)/min(v)-1)*100:.1f}%)')
for ds in ('gc10','neu'):
    for sc in ('n','s'):
        b=statistics.fmean([float(r['lat_median_ms']) for r in lat if r['dataset']==ds and r['scale']==sc and r['precision']=='fp32'])
        for pr in ('fp16','int8static','int8dyn'):
            x=statistics.fmean([float(r['lat_median_ms']) for r in lat if r['dataset']==ds and r['scale']==sc and r['precision']==pr])
            print(f'  {ds} {sc} {pr:11s} = {x/b:.2f}x FP32')
" >> "$LOG" 2>&1
say "=== 会话完整性自检(防半途中断的假完整)==="
"$VP" -c "
import csv, os
lat=list(csv.DictReader(open('out/latency.csv',encoding='utf-8-sig')))
want=os.environ.get('PTQ_SESSION','')
sess=sorted({(r.get('session') or 'EMPTY') for r in lat})
print('  行数', len(lat), ';本会话', want, ';CSV 内会话', sess)
print('  单会话完整性:', 'PASS' if (len(lat)==80 and sess==[want]) else 'FAIL-这份延迟表不得引用')
" >> "$LOG" 2>&1

say "=== ALL DONE ==="
