#!/bin/bash
# Pod 2: speed only, on the other host CPU class. Kernel tests, then A Q8 Q4 FL G x 3 rounds of
# benchmark.py (order rotated each round), then one profile + kernel counts per variant.
cd /root
PY=/opt/venv/bin/python
mkdir -p out
export PYTHONUNBUFFERED=1
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a out/plan.log; }
run() {
  local name=$1; shift
  log "START $name: $*"
  "$@" 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%H:%M:%S.%3N)" "$line"; done > "out/$name.log"
  local rc=${PIPESTATUS[0]}; log "END   $name rc=$rc"; return "$rc"
}
stop() { log "STOP: $*"; touch out/FAILED; exit 1; }
$PY hostinfo.py > out/hostinfo.json
run p1_kernel_tests $PY tests/test_rmsnorm.py || stop "kernel tests failed - nothing measured"
ORDER=(A:eager Q8:int8_naive Q4:int4_naive FL:fused_loop G:graph)
for r in 1 2 3; do
  for i in 0 1 2 3 4; do
    pair=${ORDER[$(( (i + r - 1) % 5 ))]}; tag=${pair%%:*}; v=${pair##*:}
    run r${r}_${tag}_bench $PY benchmark.py --variant $v --prompt-lens 128 --batch-sizes 1 \
      --decode-tokens 64 --max-cache-len 256 --out out/round${r}_bench_${tag}_${v}.json
  done
done
for pair in ${ORDER[@]}; do
  tag=${pair%%:*}; v=${pair##*:}
  run ${tag}_profile $PY profile_decode.py --variant $v --prompt-len 128 --max-cache-len 256 \
    --trace out/trace_${tag}_${v}.json --out out/profile_${tag}_${v}.json
  [ -f out/trace_${tag}_${v}.json ] && \
    $PY runs/2026-10-19_staircase/trace_counts.py out/trace_${tag}_${v}.json 2 > out/counts_${tag}_${v}.json
done
log "ALL DONE"; touch out/DONE
