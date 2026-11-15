#!/bin/bash
# Pod 1 extras, run after pod/quant_naive.sh finishes. Measurement only.
#   1. anchor seed check (seeds 1, 2 at the picked target/scale, both modes) - does not change the pick
#   2. int4_naive with lm_head quantized: benchmark + prefill/decode drift, band v3
#   3. three more benchmark rounds of A Q8 Q4 FL G, order rotated each round
cd /root
PY=/opt/venv/bin/python
export PYTHONUNBUFFERED=1
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a out/plan.log; }
run() {
  local name=$1; shift
  log "START $name: $*"
  "$@" 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%H:%M:%S.%3N)" "$line"; done > "out/$name.log"
  local rc=${PIPESTATUS[0]}; log "END   $name rc=$rc"; return "$rc"
}
BAND=()
if $PY -c "import json,sys; sys.exit(0 if json.load(open('out/anchor_pick.json'))['anchor'] else 1)"; then
  BAND=(--band-v3-anchor out/anchor_pick.json)
  T=$($PY -c "import json; print(json.load(open('out/anchor_pick.json'))['anchor']['target'])")
  S=$($PY -c "import json; print('%g' % json.load(open('out/anchor_pick.json'))['anchor']['scale'])")
  for seed in 1 2; do
    run x1_seed${seed}_decode $PY eval_drift.py --mode decode --variant eager --perturb $S \
      --perturb-target $T --perturb-seed $seed --reference ref_decode_eager.pt \
      --out out/seed_${T}_decode_${S}_s${seed}.json
    run x1_seed${seed}_prefill $PY eval_drift.py --variant eager --perturb $S \
      --perturb-target $T --perturb-seed $seed --reference ref_50k_bos.pt \
      --out out/seed_${T}_prefill_${S}_s${seed}.json
  done
else
  log "no anchor: seed check skipped"
fi
V=int4_naive_lm_head
run x2_bench $PY run_lm_head.py benchmark.py --variant $V --prompt-lens 128 --batch-sizes 1 \
  --decode-tokens 64 --max-cache-len 256 --out out/bench_Q4H_${V}.json
run x2_drift_prefill $PY run_lm_head.py eval_drift.py --variant $V --reference ref_50k_bos.pt \
  "${BAND[@]}" --out out/drift_prefill_Q4H_${V}.json
run x2_drift_decode $PY run_lm_head.py eval_drift.py --mode decode --variant $V \
  --reference ref_decode_eager.pt "${BAND[@]}" --out out/drift_decode_Q4H_${V}.json
ORDER=(A:eager Q8:int8_naive Q4:int4_naive FL:fused_loop G:graph)
for r in 1 2 3; do
  for i in 0 1 2 3 4; do
    pair=${ORDER[$(( (i + r) % 5 ))]}; tag=${pair%%:*}; v=${pair##*:}
    run r${r}_${tag}_bench $PY benchmark.py --variant $v --prompt-lens 128 --batch-sizes 1 \
      --decode-tokens 64 --max-cache-len 256 --out out/round${r}_bench_${tag}_${v}.json
  done
done
log "EXTRA DONE"; touch out/EXTRA_DONE
