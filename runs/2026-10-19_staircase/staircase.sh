#!/bin/bash
# Week 3 staircase driver. Every step is its own python process. Ad-hoc; not in the repo.
cd /root
PY=/opt/venv/bin/python
mkdir -p out
export TORCH_LOGS="graph_breaks,recompiles"
export PYTHONUNBUFFERED=1

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a out/staircase.log; }

run() {  # run <name> <cmd...>   -> out/<name>.log with a timestamp on every line
  local name=$1; shift
  log "START $name: $*"
  "$@" 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%H:%M:%S.%3N)" "$line"; done > "out/$name.log"
  local rc=${PIPESTATUS[0]}
  log "END   $name rc=$rc"
  return "$rc"
}

$PY hostinfo.py > out/hostinfo.json

# ---- 1. validate the decode scorer --------------------------------------------------
run s1a_save $PY eval_drift.py --mode decode --variant eager --save-reference ref_decode_eager.pt \
  || { log "1a FAILED - stopping"; touch out/FAILED; exit 1; }
run s1b_floor $PY eval_drift.py --mode decode --variant eager --reference ref_decode_eager.pt \
  --expect no-drift --out out/drift_decode_eager_floor.json \
  || { log "1b FAILED - stopping"; touch out/FAILED; exit 1; }
run s1c_perturb $PY eval_drift.py --mode decode --variant eager --reference ref_decode_eager.pt \
  --perturb 1e-3 --expect drift --out out/drift_decode_eager_perturb.json
if [ -f ref_50k_bos.pt ]; then
  run s1d_prefill_sanity $PY eval_drift.py --mode decode --variant eager \
    --reference ref_decode_eager.pt --prefill-reference ref_50k_bos.pt \
    --out out/drift_decode_eager_prefill_sanity.json
else
  log "1d SKIPPED - ref_50k_bos.pt not present"
fi

# ---- 2. staircase A B C A ------------------------------------------------------------
for pair in A:eager B:static C:compile A2:eager; do
  tag=${pair%%:*}; v=${pair##*:}
  run ${tag}_bench $PY benchmark.py --variant $v --prompt-lens 128 --batch-sizes 1 \
    --decode-tokens 64 --max-cache-len 256 --out out/bench_${tag}_${v}.json
  run ${tag}_profile $PY profile_decode.py --variant $v --prompt-len 128 --max-cache-len 256 \
    --trace out/trace_${tag}_${v}.json --out out/profile_${tag}_${v}.json
  if [ -f out/trace_${tag}_${v}.json ]; then
    $PY /root/trace_counts.py out/trace_${tag}_${v}.json 2 > out/counts_${tag}_${v}.json
    log "counts ${tag} rc=$?"
  fi
  if [ "$v" != "eager" ]; then
    run ${tag}_drift $PY eval_drift.py --mode decode --variant $v --reference ref_decode_eager.pt \
      --expect pass-band --out out/drift_decode_${tag}_${v}.json
  fi
done
log "ALL DONE"
touch out/DONE
