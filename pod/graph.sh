#!/bin/bash
# Manual CUDA graph session: A eager -> D fused_rmsnorm -> G graph -> GE graph_eager -> A eager.
# Written 2026-11-04, NOT yet run. Every step is its own python process; settings match the
# 2026-10-19 and 2026-10-28 staircases (prompt 128, batch 1, 64 decode tokens, cache 256).
#
# Expects in /root: the repo files (variants.py, graph_decode.py, benchmark.py,
# profile_decode.py, eval_drift.py, hostinfo.py, kernels/, tests/), ref_decode_eager.pt and
# runs/2026-10-19_staircase/trace_counts.py. Venv: transformers==5.19.0, datasets, pytest.
#
# Pre-stated predictions, so the run can falsify them:
#   - graph and graph_eager build: every build runs the self-check (graph replay == eager step
#     == generate()-style eager forward, bit for bit) and refuses on any mismatch.
#   - CPU launches per decode step ~1 (one cudaGraphLaunch) for G and GE, with GPU kernels/step
#     unchanged from D (~1,179) and A/static (~3,207) respectively.
#   - graph_eager decode drift == static's exactly (1.7382e-3 / 98.7971%): same kernels, same
#     cache type. Any difference means the graph or the loop computes something different.
cd /root
PY=/opt/venv/bin/python
mkdir -p out
export PYTHONUNBUFFERED=1
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a out/plan.log; }
run() {  # run <name> <cmd...> -> out/<name>.log, a timestamp on every line
  local name=$1; shift
  log "START $name: $*"
  "$@" 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%H:%M:%S.%3N)" "$line"; done > "out/$name.log"
  local rc=${PIPESTATUS[0]}; log "END   $name rc=$rc"; return "$rc"
}
stop() { log "STOP: $*"; touch out/FAILED; exit 1; }

$PY hostinfo.py > out/hostinfo.json

# 0. Decode reference portable to this pod? Rebuild it here if not.
if ! run p0_decode_floor $PY eval_drift.py --mode decode --variant eager \
     --reference ref_decode_eager.pt --expect no-drift --out out/drift_decode_eager_floor.json; then
  log "decode reference not portable - rebuilding it on this pod"
  run p0_decode_rebuild $PY eval_drift.py --mode decode --variant eager \
      --save-reference ref_decode_eager.pt || stop "could not rebuild the decode reference"
fi

# 1. The graph variant uses the fused kernel: its tests must pass first.
run p1_kernel_tests $PY tests/test_rmsnorm.py || stop "kernel tests failed - graph not measured"

# 2. The staircase. No band verdicts: band v2 has no usable FAIL line (report.md,
#    "The anchor failed"), so drift is recorded as raw KL / top-1 only.
for pair in A:eager D:fused_rmsnorm G:graph GE:graph_eager A2:eager; do
  tag=${pair%%:*}; v=${pair##*:}
  run ${tag}_bench $PY benchmark.py --variant $v --prompt-lens 128 --batch-sizes 1 \
    --decode-tokens 64 --max-cache-len 256 --out out/bench_${tag}_${v}.json
  run ${tag}_profile $PY profile_decode.py --variant $v --prompt-len 128 --max-cache-len 256 \
    --trace out/trace_${tag}_${v}.json --out out/profile_${tag}_${v}.json
  [ -f out/trace_${tag}_${v}.json ] && \
    $PY runs/2026-10-19_staircase/trace_counts.py out/trace_${tag}_${v}.json 2 > out/counts_${tag}_${v}.json
  if [ "$v" != "eager" ]; then
    run ${tag}_drift_decode $PY eval_drift.py --mode decode --variant $v \
      --reference ref_decode_eager.pt --out out/drift_decode_${tag}_${v}.json
  fi
done
log "ALL DONE"; touch out/DONE
