#!/bin/bash
# Drift FAIL anchor ladder. Written 2026-11-11, NOT yet run. Eager only; every run is its own
# python process. The rule that picks the anchor is fixed in pod/pick_anchor.py and report.md
# ("A FAIL anchor that clears the floor") BEFORE this ladder runs.
#
# Expects in /root: eval_drift.py, variants.py, graph_decode.py, hostinfo.py, kernels/,
# pod/pick_anchor.py, ref_decode_eager.pt, ref_50k_bos.pt. Venv: transformers==5.19.0,
# datasets.
#
# 3 targets x 5 scales x 2 modes = 30 drift runs, plus the reference checks and one repro.
# Pre-stated expectations, so the run can falsify them:
#   - all-q-proj @ 1e-2 decode reproduces 2026-10-28 exactly (3.254e-3 / 98.07%).
#   - o_proj, down_proj and q_proj give similar KL per noise scale (all three feed an RMSNorm;
#     the ~2e-3 floor, not q_norm, explains the flat q_proj ladder). A much steeper o_proj or
#     down_proj curve would revive the q_norm explanation.
#   - all-linears reaches KL 8.1e-3 by 3e-2.
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

# 0. Both references portable to this pod? Rebuild here if not (the ladder is internally
#    consistent either way; only comparisons with earlier pods need the originals).
if ! run p0_decode_floor $PY eval_drift.py --mode decode --variant eager \
     --reference ref_decode_eager.pt --expect no-drift --out out/drift_decode_eager_floor.json; then
  log "decode reference not portable - rebuilding it on this pod"
  run p0_decode_rebuild $PY eval_drift.py --mode decode --variant eager \
      --save-reference ref_decode_eager.pt || stop "could not rebuild the decode reference"
fi
if ! run p0_prefill_floor $PY eval_drift.py --variant eager \
     --reference ref_50k_bos.pt --expect no-drift --out out/drift_prefill_eager_floor.json; then
  log "prefill reference not portable - rebuilding it on this pod"
  run p0_prefill_rebuild $PY eval_drift.py --variant eager \
      --save-reference ref_50k_bos.pt || stop "could not rebuild the prefill reference"
fi

# 1. Repro: one point of the 2026-10-28 q_proj ladder, to tie this pod to that one.
run p1_repro_q_proj $PY eval_drift.py --mode decode --variant eager --perturb 1e-2 \
  --perturb-target all-q-proj --reference ref_decode_eager.pt --out out/repro_all-q-proj_decode_1e-2.json

# 2. The ladder. all-linears first: it is the anchor target, the other two are diagnostics.
#    No band flags: there is no usable FAIL line yet - that is what this run is for.
for t in all-linears all-o-proj all-down-proj; do
  for s in 1e-3 3e-3 1e-2 3e-2 1e-1; do
    run ladder_${t}_decode_$s $PY eval_drift.py --mode decode --variant eager --perturb $s \
      --perturb-target $t --reference ref_decode_eager.pt --out out/ladder_${t}_decode_$s.json
    run ladder_${t}_prefill_$s $PY eval_drift.py --variant eager --perturb $s \
      --perturb-target $t --reference ref_50k_bos.pt --out out/ladder_${t}_prefill_$s.json
  done
done

# 3. Pick the anchor by the pre-stated rule.
run p3_pick $PY pod/pick_anchor.py out
log "ALL DONE"; touch out/DONE
