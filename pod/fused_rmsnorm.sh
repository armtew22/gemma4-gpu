#!/bin/bash
# Next pod session: band v2 FAIL anchor first, then A eager -> D fused_rmsnorm -> A eager.
# Written 2026-10-28, NOT yet run. Every step is its own python process.
#
# Expects in /root: the repo files (variants.py, benchmark.py, profile_decode.py,
# eval_drift.py, hostinfo.py, kernels/, tests/), ref_50k_bos.pt, ref_decode_eager.pt, and
# runs/2026-10-19_staircase/trace_counts.py. Venv: transformers==5.19.0, datasets, pytest.
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

# 0. Is the decode reference from the 2026-10-19 pod portable to this one? (The prefill
#    reference already proved portable.) If not, rebuild it here; every decode verdict uses it.
if ! run p0_decode_floor $PY eval_drift.py --mode decode --variant eager \
     --reference ref_decode_eager.pt --expect no-drift --out out/drift_decode_eager_floor.json; then
  log "decode reference not portable - rebuilding it on this pod"
  run p0_decode_rebuild $PY eval_drift.py --mode decode --variant eager \
      --save-reference ref_decode_eager.pt || stop "could not rebuild the decode reference"
fi

# 1. Band v2 FAIL anchor, in both modes, before any variant is judged.
run p1_anchor_decode $PY eval_drift.py --mode decode --variant eager --perturb-preset v2-anchor \
  --reference ref_decode_eager.pt --out out/drift_anchor_decode.json || stop "anchor (decode) failed"
run p1_anchor_prefill $PY eval_drift.py --variant eager --perturb-preset v2-anchor \
  --reference ref_50k_bos.pt --out out/drift_anchor_prefill.json || stop "anchor (prefill) failed"
# Pre-stated rule (report.md, 2026-10-26): FAIL KL = anchor KL rounded DOWN to 2 significant
# figures, FAIL top-1 = anchor top-1 rounded UP to 0.1, so the anchor itself lands in FAIL.
# Applied per mode, so it holds in each mode.
thresholds() {
  $PY - "$1" <<'EOF'
import json, math, sys
d = json.load(open(sys.argv[1]))
kl, t1 = d["mean_kl"], d["top1_agreement_pct"]
e = math.floor(math.log10(kl)) - 1
print(f"{math.floor(kl / 10**e) * 10**e:.3g} {math.ceil(t1 * 10) / 10:.1f}")
EOF
}
read DEC_KL DEC_T1 < <(thresholds out/drift_anchor_decode.json)
read PRE_KL PRE_T1 < <(thresholds out/drift_anchor_prefill.json)
log "band v2 FAIL: decode KL>=$DEC_KL or top-1<=$DEC_T1 | prefill KL>=$PRE_KL or top-1<=$PRE_T1"
echo "{\"decode\": [$DEC_KL, $DEC_T1], \"prefill\": [$PRE_KL, $PRE_T1]}" > out/band_v2_fail.json

# 2. Kernel correctness. Nothing measures D unless this passes.
run p2_kernel_tests $PY tests/test_rmsnorm.py || stop "kernel tests failed - D not measured"

# 3. A eager -> D fused_rmsnorm -> A eager, each step in a fresh process.
for pair in A:eager D:fused_rmsnorm A2:eager; do
  tag=${pair%%:*}; v=${pair##*:}
  run ${tag}_bench $PY benchmark.py --variant $v --prompt-lens 128 --batch-sizes 1 \
    --decode-tokens 64 --max-cache-len 256 --out out/bench_${tag}_${v}.json
  run ${tag}_profile $PY profile_decode.py --variant $v --prompt-len 128 --max-cache-len 256 \
    --trace out/trace_${tag}_${v}.json --out out/profile_${tag}_${v}.json
  [ -f out/trace_${tag}_${v}.json ] && \
    $PY runs/2026-10-19_staircase/trace_counts.py out/trace_${tag}_${v}.json 2 > out/counts_${tag}_${v}.json
  if [ "$v" != "eager" ]; then
    run ${tag}_drift_prefill $PY eval_drift.py --variant $v --reference ref_50k_bos.pt \
      --expect pass-band --band-v2-fail-kl $PRE_KL --band-v2-fail-top1 $PRE_T1 \
      --out out/drift_prefill_${tag}_${v}.json
    run ${tag}_drift_decode $PY eval_drift.py --mode decode --variant $v --reference ref_decode_eager.pt \
      --expect pass-band --band-v2-fail-kl $DEC_KL --band-v2-fail-top1 $DEC_T1 \
      --out out/drift_decode_${tag}_${v}.json
  fi
done
log "ALL DONE"; touch out/DONE
