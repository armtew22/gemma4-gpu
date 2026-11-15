#!/bin/bash
# Naive weight-only quantization session (Q2). Written 2026-11-11, NOT yet run.
#
#   1. pod/anchor.sh + pod/pick_anchor.py: the band v3 FAIL line is fixed BEFORE any quantized
#      model is scored.
#   2. A eager -> Q8 int8_naive -> Q4 int4_naive -> FL fused_loop -> G graph -> A2 eager, each
#      step in a fresh process: benchmark (speed + memory), profile + kernel counts, and drift
#      with band v3 verdicts - prefill and decode for Q8/Q4, decode for FL/G (generate-only).
#
# Settings match the earlier staircases: prompt 128, batch 1, 64 decode tokens, cache 256.
# Expects in /root everything pod/anchor.sh needs, plus benchmark.py, profile_decode.py,
# quant/, graph_decode.py, kernels/, tests/test_rmsnorm.py, runs/2026-10-19_staircase/
# trace_counts.py. Venv: transformers==5.19.0, datasets, pytest.
#
# Pre-stated predictions, so the run can falsify them:
#   - int8_naive and int4_naive are SLOWER than eager: every step dequantizes each whole weight
#     to bf16 and reads it again, so more bytes move per token, not fewer.
#   - Weight memory drops: text-decoder linears 3.75 GB bf16 -> ~1.88 GB int8, ~0.97 GB int4
#     (scripts/quant_bytes.py); whole-model params+buffers fall by about that much.
#   - int8_naive drift lands in or near band v3 PASS.
#   - int4_naive drift lands in band v3 INVESTIGATE or FAIL.
#   - fused_loop sits between fused_rmsnorm and graph in tok/s: G vs FL is the CUDA graph
#     alone, FL vs fused_rmsnorm (2026-11-04) is dropping generate().
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

# 1. The FAIL anchor first (also records hostinfo and checks both references).
bash pod/anchor.sh || stop "anchor ladder failed - no quantized model scored"
[ -f out/ANCHOR_DONE ] || stop "anchor ladder did not finish"
if $PY -c "import json,sys; sys.exit(0 if json.load(open('out/anchor_pick.json'))['anchor'] else 1)"; then
  BAND=(--band-v3-anchor out/anchor_pick.json)
  log "band v3 FAIL line fixed: $($PY -c "import json; a=json.load(open('out/anchor_pick.json'))['anchor']; print(a['target'], a['scale'], 'KL >=', a['fail_kl_at_least'], 'top-1 <=', a['fail_top1_at_most'])")"
else
  BAND=()
  log "NO ANCHOR by the pre-stated rule: drift is recorded raw, with NO band v3 verdicts"
fi

# 2. The fused kernel is in fused_loop and graph: its tests must pass first.
run p2_kernel_tests $PY tests/test_rmsnorm.py || stop "kernel tests failed - nothing measured"

# 3. The staircase.
for pair in A:eager Q8:int8_naive Q4:int4_naive FL:fused_loop G:graph A2:eager; do
  tag=${pair%%:*}; v=${pair##*:}
  run ${tag}_bench $PY benchmark.py --variant $v --prompt-lens 128 --batch-sizes 1 \
    --decode-tokens 64 --max-cache-len 256 --out out/bench_${tag}_${v}.json
  run ${tag}_profile $PY profile_decode.py --variant $v --prompt-len 128 --max-cache-len 256 \
    --trace out/trace_${tag}_${v}.json --out out/profile_${tag}_${v}.json
  [ -f out/trace_${tag}_${v}.json ] && \
    $PY runs/2026-10-19_staircase/trace_counts.py out/trace_${tag}_${v}.json 2 > out/counts_${tag}_${v}.json
  case $v in
    eager) ;;
    int8_naive|int4_naive)
      run ${tag}_drift_prefill $PY eval_drift.py --variant $v --reference ref_50k_bos.pt \
        "${BAND[@]}" --out out/drift_prefill_${tag}_${v}.json
      run ${tag}_drift_decode $PY eval_drift.py --mode decode --variant $v \
        --reference ref_decode_eager.pt "${BAND[@]}" --out out/drift_decode_${tag}_${v}.json ;;
    *)
      run ${tag}_drift_decode $PY eval_drift.py --mode decode --variant $v \
        --reference ref_decode_eager.pt "${BAND[@]}" --out out/drift_decode_${tag}_${v}.json ;;
  esac
done
log "ALL DONE"; touch out/DONE
