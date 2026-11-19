#!/bin/bash
# Quantization quality sweep. Written 2026-11-18, NOT yet run. Drift only - no speed runs.
#
# Why does naive int8 fail band v3 (2026-11-15: KL 1.24e-2 prefill / 1.15e-2 decode), and does
# a smaller group or a single family pass? All variants are naive (dequantize the whole weight,
# then F.linear); a fused kernel computes the same dequantized values, so drift carries over.
#
# Expects in /root: eval_drift.py, variants.py, graph_decode.py, hostinfo.py, kernels/, quant/,
# tests/test_quant_primitives.py, ref_decode_eager.pt, ref_50k_bos.pt, and anchor_pick.json
# from runs/2026-11-15_quant_naive/pod1/ (the band v3 FAIL line). Venv: transformers==5.19.0,
# datasets, pytest, safetensors.
#
# Pre-stated predictions. Layer-0 relative error (tests/test_quant_primitives.py) on the
# anchor's noise scale: int8 per-row ~1.0e-2, int8g128 ~0.7e-2, int8g32 ~0.57e-2, int4g32
# ~0.10. Read off the all-linears ladder (KL ~ 1.9e-3 + 86 x scale^2 between 3e-3 and 1e-2):
#   - int8_naive reproduces 2026-11-15 exactly (1.2412e-2 / 1.1488e-2): the control.
#   - Smaller groups lower KL: int8 > int8g128 > int8g32, in both modes.
#   - Neither group size reaches PASS: int8g128 decode KL ~6e-3, int8g32 ~4.5e-3 -> both
#     INVESTIGATE (between the 2.7e-3 ceiling and the 1.0e-2 FAIL line).
#   - No single family accounts for most of int8's drift. Each alone sits well below FAIL;
#     mlp (83% of the quantized parameters) is the largest, ple the smallest. The per-family
#     excesses over the ~2e-3 floor roughly add up to int8's.
#   - int4g32 is still FAIL (KL of order 0.5).
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
cp anchor_pick.json out/anchor_pick.json

# 0. Reference floors. NO rebuild: the anchor and int8_naive were scored against these exact
#    references, so a rebuilt one would not be comparable.
run p0_decode_floor $PY eval_drift.py --mode decode --variant eager \
  --reference ref_decode_eager.pt --expect no-drift --out out/drift_decode_eager_floor.json \
  || stop "decode reference not portable to this pod"
run p0_prefill_floor $PY eval_drift.py --variant eager \
  --reference ref_50k_bos.pt --expect no-drift --out out/drift_prefill_eager_floor.json \
  || stop "prefill reference not portable to this pod"

# 1. The primitives the sweep uses - including the grouped int8 bodies - must pass their tests.
run p1_quant_tests $PY -m pytest -q -s tests/test_quant_primitives.py \
  || stop "quantization primitive tests failed - nothing scored"

# 2. The sweep: int8_naive first, as the control.
for v in int8_naive int8g128 int8g32 int8_attn int8_mlp int8_ple int4g32; do
  run ${v}_drift_prefill $PY eval_drift.py --variant $v --reference ref_50k_bos.pt \
    --band-v3-anchor out/anchor_pick.json --out out/drift_prefill_${v}.json
  run ${v}_drift_decode $PY eval_drift.py --mode decode --variant $v \
    --reference ref_decode_eager.pt --band-v3-anchor out/anchor_pick.json \
    --out out/drift_decode_${v}.json
done

# 3. One table: raw KL, top-1, perplexity and the band v3 verdict per variant and mode.
$PY - <<'PYEOF' | tee out/sweep_table.txt
import json, os
print(f"{'variant':12s} {'mode':8s} {'KL':>11s} {'top-1':>9s} {'ppl':>9s}  band v3")
for v in ("int8_naive", "int8g128", "int8g32", "int8_attn", "int8_mlp", "int8_ple", "int4g32"):
    for m in ("prefill", "decode"):
        p = f"out/drift_{m}_{v}.json"
        if not os.path.exists(p):
            print(f"{v:12s} {m:8s} missing"); continue
        d = json.load(open(p))
        print(f"{v:12s} {m:8s} {d['mean_kl']:11.4e} {d['top1_agreement_pct']:8.3f}% "
              f"{d['perplexity']:9.3f}  {d.get('band_v3', {}).get('verdict')}")
PYEOF
log "ALL DONE"; touch out/DONE
