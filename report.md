# Gemma 4 E2B — GPU inference baseline

Working notes and measurements for the Gemma 4 E2B optimization project.
Last updated 2026-10-01.

## Target

`google/gemma-4-E2B-it` — Gemma 4, released 2026-07-02. 2.3B effective / 5.1B total
parameters (selective parameter activation), 128K context, multimodal (text/image/audio),
Apache 2.0. **Not gated** — downloads without a Hugging Face token.

Loads via `AutoProcessor` + `AutoModelForMultimodalLM` (transformers 5.x).

## Test rig

RunPod L4 (23034 MiB, driver 595.91.07, EU-RO-1), template `runpod-torch-v280`
(`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`), torch 2.8.0+cu128, Python 3.12.3,
transformers 5.18.0, accelerate 1.15.0. bf16, resident weights 10.21 GB.

## Measured memory bandwidth

STREAM-style probe (`bandwidth.py`, bf16, 2 GB buffers, 20 trials, median):

| Operation | Achieved |
| --- | --- |
| Read (`sum`) | **261.2 GB/s** |
| Copy | 230.1 GB/s |
| Triad | 231.8 GB/s |

**Achieved peak 261.2 GB/s.** The L4's advertised spec is ~300 GB/s, so real achieved is
~87% of nameplate — normal. Use 261.2, not 300, for any roofline arithmetic.

**Dense roofline: 25.6 tok/s** — the decode ceiling if every step reads the full 10.21 GB
weight set (261.2 / 10.21). Measured batch-1 decode is 17.6 tok/s, i.e. **69% of that
ceiling**, so ~31% of the bf16 batch-1 gap is addressable by better kernels. The ceiling
itself can only move by reducing bytes-per-token (quantization).

Note: decode never *exceeded* the dense roofline, so these measurements give no evidence
that selective parameter activation reduces bytes-moved per decode step in this bf16 HF
path. Earlier speculation that it might was not borne out.

## Sweep results

`benchmark.py`, 3 discarded warmups, 10 trials per config, median, `--decode-tokens 64`,
`do_sample=False`. Raw data in `results.json`, console log in `sweep.log`.

| prompt_len | batch | TTFT (ms) | prefill tok/s | decode tok/s/seq | decode tok/s total | peak GB |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 1 | 64.2 | 1992.9 | 17.6 | 17.6 | 10.24 |
| 128 | 4 | 69.2 | 7403.3 | 17.4 | 69.6 | 10.29 |
| 128 | 8 | 110.5 | 9269.5 | 17.3 | 138.8 | 10.37 |
| 512 | 1 | 72.9 | 7027.5 | 17.4 | 17.4 | 10.29 |
| 512 | 4 | 236.0 | 8676.6 | 16.8 | 67.3 | 10.52 |
| 512 | 8 | 503.3 | 8138.8 | 16.6 | 133.2 | 10.82 |
| 2048 | 1 | 270.5 | 7572.1 | 17.1 | 17.1 | 10.52 |
| 2048 | 4 | 1387.1 | 5905.9 | 16.7 | 66.8 | 11.42 |
| 2048 | 8 | 2841.3 | 5766.4 | 16.7 | 133.4 | 12.62 |

No OOM at any config.

### What the sweep shows

**1. Per-sequence decode is flat — 16.6 to 17.6 tok/s across all nine configs.** It barely
responds to prompt length (128 → 2048) or batch size (1 → 8); total spread is under 6%.
This is the signature of bandwidth-bound decode: the cost per step is dominated by reading
weights, which is a fixed cost independent of how many sequences share it.

**2. Batching is close to free.** Aggregate decode scales 3.95x at batch 4 and 7.89x at
batch 8 (vs. a 4x / 8x ideal) — 99% scaling efficiency. Weights are read once per step and
amortized across the batch, so arithmetic intensity rises while bytes-moved stays constant.
**This is the single largest available win and it needs no kernel work.**

**3. There is far more memory headroom than expected.** Peak is 10.24 GB at batch 1 and only
12.62 GB at batch 8 × 2048 — the KV cache costs just 2.41 GB at the largest config, leaving
~10 GB of the 23 GB card unused. Batch 32+ is very likely feasible and untested. The sweep
range (1/4/8) was too conservative.

**4. Prefill is compute-bound and degrades with length.** Prefill peaks at 9269 tok/s
(128 × 8) and falls 38% to 5766 tok/s at 2048 × 8 — consistent with O(n²) attention. TTFT
at 2048 × 8 is 2.84 s, which is the user-visible latency cost of long prompts.

### Discrepancy with the earlier smoke test — resolved against the smoke test

`run.py` reported **23.7 tok/s**; the sweep reports **17.6 tok/s** for the nearest config.
The sweep number is the correct one. The smoke test was unreliable for three reasons:

- single trial, no warmup, run immediately after model load (cold GPU, boost clocks)
- its denominator included prefill while its numerator was however many tokens it emitted
  before EOS — neither quantity was controlled
- it used a chat-templated prompt of ~20 tokens, not a fixed length

**Unverified:** decode also drifts downward *within* the sweep (17.6 on the first config,
16.6–16.7 on later ones), which is consistent with sustained-load clock settling on a 72 W
card — but no clock, power or temperature data was collected, so this is a hypothesis, not
a finding. Sampling `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu` during a
long run would confirm or kill it, at a couple of cents of pod time.

Treat **17.6 tok/s batch-1 / 138.8 tok/s batch-8** as the project baseline. Discard 23.7.

### Early-stopping fix and re-verification

The original harness passed only `max_new_tokens=64` while dividing a hardcoded 63 tokens
by the measured time. `generate()` can stop early at EOS, so any run that ended short would
have been credited with 63 tokens over too little time — **inflating** the reported rate.
Fixed by setting `min_new_tokens=max_new_tokens`, which makes the generated length exact.

Re-ran the batch-1 / 128-token config with the fix (`verify.json`):

| | Original (5.18.0) | Fixed (5.19.0) | Δ |
| --- | --- | --- | --- |
| decode tok/s | 17.6 | **18.4** | +4.5% |
| TTFT | 64.2 ms | 61.5 ms | −4.2% |
| peak memory | 10.24 GB | 10.24 GB | — |

**The bug was not firing.** The fix can only ever *lower* a measured rate — it forces the
full token count, lengthening the denominator. The number went *up* 4.5%, so the original
run was already generating all 64 tokens. The sweep stands.

The +4.5% is therefore run-to-run variance, not a correction. It came from a different
physical host (drawn at $0.59/hr vs $0.498/hr) and transformers 5.19.0 vs 5.18.0, both
uncontrolled.

**Useful consequence: the measurement noise floor across pods is ~5%.** Any future
optimization claiming less than that is indistinguishable from re-rolling the machine.
Optimization results should either clear 5% or be measured on a single pod in one session.

Caveat: only the batch-1 config was re-verified. The other eight were not, though batch > 1
is structurally less exposed — `generate()` runs until every sequence in the batch finishes.

## Comparison: local M3 Pro Mac, 4-bit

`ollama run gemma4:e2b --verbose` (Q4_K_M, 5.6 GB on disk).

| Metric | Value |
| --- | --- |
| Decode rate | 49.10 tok/s |
| Prompt eval rate | 217.97 tok/s |
| Load duration | 3.87 s |

The laptop beats the rented GPU by **2.8x** at batch 1 (49.1 vs 17.6). Not a hardware
result — two confounds, both favoring the Mac:

1. **Precision.** The L4 run is bf16 (10.21 GB read per decode step). Ollama runs Q4_K_M,
   roughly a quarter of the bytes. Decode is bandwidth-bound, so bytes-per-token dominates.
2. **Runtime quality.** Ollama is llama.cpp — fused kernels, quantized matmul, Metal-tuned.
   HF `generate()` in eager mode is the unoptimized reference path.

So this is *optimized 4-bit* vs *naive bf16*. That makes 17.6 tok/s the right baseline for
this project: Ollama is fast because it already did, in C++/Metal, what the CUDA/Triton
phases are meant to do by hand. Note the L4 still wins decisively on aggregate throughput
(138.8 tok/s at batch 8) — the Mac comparison only holds at batch 1.

## Implications for the optimization phases

Ranked by expected payoff, from the data above:

1. **Batching** — already 99% efficient and memory headroom is large. Free throughput, no
   kernel work. Test batch 16/32/64 next.
2. **Quantization** — the only lever that moves the 25.6 tok/s roofline itself, since it
   cuts bytes-per-token. The Mac result is evidence of the size of this effect.
3. **Kernel work (CUDA/Triton)** — bounded by the 31% gap between 17.6 and the 25.6 dense
   roofline at batch 1. Real, but the smallest of the three unless it also enables (2).
4. **Prefill / attention** — separate problem with its own bottleneck; matters for TTFT on
   long prompts, not for decode throughput.

## Benchmark harness

- `run.py` — single-shot smoke test. Superseded for measurement; keep for "does it load".
- `benchmark.py` — the real harness: 3 discarded warmups, 10 trials, median, prefill and
  decode timed separately (per trial: a 1-token generate for TTFT, a full generate, decode
  is the difference), sweeps prompt length × batch size, records OOM instead of crashing.
- `bandwidth.py` — STREAM-style bandwidth probe plus roofline arithmetic.

Prompts are built by slicing a tokenized filler string to an exact token count, which
bypasses the chat template. Deliberate: a perf harness needs exact sequence lengths, and
output text is irrelevant to timing.

```bash
/opt/venv/bin/python benchmark.py                                     # full sweep
/opt/venv/bin/python benchmark.py --batch-sizes 16 32 64              # headroom probe
/opt/venv/bin/python bandwidth.py --decode-tok-s 17.6                 # roofline
```

## Environment notes

Things that cost time to rediscover:

- **The PyTorch template blocks plain `pip`** (PEP 668, externally-managed environment).
  Use a venv that inherits the preinstalled torch, or you re-download ~3 GB of CUDA wheels:
  ```bash
  python3 -m venv --system-site-packages /opt/venv
  /opt/venv/bin/pip install -U transformers accelerate huggingface_hub
  ```
- **Only `/workspace` survives a pod stop**, and only if a network volume is mounted there.
  `/root`, apt packages and `/tmp` are wiped. No volume used so far — setup is ~3 min to
  rebuild, which has been cheaper than paying for idle storage.
- **`runpodctl` v2.14.0 has no `--terminate-after` / `--stop-after` flag** — the cost guard
  in RunPod's own golden-path docs no longer exists. Teardown is manual, every time.
- **A stopped pod still bills for disk.** Only `pod remove` takes the spend rate to zero.
- **Launch long runs detached** (`nohup … &` with output to a log) and poll from a fresh
  connection. An interactive SSH holding the channel open will hang the caller.
- No Hugging Face token needed for this model.

## Reproducing the pod

```bash
runpodctl pod create --name gemma4-bench \
  --template-id runpod-torch-v280 --gpu-id "NVIDIA L4" \
  --container-disk-in-gb 60 --ports "22/tcp" --ssh --wait --wait-timeout 10m

runpodctl ssh info <pod-id>          # ip, external port, key path
runpodctl pod remove <pod-id>        # REQUIRED - stops billing
runpodctl pod list                   # confirm empty
runpodctl user                       # confirm currentSpendPerHr: 0
```

SSH key must be registered on the account **before** the pod boots — keys are injected at
startup. The external TCP port changes on every stop/start, and `ssh info` can return a
stale port right after `start`.

**Cost:** L4 bills at **$0.498/hr** as configured — $0.44 GPU plus ~$0.06 for the 60 GB
container disk. Spend to date **$0.17** across two pods (2026-10-01): $0.05 smoke test,
$0.10 sweep + bandwidth.

## Status

**Done:** model identified and verified on GPU; pod provisioning and teardown reproducible;
benchmark harness written and executed; full 3×3 sweep collected; real memory bandwidth
measured and roofline established; smoke-test number invalidated and replaced.

**Baseline:** 17.6 tok/s decode at batch 1, 138.8 tok/s at batch 8, TTFT 64 ms at 128
tokens, 69% of a 25.6 tok/s dense roofline.

**Next:** (a) extend the batch sweep upward — 10 GB of headroom is unused and batching is
the cheapest win; (b) confirm or kill the clock-throttling hypothesis with `nvidia-smi`
sampling under sustained load; (c) begin the CUDA and Triton phases against the 31% batch-1
gap.
