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

~~**Dense roofline: 25.6 tok/s**~~ — **superseded.** This assumed every decode step reads
the full 10.21 GB resident weight set (261.2 / 10.21), which is wrong: 55% of resident
bytes are lookup tables that are indexed, not multiplied. See
[Bytes actually read per decode step](#bytes-actually-read-per-decode-step) for the
corrected figure of **57.3 tok/s**. The 25.6 number is kept here only because the sweep
analysis below was originally written against it.

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

~~**Useful consequence: the measurement noise floor across pods is ~5%.**~~ **Wrong — see
[Host variance](#host-variance-the-same-l4-differs-176x).** That 5% figure generalized from
two hosts that happened to be similar. A third pod ran the identical config 1.76x faster.
Cross-host spread is up to **76%**, not 5%.

Caveat: only the batch-1 config was re-verified. The other eight were not, though batch > 1
is structurally less exposed — `generate()` runs until every sequence in the batch finishes.

## Host variance: the same L4 differs 1.76x

Four pods were rented over this project, all requesting `--gpu-id "NVIDIA L4"`, all served
an L4 with 23034 MiB and the same driver. Decode throughput on identical code:

| pod | vCPU | CPU | $/hr | decode (batch 1, 128-tok prompt) | TTFT |
| --- | --- | --- | --- | --- | --- |
| `jdeeuelkbgiur8` (sweep) | — | not recorded | 0.498 | 17.6 tok/s | 64.2 ms |
| `x55xbyeb2m1yfe` (verify) | 16 | not recorded | 0.59 | 18.4 tok/s | 61.5 ms |
| `bi4sgf93eyuu0n` (profile) | **48** | AMD EPYC 9254 | 0.59 | **32.4 tok/s** | **39.9 ms** |

Confirmed not to be a methodology artifact: the 64-token and 32-token configs were both run
on the fast host and agree (32.4 vs 32.6 tok/s), so decode length is not the variable. Same
code, same config, same GPU model — **the host is the variable.**

**Likely mechanism.** A decode step launches ~3085 CUDA kernels (see the profile below).
Every launch is CPU work: Python → PyTorch → CUDA driver. The GPU executes only as fast as
the CPU issues work, so host CPU throughput gates decode at batch 1. On the 48-vCPU host the
GPU is 76.5% busy; if the slow hosts performed the same 23.39 ms of kernel work in 54.3 ms of
wall time, they sat at ~43% busy — overhead-bound.

**This is a hypothesis, not a measurement.** The CPU model was never recorded on the slow
pods and busy% was never measured on one. Confirming it means running `profile_decode.py` on
a deliberately low-core host and checking whether busy% collapses toward 43%.

### Consequences for methodology

- **No single absolute tok/s is "the baseline."** The honest statement is a range,
  **17.6–32.4 tok/s**, qualified by host.
- **Before/after comparisons must happen on one pod in one session.** A `torch.compile`
  speedup measured across two pods could be entirely the machine — a 1.7x phantom win.
- **Every results file now records host identity** (`hostinfo.py`): CPU model, vCPU count,
  GPU name, memory, driver, max SM clock, power limit, torch and transformers versions.
  This was added precisely because the variance went unnoticed for three pods.
- **What survives unaffected:** the 3×3 sweep was collected on one pod in one session, so
  every *relative* conclusion holds — 7.89x batch-8 scaling, flat per-sequence decode,
  38% prefill degradation, 2.41 GB KV cache at the largest config. Only absolute throughput
  is host-dependent.

## Bytes actually read per decode step

The dense roofline above charged every decode step for all 10.21 GB of resident weights.
That is wrong. `bytes_per_token.py` builds the model on the meta device (no weights loaded,
runs on a laptop in seconds), walks all 1973 parameters and buffers, and sorts each into
one of three buckets by module prefix, failing loudly on anything unmatched.

| bucket | tensors | params | resident | read/step |
| --- | --- | --- | --- | --- |
| `text_decoder` — attention, MLP, norms, `lm_head` | 543 | 2279.76 M | 3.754 GB | **4.560 GB** |
| `lookup_tables` — `embed_tokens`, `embed_tokens_per_layer` | 4 | 2751.46 M | 5.503 GB | 0.000 GB |
| `non_text` — vision/audio towers and projectors | 1426 | 475.73 M | 0.951 GB | 0.000 GB |
| **TOTAL** | 1973 | 5506.95 M | **10.209 GB** | **4.560 GB** |

Classification, for audit:

```
lm_head                                      -> text_decoder
model.language_model                         -> text_decoder
model.language_model.embed_tokens            -> lookup_tables
model.language_model.embed_tokens_per_layer  -> lookup_tables
model.audio_tower / embed_audio              -> non_text
model.vision_tower / embed_vision            -> non_text
```

**Why the lookup tables don't count.** `embed_tokens_per_layer` — the Per-Layer Embedding
table that makes the E-series "effective 2B" — is (262144, 8960): 4.698 GB resident, but it
is *indexed*, so a single token reads one row, 17920 bytes. `embed_tokens` is (262144, 1536)
and likewise reads 3072 bytes per token. Together they are 55% of resident bytes and
essentially 0% of bytes read.

**`lm_head` is tied** to `embed_tokens` (`_tied_weights_keys` confirms it). Its resident
bytes are therefore counted once, under the embedding — which is what makes the total land
on exactly 10.209 GB. But its *read* is charged in full, 0.805 GB, because the vocabulary
projection touches the entire matrix every step.

### Two independent validations

- **Resident total is 10.209 GB** against the 10.209 GB that
  `torch.cuda.max_memory_allocated()` reported on the pod. Exact match.
- **`text_decoder` holds 2279.76 M params ≈ 2.3B**, which is precisely Google's published
  "effective parameter" count for E2B. The bucketing reproduces their figure independently.

### Corrected roofline

| | |
| --- | --- |
| bytes read per decode step | **4.560 GB** (44.7% of resident) |
| bandwidth | 261.2 GB/s |
| **corrected roofline** | **57.3 tok/s** |
| measured (batch 1) | 18.4 tok/s = **32.1% of corrected roofline** |
| old dense roofline | 25.6 tok/s — wrong, assumed full resident read |

**This overturns the earlier conclusion.** The baseline is not at 69–72% of ceiling with
~1.4x of headroom; it is at **32%, with roughly 3x available**. Kernel work is a far more
attractive target than the dense roofline implied, because the implementation is leaving
about two thirds of achievable bandwidth unused.

**Caveat:** KV cache reads are excluded. At 128-token prompts they are small relative to
4.56 GB of weights, but they scale with context and batch, so 57.3 tok/s is optimistic for
long sequences. Quantifying the KV term is the obvious next refinement.

## Where the decode time actually goes

`profile_decode.py` on the fast host (AMD EPYC 9254, 48 vCPU, L4), batch 1, 128-token
prompt, 32 decode tokens. Wall clock measured with the profiler detached (10 trials,
median); kernel time measured under `torch.profiler` in a separate run, so profiler
overhead does not leak into the busy% verdict.

```
wall clock               30.56 ms/step   (32.7 tok/s)
CUDA kernels launched      3085 per step
GPU kernel time          23.39 ms
GPU busy                   76.5 %
GPU idle                   23.5 %
```

| category | ms/step | share of kernel time | kernels/step |
| --- | --- | --- | --- |
| matmul/gemm | 18.464 | 78.9% | 277 |
| elementwise/activation | 2.857 | 12.2% | **2031** |
| memory copies | 1.038 | 4.4% | 472 |
| other | 0.590 | 2.5% | 242 |
| attention | 0.438 | 1.9% | 63 |

### The matmuls are already near-optimal

The top two kernels are both `internal::gemvx::kernel` — **GEMV**, matrix-*vector*, because
batch 1 degenerates every matmul into a vector product with no data reuse. Together they
account for 18.3 of the 18.46 ms.

Against the byte accounting: 4.56 GB / 261.2 GB/s = **17.46 ms of necessary memory
traffic**, versus 18.46 ms actually spent. The GEMV kernels run at **94.6% of achievable
bandwidth.**

**So a hand-written Triton matmul gains approximately nothing at bf16 batch 1.** cuBLAS has
already won that fight. The earlier claim that "kernel work is the largest lever" was right
that headroom exists and wrong about where it is.

### The real budget

| | ms | share of wall | addressable |
| --- | --- | --- | --- |
| GEMV — memory traffic | 18.46 | 60% | **No** — already 95% of bandwidth |
| other kernels | 4.93 | 16% | Yes — kernel fusion |
| GPU idle — launch gaps | 7.17 | 24% | Yes — CUDA graphs |

**2031 elementwise kernels per step at ~1.4 µs each** is the finding. 35 layers each firing
~58 microscopic kernels where launch cost dominates execution, plus 471 `direct_copy`
kernels at 2.2 µs. Fusing these removes the 4.93 ms *and* most of the 7.17 ms idle, because
the gaps exist between those launches.

Eliminating all 12.1 ms of addressable time leaves 18.46 ms/step ≈ **54 tok/s**, which lands
on the 57.3 tok/s roofline. The accounting closes.

Full trace in `trace_decode.json` (217 MB, gitignored) — open in ui.perfetto.dev to see the
launch gaps directly. Summary in `profile_decode.json`.

## Comparison: local M3 Pro Mac, 4-bit

`ollama run gemma4:e2b --verbose` (Q4_K_M, 5.6 GB on disk).

| Metric | Value |
| --- | --- |
| Decode rate | 49.10 tok/s |
| Prompt eval rate | 217.97 tok/s |
| Load duration | 3.87 s |

The laptop beats the rented GPU by **2.8x** at batch 1 (49.1 vs 17.6). Not a hardware
result — two confounds, both favoring the Mac:

1. **Precision.** The L4 run is bf16 (4.56 GB read per decode step). Ollama runs Q4_K_M,
   roughly a quarter of the bytes. Decode is bandwidth-bound, so bytes-per-token dominates.
2. **Runtime quality.** Ollama is llama.cpp — fused kernels, quantized matmul, Metal-tuned.
   HF `generate()` in eager mode is the unoptimized reference path.

So this is *optimized 4-bit* vs *naive bf16*. That makes 17.6 tok/s the right baseline for
this project: Ollama is fast because it already did, in C++/Metal, what the CUDA/Triton
phases are meant to do by hand. Note the L4 still wins decisively on aggregate throughput
(138.8 tok/s at batch 8) — the Mac comparison only holds at batch 1.

## Implications for the optimization phases

Re-ranked against the profile. Both earlier versions of this list were wrong: the dense
roofline under-rated kernel work, and the corrected roofline over-rated *matmul* kernel work
specifically. The profile settles it.

1. **Kernel fusion + CUDA graphs (`torch.compile`)** — the 12.1 ms of addressable time is
   4.93 ms of tiny non-GEMV kernels plus 7.17 ms of launch-gap idle. Best case ~54 tok/s on
   the fast host, and proportionally far more on a CPU-starved host where idle dominates.
   Largest win, and it requires no hand-written kernel.
2. **Quantization + a fused dequantize-GEMV Triton kernel** — the only way to move the
   18.46 ms GEMV floor, since that floor *is* memory traffic running at 95% efficiency.
   This is where hand-written kernel work genuinely pays, and it stacks on top of (1).
   Note the prize is smaller than the dense roofline implied: only the 4.56 GB read shrinks,
   and the 4.7 GB PLE table is already effectively free, so 4-bit does not buy a clean 4x.
3. **Batching** — 99% efficient already with ~10 GB VRAM unused. Free aggregate throughput;
   test 16/32/64. Improves aggregate, not per-sequence latency. Also raises arithmetic
   intensity, which moves GEMV toward GEMM and changes the picture in (2).
4. **Prefill / attention** — separate bottleneck, compute-bound, degrades 38% from 128→2048.
   Matters for TTFT, not decode throughput.

**What not to do:** write a bf16 matmul kernel for batch-1 decode. It is already at 94.6% of
achievable bandwidth and there is nothing there to win.

## Week 3 plan: static KV cache + CUDA graphs

Target the 12.1 ms of addressable time, in this order:

1. **`cache_implementation="static"` on `generate()`.** Non-negotiable prerequisite, not an
   optimization in itself. The default cache grows by one token per step, so shapes change
   every step and CUDA graphs cannot be captured — graphs require fixed shapes. Skipping
   this would produce little speedup and the wrong conclusion that compile does not help.
   The 471 `direct_copy` kernels per step are partly cache churn and should drop here.
2. **`torch.compile` on the decode path**, mode `reduce-overhead` (which enables CUDA
   graphs). Expect the 2031 elementwise kernels to collapse into a handful of fused ones and
   the 7.17 ms idle to shrink with them.
3. **Re-profile with `profile_decode.py`** and compare kernel count, busy%, and ms/step.
   Kernel count per step is the cleanest signal that fusion actually happened — if it stays
   near 3085, compile silently fell back to eager.
4. **Then** quantization and the fused dequant-GEMV Triton kernel, against whatever GEMV
   floor remains.

**Measurement discipline:** before and after must run on the same pod in one session. Given
1.76x host variance, a cross-pod comparison could manufacture or hide the entire effect.
Record `nproc` and CPU model with every number — `hostinfo.py` now does this automatically.

## Benchmark harness

- `run.py` — single-shot smoke test. Superseded for measurement; keep for "does it load".
- `benchmark.py` — the real harness: 3 discarded warmups, 10 trials, median, prefill and
  decode timed separately (per trial: a 1-token generate for TTFT, a full generate, decode
  is the difference), sweeps prompt length × batch size, records OOM instead of crashing.
- `bandwidth.py` — STREAM-style bandwidth probe plus roofline arithmetic.
- `bytes_per_token.py` — meta-device parameter audit: buckets every tensor and computes
  bytes genuinely read per decode step, then the corrected roofline. Needs no GPU and no
  weights; runs locally in seconds. Fails loudly on any unclassified tensor.
- `profile_decode.py` — per-decode-step kernel profile. Wall clock is measured with the
  profiler detached and kernel time under it, so profiler overhead stays out of busy%.
  Kernels are attributed to forward passes through trace correlation ids, not timestamps:
  kernels run asynchronously and do not fall inside the CPU-side annotation windows, but the
  `cudaLaunchKernel` that issued each one does. Forward passes are labelled with hooks rather
  than by wrapping `model.forward`, because transformers inspects the forward signature in
  `_validate_model_kwargs` and a wrapper hides it, making `generate()` reject
  `attention_mask`.
- `hostinfo.py` — host identity recorded into every results file: CPU model, vCPU count, GPU
  name/memory/driver/clock/power, torch and transformers versions.

Data files: `results.json` (3×3 sweep), `bandwidth.json`, `verify.json`,
`bytes_per_token.json`, `profile_decode.json`, and
`host_epyc48_{64,32}tok.json` — the two runs on the fast host that established host variance.
Those two predate `hostinfo.py`, so their host is recorded in the filename instead.

`timed_generate` returns the number of tokens generated and asserts it equals
`max_new_tokens` on every call, so early stopping cannot silently corrupt a run. Decode
token counts are derived from that measurement rather than assumed.

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

**Cost:** L4 bills **$0.50–0.59/hr** as configured — the GPU plus ~$0.06 for the 60 GB
container disk, and the hourly rate varies with the host drawn. Spend to date **~$0.26**
across three pods: $0.05 smoke test, $0.10 sweep + bandwidth, ~$0.08 verification run.

## Status

**Done:** model identified and verified on GPU; pod provisioning and teardown reproducible;
benchmark harness written, hardened against early stopping, and executed; full 3×3 sweep
collected; real memory bandwidth measured; per-token byte accounting completed and the
roofline corrected; decode profiled and the time budget fully attributed; 1.76x host
variance discovered and host identity now recorded in every results file.

**Baseline:** **17.6–32.4 tok/s** at batch 1 depending on host — not a single number. On the
fast host (48 vCPU EPYC): 32.4 tok/s, 30.56 ms/step, 76.5% GPU busy, 57% of the 57.3 tok/s
roofline. Batch-8 aggregate 138.8 tok/s on the slow host.

**Time budget per decode step (fast host):** 18.46 ms GEMV at 94.6% of achievable bandwidth
(irreducible at bf16), 4.93 ms small kernels, 7.17 ms launch-gap idle. **12.1 ms addressable.**

**Next (week 3):** static KV cache → `torch.compile` with CUDA graphs → re-profile →
quantized dequant-GEMV Triton kernel. See the week 3 plan above.

**Still open:** (a) confirm the CPU-starvation hypothesis by profiling a low-core host;
(b) extend the batch sweep to 16/32/64, ~10 GB VRAM unused; (c) add the KV cache term to the
roofline so long-context numbers are honest; (d) the sustained-load clock hypothesis is now
largely superseded by host variance as the better explanation, but remains unmeasured.

**Measurement standard:** host variance reaches 1.76x, so absolute throughput is meaningless
without the host. Every before/after must run on one pod in one session. Every results file
records CPU model and vCPU count via `hostinfo.py`.

## Worklog

What changed each day and why. `git log` has the diffs; this is the narrative.

### 2026-10-09 — baseline established
Created the repo and pushed the first measured baseline: 3×3 sweep (prompt length × batch
size), measured 261.2 GB/s bandwidth, 25.6 tok/s dense roofline, 17.6 tok/s at batch 1.
Invalidated an earlier 23.7 tok/s smoke-test figure that had no warmup and an uncontrolled
token count. *Commit: `744033a`.*

### 2026-10-11 — harness hardened, roofline corrected
Found and fixed a latent early-stopping bug: `generate()` could stop at EOS while the
harness divided a hardcoded 63 tokens by the measured time, inflating throughput. Re-ran and
**proved it had never fired** — the fix can only lower a rate, and the number rose to 18.4,
so the original run had produced all 64 tokens. Then audited bytes actually read per decode
step on the meta device and found the dense roofline was wrong: 55% of resident bytes are
lookup tables that are indexed, not multiplied. `embed_tokens_per_layer` is 4.7 GB resident
but 17,920 bytes read per token. Corrected roofline 25.6 → **57.3 tok/s**.
*Commits: `00a0522`, `d92eb30`, `33de167`, `64601a8`.*

### 2026-10-12 — decode profiled, host variance found
Profiled a decode step and attributed the full time budget. Two findings, one expected and
one not. Expected: 2031 tiny elementwise kernels per step and 23.5% GPU idle — fusion and
CUDA graphs are the lever. **Unexpected: the GEMV matmuls run at 94.6% of achievable
bandwidth**, so a hand-written bf16 matmul kernel would gain nothing, which inverted the
previous week's priority list. Separately, a third pod ran the identical config **1.76x
faster** than the first two, revealing that "NVIDIA L4" pods differ enormously by host CPU.
Retracted the ~5% noise-floor claim (wrong by an order of magnitude) and added `hostinfo.py`
so CPU model and vCPU count are recorded in every results file from now on. Set the week 3
plan: static KV cache, then CUDA graphs, then the quantized kernel.

### A note on corrections
Three claims in this report have been retracted: the 23.7 tok/s baseline, the 25.6 tok/s
dense roofline, and the ~5% noise floor. Each is struck through in place rather than deleted,
with the reason, so the reasoning trail stays auditable. Two were analysis errors from
reasoning ahead of the measurement; one was insufficient sampling. The pattern is worth
noting: every correction came from measuring something that had previously been assumed.
