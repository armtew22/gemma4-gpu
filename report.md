# Gemma 4 E2B — GPU inference baseline

Working notes and measurements for the Gemma 4 E2B optimization project.
Last updated 2026-10-28.

## Target

`google/gemma-4-E2B-it` — Gemma 4, released 2026-04-02. 2.3B effective / 5.1B total
parameters via Per-Layer Embeddings (PLE), 128K context, multimodal (text/image/audio),
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
two hosts that happened to be similar. The profile pod, the third to produce a comparable
decode number, ran the identical config 1.76x faster.
Cross-host spread is up to **76%**, not 5%.

Caveat: only the batch-1 config was re-verified. The other eight were not, though batch > 1
is structurally less exposed — `generate()` runs until every sequence in the batch finishes.

## Host variance: the same L4 differs 1.76x

Six pods were rented over this project, all requesting `--gpu-id "NVIDIA L4"`, all served
an L4 with 23034 MiB. Drivers were **not** uniform: 595.91.07 on the first pod,
570.195.03 on both 2026-10-13 pods, unrecorded on the rest. Three pods produced a comparable
decode measurement:

| pod | date | purpose | vCPU | CPU | $/hr | decode (batch 1, 128-tok) | TTFT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bcqy10mcg62h4i` | 10-01 | smoke test | 18 ᵃ | not recorded | — | ~~23.7~~ ᶜ | — |
| `jdeeuelkbgiur8` | 10-01 | sweep + bandwidth | not recorded | not recorded | 0.498 | **17.6** | 64.2 ms |
| `x55xbyeb2m1yfe` | 10-11 | early-stop verify | 16 ᵃ | not recorded | 0.59 | **18.4** | 61.5 ms |
| `bi4sgf93eyuu0n` | 10-12 | profile | 48 ᵇ | AMD EPYC 9254 | 0.59 | **32.4** | **39.9 ms** |
| `x73a0805a42pzl` | 10-13 | drift — **lost GPU** | 128 ᵇ | AMD EPYC 7542 | 0.59 | — | — |
| `d6y0tgczjs289n` | 10-13 | drift | 128 ᵇ | AMD EPYC 7542 | 0.59 | — | — |

ᵃ RunPod's allocated `vcpuCount`, from the create response.
ᵇ `nproc` inside the container — **visible** cores, which can be the whole host rather than
  the pod's allocation. 128 cores for a $0.59/hr L4 pod is almost certainly the host.
ᶜ Invalidated: single trial, no warmup, uncontrolled token count. Not comparable.

**The vCPU column mixes two measurements and cannot support a core-count comparison.**
"16 vs 48" compares an allocation against a visible-core count, so it does not establish
that the fast host had 3x the CPU. The CPU *model* (EPYC 9254 is a newer, higher-clocked
part than the 7542) is the more defensible variable, and it was recorded on only one of the
three comparable pods. `hostinfo.py` currently records `os.cpu_count()`, which has the same
visible-core problem — recording the cgroup CPU quota (`/sys/fs/cgroup/cpu.max`) would fix it.

Confirmed not to be a methodology artifact: the 64-token and 32-token configs were both run
on the fast host and agree (32.4 vs 32.6 tok/s), so decode length is not the variable. Same
code, same config, same GPU model — **the host is the variable.**

**Likely mechanism.** A decode step launches ~3085 CUDA kernels (see the profile below).
Every launch is CPU work: Python → PyTorch → CUDA driver. The GPU executes only as fast as
the CPU issues work, so host CPU throughput gates decode at batch 1. On the EPYC 9254 host the
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

`profile_decode.py` on the fast host (AMD EPYC 9254, 48 visible cores, L4), batch 1, 128-token
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

**The `norm` category reads 0 — not because normalization is free, but because eager
RMSNorm never launches a norm kernel.** It decomposes into generic ops the categorizer
files elsewhere. The trace shows exactly **242 calls per step each** of `pow<float>`,
`MeanOps<float>` (a `reduce_kernel`), `add<float>` and `rsqrt`, plus **471**
`direct_copy` kernels that are `LoadWithCast<float>` — RMSNorm upcasting input and weight to
fp32, ~2 per call. So RMSNorm's real cost is spread across "elementwise", "memory copies"
and "other". At ~8 kernels per call, it is roughly **1,900 of the 3,085 launches per step**.

**2031 elementwise kernels per step at ~1.4 µs each** is the finding. 35 layers each firing
~58 microscopic kernels where launch cost dominates execution, plus 471 `direct_copy`
kernels at 2.2 µs. Fusing these removes the 4.93 ms *and* most of the 7.17 ms idle, because
the gaps exist between those launches.

Eliminating all 12.1 ms of addressable time leaves 18.46 ms/step ≈ **54 tok/s**, which lands
on the 57.3 tok/s roofline. The accounting closes.

Full trace in `trace_decode.json` (217 MB, gitignored) — open in ui.perfetto.dev to see the
launch gaps directly. Summary in `profile_decode.json`.

## Output drift evaluation

Speed is worthless if the outputs changed. Before any optimization lands, `eval_drift.py`
establishes how to tell. It scores a fixed evaluation set — wikitext-2-raw-v1 test,
non-overlapping 1024-token windows, BOS prepended — and compares any variant against a
saved bf16 eager reference on two metrics:

- **mean KL(reference ‖ current)**, approximated over the reference's top-100 tokens per
  position with both distributions renormalized over that set. Full 262k-vocab logits for
  50k positions would be ~26 GB, so the reference stores only top-100 ids and log-probs.
- **top-1 agreement** — fraction of positions where the argmax is unchanged.

Perplexity and top-100 coverage are reported alongside. The eval set is pinned by a
sha256 over its tokens *and* window structure; a comparison refuses to run if any setting
differs, since KL across different evaluation sets is meaningless.

Run on an L4, AMD EPYC 7542, 128 vCPU, driver 570.195.03 (2026-10-13).

### BOS is mandatory

| | perplexity (4096 tokens) |
| --- | --- |
| BOS prepended | **212.0** |
| no BOS | **15,713.7** |

A **74x** difference. Gemma 4's tokenizer reports `add_bos_token=False`, and
`add_special_tokens=True` empirically adds none — but that reflects the chat template
supplying BOS, and this harness bypasses the chat template. Without BOS the model is
far off-distribution, which would make every drift measurement describe a state the
model never runs in. BOS is on by default; `--no-bos` exists only to reproduce this.

The initial implementation defaulted BOS *off*, reasoning from the tokenizer config. That
was literally correct and the wrong conclusion. Measured on the first pod run and
reproduced to three decimals on a second.

### The harness is validated in both directions

| check | KL | top-1 | result |
| --- | --- | --- | --- |
| eager vs eager, **separate process** | **0.000e+00** | **100.0000%** | no drift, as required |
| eager, one layer perturbed by 1e-3 | **8.879e-04** | **98.9014%** | drift detected, as required |

**The noise floor is exactly zero.** bf16 eager decode is bitwise deterministic across
processes on this setup. Any nonzero KL from a later variant is attributable entirely to
that variant — e.g. fused kernels summing in a different order under `torch.compile` —
with no baseline noise to subtract.

**The negative control is what makes the zero meaningful.** A check that only confirms
"KL is 0 when nothing changed" would also pass if the comparison were broken and always
returned 0. Perturbing `layers.17.self_attn.q_proj.weight` (2048 × 1536) by gaussian noise
at 1e-3 of its std moves KL clearly off zero and flips 1.1% of top-1 predictions. The
harness can see a change, so its silence means something.

Every variant should be read between these two numbers: 0 is identical, 8.9e-4 / 98.9% is
what a deliberately damaged model looks like.

### Pre-registered pass/fail bands

The eager floor is exactly zero, but zero is the wrong bar for an optimized variant.
`torch.compile` and fused kernels change floating-point summation order, so their bf16
outputs **cannot** be bitwise identical to eager even when they are correct. Demanding KL 0
would fail every legitimate optimization; accepting "small" after the fact would let any
result rationalize itself.

So the verdict thresholds were fixed **before any variant was run**, anchored on the
measured negative control, and are recorded in every `drift_*.json`:

| verdict | rule |
| --- | --- |
| **PASS** | mean KL < 1e-4 **and** top-1 ≥ 99.5% |
| **FAIL** | mean KL ≥ 8.9e-4 **or** top-1 ≤ 98.9% |
| **INVESTIGATE** | anything in between |

`--expect pass-band` exits 0 on PASS, 1 on FAIL, 2 on INVESTIGATE, so it can gate a script.
The existing `no-drift` and `drift` modes are unchanged; they validate the harness, while
`pass-band` judges a variant.

FAIL is the negative control rounded to 8.9e-4 / 98.9%. In that rounding direction the
control itself (8.879e-4 / 98.9014%) classifies as **INVESTIGATE**, by margins of 0.2% on
KL and 0.0014 points on top-1. The eager floor classifies as PASS.

### Perplexity is a poor drift metric

The perturbation that flipped 1.1% of argmax predictions moved perplexity from 212.0098 to
212.0292 — **0.009%**. Tracking perplexity alone, that damage is invisible. KL and top-1
agreement caught it immediately. This is the measured justification for building the
harness around them.

### The top-100 approximation holds at bf16

**Top-100 coverage is 98.9%**: the current model places 98.9% of its full-vocabulary
probability mass on the reference's top-100 tokens, so the approximated KL is describing
nearly all of the real distribution. This is the number to watch under 4-bit quantization —
if coverage falls, the KL figure is measuring less of what changed.

### The 50k reference

`ref_50k_bos.pt`: 48 windows × 1024 tokens = **49,152 scored positions**, top-100,
perplexity **184.56**, 39.9 MB, sha256 `49dab96ff948f95b…`. Gitignored for size; it must be
uploaded to each pod. Ready for the week 3 compile staircase.

### Caveats

- **Absolute perplexity is high** — 184–212 on wikitext-2, where a model this size usually
  lands around 10–30. Likely contributors: this is the instruction-tuned `-it` variant,
  which typically scores worse on raw text than a base model; the 262k vocabulary makes
  per-token perplexity incomparable with ~32k-vocab models; and each window is scored
  independently with no carried-over context. None of this affects drift, which is relative
  to the reference, but **184 should not be cited as a quality number.** Scoring the base
  `gemma-4-E2B` once would isolate the instruction-tuning effect.
- **Cross-pod portability of the reference is likely but unproven.** Two containers
  reproduced the identical eval-set sha256 and perplexity 212.010 to three decimals, which
  is good evidence. Both were EPYC 7542 / 128 vCPU hosts and may have been the same
  physical machine. Until a different host class confirms it, run the `--expect no-drift`
  floor check on any new pod before trusting comparisons against the shared reference.

### Decode mode

`--mode prefill` (the default, unchanged) scores each window with one forward pass. That
never enters `generate()`, so it cannot see the static cache, `torch.compile` or CUDA graphs,
which exist only in `generate()`'s decode loop. `--mode decode` scores through the real
`generate()` with **teacher forcing**: 8 windows × 256 tokens, prompt = BOS + 16 tokens, the
other 240 generated. A logits processor records each step's **raw** distribution (top-100
ids, log-probs, argmax) and only then sets every logit except the true next token to −inf,
so greedy decoding stays on the evaluation text. After each window the generated tokens are
asserted equal to the true ones. Step 0 of each window is produced by the prefill pass, so
only the 239 decode steps per window are scored — 1,912 positions in all.

**The recorder must run first, and transformers puts custom processors last.** With
`min_new_tokens` set, `generate()` builds `MinNewTokensLengthLogitsProcessor`, which sets
Gemma's EOS ids to −inf *before* any user processor sees the logits. A recorder passed the
normal way would record distorted, renormalized log-probs. Decode mode therefore places the
recorder at position 0 for the scoring call (the forward pass is untouched) and asserts it
was first. Verified on a tiny model: placed normally, the recorder saw EOS at −inf; placed
first, it matched an independent full forward pass to 4.8e-07.

**A decode-mode eager reference is the baseline for `static` and `compile`, not the prefill
reference.** Decode uses different kernels (GEMV rather than GEMM, cached attention), so it
will not be bitwise equal to prefill. As a sanity check, `--prefill-reference` compares a
decode run with the prefill reference on the 478 positions whose context is identical
(decode windows 0 and 4, which start exactly where prefill windows start); the expected
result is the PASS band, not 0. Decode references carry their own sha256, covering mode and
prompt length, and comparisons refuse any mode or eval-set mismatch.

Nothing in decode mode has run on a GPU yet. The planned validation, in order: decode-eager
vs decode-eager in a fresh process (expect KL 0, top-1 100%), `--perturb 1e-3` (expect
drift), the prefill sanity check (expect PASS), then `static` and `compile` against the
decode-eager reference for their pass-band verdicts.

## Comparison: local M3 Pro Mac, 4-bit

`ollama run gemma4:e2b --verbose` (Q4_K_M, 5.6 GB on disk).

| Metric | Value |
| --- | --- |
| Decode rate | 49.10 tok/s |
| Prompt eval rate | 217.97 tok/s |
| Load duration | 3.87 s |

The laptop beats the rented GPU by **1.5x–2.8x** at batch 1 depending on host (49.1 vs
32.4–17.6 tok/s). Not a hardware
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
   **Prediction: static cache alone will not reduce the 471 `direct_copy` kernels.** The
   profile identifies them as `LoadWithCast<float>` — RMSNorm upcasting its input and weight
   to fp32, ~2 per call — not cache traffic. (An earlier draft of this plan called them
   "partly cache churn"; that was a guess, and the trace contradicts it.)
2. **`torch.compile` on the decode path**, mode `reduce-overhead` (which enables CUDA
   graphs). Expect the 2031 elementwise kernels to collapse into a handful of fused ones and
   the 7.17 ms idle to shrink with them. This step — or a hand-fused RMSNorm — is what
   should eliminate the `LoadWithCast` kernels. RMSNorm is the single largest target:
   ~242 calls × ~8 kernels ≈ **~1,900 of the 3,085 launches per step**.
3. **Re-profile with `profile_decode.py`** and compare kernel count, busy%, and ms/step.
   Kernel count per step is the cleanest signal that fusion actually happened — if it stays
   near 3085, compile silently fell back to eager.
4. **Then** quantization and the fused dequant-GEMV Triton kernel, against whatever GEMV
   floor remains.

**Measurement discipline:** before and after must run on the same pod in one session. Given
1.76x host variance, a cross-pod comparison could manufacture or hide the entire effect.
Record `nproc` and CPU model with every number — `hostinfo.py` now does this automatically.

Steps 1 and 2 are now `--variant static` and `--variant compile` on `benchmark.py` and
`profile_decode.py` — see [Model variants](#model-variants).

## Week 3 staircase — raw results (2026-10-19)

Raw numbers only; the full writeup is a separate commit. One L4 pod, one session, every step
in a fresh process: AMD EPYC 7542 (128 visible cores), driver 595.91.07, torch 2.8.0+cu128,
transformers pinned to 5.19.0 (`pip -U` resolved 5.20.2; variants were verified against
5.19.0). Batch 1, 128-token prompt, `--max-cache-len 256`. All files in
`runs/2026-10-19_staircase/`.

**Decode scorer validation** (`eval_drift.py --mode decode`, 1,912 positions):

| check | KL | top-1 | result |
| --- | --- | --- | --- |
| eager vs eager, fresh process | 0.000 | 100.0000% | no drift ✓ |
| `--perturb 1e-3` | 8.968e-4 | 99.1632% | drift ✓ (band: FAIL) |
| decode vs prefill ref, 478 identical-context positions | 1.797e-3 | 98.3264% | **FAIL band — expected PASS** |
| prefill-mode eager vs `ref_50k_bos.pt` (49,152 pos., Oct 13 pod) | 0.000 | 100.0000% | prefill ref is portable across pods/drivers ✓ |

**Staircase** — tok/s from `benchmark.py` (64 tokens), the rest from `profile_decode.py`
(32 tokens) and a read-only trace counter; KL/top-1 vs the decode-eager reference:

| variant | tok/s | ms/step | GPU kernels/step | CPU launches/step | GPU busy | KL | top-1 | verdict | graph breaks | recompiles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A eager | 17.58 | 56.85 | 3,085 | 3,084 | 41.3% | — | — | ref | 0 | 0 |
| B static | 16.83 | 58.45 | 3,207 | 3,206 | 40.9% | 1.738e-3 | 98.80% | **FAIL** | 0 | 0 |
| C compile | 17.05 | 58.05 | 1,936 | 1,931 ᵃ | 37.6% | 1.911e-3 | 98.85% | **FAIL** | 1 site ×16 | 17, 2 limit hits |
| A eager (repeat) | 18.56 | 54.82 | 3,085 | 3,084 | 42.9% | — | — | — | 0 | 0 |

ᵃ 1,752 `cudaLaunchKernel` + 179 `cudaGraphLaunch` per step.

- **The two A runs differ by 4–6% in wall time** (17.58 vs 18.56 tok/s) on identical GPU work
  (3,085 kernels, 23.50 ms). B and C fall inside that band: neither is faster than eager.
- **Compile mostly fell back to eager.** One graph break at `modeling_gemma4.py:1677`
  (`kwargs.pop("shared_kv_states", UserDict())`, Gemma 4's KV-sharing state, rebuilt every call)
  and per-layer guards on the static cache's `cumulative_length_int` plus differing rotary
  shapes (256 vs 512) drove both frames to `recompile_limit` (8). What stays compiled is ~179
  one-kernel CUDA graphs per step. `variants.finalize()` still reports `engaged=True`.
- **No compile time reached a timed run**: every recompile and both limit hits finished by
  01:42:42, inside the first warmup generation, and are triggered per decode step.
- **Static and compile FAIL the pre-registered bands** — and so does decode vs prefill on the
  same weights. Three legitimate kernel-path changes all land at KL 1.7–1.9e-3, about 2× the
  1e-3 perturbation that anchors FAIL. The verdicts stand as recorded; whether the bands need
  recalibrating is an open decision, and any change now would be post hoc.
- **This host is CPU-bound**: the same 23.5 ms of GPU work takes 56.9 ms here vs 30.6 ms on
  the Oct 12 EPYC 9254, so the GPU is 41% busy — close to the ~43% predicted for slow hosts.
  It has 128 visible cores, so per-core CPU speed, not core count, is the likely variable.
- Cost ~$0.35–0.44 (one pod, ~44 min, including ~5 min idle after a bug in the stock watcher).

## Drift band recalibration (2026-10-26)

**Why.** The v1 bands were miscalibrated. On 2026-10-19 three changes that do not damage the
model — decode vs prefill on the same weights, the static cache, and `torch.compile` — all
landed at KL 1.7–1.9e-3 and top-1 98.3–98.85%, about twice v1's FAIL anchor. A single q_proj
matrix perturbed by 1e-3 was too gentle to define "broken": it sits *below* ordinary bf16
kernel-path noise.

**Band v2**, now the default. This was decided before testing any new variant.

| verdict | rule | source |
| --- | --- | --- |
| PASS | KL < 2.7e-3 **and** top-1 ≥ 98.0% | benign noise: 1.5 × decode-vs-prefill KL (1.797e-3); floor below its 98.33% |
| FAIL | anchor-derived — **not yet measured** | `--perturb-preset v2-anchor`: 1e-3 noise on all 35 q_proj |
| INVESTIGATE | in between | — |

- **The PASS ceiling uses only decode vs prefill**, the one change benign by construction
  (same weights, two standard eager kernel paths). Static and compile are deliberately
  excluded: they are what the band judges, so letting them set its ceiling would be circular.
- **The FAIL anchor** keeps v1's per-matrix noise but applies it to every layer instead of one.
  1e-2 was considered and rejected: KL grows roughly with the square of the noise, so it would
  likely destroy the model, and a FAIL line anchored there would almost never trigger.
- **Until the anchor is measured, band v2 reports only PASS / NOT-PASS.** It is measured on
  the next pod, before any variant is judged. The rule is fixed now: FAIL KL = anchor KL
  rounded *down* to 2 significant figures, FAIL top-1 = anchor top-1 rounded *up* to 0.1, so
  the anchor itself lands in FAIL. v1 rounded the other way, and its own control read
  INVESTIGATE.

**v1 verdicts stand as recorded.** Band v1 is still computed and written to every JSON, and
recomputing it on all recorded results reproduces them exactly. Applied after the fact,
band v2 would read static, compile and decode-vs-prefill as PASS — and the v1 anchor too,
which is the point. Those v2 readings are post hoc and are not pre-registered evidence about
static or compile.

## Fused RMSNorm (2026-10-28, built, not yet run on a GPU)

RMSNorm is the single largest source of decode launches: 242 calls per step at ~8 eager
kernels each, roughly 1,900 of the 3,085. `fused_rmsnorm` replaces each call with one Triton
kernel (`kernels/rmsnorm.py`). The kernel body is deliberately left as a TODO to be written by
hand; until then it fails at compile time rather than producing output.

**Gemma 4's RMSNorm, read from transformers 5.19.0** — one class, `Gemma4RMSNorm`:

- upcast `x.float()` first; `ms = x.pow(2).mean(-1) + eps`; `y = x * torch.pow(ms, -0.5)`
  (`pow(-0.5)` for JAX parity — it runs as `rsqrt` on CUDA)
- multiply by the **plain** `weight.float()` in fp32 — **not `(1 + weight)`** as in Gemma 1–3;
  the weight is initialised to ones
- cast back with `.type_as(x)` only at the very end; eps is 1e-6 for every text norm
- `v_norm` uses `with_scale=False` and has no weight

The pure-PyTorch reference reproduces the real module **bit for bit** on CPU, in bf16 and
fp32, for every shape below; the `(1 + weight)` form does not match.

| N | weight | text-decoder modules | where |
| --- | --- | --- | --- |
| 1536 | yes | 176 | 5 per layer × 35, plus the final norm |
| 256 | yes | 41 | q_norm (28 sliding layers), k_norm (12), per-layer projection norm (1) |
| 256 | no | 12 | v_norm, sliding layers |
| 512 | yes | 10 | q_norm (7 full layers), k_norm (3) |
| 512 | no | 3 | v_norm, full layers |

That is exactly **242**, matching the profile's 242 calls per step; k/v norms exist only on
the 15 layers that own a KV cache. The cast count closes too: 242 input upcasts plus 227
weight upcasts (242 minus the 15 weightless v_norms) is 469, against 471 `LoadWithCast`
kernels measured. The 222 vision/audio RMSNorms and 2 LayerNorms are left eager — they never
run during text decode.

`fused_rmsnorm` changes every forward pass, so it is **not** generate-only: prefill and decode
drift both apply. At build time on a GPU it checks the kernel against the reference for each
shape and refuses to continue on a mismatch. `tests/test_rmsnorm.py` compares the kernel with
the real module across 76 cases (both dtypes, 1 to 1,000 rows, three non-contiguous layouts)
and microbenchmarks it.

**Next pod plan** — `pod/fused_rmsnorm.sh`, written, not run:

1. Decode-reference floor on the new pod; rebuild `ref_decode_eager.pt` there if it is not
   portable.
2. **Measure the band v2 FAIL anchor** (`--perturb-preset v2-anchor`) in decode and prefill
   mode, and derive FAIL thresholds per mode with the pre-stated rounding rule.
3. `tests/test_rmsnorm.py` — correctness, then the microbenchmark. Nothing measures D unless
   it passes.
4. A eager → D fused_rmsnorm → A eager, each step in a fresh process: benchmark, profile
   (kernels/step, launches/step, busy %), and for D, drift in prefill and decode mode with
   band v2 verdicts.

## Model variants

`variants.py` is the single place every tool builds its model, so a variant name means the
same weights, cache and `generate()` arguments in `benchmark.py`, `profile_decode.py` and
`eval_drift.py`. Each tool takes `--variant` (default `eager`) and `--max-cache-len`, and every
results JSON carries a `variant_meta` block recording what was actually built. Everything
below was read from the installed transformers 5.19.0 source and checked locally on the meta
config; nothing in this section has run on a GPU yet.

| variant | cache | extra `generate()` arguments | compiled |
| --- | --- | --- | --- |
| `eager` | `DynamicCache` (transformers default) | none | no |
| `static` | `StaticCache`, explicit length | `cache_implementation="static"`, `max_cache_len`, `disable_compile=True` | no — explicitly disabled |
| `compile` | `StaticCache`, explicit length | as `static`, plus `compile_config=CompileConfig(mode="reduce-overhead")`, `disable_compile=False` | decode steps only |
| `graph` | — | reserved: manual CUDA graph capture, no compile | not implemented |

### The existing baselines really were eager

Gemma 4's shipped generation config sets `cache_implementation=None`, so every run so far used
the default `DynamicCache`, which transformers never auto-compiles. The `eager` variant is
byte-for-byte the old path: the old and new code were run side by side against an
instrumented model and made an identical `from_pretrained` + `eval()` call and identical
`generate()` arguments at 1 and 64 new tokens. All earlier results remain comparable.

### What transformers does with a static cache

- **The class is always `StaticCache`.** The older `"hybrid"` and `"sliding_window"` cache
  names are deprecated aliases; the sliding/full layout is inferred per layer inside
  `StaticCache`. So the meaningful record is the per-layer composition, not the class name.
- **On CUDA a static cache silently enables compilation.** `generate()` auto-compiles whenever
  the cache is compileable unless `disable_compile=True`. Without that flag, a "static cache"
  measurement would actually be static cache *plus* compile, and attribute the compile win to
  the cache. The `static` variant sets it.
- **Compilation covers decode steps only.** `get_compiled_call` is used inside the decode loop;
  prefill calls the model directly and always runs eager.
- **Static caches only grow within a process.** `generate()` stores the largest cache it has
  built (`_previous_max_cache_length`) and never shrinks it, so a requested length is a floor,
  not a guarantee. `finalize()` records the effective length after the run and flags any
  mismatch.

### Only 15 of the 35 layers own a KV cache

Building the cache from the config gives **15 cache layers, not 35**: 12 sliding-window and 3
full-attention. The config's `num_kv_shared_layers = 20` explains it — the last 20 layers
reuse earlier layers' keys and values instead of storing their own. That is why the KV cache
measured only 2.41 GB at batch 8 × 2048 tokens, and it makes the still-open KV term in the
roofline smaller than a naive 35-layer estimate.

In the static variants the 12 sliding layers cap at their **512-token window** and the 3 full
layers take the requested `max_cache_len`. That is the hybrid behaviour the old `"hybrid"`
cache name described, now recorded exactly in every results file.

### Known gaps before these numbers can be trusted

- **Drift for `static` and `compile` needs `--mode decode`.** Prefill mode scores with one
  plain forward pass and still refuses them, since it would run them as eager and report a
  false PASS. Decode mode (see [Decode mode](#decode-mode)) scores through `generate()`, but
  has not yet run on a GPU.
- **One cache length per sweep.** `benchmark.py` defaults to the largest prompt + decode
  length, rounded up to 128, for every config, so the 128-token config decodes against a
  2176-slot cache in the 3 full-attention layers. Isolate it with
  `--prompt-lens 128 --max-cache-len 256`.
- **`profile_decode.py` may not survive `compile`.** It labels decode steps with Python forward
  hooks; inside a compiled, CUDA-graphed step those can be traced away or force graph breaks.
  If so it fails loudly with "no gen_fwd_* annotations"; the wall-clock measurement is
  unaffected.
- **`fullgraph=False`** is transformers' default and is used as-is, so graph breaks are
  silent. Kernel count per step is the check that compilation actually happened.

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
- `variants.py` — builds every model variant (`eager`, `static`, `compile`) for all three
  tools; `build(variant, max_cache_len)` returns the model, the extra `generate()` arguments,
  and the meta dict, and `finalize()` records what actually happened at run time.
- `hostinfo.py` — host identity recorded into every results file: CPU model, vCPU count, GPU
  name/memory/driver/clock/power, torch and transformers versions.

- `eval_drift.py` — output-drift harness: mean KL over the reference top-100, top-1
  agreement, perplexity and top-k coverage against a saved bf16 eager reference. Supports
  `--expect no-drift|drift` assertions and `--perturb` as a negative control. Variants are
  registered in one `VARIANTS` dict so `static`, `compile` and `fused_rmsnorm` drop in.

Data files: `results.json` (3×3 sweep), `bandwidth.json`, `verify.json`,
`drift_eager_floor.json` and `drift_perturb.json` (harness validation),
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
- **The shipped generation config has `do_sample=True`.** Every `generate()` call here passes
  `do_sample=False` explicitly; one that omitted it would sample, and its output would no
  longer be deterministic.

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
container disk, and the hourly rate varies with the host drawn. Spend to date **$0.35**
across six pods ($20.00 → $19.65 account balance), including one pod that lost its GPU
mid-session and was discarded.

## Status

**Done:** model identified and verified on GPU; pod provisioning and teardown reproducible;
benchmark harness written, hardened against early stopping, and executed; full 3×3 sweep
collected; real memory bandwidth measured; per-token byte accounting completed and the
roofline corrected; decode profiled and the time budget fully attributed; 1.76x host
variance discovered and host identity now recorded in every results file; output-drift
harness built and validated in both directions, with a 50k-token bf16 reference saved.

**Drift baseline:** noise floor KL 0.000, top-1 100%; a deliberately damaged model reads
KL 8.9e-4, top-1 98.9%. Every optimization is judged between those two.

**Baseline:** **17.6–32.4 tok/s** at batch 1 depending on host — not a single number. On the
fast host (EPYC 9254): 32.4 tok/s, 30.56 ms/step, 76.5% GPU busy, 57% of the 57.3 tok/s
roofline. Batch-8 aggregate 138.8 tok/s on the slow host.

**Time budget per decode step (fast host):** 18.46 ms GEMV at 94.6% of achievable bandwidth
(irreducible at bf16), 4.93 ms small kernels, 7.17 ms launch-gap idle. **12.1 ms addressable.**

**Next (week 3):** run `--variant static`, then `--variant compile`, through `benchmark.py`
and `profile_decode.py` on one pod in one session → quantized dequant-GEMV Triton kernel. See
the week 3 plan and Model variants above. Speed is measurable now, and drift for these two
variants is scored with `eval_drift.py --mode decode` against a decode-mode eager reference,
once that mode has been validated on a GPU.

**Still open:** (a) confirm the CPU-starvation hypothesis by profiling a low-core host;
(b) extend the batch sweep to 16/32/64, ~10 GB VRAM unused; (c) add the KV cache term to the
roofline so long-context numbers are honest, now known to cover only 15 cache-owning layers;
(d) the sustained-load clock hypothesis is now largely superseded by host variance as the
better explanation, but remains unmeasured; (e) confirm the drift reference is portable to a
different host class; (f) score the base `gemma-4-E2B` once to explain the high absolute
perplexity; (g) validate decode mode on a GPU — fresh-process floor, perturbation, prefill
sanity check — before trusting any `static` or `compile` drift verdict; (h) check whether `profile_decode.py`'s step labels survive `compile`; (i) decide
whether to run `compile` with `fullgraph=True` so graph breaks fail loudly; (j) record the
cgroup CPU quota in `hostinfo.py` instead of visible cores; (k) measure the band v2 FAIL
anchor (`--perturb-preset v2-anchor`) on the next pod, before judging any new variant;
(l) write the fused RMSNorm kernel body, then run `pod/fused_rmsnorm.sh`.

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
previous week's priority list. Separately, the profile pod ran the identical config **1.76x
faster** than the first two, revealing that "NVIDIA L4" pods differ enormously by host CPU.
Retracted the ~5% noise-floor claim (wrong by an order of magnitude) and added `hostinfo.py`
so CPU model and vCPU count are recorded in every results file from now on. Set the week 3
plan: static KV cache, then CUDA graphs, then the quantized kernel.

*Commit: `5a45118`.*

### 2026-10-13 — drift harness built and validated
Built `eval_drift.py` so every later optimization is judged on output fidelity, not just
speed. Settled the BOS question with data: perplexity **212 with BOS, 15,714 without**, a 74x
gap — overturning an initial default that had followed the tokenizer config literally.
Validated the harness in both directions: a fresh-process eager-vs-eager run gives KL
exactly 0 and 100% top-1, and a 1e-3 perturbation of one layer gives KL 8.9e-4 and 98.9% top-1,
so the zero is meaningful rather than the output of a comparison that cannot fail. Found
that perplexity moved only 0.009% under that same perturbation, which justifies building on
KL and top-1. Saved a 49,152-position bf16 reference for the week 3 staircase. One pod lost its
GPU mid-session (`Failed to initialize NVML`, container restarting); it was discarded rather
than waited on, and its BOS numbers matched the replacement exactly. Later the same day,
corrected stale facts across the report and README (release date, PLE, a six-pod table, the
`LoadWithCast` kernels) and pre-registered the pass/fail bands before any variant was run.
*Commits: `fbf2f20`, `e38614a`, `d316da5`.*

### 2026-10-14 — model variants, no GPU
Built `variants.py` so `eager`, `static` and `compile` are constructed in one place and every
tool measures the identical model. Read the transformers 5.19 source rather than assuming its
behaviour, and found three things that would have quietly skewed week 3: a static cache on
CUDA auto-compiles unless `disable_compile=True`, so a naive "static" run would have measured
compile too; the cache class is always `StaticCache` with the sliding layout inferred per
layer; and static caches never shrink within a process. Confirmed the earlier baselines were
genuinely eager (`cache_implementation=None`, `DynamicCache`) and proved the `eager` variant
identical to the old path by running both against an instrumented model. Found that only 15
of 35 layers own a KV cache (`num_kv_shared_layers = 20`). Made `eval_drift.py` refuse the
generate-only variants rather than score them as eager and report a false PASS. Everything
today was verified on the Mac; the GPU checks are listed under Model variants.
*Commit: `b6e5200`.*

### 2026-10-19 — week 3 staircase run
No L4 stock at first; a watcher took one at 21:19. Validated the decode scorer (floor 0,
perturbation detected), showed the prefill reference reproduces bit-for-bit on a different
pod and driver, then ran eager → static → compile → eager. Neither static nor compile was
faster than eager; compile fell back to mostly eager on a Gemma 4 graph break and two
recompile-limit hits, and both failed the drift bands at the same KL as decode vs prefill.
Raw results only; see Week 3 staircase above.

### 2026-10-26 — drift bands recalibrated, no GPU
Kept band v1 intact and added band v2 as the default: PASS from benign decode-vs-prefill noise,
FAIL from a stronger all-layer perturbation that is measured on the next pod before any
variant is judged. v1 verdicts unchanged.

### 2026-10-28 — fused RMSNorm scaffolding, no GPU
Read Gemma 4's RMSNorm from source — plain weight rather than `(1 + weight)`, fp32
throughout, one cast back at the end — and confirmed the 242 norms per decode step one by one
on the meta model. Built the reference (bit-exact against the real module), the wrapper, the
76-case GPU test file, the `fused_rmsnorm` variant and the next pod plan. The kernel body is
left for hand-writing.

### A note on corrections
Four conclusions have been overturned by measurement: the 23.7 tok/s baseline, the 25.6 tok/s
dense roofline, the ~5% noise floor, and defaulting BOS off. The first three are struck
through in place rather than deleted, with the reason, so the reasoning trail stays
auditable; the BOS default was caught before it reached a commit. Two were analysis errors
from reasoning ahead of the measurement, one was insufficient sampling, and one was trusting
a config file over the model's known behaviour. The pattern holds: every correction came from
measuring something that had previously been assumed.
