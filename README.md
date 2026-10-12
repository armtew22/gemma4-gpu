# Gemma 4 E2B inference baseline — NVIDIA L4

Week 1: establish a trustworthy performance baseline for `google/gemma-4-E2B-it` before
attempting any optimization, and determine empirically where the bottleneck actually is.

## Finding

Decode is **memory-bandwidth-bound**, not compute-bound. Generating one token requires
reading the full 10.2 GB bf16 weight set out of VRAM, so throughput is governed by bytes
moved per token rather than arithmetic performed.

| | |
| --- | --- |
| Measured memory bandwidth | **261.2 GB/s** (spec claims ~300) |
| Resident weights | 10.21 GB (bf16) |
| **Bytes genuinely read per decode step** | **4.56 GB** — only 45% of resident |
| **Corrected roofline** | 261.2 / 4.56 = **57.3 tok/s** |
| Measured decode, batch 1 | **17.6–32.4 tok/s**, host-dependent (see below) |
| Batch-8 aggregate | **138.8 tok/s** — 7.89x, 99% scaling efficiency |
| GPU busy during decode | 76.5% — 18.46 ms GEMV, 4.93 ms small kernels, 7.17 ms idle |

The gap between resident and read bytes is the interesting part. `embed_tokens_per_layer`,
the Per-Layer Embedding table that makes the E-series "effective 2B", is (262144, 8960) —
4.7 GB resident, but *indexed*, so one token reads one 17,920-byte row. Charging decode for
the full resident set gives a roofline of 25.6 tok/s and the false conclusion that the
baseline is already at ~70% of achievable. Accounting for it properly puts the baseline at
32%, with roughly **3x of headroom at batch 1 before changing precision at all**.

The byte accounting (`bytes_per_token.py`) validates two ways: resident total comes out at
10.209 GB against 10.209 GB measured by `torch.cuda.max_memory_allocated()` on the GPU, and
the text-decoder bucket holds 2279.76 M params — Google's published "effective parameter"
count for E2B, reproduced independently.

For reference, a 4-bit Ollama build on an M3 Pro laptop reaches 49.1 tok/s — faster than the
bf16 GPU baseline at batch 1.

## Two findings that changed the plan

**The matmuls are already optimal.** Profiling a decode step shows the dominant kernels are
`gemvx` — matrix-*vector*, since batch 1 has no data reuse — running at **94.6% of achievable
bandwidth** (18.46 ms against 17.46 ms of theoretically necessary traffic). A hand-written
bf16 matmul kernel would gain nothing. The addressable time is elsewhere: **2031 tiny
elementwise kernels per step** at ~1.4 µs each, plus **23.5% GPU idle** in launch gaps. That
is a kernel-fusion and CUDA-graphs problem, not a matmul problem.

**Two pods labelled "NVIDIA L4" differ 1.76x.** Identical code and config gave 17.6 tok/s on
a 16-vCPU host and 32.4 tok/s on a 48-vCPU AMD EPYC 9254. At ~3085 kernel launches per decode
step, host CPU throughput gates decode, so absolute numbers are meaningless without the host.
Every results file now records CPU model and vCPU count (`hostinfo.py`), and before/after
comparisons must run on one pod in one session.

## Files

| | |
| --- | --- |
| `report.md` | full methodology, 3x3 sweep results, analysis, environment notes |
| `benchmark.py` | prefill/decode harness — warmup, medians, prompt-length × batch sweep |
| `bandwidth.py` | STREAM-style bandwidth probe + roofline arithmetic |
| `bytes_per_token.py` | meta-device parameter audit → bytes read per step, corrected roofline (no GPU needed) |
| `run.py` | single-shot smoke test (superseded for measurement) |
| `results.json`, `bandwidth.json`, `verify.json`, `bytes_per_token.json`, `sweep.log` | raw measured output |

## Running

Needs an NVIDIA GPU. Tested on a RunPod L4, template `runpod-torch-v280`
(torch 2.8.0+cu128, transformers 5.18.0).

```bash
python -m venv --system-site-packages /opt/venv     # image blocks plain pip (PEP 668)
/opt/venv/bin/pip install -U transformers accelerate huggingface_hub

/opt/venv/bin/python bandwidth.py --decode-tok-s 17.6
/opt/venv/bin/python benchmark.py                   # full sweep -> results.json
```

## Method notes

Measurements use 3 discarded warmup iterations and the median of 10 trials. Prefill and
decode are separated by timing a 1-token generate (TTFT) against a full generate and taking
the difference. Prompts are built by slicing a tokenized filler string to an exact token
count, bypassing the chat template, so sequence length is controlled exactly.

An earlier single-trial smoke test reported 23.7 tok/s; it was invalidated and replaced by
the 17.6 figure. See `report.md` for why, and for one hypothesis (sustained-load clock
behavior) that remains unverified.
