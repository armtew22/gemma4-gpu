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
| Measured decode, batch 1 | **18.4 tok/s** — **32% of roofline** |
| Batch-8 aggregate | **138.8 tok/s** — 7.89x, 99% scaling efficiency |

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

For reference, a 4-bit Ollama build on an M3 Pro laptop reaches 49.1 tok/s, 2.7x the bf16
GPU baseline at batch 1.

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
