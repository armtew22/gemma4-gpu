"""Prefill/decode benchmark for google/gemma-4-E2B-it across prompt lengths and batch sizes."""

import argparse
import json
import statistics
import time

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E2B-it"
FILLER = (
    "The quick brown fox jumps over the lazy dog while the compiler emits vectorized "
    "instructions for the inner loop and the memory controller prefetches the next cache line. "
)


def build_inputs(tokenizer, prompt_len, batch_size, device):
    """Exact-length token batch. Bypasses the chat template so prompt_len is precise."""
    ids = tokenizer(FILLER * (prompt_len // 10 + 2), return_tensors="pt").input_ids[0]
    if ids.numel() < prompt_len:
        ids = ids.repeat(prompt_len // ids.numel() + 2)
    ids = ids[:prompt_len].unsqueeze(0).repeat(batch_size, 1)
    return {
        "input_ids": ids.to(device),
        "attention_mask": torch.ones_like(ids).to(device),
    }


def timed_generate(model, inputs, max_new_tokens):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def measure(model, inputs, decode_tokens, trials, warmup):
    """One trial = a 1-token run (prefill/TTFT) plus a full run; decode is the difference."""
    for _ in range(warmup):
        timed_generate(model, inputs, decode_tokens)

    ttfts, decode_times = [], []
    for _ in range(trials):
        ttft = timed_generate(model, inputs, 1)
        full = timed_generate(model, inputs, decode_tokens)
        ttfts.append(ttft)
        decode_times.append(full - ttft)
    return ttfts, decode_times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-lens", type=int, nargs="+", default=[128, 512, 2048])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--decode-tokens", type=int, default=64)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", default="results.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - this benchmark needs an NVIDIA GPU.")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16
    )
    model.eval()

    gpu = torch.cuda.get_device_name(0)
    weights_gb = torch.cuda.memory_allocated() / 1e9
    print(f"gpu={gpu}  torch={torch.__version__}  weights={weights_gb:.2f} GB\n")

    rows = []
    for prompt_len in args.prompt_lens:
        for batch_size in args.batch_sizes:
            inputs = build_inputs(processor.tokenizer, prompt_len, batch_size, model.device)
            torch.cuda.reset_peak_memory_stats()
            try:
                ttfts, decode_times = measure(
                    model, inputs, args.decode_tokens, args.trials, args.warmup
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append({"prompt_len": prompt_len, "batch_size": batch_size, "status": "OOM"})
                print(f"prompt={prompt_len:5d} batch={batch_size:2d}  OOM")
                continue

            ttft = statistics.median(ttfts)
            decode_time = statistics.median(decode_times)
            decoded = args.decode_tokens - 1
            row = {
                "prompt_len": prompt_len,
                "batch_size": batch_size,
                "status": "ok",
                "ttft_ms": ttft * 1000,
                "prefill_tok_s": prompt_len * batch_size / ttft,
                "decode_tok_s_per_seq": decoded / decode_time,
                "decode_tok_s_total": decoded * batch_size / decode_time,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                "ttft_ms_min": min(ttfts) * 1000,
                "ttft_ms_max": max(ttfts) * 1000,
            }
            rows.append(row)
            print(
                f"prompt={prompt_len:5d} batch={batch_size:2d}  "
                f"ttft={row['ttft_ms']:7.1f}ms  "
                f"prefill={row['prefill_tok_s']:8.1f} tok/s  "
                f"decode={row['decode_tok_s_per_seq']:6.1f} tok/s/seq  "
                f"{row['decode_tok_s_total']:7.1f} tok/s total  "
                f"peak={row['peak_mem_gb']:5.2f} GB"
            )

    with open(args.out, "w") as f:
        json.dump(
            {
                "model": MODEL_ID,
                "gpu": gpu,
                "torch": torch.__version__,
                "dtype": "bfloat16",
                "weights_gb": weights_gb,
                "decode_tokens": args.decode_tokens,
                "trials": args.trials,
                "warmup": args.warmup,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
