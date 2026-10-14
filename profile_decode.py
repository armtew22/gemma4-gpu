"""Where does the time go in one batch-1 decode step of google/gemma-4-E2B-it (bf16)?

Answers one question: is decode memory/compute-bound (GPU busy inside big matmuls) or
overhead-bound (GPU idle between small kernels while Python launches them)?

Two measurements, deliberately separate:
  1. Wall clock with no profiler attached, since profiling inflates step time.
  2. Kernel accounting under torch.profiler, attributed per decode step.

GPU busy% compares (2) against (1), so the profiler's own overhead does not leak into the
verdict.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoProcessor

from benchmark import MODEL_ID, build_inputs, timed_generate
from hostinfo import print_host
from variants import VARIANTS, build, finalize, resolve_max_cache_len

# Checked in order; first match wins. Deliberately specific - the top-15 kernel listing
# printed at the end is the audit for these rules.
CATEGORY_RULES = [
    ("memory copies", r"memcpy|memset|copy_device|direct_copy"),
    ("attention", r"attn|attention|flash|fmha|sdpa|scaled_dot"),
    ("matmul/gemm", r"gemm|gemv|cutlass|cublas|matmul|tensorop|s\d{4,}|wgrad|dgrad"),
    ("norm", r"norm|rms"),
    ("elementwise/activation", r"elementwise|vectorized|gelu|silu|swish|relu|softmax|"
                               r"activation|mul|add|cat|index|gather|embedding"),
]


def categorize(name):
    low = name.lower()
    for label, pattern in CATEGORY_RULES:
        if re.search(pattern, low):
            return label
    return "other"


def label_forward_passes(model, counter):
    """Label every generate() forward pass so prefill and each decode step are separable.

    Uses hooks rather than wrapping model.forward: transformers inspects the forward
    signature in _validate_model_kwargs, and a wrapper hides it, which makes generate()
    reject attention_mask.
    """
    open_ctx = []

    def pre_hook(module, args, kwargs):
        ctx = record_function(f"gen_fwd_{counter['n']:03d}")
        counter["n"] += 1
        ctx.__enter__()
        open_ctx.append(ctx)

    def post_hook(module, args, kwargs, output):
        open_ctx.pop().__exit__(None, None, None)
        return output

    return [
        model.register_forward_pre_hook(pre_hook, with_kwargs=True),
        model.register_forward_hook(post_hook, with_kwargs=True),
    ]


def analyze_trace(path, skip_decode_steps):
    """Attribute CUDA kernels to forward passes via trace correlation IDs.

    Kernel events sit on the GPU timeline and do not overlap the CPU-side annotation
    windows, so they cannot be matched by timestamp. Each kernel carries a correlation id
    shared with the cudaLaunchKernel runtime call that issued it, and that runtime call
    *is* on the CPU timeline inside the annotation window.
    """
    with open(path) as f:
        events = json.load(f)["traceEvents"]

    windows = {}
    kernels_by_corr = defaultdict(list)
    launches = []

    for e in events:
        cat, name = e.get("cat"), str(e.get("name", ""))
        if cat == "user_annotation" and name.startswith("gen_fwd_"):
            windows[name] = (e["ts"], e["ts"] + e.get("dur", 0))
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                kernels_by_corr[corr].append(e)
        elif cat in ("cuda_runtime", "cuda_driver", "runtime"):
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                launches.append((e["ts"], corr))

    if not windows:
        raise SystemExit(
            "no gen_fwd_* annotations in the trace - the forward hooks did not fire"
        )
    launches.sort()

    names = sorted(windows)
    prefill_name, decode_names = names[0], names[1:]
    analyzed = decode_names[skip_decode_steps:]
    if not analyzed:
        raise SystemExit(
            f"only {len(decode_names)} decode steps found, cannot skip {skip_decode_steps}"
        )

    def kernels_in(window):
        lo, hi = windows[window]
        out = []
        for ts, corr in launches:
            if lo <= ts <= hi:
                out.extend(kernels_by_corr.get(corr, ()))
        return out

    per_step_counts, per_step_us = [], []
    cat_us, cat_counts = defaultdict(float), defaultdict(int)
    kernel_us, kernel_counts = defaultdict(float), defaultdict(int)

    for step in analyzed:
        ks = kernels_in(step)
        per_step_counts.append(len(ks))
        per_step_us.append(sum(k.get("dur", 0) for k in ks))
        for k in ks:
            name, dur = str(k.get("name", "?")), k.get("dur", 0)
            cat_us[categorize(name)] += dur
            cat_counts[categorize(name)] += 1
            kernel_us[name] += dur
            kernel_counts[name] += 1

    n = len(analyzed)
    return {
        "prefill_annotation": prefill_name,
        "decode_steps_found": len(decode_names),
        "decode_steps_analyzed": n,
        "kernels_per_step": statistics.mean(per_step_counts),
        "kernel_ms_per_step": statistics.mean(per_step_us) / 1000.0,
        "categories": {
            c: {"ms_per_step": cat_us[c] / n / 1000.0, "kernels_per_step": cat_counts[c] / n}
            for c in cat_us
        },
        "top_kernels": [
            {
                "name": name,
                "ms_per_step": kernel_us[name] / n / 1000.0,
                "calls_per_step": kernel_counts[name] / n,
                "avg_us": kernel_us[name] / kernel_counts[name],
            }
            for name in sorted(kernel_us, key=lambda x: -kernel_us[x])[:15]
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--decode-tokens", type=int, default=32)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--skip-decode-steps", type=int, default=2)
    p.add_argument("--trace", default="trace_decode.json")
    p.add_argument("--out", default="profile_decode.json")
    p.add_argument("--variant", default="eager", choices=sorted(VARIANTS))
    p.add_argument("--max-cache-len", type=int, default=None,
                   help="static-cache length. Default: prompt length + decode tokens, "
                        "rounded up to a multiple of 128")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - this profile needs an NVIDIA GPU.")

    max_cache_len = resolve_max_cache_len(
        args.variant, args.max_cache_len, args.prompt_len, args.decode_tokens
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model, gen_kwargs, meta = build(args.variant, max_cache_len)
    inputs = build_inputs(processor.tokenizer, args.prompt_len, 1, model.device)

    host = print_host()
    gpu = torch.cuda.get_device_name(0)
    print(f"torch={torch.__version__}  "
          f"prompt={args.prompt_len}  decode_tokens={args.decode_tokens}")
    print(f"variant={args.variant}  max_cache_len={meta['max_cache_len']}  "
          f"compile_requested={meta['compile']['requested']}\n")

    for _ in range(args.warmup):
        timed_generate(model, inputs, args.decode_tokens, gen_kwargs)

    # ---- Measurement 1: wall clock, profiler detached ----
    per_step = []
    for _ in range(args.trials):
        ttft, _ = timed_generate(model, inputs, 1, gen_kwargs)
        full, _ = timed_generate(model, inputs, args.decode_tokens, gen_kwargs)
        per_step.append((full - ttft) / (args.decode_tokens - 1))
    wall_ms = statistics.median(per_step) * 1000.0
    tok_s = 1000.0 / wall_ms
    print(f"wall clock   {wall_ms:7.2f} ms per decode step   ({tok_s:.1f} tok/s)\n")

    # ---- Measurement 2: profiler ----
    counter = {"n": 0}
    handles = label_forward_passes(model, counter)
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            model.generate(
                **inputs,
                max_new_tokens=args.decode_tokens,
                min_new_tokens=args.decode_tokens,
                do_sample=False,
                **gen_kwargs,
            )
    finally:
        for h in handles:
            h.remove()
    prof.export_chrome_trace(args.trace)

    a = analyze_trace(args.trace, args.skip_decode_steps)
    busy = 100.0 * a["kernel_ms_per_step"] / wall_ms
    idle = 100.0 - busy

    print(f"forwards seen: {counter['n']}  "
          f"(1 prefill '{a['prefill_annotation']}' + {a['decode_steps_found']} decode)")
    print(f"analyzed {a['decode_steps_analyzed']} decode steps, "
          f"skipping the first {args.skip_decode_steps}\n")

    print(f"per decode step, averaged")
    print(f"  CUDA kernels launched   {a['kernels_per_step']:8.1f}")
    print(f"  GPU kernel time         {a['kernel_ms_per_step']:8.2f} ms")
    print(f"  wall time               {wall_ms:8.2f} ms")
    print(f"  GPU busy                {busy:8.1f} %")
    print(f"  GPU idle                {idle:8.1f} %\n")

    print(f"{'category':26s} {'ms/step':>9s} {'share':>8s} {'kernels':>9s}")
    print("-" * 55)
    for c, v in sorted(a["categories"].items(), key=lambda kv: -kv[1]["ms_per_step"]):
        share = 100.0 * v["ms_per_step"] / a["kernel_ms_per_step"]
        print(f"{c:26s} {v['ms_per_step']:9.3f} {share:7.1f}% {v['kernels_per_step']:9.1f}")
    print()

    print(f"top 15 kernels by total time (per step)")
    print(f"{'ms/step':>9s} {'calls':>7s} {'avg us':>9s}  name")
    print("-" * 100)
    for k in a["top_kernels"]:
        print(f"{k['ms_per_step']:9.3f} {k['calls_per_step']:7.1f} {k['avg_us']:9.1f}  "
              f"{k['name'][:70]}")
    print()

    if busy >= 80:
        verdict = "mostly memory/compute-bound: quantization is the lever"
    elif busy < 60:
        verdict = "overhead-bound: try CUDA graphs / torch.compile first"
    else:
        verdict = "mixed"
    print(f"VERDICT  GPU busy {busy:.1f}%  ->  {verdict}")
    print("thresholds: busy >= 80% memory/compute-bound | < 60% overhead-bound | "
          "60-80% mixed")
    print(f"\ntrace written to {args.trace} - open in ui.perfetto.dev and look at the "
          "gaps between kernels by eye")

    finalize(model, meta)
    with open(args.out, "w") as f:
        json.dump(
            {
                "model": MODEL_ID,
                "variant": args.variant,
                "variant_meta": meta,
                "host": host,
                "gpu": gpu,
                "torch": torch.__version__,
                "dtype": "bfloat16",
                "prompt_len": args.prompt_len,
                "batch_size": 1,
                "decode_tokens": args.decode_tokens,
                "trials": args.trials,
                "warmup": args.warmup,
                "wall_ms_per_step": wall_ms,
                "tok_s": tok_s,
                "gpu_busy_pct": busy,
                "gpu_idle_pct": idle,
                "verdict": verdict,
                **a,
            },
            f,
            indent=2,
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
