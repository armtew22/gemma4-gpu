"""Measure achievable GPU memory bandwidth (STREAM-style) to establish a real roofline."""

import argparse
import json
import statistics
import time

import torch

from hostinfo import print_host


def bench(fn, bytes_moved, trials, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return bytes_moved / statistics.median(times) / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size-gb", type=float, default=2.0)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--weights-gb", type=float, default=10.22,
                   help="resident weight bytes, for the dense roofline")
    p.add_argument("--decode-tok-s", type=float, default=None,
                   help="measured batch-1 decode rate, to infer bytes moved per token")
    p.add_argument("--out", default="bandwidth.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    dtype = torch.bfloat16
    itemsize = torch.finfo(dtype).bits // 8
    n = int(args.size_gb * 1e9 / itemsize)

    a = torch.randn(n, device="cuda", dtype=dtype)
    b = torch.empty_like(a)
    nbytes = n * itemsize

    results = {
        # copy reads a and writes b
        "copy_gb_s": bench(lambda: b.copy_(a), 2 * nbytes, args.trials, args.warmup),
        # sum reads a only
        "read_gb_s": bench(lambda: a.sum(), nbytes, args.trials, args.warmup),
        # triad reads a and b, writes b
        "triad_gb_s": bench(lambda: torch.add(a, b, alpha=2.0, out=b),
                            3 * nbytes, args.trials, args.warmup),
    }
    achieved = max(results.values())
    results["achieved_peak_gb_s"] = achieved
    results["gpu"] = torch.cuda.get_device_name(0)
    results["host"] = print_host()

    # Roofline if every decode step reads the full resident weight set.
    dense = achieved / args.weights_gb
    results["dense_roofline_tok_s"] = dense
    print(f"gpu                  {results['gpu']}")
    print(f"copy                 {results['copy_gb_s']:7.1f} GB/s")
    print(f"read                 {results['read_gb_s']:7.1f} GB/s")
    print(f"triad                {results['triad_gb_s']:7.1f} GB/s")
    print(f"achieved peak        {achieved:7.1f} GB/s")
    print(f"dense roofline       {dense:7.1f} tok/s  (all {args.weights_gb:.2f} GB per token)")

    if args.decode_tok_s:
        implied = achieved / args.decode_tok_s
        results["measured_decode_tok_s"] = args.decode_tok_s
        results["implied_bytes_per_token_gb"] = implied
        results["pct_of_dense_roofline"] = 100 * args.decode_tok_s / dense
        print(f"measured decode      {args.decode_tok_s:7.1f} tok/s")
        print(f"  = {results['pct_of_dense_roofline']:.0f}% of dense roofline")
        print(f"  implies <={implied:.2f} GB moved per token if purely bandwidth-bound")
        if args.decode_tok_s > dense:
            print("  NOTE: exceeds dense roofline -> less than the full weight set is read "
                  "per token (selective parameter activation)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
