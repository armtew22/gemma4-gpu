"""Bytes read per decode step, and the bandwidth ceiling, with weight-only int8 / int4 linears.

Uses the meta model and the bytes_per_token.py accounting (same buckets, same read rules), so
the bf16 row reproduces bytes_per_token.json exactly. Runs on a laptop in seconds.

"Linears" = every nn.Linear in the text decoder:
  attention  q_proj, k_proj, v_proj, o_proj
  MLP        gate_proj, up_proj, down_proj
  PLE        per_layer_input_gate, per_layer_projection (per layer),
             per_layer_model_projection (once per step)
lm_head is reported separately: bf16, or quantized like the other linears. It is tied to
embed_tokens, so quantizing it means a second, quantized copy of that matrix for the vocab
projection; the embedding lookup keeps the bf16 table (resident bytes go up).

Everything else stays bf16: the norms, the embedding tables (one row read per token), and any
other text-decoder parameter or buffer. Vision and audio towers are not read during text decode.

Quantized sizes (quant/primitives.py formats):
  int8  out * in bytes            + one bf16 scale per output row
  int4  out * in / 2 bytes        + one bf16 scale per group of 128 inputs

The ceiling assumes the dequantizing kernels reach the same 261.2 GB/s as bf16 GEMV and that
dequantization is free - an upper bound. KV cache reads are excluded, as in bytes_per_token.py.
"""

import argparse
import json
import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerate import init_empty_weights  # noqa: E402
from transformers import AutoConfig, AutoModelForMultimodalLM  # noqa: E402

from bytes_per_token import BYTES_PER_PARAM, MODEL_ID, classify  # noqa: E402
from hostinfo import host_info  # noqa: E402

GROUP_SIZE = 128
SCALE_BYTES = 2  # bf16
LM_HEAD = "lm_head.weight"
FAMILIES = {
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "ple": ("per_layer_input_gate", "per_layer_projection", "per_layer_model_projection"),
}


def family(module_name):
    leaf = module_name.rsplit(".", 1)[-1]
    for fam, leaves in FAMILIES.items():
        if leaf in leaves:
            return fam
    raise SystemExit(f"UNCLASSIFIED LINEAR: {module_name!r} - add it to FAMILIES")


def quant_bytes(shape, fmt):
    out_f, in_f = shape
    if fmt == "bf16":
        return out_f * in_f * BYTES_PER_PARAM
    if fmt == "int8":
        return out_f * in_f + out_f * SCALE_BYTES
    if fmt == "int4":
        if in_f % GROUP_SIZE:
            raise SystemExit(f"in_features {in_f} is not a multiple of {GROUP_SIZE}")
        return out_f * in_f // 2 + out_f * (in_f // GROUP_SIZE) * SCALE_BYTES
    raise ValueError(fmt)


def bf16_read_bytes(tensor, bucket):
    """The read rule of bytes_per_token.py, unchanged."""
    if bucket == "non_text":
        return 0
    if bucket == "lookup_tables":
        return (tensor.shape[-1] if tensor.dim() >= 2 else tensor.numel()) * BYTES_PER_PARAM
    return tensor.numel() * BYTES_PER_PARAM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bandwidth-gb-s", type=float, default=261.2,
                   help="measured achievable bandwidth (see bandwidth.json)")
    p.add_argument("--measured-tok-s", type=float, default=48.38,
                   help="best measured bf16 decode rate (graph, 2026-11-04)")
    p.add_argument("--out", default="quant_bytes.json")
    args = p.parse_args()

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    with init_empty_weights():
        model = AutoModelForMultimodalLM.from_config(cfg)

    linears = {}  # weight name -> (family, shape)
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        wname = f"{name}.weight"
        if wname == LM_HEAD or classify(wname)[0] != "text_decoder":
            continue
        if mod.bias is not None:
            raise SystemExit(f"{name} has a bias - the byte model assumes weight only")
        linears[wname] = (family(name), (mod.out_features, mod.in_features))

    tensors = [(n, t) for n, t in model.named_parameters()]
    tensors += [(n, t) for n, t in model.named_buffers()]

    # The fixed part: everything that stays bf16 whatever the scenario.
    fixed = {"norms": 0, "embedding rows": 0, "other": 0}
    fixed_counts = {k: 0 for k in fixed}
    bf16_total, lm_head_shape, seen = 0, None, set()
    for name, t in tensors:
        bucket, _ = classify(name)
        read = bf16_read_bytes(t, bucket)
        bf16_total += read
        if name == LM_HEAD:
            lm_head_shape = tuple(t.shape)
        elif name in linears:
            seen.add(name)
        elif read:
            kind = ("norms" if "norm" in name else
                    "embedding rows" if bucket == "lookup_tables" else "other")
            fixed[kind] += read
            fixed_counts[kind] += 1
    if seen != set(linears):
        raise SystemExit(f"linears not found among parameters: {sorted(set(linears) - seen)[:5]}")
    if lm_head_shape is None:
        raise SystemExit("lm_head.weight not found among parameters")

    ref = None
    if os.path.exists("bytes_per_token.json"):
        with open("bytes_per_token.json") as f:
            ref = json.load(f)["read_bytes_per_step"]
        if ref != bf16_total:
            raise SystemExit(f"bf16 total {bf16_total} != bytes_per_token.json {ref}")

    fam_params = {fam: 0 for fam in FAMILIES}
    fam_counts = {fam: 0 for fam in FAMILIES}
    for fam, shape in linears.values():
        fam_params[fam] += shape[0] * shape[1]
        fam_counts[fam] += 1

    print(f"model: {MODEL_ID}   bandwidth {args.bandwidth_gb_s} GB/s   int4 group {GROUP_SIZE}\n")
    print("quantized linears (text decoder, weight only)")
    for fam in FAMILIES:
        print(f"  {fam:10s} {fam_counts[fam]:4d} tensors {fam_params[fam] / 1e6:9.2f} M params  "
              f"{', '.join(FAMILIES[fam])}")
    print(f"  {'lm_head':10s} {1:4d} tensor  {lm_head_shape[0] * lm_head_shape[1] / 1e6:9.2f} M params  "
          f"{lm_head_shape} - bf16 or quantized, per scenario")
    print("always bf16 (read per step)")
    for kind in fixed:
        print(f"  {kind:15s} {fixed_counts[kind]:4d} tensors {fixed[kind] / 1e6:9.3f} MB")
    print(f"bf16 total reproduces bytes_per_token.json: "
          f"{'yes' if ref is not None else 'not checked (file missing)'} ({bf16_total:,} B)\n")

    fixed_total = sum(fixed.values())
    scenarios = [("bf16 (today)", "bf16", "bf16"),
                 ("int8 linears, lm_head bf16", "int8", "bf16"),
                 ("int8 linears + int8 lm_head", "int8", "int8"),
                 ("int4 linears, lm_head bf16", "int4", "bf16"),
                 ("int4 linears + int4 lm_head", "int4", "int4")]
    rows = []
    hdr = (f"{'scenario':30s} {'linears':>9s} {'lm_head':>9s} {'bf16 rest':>9s} "
           f"{'read/step':>10s} {'vs bf16':>8s} {'ceiling':>11s}")
    print(hdr)
    print("-" * len(hdr))
    for label, lin_fmt, head_fmt in scenarios:
        lin = sum(quant_bytes(shape, lin_fmt) for _, shape in linears.values())
        head = quant_bytes(lm_head_shape, head_fmt)
        total = lin + head + fixed_total
        ceiling = args.bandwidth_gb_s / (total / 1e9)
        rows.append({"scenario": label, "linears": lin_fmt, "lm_head": head_fmt,
                     "linear_bytes": lin, "lm_head_bytes": head, "bf16_rest_bytes": fixed_total,
                     "read_bytes_per_step": total, "ceiling_tok_s": ceiling,
                     "extra_resident_bytes": 0 if head_fmt == "bf16" else head})
        print(f"{label:30s} {lin / 1e9:7.3f}GB {head / 1e9:7.3f}GB {fixed_total / 1e9:7.3f}GB "
              f"{total / 1e9:8.3f}GB {total / bf16_total:7.1%} {ceiling:7.1f} tok/s")
    if rows[0]["read_bytes_per_step"] != bf16_total:
        raise SystemExit("bf16 scenario does not match the bf16 walk")
    print()
    print(f"measured today: {args.measured_tok_s} tok/s (graph, bf16) = "
          f"{100 * args.measured_tok_s / rows[0]['ceiling_tok_s']:.1f}% of the bf16 ceiling")
    print("Quantizing lm_head adds a quantized copy of the tied embedding matrix to resident "
          "memory (the lookup keeps the bf16 table).")
    print("CAVEAT: upper bounds - free dequantization, bf16-GEMV bandwidth, KV cache excluded.")

    with open(args.out, "w") as f:
        json.dump({"model": MODEL_ID, "host": host_info(), "bandwidth_gb_s": args.bandwidth_gb_s,
                   "int4_group_size": GROUP_SIZE, "scale_dtype": "bfloat16",
                   "linear_families": {fam: {"leaves": FAMILIES[fam], "tensors": fam_counts[fam],
                                             "params": fam_params[fam]} for fam in FAMILIES},
                   "lm_head_shape": lm_head_shape, "bf16_fixed_bytes": fixed,
                   "bf16_matches_bytes_per_token": ref is not None,
                   "measured_tok_s": args.measured_tok_s, "scenarios": rows,
                   "kv_cache_excluded": True}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
