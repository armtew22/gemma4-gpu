"""Bytes actually read from VRAM per text decode step for google/gemma-4-E2B-it in bf16.

Builds the model on the meta device, so no weights are downloaded or loaded and this runs
on a laptop in seconds. Buckets every parameter and buffer, then recomputes the bandwidth
roofline against bytes genuinely read per step rather than total resident bytes.
"""

import argparse
import json

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForMultimodalLM

from hostinfo import host_info

MODEL_ID = "google/gemma-4-E2B-it"
BYTES_PER_PARAM = 2  # bf16, regardless of the meta dtype

# Longest prefix first: embed_tokens_per_layer must beat embed_tokens, and both must beat
# the generic language_model rule.
RULES = [
    ("model.vision_tower", "non_text"),
    ("model.audio_tower", "non_text"),
    ("model.embed_vision", "non_text"),
    ("model.embed_audio", "non_text"),
    ("model.language_model.embed_tokens_per_layer", "lookup_tables"),
    ("model.language_model.embed_tokens", "lookup_tables"),
    ("model.language_model", "text_decoder"),
    ("lm_head", "text_decoder"),
]

BUCKETS = ("text_decoder", "lookup_tables", "non_text")


def classify(name):
    for prefix, bucket in RULES:
        if name.startswith(prefix):
            return bucket, prefix
    raise SystemExit(
        f"UNCLASSIFIED TENSOR: {name!r}\n"
        "Every parameter and buffer must match a rule. Add a rule rather than letting it "
        "fall into a default bucket - a silent misclassification corrupts the roofline."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bandwidth-gb-s", type=float, default=261.2,
                   help="measured achievable bandwidth (see bandwidth.json)")
    p.add_argument("--measured-tok-s", type=float, default=18.4,
                   help="measured batch-1 decode rate (see verify.json)")
    p.add_argument("--old-roofline", type=float, default=25.6,
                   help="the superseded dense roofline, for comparison")
    p.add_argument("--out", default="bytes_per_token.json")
    args = p.parse_args()

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    with init_empty_weights():
        model = AutoModelForMultimodalLM.from_config(cfg)

    # lm_head aliases the token embedding, so its storage must not be counted twice as
    # resident - but it IS a full read every step, since logits span the whole vocab.
    tied = getattr(model, "_tied_weights_keys", None) or {}
    aliases = set(tied.keys()) if isinstance(tied, dict) else set(tied)

    tensors = [(n, t, "param") for n, t in model.named_parameters()]
    tensors += [(n, t, "buffer") for n, t in model.named_buffers()]

    prefix_map = {}
    stats = {b: {"tensors": 0, "params": 0, "resident_bytes": 0, "read_bytes": 0}
             for b in BUCKETS}
    notes = []

    for name, tensor, _kind in tensors:
        bucket, prefix = classify(name)
        prefix_map.setdefault(prefix, bucket)
        n = tensor.numel()
        resident = n * BYTES_PER_PARAM

        is_alias = name in aliases
        if is_alias:
            resident = 0  # shares storage with its tied source

        if bucket == "non_text":
            read = 0
        elif bucket == "lookup_tables":
            # Indexed, not multiplied: one row per token.
            read = (tensor.shape[-1] if tensor.dim() >= 2 else n) * BYTES_PER_PARAM
            notes.append(
                f"{name}: table {tuple(tensor.shape)} resident "
                f"{n * BYTES_PER_PARAM / 1e9:.3f} GB, one row read = {read} B"
            )
        else:
            read = n * BYTES_PER_PARAM
            if is_alias:
                notes.append(
                    f"{name}: tied to {tied[name]}; resident counted once under its source, "
                    f"read in full ({read / 1e9:.3f} GB) for the vocab projection"
                )

        s = stats[bucket]
        s["tensors"] += 1
        s["params"] += n
        s["resident_bytes"] += resident
        s["read_bytes"] += read

    print(f"model: {MODEL_ID}   bf16 ({BYTES_PER_PARAM} bytes/param)")
    print(f"tensors walked: {len(tensors)} "
          f"({sum(1 for t in tensors if t[2] == 'param')} params, "
          f"{sum(1 for t in tensors if t[2] == 'buffer')} buffers)\n")

    print("module prefix -> bucket (verify this by eye)")
    for prefix in sorted(prefix_map):
        print(f"  {prefix:48s} -> {prefix_map[prefix]}")
    print()

    print("notes")
    for n in notes:
        print(f"  {n}")
    print()

    hdr = f"{'bucket':16s} {'tensors':>8s} {'params':>14s} {'resident':>12s} {'read/step':>12s}"
    print(hdr)
    print("-" * len(hdr))
    for b in BUCKETS:
        s = stats[b]
        print(f"{b:16s} {s['tensors']:8d} {s['params']/1e6:11.2f} M "
              f"{s['resident_bytes']/1e9:9.3f} GB {s['read_bytes']/1e9:9.3f} GB")

    resident_total = sum(stats[b]["resident_bytes"] for b in BUCKETS)
    read_total = sum(stats[b]["read_bytes"] for b in BUCKETS)
    print("-" * len(hdr))
    print(f"{'TOTAL':16s} {len(tensors):8d} "
          f"{sum(stats[b]['params'] for b in BUCKETS)/1e6:11.2f} M "
          f"{resident_total/1e9:9.3f} GB {read_total/1e9:9.3f} GB")
    print()

    corrected = args.bandwidth_gb_s / (read_total / 1e9)
    pct = 100 * args.measured_tok_s / corrected
    print(f"resident total            {resident_total/1e9:8.3f} GB   "
          f"(report.md measured 10.209 GB)")
    print(f"bytes read per step       {read_total/1e9:8.3f} GB   "
          f"({100*read_total/resident_total:.1f}% of resident)")
    print(f"bandwidth                 {args.bandwidth_gb_s:8.1f} GB/s")
    print(f"corrected roofline        {corrected:8.1f} tok/s")
    print(f"measured {args.measured_tok_s:.1f} tok/s is  {pct:8.1f}%   of corrected roofline")
    print(f"old dense roofline        {args.old_roofline:8.1f} tok/s "
          f"(assumed the full resident set was read)")
    print()
    print("CAVEAT: KV cache reads are excluded. At 128-token prompts they are small "
          "relative to weights, but they grow with context and batch, so this roofline "
          "is optimistic for long sequences.")

    with open(args.out, "w") as f:
        json.dump(
            {
                "model": MODEL_ID,
                "host": host_info(),
                "bytes_per_param": BYTES_PER_PARAM,
                "prefix_map": prefix_map,
                "buckets": stats,
                "resident_bytes": resident_total,
                "read_bytes_per_step": read_total,
                "bandwidth_gb_s": args.bandwidth_gb_s,
                "corrected_roofline_tok_s": corrected,
                "measured_tok_s": args.measured_tok_s,
                "pct_of_corrected_roofline": pct,
                "old_dense_roofline_tok_s": args.old_roofline,
                "kv_cache_excluded": True,
            },
            f,
            indent=2,
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
