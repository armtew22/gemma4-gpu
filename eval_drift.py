"""How far does an optimized variant drift from the bf16 eager baseline?

Speed means nothing if the outputs changed. This measures drift two ways on a fixed
evaluation set: mean KL(reference || current) and top-1 agreement.

Usage:
    # once, on the bf16 eager model
    python eval_drift.py --variant eager --save-reference ref.pt

    # then for each optimized variant
    python eval_drift.py --variant compile --reference ref.pt

Validating the harness itself - run both before trusting any variant's number, and
always from a fresh process so nondeterminism and load-path differences are visible:

    # floor: same model, separate process -> must show no drift
    python eval_drift.py --reference ref.pt --expect no-drift

    # negative control: perturbed model -> must show drift
    python eval_drift.py --reference ref.pt --perturb 1e-3 --expect drift

Without the negative control, "KL is 0" proves nothing: a bug that always returned 0
would pass too. --self-check does save+compare in one process as a quick smoke test only.

BOS is prepended by default. Gemma 4's tokenizer reports add_bos_token=False because the
chat template adds it, and this bypasses the chat template. Measured on wikitext-2,
perplexity is 212 with BOS and 15,714 without.

Full 262k-vocab logits are far too large to store for 50k positions, so the reference
keeps each position's top-k ids and log-probs and KL is approximated over that set, with
both distributions renormalized over it.
"""

import argparse
import hashlib
import json
import time

import torch

from benchmark import MODEL_ID
from hostinfo import host_info, print_host

# Positions per log_softmax chunk. This bounds only the float32 log-prob temporaries
# (~256 x 262144 x 4 B = 268 MB). The forward pass still materialises the full bf16 logits
# for the window first (1024 x 262144 x 2 B = 537 MB), so peak overhead above the 10.2 GB
# of weights is roughly 0.8 GB - comfortable on a 23 GB L4.
POSITION_CHUNK = 256

# Current `datasets` requires a namespaced repo id; the bare "wikitext" no longer resolves.
DATASET = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"


# --- model variants -------------------------------------------------------------------
# One builder per variant. Adding "static" / "compile" / "fused_rmsnorm" later means
# adding a function here and a line to VARIANTS; nothing else changes.

def _build_eager(dtype):
    from transformers import AutoModelForMultimodalLM

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=dtype
    )
    model.eval()
    return model, {}


VARIANTS = {
    "eager": _build_eager,
}


def perturb_one_layer(model, scale, seed, pattern):
    """Add tiny relative noise to one layer's weight: the negative control.

    Without this, --self-check only proves "KL is 0 when nothing changed", which a bug
    that always returns 0 would also pass. A test that cannot fail proves nothing.
    """
    gen = torch.Generator().manual_seed(seed)
    for name, param in model.named_parameters():
        if pattern in name and param.dim() >= 2:
            with torch.no_grad():
                std = param.detach().float().std().item()
                before = param.detach().float().norm().item()
                noise = torch.randn(param.shape, generator=gen, dtype=torch.float32)
                param.add_(noise.to(param.device, param.dtype) * (scale * std))
                after = param.detach().float().norm().item()
            return {
                "param": name,
                "shape": tuple(param.shape),
                "scale": scale,
                "seed": seed,
                "weight_std": std,
                "norm_before": before,
                "norm_after": after,
            }
    raise SystemExit(f"no parameter matched {pattern!r} with dim >= 2")


def build_model(variant, dtype=torch.bfloat16):
    if variant not in VARIANTS:
        raise SystemExit(
            f"unknown variant {variant!r}; known: {sorted(VARIANTS)}. "
            "Add a builder to VARIANTS."
        )
    return VARIANTS[variant](dtype)


# --- evaluation set -------------------------------------------------------------------

def load_eval_windows(tokenizer, num_tokens, window, add_bos, dataset=DATASET):
    """Non-overlapping `window`-token windows over wikitext-2 test, first `num_tokens`.

    The corpus is tokenized once without special tokens, then chunked, so window
    boundaries are a deterministic function of (dataset, window, num_tokens) alone.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets")

    ds = load_dataset(dataset, DATASET_CONFIG, split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, add_special_tokens=False).input_ids

    n_windows = num_tokens // window
    if n_windows == 0:
        raise SystemExit(f"--num-tokens {num_tokens} is smaller than one {window} window")
    if len(ids) < n_windows * window:
        raise SystemExit(
            f"corpus has {len(ids)} tokens, need {n_windows * window}; lower --num-tokens"
        )

    bos = tokenizer.bos_token_id if add_bos else None
    windows = []
    for w in range(n_windows):
        chunk = ids[w * window : (w + 1) * window]
        windows.append(([bos] + chunk) if bos is not None else chunk)

    # Checksum over the scored targets, so a variant run can prove it scored the same
    # tokens as the reference. Settings drifting silently would make KL meaningless.
    # The structure prefix matters: the same num_tokens split into 512- vs 1024-token
    # windows flattens to an identical token sequence but gives each position a different
    # attention context, so the token bytes alone would not distinguish them.
    flat = f"{window}:{n_windows}:{bos}:".encode()
    flat += b"".join(int(t).to_bytes(4, "little") for w in windows for t in w)
    return windows, {
        "dataset": f"{dataset}/{DATASET_CONFIG}:test",
        "join": "\\n\\n",
        "window": window,
        "num_windows": n_windows,
        "tokens_requested": num_tokens,
        "add_bos": bos is not None,
        "bos_token_id": bos,
        "corpus_tokens_available": len(ids),
        "eval_set_sha256": hashlib.sha256(flat).hexdigest(),
    }


# --- scoring --------------------------------------------------------------------------

def window_logits(model, window_ids):
    ids = torch.tensor([window_ids], device=model.device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    # logits[j] predicts window_ids[j+1]; score every position with a target
    return out.logits[0, :-1, :], ids[0, 1:]


def score_reference(model, windows, topk):
    """Top-k ids/log-probs, full-vocab argmax, and target log-prob at every position."""
    acc = {k: [] for k in ("topk_ids", "topk_logprobs", "argmax_ids",
                           "argmax_logprobs", "target_logprobs")}
    for w in windows:
        logits, targets = window_logits(model, w)
        for a in range(0, logits.shape[0], POSITION_CHUNK):
            lp = torch.log_softmax(logits[a : a + POSITION_CHUNK].float(), dim=-1)
            vals, idx = lp.topk(topk, dim=-1)
            acc["topk_ids"].append(idx.to(torch.int32).cpu())
            acc["topk_logprobs"].append(vals.cpu())
            # argmax over the full vocabulary is topk's first column by construction
            acc["argmax_ids"].append(idx[:, 0].to(torch.int32).cpu())
            acc["argmax_logprobs"].append(vals[:, 0].cpu())
            tgt = targets[a : a + POSITION_CHUNK]
            acc["target_logprobs"].append(lp.gather(-1, tgt[:, None])[:, 0].cpu())
            del lp, vals, idx
    return {k: torch.cat(v) for k, v in acc.items()}


def score_against(model, windows, ref, topk):
    """mean KL(ref || current) over the reference top-k, top-1 agreement, perplexity."""
    kl_sum, agree, n = 0.0, 0, 0
    nll_sum = 0.0
    coverage_sum = 0.0
    pos = 0
    for w in windows:
        logits, targets = window_logits(model, w)
        for a in range(0, logits.shape[0], POSITION_CHUNK):
            lp = torch.log_softmax(logits[a : a + POSITION_CHUNK].float(), dim=-1)
            m = lp.shape[0]
            sl = slice(pos, pos + m)

            ref_ids = ref["topk_ids"][sl].long().to(lp.device)
            ref_lp = ref["topk_logprobs"][sl].to(lp.device)

            # renormalize both distributions over the reference's top-k support
            log_p = ref_lp - torch.logsumexp(ref_lp, dim=-1, keepdim=True)
            cur_lp = lp.gather(-1, ref_ids)
            # How much of the CURRENT model's full-vocab mass sits on the reference's
            # top-k support. If this falls (likely under 4-bit), the top-k KL
            # approximation is measuring less of the real distribution.
            coverage_sum += cur_lp.exp().sum(-1).sum().item()
            log_q = cur_lp - torch.logsumexp(cur_lp, dim=-1, keepdim=True)
            kl_sum += (log_p.exp() * (log_p - log_q)).sum(-1).sum().item()

            agree += (lp.argmax(-1).to(torch.int32)
                      == ref["argmax_ids"][sl].to(lp.device)).sum().item()

            tgt = targets[a : a + POSITION_CHUNK]
            nll_sum += -lp.gather(-1, tgt[:, None])[:, 0].sum().item()

            n += m
            pos += m
            del lp, cur_lp, log_p, log_q, ref_ids, ref_lp
    return {
        "tokens_scored": n,
        "mean_kl": kl_sum / n,
        "top1_agreement_pct": 100.0 * agree / n,
        "perplexity": float(torch.tensor(nll_sum / n).exp()),
        "mean_nll": nll_sum / n,
        "topk_coverage_pct": 100.0 * coverage_sum / n,
    }


# --- main -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="eager", choices=sorted(VARIANTS))
    p.add_argument("--save-reference", metavar="PATH")
    p.add_argument("--reference", metavar="PATH")
    p.add_argument("--self-check", action="store_true",
                   help="in-process save+compare against the same model. Convenience "
                        "smoke test only - a fresh-process --save-reference then "
                        "--reference run is the real check, since it also catches "
                        "nondeterminism and load-path differences.")
    p.add_argument("--expect", choices=("none", "no-drift", "drift"), default="none",
                   help="assert on the outcome. 'no-drift' requires KL <= tol and top-1 "
                        ">= min; 'drift' requires the opposite, which is what makes the "
                        "harness falsifiable.")
    p.add_argument("--perturb", type=float, default=0.0,
                   help="negative control: add this much relative gaussian noise to one "
                        "layer's weight before scoring (e.g. 1e-3)")
    p.add_argument("--perturb-pattern", default="language_model.layers.17.")
    p.add_argument("--perturb-seed", type=int, default=0)
    p.add_argument("--num-tokens", type=int, default=50_000)
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--no-bos", action="store_true",
                   help="do NOT prepend BOS. BOS is ON by default despite Gemma 4's "
                        "tokenizer reporting add_bos_token=False: that config reflects the "
                        "chat template adding it, and we bypass the chat template. Gemma "
                        "models are known to be sensitive to a missing BOS, which inflates "
                        "perplexity and puts the model off-distribution - a bad state in "
                        "which to measure quantization drift.")
    p.add_argument("--self-check-kl-tol", type=float, default=1e-5)
    p.add_argument("--self-check-top1-min", type=float, default=99.99)
    p.add_argument("--out", metavar="PATH")
    args = p.parse_args()

    if not (args.save_reference or args.reference or args.self_check):
        raise SystemExit("pass --save-reference, --reference, or --self-check")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - this evaluation needs an NVIDIA GPU.")

    # AutoTokenizer, not AutoProcessor: this is text-only scoring, and the multimodal
    # processor pulls in PIL for image handling we never use.
    from transformers import AutoTokenizer

    host = print_host()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    add_bos = (not args.no_bos) and tokenizer.bos_token_id is not None
    windows, settings = load_eval_windows(
        tokenizer, args.num_tokens, args.window, add_bos, args.dataset
    )
    settings["topk"] = args.topk
    settings["model"] = MODEL_ID
    settings["dtype"] = "bfloat16"
    print(f"eval set: {settings['num_windows']} x {args.window} tokens, "
          f"bos={'yes' if add_bos else 'no'}, sha256={settings['eval_set_sha256'][:16]}\n")

    model, variant_meta = build_model(args.variant)
    print(f"variant: {args.variant}  {variant_meta if variant_meta else ''}")

    if args.perturb > 0:
        pert = perturb_one_layer(
            model, args.perturb, args.perturb_seed, args.perturb_pattern
        )
        variant_meta["perturb"] = pert
        print(f"PERTURBED {pert['param']} {pert['shape']} "
              f"scale={pert['scale']:g} (relative to weight std {pert['weight_std']:.4g}); "
              f"norm {pert['norm_before']:.2f} -> {pert['norm_after']:.2f}")

    t0 = time.perf_counter()

    if args.save_reference or args.self_check:
        ref = score_reference(model, windows, args.topk)
        path = args.save_reference or "self_check_ref.pt"
        torch.save({"settings": settings, "host": host, **ref}, path)
        ppl = float((-ref["target_logprobs"].mean()).exp())
        print(f"reference saved to {path}: "
              f"{ref['argmax_ids'].numel()} positions, top-{args.topk}, "
              f"perplexity {ppl:.3f}")
        if not args.self_check:
            print(f"wall {time.perf_counter() - t0:.1f}s")
            return

    ref_path = args.reference or "self_check_ref.pt"
    blob = torch.load(ref_path, map_location="cpu", weights_only=False)
    ref_settings = blob.pop("settings")
    ref_host = blob.pop("host", None)

    for key in ("dataset", "window", "num_windows", "add_bos", "topk", "eval_set_sha256"):
        if ref_settings.get(key) != settings.get(key):
            raise SystemExit(
                f"settings mismatch on {key!r}: reference={ref_settings.get(key)!r} "
                f"current={settings.get(key)!r}. KL across different evaluation sets is "
                "meaningless."
            )

    result = score_against(model, windows, blob, args.topk)
    wall = time.perf_counter() - t0

    print()
    print(f"tokens scored         {result['tokens_scored']}")
    print(f"mean KL(ref||cur)     {result['mean_kl']:.3e}  nats, over the ref top-{args.topk}")
    print(f"top-1 agreement       {result['top1_agreement_pct']:.4f} %")
    print(f"perplexity            {result['perplexity']:.4f}")
    print(f"top-{args.topk} coverage       {result['topk_coverage_pct']:.3f} %  "
          f"(current model's mass inside the reference top-k)")
    print(f"wall                  {wall:.1f}s")

    out = args.out or f"drift_{args.variant}.json"
    with open(out, "w") as f:
        json.dump(
            {
                "variant": args.variant,
                "variant_meta": variant_meta,
                "host": host_info(),
                "reference_host": ref_host,
                "settings": settings,
                "reference_path": ref_path,
                "wall_seconds": wall,
                **result,
            },
            f,
            indent=2,
        )
    print(f"wrote {out}")

    if args.expect != "none" or args.self_check:
        expect = "no-drift" if args.expect == "none" else args.expect
        kl, t1 = result["mean_kl"], result["top1_agreement_pct"]
        print()
        if expect == "no-drift":
            ok_kl = kl <= args.self_check_kl_tol
            ok_t1 = t1 >= args.self_check_top1_min
            print(f"EXPECT no-drift  KL {kl:.3e} <= {args.self_check_kl_tol:.0e}"
                  f"   {'PASS' if ok_kl else 'FAIL'}")
            print(f"EXPECT no-drift  top-1 {t1:.4f}% >= {args.self_check_top1_min}%"
                  f"   {'PASS' if ok_t1 else 'FAIL'}")
            assert ok_kl, f"expected no drift but KL is {kl:.3e}"
            assert ok_t1, f"expected no drift but top-1 is only {t1:.4f}%"
            print("EXPECT no-drift  PASS - reports zero drift when nothing changed")
        else:
            ok_kl = kl > args.self_check_kl_tol
            ok_t1 = t1 < 100.0
            print(f"EXPECT drift     KL {kl:.3e} > {args.self_check_kl_tol:.0e}"
                  f"   {'PASS' if ok_kl else 'FAIL'}")
            print(f"EXPECT drift     top-1 {t1:.4f}% < 100%"
                  f"   {'PASS' if ok_t1 else 'FAIL'}")
            assert ok_kl, (
                f"expected drift but KL is only {kl:.3e} - the harness may be incapable "
                "of detecting a difference, which would invalidate every no-drift result"
            )
            assert ok_t1, "expected drift but top-1 agreement is still 100%"
            print("EXPECT drift     PASS - reports nonzero drift when something changed")


if __name__ == "__main__":
    main()
