"""How far does an optimized variant drift from the bf16 eager baseline?

Speed means nothing if the outputs changed. This measures drift two ways on a fixed
evaluation set: mean KL(reference || current) and top-1 agreement.

Usage:
    # once, on the bf16 eager model
    python eval_drift.py --variant eager --save-reference ref.pt

    # then for each variant that changes the forward pass itself
    python eval_drift.py --variant <variant> --reference ref.pt

Variants in variants.GENERATE_ONLY_VARIANTS (static, compile) are refused in the default
prefill mode: their behaviour exists only inside generate(), and prefill mode scores with a
plain forward pass, so they would be scored as eager and report a false PASS. Score them
with --mode decode, which runs teacher-forced generate(), against a decode-mode eager
reference:

    python eval_drift.py --mode decode --save-reference ref_decode.pt \\
        --prefill-reference ref_50k_bos.pt      # also checks decode vs prefill
    python eval_drift.py --mode decode --variant compile --reference ref_decode.pt \\
        --expect pass-band

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
from contextlib import contextmanager

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from hostinfo import host_info, print_host
from variants import GENERATE_ONLY_VARIANTS, MODEL_ID, VARIANTS, build, resolve_max_cache_len

# Positions per log_softmax chunk. This bounds only the float32 log-prob temporaries
# (~256 x 262144 x 4 B = 268 MB). The forward pass still materialises the full bf16 logits
# for the window first (1024 x 262144 x 2 B = 537 MB), so peak overhead above the 10.2 GB
# of weights is roughly 0.8 GB - comfortable on a 23 GB L4.
POSITION_CHUNK = 256

# Current `datasets` requires a namespaced repo id; the bare "wikitext" no longer resolves.
DATASET = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"


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


# --- decode-path scoring --------------------------------------------------------------
# Prefill scoring runs one forward pass per window, which never enters generate(). The static
# cache, torch.compile and CUDA graphs exist only inside generate()'s decode loop, so decode
# mode scores through the real generate() with teacher forcing: every step records the
# model's raw next-token distribution, then forces the true token so the sequence stays on
# the evaluation text.

DECODE_WINDOWS = 8
DECODE_WINDOW_LEN = 256
DECODE_PROMPT_LEN = 16
REF_KEYS = ("topk_ids", "topk_logprobs", "argmax_ids", "argmax_logprobs", "target_logprobs")


def _corpus_ids(tokenizer, dataset):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets")
    ds = load_dataset(dataset, DATASET_CONFIG, split="test")
    return tokenizer("\n\n".join(ds["text"]), add_special_tokens=False).input_ids


def load_decode_windows(tokenizer, n_windows, window_len, prompt_len, add_bos, dataset=DATASET):
    """Decode eval set: `n_windows` non-overlapping `window_len`-token windows, BOS prepended.

    Each window's prompt is BOS + its first `prompt_len` tokens; the rest are generated under
    teacher forcing. Pinned by its own sha256, which covers the mode and the prompt length as
    well as the tokens, so a decode reference cannot be compared against a different decode
    set or against a prefill set.
    """
    if not 0 < prompt_len < window_len - 1:
        raise SystemExit(
            f"--decode-prompt-len {prompt_len} must leave at least 2 generated tokens in a "
            f"{window_len}-token window: step 0 comes from the prefill pass and is not scored"
        )
    ids = _corpus_ids(tokenizer, dataset)
    if len(ids) < n_windows * window_len:
        raise SystemExit(f"corpus has {len(ids)} tokens, need {n_windows * window_len}")

    bos = tokenizer.bos_token_id if add_bos else None
    windows = []
    for w in range(n_windows):
        chunk = ids[w * window_len : (w + 1) * window_len]
        windows.append(([bos] + chunk) if bos is not None else chunk)

    generated = window_len - prompt_len
    flat = f"decode:{window_len}:{n_windows}:{prompt_len}:{bos}:".encode()
    flat += b"".join(int(t).to_bytes(4, "little") for w in windows for t in w)
    return windows, {
        "mode": "decode",
        "dataset": f"{dataset}/{DATASET_CONFIG}:test",
        "join": "\\n\\n",
        "decode_window_len": window_len,
        "decode_windows": n_windows,
        "decode_prompt_len": prompt_len,
        "prompt_tokens_with_bos": prompt_len + (1 if bos is not None else 0),
        "generated_per_window": generated,
        "scored_per_window": generated - 1,
        "scored_positions": n_windows * (generated - 1),
        "unscored": "step 0 of each window - its logits come from the prefill forward pass",
        "add_bos": bos is not None,
        "bos_token_id": bos,
        "corpus_tokens_available": len(ids),
        "eval_set_sha256": hashlib.sha256(flat).hexdigest(),
    }


class TeacherForcingRecorder(LogitsProcessor):
    """Record each step's RAW next-token distribution, then force the true next token.

    Order matters twice. Inside this processor, recording happens before forcing: after
    forcing, every variant would score perfectly. And this processor must run before every
    processor generate() builds for itself, which _processor_first() guarantees.
    """

    def __init__(self, window_ids, prompt_total, topk, gathers=None):
        self.window_ids = window_ids
        self.prompt_total = prompt_total
        self.topk = topk
        # name -> [generated, k] ids whose current log-probs to read at each step (KL needs
        # the current model's mass on the reference's top-k support)
        self.gathers = gathers or {}
        self.steps = []

    def __call__(self, input_ids, scores):
        assert input_ids.shape[0] == 1, "decode scoring runs one window at a time"
        pos = input_ids.shape[-1]  # index in the window of the token about to be generated
        step = pos - self.prompt_total
        target = self.window_ids[pos]

        # 1. record the raw distribution, moved to the CPU so GPU memory does not grow
        lp = torch.log_softmax(scores[0].float(), dim=-1)
        vals, idx = lp.topk(self.topk)
        record = {
            "step": step,
            "target": target,
            "topk_ids": idx.to(torch.int32).cpu(),
            "topk_logprobs": vals.cpu(),
            "argmax_id": int(idx[0]),
            "argmax_logprob": float(vals[0]),
            "target_logprob": float(lp[target]),
        }
        for name, ids in self.gathers.items():
            record[name] = lp[ids[step].to(lp.device)].cpu()
        self.steps.append(record)

        # 2. only then force: greedy decoding can now pick nothing but the true token
        forced = torch.full_like(scores, float("-inf"))
        forced[:, target] = 0.0
        return forced


@contextmanager
def _processor_first(model, processor):
    """Make `processor` the first logits processor generate() applies.

    generate() appends user processors AFTER the ones it builds itself
    (_merge_criteria_processor_list), and min_new_tokens makes it build
    MinNewTokensLengthLogitsProcessor, which sets Gemma's EOS ids [1, 106, 50] to -inf. A
    recorder passed the normal way would see EOS-suppressed, renormalized logits instead of
    raw ones. This reorders post-processing for the scoring call only; the model's forward
    pass is untouched.
    """
    original = getattr(model, "_get_logits_processor", None)
    if original is None:
        raise SystemExit(
            "this transformers version has no _get_logits_processor, so the recorder cannot "
            "be guaranteed to see raw logits - refusing rather than recording processed ones"
        )
    order = []

    def recorder_first(*args, **kwargs):
        procs = LogitsProcessorList([processor, *original(*args, **kwargs)])
        order[:] = [type(p).__name__ for p in procs]
        return procs

    model._get_logits_processor = recorder_first
    try:
        yield order
    finally:
        del model._get_logits_processor


def decode_window(model, window_ids, prompt_total, topk, gen_kwargs, gathers=None):
    """Teacher-forced generate() over one window. Returns the decode steps' records."""
    generated = len(window_ids) - prompt_total
    prompt = torch.tensor([window_ids[:prompt_total]], device=model.device)
    recorder = TeacherForcingRecorder(window_ids, prompt_total, topk, gathers)
    with _processor_first(model, recorder) as order:
        out = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=generated,
            min_new_tokens=generated,
            # explicit: Gemma 4's shipped generation config has do_sample=True
            do_sample=False,
            **gen_kwargs,
        )

    got = out[0, prompt_total:].tolist()
    want = list(window_ids[prompt_total:])
    if got != want:
        first = next(i for i, (a, b) in enumerate(zip(got + [None] * len(want), want)) if a != b)
        raise AssertionError(
            f"teacher forcing failed at generated position {first}: got {got[first:first + 3]}"
            f" want {want[first:first + 3]} - every number from this window would be invalid"
        )
    assert len(recorder.steps) == generated, (len(recorder.steps), generated)
    assert order and order[0] == type(recorder).__name__, f"recorder not first: {order}"
    # step 0's logits come from the prefill forward pass, so only steps 1.. are decode steps
    return recorder.steps[1:], order


def run_decode(model, windows, prompt_total, topk, gen_kwargs, gather_plans=None):
    """Teacher-forced decode over every window; stacked tensors for the decode steps.

    gather_plans: name -> one entry per window, each a [generated, k] LongTensor of ids to
    read at every step, or None to skip that window.
    """
    gather_plans = gather_plans or {}
    current = {k: [] for k in REF_KEYS}
    gathered = {name: [] for name in gather_plans}
    rows = {name: [] for name in gather_plans}
    order, offset = None, 0
    for w, window_ids in enumerate(windows):
        gathers = {n: plan[w] for n, plan in gather_plans.items() if plan[w] is not None}
        steps, order = decode_window(model, window_ids, prompt_total, topk, gen_kwargs, gathers)
        current["topk_ids"].append(torch.stack([s["topk_ids"] for s in steps]))
        current["topk_logprobs"].append(torch.stack([s["topk_logprobs"] for s in steps]))
        current["argmax_ids"].append(torch.tensor([s["argmax_id"] for s in steps], dtype=torch.int32))
        current["argmax_logprobs"].append(torch.tensor([s["argmax_logprob"] for s in steps]))
        current["target_logprobs"].append(torch.tensor([s["target_logprob"] for s in steps]))
        for name in gathers:
            gathered[name].append(torch.stack([s[name] for s in steps]))
            rows[name].append(torch.arange(offset, offset + len(steps)))
        offset += len(steps)
    return {
        "current": {k: torch.cat(v) for k, v in current.items()},
        "gathered": {n: torch.cat(v) for n, v in gathered.items() if v},
        "rows": {n: torch.cat(v) for n, v in rows.items() if v},
        "processor_order": order,
    }


def reference_gather_plan(ref, n_windows, generated):
    """Per-window [generated, k] ids from a decode reference, row 0 unused (prefill step)."""
    scored = generated - 1
    plan = []
    for w in range(n_windows):
        ids = ref["topk_ids"][w * scored : (w + 1) * scored].long()
        plan.append(torch.cat([torch.zeros(1, ids.shape[1], dtype=torch.long), ids]))
    return plan


def drift_metrics(ref_topk_lp, ref_argmax, cur_at_ref, cur_argmax, cur_target_lp):
    """The same formulas as score_against, over tensors already collected."""
    n = ref_topk_lp.shape[0]
    log_p = ref_topk_lp - torch.logsumexp(ref_topk_lp, dim=-1, keepdim=True)
    log_q = cur_at_ref - torch.logsumexp(cur_at_ref, dim=-1, keepdim=True)
    kl_sum = (log_p.exp() * (log_p - log_q)).sum(-1).sum().item()
    agree = (cur_argmax.to(torch.int32) == ref_argmax.to(torch.int32)).sum().item()
    nll_sum = -cur_target_lp.sum().item()
    return {
        "tokens_scored": n,
        "mean_kl": kl_sum / n,
        "top1_agreement_pct": 100.0 * agree / n,
        "perplexity": float(torch.tensor(nll_sum / n).exp()),
        "mean_nll": nll_sum / n,
        "topk_coverage_pct": 100.0 * cur_at_ref.exp().sum(-1).sum().item() / n,
    }


def prefill_overlap_plan(decode_settings, prefill_settings, prefill_ref):
    """Map decode positions onto a prefill reference wherever the context is identical.

    Both modes prepend BOS at their own window start, so a decode position matches the
    prefill reference only when its decode window starts exactly where a prefill window
    starts. Anywhere else the same token is predicted from a different context and the
    comparison would measure that, not drift. With BOS, the prefill reference stores the
    prediction for global token g at flat index g.
    """
    if not (decode_settings["add_bos"] and prefill_settings.get("add_bos")):
        return None, "both evaluation sets must prepend BOS"
    if decode_settings["dataset"] != prefill_settings.get("dataset"):
        return None, "different datasets"
    ld, p = decode_settings["decode_window_len"], decode_settings["decode_prompt_len"]
    lp, nwp = prefill_settings["window"], prefill_settings["num_windows"]
    if ld > lp:
        return None, f"decode window {ld} is longer than the prefill window {lp}"
    generated = ld - p
    gather, flat_idx, aligned = [], [], []
    for w in range(decode_settings["decode_windows"]):
        start = w * ld
        if start % lp or start // lp >= nwp:
            gather.append(None)
            continue
        g = torch.arange(start + p, start + p + generated)  # global token each step predicts
        gather.append(prefill_ref["topk_ids"][g].long())
        flat_idx.append(g[1:])  # scored decode steps only
        aligned.append(w)
    if not aligned:
        return None, "no decode window starts on a prefill window boundary"
    return {"gather": gather, "flat_idx": torch.cat(flat_idx), "windows": aligned}, None


# --- pre-registered pass/fail bands ---------------------------------------------------
# The eager-vs-eager floor is exactly KL 0 / top-1 100%, but that cannot be the bar for an
# optimized variant: torch.compile and fused kernels change floating-point summation order,
# so their bf16 outputs cannot be bitwise identical to eager even when they are correct.
# These bands give every variant a verdict that was fixed BEFORE any variant was run, so a
# result cannot quietly move the goalposts. FAIL is set from the measured negative control
# (one layer perturbed by 1e-3: KL 8.879e-4, top-1 98.9014%), rounded to 8.9e-4 / 98.9%.
# Note the rounding direction: the control itself lands just inside INVESTIGATE, not FAIL.
# PASS sits roughly an order of magnitude below the control on KL. The gap between them is
# INVESTIGATE: not proven wrong, not trusted either.

BAND_PASS_KL = 1e-4
BAND_PASS_TOP1 = 99.5
BAND_FAIL_KL = 8.9e-4
BAND_FAIL_TOP1 = 98.9


def classify_band(kl, top1, pass_kl, pass_top1, fail_kl, fail_top1):
    if kl >= fail_kl or top1 <= fail_top1:
        return "FAIL"
    if kl < pass_kl and top1 >= pass_top1:
        return "PASS"
    return "INVESTIGATE"


# --- main -----------------------------------------------------------------------------

def _load_prefill_sanity(path, decode_settings, tokenizer, dataset):
    """Load a prefill reference and map it onto the decode eval set for the sanity check."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    prefill_settings = blob.pop("settings")
    blob.pop("host", None)
    if prefill_settings.get("mode", "prefill") != "prefill":
        raise SystemExit(f"--prefill-reference {path} is not a prefill-mode reference")
    # Prove the reference came from this exact token stream and windowing, so flat index g
    # really is the prediction for global token g.
    _, rebuilt = load_eval_windows(
        tokenizer, prefill_settings["num_windows"] * prefill_settings["window"],
        prefill_settings["window"], prefill_settings["add_bos"], dataset,
    )
    if rebuilt["eval_set_sha256"] != prefill_settings["eval_set_sha256"]:
        raise SystemExit(
            f"--prefill-reference {path} was not built from this corpus and windowing "
            "(eval-set sha256 differs); positions would not line up"
        )
    plan, why = prefill_overlap_plan(decode_settings, prefill_settings, blob)
    if plan is None:
        raise SystemExit(f"no usable overlap with --prefill-reference {path}: {why}")
    print(f"prefill sanity: {len(plan['flat_idx'])} positions with identical context, in "
          f"decode windows {plan['windows']}")
    return {"path": path, "settings": prefill_settings, "ref": blob, "plan": plan}


def _prefill_sanity_metrics(sanity, run, args):
    """Decode-mode run vs the prefill reference on positions with identical context."""
    ref, idx, rows = sanity["ref"], sanity["plan"]["flat_idx"], run["rows"]["prefill"]
    m = drift_metrics(
        ref["topk_logprobs"][idx], ref["argmax_ids"][idx], run["gathered"]["prefill"],
        run["current"]["argmax_ids"][rows], run["current"]["target_logprobs"][rows],
    )
    m["verdict"] = classify_band(
        m["mean_kl"], m["top1_agreement_pct"],
        args.band_pass_kl, args.band_pass_top1, args.band_fail_kl, args.band_fail_top1,
    )
    m["aligned_decode_windows"] = sanity["plan"]["windows"]
    m["prefill_reference"] = sanity["path"]
    print(f"\nPREFILL SANITY  decode run vs prefill reference, {m['tokens_scored']} positions")
    print(f"PREFILL SANITY  KL {m['mean_kl']:.3e}  top-1 {m['top1_agreement_pct']:.4f}%  "
          f"-> {m['verdict']}  (expected PASS, not 0: decode and prefill use different "
          f"kernels)")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="eager", choices=sorted(VARIANTS))
    p.add_argument("--max-cache-len", type=int, default=None,
                   help="static-cache length, decode mode only. Default: prompt + generated "
                        "tokens per window, rounded up to a multiple of 128")
    p.add_argument("--mode", choices=("prefill", "decode"), default="prefill",
                   help="prefill: one forward pass per window (default, unchanged). decode: "
                        "teacher-forced generate(), so the static cache, torch.compile and "
                        "CUDA graphs are exercised. The two modes need separate references.")
    p.add_argument("--decode-windows", type=int, default=DECODE_WINDOWS)
    p.add_argument("--decode-window-len", type=int, default=DECODE_WINDOW_LEN)
    p.add_argument("--decode-prompt-len", type=int, default=DECODE_PROMPT_LEN,
                   help="tokens after BOS given as the prompt; the rest of each window is "
                        "generated under teacher forcing")
    p.add_argument("--prefill-reference", metavar="PATH",
                   help="decode mode: also compare against this prefill reference on "
                        "positions with identical context, as a sanity check. Expect the "
                        "PASS band, not 0 - decode uses different kernels than prefill.")
    p.add_argument("--save-reference", metavar="PATH")
    p.add_argument("--reference", metavar="PATH")
    p.add_argument("--self-check", action="store_true",
                   help="in-process save+compare against the same model. Convenience "
                        "smoke test only - a fresh-process --save-reference then "
                        "--reference run is the real check, since it also catches "
                        "nondeterminism and load-path differences.")
    p.add_argument("--expect", choices=("none", "no-drift", "drift", "pass-band"),
                   default="none",
                   help="assert on the outcome. 'no-drift' requires KL <= tol and top-1 "
                        ">= min; 'drift' requires the opposite, which is what makes the "
                        "harness falsifiable. 'pass-band' applies the pre-registered bands "
                        "for optimized variants: exit 0 on PASS, 1 on FAIL, 2 on "
                        "INVESTIGATE.")
    p.add_argument("--band-pass-kl", type=float, default=BAND_PASS_KL,
                   help="PASS requires mean KL strictly below this")
    p.add_argument("--band-pass-top1", type=float, default=BAND_PASS_TOP1,
                   help="PASS requires top-1 agreement %% at or above this")
    p.add_argument("--band-fail-kl", type=float, default=BAND_FAIL_KL,
                   help="FAIL if mean KL is at or above this (the negative control)")
    p.add_argument("--band-fail-top1", type=float, default=BAND_FAIL_TOP1,
                   help="FAIL if top-1 agreement %% is at or below this (the negative "
                        "control)")
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
    if args.mode == "prefill" and args.variant in GENERATE_ONLY_VARIANTS:
        raise SystemExit(
            f"variant {args.variant!r} cannot be scored in prefill mode. Its static cache and "
            "compiled decode step exist only inside generate(), but prefill mode scores each "
            "window with one plain forward pass, so the variant would run as plain eager and "
            "report KL 0 - a false PASS. Use --mode decode, against a decode-mode eager "
            "reference."
        )
    if args.prefill_reference and args.mode != "decode":
        raise SystemExit("--prefill-reference is a decode-mode sanity check; add --mode decode")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - this evaluation needs an NVIDIA GPU.")

    # AutoTokenizer, not AutoProcessor: this is text-only scoring, and the multimodal
    # processor pulls in PIL for image handling we never use.
    from transformers import AutoTokenizer

    host = print_host()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    add_bos = (not args.no_bos) and tokenizer.bos_token_id is not None
    if args.mode == "decode":
        windows, settings = load_decode_windows(
            tokenizer, args.decode_windows, args.decode_window_len, args.decode_prompt_len,
            add_bos, args.dataset,
        )
        print(f"eval set: decode, {settings['decode_windows']} x "
              f"{settings['decode_window_len']} tokens, prompt "
              f"{settings['prompt_tokens_with_bos']} incl. BOS, "
              f"{settings['scored_positions']} scored decode steps, "
              f"sha256={settings['eval_set_sha256'][:16]}\n")
        max_cache_len = resolve_max_cache_len(
            args.variant, args.max_cache_len, settings["prompt_tokens_with_bos"],
            settings["generated_per_window"],
        )
    else:
        windows, settings = load_eval_windows(
            tokenizer, args.num_tokens, args.window, add_bos, args.dataset
        )
        settings["mode"] = "prefill"
        print(f"eval set: {settings['num_windows']} x {args.window} tokens, "
              f"bos={'yes' if add_bos else 'no'}, sha256={settings['eval_set_sha256'][:16]}\n")
        max_cache_len = resolve_max_cache_len(args.variant, args.max_cache_len, args.window, 0)
    settings["topk"] = args.topk
    settings["model"] = MODEL_ID
    settings["dtype"] = "bfloat16"

    model, gen_kwargs, variant_meta = build(args.variant, max_cache_len)
    print(f"variant: {args.variant}  cache={variant_meta['cache']['class']}  "
          f"compile_requested={variant_meta['compile']['requested']}")

    if args.perturb > 0:
        pert = perturb_one_layer(
            model, args.perturb, args.perturb_seed, args.perturb_pattern
        )
        variant_meta["perturb"] = pert
        print(f"PERTURBED {pert['param']} {pert['shape']} "
              f"scale={pert['scale']:g} (relative to weight std {pert['weight_std']:.4g}); "
              f"norm {pert['norm_before']:.2f} -> {pert['norm_after']:.2f}")

    sanity = None
    if args.prefill_reference:
        sanity = _load_prefill_sanity(args.prefill_reference, settings, tokenizer, args.dataset)

    t0 = time.perf_counter()

    if args.save_reference or args.self_check:
        if args.mode == "decode":
            plans = {"prefill": sanity["plan"]["gather"]} if sanity else {}
            run = run_decode(model, windows, settings["prompt_tokens_with_bos"], args.topk,
                             gen_kwargs, plans)
            ref = run["current"]
            settings["processor_order"] = run["processor_order"]
            if sanity:
                sanity["result"] = _prefill_sanity_metrics(sanity, run, args)
                settings["prefill_sanity"] = sanity["result"]
        else:
            ref = score_reference(model, windows, args.topk)
        path = args.save_reference or "self_check_ref.pt"
        settings["reference_variant"] = args.variant
        torch.save({"settings": settings, "host": host, **ref}, path)
        ppl = float((-ref["target_logprobs"].mean()).exp())
        print(f"reference saved to {path}: "
              f"{ref['argmax_ids'].numel()} positions, top-{args.topk}, "
              f"perplexity {ppl:.3f}")
        if sanity and not args.self_check:
            out = args.out or "drift_decode_vs_prefill.json"
            with open(out, "w") as f:
                json.dump({"variant": args.variant, "variant_meta": variant_meta,
                           "host": host_info(), "settings": settings,
                           "prefill_sanity": sanity["result"]}, f, indent=2)
            print(f"wrote {out}")
        if not args.self_check:
            print(f"wall {time.perf_counter() - t0:.1f}s")
            return

    ref_path = args.reference or "self_check_ref.pt"
    blob = torch.load(ref_path, map_location="cpu", weights_only=False)
    ref_settings = blob.pop("settings")
    ref_host = blob.pop("host", None)

    # references saved before decode mode existed carry no "mode" key; they are prefill
    if ref_settings.get("mode", "prefill") != args.mode:
        raise SystemExit(
            f"mode mismatch: reference is {ref_settings.get('mode', 'prefill')!r}, this run is "
            f"{args.mode!r}. Decode and prefill use different kernels and different evaluation "
            "sets, so each needs its own reference."
        )
    if args.mode == "decode":
        keys = ("dataset", "decode_window_len", "decode_windows", "decode_prompt_len",
                "add_bos", "topk", "eval_set_sha256")
    else:
        keys = ("dataset", "window", "num_windows", "add_bos", "topk", "eval_set_sha256")
    for key in keys:
        if ref_settings.get(key) != settings.get(key):
            raise SystemExit(
                f"settings mismatch on {key!r}: reference={ref_settings.get(key)!r} "
                f"current={settings.get(key)!r}. KL across different evaluation sets is "
                "meaningless."
            )

    if args.mode == "decode":
        plans = {"ref": reference_gather_plan(blob, settings["decode_windows"],
                                              settings["generated_per_window"])}
        if sanity:
            plans["prefill"] = sanity["plan"]["gather"]
        run = run_decode(model, windows, settings["prompt_tokens_with_bos"], args.topk,
                         gen_kwargs, plans)
        settings["processor_order"] = run["processor_order"]
        result = drift_metrics(
            blob["topk_logprobs"], blob["argmax_ids"], run["gathered"]["ref"],
            run["current"]["argmax_ids"], run["current"]["target_logprobs"],
        )
        if sanity:
            sanity["result"] = _prefill_sanity_metrics(sanity, run, args)
    else:
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

    band = {
        "pass_kl_below": args.band_pass_kl,
        "pass_top1_at_least": args.band_pass_top1,
        "fail_kl_at_least": args.band_fail_kl,
        "fail_top1_at_most": args.band_fail_top1,
        "verdict": classify_band(
            result["mean_kl"], result["top1_agreement_pct"],
            args.band_pass_kl, args.band_pass_top1,
            args.band_fail_kl, args.band_fail_top1,
        ),
    }
    print(f"band verdict          {band['verdict']}")

    default_out = f"drift_{args.variant}_decode.json" if args.mode == "decode" else (
        f"drift_{args.variant}.json")
    out = args.out or default_out
    with open(out, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "variant": args.variant,
                "variant_meta": variant_meta,
                "host": host_info(),
                "reference_host": ref_host,
                "settings": settings,
                "reference_path": ref_path,
                "wall_seconds": wall,
                **result,
                "band": band,
                **({"prefill_sanity": sanity["result"]} if sanity else {}),
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
        elif expect == "drift":
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
        elif expect == "pass-band":
            print(f"PASS-BAND  PASS        KL < {band['pass_kl_below']:.0e} "
                  f"and top-1 >= {band['pass_top1_at_least']}%")
            print(f"PASS-BAND  FAIL        KL >= {band['fail_kl_at_least']:.1e} "
                  f"or top-1 <= {band['fail_top1_at_most']}%   (negative control)")
            print(f"PASS-BAND  INVESTIGATE anything in between")
            print(f"PASS-BAND  measured    KL {kl:.3e}, top-1 {t1:.4f}%  ->  "
                  f"{band['verdict']}")
            if band["verdict"] == "FAIL":
                raise SystemExit(1)
            if band["verdict"] == "INVESTIGATE":
                raise SystemExit(2)


if __name__ == "__main__":
    main()
