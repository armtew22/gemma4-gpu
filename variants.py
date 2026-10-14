"""Single source of truth for model variants.

benchmark.py, profile_decode.py and eval_drift.py all obtain their model through build(), so
a variant name means the same weights, the same cache and the same generate() arguments in
every tool. Behaviour below was read from the installed transformers 5.19.0 source:

- cache_implementation="static" always instantiates StaticCache. The older "hybrid" and
  "sliding_window" names are deprecated aliases; StaticCache infers the sliding/full layout
  per layer from the config, so the cache *class* is StaticCache either way and the layer
  composition is what tells you how sliding-window layers are handled.
- On CUDA, generate() auto-compiles the forward whenever the cache is compileable (StaticCache
  is, DynamicCache is not) unless generation_config.disable_compile is set.
- The compiled call is used for decode steps only. Prefill always runs eager: _prefill calls
  self(...) directly.
- generate() remembers the largest static cache it has built (model._previous_max_cache_length)
  and never shrinks it, so finalize() records the effective length, not just the request.
"""

import math
from collections import Counter

import torch
import transformers
from transformers import AutoModelForMultimodalLM, CompileConfig, DynamicCache, StaticCache

MODEL_ID = "google/gemma-4-E2B-it"
CACHE_LEN_MULTIPLE = 128


def load_model():
    """The exact loading call every tool used before this module existed.

    Do not change it: eager results from earlier runs stay comparable only while this is
    identical.
    """
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16
    )
    model.eval()
    return model


def default_max_cache_len(prompt_len, decode_tokens):
    """prompt_len + decode_tokens, rounded up to a multiple of 128."""
    return math.ceil((prompt_len + decode_tokens) / CACHE_LEN_MULTIPLE) * CACHE_LEN_MULTIPLE


def resolve_max_cache_len(variant, explicit, prompt_len, decode_tokens):
    """The max cache length a tool should pass to build().

    Eager uses a DynamicCache with no fixed length, so it gets None unless one was given
    explicitly (which build() then records as ignored).
    """
    if explicit is not None:
        return explicit
    if VARIANTS[variant]["static_cache"]:
        return default_max_cache_len(prompt_len, decode_tokens)
    return None


def _describe_cache(cache, source):
    kinds = Counter(type(layer).__name__ for layer in cache.layers)
    lengths = {}
    for layer in cache.layers:
        lengths.setdefault(type(layer).__name__, set()).add(getattr(layer, "max_cache_len", None))
    return {
        "class": type(cache).__name__,
        "source": source,
        "compileable": bool(getattr(cache, "is_compileable", False)),
        "num_cache_layers": len(cache.layers),
        "layer_classes": dict(kinds),
        "layer_max_cache_len": {
            k: sorted(v, key=lambda x: (x is None, x if x is not None else 0))
            for k, v in lengths.items()
        },
    }


# --- variants -------------------------------------------------------------------------
# Each builder takes (max_cache_len, text_config) and returns (generate_kwargs, meta). It
# needs only the config, never the weights, so describe() can run on a laptop.

def _eager(max_cache_len, text_config):
    cache = DynamicCache(config=text_config)
    meta = {
        "cache": _describe_cache(
            cache, "generate() default: cache_implementation is None -> "
                   "DynamicCache(config=text_config)"
        ),
        "max_cache_len": None,
        "compile": {
            "requested": False,
            "disable_compile": None,
            "reason": "generate() auto-compiles only with a compileable cache; DynamicCache "
                      "is not compileable",
            "config": None,
        },
    }
    if max_cache_len is not None:
        meta["max_cache_len_ignored"] = max_cache_len
    # Empty on purpose: generate() must receive exactly the arguments it always has.
    return {}, meta


def _static(max_cache_len, text_config):
    cache = StaticCache(config=text_config, max_cache_len=max_cache_len, offloading=False)
    kwargs = {
        "cache_implementation": "static",
        "max_cache_len": max_cache_len,
        # Without this, generate() on CUDA would silently compile the decode step, and
        # "static" would measure compile + static cache instead of the cache alone.
        "disable_compile": True,
    }
    meta = {
        "cache": _describe_cache(
            cache, "StaticCache built with generate()'s own arguments "
                   "(_prepare_static_cache: config=text_config, max_cache_len, "
                   "offloading=False)"
        ),
        "max_cache_len": max_cache_len,
        "compile": {
            "requested": False,
            "disable_compile": True,
            "reason": "disable_compile=True overrides generate()'s auto-compile for "
                      "compileable caches",
            "config": None,
        },
    }
    return kwargs, meta


def _compile(max_cache_len, text_config):
    compile_config = CompileConfig(mode="reduce-overhead")
    kwargs = {
        "cache_implementation": "static",
        "max_cache_len": max_cache_len,
        "disable_compile": False,
        "compile_config": compile_config,
    }
    _, meta = _static(max_cache_len, text_config)
    meta["compile"] = {
        "requested": True,
        "disable_compile": False,
        "scope": "decode steps inside generate() only; prefill runs eager",
        "requires": "CUDA device and a compileable cache (StaticCache)",
        "reason": "static cache makes generate() compile the decode forward via "
                  "get_compiled_call(compile_config)",
        "config": compile_config.to_dict(),
    }
    return kwargs, meta


# --- future: "graph" ------------------------------------------------------------------
# Manual torch.cuda.graph capture of the decode step with a static cache, and NO
# torch.compile. The point is attribution: "compile" combines kernel fusion with CUDA graphs,
# and "graph" would isolate how much of the win is launch overhead alone. It needs its own
# capture per (batch, cache length) and a forward wrapper or decode loop that replays it.
# Not implemented. Register it below once it exists, and decide whether it belongs in
# GENERATE_ONLY_VARIANTS.

VARIANTS = {
    "eager": {"builder": _eager, "static_cache": False},
    "static": {"builder": _static, "static_cache": True},
    "compile": {"builder": _compile, "static_cache": True},
}

# Variants whose distinguishing behaviour exists only inside generate(). A tool that scores
# with a plain forward pass (eval_drift.py) would silently run these as eager.
GENERATE_ONLY_VARIANTS = frozenset({"static", "compile"})


def describe(variant, max_cache_len, config):
    """generate_kwargs and meta for a variant, from the model config alone."""
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
    spec = VARIANTS[variant]
    if spec["static_cache"] and max_cache_len is None:
        raise SystemExit(
            f"variant {variant!r} uses a static cache and needs max_cache_len; use "
            "resolve_max_cache_len()"
        )
    text_config = config.get_text_config(decoder=True)
    kwargs, meta = spec["builder"](max_cache_len, text_config)
    meta.update({
        "variant": variant,
        "model": MODEL_ID,
        "dtype": "bfloat16",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "num_hidden_layers": text_config.num_hidden_layers,
        "generate_kwargs": {
            k: (v.to_dict() if isinstance(v, CompileConfig) else v) for k, v in kwargs.items()
        },
        "generate_only": variant in GENERATE_ONLY_VARIANTS,
    })
    meta["compile"]["engaged"] = None  # filled in by finalize() after a generate() call
    return kwargs, meta


def build(variant, max_cache_len=None):
    """Load the model and return (model, generate_kwargs, meta).

    Every generate() call must pass **generate_kwargs. meta goes into every results JSON.
    """
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
    model = load_model()
    kwargs, meta = describe(variant, max_cache_len, model.config)
    return model, kwargs, meta


def finalize(model, meta):
    """Record what actually happened at run time. Call after at least one generate()."""
    effective = getattr(model, "_previous_max_cache_length", None)
    engaged = hasattr(model, "_compiled_call")
    meta["cache"]["max_cache_len_effective"] = effective
    meta["compile"]["engaged"] = engaged

    problems = []
    if meta["compile"]["requested"] and not engaged:
        problems.append("compile was requested but generate() never built a compiled call")
    if not meta["compile"]["requested"] and engaged:
        problems.append("compile was NOT requested but generate() built a compiled call")
    requested = meta.get("max_cache_len")
    if requested is not None and effective is not None and effective != requested:
        problems.append(
            f"static cache effective length {effective} != requested {requested} "
            "(generate() grows it to fit a call and never shrinks it)"
        )
    meta["consistency_problems"] = problems
    for message in problems:
        print(f"WARNING variants: {message}")
    return meta
