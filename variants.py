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
from functools import partial

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


def _fused_rmsnorm(max_cache_len, text_config):
    # Same cache and generate() arguments as eager; the difference is applied to the loaded
    # model by patch_fused_rmsnorm(), so it shows up in plain forward passes too.
    kwargs, meta = _eager(max_cache_len, text_config)
    meta["kernel"] = {
        "name": "fused RMSNorm (Triton, one program per row)",
        "file": "kernels/rmsnorm.py",
        "replaces": "Gemma4RMSNorm.forward on the text decoder",
    }
    return kwargs, meta


def _norm_size(name, module, modules):
    if getattr(module, "with_scale", False):
        return module.weight.shape[0]
    # a weightless norm (v_norm) stores no size; its parent attention knows the head dim
    return getattr(modules[name.rsplit(".", 1)[0]], "head_dim", None)


def patch_fused_rmsnorm(model, verify=True):
    """Route every text-decoder Gemma4RMSNorm through the fused kernel. Returns a report.

    Vision and audio norms are left eager on purpose: they never run during text decode, and
    their inputs are not covered by tests/test_rmsnorm.py, so replacing them would change code
    that nothing measures. Anything that is not a Gemma4RMSNorm is left alone as well.
    """
    import torch.nn as nn
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    from kernels.rmsnorm import rmsnorm

    def fused_forward(module):
        def forward(hidden_states):
            return rmsnorm(hidden_states, module.weight if module.with_scale else None,
                           module.eps, module.with_scale)
        return forward

    modules = dict(model.named_modules())
    replaced, left, shapes = Counter(), Counter(), {}
    for name, module in modules.items():
        if isinstance(module, Gemma4RMSNorm):
            if ".language_model." not in name:
                left["Gemma4RMSNorm (vision/audio): not on the text decode path, untested inputs"] += 1
                continue
            n = _norm_size(name, module, modules)
            key = f"Gemma4RMSNorm N={n} with_scale={module.with_scale}"
            replaced[key] += 1
            shapes[key] = (n, module.with_scale, module.eps)
            module.forward = fused_forward(module)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            left[f"{type(module).__name__}: not an RMSNorm"] += 1

    report = {
        "replaced_total": sum(replaced.values()),
        "replaced": dict(replaced),
        "left_on_eager": dict(left),
    }
    if verify:
        report["build_check"] = _verify_fused(model, shapes)
    return report


def _verify_fused(model, shapes):
    """On a GPU, compare the kernel with the reference for each distinct norm shape at build
    time, so a broken kernel stops the run before anything is measured."""
    if model.device.type != "cuda":
        return "skipped: model not on CUDA"
    from kernels.rmsnorm import rmsnorm, rmsnorm_reference

    worst = {}
    for key, (n, with_scale, eps) in shapes.items():
        gen = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(8, n, device="cuda", dtype=model.dtype, generator=gen) * 3
        w = (1 + 0.1 * torch.randn(n, device="cuda", dtype=model.dtype, generator=gen)
             if with_scale else None)
        err = (rmsnorm(x, w, eps, with_scale).float()
               - rmsnorm_reference(x, w, eps, with_scale).float()).abs().max().item()
        worst[key] = err
        if not err < 0.1:
            raise SystemExit(f"fused RMSNorm disagrees with the reference for {key}: max abs "
                             f"error {err:.3g}. Run tests/test_rmsnorm.py before measuring.")
    return {"max_abs_error_vs_reference": worst}


def _graph_common(max_cache_len, text_config, fused):
    # Same StaticCache as "static"; no generate() kwargs because the runner has its own loop.
    _, meta = _static(max_cache_len, text_config)
    meta["cache"]["source"] = ("StaticCache built by graph_decode.GraphDecoder with generate()'s "
                               "own arguments (config=text_config, max_cache_len, offloading=False)")
    meta["compile"] = {
        "requested": False,
        "disable_compile": None,
        "reason": "no torch.compile: each decode step replays a manually captured CUDA graph",
        "config": None,
    }
    meta["decode"] = {
        "loop": "graph_decode.GraphDecoder: eager prefill, one CUDA graph replay per decode step",
        "norms": "fused Triton RMSNorm" if fused else "eager Gemma4RMSNorm",
    }
    return {}, meta


def _graph(max_cache_len, text_config):
    return _graph_common(max_cache_len, text_config, fused=True)


def _graph_eager(max_cache_len, text_config):
    return _graph_common(max_cache_len, text_config, fused=False)


def wrap_graph(model, max_cache_len, meta):
    """Capture the decode step and hand back the runner in place of the model."""
    from graph_decode import GraphDecoder

    runner = GraphDecoder(model, max_cache_len)
    meta["graph"] = runner.meta
    return runner


def _fused_loop(max_cache_len, text_config):
    kwargs, meta = _graph_common(max_cache_len, text_config, fused=True)
    meta["compile"]["reason"] = ("no torch.compile and no CUDA graph: each decode step runs "
                                 "eagerly in the graph runner's own loop")
    meta["decode"]["loop"] = ("graph_decode.GraphDecoder(eager_steps=True): eager prefill, one "
                              "eager step per decode token - no capture")
    return kwargs, meta


def wrap_fused_loop(model, max_cache_len, meta):
    """The graph runner's decode loop, static cache and dict mask, stepping eagerly."""
    from graph_decode import GraphDecoder

    runner = GraphDecoder(model, max_cache_len, capture=False, eager_steps=True)
    meta["loop"] = runner.meta
    return runner


# --- naive weight-only quantization (quant/linear.py) -----------------------------------
# Every text-decoder nn.Linear except lm_head - the same 276 matrices as eval_drift.py's
# --perturb-target all-linears - becomes a QuantLinear built with quant/primitives.py. lm_head
# stays bf16 by default: it is tied to the embedding, so quantizing it adds a copy instead of
# replacing one (quantize_lm_head=True does it anyway). forward() dequantizes the whole weight
# to bf16 every call, so these save memory at rest, not bytes per token.

QUANT_GROUP_SIZE = 128
EXPECTED_TEXT_LINEARS = 276


def _quant_naive(fmt):
    def builder(max_cache_len, text_config):
        kwargs, meta = _eager(max_cache_len, text_config)
        meta["quant"] = {
            "format": fmt,
            "scheme": ("symmetric, per output channel" if fmt == "int8"
                       else f"symmetric, group-wise along the input dim, group {QUANT_GROUP_SIZE}"),
            "group_size": QUANT_GROUP_SIZE if fmt == "int4" else None,
            "scales": "bf16",
            "forward": "dequantize the whole weight to bf16, then F.linear (naive)",
        }
        return kwargs, meta
    return builder


def _text_linear_names(model, quantize_lm_head):
    import torch.nn as nn
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and ".language_model." in f".{n}"
             and not n.endswith("lm_head")]
    if len(names) != EXPECTED_TEXT_LINEARS:
        raise SystemExit(f"found {len(names)} text-decoder linears, expected "
                         f"{EXPECTED_TEXT_LINEARS} - the model is not the one this was audited on")
    if quantize_lm_head:
        if not isinstance(getattr(model, "lm_head", None), nn.Linear):
            raise SystemExit("quantize_lm_head: model has no nn.Linear lm_head")
        names.append("lm_head")
    return names


def patch_quant(model, fmt, group_size=QUANT_GROUP_SIZE, quantize_lm_head=False, verify=True):
    """Replace the text-decoder linears with QuantLinear and free their bf16 weights."""
    import gc
    import weakref

    import torch.nn.functional as F

    from quant.linear import QuantLinear

    names = _text_linear_names(model, quantize_lm_head)
    modules = dict(model.named_modules())
    tied = {id(p) for n, p in model.named_parameters(remove_duplicate=False)
            if not n.startswith("lm_head")}
    counts, bytes_before, bytes_after, refs, check = Counter(), 0, 0, [], None
    on_cuda = model.device.type == "cuda"
    mem_before = torch.cuda.memory_allocated() if on_cuda else None

    for i, name in enumerate(names):
        linear = modules[name]
        w = linear.weight
        bytes_before += w.numel() * w.element_size()
        ql = QuantLinear.from_linear(linear, fmt, group_size)
        bytes_after += ql.stored_bytes()
        if i == 0 and verify:
            check = _verify_quant(name, w, ql, F)
        counts[name.rsplit(".", 1)[-1]] += 1
        parent, _, child = name.rpartition(".")
        setattr(modules[parent] if parent else model, child, ql)
        if not (name == "lm_head" and id(w) in tied):
            refs.append((name, weakref.ref(w)))
        del linear, w, ql

    modules = None
    gc.collect()
    alive = [n for n, r in refs if r() is not None]
    if alive:
        raise SystemExit(f"{len(alive)} replaced bf16 weights are still referenced (e.g. "
                         f"{alive[:3]}) - the memory saving would not be real")
    if on_cuda:
        torch.cuda.empty_cache()

    report = {
        "format": fmt,
        "group_size": group_size if fmt == "int4" else None,
        "replaced_total": len(names),
        "replaced": dict(counts),
        "lm_head": "quantized (tied bf16 embedding kept)" if quantize_lm_head else "bf16",
        "weight_bytes_before": bytes_before,
        "weight_bytes_after": bytes_after,
        "bf16_weights_freed": len(refs),
    }
    if on_cuda:
        report["cuda_allocated_drop_bytes"] = mem_before - torch.cuda.memory_allocated()
    if verify:
        report["build_check"] = check
    return report


@torch.no_grad()
def _verify_quant(name, w, ql, F):
    """For one layer: forward == F.linear on the dequantized weight, bit for bit, and the
    dequantized weight is within scale/2 of the original."""
    gen = torch.Generator(device=w.device).manual_seed(0)
    x = torch.randn(3, w.shape[1], device=w.device, generator=gen).to(w.dtype)
    deq = ql.dequantized_weight()
    if not torch.equal(ql(x), F.linear(x, deq, ql.bias)):
        raise SystemExit(f"QuantLinear({name}) forward differs from F.linear on its dequantized "
                         "weight")
    exact = ql.dequantized_weight(torch.float32)
    if ql.fmt == "int8":
        half = ql.scale.float().unsqueeze(1) / 2
    else:
        half = ql.scale.float().repeat_interleave(ql.group_size, dim=1) / 2
    err = (exact - w.float()).abs()
    worst = (err - half * (1 + 2**-20) - w.float().abs() * 2**-23).max().item()
    if worst > 0:
        raise SystemExit(f"QuantLinear({name}) dequantized weight exceeds scale/2 by {worst:.3g}")
    return {"layer": name, "forward_equals_F_linear_on_dequantized": True,
            "max_abs_error_vs_bf16_weight": err.max().item(),
            "max_abs_error_of_bf16_dequant": (deq.float() - w.float()).abs().max().item()}




VARIANTS = {
    "eager": {"builder": _eager, "static_cache": False},
    "static": {"builder": _static, "static_cache": True},
    "compile": {"builder": _compile, "static_cache": True},
    # Not generate-only: the norms change in every forward pass, so prefill-mode drift sees it.
    "fused_rmsnorm": {"builder": _fused_rmsnorm, "static_cache": False,
                      "patch": patch_fused_rmsnorm},
    # Manual CUDA graph decode (graph_decode.py), with and without the fused norms, so the
    # graph and the fusion can be attributed separately.
    "graph": {"builder": _graph, "static_cache": True, "patch": patch_fused_rmsnorm,
              "wrap": wrap_graph},
    "graph_eager": {"builder": _graph_eager, "static_cache": True, "wrap": wrap_graph},
    # The graph runner's loop without the graph: separates "the CUDA graph" from "dropping
    # generate()" (fused_rmsnorm -> fused_loop -> graph).
    "fused_loop": {"builder": _fused_loop, "static_cache": True, "patch": patch_fused_rmsnorm,
                   "wrap": wrap_fused_loop},
    # Naive weight-only quantization. Not generate-only: the weights change in every forward,
    # so prefill-mode drift sees it.
    "int8_naive": {"builder": _quant_naive("int8"), "static_cache": False,
                   "patch": partial(patch_quant, fmt="int8")},
    "int4_naive": {"builder": _quant_naive("int4"), "static_cache": False,
                   "patch": partial(patch_quant, fmt="int4")},
}

# Variants whose distinguishing behaviour exists only inside generate(). A tool that scores
# with a plain forward pass (eval_drift.py) would silently run these as eager.
GENERATE_ONLY_VARIANTS = frozenset({"static", "compile", "graph", "graph_eager", "fused_loop"})


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
    patch = VARIANTS[variant].get("patch")
    if patch is not None:
        meta["patch"] = patch(model)
    wrap = VARIANTS[variant].get("wrap")
    if wrap is not None:
        model = wrap(model, max_cache_len, meta)
    return model, kwargs, meta


def model_weight_bytes(model):
    """Bytes held by the model's parameters and buffers, each tensor counted once (tied
    weights share storage). Unwraps the graph runner."""
    module = model.model if getattr(model, "is_graph_runner", False) else model
    seen, total = set(), 0
    for t in list(module.parameters()) + list(module.buffers()):
        key = (t.device, t.data_ptr())
        if key not in seen:
            seen.add(key)
            total += t.numel() * t.element_size()
    return total


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
