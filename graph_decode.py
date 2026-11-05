"""Manual CUDA graph decode for Gemma 4: one captured graph per decode step, no torch.compile.

torch.compile with a static cache falls back to eager on Gemma 4 (a graph break on
shared_kv_states and the static cache's recompile guards). This captures one decode step with
torch.cuda.CUDAGraph directly and replays it, with its own decode loop instead of generate().

A CUDA graph replays recorded kernels; no Python runs during replay. Every value that changes
between steps must therefore be a GPU tensor updated in place before the replay, and any
Python-side state read while capturing is frozen into the graph. Audit of transformers 5.19.0
(models/gemma4/modeling_gemma4.py, cache_utils.py, masking_utils.py), item by item:

 1. Token id - static GPU buffer `input_ids` [1, 1], copy_() in place before each replay.
 2. Position - static GPU buffer `position_ids` [1, 1], fill_() before each replay, and always
    passed explicitly. Omitted, Gemma4TextModel derives it from past_key_values.get_seq_length(),
    which for layer 0 (sliding) is the Python int cumulative_length_int: frozen at capture.
 3. Rotary cos/sin - computed inside the graph from the position buffer. Rope types here are
    "default" (sliding) and "proportional" (full); @dynamic_rope_update is a no-op for both.
    Guard: refuse any other rope type.
 4. cache_position - not an argument of the 5.19 text forward. StaticLayer writes at
    arange(1) + its GPU counter `cumulative_length` (index_copy_, then add_(1) in place), so
    replays advance the write slot on the GPU.
 5. Sliding-window cache branch - StaticSlidingWindowLayer picks "not full" / "becoming full" /
    "full (roll)" in Python from cumulative_length_int. The graph freezes the branch seen at
    capture ("not full") and the int stops advancing during replay. Safe only while every step
    stays "not full": guard max_cache_len <= sliding_window at build, and
    prompt + new_tokens - 1 <= max_cache_len on every call.
 6. Masks - masking_utils is NOT safe inside the graph. With attention_mask=None and a 1-token
    query, _ignore_causal_mask_sdpa skips the mask, so the query would attend to unwritten cache
    slots (zero keys score 0, not -inf). A fixed-length 2D mask instead makes fast_all() turn a
    GPU tensor into a Python bool, a sync that breaks capture. And for sliding layers q_offset is
    get_query_offset() = the frozen Python int. Instead the step computes one boolean mask,
    kv_index <= position, inside the graph from the position buffer, and passes it as the dict
    {"full_attention", "sliding_attention"}, which Gemma4Model and Gemma4TextModel use as-is.
    With max_cache_len <= sliding_window, batch 1 and no padding, it is exactly the mask
    masking_utils builds for eager static decode - checked at build time, see below.
 7. Mask sizes / kv_offset - get_mask_sizes() Python ints. Not called in the graph (masks are
    bypassed); constant in this regime anyway (kv_length = max_cache_len, kv_offset = 0).
 8. shared_kv_states (20 KV-sharing layers) - a fresh UserDict built and consumed within one
    forward, holding what past_key_values.update() returns: in the "not full" regime, the static
    cache buffers themselves. Fixed addresses, graph-safe. Never passed in or reused across steps.
 9. Per-layer inputs (PLE) - computed in the graph from the input_ids buffer.
10. Multimodal wrapper - get_placeholder_mask does tensor comparisons only; the image, audio and
    video branches are behind `is not None` checks on arguments that are always None here.
11. Logits - lm_head on the last position plus final_logit_softcapping (tanh): tensor ops, read
    from the graph's static output after each replay and copied to fp32 as generate() does.
12. Cache reset between calls - StaticCache.reset() zeroes keys, values and the GPU counter in
    place and sets cumulative_length_int = 0. Buffer addresses are unchanged, so the graph holds.
13. Warmup and capture side effects - warmup executes the step (advancing both counters);
    capture runs the Python again (advancing cumulative_length_int) without executing kernels.
    The cache is reset after capture.
14. Host syncs (.item(), .tolist()) - any in the forward raises during capture: loud, not silent.
15. Fused RMSNorm ("graph" variant) - the Triton kernels compile during warmup; their launches
    and the wrapper's torch.empty are captured on the current stream and the graph's pool.

Build-time self-check, refusing to build on any mismatch: from identical cache states, decode
the same tokens three ways - (A) the step run eagerly, (B) graph replay, (C) a plain eager
forward with a growing 2D attention_mask through masking_utils, as generate() does it. B must
equal A bit for bit (nothing is frozen) and A must equal C bit for bit (the dict mask is the
same mask).

The decode loop also drops generate()'s per-step stopping-criteria host sync, so comparing
"graph_eager" with "static" measures the graph and the loop together, not the graph alone.
"""

import time
from contextlib import nullcontext

import torch
from torch.profiler import record_function
from transformers import StaticCache
from transformers.cache_utils import StaticLayer, StaticSlidingWindowLayer

SAFE_ROPE_TYPES = {"default", "proportional"}
SAFE_CACHE_LAYERS = (StaticLayer, StaticSlidingWindowLayer)


class GraphDecoder:
    """Greedy decode for batch 1: eager prefill, then one CUDA graph replay per new token.

    Exposes generate(...) with the subset of the HF signature this repo's tools use, so
    benchmark.py, profile_decode.py and eval_drift.py drive it like a model.
    """

    is_graph_runner = True

    def __init__(self, model, max_cache_len, capture=True, warmup_steps=3, self_check=True):
        self.model = model
        self.config = model.config
        self.max_cache_len = max_cache_len
        text = model.config.get_text_config(decoder=True)
        self._guard(text, capture)

        self.cache = StaticCache(config=text, max_cache_len=max_cache_len, offloading=False)
        bad = sorted({type(l).__name__ for l in self.cache.layers
                      if type(l) not in SAFE_CACHE_LAYERS})
        if bad:
            raise SystemExit(f"graph runner refuses to build: unaudited cache layer types {bad}")

        dev = model.device
        self.input_ids = torch.zeros((1, 1), dtype=torch.long, device=dev)
        self.position_ids = torch.zeros((1, 1), dtype=torch.long, device=dev)
        self.kv_index = torch.arange(max_cache_len, device=dev)
        self.graph = None
        self.static_logits = None
        self.annotate = False       # profile_decode.py turns this on for its profiled run
        self.allow_eager_steps = False   # tests only: run the step eagerly instead of replaying
        self.meta = {"max_cache_len": max_cache_len, "batch_size": 1, "warmup_steps": warmup_steps}

        if capture:
            self._capture(warmup_steps)
            if self_check:
                self.meta["self_check"] = self._self_check()

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    def _guard(self, text, capture):
        problems = []
        if capture and self.model.device.type != "cuda":
            problems.append("CUDA graph capture needs the model on a CUDA device")
        window = getattr(text, "sliding_window", None)
        if window is not None and self.max_cache_len > window:
            problems.append(
                f"max_cache_len {self.max_cache_len} > sliding_window {window}: sliding layers "
                "could reach their 'full (roll)' branch, which capture would freeze out")
        rope = getattr(text, "rope_parameters", None) or {}
        rope_types = {v.get("rope_type") for v in rope.values() if isinstance(v, dict)}
        if rope.get("rope_type"):
            rope_types.add(rope["rope_type"])
        if rope_types - SAFE_ROPE_TYPES:
            problems.append(f"unaudited rope types {sorted(rope_types - SAFE_ROPE_TYPES)}")
        impl = getattr(self.config, "_attn_implementation", None) or getattr(
            text, "_attn_implementation", None)
        if impl != "sdpa":
            problems.append(f"attention implementation {impl!r}: the boolean dict mask is "
                            "audited for sdpa only")
        if problems:
            raise SystemExit("graph runner refuses to build:\n  - " + "\n  - ".join(problems))

    # --- the captured step -------------------------------------------------------------

    def _step(self):
        # kv_index <= position, computed from the GPU buffer so it changes with every replay
        mask = (self.kv_index <= self.position_ids[:, :1]).view(1, 1, 1, -1)
        out = self.model(
            input_ids=self.input_ids,
            position_ids=self.position_ids,
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        return out.logits

    def _prefill(self, input_ids):
        self.cache.reset()
        p = input_ids.shape[1]
        out = self.model(
            input_ids=input_ids,
            position_ids=torch.arange(p, device=input_ids.device).unsqueeze(0),
            attention_mask=torch.ones_like(input_ids),
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        return out.logits

    def _replay(self):
        if self.graph is not None:
            self.graph.replay()
        elif self.allow_eager_steps:
            self.static_logits = self._step()
        else:
            raise RuntimeError("graph not captured - refusing to fall back to eager steps")

    @torch.no_grad()
    def _capture(self, warmup_steps):
        dummy = torch.full((1, 8), 100, dtype=torch.long, device=self.device)
        self._prefill(dummy)
        self.input_ids.fill_(100)
        self.position_ids.fill_(dummy.shape[1])

        # warm up on a side stream (PyTorch's recommended pattern): compiles Triton kernels,
        # allocates workspaces and initializes cuBLAS handles before capture
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup_steps):
                self._step()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.graph(graph):
            self.static_logits = self._step()
        torch.cuda.synchronize()
        self.meta["capture_ms"] = (time.perf_counter() - t0) * 1000.0
        self.graph = graph
        self.cache.reset()  # warmup and capture both moved the counters

    @torch.no_grad()
    def _self_check(self, steps=4):
        """A eager step == B graph replay == C generate()-style eager forward, bit for bit."""
        prompt = torch.tensor([[2, 1000, 2000, 3000, 4000, 5000, 6000, 7000]], device=self.device)
        feed = [111, 222, 333, 444][:steps]
        p = prompt.shape[1]

        def run(kind):
            self._prefill(prompt)
            logits = []
            for j, tok in enumerate(feed):
                self.input_ids.fill_(tok)
                self.position_ids.fill_(p + j)
                if kind == "A":
                    out = self._step()
                elif kind == "B":
                    self.graph.replay()
                    out = self.static_logits
                else:
                    out = self.model(
                        input_ids=self.input_ids, position_ids=self.position_ids,
                        attention_mask=torch.ones((1, p + j + 1), dtype=torch.long,
                                                  device=self.device),
                        past_key_values=self.cache, use_cache=True, logits_to_keep=1,
                    ).logits
                logits.append(out[:, -1].float().clone())
            return torch.stack(logits)

        a, b, c = run("A"), run("B"), run("C")
        self.cache.reset()
        ab, ac = (a - b).abs().max().item(), (a - c).abs().max().item()
        if not torch.equal(a, b):
            raise SystemExit(f"graph runner refuses to build: graph replay differs from the eager "
                             f"step (max abs {ab:.3g}) - some per-step state was frozen at capture")
        if not torch.equal(a, c):
            raise SystemExit(f"graph runner refuses to build: the dict mask gives different logits "
                             f"from generate()'s 2D-mask path (max abs {ac:.3g})")
        return {"steps": steps, "replay_vs_eager_step": "bit-identical",
                "dict_mask_vs_generate_mask": "bit-identical"}

    # --- decode loop ----------------------------------------------------------------------

    def _label(self, i):
        return record_function(f"gen_fwd_{i:03d}") if self.annotate else nullcontext()

    @staticmethod
    def _next_token(ids, scores, processors):
        for proc in processors:
            scores = proc(ids, scores)
        return scores.argmax(dim=-1)  # greedy, as generate() does with do_sample=False

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, min_new_tokens=None,
                 do_sample=False, logits_processor=None, **unsupported):
        if unsupported:
            raise TypeError(f"graph runner does not support generate() arguments "
                            f"{sorted(unsupported)}")
        if do_sample:
            raise ValueError("graph runner decodes greedily only")
        if input_ids.shape[0] != 1:
            raise ValueError(f"graph runner is captured for batch 1, got {input_ids.shape[0]}")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids (no padding)")
        n = max_new_tokens
        if min_new_tokens not in (None, n):
            raise ValueError("graph runner always generates exactly max_new_tokens")
        p = input_ids.shape[1]
        if p + n - 1 > self.max_cache_len:
            raise ValueError(f"prompt {p} + {n} new tokens needs {p + n - 1} cache slots; "
                             f"captured with {self.max_cache_len}")

        processors = list(logits_processor or [])
        out = torch.empty((1, p + n), dtype=input_ids.dtype, device=input_ids.device)
        out[:, :p] = input_ids

        with self._label(0):
            logits = self._prefill(input_ids)
        scores = logits[:, -1].to(copy=True, dtype=torch.float32)
        token = self._next_token(out[:, :p], scores, processors)
        out[:, p] = token
        for j in range(n - 1):
            self.input_ids.copy_(token.view(1, 1))
            self.position_ids.fill_(p + j)
            with self._label(j + 1):
                self._replay()
            scores = self.static_logits[:, -1].to(copy=True, dtype=torch.float32)
            token = self._next_token(out[:, : p + j + 1], scores, processors)
            out[:, p + j + 1] = token
        return out
