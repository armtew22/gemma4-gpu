"""Fused RMSNorm kernel vs the real transformers Gemma4RMSNorm module. GPU only.

    pytest -s tests/test_rmsnorm.py            # correctness, prints max abs / rel error
    python tests/test_rmsnorm.py               # correctness summary + microbenchmark

Every (N, with_scale) case comes from walking the meta model: the 242 text-decoder norms are
N=1536 with weight (176), N=256 with weight (41) and without (12), N=512 with weight (10) and
without (3). The model runs bf16, so bf16 is the case that matters; fp32 is checked too.
"""

import os
import statistics
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    TRITON = True
except ImportError:
    TRITON = False

pytestmark = pytest.mark.skipif(not (CUDA and TRITON), reason="needs a CUDA GPU with Triton")

CASES = [(1536, True), (256, True), (256, False), (512, True), (512, False)]
ROWS = [1, 3, 7, 8, 33, 129, 1000]           # 1 and odd counts on purpose
# bf16 keeps ~8 bits of mantissa: allow two ulps relative (2 * 2^-7 = 1.6e-2) plus a small
# absolute floor for values near zero. fp32 output should agree to rounding of the reductions.
TOL = {torch.bfloat16: dict(rtol=1.6e-2, atol=1e-3), torch.float32: dict(rtol=1e-5, atol=1e-6)}


def _module(n, with_scale, dtype):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
    mod = Gemma4RMSNorm(n, eps=1e-6, with_scale=with_scale).to("cuda", dtype)
    if with_scale:
        # trained weights scatter around the ones they were initialised to
        with torch.no_grad():
            mod.weight.copy_(1 + 0.1 * torch.randn(n, device="cuda", dtype=dtype))
    return mod


def _fused(mod, x):
    from kernels.rmsnorm import rmsnorm
    return rmsnorm(x, mod.weight if mod.with_scale else None, mod.eps, mod.with_scale)


def _errors(out, ref):
    o, r = out.float(), ref.float()
    abs_err = (o - r).abs()
    rel_err = abs_err / r.abs().clamp_min(1e-6)
    return abs_err.max().item(), rel_err.max().item(), (out == ref).float().mean().item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("n,with_scale", CASES, ids=[f"N{n}-{'w' if s else 'nw'}" for n, s in CASES])
@pytest.mark.parametrize("rows", ROWS)
def test_matches_gemma4_module(n, with_scale, dtype, rows):
    torch.manual_seed(rows * 1000 + n)
    mod = _module(n, with_scale, dtype)
    x = torch.randn(rows, n, device="cuda", dtype=dtype) * 3
    with torch.no_grad():
        ref, out = mod(x), _fused(mod, x)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    max_abs, max_rel, exact = _errors(out, ref)
    print(f"\n  N={n:5d} scale={with_scale!s:5s} {str(dtype)[6:]:8s} rows={rows:4d}  "
          f"max_abs={max_abs:.2e} max_rel={max_rel:.2e} bit-exact={100 * exact:.2f}%")
    torch.testing.assert_close(out, ref, **TOL[dtype])


@pytest.mark.parametrize("n,with_scale", CASES, ids=[f"N{n}-{'w' if s else 'nw'}" for n, s in CASES])
def test_non_contiguous(n, with_scale):
    """Inputs the wrapper must copy or stride over rather than misread."""
    torch.manual_seed(n)
    mod = _module(n, with_scale, torch.bfloat16)
    base = torch.randn(5, 8, n, device="cuda", dtype=torch.bfloat16)
    cases = {
        "transposed (heads/seq swapped)": base.transpose(0, 1),          # rows not uniformly strided
        "strided last dim": torch.randn(9, 2 * n, device="cuda", dtype=torch.bfloat16)[:, ::2],
        "row slice": base[:, 2:7, :],
    }
    for name, x in cases.items():
        assert not x.is_contiguous(), name
        with torch.no_grad():
            ref, out = mod(x), _fused(mod, x)
        assert out.shape == x.shape, name
        torch.testing.assert_close(out, ref, **TOL[torch.bfloat16], msg=lambda m: f"{name}: {m}")


def test_fused_path_actually_runs():
    """The variant must not silently fall back to eager: the wrapper rejects CPU tensors."""
    from kernels.rmsnorm import rmsnorm
    with pytest.raises(RuntimeError):
        rmsnorm(torch.randn(2, 256), torch.ones(256))


# --- microbenchmark ---------------------------------------------------------------------

def _time_us(fn, warmup=25, reps=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def benchmark():
    """Eager module vs fused kernel per decode-step shape, microseconds (median of 200)."""
    print(f"\n{'shape':28s} {'eager us':>9s} {'fused us':>9s} {'speedup':>8s}")
    print("-" * 58)
    # decode-step shapes: hidden norms see 1 row; q/k/v norms see one row per head (8)
    for n, with_scale, rows, label in [
        (1536, True, 1, "hidden norm, 1 x 1536"),
        (256, True, 8, "q/k norm sliding, 8 x 256"),
        (256, False, 8, "v norm sliding, 8 x 256"),
        (512, True, 8, "q/k norm full, 8 x 512"),
        (512, False, 8, "v norm full, 8 x 512"),
        (1536, True, 128, "prefill hidden, 128 x 1536"),
    ]:
        mod = _module(n, with_scale, torch.bfloat16)
        x = torch.randn(rows, n, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            eager = _time_us(lambda: mod(x))
            fused = _time_us(lambda: _fused(mod, x))
        print(f"{label:28s} {eager:9.1f} {fused:9.1f} {eager / fused:7.2f}x")
    print("\nAt batch 1 both are launch-bound, so expect the win to show as fewer launches "
          "per call (~8 -> 1), not as a faster single kernel.")


if __name__ == "__main__":
    if not (CUDA and TRITON):
        raise SystemExit("needs a CUDA GPU with Triton")
    sys.exit(pytest.main([__file__, "-q", "-s"]) or benchmark())
