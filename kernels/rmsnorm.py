"""Fused RMSNorm for Gemma 4's Gemma4RMSNorm: one Triton program per row.

Gemma 4's math, read from transformers 5.19.0 (models/gemma4/modeling_gemma4.py, the only
RMSNorm class Gemma 4 uses):

    x32 = x.float()                                   # upcast first
    ms  = x32.pow(2).mean(-1, keepdim=True) + eps     # eps = 1e-6 for every text-decoder norm
    y32 = x32 * torch.pow(ms, -0.5)                   # pow(-0.5) in source; runs as rsqrt on CUDA
    if with_scale:
        y32 = y32 * weight.float()                    # PLAIN weight (initialised to ones) -
                                                      # NOT (1 + weight) as in Gemma 1-3
    y   = y32.type_as(x)                              # the only cast back, at the very end

v_norm is built with with_scale=False and has no weight at all. Eager runs one call as ~8
kernels (two fp32 upcasts, pow, mean, add, rsqrt, two multiplies); 242 calls per decode step
make RMSNorm roughly 1,900 of the ~3,085 launches. This replaces each call with one kernel.

Triton is imported only when the fused path actually runs, so every other tool in the repo
still imports on a Mac.
"""

import torch

EPS = 1e-6


def rmsnorm_reference(x, weight=None, eps=EPS, with_scale=True):
    """Gemma4RMSNorm.forward, op for op. Bitwise identical to the module on the same device."""
    x32 = x.float()
    ms = x32.pow(2).mean(-1, keepdim=True) + eps
    y32 = x32 * torch.pow(ms, -0.5)
    if with_scale:
        y32 = y32 * weight.float()
    return y32.type_as(x)


def _num_warps(block_size):
    # one warp per 256 elements of the row, clamped to [1, 8]: N=256 -> 1, 512 -> 2, 1536 -> 8
    return max(1, min(8, block_size // 256))


def rmsnorm(x, weight=None, eps=EPS, with_scale=True):
    """Fused RMSNorm over the last dimension of `x`, any leading shape."""
    if not x.is_cuda:
        raise RuntimeError("the fused RMSNorm kernel needs a CUDA tensor; use rmsnorm_reference")
    import triton

    from kernels._rmsnorm_triton import rmsnorm_fwd_kernel

    n = x.shape[-1]
    if with_scale:
        if weight is None or weight.shape != (n,):
            raise ValueError(f"weight must have shape ({n},), got "
                             f"{None if weight is None else tuple(weight.shape)}")
        weight = weight.contiguous()

    # Flatten leading dims into rows. reshape() is a view whenever the layout allows and a copy
    # otherwise; the kernel then only needs the last dim to be unit-stride and a fixed row
    # stride, so anything else is made contiguous here rather than read wrongly.
    x2 = x.reshape(-1, n)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    rows = x2.shape[0]
    y = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    if rows == 0:
        return y.reshape(x.shape)

    block = triton.next_power_of_2(n)
    rmsnorm_fwd_kernel[(rows,)](
        x2, weight if with_scale else x2, y,
        x2.stride(0), y.stride(0),
        n, eps,
        HAS_WEIGHT=with_scale,
        BLOCK_SIZE=block,
        num_warps=_num_warps(block),
    )
    return y.reshape(x.shape)
