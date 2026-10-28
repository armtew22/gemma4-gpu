"""The Triton kernel behind kernels.rmsnorm.rmsnorm. Imported only on a CUDA machine.

Written by hand, one commented step at a time. Not yet run on a GPU: tests/test_rmsnorm.py,
which compares it against the real Gemma4RMSNorm module, must pass before anything measures
the fused_rmsnorm variant (pod/fused_rmsnorm.sh stops if it does not).
"""

import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    X,              # *input, [rows, N] with unit stride along N
    W,              # *weight [N]; a dummy pointer when HAS_WEIGHT is False (never read)
    Y,              # *output, [rows, N], same dtype as X
    stride_x_row,   # elements between consecutive rows of X
    stride_y_row,   # elements between consecutive rows of Y
    N,              # REAL row length: 1536, 512 or 256 for Gemma 4
    eps,            # 1e-6
    HAS_WEIGHT: tl.constexpr,   # False only for v_norm (with_scale=False)
    BLOCK_SIZE: tl.constexpr,   # triton.next_power_of_2(N), so >= N
):
    # ======================= kernel body: one program per row =======================
    #
    # 1. Which row is this program? One program per row, so the row index is the program id
    #    along axis 0. Move the X and Y base pointers to the start of that row using the row
    #    strides.
    row = tl.program_id(0)
    X = X + row * stride_x_row
    Y = Y + row * stride_y_row

    #
    # 2. Column offsets 0..BLOCK_SIZE-1, and a mask that is True only for offsets < N.
    #    BLOCK_SIZE is a power of two >= N (2048 for N=1536), so the tail must be masked off.
    #
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    # 3. Load the row with the mask and other=0.0, then cast to fp32. Gemma upcasts the input
    #    BEFORE squaring. The zero-filled tail contributes nothing to the sum in step 4.
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    # 4. Mean of squares: sum(x * x) divided by the REAL N, not BLOCK_SIZE. Dividing by
    #    BLOCK_SIZE would silently shrink the mean for N=1536 (2048 slots, 512 of them zero).
    
    x_mean_of_sqrs = tl.sum(x * x, axis=0) / N
    # 5. Reciprocal RMS: rstd = 1 / sqrt(mean + eps). Gemma writes torch.pow(ms, -0.5), which
    #    runs as rsqrt on CUDA.
    rstd = 1/ tl.sqrt(x_mean_of_sqrs + eps)
    #
    # 6. Normalize in fp32: y = x * rstd. Keep Gemma's order - (x * rstd) first, then the
    #    weight - because fp32 multiplication is not associative.
    y = x * rstd
    
    #
    # 7. If HAS_WEIGHT: load the weight with the same mask, cast to fp32, and multiply by the
    #    PLAIN weight. Gemma 4 does NOT add 1 (Gemma 1-3 did); the weight is initialised to ones.
    #    HAS_WEIGHT is a constexpr, so the untaken branch is compiled away.
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = y * w
    #
    # 8. Cast back to the output dtype only now - Gemma's single .type_as(x) at the very end -
    #    and store with the mask.
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
