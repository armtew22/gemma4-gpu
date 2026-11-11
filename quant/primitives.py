"""Weight-only quantization primitives for the text-decoder linears. CPU reference code.

Two formats, both symmetric (no zero point) and weight-only (activations stay bf16):

  int8  per output channel   one scale per row of W [out, in]       q in [-127, 127]
  int4  group-wise           one scale per group_size inputs of a row q in [-7, 7]

Conventions shared by both:
  - W is a 2-D linear weight [out_features, in_features], as stored by nn.Linear.
  - All arithmetic is fp32, whatever the input dtype.
  - Scales are STORED in bf16. A scale is rounded UP to the next bf16 value when bf16 rounding
    would shrink it, so absmax / scale never exceeds qmax and nothing has to be clipped. The
    quantized values are computed against the stored bf16 scale, never the fp32 one, so
    dequantize(quantize(W)) is off by at most scale / 2 per element.
  - An all-zero row (int8) or group (int4) gets scale 1.0 and quantizes to all zeros, so it
    dequantizes to exact zeros instead of 0 / 0 = NaN.
  - Dequantization returns fp32 by default. q * scale is exact in fp32 (a 7-bit integer times
    a bf16 scale with an 8-bit significand fits in fp32's 24 bits).

int4 packing layout (a Triton kernel will unpack this - keep it in sync):

  - Codes are offset binary: nibble = q + 8, so q in [-7, 7] is stored as 1..15. Nibble 0
    (q = -8) is never produced. Unpacking is (nibble - 8), with no sign extension.
  - Two codes per byte along the INPUT dim, row-major. For a row of codes c[0..in-1]:
        packed[row, k] = (c[2k] + 8) | ((c[2k+1] + 8) << 4)
    so element 2k is the LOW nibble (bits 0-3) and element 2k+1 the HIGH nibble (bits 4-7).
  - Unpack:  c[2k] = (b & 0xF) - 8,  c[2k+1] = (b >> 4) - 8
  - packed: uint8 [out, in / 2], contiguous. scales: bf16 [out, in / group_size],
    contiguous; scales[r, g] covers inputs g*group_size .. (g+1)*group_size - 1 of row r.
  - group_size must be even, so a byte never straddles two groups.
"""

import torch

INT8_QMAX = 127
INT4_QMAX = 7
INT4_OFFSET = 8
DEFAULT_GROUP_SIZE = 128


def _check_weight(w):
    if w.dim() != 2:
        raise ValueError(f"expected a 2-D linear weight [out, in], got shape {tuple(w.shape)}")
    if not w.is_floating_point():
        raise ValueError(f"expected a floating-point weight, got {w.dtype}")
    if not torch.isfinite(w).all():
        raise ValueError("weight contains inf or NaN - refusing to quantize it")


def _bf16_scale(absmax, qmax):
    """Per-row/group scale absmax / qmax as bf16, rounded up so absmax / scale <= qmax.

    absmax is fp32. Zero absmax (an all-zero row or group) gets scale 1.0: any nonzero scale
    quantizes zeros to zeros, and it avoids dividing by zero.
    """
    s = absmax / qmax
    s = torch.where(absmax == 0, torch.ones_like(s), s)
    sb = s.to(torch.bfloat16)
    up = torch.nextafter(sb, torch.full_like(sb, float("inf")))
    return torch.where(sb.float() < s, up, sb)


# ---------------------------------------------------------------------------- int8, per row

def quantize_int8(w):
    """Symmetric int8, one scale per output channel.

    Returns (q, scale): q int8 [out, in] in [-127, 127]; scale bf16 [out].
    """
    _check_weight(w)
    # TODO(armaan): write this body. One step per comment:
    #   1. Upcast: w32 = w as fp32. All math below is fp32.
    #   2. Per-row absmax: the largest |value| in each row (reduce over the input dim).
    #   3. Scale: absmax / 127, stored in bf16. _bf16_scale(absmax, INT8_QMAX) does this and
    #      handles the two traps - an all-zero row, and bf16 rounding the scale down so the
    #      biggest value no longer fits. Read it before you use it.
    #   4. Divide each row by its OWN stored scale (the bf16 one, back in fp32), so the error
    #      bound holds for the scale that is actually saved. Mind the broadcast shape.
    #   5. Round to the nearest integer.
    #   6. Clamp to [-127, 127]. (-128 is left unused so the range stays symmetric.)
    #   7. Cast to int8 and return (q, scale).
    raise NotImplementedError("quantize_int8: TODO - write the body (see the steps above)")


def dequantize_int8(q, scale, dtype=torch.float32):
    """Inverse of quantize_int8: q * scale per row. Returns dtype (fp32 by default)."""
    # TODO(armaan): write this body. One step per comment:
    #   1. Upcast q to fp32.
    #   2. Upcast scale to fp32 and shape it so it broadcasts over each row ([out] -> [out, 1]).
    #   3. Multiply.
    #   4. Cast to dtype and return.
    raise NotImplementedError("dequantize_int8: TODO - write the body (see the steps above)")


# ------------------------------------------------------------------------- int4, group-wise

def pack_int4(codes):
    """Pack int4 codes in [-7, 7] two per byte along the last dim (layout in the module doc).

    An odd last dim is padded with one zero code, which unpack_int4 drops given the length.
    """
    if codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"codes must be an integer tensor, got {codes.dtype}")
    if codes.numel() and (codes.min() < -INT4_QMAX or codes.max() > INT4_QMAX):
        raise ValueError(f"int4 codes must be in [-{INT4_QMAX}, {INT4_QMAX}]")
    n = codes.shape[-1]
    if n % 2:
        codes = torch.nn.functional.pad(codes, (0, 1))  # code 0 -> nibble 8
    nib = (codes.to(torch.int16) + INT4_OFFSET).to(torch.uint8)
    lo, hi = nib[..., 0::2], nib[..., 1::2]
    return (lo | (hi << 4)).contiguous()


def unpack_int4(packed, n):
    """Inverse of pack_int4: int8 codes [..., n] from uint8 [..., ceil(n / 2)]."""
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed must be uint8, got {packed.dtype}")
    if packed.shape[-1] != (n + 1) // 2:
        raise ValueError(f"packed last dim {packed.shape[-1]} does not hold {n} codes")
    lo = (packed & 0xF).to(torch.int8) - INT4_OFFSET
    hi = (packed >> 4).to(torch.int8) - INT4_OFFSET
    codes = torch.stack((lo, hi), dim=-1).flatten(-2)
    return codes[..., :n]


def _check_groups(in_features, group_size):
    if group_size < 2 or group_size % 2:
        raise ValueError(f"group_size must be even and >= 2, got {group_size}")
    if in_features % group_size:
        # Padding would change the packed layout the kernel reads, and every Gemma 4 text
        # linear has in_features divisible by 128 anyway, so refuse instead.
        raise ValueError(
            f"in_features {in_features} is not a multiple of group_size {group_size}; "
            "pick a group_size that divides it"
        )


def quantize_int4(w, group_size=DEFAULT_GROUP_SIZE):
    """Symmetric int4, one scale per group_size consecutive inputs of each row.

    Returns (packed, scales): packed uint8 [out, in / 2]; scales bf16 [out, in / group_size].
    """
    _check_weight(w)
    out_f, in_f = w.shape
    _check_groups(in_f, group_size)
    g = w.float().reshape(out_f, in_f // group_size, group_size)
    absmax = g.abs().amax(dim=-1)                                   # [out, groups]
    scales = _bf16_scale(absmax, INT4_QMAX)
    q = torch.round(g / scales.float().unsqueeze(-1))
    q = q.clamp(-INT4_QMAX, INT4_QMAX).to(torch.int8).reshape(out_f, in_f)
    return pack_int4(q), scales


def dequantize_int4(packed, scales, shape, group_size=DEFAULT_GROUP_SIZE, dtype=torch.float32):
    """Inverse of quantize_int4. shape is the original (out, in). Returns dtype (fp32 default)."""
    out_f, in_f = shape
    _check_groups(in_f, group_size)
    if tuple(packed.shape) != (out_f, in_f // 2):
        raise ValueError(f"packed shape {tuple(packed.shape)} != {(out_f, in_f // 2)}")
    if tuple(scales.shape) != (out_f, in_f // group_size):
        raise ValueError(f"scales shape {tuple(scales.shape)} != {(out_f, in_f // group_size)}")
    q = unpack_int4(packed, in_f).float().reshape(out_f, in_f // group_size, group_size)
    w = q * scales.float().unsqueeze(-1)
    return w.reshape(out_f, in_f).to(dtype)
