"""QuantLinear: a weight-only int8 / int4 replacement for nn.Linear. Deliberately naive.

The weight is stored quantized (quant/primitives.py formats) and the original bf16 weight is
freed, so memory at rest drops. forward() dequantizes the WHOLE weight back to bf16 and calls
F.linear - so every step reads the quantized weight, writes a bf16 copy and reads it again.
That saves memory, not bytes per token; a fused dequant-GEMV kernel is what saves bandwidth.

Note the bf16 dequant: q * scale is exact in fp32, but the bf16 weight F.linear sees is that
product rounded to bf16, a little extra error on top of the quantization error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.primitives import (DEFAULT_GROUP_SIZE, dequantize_int4, dequantize_int8,
                              dequantize_int8_grouped, quantize_int4, quantize_int8,
                              quantize_int8_grouped)

# int8: one scale per row. int8_grouped / int4: one scale per group_size inputs of a row.
FORMATS = ("int8", "int8_grouped", "int4")


class QuantLinear(nn.Module):
    def __init__(self, q, scale, out_features, in_features, fmt, group_size, weight_dtype,
                 bias=None):
        super().__init__()
        if fmt not in FORMATS:
            raise ValueError(f"unknown format {fmt!r}; known: {FORMATS}")
        self.in_features, self.out_features = in_features, out_features
        self.fmt, self.group_size, self.weight_dtype = fmt, group_size, weight_dtype
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.bias = bias

    @classmethod
    def from_linear(cls, linear, fmt, group_size=DEFAULT_GROUP_SIZE):
        """Quantize linear.weight with the quant/primitives.py function for fmt, on its own
        device."""
        w = linear.weight.detach()
        if fmt == "int8":
            q, scale = quantize_int8(w)
            group_size = None
        elif fmt == "int8_grouped":
            q, scale = quantize_int8_grouped(w, group_size)
        elif fmt == "int4":
            q, scale = quantize_int4(w, group_size)
        else:
            raise ValueError(f"unknown format {fmt!r}; known: {FORMATS}")
        bias = linear.bias if linear.bias is not None else None
        return cls(q, scale, linear.out_features, linear.in_features, fmt, group_size,
                   w.dtype, bias)

    def dequantized_weight(self, dtype=None):
        """The full weight, in the original dtype (bf16) unless dtype is given."""
        dtype = dtype or self.weight_dtype
        if self.fmt == "int8":
            return dequantize_int8(self.q, self.scale, dtype=dtype)
        if self.fmt == "int8_grouped":
            return dequantize_int8_grouped(self.q, self.scale, self.group_size, dtype=dtype)
        return dequantize_int4(self.q, self.scale, (self.out_features, self.in_features),
                               self.group_size, dtype=dtype)

    def forward(self, x):
        return F.linear(x, self.dequantized_weight(), self.bias)

    def stored_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.q, self.scale))

    def extra_repr(self):
        g = f", group_size={self.group_size}" if self.group_size else ""
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"fmt={self.fmt}{g}, bias={self.bias is not None}")
