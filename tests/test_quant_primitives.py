"""int8 / int4 weight quantization primitives (quant/primitives.py). CPU only, runs on a Mac.

    pytest -s tests/test_quant_primitives.py    # -s prints the real-weight error table

The int8 tests fail with NotImplementedError until quantize_int8 / dequantize_int8 are written.

The real-weight tests use q_proj, o_proj, gate_proj and down_proj of text layer 0. They are
range-fetched from the Hub's model.safetensors (~50 MB, not the 10 GB file) into
~/.cache/gemma4-gpu/ on first use; without network access those tests skip.

Relative error definitions used in the real-weight table:
  fro   ||W_hat - W||_F / ||W||_F
  mean  mean|W_hat - W| / mean|W|
  max   max over rows (int8) or groups (int4) of  max|W_hat - W| / absmax(W)  - bounded by
        1 / (2 * qmax), times the bf16 round-up of the scale (at most 1 + 2^-7)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quant.primitives as P  # noqa: E402  (module attribute, so the TODO bodies are picked up)

MODEL_ID = "google/gemma-4-E2B-it"
REVISION = "4adf15748de9e1a0097b6274cee9c15836c41c37"   # pinned so the numbers reproduce
SAMPLE = os.path.expanduser("~/.cache/gemma4-gpu/gemma4_e2b_layer0_linears.safetensors")
REAL = {
    "q_proj": "model.language_model.layers.0.self_attn.q_proj.weight",
    "o_proj": "model.language_model.layers.0.self_attn.o_proj.weight",
    "gate_proj": "model.language_model.layers.0.mlp.gate_proj.weight",
    "down_proj": "model.language_model.layers.0.mlp.down_proj.weight",
}
SHAPES = [(1, 128), (3, 256), (7, 384), (64, 1536), (5, 2048)]   # odd row counts on purpose
DTYPES = [torch.bfloat16, torch.float32]
GROUPS = [32, 64, 128]


def _weight(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (0.02 * torch.randn(shape, generator=g)).to(dtype)


def _assert_within_half_scale(deq, w, scale_per_elem):
    """|dequant - w| <= scale / 2 per element, with slack for one fp32 rounding of w / scale."""
    err = (deq - w.float()).abs()
    bound = scale_per_elem.float() / 2 * (1 + 2**-20) + w.float().abs() * 2**-23
    worst = (err - bound).max().item()
    assert worst <= 0, f"error exceeds scale/2 by {worst:.3e}"


# ------------------------------------------------------------------------------ int4 packing

def test_int4_pack_layout():
    # documented layout: element 2k -> low nibble, 2k+1 -> high nibble, nibble = code + 8
    codes = torch.tensor([[-7, 7, 0, 1]], dtype=torch.int8)
    packed = P.pack_int4(codes)
    assert packed.dtype == torch.uint8
    assert packed.tolist() == [[0x1 | (0xF << 4), 0x8 | (0x9 << 4)]]


@pytest.mark.parametrize("n", [1, 2, 3, 127, 128, 1535, 1536])
def test_int4_pack_roundtrip_exact(n):
    g = torch.Generator().manual_seed(n)
    codes = torch.randint(-7, 8, (5, n), generator=g, dtype=torch.int8)
    packed = P.pack_int4(codes)
    assert packed.shape == (5, (n + 1) // 2)
    assert torch.equal(P.unpack_int4(packed, n), codes)


def test_int4_pack_roundtrip_every_code_pair():
    lo, hi = torch.meshgrid(torch.arange(-7, 8), torch.arange(-7, 8), indexing="ij")
    codes = torch.stack((lo.flatten(), hi.flatten()), dim=-1).to(torch.int8)   # [225, 2]
    assert torch.equal(P.unpack_int4(P.pack_int4(codes), 2), codes)


@pytest.mark.parametrize("bad", [-8, 8])
def test_int4_pack_refuses_out_of_range(bad):
    with pytest.raises(ValueError, match="int4 codes"):
        P.pack_int4(torch.tensor([[0, bad]], dtype=torch.int8))


def test_int4_unpack_refuses_wrong_length():
    with pytest.raises(ValueError, match="does not hold"):
        P.unpack_int4(torch.zeros(2, 4, dtype=torch.uint8), 10)


# -------------------------------------------------------------------------------------- int4

def _int4_scale_per_elem(scales, shape, group_size):
    return scales.float().repeat_interleave(group_size, dim=1).reshape(shape)


@pytest.mark.parametrize("group_size", GROUPS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_int4_roundtrip_bound(shape, dtype, group_size):
    w = _weight(shape, dtype)
    packed, scales = P.quantize_int4(w, group_size)
    assert packed.dtype == torch.uint8 and packed.shape == (shape[0], shape[1] // 2)
    assert scales.dtype == torch.bfloat16 and scales.shape == (shape[0], shape[1] // group_size)
    assert packed.is_contiguous() and scales.is_contiguous()
    deq = P.dequantize_int4(packed, scales, shape, group_size)
    assert deq.dtype == torch.float32
    _assert_within_half_scale(deq, w, _int4_scale_per_elem(scales, shape, group_size))
    codes = P.unpack_int4(packed, shape[1])
    assert codes.min() >= -7 and codes.max() <= 7


@pytest.mark.parametrize("group_size", GROUPS)
def test_int4_exact_on_grid(group_size):
    # Every group holds +-7 * s for a bf16 s, so the derived scale is exactly s and every
    # value is a multiple of it: the round trip must be bit-exact.
    out_f, in_f = 6, 4 * group_size
    g = torch.Generator().manual_seed(1)
    s = (2.0 ** torch.randint(-12, -2, (out_f, in_f // group_size), generator=g)).float()
    s = (s * (1 + torch.randint(0, 128, s.shape, generator=g) / 128)).to(torch.bfloat16).float()
    codes = torch.randint(-7, 8, (out_f, in_f // group_size, group_size), generator=g)
    codes[..., 0] = 7
    codes[..., 1] = -7
    w = (codes.float() * s.unsqueeze(-1)).reshape(out_f, in_f)
    packed, scales = P.quantize_int4(w, group_size)
    assert torch.equal(scales.float(), s)
    assert torch.equal(P.dequantize_int4(packed, scales, w.shape, group_size), w)


def test_int4_zero_group_and_zero_matrix_no_nan():
    w = _weight((4, 512), torch.float32)
    w[1, 128:256] = 0          # one all-zero group
    w[3] = 0                   # one all-zero row
    packed, scales = P.quantize_int4(w, 128)
    deq = P.dequantize_int4(packed, scales, w.shape, 128)
    assert torch.isfinite(deq).all() and torch.isfinite(scales.float()).all()
    assert torch.equal(deq[1, 128:256], torch.zeros(128))
    assert torch.equal(deq[3], torch.zeros(512))
    z = torch.zeros(3, 256)
    pz, sz = P.quantize_int4(z, 128)
    assert torch.equal(P.dequantize_int4(pz, sz, z.shape, 128), z)


def test_int4_outlier_stays_in_its_group():
    w = _weight((2, 512), torch.float32)
    clean = P.dequantize_int4(*P.quantize_int4(w, 128), w.shape, 128)
    w_out = w.clone()
    w_out[0, 300] = 1e3        # group 2 of row 0
    packed, scales = P.quantize_int4(w_out, 128)
    deq = P.dequantize_int4(packed, scales, w.shape, 128)
    _assert_within_half_scale(deq, w_out, _int4_scale_per_elem(scales, w.shape, 128))
    assert abs(deq[0, 300].item() - 1e3) <= scales[0, 2].float().item() / 2
    # every other group is quantized exactly as without the outlier
    mask = torch.ones_like(w, dtype=torch.bool)
    mask[0, 256:384] = False
    assert torch.equal(deq[mask], clean[mask])


@pytest.mark.parametrize("in_f,group_size", [(200, 128), (130, 64), (1536, 100)])
def test_int4_refuses_non_multiple(in_f, group_size):
    with pytest.raises(ValueError, match="not a multiple of group_size"):
        P.quantize_int4(torch.randn(4, in_f), group_size)


@pytest.mark.parametrize("group_size", [0, 1, 63])
def test_int4_refuses_bad_group_size(group_size):
    with pytest.raises(ValueError, match="group_size must be even"):
        P.quantize_int4(torch.randn(4, 126), group_size)


def test_int4_scale_rounded_up_to_bf16():
    # Find absmax values whose fp32 scale bf16 rounds DOWN; those must be bumped up so the
    # largest value still fits in [-7, 7] without being clipped.
    g = torch.Generator().manual_seed(2)
    absmax = torch.rand(4096, generator=g) + 0.5
    s = absmax / 7
    down = s.to(torch.bfloat16).float() < s
    assert down.sum() > 100
    w = torch.zeros(int(down.sum()), 128)
    w[:, 0] = absmax[down]
    packed, scales = P.quantize_int4(w, 128)
    assert (scales.float()[:, 0] >= s[down]).all()
    assert (P.unpack_int4(packed, 128)[:, 0] == 7).all()
    deq = P.dequantize_int4(packed, scales, w.shape, 128)
    _assert_within_half_scale(deq, w, _int4_scale_per_elem(scales, w.shape, 128))


# ------------------------------------------------------------------- refusals, both formats

@pytest.mark.parametrize("fmt", ["int8", "int4"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_refuses_non_finite(fmt, bad):
    w = torch.randn(2, 128)
    w[1, 5] = bad
    with pytest.raises(ValueError, match="inf or NaN"):
        getattr(P, f"quantize_{fmt}")(w)


@pytest.mark.parametrize("fmt", ["int8", "int4"])
def test_refuses_non_2d(fmt):
    with pytest.raises(ValueError, match="2-D"):
        getattr(P, f"quantize_{fmt}")(torch.randn(2, 3, 128))


# ---------------------------------------------- int8 (fail with NotImplementedError until TODOs)

@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES)
def test_int8_roundtrip_bound(shape, dtype):
    w = _weight(shape, dtype)
    q, scale = P.quantize_int8(w)
    assert q.dtype == torch.int8 and q.shape == shape
    assert scale.dtype == torch.bfloat16 and scale.shape == (shape[0],)
    assert q.min() >= -127 and q.max() <= 127
    deq = P.dequantize_int8(q, scale)
    assert deq.dtype == torch.float32
    _assert_within_half_scale(deq, w, scale.float().unsqueeze(1).expand(shape))


def test_int8_exact_on_grid():
    out_f, in_f = 6, 640
    g = torch.Generator().manual_seed(3)
    s = (2.0 ** torch.randint(-14, -4, (out_f,), generator=g)).float()
    s = (s * (1 + torch.randint(0, 128, s.shape, generator=g) / 128)).to(torch.bfloat16).float()
    codes = torch.randint(-127, 128, (out_f, in_f), generator=g)
    codes[:, 0] = 127
    codes[:, 1] = -127
    w = codes.float() * s.unsqueeze(1)
    q, scale = P.quantize_int8(w)
    assert torch.equal(scale.float(), s)
    assert torch.equal(P.dequantize_int8(q, scale), w)


def test_int8_zero_row_no_nan():
    w = _weight((3, 256), torch.float32)
    w[1] = 0
    q, scale = P.quantize_int8(w)
    deq = P.dequantize_int8(q, scale)
    assert torch.isfinite(deq).all() and torch.isfinite(scale.float()).all()
    assert torch.equal(deq[1], torch.zeros(256))
    z = torch.zeros(2, 128)
    assert torch.equal(P.dequantize_int8(*P.quantize_int8(z)), z)


def test_int8_outlier_bound_holds():
    # Per-row scaling: one outlier coarsens its whole row (unlike int4 groups), but the
    # scale/2 bound must still hold and the other rows must be untouched.
    w = _weight((3, 512), torch.float32)
    clean = P.dequantize_int8(*P.quantize_int8(w))
    w_out = w.clone()
    w_out[0, 7] = 1e3
    q, scale = P.quantize_int8(w_out)
    deq = P.dequantize_int8(q, scale)
    _assert_within_half_scale(deq, w_out, scale.float().unsqueeze(1).expand(w.shape))
    assert torch.equal(deq[1:], clean[1:])


def test_int8_dequant_dtype():
    q, scale = P.quantize_int8(_weight((2, 128), torch.float32))
    assert P.dequantize_int8(q, scale, dtype=torch.bfloat16).dtype == torch.bfloat16


# ---------------------------------------------------------------------- real Gemma 4 weights

def _fetch_sample():
    """Range-fetch the four layer-0 tensors from the Hub's single model.safetensors."""
    from huggingface_hub import get_safetensors_metadata, get_session, hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    from safetensors.torch import save_file

    meta = get_safetensors_metadata(MODEL_ID, revision=REVISION)
    info = meta.files_metadata["model.safetensors"].tensors
    url = hf_hub_url(MODEL_ID, "model.safetensors", revision=REVISION)
    session, headers = get_session(), build_hf_headers()

    def fetch(start, end):  # [start, end)
        r = session.get(url, headers={**headers, "Range": f"bytes={start}-{end - 1}"},
                        follow_redirects=True, timeout=120)
        r.raise_for_status()
        if len(r.content) != end - start:
            raise RuntimeError(f"range fetch returned {len(r.content)} bytes, wanted {end - start}")
        return r.content

    data_start = 8 + int.from_bytes(fetch(0, 8), "little")
    out = {}
    for key in REAL.values():
        t = info[key]
        if t.dtype != "BF16":
            raise RuntimeError(f"{key} is {t.dtype}, expected BF16")
        lo, hi = t.data_offsets
        buf = bytearray(fetch(data_start + lo, data_start + hi))
        out[key] = torch.frombuffer(buf, dtype=torch.bfloat16).reshape(t.shape).clone()
    os.makedirs(os.path.dirname(SAMPLE), exist_ok=True)
    save_file(out, SAMPLE, metadata={"model": MODEL_ID, "revision": REVISION})


@pytest.fixture(scope="module")
def real_weights():
    if not os.path.exists(SAMPLE):
        try:
            _fetch_sample()
        except Exception as e:  # offline, Hub down, auth
            pytest.skip(f"could not fetch the real Gemma 4 weights: {type(e).__name__}: {e}")
    from safetensors.torch import load_file
    tensors = load_file(SAMPLE)
    return {name: tensors[key] for name, key in REAL.items()}


def _rel_errors(w, deq, unit_absmax, unit_err_max):
    w32 = w.float()
    e = deq - w32
    fro = (e.norm() / w32.norm()).item()
    mean = (e.abs().mean() / w32.abs().mean()).item()
    mx = (unit_err_max / unit_absmax.clamp_min(1e-30)).max().item()
    return fro, mean, mx


@pytest.mark.parametrize("name", list(REAL))
@pytest.mark.parametrize("fmt", ["int8", "int4"])
def test_real_weight_error(real_weights, fmt, name):
    w = real_weights[name]
    out_f, in_f = w.shape
    if fmt == "int8":
        q, scale = P.quantize_int8(w)
        deq = P.dequantize_int8(q, scale)
        per_elem = scale.float().unsqueeze(1).expand(w.shape)
        absmax = w.float().abs().amax(dim=1)
        err_max = (deq - w.float()).abs().amax(dim=1)
        qmax = 127
    else:
        packed, scales = P.quantize_int4(w, P.DEFAULT_GROUP_SIZE)
        deq = P.dequantize_int4(packed, scales, w.shape)
        per_elem = _int4_scale_per_elem(scales, w.shape, P.DEFAULT_GROUP_SIZE)
        groups = (out_f, in_f // P.DEFAULT_GROUP_SIZE, P.DEFAULT_GROUP_SIZE)
        absmax = w.float().abs().reshape(groups).amax(dim=-1)
        err_max = (deq - w.float()).abs().reshape(groups).amax(dim=-1)
        qmax = 7
    _assert_within_half_scale(deq, w, per_elem)
    fro, mean, mx = _rel_errors(w, deq, absmax, err_max)
    assert mx <= (1 + 2**-7) / (2 * qmax) * (1 + 1e-6)
    print(f"\n  {fmt}  layer0 {name:9s} {str(tuple(w.shape)):14s} "
          f"rel err  fro {fro:.4e}  mean {mean:.4e}  max {mx:.4e}", end="")


if __name__ == "__main__":
    sys.exit(pytest.main(["-s", "-q", __file__]))
