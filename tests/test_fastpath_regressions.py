import types
import mlx.core as mx
import mlx.nn as nn
import pytest

import inkling2_fastpath
from inkling2_fastpath import fastpath as fp
from mlx_lm.models.inkling2 import layers as L


def _make_qlinear(in_dims=64, out_dims=64, bits=4, group_size=64, bias=False):
    lin = nn.Linear(in_dims, out_dims, bias=bias)
    mx.eval(lin.parameters())
    return nn.QuantizedLinear.from_linear(lin, group_size=group_size, bits=bits)


def test_proj_skips_qrgs_when_same_kv_uses_all6():
    fp.revert(None)
    fp.OPTS.update(fp.BEST)
    attn = types.SimpleNamespace(
        k_proj=lambda x: x,
        v_proj=lambda x: x,
        q_proj=lambda x: x,
        r_proj=lambda x: x,
        gate_proj=lambda x: x,
        sink_proj=lambda x: x,
    )
    calls = {"qrgs": 0, "all6": 0}

    def qrgs_fn(x):
        calls["qrgs"] += 1
        return mx.zeros((1, 1, 16), dtype=x.dtype)

    def all6_fn(x):
        calls["all6"] += 1
        return mx.zeros((1, 1, 24), dtype=x.dtype)

    fp._fused[id(attn)] = {
        "qrgs": qrgs_fn,
        "qrgs_split": [4, 8, 12],
        "all6": all6_fn,
        "all6_split": [4, 8, 12, 16, 20],
    }
    try:
        qi = mx.ones((1, 1, 8), dtype=mx.bfloat16)
        out = fp._proj(None, attn, qi, qi, qi, same_kv=True)
        assert len(out) == 6
        assert calls["all6"] == 1
        assert calls["qrgs"] == 0, "qrgs should not run when same_kv uses all6"
    finally:
        fp.revert(None)


def test_fuse_skips_non_sliding_all6_and_mixed_precision_mismatch():
    fp.revert(None)
    # Build two mock layers: one SlidingDecoderLayer (uniform 4-bit), one FullAttention (mixed 8-bit/4-bit)
    sliding_layer = L.SlidingDecoderLayer.__new__(L.SlidingDecoderLayer)
    sliding_layer.self_attn = types.SimpleNamespace(
        q_proj=_make_qlinear(64, 64, bits=4),
        k_proj=_make_qlinear(64, 32, bits=4),
        v_proj=_make_qlinear(64, 32, bits=4),
        r_proj=_make_qlinear(64, 32, bits=4),
        gate_proj=_make_qlinear(64, 64, bits=4),
        sink_proj=_make_qlinear(64, 32, bits=4),
    )

    full_layer = types.SimpleNamespace(
        self_attn=types.SimpleNamespace(
            q_proj=_make_qlinear(64, 64, bits=8),
            k_proj=_make_qlinear(64, 32, bits=8),
            v_proj=_make_qlinear(64, 32, bits=8),
            r_proj=_make_qlinear(64, 32, bits=4),  # mixed bit-width!
            gate_proj=_make_qlinear(64, 64, bits=8),
            sink_proj=_make_qlinear(64, 32, bits=4),
        )
    )
    full_uniform = types.SimpleNamespace(
        self_attn=types.SimpleNamespace(
            q_proj=_make_qlinear(64, 64, bits=4),
            k_proj=_make_qlinear(64, 32, bits=4),
            v_proj=_make_qlinear(64, 32, bits=4),
            r_proj=_make_qlinear(64, 32, bits=4),
            gate_proj=_make_qlinear(64, 64, bits=4),
            sink_proj=_make_qlinear(64, 32, bits=4),
        )
    )
    mock_model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            language_model=types.SimpleNamespace(layers=[sliding_layer, full_layer, full_uniform])
        )
    )
    try:
        fp._fuse(mock_model)
        assert "qrgs" in fp._fused[id(sliding_layer.self_attn)]
        assert "all6" in fp._fused[id(sliding_layer.self_attn)]
        assert id(full_layer.self_attn) not in fp._fused
        assert "qrgs" in fp._fused[id(full_uniform.self_attn)]
        assert "all6" not in fp._fused[id(full_uniform.self_attn)]

        # Verify disable/revert releases all cached fused weights
        inkling2_fastpath.disable(mock_model)
        assert len(fp._fused) == 0
        assert len(fp._msgw) == 0
        assert len(fp._bw) == 0
        assert len(fp._cache) == 0
    finally:
        fp.revert(None)


@pytest.mark.parametrize("state_dtype", [mx.float32, mx.bfloat16])
def test_sconv_kernel_preserves_state_dtype_and_matches_stock(state_dtype):
    conv = L.ShortConvolution(64, 4)
    mx.eval(conv.parameters())
    x = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
    st = mx.random.normal((1, 3, 64)).astype(state_dtype)
    mx.eval(x, st)

    fp.OPTS["conv_kernel"] = False
    ref_o, ref_ns = fp._sconv(conv, x, st)
    mx.eval(ref_o, ref_ns)

    fp.OPTS["conv_kernel"] = True
    got_o, got_ns = fp._sconv(conv, x, st)
    mx.eval(got_o, got_ns)

    assert got_ns.dtype == state_dtype
    assert got_o.dtype == x.dtype
    assert mx.max(mx.abs(got_o.astype(mx.float32) - ref_o.astype(mx.float32))).item() < 2e-2
    assert mx.max(mx.abs(got_ns.astype(mx.float32) - ref_ns.astype(mx.float32))).item() < 2e-2
