"""Torch-flavored golden tests for sfa_grad_core_standalone._sfa_grad_core.

Mirrors test_sfa_grad_triton.test_golden/test_basic but uses torch tensors and
calls _sfa_grad_core directly (no MindSpore cell). The forward stats (out/smax/
ssum) are produced by run_sfa from sfa_torch_utils. Run:
    pytest test_sfa_grad_torch.py -v
    python test_sfa_grad_torch.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sfa_torch_utils import (
    D_ROPE, make_inputs, make_sparse_indices, run_sfa, to_np_f32,
)
from sfa_grad_torch_utils import make_dout, run_sfa_grad
from sparse_flash_attention_numpy import BF16
from sparse_flash_attention_grad_numpy import sparse_flash_attention_grad_golden_bsnd

_DT_NP = {torch.float16: np.float16, torch.bfloat16: BF16, torch.float32: np.float32}


def _allclose(a, b, bf16=False, scale=1.0, pct_thd=99.5):
    if bf16:
        rtol = 6e-2
        atol = 5e-2 * scale
    else:
        rtol = 3e-2
        atol = 2e-2 * scale
    max_diff_hd = 10

    a = np.asarray(a, np.float32).flatten()
    b = np.asarray(b, np.float32).flatten()
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    fail_mask = ~close
    if fail_mask.any():
        fa, fb = a[fail_mask], b[fail_mask]
        diff = np.abs(fa - fb)
        rtol_only = np.abs(fb) * rtol
        print(f"[diag] failed {fail_mask.sum()}/{fail_mask.size}  "
              f"atol={atol:.4e} rtol={rtol}")
        print(f"[diag]   |diff|  min={diff.min():.6e} max={diff.max():.6e}")
        print(f"[diag]   rtol*|b| min={rtol_only.min():.6e} max={rtol_only.max():.6e}")
        print(f"[diag]   atol-dominated (|b|<{atol/rtol:.4f}): "
              f"{(np.abs(fb) < atol/rtol).sum()}")
    pass_pct = close.sum() / close.size * 100.0
    if pass_pct < pct_thd:
        return False

    diff_abs = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-10
    rel_err = diff_abs / denom
    if np.any(rel_err[~close] >= max_diff_hd):
        return False
    return True


# ---------------------------------------------------------------------------
# triton vs numpy golden -- algorithm correctness, runs on any shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,S1,S2,N1,sc,bs,mode", [
    (1, 4, 128, 8, 64, 1, 3),       # token-wise, rightDownCausal
    (2, 16, 256, 16, 128, 1, 3),    # bigger, multi-batch
    (1, 8, 128, 8, 32, 2, 3),       # block-wise (block_size=2)
    (1, 8, 256, 16, 32, 4, 3),      # block-wise (block_size=4)
    (1, 1, 128, 8, 16, 1, 3),       # S1=1 single query row
    (1, 4, 128, 1, 64, 1, 3),       # N1=1 single head (MQA degenerate, head mask)
    (1, 4, 128, 128, 64, 1, 3),     # N1=128 upper bound (BLOCK_G head tiling)
    (1, 4, 2048, 8, 2048, 1, 3),    # topK=2048, rightDownCausal
])
@pytest.mark.parametrize("D", [128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_golden(B, S1, S2, N1, sc, bs, mode, D, dtype):
    """Compare triton SFA backward with the numpy golden (D 128/256/512, fp16/bf16)."""
    np_dtype = _DT_NP[dtype]
    q, k, qr, kr = make_inputs(B, S1, S2, N1, D, dtype)
    do = make_dout(B, S1, N1, D, dtype)
    si = make_sparse_indices(B, S1, S2, sc, bs, mode)
    scale = 1.0 / np.sqrt(D + D_ROPE)

    # Forward to get out/smax/ssum (consumed by the backward)
    out, smax, ssum = run_sfa(q, k, qr, kr, si, bs, mode, scale, return_lse=True)

    # Backward
    dq, dk, dv, dqr, dkr = run_sfa_grad(
        q, k, qr, kr, si, do, out, smax, ssum, bs, mode, scale)

    # Numpy golden
    g_dq, g_dk, g_dv, g_dqr, g_dkr = sparse_flash_attention_grad_golden_bsnd(
        to_np_f32(q), to_np_f32(k), to_np_f32(k), si.cpu().numpy(),
        to_np_f32(do), to_np_f32(out), to_np_f32(smax), to_np_f32(ssum),
        to_np_f32(qr), to_np_f32(kr),
        scale, [S1] * B, [S2] * B,
        sparse_block_size=bs, sparse_mode=mode, dtype=np_dtype,
    )

    bf16 = (dtype == torch.bfloat16)
    sc_atol = math.sqrt(S1)
    assert _allclose(to_np_f32(dq), g_dq, bf16=bf16), "d_query mismatch vs golden"
    assert _allclose(to_np_f32(dqr), g_dqr, bf16=bf16), "d_query_rope mismatch vs golden"
    assert _allclose(to_np_f32(dk), g_dk, bf16=bf16, scale=sc_atol), "d_key mismatch vs golden"
    assert _allclose(to_np_f32(dv), g_dv, bf16=bf16, scale=sc_atol), "d_value mismatch vs golden"
    assert _allclose(to_np_f32(dkr), g_dkr, bf16=bf16, scale=sc_atol), "d_key_rope mismatch vs golden"


# ---------------------------------------------------------------------------
# functional self-checks -- shapes beyond CANN reference constraints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,S1,S2,N1,sc", [
    (1, 128, 1024, 64, 512),
    (2, 64, 512, 32, 256),
    (1, 16, 2048, 64, 2048),   # topK=2048
    (1, 512, 4096, 64, 2048),  # perf shape, topK=2048 (sparse_count > S2, clamp)
])
@pytest.mark.parametrize("D", [128, 256, 512])
@pytest.mark.parametrize("sparse_mode", [3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_basic(B, S1, S2, N1, sc, D, sparse_mode, dtype):
    """Shape / dtype / finiteness checks (no reference comparison)."""
    q, k, qr, kr = make_inputs(B, S1, S2, N1, D, dtype)
    do = make_dout(B, S1, N1, D, dtype)
    si = make_sparse_indices(B, S1, S2, sc, 1, sparse_mode)
    scale = 1.0 / np.sqrt(D + D_ROPE)

    out, smax, ssum = run_sfa(q, k, qr, kr, si, 1, sparse_mode, scale, return_lse=True)
    dq, dk, dv, dqr, dkr = run_sfa_grad(
        q, k, qr, kr, si, do, out, smax, ssum, 1, sparse_mode, scale)

    assert dq.shape == (B, S1, N1, D) and dq.dtype == dtype, f"dq {dq.shape}/{dq.dtype}"
    assert dqr.shape == (B, S1, N1, D_ROPE) and dqr.dtype == dtype, f"dqr {dqr.shape}"
    assert dk.shape == (B, S2, 1, D) and dk.dtype == dtype, f"dk {dk.shape}/{dk.dtype}"
    assert dv.shape == (B, S2, 1, D) and dv.dtype == dtype, f"dv {dv.shape}"
    assert dkr.shape == (B, S2, 1, D_ROPE) and dkr.dtype == dtype, f"dkr {dkr.shape}"
    for name, t in (("dq", dq), ("dk", dk), ("dv", dv), ("dqr", dqr), ("dkr", dkr)):
        assert bool(torch.isfinite(t).all()), f"{name} has NaN/inf"


# ---------------------------------------------------------------------------
# smoke -- fast per-edit regression. Each case is a FULL param tuple (no D/dtype
# cross-product), so the count is exactly what's listed. Reuses test_golden
# bodies. Run after every edit:  pytest -k smoke
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,S1,S2,N1,sc,bs,mode,D,dtype", [
    (2, 16, 256, 16, 128, 1, 3, 512, torch.float16),    # multi-batch
    (1, 4, 128, 8, 64, 1, 3, 512, torch.bfloat16),      # bf16
    (1, 1, 128, 8, 16, 1, 3, 128, torch.float16),       # S1=1 single row, D=128
    (1, 8, 128, 8, 32, 2, 3, 512, torch.float16),       # block-wise (bs=2)
    (1, 4, 2048, 8, 2048, 1, 3, 256, torch.float16),    # topK=2048
])
def test_smoke_golden(B, S1, S2, N1, sc, bs, mode, D, dtype):
    """Fast backward golden subset covering the BLOCK_S1 folding risk points."""
    test_golden(B, S1, S2, N1, sc, bs, mode, D, dtype)


if __name__ == "__main__":
    test_golden(1, 4, 128, 8, 64, 1, 3, 512, torch.float16)
    print("grad golden test (D=512, token-wise, mode3, fp16) passed!")
    test_golden(1, 4, 128, 8, 64, 1, 3, 128, torch.float16)
    print("grad golden test (D=128, token-wise, mode3, fp16) passed!")
    test_golden(1, 8, 128, 8, 32, 2, 3, 256, torch.float16)
    print("grad golden test (D=256, block-wise bs=2, mode3, fp16) passed!")
    test_golden(1, 4, 128, 8, 64, 1, 3, 512, torch.bfloat16)
    print("grad golden test (D=512, token-wise, mode3, bf16) passed!")
    test_golden(1, 4, 2048, 8, 2048, 1, 3, 256, torch.float16)
    print("grad golden test (D=256, topK=2048, mode3, fp16) passed!")
