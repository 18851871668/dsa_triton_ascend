"""Torch-flavored golden tests for sfa_core_standalone._sfa_core.

Mirrors test_sfa_triton.test_golden/test_basic but uses torch tensors and calls
_sfa_core directly (no MindSpore cell). Run:
    pytest test_sfa_torch.py -v
    python test_sfa_torch.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sfa_torch_utils import (
    D_ROPE, make_inputs, make_sparse_indices, run_sfa, to_np_f32,
)
from sparse_flash_attention_numpy import sparse_flash_attention_golden_bsnd, BF16

_DT_NP = {torch.float16: np.float16, torch.bfloat16: BF16, torch.float32: np.float32}


def _allclose(a, b, bf16=False, pct_thd=None, rtol=None, atol=None, max_diff_hd=10):
    if bf16:
        rtol = 7.8125e-3 if rtol is None else rtol
        atol = 7e-4 if atol is None else atol
    else:
        rtol = 5e-3 if rtol is None else rtol
        atol = 2.5e-4 if atol is None else atol
    pct_thd = 99.5 if pct_thd is None else pct_thd

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
              f"atol={atol} rtol={rtol}")
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
# triton vs numpy golden — algorithm correctness, runs on any shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,S1,S2,N1,sc,bs,mode", [
    (1, 4, 128, 8, 64, 1, 3),       # token-wise, rightDownCausal
    (2, 16, 256, 16, 128, 1, 3),    # bigger, multi-batch
    (1, 8, 128, 8, 32, 2, 3),       # block-wise (block_size=2)
    (1, 8, 256, 16, 32, 4, 3),      # block-wise (block_size=4)
    (1, 1, 128, 8, 16, 1, 3),       # S1=1 single query row
    (1, 4, 2048, 8, 2048, 1, 3),    # two-pass path, rightDownCausal
])
@pytest.mark.parametrize("D", [128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_golden(B, S1, S2, N1, sc, bs, mode, D, dtype):
    np_dtype = _DT_NP[dtype]
    q, k, qr, kr = make_inputs(B, S1, S2, N1, D, dtype)
    si = make_sparse_indices(B, S1, S2, sc, bs, mode)
    scale = 1.0 / np.sqrt(D + D_ROPE)

    out, smax, ssum = run_sfa(q, k, qr, kr, si, bs, mode, scale, return_lse=True)

    out_g, smax_g, ssum_g = sparse_flash_attention_golden_bsnd(
        to_np_f32(q), to_np_f32(k), to_np_f32(k), si.cpu().numpy(),
        to_np_f32(qr), to_np_f32(kr),
        scale, [S1] * B, [S2] * B,
        sparse_block_size=bs, sparse_mode=mode,
        return_softmax_lse=True, dtype=np_dtype,
    )
    bf16 = (dtype == torch.bfloat16)
    pct = 99.0 if bf16 else 99.5
    assert _allclose(to_np_f32(out), out_g, bf16=bf16, pct_thd=pct), "out mismatch vs golden"
    assert _allclose(to_np_f32(smax), smax_g, bf16=bf16), "smax mismatch vs golden"
    assert _allclose(to_np_f32(ssum), ssum_g, bf16=bf16), "ssum mismatch vs golden"


# ---------------------------------------------------------------------------
# functional self-checks — shapes beyond CANN reference constraints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,S1,S2,N1,sc", [
    (1, 128, 1024, 64, 512),
    (2, 64, 512, 32, 256),
    (1, 16, 2048, 64, 2048),   # two-pass path
    (1, 512, 4096, 64, 2048),  # perf shape
])
@pytest.mark.parametrize("D", [128, 256, 512])
@pytest.mark.parametrize("sparse_mode", [3])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_basic(B, S1, S2, N1, sc, D, sparse_mode, return_lse, dtype):
    q, k, qr, kr = make_inputs(B, S1, S2, N1, D, dtype)
    si = make_sparse_indices(B, S1, S2, sc, 1, sparse_mode)
    scale = 1.0 / np.sqrt(D + D_ROPE)
    out, smax, ssum = run_sfa(q, k, qr, kr, si, 1, sparse_mode, scale, return_lse)

    assert out.shape == (B, S1, N1, D), f"out shape {out.shape}"
    assert out.dtype == dtype, f"out dtype {out.dtype}"
    assert smax.shape == (B, 1, S1, N1), f"smax shape {smax.shape}"
    assert ssum.shape == (B, 1, S1, N1), f"ssum shape {ssum.shape}"
    assert bool(torch.isfinite(out).all()), "out has NaN/inf"


# ---------------------------------------------------------------------------
# smoke — fast per-edit regression. Each case is a FULL param tuple (no D/dtype
# cross-product), so the count is exactly what's listed. Reuses test_golden
# bodies. Run after every edit:  pytest -k smoke
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,S1,S2,N1,sc,bs,mode,D,dtype", [
    (2, 16, 256, 16, 128, 1, 3, 512, torch.float16),  # multi-batch: pid crosses batch boundary
    (1, 4, 128, 8, 64, 1, 3, 512, torch.bfloat16),    # B_S1<BLOCK_S1: tail-program masking + bf16
    (1, 1, 128, 8, 16, 1, 3, 128, torch.float16),     # S1=1 single row, D=128
    (1, 8, 128, 8, 32, 2, 3, 512, torch.float16),     # block-wise (bs=2)
    (1, 4, 2048, 8, 2048, 1, 3, 256, torch.float16),   # two-pass path
])
def test_smoke_golden(B, S1, S2, N1, sc, bs, mode, D, dtype):
    """Fast golden subset covering the BLOCK_S1 folding risk points."""
    test_golden(B, S1, S2, N1, sc, bs, mode, D, dtype)


if __name__ == "__main__":
    test_golden(1, 4, 128, 8, 64, 1, 3, 512, torch.float16)
    print("golden test (D=512, token-wise, mode3, fp16) passed!")
    test_golden(1, 4, 128, 8, 64, 1, 3, 128, torch.float16)
    print("golden test (D=128, token-wise, mode3, fp16) passed!")
    test_golden(1, 8, 128, 8, 32, 2, 3, 256, torch.float16)
    print("golden test (D=256, block-wise bs=2, mode3, fp16) passed!")
    test_golden(1, 4, 128, 8, 64, 1, 3, 512, torch.bfloat16)
    print("golden test (D=512, token-wise, mode3, bf16) passed!")
    test_golden(1, 4, 2048, 8, 2048, 1, 3, 256, torch.float16)
    print("golden test (D=256, topK=2048, two-pass, mode3, fp16) passed!")
