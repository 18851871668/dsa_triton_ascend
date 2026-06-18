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
    D_ROPE, make_inputs, make_sparse_indices, run_sfa, to_np_f32, allclose,
)
from sparse_flash_attention_numpy import sparse_flash_attention_golden_bsnd, BF16

_DT_NP = {torch.float16: np.float16, torch.bfloat16: BF16, torch.float32: np.float32}


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
    assert allclose(to_np_f32(out), out_g, bf16=bf16, pct_thd=pct), "out mismatch vs golden"
    assert allclose(to_np_f32(smax), smax_g, bf16=bf16), "smax mismatch vs golden"
    assert allclose(to_np_f32(ssum), ssum_g, bf16=bf16), "ssum mismatch vs golden"


@pytest.mark.parametrize("B,S1,S2,N1,sc", [
    (1, 128, 1024, 64, 512),
    (2, 64, 512, 32, 256),
    (1, 16, 2048, 64, 2048),   # two-pass path
])
@pytest.mark.parametrize("D", [128, 256, 512])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_basic(B, S1, S2, N1, sc, D, return_lse, dtype):
    q, k, qr, kr = make_inputs(B, S1, S2, N1, D, dtype)
    si = make_sparse_indices(B, S1, S2, sc, 1, 3)
    scale = 1.0 / np.sqrt(D + D_ROPE)
    out, smax, ssum = run_sfa(q, k, qr, kr, si, 1, 3, scale, return_lse)

    assert out.shape == (B, S1, N1, D), f"out shape {out.shape}"
    assert out.dtype == dtype, f"out dtype {out.dtype}"
    assert smax.shape == (B, 1, S1, N1), f"smax shape {smax.shape}"
    assert ssum.shape == (B, 1, S1, N1), f"ssum shape {ssum.shape}"
    assert bool(torch.isfinite(out).all()), "out has NaN/inf"


if __name__ == "__main__":
    test_golden(1, 4, 128, 8, 64, 1, 3, 512, torch.float16)
    print("golden (D=512, token-wise, mode3, fp16) passed!")
    test_golden(1, 4, 128, 8, 64, 1, 3, 128, torch.float16)
    print("golden (D=128, token-wise, mode3, fp16) passed!")
    test_golden(1, 8, 128, 8, 32, 2, 3, 256, torch.float16)
    print("golden (D=256, block-wise bs=2, mode3, fp16) passed!")
    test_golden(1, 4, 2048, 8, 2048, 1, 3, 256, torch.float16)
    print("golden (D=256, topK=2048, two-pass, mode3, fp16) passed!")
