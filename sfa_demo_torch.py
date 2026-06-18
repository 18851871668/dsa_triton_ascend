"""Minimal torch caller demo for sfa_core_standalone._sfa_core.

Builds BSND inputs, packs to flat, allocates buffers, launches the kernel,
prints output metadata, and verifies against the numpy golden. Run on Ascend:
    python sfa_demo_torch.py
"""
from __future__ import annotations

import numpy as np
import torch

from sfa_torch_utils import (
    D_ROPE, make_inputs, make_sparse_indices, run_sfa, to_np_f32, allclose,
)
from sparse_flash_attention_numpy import sparse_flash_attention_golden_bsnd


def main():
    B, S1, S2, N1 = 1, 4, 128, 8
    sparse_count, sparse_block_size, sparse_mode = 64, 1, 3
    D = 512
    dtype = torch.float16
    np_dtype = np.float16
    scale = 1.0 / np.sqrt(D + D_ROPE)

    q, k, qr, kr = make_inputs(B, S1, S2, N1, D, dtype)
    si = make_sparse_indices(B, S1, S2, sparse_count, sparse_block_size, sparse_mode)

    out, smax, ssum = run_sfa(
        q, k, qr, kr, si, sparse_block_size, sparse_mode, scale, return_lse=True)

    print(f"out   {tuple(out.shape)} {out.dtype}")
    print(f"smax  {tuple(smax.shape)} {smax.dtype}")
    print(f"ssum  {tuple(ssum.shape)} {ssum.dtype}")
    print(f"out finite: {bool(torch.isfinite(out).all())}")

    out_g, smax_g, ssum_g = sparse_flash_attention_golden_bsnd(
        to_np_f32(q), to_np_f32(k), to_np_f32(k), si.cpu().numpy(),
        to_np_f32(qr), to_np_f32(kr),
        scale, [S1] * B, [S2] * B,
        sparse_block_size=sparse_block_size, sparse_mode=sparse_mode,
        return_softmax_lse=True, dtype=np_dtype,
    )
    bf16 = (dtype == torch.bfloat16)
    assert allclose(to_np_f32(out), out_g, bf16=bf16), "out mismatch vs golden"
    assert allclose(to_np_f32(smax), smax_g, bf16=bf16), "smax mismatch vs golden"
    assert allclose(to_np_f32(ssum), ssum_g, bf16=bf16), "ssum mismatch vs golden"
    print("golden check passed!")


if __name__ == "__main__":
    main()
