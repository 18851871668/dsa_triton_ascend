"""Torch helpers for sfa_grad_core_standalone._sfa_grad_core: input construction,
BSND->flat packing, buffer allocation, kernel launch, and output reshaping.

Reuses make_inputs/make_sparse_indices/expand_block_indices/run_sfa from
sfa_torch_utils for the forward pass (needed to produce out/smax/ssum that the
backward consumes). MLA-absorb: value = key (v_ptr aliases k_flat).
"""
from __future__ import annotations

import numpy as np
import torch

from sfa_torch_utils import (
    D_ROPE, DEVICE, _np, make_inputs, make_sparse_indices,
    expand_block_indices, run_sfa, to_np_f32,
)


def make_dout(B, S1, N1, D, dtype=torch.float16, device=DEVICE, seed=43):
    rng = np.random.RandomState(seed)
    return _np(rng.randn(B, S1, N1, D), dtype, device)


def run_sfa_grad(q, k, qr, kr, sparse_indices, do, out, smax, ssum,
                 sparse_block_size, sparse_mode, scale_value, device=DEVICE):
    """Launch the SFA backward kernel via _sfa_grad_core.

    Args:
        q:  [B, S1, N1, D]         query (from make_inputs)
        k:  [B, S2, 1, D]          key (from make_inputs)
        qr: [B, S1, N1, D_ROPE]    query_rope (from make_inputs)
        kr: [B, S2, 1, D_ROPE]     key_rope (from make_inputs)
        sparse_indices: [B, S1, 1, sparse_count] int32 (from make_sparse_indices)
        do: [B, S1, N1, D]         gradient of attention output (from make_dout)
        out:   [B, S1, N1, D]      forward attention output (from run_sfa)
        smax:  [B, 1, S1, N1] fp32 forward softmax_max (from run_sfa)
        ssum:  [B, 1, S1, N1] fp32 forward softmax_sum (from run_sfa)

    Returns:
        dq:  [B, S1, N1, D]
        dk:  [B, S2, 1, D]
        dv:  [B, S2, 1, D]
        dqr: [B, S1, N1, D_ROPE]
        dkr: [B, S2, 1, D_ROPE]
    """
    from sfa_grad_core_standalone import _sfa_grad_core
    B, S1, N1, D = q.shape
    S2 = k.shape[1]
    si_tok = expand_block_indices(sparse_indices, sparse_block_size)
    topK = si_tok.shape[-1]

    q_flat = q.contiguous()
    qr_flat = qr.contiguous()
    do_flat = do.contiguous()
    o_flat = out.contiguous()
    k_flat = k.reshape(B * S2, D).contiguous()
    kr_flat = kr.reshape(B * S2, D_ROPE).contiguous()
    v_flat = k_flat
    sparse_flat = si_tok.reshape(B * S1, topK).to(torch.int32).contiguous()
    sm_max_flat = smax.reshape(B * S1 * N1).to(torch.float32).contiguous()
    sm_sum_flat = ssum.reshape(B * S1 * N1).to(torch.float32).contiguous()

    dq_buf = torch.zeros((B, S1, N1, D), dtype=q.dtype, device=device)
    dqr_buf = torch.zeros((B, S1, N1, D_ROPE), dtype=qr.dtype, device=device)
    dk_buf = torch.zeros((B * S2, D), dtype=torch.float32, device=device)
    dkr_buf = torch.zeros((B * S2, D_ROPE), dtype=torch.float32, device=device)
    dv_buf = torch.zeros((B * S2, D), dtype=torch.float32, device=device)

    act_q = torch.full((B,), S1, dtype=torch.int32, device=device)
    act_k = torch.full((B,), S2, dtype=torch.int32, device=device)

    dq, dqr, dk, dkr, dv = _sfa_grad_core(
        q_flat, qr_flat, k_flat, kr_flat, v_flat, sparse_flat,
        do_flat, o_flat, sm_max_flat, sm_sum_flat,
        dq_buf, dqr_buf, dk_buf, dkr_buf, dv_buf,
        act_q, act_k,
        B * S1, S1, S2, N1, topK, D, D_ROPE,
        float(scale_value), sparse_mode,
    )

    dk = dk.reshape(B, S2, 1, D).to(k.dtype)
    dkr = dkr.reshape(B, S2, 1, D_ROPE).to(kr.dtype)
    dv = dv.reshape(B, S2, 1, D).to(k.dtype)

    return dq, dk, dv, dqr, dkr
