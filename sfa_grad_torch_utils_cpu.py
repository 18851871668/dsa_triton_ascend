"""Torch helpers for sfa_grad_core_standalone._sfa_grad_core — CPU-generate variant.

Inputs are generated on CPU (no NPU memory used for random data). Forward stats
(out/smax/ssum) are produced by run_sfa_kernel on NPU. Backward inputs are
flattened on CPU (for CPU-origin tensors) or NPU (for forward outputs), then all
moved to NPU before timing.

MLA-absorb: value = key (v_ptr aliases k_flat).
"""
from __future__ import annotations

import numpy as np
import torch

from sfa_torch_utils_cpu import (
    D_ROPE, DEVICE, make_inputs, make_sparse_indices,
    expand_block_indices, to_device, run_sfa,
)


def make_dout(B, S1, N1, D, dtype=torch.float16, seed=43):
    rng = np.random.RandomState(seed)
    return torch.from_numpy(np.ascontiguousarray(rng.randn(B, S1, N1, D))).to(dtype)


def prepare_sfa_grad_inputs(q, k, qr, kr, sparse_indices, do, out, smax, ssum,
                            sparse_block_size):
    """Flatten all backward inputs on their current device."""
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
    sparse_flat = si_tok.reshape(B * S1, topK).to(torch.int32).contiguous()
    sm_max_flat = smax.reshape(B * S1 * N1).to(torch.float32).contiguous()
    sm_sum_flat = ssum.reshape(B * S1 * N1).to(torch.float32).contiguous()

    # Pre-gather K/KR into contiguous [B*S1, topK, *] for sequential kernel access.
    dev = q_flat.device
    batch_offsets = torch.arange(B, dtype=torch.int32, device=dev) * S2
    sparse_global = sparse_flat.reshape(B, S1, topK) + batch_offsets.reshape(B, 1, 1)
    sparse_1d = sparse_global.reshape(-1).clamp(min=0)
    k_gathered = torch.index_select(k_flat, 0, sparse_1d).reshape(B * S1, topK, D).contiguous()
    kr_gathered = torch.index_select(kr_flat, 0, sparse_1d).reshape(B * S1, topK, D_ROPE).contiguous()
    v_gathered = k_gathered

    return (q_flat, qr_flat, k_gathered, kr_gathered, v_gathered, sparse_flat,
            do_flat, o_flat, sm_max_flat, sm_sum_flat,
            B, S1, S2, N1, topK, D)


def prepare_sfa_grad_inputs_cpu(q, k, qr, kr, sparse_indices, do, out, smax, ssum,
                                sparse_block_size, device=DEVICE):
    """Flatten CPU-origin tensors on CPU, move to device; reshape NPU tensors on device.

    q/k/qr/kr/do/sparse_indices are on CPU (from make_inputs/make_dout).
    out/smax/ssum are on NPU (from run_sfa_kernel).
    """
    B, S1, N1, D = q.shape
    S2 = k.shape[1]
    si_tok = expand_block_indices(sparse_indices, sparse_block_size)  # CPU
    topK = si_tok.shape[-1]

    # Flatten on CPU
    q_flat = q.contiguous()
    qr_flat = qr.contiguous()
    do_flat = do.contiguous()
    k_flat = k.reshape(B * S2, D).contiguous()
    kr_flat = kr.reshape(B * S2, D_ROPE).contiguous()
    sparse_flat = si_tok.reshape(B * S1, topK).to(torch.int32).contiguous()

    # Pre-gather K/KR on CPU (index_select is efficient on CPU too)
    batch_offsets = torch.arange(B, dtype=torch.int32) * S2
    sparse_global = sparse_flat.reshape(B, S1, topK) + batch_offsets.reshape(B, 1, 1)
    sparse_1d = sparse_global.reshape(-1).clamp(min=0)
    k_gathered = torch.index_select(k_flat, 0, sparse_1d).reshape(B * S1, topK, D).contiguous()
    kr_gathered = torch.index_select(kr_flat, 0, sparse_1d).reshape(B * S1, topK, D_ROPE).contiguous()
    v_gathered = k_gathered

    # Move CPU tensors to device
    q_flat = q_flat.to(device)
    qr_flat = qr_flat.to(device)
    do_flat = do_flat.to(device)
    k_gathered = k_gathered.to(device)
    kr_gathered = kr_gathered.to(device)
    v_gathered = k_gathered
    sparse_flat = sparse_flat.to(device)

    # Reshape NPU tensors (already on device)
    o_flat = out.contiguous()
    sm_max_flat = smax.reshape(B * S1 * N1).to(torch.float32).contiguous()
    sm_sum_flat = ssum.reshape(B * S1 * N1).to(torch.float32).contiguous()

    return (q_flat, qr_flat, k_gathered, kr_gathered, v_gathered, sparse_flat,
            do_flat, o_flat, sm_max_flat, sm_sum_flat,
            B, S1, S2, N1, topK, D)


def run_sfa_grad_kernel(q_flat, qr_flat, k_flat, kr_flat, v_flat, sparse_flat,
                        do_flat, o_flat, sm_max_flat, sm_sum_flat,
                        B, S1, S2, N1, topK, D, scale_value, sparse_mode):
    """Launch backward kernel with pre-prepared flat tensors. Only allocs output
    buffers + act_q/act_k on device, then launches. Used for perf (excludes
    host-side prep + H2D)."""
    from sfa_grad_core_standalone import _sfa_grad_core
    device = q_flat.device
    dq_buf = torch.zeros((B, S1, N1, D), dtype=q_flat.dtype, device=device)
    dqr_buf = torch.zeros((B, S1, N1, D_ROPE), dtype=qr_flat.dtype, device=device)
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

    dk = dk.reshape(B, S2, 1, D).to(k_flat.dtype)
    dkr = dkr.reshape(B, S2, 1, D_ROPE).to(kr_flat.dtype)
    dv = dv.reshape(B, S2, 1, D).to(k_flat.dtype)
    return dq, dk, dv, dqr, dkr


def run_sfa_grad(q, k, qr, kr, sparse_indices, do, out, smax, ssum,
                 sparse_block_size, sparse_mode, scale_value):
    """Full backward flow: flatten inputs + launch kernel. For correctness tests."""
    q_flat, qr_flat, k_flat, kr_flat, v_flat, sparse_flat, \
        do_flat, o_flat, sm_max_flat, sm_sum_flat, \
        B, S1, S2, N1, topK, D = prepare_sfa_grad_inputs(
            q, k, qr, kr, sparse_indices, do, out, smax, ssum, sparse_block_size)
    return run_sfa_grad_kernel(
        q_flat, qr_flat, k_flat, kr_flat, v_flat, sparse_flat,
        do_flat, o_flat, sm_max_flat, sm_sum_flat,
        B, S1, S2, N1, topK, D, scale_value, sparse_mode)
