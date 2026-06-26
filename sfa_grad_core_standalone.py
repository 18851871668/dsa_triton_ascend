"""Standalone torch-flavored SFA grad core op (extracted from sparse_flash_attention_grad_triton).

Contains only the triton kernel (_sfa_grad_kernel) and the _sfa_grad_core launcher.
No mindspore dependency: buffers are passed in pre-allocated by the caller
(torch.zeros on device). Type annotations are lazy (PEP 563) so the module loads
even without torch installed; torch is only needed at call time.
"""
from __future__ import annotations

import os
if os.environ.get("TRITON_ENABLE_TASKQUEUE", "true").lower() in ("true", "1"):
    os.environ["TRITON_ENABLE_TASKQUEUE"] = "false"

import triton
import triton.language as tl
import triton.backends.ascend.runtime


def _next_pow2(x):
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _select_block_config(D, N1, topK):
    block_g = min(16, max(8, N1))
    block_k = 256
    if topK < block_k:
        block_k = 1 << (topK - 1).bit_length()
    return {"BLOCK_G": block_g, "BLOCK_K": block_k}


@triton.jit
def _sfa_grad_kernel(
    q_ptr, qr_ptr,
    k_ptr, kr_ptr, v_ptr,
    sparse_ptr,
    do_ptr, o_ptr,
    sm_max_ptr, sm_sum_ptr,
    dq_ptr, dqr_ptr,
    ds_ptr, dp_ptr,
    act_q_ptr, act_k_ptr,
    B_S1, S1, S2, N1, topK,
    D: tl.constexpr, D_ROPE: tl.constexpr,
    scale_value,
    sparse_mode: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_HC: tl.constexpr,
    NEED_CLAMP: tl.constexpr,
):
    """SFA backward — single pass, no atomic_add.

    Grid: (_next_pow2(B*S1),), pow2-padded.
    Each program owns one (b,s1) and BLOCK_G query heads.

    Single pass: for each hc, iterate blk_start, compute scores/P/dS,
    accumulate dq/dqr in UB, and store dS/P to GMEM workspace (sequential
    write, no atomic). dk/dv/dkr scatter-add is done on host via bmm +
    index_add_ after the kernel.
    """
    pid_bs1 = tl.program_id(0)
    bs1_in_range = pid_bs1 < B_S1
    pid_bs1 = tl.where(bs1_in_range, pid_bs1, 0)

    b = pid_bs1 // S1
    s1 = pid_bs1 % S1

    d_offs = tl.arange(0, D)
    dr_offs = tl.arange(0, D_ROPE)
    blk_k_offs = tl.arange(0, BLOCK_K)

    act_q = tl.load(act_q_ptr + b)
    act_k = tl.load(act_k_ptr + b)

    if sparse_mode == 0:
        threshold = act_k
    else:
        threshold = act_k - act_q + s1 + 1
    row_active = bs1_in_range & (s1 < act_q) & (threshold > 0)

    upper = tl.minimum(threshold, act_k)
    upper_f = upper.to(tl.float32)

    sp_base = pid_bs1 * topK
    k_rd_base = pid_bs1 * topK * D
    kr_rd_base = pid_bs1 * topK * D_ROPE
    dq_row_base = pid_bs1 * N1 * D
    dqr_row_base = pid_bs1 * N1 * D_ROPE
    ds_base = pid_bs1 * N1 * topK

    if NUM_HC == 1:
        HC_LOOP: tl.constexpr = 2
    else:
        HC_LOOP: tl.constexpr = NUM_HC
    for hc in range(HC_LOOP):
        g_offs = hc * BLOCK_G + tl.arange(0, BLOCK_G)
        g_valid = g_offs < N1
        if NEED_CLAMP:
            g_offs_s = tl.where(g_valid, g_offs, 0)
        else:
            g_offs_s = g_offs

        q_nope = tl.load(q_ptr + dq_row_base + g_offs_s[:, None] * D + d_offs[None, :],
                         mask=g_valid[:, None], other=0.0)
        q_rope = tl.load(qr_ptr + dqr_row_base + g_offs_s[:, None] * D_ROPE + dr_offs[None, :],
                         mask=g_valid[:, None], other=0.0)
        do_tile = tl.load(do_ptr + dq_row_base + g_offs_s[:, None] * D + d_offs[None, :],
                          mask=g_valid[:, None], other=0.0)
        o_tile = tl.load(o_ptr + dq_row_base + g_offs_s[:, None] * D + d_offs[None, :],
                         mask=g_valid[:, None], other=0.0)

        sm_max = tl.load(sm_max_ptr + pid_bs1 * N1 + g_offs_s, mask=g_valid, other=0.0)
        sm_sum = tl.load(sm_sum_ptr + pid_bs1 * N1 + g_offs_s, mask=g_valid, other=1.0)
        sm_sum = tl.where(sm_sum > 0.0, sm_sum, 1.0)

        delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)

        acc_dq = tl.zeros([BLOCK_G, D], dtype=tl.float32)
        acc_dqr = tl.zeros([BLOCK_G, D_ROPE], dtype=tl.float32)

        for blk_start in range(0, topK, BLOCK_K):
            blk_offs = blk_start + blk_k_offs
            blk_in_count = blk_offs < topK
            tok = tl.load(sparse_ptr + sp_base + blk_offs, mask=blk_in_count, other=-1)

            tok_f = tok.to(tl.float32)
            tok_valid = (blk_in_count
                         & (tok != -1)
                         & (tok_f < upper_f)
                         & row_active)

            k_full = tl.load(
                k_ptr + k_rd_base + blk_offs[:, None] * D + d_offs[None, :])
            kr_full = tl.load(
                kr_ptr + kr_rd_base + blk_offs[:, None] * D_ROPE + dr_offs[None, :])

            scores = tl.dot(q_nope, tl.trans(k_full)).to(tl.float32)
            scores += tl.dot(q_rope, tl.trans(kr_full)).to(tl.float32)
            scores = scores * scale_value

            P = tl.exp(scores - sm_max[:, None]) / sm_sum[:, None]
            P = tl.where(tok_valid[None, :], P, 0.0)

            dPv = tl.dot(do_tile, tl.trans(k_full)).to(tl.float32)
            dS = P * (dPv - delta[:, None]) * scale_value
            dS = tl.where(tok_valid[None, :], dS, 0.0)

            acc_dq += tl.dot(dS.to(k_full.dtype), k_full).to(tl.float32)
            acc_dqr += tl.dot(dS.to(kr_full.dtype), kr_full).to(tl.float32)

            ds_offs = ds_base + g_offs_s[:, None] * topK + blk_offs[None, :]
            store_mask = g_valid[:, None] & blk_in_count[None, :]
            tl.store(ds_ptr + ds_offs, dS, mask=store_mask)
            tl.store(dp_ptr + ds_offs, P, mask=store_mask)

        tl.store(dq_ptr + dq_row_base + g_offs_s[:, None] * D + d_offs[None, :],
                 acc_dq.to(dq_ptr.dtype.element_ty),
                 mask=g_valid[:, None] & row_active)
        tl.store(dqr_ptr + dqr_row_base + g_offs_s[:, None] * D_ROPE + dr_offs[None, :],
                 acc_dqr.to(dqr_ptr.dtype.element_ty),
                 mask=g_valid[:, None] & row_active)


def _sfa_grad_core(
    q_flat: torch.Tensor, qr_flat: torch.Tensor,
    k_flat: torch.Tensor, kr_flat: torch.Tensor, v_flat: torch.Tensor,
    sparse_flat: torch.Tensor,
    do_flat: torch.Tensor, o_flat: torch.Tensor,
    sm_max_flat: torch.Tensor, sm_sum_flat: torch.Tensor,
    dq_buf: torch.Tensor, dqr_buf: torch.Tensor,
    dk_buf: torch.Tensor, dkr_buf: torch.Tensor, dv_buf: torch.Tensor,
    act_q: torch.Tensor, act_k: torch.Tensor,
    B_S1: int, S1: int, S2: int, N1: int, topK: int,
    D: int, D_ROPE: int,
    scale_value: float,
    sparse_mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch
    cfg = _select_block_config(D, N1, topK)
    block_g = cfg["BLOCK_G"]
    num_hc = triton.cdiv(N1, block_g)
    need_clamp = (N1 % block_g) != 0

    grid = (_next_pow2(B_S1),)

    device = q_flat.device
    B = B_S1 // S1
    ds_buf = torch.zeros((B_S1, N1, topK), dtype=torch.float32, device=device)
    dp_buf = torch.zeros((B_S1, N1, topK), dtype=torch.float32, device=device)

    _sfa_grad_kernel[grid](
        q_flat, qr_flat,
        k_flat, kr_flat, v_flat,
        sparse_flat,
        do_flat, o_flat,
        sm_max_flat, sm_sum_flat,
        dq_buf, dqr_buf,
        ds_buf, dp_buf,
        act_q, act_k,
        B_S1, S1, S2, N1, topK,
        D, D_ROPE,
        scale_value,
        sparse_mode=sparse_mode,
        BLOCK_G=block_g,
        BLOCK_K=cfg["BLOCK_K"],
        NUM_HC=num_hc,
        NEED_CLAMP=need_clamp,
        multibuffer=False,
    )

    batch_offsets = torch.arange(B, dtype=torch.int32, device=device) * S2
    sparse_global = sparse_flat.reshape(B, S1, topK) + batch_offsets.reshape(B, 1, 1)
    sparse_1d = sparse_global.reshape(-1).long()

    dk_contrib = torch.bmm(ds_buf.reshape(B_S1, N1, topK).transpose(1, 2),
                           q_flat.reshape(B_S1, N1, D).to(torch.float32))
    dk_buf.index_add_(0, sparse_1d, dk_contrib.reshape(-1, D))

    dv_contrib = torch.bmm(dp_buf.reshape(B_S1, N1, topK).transpose(1, 2),
                           do_flat.reshape(B_S1, N1, D).to(torch.float32))
    dv_buf.index_add_(0, sparse_1d, dv_contrib.reshape(-1, D))

    dkr_contrib = torch.bmm(ds_buf.reshape(B_S1, N1, topK).transpose(1, 2),
                            qr_flat.reshape(B_S1, N1, D_ROPE).to(torch.float32))
    dkr_buf.index_add_(0, sparse_1d, dkr_contrib.reshape(-1, D_ROPE))

    return dq_buf, dqr_buf, dk_buf, dkr_buf, dv_buf
