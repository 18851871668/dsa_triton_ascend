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
    # Ascend requires kernel grid dims to be powers of 2.
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _select_block_config(D, N1):
    block_g = min(16, max(8, N1))
    return {"BLOCK_G": block_g, "BLOCK_K": 32, "BLOCK_D": 128}


@triton.jit
def _sfa_grad_kernel(
    q_ptr, qr_ptr,                       # query[B,S1,N1,D], query_rope[B,S1,N1,Dr]
    k_ptr, kr_ptr, v_ptr,                # key/key_rope/value, all [B,S2,1,*] (v aliases k)
    sparse_ptr,                          # token indices [B,S1,1,topK] int32 (block pre-expanded)
    do_ptr, o_ptr,                       # d_out[B,S1,N1,D], out[B,S1,N1,D]
    sm_max_ptr, sm_sum_ptr,              # forward softmax stats, flat (b*S1+s1)*N1 + g
    dq_ptr, dqr_ptr,                     # outputs: d_query, d_query_rope
    dk_ptr, dkr_ptr, dv_ptr,             # outputs (fp32 workspace): d_key, d_key_rope, d_value
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
    """SFA backward over sparsely gathered KV (BSND, MQA / N2=1), single pass.

    Grid: (_next_pow2(B*S1),), pow2-padded.
    Each program owns one (b,s1) and BLOCK_G query heads.

    Per topK block: rebuild scores (q*k) -> P (from saved softmax stats) -> dPv
    (dO*v, v=k_nope) -> dS = P*(dPv - delta)*scale. Accumulate dq/dqr resident
    (no cross-program contention -- each head row owned by one program); scatter-add
    dk/dkr (dS*q) and dv (P*dO) into fp32 workspaces (many s1 rows hit one KV token).

    No early-return (triton-ascend drops stores after early-return): inactive rows
    are folded into tok_valid so dS/P become 0 and all contributions vanish.
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

    # Base offsets (shared across all stages)
    sp_base = pid_bs1 * topK
    k_base = b * S2 * D
    kr_base = b * S2 * D_ROPE
    v_base = b * S2 * D
    dq_row_base = pid_bs1 * N1 * D       # [B*S1, N1, D]
    dqr_row_base = pid_bs1 * N1 * D_ROPE  # [B*S1, N1, D_ROPE]

    # ──────────────────────────────────────────────────────────────
    # Stage A: Compute dq/dqr for each head chunk.
    # hc-outer, blk_start-inner loop.
    # NO MTE3 writes in the hot loop -> Cube not blocked by Vector.
    # ──────────────────────────────────────────────────────────────
    # Use HC_LOOP to avoid Triton compiler crash on `for hc in range(1)`
    # (scf.For assertion failure in ttir_to_linalg). When NUM_HC == 1,
    # the extra hc=1 iteration has g_valid all-False, contributing nothing.
    if NUM_HC == 1:
        HC_LOOP: tl.constexpr = 2
    else:
        HC_LOOP: tl.constexpr = NUM_HC
    for hc in range(HC_LOOP):
        g_offs = hc * BLOCK_G + tl.arange(0, BLOCK_G)
        g_valid = g_offs < N1
        # When NEED_CLAMP is False (N1 is a multiple of BLOCK_G), skip the
        # tl.where and use g_offs directly -- avoids a select instruction per
        # load address, which measurably hurts performance on Ascend.
        if NEED_CLAMP:
            g_offs_s = tl.where(g_valid, g_offs, 0)
        else:
            g_offs_s = g_offs

        # Load per-head-chunk inputs
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
            tok_clamped = tl.where(tok_valid, tok, 0)

            k_full = tl.load(
                k_ptr + k_base + tok_clamped[:, None] * D + d_offs[None, :],
                mask=tok_valid[:, None], other=0.0)
            kr_full = tl.load(
                kr_ptr + kr_base + tok_clamped[:, None] * D_ROPE + dr_offs[None, :],
                mask=tok_valid[:, None], other=0.0)

            scores = tl.dot(q_nope, tl.trans(k_full)).to(tl.float32)
            scores += tl.dot(q_rope, tl.trans(kr_full)).to(tl.float32)
            scores = scores * scale_value

            P = tl.exp(scores - sm_max[:, None]) / sm_sum[:, None]

            dPv = tl.dot(do_tile, tl.trans(k_full)).to(tl.float32)
            dS = P * (dPv - delta[:, None]) * scale_value

            acc_dq += tl.dot(dS.to(k_full.dtype), k_full).to(tl.float32)
            acc_dqr += tl.dot(dS.to(kr_full.dtype), kr_full).to(tl.float32)

        # Store dq/dqr for this head chunk (tl.store, no atomic -- one program per row)
        tl.store(dq_ptr + dq_row_base + g_offs_s[:, None] * D + d_offs[None, :],
                 acc_dq.to(dq_ptr.dtype.element_ty),
                 mask=g_valid[:, None] & row_active)
        tl.store(dqr_ptr + dqr_row_base + g_offs_s[:, None] * D_ROPE + dr_offs[None, :],
                 acc_dqr.to(dqr_ptr.dtype.element_ty),
                 mask=g_valid[:, None] & row_active)

    # ──────────────────────────────────────────────────────────────
    # Stage B1: Accumulate dk/dkr across head chunks.
    # blk_start-outer, hc-inner loop.
    # dk_acc/dkr_acc persist across hc iterations within one blk_start.
    # After all hc: single atomic_add per output.
    # ──────────────────────────────────────────────────────────────
    for blk_start in range(0, topK, BLOCK_K):
        blk_offs = blk_start + blk_k_offs
        blk_in_count = blk_offs < topK
        tok = tl.load(sparse_ptr + sp_base + blk_offs, mask=blk_in_count, other=-1)

        tok_f = tok.to(tl.float32)
        tok_valid = blk_in_count & (tok != -1) & (tok_f < upper_f) & row_active
        tok_clamped = tl.where(tok_valid, tok, 0)

        # Load k_full/kr_full once per blk_start (shared across all hc)
        k_full = tl.load(
            k_ptr + k_base + tok_clamped[:, None] * D + d_offs[None, :],
            mask=tok_valid[:, None], other=0.0)
        kr_full = tl.load(
            kr_ptr + kr_base + tok_clamped[:, None] * D_ROPE + dr_offs[None, :],
            mask=tok_valid[:, None], other=0.0)

        dk_acc = tl.zeros([BLOCK_K, D], dtype=tl.float32)
        dkr_acc = tl.zeros([BLOCK_K, D_ROPE], dtype=tl.float32)

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

            scores = tl.dot(q_nope, tl.trans(k_full)).to(tl.float32)
            scores += tl.dot(q_rope, tl.trans(kr_full)).to(tl.float32)
            scores = scores * scale_value

            P = tl.exp(scores - sm_max[:, None]) / sm_sum[:, None]

            dPv = tl.dot(do_tile, tl.trans(k_full)).to(tl.float32)
            dS = P * (dPv - delta[:, None]) * scale_value

            dk_acc += tl.dot(tl.trans(dS).to(q_nope.dtype), q_nope).to(tl.float32)
            dkr_acc += tl.dot(tl.trans(dS).to(q_rope.dtype), q_rope).to(tl.float32)

        # Single atomic_add for dk (accumulated across all head chunks)
        dk_offs = v_base + tok_clamped[:, None] * D + d_offs[None, :]
        tl.atomic_add(dk_ptr + dk_offs, dk_acc, mask=tok_valid[:, None])

        dkr_offs = kr_base + tok_clamped[:, None] * D_ROPE + dr_offs[None, :]
        tl.atomic_add(dkr_ptr + dkr_offs, dkr_acc, mask=tok_valid[:, None])

    # ──────────────────────────────────────────────────────────────
    # Stage B2: Accumulate dv across head chunks.
    # Separate from B1 to reduce peak UB (dv_acc and dk_acc don't
    # coexist).
    # ──────────────────────────────────────────────────────────────
    for blk_start in range(0, topK, BLOCK_K):
        blk_offs = blk_start + blk_k_offs
        blk_in_count = blk_offs < topK
        tok = tl.load(sparse_ptr + sp_base + blk_offs, mask=blk_in_count, other=-1)

        tok_f = tok.to(tl.float32)
        tok_valid = (blk_in_count
                     & (tok != -1)
                     & (tok_f < upper_f)
                     & row_active)
        tok_clamped = tl.where(tok_valid, tok, 0)

        k_full = tl.load(
            k_ptr + k_base + tok_clamped[:, None] * D + d_offs[None, :],
            mask=tok_valid[:, None], other=0.0)
        kr_full = tl.load(
            kr_ptr + kr_base + tok_clamped[:, None] * D_ROPE + dr_offs[None, :],
            mask=tok_valid[:, None], other=0.0)

        dv_acc = tl.zeros([BLOCK_K, D], dtype=tl.float32)

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

            sm_max = tl.load(sm_max_ptr + pid_bs1 * N1 + g_offs_s, mask=g_valid, other=0.0)
            sm_sum = tl.load(sm_sum_ptr + pid_bs1 * N1 + g_offs_s, mask=g_valid, other=1.0)
            sm_sum = tl.where(sm_sum > 0.0, sm_sum, 1.0)

            scores = tl.dot(q_nope, tl.trans(k_full)).to(tl.float32)
            scores += tl.dot(q_rope, tl.trans(kr_full)).to(tl.float32)
            scores = scores * scale_value

            P = tl.exp(scores - sm_max[:, None]) / sm_sum[:, None]

            dv_acc += tl.dot(tl.trans(P).to(do_tile.dtype), do_tile).to(tl.float32)

        # Single atomic_add for dv
        dv_offs = v_base + tok_clamped[:, None] * D + d_offs[None, :]
        tl.atomic_add(dv_ptr + dv_offs, dv_acc, mask=tok_valid[:, None])


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
    # Fixed block config + single launch (NO autotune): autotune re-runs the
    # kernel many times to benchmark, which double-counts the atomic_add scatter
    # into dk/dkr/dv. See _select_block_config.
    cfg = _select_block_config(D, N1)
    block_g = cfg["BLOCK_G"]
    num_hc = triton.cdiv(N1, block_g)  # number of head chunks
    need_clamp = (N1 % block_g) != 0   # need g_offs clamping only when N1 not multiple of BLOCK_G

    # 1D grid, one program per query row
    grid = (_next_pow2(B_S1),)

    _sfa_grad_kernel[grid](
        q_flat, qr_flat,
        k_flat, kr_flat, v_flat,
        sparse_flat,
        do_flat, o_flat,
        sm_max_flat, sm_sum_flat,
        dq_buf, dqr_buf,
        dk_buf, dkr_buf, dv_buf,
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
    return dq_buf, dqr_buf, dk_buf, dkr_buf, dv_buf
