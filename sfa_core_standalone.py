"""Standalone torch-flavored SFA core op (extracted from sparse_flash_attention_triton).

Contains only the triton kernel chain (_sfa_scores_block, _sfa_kernel) and the
_sfa_core launcher. No mindspore dependency: buffers are passed in pre-allocated
by the caller (torch.zeros on device). Type annotations are lazy (PEP 563) so
the module loads even without torch installed; torch is only needed at call time.
"""
from __future__ import annotations

import triton
import triton.language as tl
import triton.backends.ascend.runtime

_D_ROPE = 64
_GRID_CAP = 1024


def _next_pow2(x):
    # Ascend 要求 kernel grid 每维都是 2 的幂, 否则分核映射出错 -> aicore trap。
    # padding 多出的 program 在 kernel 里靠 in_range 掩码空转。
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _prune_configs(configs, named_args, **kwargs):
    """autotune config 过滤 (UB 上限 + grid pow2 + grid 总数上限)。

    Phased UB estimation: the chunked path has two non-overlapping phases
    per chunk — (1) score: Q@K^T, (2) P@V + accumulation. Buffers from
    phase 1 (q_tile, k_tile) are freed before phase 2 starts, so the peak
    UB is max(phase1, phase2), not the sum.
    """
    _UB_LIMIT_BYTES = 180 * 1024
    _GRID_LIMIT = 131072

    def _get(name):
        if name in named_args:
            return named_args[name]
        return kwargs.get(name, None)

    N1 = _get("N1")
    BS1 = _get("B_S1")
    D = _get("D")
    # Conservative multiplier for D<=128: the compiler may keep all tiles
    # (nope + rope + scores) alive simultaneously across d-loops, and may
    # retain intermediate dot products. D>=256 uses 1.0 (verified safe).
    ub_multiplier = 2.5 if (D is not None and D <= 128) else 1.0

    def _estimate_ub_bytes(block_g, block_k, block_d, block_dv):
        if None in (block_g, block_k, block_d, block_dv):
            return 0
        # Phase 1 (score computation): the compiler may keep all tiles alive
        # simultaneously across nope/rope d-loops (q, k, qr, kr, scores, m/l).
        # Critical for D<=128 where BLOCK_D covers both D and D_ROPE=64 in one
        # iteration, making all tiles peak-resident at the same time.
        rope_d = min(block_d, _D_ROPE)
        q_tile = block_g * block_d * 2
        k_tile = block_k * block_d * 2
        qr_tile = block_g * rope_d * 2
        kr_tile = block_k * rope_d * 2
        s_tile = block_g * block_k * 4
        m_l = block_g * 2 * 4
        phase1 = q_tile + k_tile + qr_tile + kr_tile + s_tile + m_l

        # Phase 2 (P@V + accumulation): p_raw + v_tile + pv_tile + acc_dv
        p_tile = block_g * block_k * 4  # p_raw in fp32 (kept across dv-tiles)
        v_tile = block_k * block_dv * 2
        pv_tile = block_g * block_dv * 4
        acc_dv = block_g * block_dv * 4
        phase2 = p_tile + v_tile + pv_tile + acc_dv + m_l

        total = max(phase1, phase2)
        return int(total * ub_multiplier)

    kept = []
    for c in configs:
        bg = c.kwargs.get("BLOCK_G")
        bk = c.kwargs.get("BLOCK_K")
        bd = c.kwargs.get("BLOCK_D")
        bdv = c.kwargs.get("BLOCK_DV")

        if _estimate_ub_bytes(bg, bk, bd, bdv) > _UB_LIMIT_BYTES:
            continue
        # index_select_simd has no mask; BLOCK_D/DV > D causes out-of-bounds MPU access.
        if bd is not None and D is not None and bd > D:
            continue
        if bdv is not None and D is not None and bdv > D:
            continue
        # Hard limit: BLOCK_G > 16 causes UB overflow for D=128 regardless of
        # multiplier (verified: BG=32/64 crash, BG=16/8 safe across all shapes).
        if D is not None and D <= 128 and bg is not None and bg > 16:
            continue
        # NB: BLOCK_G may exceed N1; the kernel masks padded heads (g_valid),
        # so we do NOT prune on bg > N1 (would kill all configs for small N1).
        if None not in (BS1, N1) and bg:
            grid_g = _next_pow2((N1 + bg - 1) // bg)
            total_work = BS1 * grid_g
            if total_work > _GRID_LIMIT:
                continue
        kept.append(c)

    if not kept:
        print('Warning: all autotune params pruned')
        kept = [min(configs, key=lambda c: _estimate_ub_bytes(
            c.kwargs.get("BLOCK_G"), c.kwargs.get("BLOCK_K"),
            c.kwargs.get("BLOCK_D"), c.kwargs.get("BLOCK_DV")))]
    return kept


@triton.jit
def _sfa_scores_block(
    q_ptr, q_base, qr_ptr, qr_base,
    k_ptr, k_base, kr_ptr, kr_base,
    tok_clamped, tok_valid, g_offs, g_valid,
    scale_value: tl.constexpr, D: tl.constexpr, D_ROPE: tl.constexpr,
    BLOCK_G: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """scores[BLOCK_G, BLOCK_K] = (q_nope·k_nope + q_rope·k_rope) * scale.

    Shared by both passes; recomputed (not cached) so the kernel keeps no large
    resident buffer. Invalid gathered tokens are masked to -inf.
    """
    scores = tl.zeros([BLOCK_G, BLOCK_K], dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_valid = d_offs < D
        q_tile = tl.load(
            q_ptr + q_base + g_offs[:, None] * D + d_offs[None, :],
            mask=g_valid[:, None] & d_valid[None, :], other=0.0,
            care_padding=False)
        k_tile = tl.load(
            k_ptr + k_base + tok_clamped[:, None] * D + d_offs[None, :],
            mask=tok_valid[:, None] & d_valid[None, :], other=0.0,
            care_padding=False)
        scores += tl.dot(q_tile, tl.trans(k_tile))
    for d_start in range(0, D_ROPE, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_valid = d_offs < D_ROPE
        qr_tile = tl.load(
            qr_ptr + qr_base + g_offs[:, None] * D_ROPE + d_offs[None, :],
            mask=g_valid[:, None] & d_valid[None, :], other=0.0,
            care_padding=False)
        kr_tile = tl.load(
            kr_ptr + kr_base + tok_clamped[:, None] * D_ROPE + d_offs[None, :],
            mask=tok_valid[:, None] & d_valid[None, :], other=0.0,
            care_padding=False)
        scores += tl.dot(qr_tile, tl.trans(kr_tile))
    scores = scores * scale_value
    return tl.where(tok_valid[None, :], scores, float('-inf'))


@triton.autotune(
    configs=[
        # Original configs (BLOCK_G=16)
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 64, "BLOCK_D": 128, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 64, "BLOCK_D": 64,  "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 32, "BLOCK_D": 128, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 128, "BLOCK_D": 64,  "BLOCK_DV": 64}),
        # BLOCK_G=8: lower UB pressure, finer grid for small N1
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 64, "BLOCK_D": 128, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 128, "BLOCK_D": 64,  "BLOCK_DV": 64}),
        # BLOCK_G=32: higher per-core work density for large N1
        triton.Config({"BLOCK_G": 32, "BLOCK_K": 32, "BLOCK_D": 128, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 32, "BLOCK_K": 64, "BLOCK_D": 64,  "BLOCK_DV": 64}),
        # BLOCK_G=64: one program covers all 64 heads, minimizing grid1
        triton.Config({"BLOCK_G": 64, "BLOCK_K": 32, "BLOCK_D": 128, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 64, "BLOCK_K": 64, "BLOCK_D": 64,  "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 64, "BLOCK_K": 64, "BLOCK_D": 128, "BLOCK_DV": 64}),
        # BLOCK_G=64 with larger BLOCK_K to halve chunked loop count (32 -> 16
        # chunks at topK=2048). Larger BLOCK_K also makes the K_nope gather
        # better-aligned with index_select_simd's preferred chunk size.
        triton.Config({"BLOCK_G": 64, "BLOCK_K": 128, "BLOCK_D": 128, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 64, "BLOCK_K": 128, "BLOCK_D": 64,  "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 64, "BLOCK_K": 256, "BLOCK_D": 64,  "BLOCK_DV": 64}),
        # Wider BLOCK_K ranges
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 256, "BLOCK_D": 64,  "BLOCK_DV": 64}),
        # Larger BLOCK_DV: fewer dv-tile iterations, fewer fp32_acc GM round-trips
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 64, "BLOCK_D": 64,  "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 128, "BLOCK_D": 64,  "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 64, "BLOCK_D": 64,  "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 64, "BLOCK_D": 64,  "BLOCK_DV": 256}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 64, "BLOCK_D": 64,  "BLOCK_DV": 256}),
        # Larger BLOCK_D=256: fewer nope d-tile iterations (2 vs 4 for D=512)
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 64, "BLOCK_D": 256, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 128, "BLOCK_D": 256, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 64, "BLOCK_D": 256, "BLOCK_DV": 64}),
        # Large BLOCK_DV + BLOCK_D combos: minimize both dv and d iterations
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 64, "BLOCK_D": 128, "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 64, "BLOCK_D": 256, "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 64, "BLOCK_D": 128, "BLOCK_DV": 256}),
        # BLOCK_DV=512 (full D=512 in one dv tile, no inner dv loop)
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 64, "BLOCK_D": 128, "BLOCK_DV": 512}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 32, "BLOCK_D": 128, "BLOCK_DV": 512}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 32, "BLOCK_D": 256, "BLOCK_DV": 512}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 32, "BLOCK_D": 128, "BLOCK_DV": 512}),
        # Previously pruned by sum-based UB estimator; now allowed by phased estimator.
        # BK=512+BD=128: 1 nope d-tile, 4 chunks (vs 16 for BK=128)
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 512, "BLOCK_D": 128, "BLOCK_DV": 64}),
        # BK=256+BD=256: 2 nope d-tiles, 8 chunks; high cube utilization
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 256, "BLOCK_D": 256, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 256, "BLOCK_D": 256, "BLOCK_DV": 128}),
        # BG=64 + BK=256 + BD=128: covers all 64 heads, larger tiles
        triton.Config({"BLOCK_G": 64, "BLOCK_K": 256, "BLOCK_D": 128, "BLOCK_DV": 64}),
        # BG=64 + BK=128 + BD=128 + BDV=128: full-head with wider dv tiles
        triton.Config({"BLOCK_G": 64, "BLOCK_K": 128, "BLOCK_D": 128, "BLOCK_DV": 128}),
        # BG=64 full-head + large BDV=256: halve dv-tile iterations (D=512: 2 vs 4)
        triton.Config({"BLOCK_G": 64, "BLOCK_K": 64,  "BLOCK_D": 128, "BLOCK_DV": 256}),
        triton.Config({"BLOCK_G": 64, "BLOCK_K": 64,  "BLOCK_D": 64,  "BLOCK_DV": 256}),
        triton.Config({"BLOCK_G": 64, "BLOCK_K": 128, "BLOCK_D": 64,  "BLOCK_DV": 128}),
        # BK=256 + BDV=128: fewer dv-tile iterations
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 256, "BLOCK_D": 128, "BLOCK_DV": 128}),
        # More configs unlocked by phased UB estimator
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 512, "BLOCK_D": 64,  "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 32, "BLOCK_K": 256, "BLOCK_D": 128, "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 256, "BLOCK_D": 128, "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 128, "BLOCK_D": 128, "BLOCK_DV": 256}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 128, "BLOCK_D": 256, "BLOCK_DV": 256}),
        # More configs unlocked by ub_multiplier=1.1
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 256, "BLOCK_D": 64,  "BLOCK_DV": 256}),
        # Extreme configs unlocked by ub_multiplier=1.0
        triton.Config({"BLOCK_G": 32, "BLOCK_K": 512, "BLOCK_D": 64,  "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 512, "BLOCK_D": 64,  "BLOCK_DV": 128}),
        # Large BLOCK_DV=256/512 configs: minimize dv-tile iterations
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 128, "BLOCK_D": 64,  "BLOCK_DV": 256}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 128, "BLOCK_D": 128, "BLOCK_DV": 256}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 128, "BLOCK_D": 64,  "BLOCK_DV": 512}),
        # BLOCK_D=512: single nope d-tile iteration for D=512, eliminating loop overhead
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 128, "BLOCK_D": 512, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 128, "BLOCK_D": 512, "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 64,  "BLOCK_D": 512, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 64,  "BLOCK_D": 512, "BLOCK_DV": 256}),
    ],
    key=["B_S1", "N1", "S2", "topK", "D", "D_ROPE"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _sfa_kernel(
    q_ptr, qr_ptr,                       # query[B,S1,N1,D], query_rope[B,S1,N1,Dr]
    k_ptr, kr_ptr, v_ptr,                # key/key_rope/value, all [B,S2,1,*]
    sparse_ptr,                          # token indices [B,S1,1,topK] int32 (block-wise pre-expanded on host)
    out_ptr, sm_max_ptr, sm_sum_ptr,     # outputs
    fp32_acc_ptr,                        # fp32 accumulator [B,S1,N1,D] for chunked path
    act_q_ptr, act_k_ptr,
    S2, N1, topK,
    B_S1: tl.constexpr, S1: tl.constexpr,
    D: tl.constexpr, D_ROPE: tl.constexpr,
    scale_value,
    sparse_mode: tl.constexpr,
    return_lse: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    GRID_CAP: tl.constexpr,
):
    """Flash attention over sparsely gathered KV (BSND, MQA / N2=1).

    Grid: (_next_pow2(B*S1), _next_pow2(cdiv(N1, BLOCK_G))), both pow2-padded.
    Each program: one (b,s1) position, BLOCK_G query heads. Inline-gathers KV
    rows by sparse token indices.

    SINGLE_BLOCK (topK fits one BLOCK_TOPK block): scores/P computed ONCE and kept
        resident, then dv-tiled P@V. Score/gather recompute is O(1), not
        O(dv_tiles*k_blocks) as in the chunked fallback below. Used for topK<=128.
    else (chunked online-softmax, large topK): single pass over KV chunks with
        per-chunk correction of fp32 global accumulator (fp32_acc_ptr). Per chunk:
        compute scores, p_raw = exp(scores - m_chunk), then dv-tiled P@V with
        alpha_old/alpha_new correction applied via load-modify-store on fp32_acc.
        Eliminates two-pass score recompute, reduces dots ~69%.

    sparse_ptr holds token positions directly; block-wise (sparse_block_size>1)
    is pre-expanded on host into per-token indices, so this kernel is token-wise.
    """
    # Alignment hints: tensors are contiguous and start at least 128-byte aligned,
    # so vector loads/stores can use aligned addressing (vector_core_partition.md).
    q_ptr = tl.multiple_of(q_ptr, 128)
    qr_ptr = tl.multiple_of(qr_ptr, 128)
    k_ptr = tl.multiple_of(k_ptr, 128)
    kr_ptr = tl.multiple_of(kr_ptr, 128)
    v_ptr = tl.multiple_of(v_ptr, 128)
    out_ptr = tl.multiple_of(out_ptr, 128)
    fp32_acc_ptr = tl.multiple_of(fp32_acc_ptr, 128)
    sparse_ptr = tl.multiple_of(sparse_ptr, 128)
    sm_max_ptr = tl.multiple_of(sm_max_ptr, 128)
    sm_sum_ptr = tl.multiple_of(sm_sum_ptr, 128)

    pid_flat = tl.program_id(0)
    grid_g = _next_pow2((N1 + BLOCK_G - 1) // BLOCK_G)
    total_work = B_S1 * grid_g
    grid_size = _next_pow2(min(total_work, GRID_CAP))

    for work_id in range(pid_flat, total_work, grid_size):
        pid_bs1 = work_id // grid_g
        pid_g = work_id % grid_g

        bs1_in_range = pid_bs1 < B_S1
        pid_bs1 = tl.where(bs1_in_range, pid_bs1, 0)

        b = pid_bs1 // S1
        s1 = pid_bs1 % S1

        g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        g_valid = g_offs < N1

        act_q = tl.load(act_q_ptr + b)
        act_k = tl.load(act_k_ptr + b)

        # causal window upper bound (token threshold)
        if sparse_mode == 0:
            threshold = act_k
        else:
            threshold = act_k - act_q + s1 + 1

        # rightDownCausal: leading rows (query longer than key) are fully hidden.
        row_active = bs1_in_range & (s1 < act_q) & (threshold > 0)

        # base offsets (memory layout: q[B,S1,N1,D], k/v[B,S2,1,D], rope analogous)
        q_base = (b * S1 + s1) * N1 * D
        qr_base = (b * S1 + s1) * N1 * D_ROPE
        k_base = b * S2 * D
        kr_base = b * S2 * D_ROPE
        v_base = b * S2 * D
        sp_base = (b * S1 + s1) * topK

        if SINGLE_BLOCK:
            # ---- fast path: one block covers the whole topK window; scores/P computed
            # ONCE and kept resident, then dv-tiled P@V. Score/gather recompute is O(1),
            # not O(dv_tiles*k_blocks) as in the two-pass fallback below. ----
            blk_offs = tl.arange(0, BLOCK_TOPK)
            blk_in_count = blk_offs < topK
            tok = tl.load(sparse_ptr + sp_base + blk_offs, mask=blk_in_count, other=-1)
            tok_valid = blk_in_count & (tok != -1) & (tok < threshold) & (tok < act_k) & row_active
            tok_clamped = blk_offs

            scores = _sfa_scores_block(
                q_ptr, q_base, qr_ptr, qr_base,
                k_ptr, k_base, kr_ptr, kr_base,
                tok_clamped, tok_valid, g_offs, g_valid,
                scale_value, D, D_ROPE, BLOCK_G, BLOCK_TOPK, BLOCK_D)

            m_i = tl.max(scores, axis=1)
            m_safe = tl.where(m_i == float('-inf'), 0.0, m_i)
            p = tl.exp(scores - m_safe[:, None])
            p = tl.where(tok_valid[None, :], p, 0.0)
            l_i = tl.sum(p, axis=1)
            l_safe = tl.where(l_i > 0.0, l_i, 1.0)

            out_base = (b * S1 + s1) * N1 * D
            p_norm = p / l_safe[:, None]
            for dv_start in range(0, D, BLOCK_DV):
                dv_offs = dv_start + tl.arange(0, BLOCK_DV)
                dv_valid = dv_offs < D
                v_tile = tl.load(
                    v_ptr + v_base + tok_clamped[:, None] * D + dv_offs[None, :],
                    mask=tok_valid[:, None] & dv_valid[None, :], other=0.0)
                out_tile = tl.dot(p_norm.to(v_tile.dtype), v_tile)
                tl.store(
                    out_ptr + out_base + g_offs[:, None] * D + dv_offs[None, :],
                    out_tile.to(out_ptr.dtype.element_ty),
                    mask=g_valid[:, None] & dv_valid[None, :] & row_active)

            if return_lse:
                sm_base = (b * S1 + s1) * N1
                store_mask = g_valid & row_active & (l_i > 0.0)
                tl.store(sm_max_ptr + sm_base + g_offs, m_i, mask=store_mask)
                tl.store(sm_sum_ptr + sm_base + g_offs, l_i, mask=store_mask)
        else:
            # ---- chunked online-softmax: one pass over KV chunks, per-chunk
            # correction of fp32 global accumulator. Eliminates two-pass score
            # recompute, reduces dots ~69% (928 -> 288 for the profiled shape).
            # Score computation inlined (was _sfa_scores_block) so the compiler
            # can see Q/QR loads are loop-invariant across blk_start and
            # potentially reuse/cache them (loop-invariant-hoisting.md). ----
            m_i = tl.full([BLOCK_G], float('-inf'), dtype=tl.float32)
            l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
            fp32_base = (b * S1 + s1) * N1 * D

            for blk_start in range(0, topK - BLOCK_K, BLOCK_K):
                blk_offs = blk_start + tl.arange(0, BLOCK_K)
                blk_in_count = blk_offs < topK
                tok = tl.load(sparse_ptr + sp_base + blk_offs, mask=blk_in_count, other=-1)
                tok_valid = blk_in_count & (tok != -1) & (tok < threshold) & (tok < act_k) & row_active
                tok_clamped = blk_offs

                # Inline score computation (inlined from _sfa_scores_block) so the
                # compiler sees the full loop structure and can detect that Q/QR
                # loads are loop-invariant across blk_start iterations.
                # Load order: Q before K (Q has no dep on tok_clamped, can overlap
                # with prev iter's fp32_acc store per load-order.md).
                scores = tl.zeros([BLOCK_G, BLOCK_K], dtype=tl.float32)
                for d_start in range(0, D, BLOCK_D):
                    d_offs = d_start + tl.arange(0, BLOCK_D)
                    d_valid = d_offs < D
                    q_tile = tl.load(
                        q_ptr + q_base + g_offs[:, None] * D + d_offs[None, :],
                        mask=g_valid[:, None] & d_valid[None, :], other=0.0)
                    k_tile = tl.load(
                        k_ptr + k_base + tok_clamped[:, None] * D + d_offs[None, :],
                        mask=tok_valid[:, None] & d_valid[None, :], other=0.0)
                    scores += tl.dot(q_tile, tl.trans(k_tile))
                for d_start in range(0, D_ROPE, BLOCK_D):
                    d_offs = d_start + tl.arange(0, BLOCK_D)
                    d_valid = d_offs < D_ROPE
                    qr_tile = tl.load(
                        qr_ptr + qr_base + g_offs[:, None] * D_ROPE + d_offs[None, :],
                        mask=g_valid[:, None] & d_valid[None, :], other=0.0)
                    kr_tile = tl.load(
                        kr_ptr + kr_base + tok_clamped[:, None] * D_ROPE + d_offs[None, :],
                        mask=tok_valid[:, None] & d_valid[None, :], other=0.0)
                    scores += tl.dot(qr_tile, tl.trans(kr_tile))
                scores = scores * scale_value
                scores = tl.where(tok_valid[None, :], scores, float('-inf'))

                m_blk = tl.max(scores, axis=1)
                m_new = tl.maximum(m_i, m_blk)
                m_new_safe = tl.where(m_new == float('-inf'), 0.0, m_new)
                alpha_old = tl.exp(m_i - m_new_safe)
                alpha_new = tl.exp(m_blk - m_new_safe)

                m_blk_safe = tl.where(m_blk == float('-inf'), 0.0, m_blk)
                p_raw = tl.exp(scores - m_blk_safe[:, None])
                # p_raw = tl.where(tok_valid[None, :], p_raw, 0.0)
                l_chunk = tl.sum(p_raw, axis=1)
                l_i = l_i * alpha_old + l_chunk * alpha_new

                for dv_start in range(0, D, BLOCK_DV):
                    dv_offs = dv_start + tl.arange(0, BLOCK_DV)
                    dv_valid = dv_offs < D
                    # Load fp32_acc before V so the independent load can overlap
                    # with the previous iteration's fp32_acc store (load-order.md).
                    acc_dv = tl.load(
                        fp32_acc_ptr + fp32_base + g_offs[:, None] * D + dv_offs[None, :],
                        mask=g_valid[:, None] & dv_valid[None, :], other=0.0)
                    v_tile = tl.load(
                        v_ptr + v_base + tok_clamped[:, None] * D + dv_offs[None, :],
                        mask=tok_valid[:, None] & dv_valid[None, :], other=0.0)
                    pv_tile = tl.dot(p_raw.to(v_tile.dtype), v_tile) * alpha_new[:, None]
                    acc_dv = acc_dv * alpha_old[:, None] + pv_tile
                    tl.store(
                        fp32_acc_ptr + fp32_base + g_offs[:, None] * D + dv_offs[None, :],
                        acc_dv,
                        mask=g_valid[:, None] & dv_valid[None, :] & row_active)

                m_i = m_new

            # Last k-block: compute scores/softmax as above, but fuse the
            # fp32_acc normalization (divide by l_safe) and write directly to
            # out_ptr.  Eliminates the separate post-loop dv-tile pass that
            # reads fp32_acc back from GM (discrete_memory_access.md: eliminate
            # redundant GM round-trips).
            last_blk_start = topK - BLOCK_K
            blk_offs = last_blk_start + tl.arange(0, BLOCK_K)
            blk_in_count = blk_offs < topK
            tok = tl.load(sparse_ptr + sp_base + blk_offs, mask=blk_in_count, other=-1)
            tok_valid = blk_in_count & (tok != -1) & (tok < threshold) & (tok < act_k) & row_active
            tok_clamped = blk_offs

            scores = tl.zeros([BLOCK_G, BLOCK_K], dtype=tl.float32)
            for d_start in range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                q_tile = tl.load(
                    q_ptr + q_base + g_offs[:, None] * D + d_offs[None, :],
                    mask=g_valid[:, None] & d_valid[None, :], other=0.0)
                k_tile = tl.load(
                    k_ptr + k_base + tok_clamped[:, None] * D + d_offs[None, :],
                    mask=tok_valid[:, None] & d_valid[None, :], other=0.0)
                scores += tl.dot(q_tile, tl.trans(k_tile))
            for d_start in range(0, D_ROPE, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D_ROPE
                qr_tile = tl.load(
                    qr_ptr + qr_base + g_offs[:, None] * D_ROPE + d_offs[None, :],
                    mask=g_valid[:, None] & d_valid[None, :], other=0.0)
                kr_tile = tl.load(
                    kr_ptr + kr_base + tok_clamped[:, None] * D_ROPE + d_offs[None, :],
                    mask=tok_valid[:, None] & d_valid[None, :], other=0.0)
                scores += tl.dot(qr_tile, tl.trans(kr_tile))
            scores = scores * scale_value
            scores = tl.where(tok_valid[None, :], scores, float('-inf'))

            m_blk = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_blk)
            m_new_safe = tl.where(m_new == float('-inf'), 0.0, m_new)
            alpha_old = tl.exp(m_i - m_new_safe)
            alpha_new = tl.exp(m_blk - m_new_safe)

            m_blk_safe = tl.where(m_blk == float('-inf'), 0.0, m_blk)
            p_raw = tl.exp(scores - m_blk_safe[:, None])
            p_raw = tl.where(tok_valid[None, :], p_raw, 0.0)
            l_chunk = tl.sum(p_raw, axis=1)
            l_i = l_i * alpha_old + l_chunk * alpha_new
            l_safe = tl.where(l_i > 0.0, l_i, 1.0)

            out_base = (b * S1 + s1) * N1 * D
            for dv_start in range(0, D, BLOCK_DV):
                dv_offs = dv_start + tl.arange(0, BLOCK_DV)
                dv_valid = dv_offs < D
                # Load fp32_acc before V to overlap with previous store.
                acc_dv = tl.load(
                    fp32_acc_ptr + fp32_base + g_offs[:, None] * D + dv_offs[None, :],
                    mask=g_valid[:, None] & dv_valid[None, :], other=0.0)
                v_tile = tl.load(
                    v_ptr + v_base + tok_clamped[:, None] * D + dv_offs[None, :],
                    mask=tok_valid[:, None] & dv_valid[None, :], other=0.0)
                pv_tile = tl.dot(p_raw.to(v_tile.dtype), v_tile) * alpha_new[:, None]
                # Fuse normalization: write directly to out_ptr instead of
                # fp32_acc_ptr, saving a full dv-tile GM read+write pass.
                out_tile = (acc_dv * alpha_old[:, None] + pv_tile) / l_safe[:, None]
                tl.store(
                    out_ptr + out_base + g_offs[:, None] * D + dv_offs[None, :],
                    out_tile.to(out_ptr.dtype.element_ty),
                    mask=g_valid[:, None] & dv_valid[None, :] & row_active)

            m_i = m_new

            if return_lse:
                sm_base = (b * S1 + s1) * N1
                store_mask = g_valid & row_active & (l_i > 0.0)
                tl.store(sm_max_ptr + sm_base + g_offs, m_i, mask=store_mask)
                tl.store(sm_sum_ptr + sm_base + g_offs, l_i, mask=store_mask)


def _sfa_core(
    q_flat: torch.Tensor, qr_flat: torch.Tensor,
    k_flat: torch.Tensor, kr_flat: torch.Tensor, v_flat: torch.Tensor,
    sparse_flat: torch.Tensor,
    out_buf: torch.Tensor, sm_max_buf: torch.Tensor, sm_sum_buf: torch.Tensor,
    fp32_acc_buf: torch.Tensor,
    act_q: torch.Tensor, act_k: torch.Tensor,
    B_S1: int, S1: int, S2: int, N1: int, topK: int,
    D: int, D_ROPE: int,
    scale_value: float,
    sparse_mode: int,
    return_lse: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1D flat grid: flatten (bs1, g) into a single dim, cap at _GRID_CAP to
    # match NPU core count (32 cores × 32 programs/core = 1024). Each program
    # loops over multiple work items when total_work > grid_size.
    def grid_fn(meta):
        grid_g = _next_pow2(triton.cdiv(N1, meta["BLOCK_G"]))
        total_work = B_S1 * grid_g
        return (_next_pow2(min(total_work, _GRID_CAP)),)

    block_topk = _next_pow2(topK)
    single_block = block_topk <= 128

    _sfa_kernel[grid_fn](
        q_flat, qr_flat,
        k_flat, kr_flat, v_flat,
        sparse_flat,
        out_buf, sm_max_buf, sm_sum_buf,
        fp32_acc_buf,
        act_q, act_k,
        S2, N1, topK,
        B_S1=B_S1, S1=S1,
        D=D, D_ROPE=D_ROPE,
        scale_value=scale_value,
        sparse_mode=sparse_mode,
        return_lse=return_lse,
        SINGLE_BLOCK=single_block,
        BLOCK_TOPK=block_topk,
        GRID_CAP=_GRID_CAP,
    )
    return out_buf, sm_max_buf, sm_sum_buf
