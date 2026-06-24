"""Triton-ascend implementation of sparse_flash_attention (SFA).

Interface aligned with MindSpore ops.sparse_flash_attention (drop-in replacement
for the call site in mindformers .../transformer/dsa/dsa_attention.py).

Computes  softmax(Q @ K^T / sqrt(d)) @ V  over a sparsely gathered subset of KV,
in MLA-absorb mode (attention_mode=2):
  q_full = concat(query[..,:D], query_rope[..,:Dr])     (D in 128/256/512, Dr=64)
  k_full = concat(key[..,:D],   key_rope[..,:Dr])
  value  = key[..,:D]  (K and V share the compressed latent c_kv; the passed
           `value` tensor is IGNORED, matching CANN's attention_mode=2 contract)

MQA only (N2=1): the N1 query heads share one gathered KV per (b,s1).

Layout support:
  - layout_query: BSND / TND
  - layout_kv:    BSND / TND / PA_BSND (paged KV via block_table)
TND and PA_BSND are normalized to BSND on host (PyNative only); for GRAPH_MODE
the caller should pass BSND directly (mirrors lightning_indexer_triton).

sparse_mode 0 (full) and 3 (rightDownCausal) supported.
sparse_block_size 1 (token-wise) and 2^n in [1,128] (block-wise) supported.
"""
import triton
import triton.language as tl
import triton.backends.ascend.runtime


import mindspore as ms
from mindspore import ops

INT64_MAX = 9223372036854775807

# CANN constraints. attention_mode=2 (MLA-absorb) fixes Dr=64; D (kv_lora_rank,
# the nope latent dim) is generalized here to 128/256/512 — CANN itself only
# does 512, so D!=512 has no CANN reference (verify against the numpy golden).
_VALID_D = (128, 256, 512)
_D_ROPE = 64
_VALID_N1 = (1, 2, 4, 8, 16, 32, 64, 128)


def _next_pow2(x):
    # Ascend 要求 kernel grid 每维都是 2 的幂, 否则分核映射出错 -> aicore trap。
    # padding 多出的 program 在 kernel 里靠 in_range 掩码空转。
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _patch_triton_ascend_mindspore_dtype_bytes():
    """修补 triton-ascend autotuner 的 dtype 字节数查询 (见 lightning_indexer_triton)。"""
    try:
        from triton.backends.ascend.runtime import utils as ascend_utils
        from triton.backends.ascend.runtime import autotuner as ascend_autotuner
    except ImportError:
        return

    origin_func = getattr(ascend_utils, "get_byte_per_numel", None)
    if origin_func is None:
        return
    if getattr(origin_func, "_mindspore_dtype_patched", False):
        if hasattr(ascend_autotuner, "get_byte_per_numel"):
            ascend_autotuner.get_byte_per_numel = origin_func
        return

    dtype_bytes = {}
    for byte_size, names in (
        (1, ("int8", "uint8", "bool_")),
        (2, ("float16", "bfloat16", "int16", "uint16")),
        (4, ("float32", "int32", "uint32")),
        (8, ("float64", "int64", "uint64")),
    ):
        for name in names:
            dt = getattr(ms, name, None)
            if dt is not None:
                dtype_bytes[dt] = byte_size

    def patched_get_byte_per_numel(dtype):
        try:
            if dtype in dtype_bytes:
                return dtype_bytes[dtype]
        except TypeError:
            pass
        return origin_func(dtype)

    patched_get_byte_per_numel._mindspore_dtype_patched = True
    ascend_utils.get_byte_per_numel = patched_get_byte_per_numel
    if hasattr(ascend_autotuner, "get_byte_per_numel"):
        ascend_autotuner.get_byte_per_numel = patched_get_byte_per_numel


_patch_triton_ascend_mindspore_dtype_bytes()


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
            grid0 = _next_pow2(BS1)
            grid1 = _next_pow2((N1 + bg - 1) // bg)
            if grid0 * grid1 > _GRID_LIMIT:
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
    blk_offs, tok_valid, g_offs, g_valid,
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
            k_ptr + k_base + blk_offs[:, None] * D + d_offs[None, :],
            mask=d_valid[None, :], other=0.0,
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
            kr_ptr + kr_base + blk_offs[:, None] * D_ROPE + d_offs[None, :],
            mask=d_valid[None, :], other=0.0,
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
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 16, "BLOCK_D": 128, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 256, "BLOCK_D": 64,  "BLOCK_DV": 64}),
        # Larger BLOCK_DV: fewer dv-tile iterations, fewer fp32_acc GM round-trips
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 64, "BLOCK_D": 64,  "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 128, "BLOCK_D": 64, "BLOCK_DV": 128}),
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
        # BK=256 + BDV=128: fewer dv-tile iterations
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 256, "BLOCK_D": 128, "BLOCK_DV": 128}),
        # More configs unlocked by phased UB estimator
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 512, "BLOCK_D": 64,  "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 32, "BLOCK_K": 256, "BLOCK_D": 128, "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 16, "BLOCK_K": 256, "BLOCK_D": 128, "BLOCK_DV": 128}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 128, "BLOCK_D": 128, "BLOCK_DV": 256}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 128, "BLOCK_D": 256, "BLOCK_DV": 256}),
        # More configs unlocked by ub_multiplier=1.1
        triton.Config({"BLOCK_G": 64, "BLOCK_K": 256, "BLOCK_D": 128, "BLOCK_DV": 64}),
        triton.Config({"BLOCK_G": 8,  "BLOCK_K": 256, "BLOCK_D": 64,  "BLOCK_DV": 256}),
        triton.Config({"BLOCK_G": 4,  "BLOCK_K": 128, "BLOCK_D": 128, "BLOCK_DV": 512}),
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

    pid_bs1 = tl.program_id(0)
    pid_g = tl.program_id(1)

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

    # base offsets (q/qr: [B,S1,N1,D]; k/kr/v: pre-gathered [B*S1,topK,D])
    q_base = (b * S1 + s1) * N1 * D
    qr_base = (b * S1 + s1) * N1 * D_ROPE
    k_base = pid_bs1 * topK * D
    kr_base = pid_bs1 * topK * D_ROPE
    v_base = pid_bs1 * topK * D
    sp_base = (b * S1 + s1) * topK

    if SINGLE_BLOCK:
        # ---- fast path: one block covers the whole topK window; scores/P computed
        # ONCE and kept resident, then dv-tiled P@V. Score/gather recompute is O(1),
        # not O(dv_tiles*k_blocks) as in the two-pass fallback below. ----
        blk_offs = tl.arange(0, BLOCK_TOPK)
        blk_in_count = blk_offs < topK
        tok = tl.load(sparse_ptr + sp_base + blk_offs, mask=blk_in_count, other=-1)
        tok_valid = blk_in_count & (tok != -1) & (tok < threshold) & (tok < act_k) & row_active
        scores = _sfa_scores_block(
            q_ptr, q_base, qr_ptr, qr_base,
            k_ptr, k_base, kr_ptr, kr_base,
            blk_offs, tok_valid, g_offs, g_valid,
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
                v_ptr + v_base + blk_offs[:, None] * D + dv_offs[None, :],
                mask=dv_valid[None, :], other=0.0)
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
            # Inline score computation (inlined from _sfa_scores_block) so the
            # compiler sees the full loop structure and can detect that Q/QR
            # loads are loop-invariant across blk_start iterations.
            # Load order: Q before K (Q has no dep on blk_offs, can overlap
            # with prev iter's fp32_acc store per load-order.md).
            scores = tl.zeros([BLOCK_G, BLOCK_K], dtype=tl.float32)
            for d_start in range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                q_tile = tl.load(
                    q_ptr + q_base + g_offs[:, None] * D + d_offs[None, :],
                    mask=g_valid[:, None] & d_valid[None, :], other=0.0)
                k_tile = tl.load(
                    k_ptr + k_base + blk_offs[:, None] * D + d_offs[None, :],
                    mask=d_valid[None, :], other=0.0)
                scores += tl.dot(q_tile, tl.trans(k_tile))
            for d_start in range(0, D_ROPE, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D_ROPE
                qr_tile = tl.load(
                    qr_ptr + qr_base + g_offs[:, None] * D_ROPE + d_offs[None, :],
                    mask=g_valid[:, None] & d_valid[None, :], other=0.0)
                kr_tile = tl.load(
                    kr_ptr + kr_base + blk_offs[:, None] * D_ROPE + d_offs[None, :],
                    mask=d_valid[None, :], other=0.0)
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
                    v_ptr + v_base + blk_offs[:, None] * D + dv_offs[None, :],
                    mask=dv_valid[None, :], other=0.0)
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
        scores = tl.zeros([BLOCK_G, BLOCK_K], dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_offs < D
            q_tile = tl.load(
                q_ptr + q_base + g_offs[:, None] * D + d_offs[None, :],
                mask=g_valid[:, None] & d_valid[None, :], other=0.0)
            k_tile = tl.load(
                k_ptr + k_base + blk_offs[:, None] * D + d_offs[None, :],
                mask=d_valid[None, :], other=0.0)
            scores += tl.dot(q_tile, tl.trans(k_tile))
        for d_start in range(0, D_ROPE, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_offs < D_ROPE
            qr_tile = tl.load(
                qr_ptr + qr_base + g_offs[:, None] * D_ROPE + d_offs[None, :],
                mask=g_valid[:, None] & d_valid[None, :], other=0.0)
            kr_tile = tl.load(
                kr_ptr + kr_base + blk_offs[:, None] * D_ROPE + d_offs[None, :],
                mask=d_valid[None, :], other=0.0)
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
                v_ptr + v_base + blk_offs[:, None] * D + dv_offs[None, :],
                mask=dv_valid[None, :], other=0.0)
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

# ---------------------------------------------------------------------------
# host-side helpers
# ---------------------------------------------------------------------------
def _default_actual_seq_lens(actual_seq_lens, batch_size, seq_len):
    # None -> full seq; list/tuple -> int32 tensor; else pass-through.
    return ms.ops.fill(ms.int32, (batch_size,), seq_len) if actual_seq_lens is None else \
           ms.Tensor(list(actual_seq_lens), dtype=ms.int32) if isinstance(actual_seq_lens, (list, tuple)) else \
           actual_seq_lens


def _tnd_cumsum_to_per_batch(cumsum):
    # TND actual_seq_lengths are cumulative prefix sums; diff back to per-batch.
    return cumsum - ops.pad(cumsum[:-1], (1, 0))


def _tnd_to_bsnd(tensor, act_seq_per_batch):
    """[T, N, ...] -> [B, max_S, N, ...]; PyNative-only (data-dependent slicing)."""
    # assert ms.get_context('mode') == ms.PYNATIVE_MODE, "TND path is PyNative-only."
    B = act_seq_per_batch.shape[0]
    lengths = [int(act_seq_per_batch[i].asnumpy().item()) for i in range(B)]
    max_seq = max(lengths) if lengths else 0
    out = ms.ops.zeros((B, max_seq, *tensor.shape[1:]), dtype=tensor.dtype)
    start = 0
    for b_idx in range(B):
        length = lengths[b_idx]
        if length > 0:
            out[b_idx, :length] = tensor[start:start + length]
            start += length
    return out


def _bsnd_to_tnd(tensor, act_seq_per_batch):
    """[B, S, N, ...] -> [T, N, ...]; inverse of _tnd_to_bsnd (PyNative-only)."""
# assert ms.get_context('mode') == ms.PYNATIVE_MODE, "TND path is PyNative-only."
    B = act_seq_per_batch.shape[0]
    lengths = [int(act_seq_per_batch[i].asnumpy().item()) for i in range(B)]
    total_t = sum(lengths)
    out = ms.ops.zeros((total_t, *tensor.shape[2:]), dtype=tensor.dtype)
    start = 0
    for b_idx in range(B):
        length = lengths[b_idx]
        if length > 0:
            out[start:start + length] = tensor[b_idx, :length]
            start += length
    return out


def _pa_to_bsnd(cache, block_table, act_k_per_batch, max_s2):
    """PageAttention cache [block_num, block_size, N, Dx] -> dense [B, max_s2, N, Dx].

    Reverses paging using block_table[B, max_blocks] (PyNative-only). -1 block ids
    are skipped. Mirrors the golden's tensor_to_pa inverse.
    """
    # assert ms.get_context('mode') == ms.PYNATIVE_MODE, "PA_BSND path is PyNative-only."
    block_num, block_size, N, Dx = cache.shape
    B = block_table.shape[0]
    out = ms.ops.zeros((B, max_s2, N, Dx), dtype=cache.dtype)
    bt = block_table.asnumpy()
    for b in range(B):
        for blk_i in range(block_table.shape[1]):
            blk_id = int(bt[b, blk_i])
            if blk_id == -1:
                continue
            dst = blk_i * block_size
            if dst >= max_s2:
                break
            end = min(dst + block_size, max_s2)
            out[b, dst:end] = cache[blk_id, :end - dst]
    return out


def _expand_block_indices(sparse_indices, sparse_block_size):
    """Block ids [.., topK] -> token indices [.., topK*block_size].

    Each block id b expands to tokens [b*bs, b*bs+bs); -1 stays -1. The kernel
    then treats the result as plain token indices (token-wise path). Pure static
    tensor ops, so this works in GRAPH_MODE as well as PyNative.
    """
    if sparse_block_size == 1:
        return sparse_indices
    bs = sparse_block_size
    base = sparse_indices.astype(ms.int32)
    *lead, topK = base.shape
    base = base.reshape(*lead, topK, 1)
    offs = ms.ops.arange(0, bs, dtype=ms.int32).reshape(*([1] * len(lead)), 1, bs)
    tokens = ms.ops.where(base == -1, ms.Tensor(-1, ms.int32), base * bs + offs)
    return tokens.reshape(*lead, topK * bs)


# ---------------------------------------------------------------------------
# _ms_pyfunc core (launches the triton kernel). Type annotations are required by
# _ms_pyfunc shape/dtype inference and must match between infer_func and core.
# ---------------------------------------------------------------------------
def _infer_sfa(
    q_flat: ms.Tensor, qr_flat: ms.Tensor,
    k_flat: ms.Tensor, kr_flat: ms.Tensor, v_flat: ms.Tensor,
    sparse_flat: ms.Tensor,
    out_buf: ms.Tensor, sm_max_buf: ms.Tensor, sm_sum_buf: ms.Tensor,
    fp32_acc_buf: ms.Tensor,
    act_q: ms.Tensor, act_k: ms.Tensor,
    B_S1: int, S1: int, S2: int, N1: int, topK: int,
    D: int, D_ROPE: int,
    scale_value: float,
    sparse_mode: int,
    return_lse: int,
) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
    return (ms.mint.empty_like(out_buf),
            ms.mint.empty_like(sm_max_buf),
            ms.mint.empty_like(sm_sum_buf))

import os
import pickle
from datetime import datetime

def _save_sfa_inputs(
    q_flat, qr_flat, k_flat, kr_flat, v_flat,
    sparse_flat, out_buf, sm_max_buf, sm_sum_buf,
    fp32_acc_buf, act_q, act_k,
    B_S1, S1, S2, N1, topK,
    D, D_ROPE, scale_value, sparse_mode, return_lse,
    single_block, block_topk,
    save_dir="/tmp/sfa_inputs"
):
    """保存 _sfa_kernel 的所有输入到本地文件"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = os.path.join(save_dir, f"sfa_inputs_{timestamp}.pkl")
    
    import torch
    import numpy as np
    import mindspore as ms
    
    def ms_tensor_to_torch(ms_t):
        """将 ms.Tensor 转为 torch.Tensor，兼容 bfloat16"""
        if not hasattr(ms_t, "dtype"):
            return ms_t  # 不是 ms.Tensor
        
        ms_dtype = ms_t.dtype
        shape = ms_t.shape
        
        # bfloat16 特殊处理：通过 uint16 字节视图中转
        if ms_dtype == ms.bfloat16:
            # 方法：先 view 为 uint16，再 asnumpy，再 torch.from_numpy，最后 view 回 bfloat16
            uint16_tensor = ms_t.view(ms.uint16)
            np_uint16 = uint16_tensor.asnumpy()
            torch_uint16 = torch.from_numpy(np_uint16)
            return torch_uint16.view(torch.bfloat16).reshape(shape)
        
        # 其他类型正常转换
        try:
            np_arr = ms_t.asnumpy()
            dtype_map = {
                ms.float16: torch.float16,
                ms.float32: torch.float32,
                ms.float64: torch.float64,
                ms.int8: torch.int8,
                ms.int16: torch.int16,
                ms.int32: torch.int32,
                ms.int64: torch.int64,
                ms.uint8: torch.uint8,
                ms.bool_: torch.bool,
            }
            torch_dtype = dtype_map.get(ms_dtype, torch.float32)
            return torch.from_numpy(np_arr).to(torch_dtype)
        except Exception as e:
            # 兜底：float32 中转
            np_f32 = ms_t.astype(ms.float32).asnumpy()
            target_dtype = dtype_map.get(ms_dtype, torch.float32)
            return torch.from_numpy(np_f32).to(target_dtype)

    inputs = {
        "q_flat":       ms_tensor_to_torch(q_flat),
        "qr_flat":      ms_tensor_to_torch(qr_flat),
        "k_flat":       ms_tensor_to_torch(k_flat),
        "kr_flat":      ms_tensor_to_torch(kr_flat),
        "v_flat":       ms_tensor_to_torch(v_flat),
        "sparse_flat":  ms_tensor_to_torch(sparse_flat),
        "out_buf":      ms_tensor_to_torch(out_buf),
        "sm_max_buf":   ms_tensor_to_torch(sm_max_buf),
        "sm_sum_buf":   ms_tensor_to_torch(sm_sum_buf),
        "fp32_acc_buf": ms_tensor_to_torch(fp32_acc_buf),
        "act_q":        ms_tensor_to_torch(act_q),
        "act_k":        ms_tensor_to_torch(act_k),
        "B_S1": B_S1, "S1": S1, "S2": S2, "N1": N1, "topK": topK,
        "D": D, "D_ROPE": D_ROPE, "scale_value": scale_value,
        "sparse_mode": sparse_mode, "return_lse": return_lse,
        "single_block": single_block, "block_topk": block_topk,
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(inputs, f)
    
    print(f"[SFA] 输入已保存到: {save_path}")
    return save_path

@ms.ops._ms_pyfunc(infer_func=_infer_sfa)
def _sfa_core(
    q_flat: ms.Tensor, qr_flat: ms.Tensor,
    k_flat: ms.Tensor, kr_flat: ms.Tensor, v_flat: ms.Tensor,
    sparse_flat: ms.Tensor,
    out_buf: ms.Tensor, sm_max_buf: ms.Tensor, sm_sum_buf: ms.Tensor,
    fp32_acc_buf: ms.Tensor,
    act_q: ms.Tensor, act_k: ms.Tensor,
    B_S1: int, S1: int, S2: int, N1: int, topK: int,
    D: int, D_ROPE: int,
    scale_value: float,
    sparse_mode: int,
    return_lse: int,
) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
    # grid both dims pow2-padded (Ascend traps on non-pow2 grid); out-of-range
    # programs idle via in_range masks. Padding must match _prune_configs.
    def grid_fn(meta): return (
        _next_pow2(B_S1),
        _next_pow2(triton.cdiv(N1, meta["BLOCK_G"])),
    )

    # fast path when one block (BLOCK_TOPK = pow2(topK), capped at 128) covers the
    # whole sparse window: scores/P computed once, dv-tiled P@V. Larger topK uses
    # chunked online-softmax (fp32 global accumulator, no two-pass recompute).
    block_topk = _next_pow2(topK)
    single_block = block_topk <= 128

    # _save_sfa_inputs(
    #         q_flat, qr_flat, k_flat, kr_flat, v_flat,
    #         sparse_flat, out_buf, sm_max_buf, sm_sum_buf,
    #         fp32_acc_buf, act_q, act_k,
    #         B_S1, S1, S2, N1, topK,
    #         D, D_ROPE, scale_value, sparse_mode, return_lse,
    #         single_block, block_topk,
    #         save_dir="/home/z00841464/SFA/data/sfa_inputs"  # 可修改保存路径
    #     )

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
    )
    return out_buf, sm_max_buf, sm_sum_buf


# ---------------------------------------------------------------------------
# public API — aligned with MindSpore ops.sparse_flash_attention signature
# ---------------------------------------------------------------------------
def sparse_flash_attention_triton(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    sparse_indices: ms.Tensor,
    scale_value: float = 1.0,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sparse_block_size: int = 1,
    layout_query: str = "BSND",
    layout_kv: str = "BSND",
    sparse_mode: int = 0,
    pre_tokens: int = INT64_MAX,
    next_tokens: int = INT64_MAX,
    attention_mode: int = 2,
    return_softmax_lse: bool = False,
    block_size: int = 0,
):
    """Drop-in replacement for ops.sparse_flash_attention (MLA-absorb, MQA).

    Args:
        query: [B,S1,N1,D] BSND or [T1,N1,D] TND, fp16/bf16 (D in 128/256/512)
        key:   [B,S2,1,D] BSND / [T2,1,D] TND / [block_num,block_size,1,D] PA_BSND
        value: same layout/shape as key; IGNORED in MLA-absorb mode (value is
               taken as key[..,:D]). Kept for API parity with CANN.
        sparse_indices: [B,S1,1,sparse_count] (or TND) int32, block ids, -1 invalid
        scale_value: softmax scale (1/sqrt(d_k))
        block_table: [B, max_blocks] int32, required for PA_BSND
        actual_seq_lengths_query/kv: [B] int32 / list / None (cumulative for TND)
        query_rope/key_rope: [.,N,Dr] (Dr=64), required (no empty rope)
        sparse_block_size: 1 (token-wise) or 2^n in [1,128] (block-wise)
        layout_query: "BSND" / "TND"
        layout_kv: "BSND" / "TND" / "PA_BSND"
        sparse_mode: 0 (full) / 3 (rightDownCausal)
        pre_tokens/next_tokens: only default INT64_MAX supported
        attention_mode: only 2 (MLA-absorb) supported
        return_softmax_lse: also return softmax_max / softmax_sum
        block_size: PageAttention block token count (PA_BSND only)

    Returns:
        (attention_out, softmax_max, softmax_sum)
        softmax_max/sum: (B,1,S1,N1) BSND / (1,T1,N1) TND; zeros if not requested.
    """
    if attention_mode != 2:
        raise ValueError("Only attention_mode=2 (MLA-absorb) is supported")
    if pre_tokens != INT64_MAX or next_tokens != INT64_MAX:
        raise ValueError("pre_tokens/next_tokens only support default INT64_MAX")
    if sparse_mode not in (0, 3):
        raise ValueError("Only sparse_mode 0 (full) / 3 (rightDownCausal) supported")
    if query_rope is None or key_rope is None:
        raise ValueError("query_rope and key_rope are required (no empty rope)")
    if sparse_block_size < 1 or (sparse_block_size & (sparse_block_size - 1)) != 0 \
            or sparse_block_size > 128:
        raise ValueError("sparse_block_size must be a power of 2 in [1,128]")

    is_tnd_q = (layout_query == "TND")
    is_tnd_kv = (layout_kv == "TND")
    is_pa = (layout_kv == "PA_BSND")

    # Pre-init cross-branch vars: GRAPH_MODE's parser checks variable definition
    # across all branches (before dead-branch elimination), so anything used after
    # the merge must be defined in every path even when only BSND runs at runtime.
    act_q_pb = None
    act_k_pb = None

    # ---- normalize all layouts to dense BSND ----
    if is_tnd_q:
        act_q_pb = _tnd_cumsum_to_per_batch(actual_seq_lengths_query)
        q_bsnd = _tnd_to_bsnd(query, act_q_pb)
        qr_bsnd = _tnd_to_bsnd(query_rope, act_q_pb)
        si_bsnd = _tnd_to_bsnd(sparse_indices, act_q_pb)
    else:
        q_bsnd, qr_bsnd, si_bsnd = query, query_rope, sparse_indices

    if is_tnd_kv:
        act_k_pb = _tnd_cumsum_to_per_batch(actual_seq_lengths_kv)
        k_bsnd = _tnd_to_bsnd(key, act_k_pb)
        kr_bsnd = _tnd_to_bsnd(key_rope, act_k_pb)
    elif is_pa:
        if block_table is None:
            raise ValueError("PA_BSND requires block_table")
        act_k_list = list(actual_seq_lengths_kv)
        pa_block_size = block_size or key.shape[1]
        max_s2 = block_table.shape[1] * pa_block_size
        k_bsnd = _pa_to_bsnd(key, block_table, act_k_list, max_s2)
        kr_bsnd = _pa_to_bsnd(key_rope, block_table, act_k_list, max_s2)
    else:
        k_bsnd, kr_bsnd = key, key_rope

    B, S1, N1, D = q_bsnd.shape
    S2 = k_bsnd.shape[1]
    D_ROPE = qr_bsnd.shape[-1]

    if D not in _VALID_D:
        raise ValueError(f"D must be one of {_VALID_D}, got {D}")
    if D_ROPE != _D_ROPE:
        raise ValueError(f"rope dim must be {_D_ROPE}, got {D_ROPE}")
    if k_bsnd.shape[2] != 1:
        raise ValueError("Only N2=1 (MQA) is supported")
    if N1 not in _VALID_N1:
        raise ValueError(f"N1 must be one of {_VALID_N1}, got {N1}")

    act_q = _default_actual_seq_lens(
        None if is_tnd_q else actual_seq_lengths_query, B, S1)
    act_k = _default_actual_seq_lens(
        None if (is_tnd_kv or is_pa) else actual_seq_lengths_kv, B, S2)
    if is_tnd_q:
        act_q = act_q_pb.to(ms.int32)
    if is_tnd_kv:
        act_k = act_k_pb.to(ms.int32)
    elif is_pa:
        act_k = ms.Tensor(list(actual_seq_lengths_kv), dtype=ms.int32)

    # block-wise -> token-wise indices (kernel is purely token-wise)
    si_tok = _expand_block_indices(si_bsnd, sparse_block_size)
    topK = si_tok.shape[-1]

    # flatten (N2=1 / MQA): q[B*S1,N1,D], k/v[B*S2,D], sparse[B*S1,topK]
    # MLA-absorb: value = key[..,:D], so the kernel's v_ptr aliases k_flat
    # (the passed `value` tensor is ignored, matching CANN attention_mode=2).
    q_flat = q_bsnd.contiguous()
    qr_flat = qr_bsnd.contiguous()
    k_flat = k_bsnd.reshape(B * S2, D).contiguous()
    kr_flat = kr_bsnd.reshape(B * S2, D_ROPE).contiguous()
    sparse_flat = si_tok.reshape(B * S1, topK).to(ms.int32).contiguous()

    sparse_1d = sparse_flat.clamp(min=0).reshape(-1)
    k_gathered = ms.ops.gather(k_flat, sparse_1d, axis=0).reshape(B * S1, topK, D).contiguous()
    kr_gathered = ms.ops.gather(kr_flat, sparse_1d, axis=0).reshape(B * S1, topK, D_ROPE).contiguous()
    v_gathered = k_gathered

    # Output buffers must live on-device; mint.zeros defaults to CPU, which makes
    # triton reject the pointer ("cannot be accessed from Triton (cpu tensor?)").
    out_buf = ms.mint.zeros((B, S1, N1, D), dtype=q_bsnd.dtype).to('Ascend')
    sm_max_buf = ms.mint.zeros((B, 1, S1, N1), dtype=ms.float32).to('Ascend')
    sm_sum_buf = ms.mint.zeros((B, 1, S1, N1), dtype=ms.float32).to('Ascend')
    fp32_acc_buf = ms.mint.zeros((B, S1, N1, D), dtype=ms.float32).to('Ascend')

    out_buf, sm_max_buf, sm_sum_buf = _sfa_core(
        q_flat, qr_flat,
        k_gathered, kr_gathered, v_gathered,
        sparse_flat,
        out_buf, sm_max_buf, sm_sum_buf,
        fp32_acc_buf,
        act_q.to('Ascend'), act_k.to('Ascend'),
        B * S1, S1, S2, N1, topK,
        D, D_ROPE,
        float(scale_value),
        sparse_mode,
        1 if return_softmax_lse else 0,
    )

    if is_tnd_q:
        attention_out = _bsnd_to_tnd(out_buf, act_q_pb)
        # softmax (B,1,S1,N1) -> (1,T1,N1): squeeze N2, tnd-pack S1, restore N2 axis
        sm_max = _bsnd_to_tnd(sm_max_buf.transpose(0, 2, 1, 3), act_q_pb).transpose(1, 0, 2)
        sm_sum = _bsnd_to_tnd(sm_sum_buf.transpose(0, 2, 1, 3), act_q_pb).transpose(1, 0, 2)
    else:
        attention_out = out_buf
        sm_max, sm_sum = sm_max_buf, sm_sum_buf

    return attention_out, sm_max, sm_sum


class SparseFlashAttentionTriton(ms.nn.Cell):
    """nn.Cell wrapper around sparse_flash_attention_triton; see that function."""

    def __init__(
        self,
        scale_value=1.0,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=False,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        block_size=0,
    ):
        super().__init__()
        self.scale_value = scale_value
        self.sparse_block_size = sparse_block_size
        self.layout_query = layout_query
        self.layout_kv = layout_kv
        self.sparse_mode = sparse_mode
        self.attention_mode = attention_mode
        self.return_softmax_lse = return_softmax_lse
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.block_size = block_size

    def construct(
        self,
        query, key, value, sparse_indices,
        query_rope=None, key_rope=None,
        actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
        block_table=None,
    ):
        return sparse_flash_attention_triton(
            query, key, value, sparse_indices,
            scale_value=self.scale_value,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=query_rope, key_rope=key_rope,
            sparse_block_size=self.sparse_block_size,
            layout_query=self.layout_query,
            layout_kv=self.layout_kv,
            sparse_mode=self.sparse_mode,
            pre_tokens=self.pre_tokens,
            next_tokens=self.next_tokens,
            attention_mode=self.attention_mode,
            return_softmax_lse=self.return_softmax_lse,
            # block_size=self.block_size,   目前mindformers不用，先注释掉 by guod
        )