"""Triton-ascend implementation of sparse_flash_attention_grad (SFA backward).

Interface aligned with CANN aclnnSparseFlashAttentionGrad (op_host/
sparse_flash_attention_grad_def.cpp): 12 inputs / 7 attrs / 5 outputs, MLA-absorb
(attention_mode=2 semantics), MQA (N2=1). Pairs with the forward
sparse_flash_attention_triton.py; reuses the forward softmax_max / softmax_sum to
rebuild the attention probabilities P (no online-softmax recompute).

Math (per (b,s1), N2=1 / MQA, g = N1 heads share one gathered KV):
  score[h,p] = (q_nope·k_nope + q_rope·k_rope) * scale
  P[h,p]     = exp(score[h,p] - softmax_max[h]) / softmax_sum[h]
  delta[h]   = sum_d dO[h,d]·O[h,d]                       (rowsum(dO*O))
  dS[h,p]    = P[h,p] · (dO[h]·k_nope[p] - delta[h]) · scale
  d_query[h]      = sum_p dS[h,p] · k_nope[p]
  d_query_rope[h] = sum_p dS[h,p] · k_rope[p]
  d_key[u]       += sum_h dS[h,p->u] · q_nope[h]          (scatter-add over s1)
  d_key_rope[u]  += sum_h dS[h,p->u] · q_rope[h]          (scatter-add over s1)
  d_value[u]     += sum_h P[h,p->u]  · dO[h]              (scatter-add, P@dO path)

d_key (QK path) and d_value (P@dO path) are returned SEPARATELY, matching CANN's
5-output contract (key/value are independent inputs even though value=key in
MLA-absorb). A caller that shares the tensor sums the two itself.

Layout: BSND (TND/PA normalized to BSND on host, PyNative-only — same as forward).
sparse_mode 0 (full) / 3 (rightDownCausal). sparse_block_size 1 or 2^n in [1,128]
(block-wise pre-expanded to token indices on host). pre/next_tokens only default.
deterministic: only False (the scatter-add path is order-nondeterministic by
nature; True would require a serialized reduction not implemented here).

--------------------------------------------------------------------------------
mindformers integration (P.Morph bprop) — see CLAUDE.md 接入点 table
--------------------------------------------------------------------------------
The forward call site mindformers .../transformer/dsa/dsa_attention.py wraps the
forward in `P.Morph(self._sparse_flash_attention_forward, self.sfa_infer_shape,
...)` WITHOUT a bprop, relying on autodiff through the inner
ops.sparse_flash_attention (which lowers to CANN SparseFlashAttentionGrad).
After swapping the forward to the triton `_ms_pyfunc` op, that autodiff chain is
broken, so the Morph MUST be given an explicit bprop that calls this op. Paste:

    # in DSAttention.__init__, replacing the P.Morph(...) construction:
    self.sparse_flash_attention = P.Morph(
        self._sparse_flash_attention_forward,
        self.sfa_infer_shape,
        lambda *args: (args[0], mstype.float32, mstype.float32),
    ).add_prim_attr("self_define_shard", True)
    self.sparse_flash_attention.bprop = self._sfa_bprop   # attach backward

    # forward signature is _sparse_flash_attention_forward(self, q, k, v,
    #   topk_indices, query_rope, key_rope, actual_seq_qlen, actual_seq_kvlen);
    # Morph bprop receives (*inputs, out, dout) where out/dout are 3-tuples
    # (attention_out, softmax_max, softmax_sum). Only d(attention_out) feeds back.
    def _sfa_bprop(self, q, k, v, topk_indices, query_rope, key_rope,
                   actual_seq_qlen, actual_seq_kvlen, out, dout):
        attention_out, softmax_max, softmax_sum = out
        d_attention_out = dout[0]
        dq, dk, dv, dqr, dkr = sparse_flash_attention_grad_triton(
            q, k, v, topk_indices, d_attention_out, attention_out,
            softmax_max, softmax_sum, self.softmax_scale,
            query_rope=query_rope, key_rope=key_rope,
            actual_seq_lengths_query=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_kvlen,
            layout=self.input_layout, sparse_mode=<forward sparse_mode>)
        # k and v share the compressed latent c_kv -> the caller's split feeds
        # both back into the same key tensor; sum dk+dv for the merged k grad.
        d_qkv_rope_pad = self.zeros_like(query_rope)  # if rope grads are unused
        return (dq, dk + dv, self.zeros_like(v), dqr, dkr,
                None if actual_seq_qlen is None else self.zeros_like(actual_seq_qlen),
                None if actual_seq_kvlen is None else self.zeros_like(actual_seq_kvlen))
    # NB: align the returned tuple arity/order with the Morph forward inputs;
    # q_rope/k_rope grads (dqr/dkr) route to the upstream concat that built q/k.
--------------------------------------------------------------------------------
"""
import triton
import triton.language as tl
import triton.backends.ascend.runtime

import mindspore as ms
from mindspore import ops
import os
if os.environ.get("TRITON_ENABLE_TASKQUEUE", "true").lower() in ("true", "1"):
    os.environ["TRITON_ENABLE_TASKQUEUE"] = "false"

# host-side layout helpers shared with the forward (same normalization rules)
from sparse_flash_attention_triton import (
    _default_actual_seq_lens, _tnd_cumsum_to_per_batch,
    _tnd_to_bsnd, _bsnd_to_tnd, _expand_block_indices,
)

INT64_MAX = 9223372036854775807

_VALID_D = (128, 256, 512)
_D_ROPE = 64
_VALID_N1 = (1, 2, 4, 8, 16, 32, 64, 128)


def _next_pow2(x):
    # Ascend 要求 kernel grid 每维都是 2 的幂, 否则分核映射出错 -> aicore trap。
    return 1 << (x - 1).bit_length() if x > 1 else 1


def _patch_triton_ascend_mindspore_dtype_bytes():
    """修补 triton-ascend autotuner 的 dtype 字节数查询 (见 sparse_flash_attention_triton)。"""
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
    index_add after the kernel.
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
        # When NEED_CLAMP is False (N1 is a multiple of BLOCK_G), skip the
        # tl.where and use g_offs directly — avoids a select instruction per
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


# ---------------------------------------------------------------------------
# _ms_pyfunc core. Type annotations are required by _ms_pyfunc shape/dtype
# inference and must match between infer_func and core.
# ---------------------------------------------------------------------------
def _infer_sfa_grad(
    q_flat: ms.Tensor, qr_flat: ms.Tensor,
    k_flat: ms.Tensor, kr_flat: ms.Tensor, v_flat: ms.Tensor,
    sparse_flat: ms.Tensor,
    do_flat: ms.Tensor, o_flat: ms.Tensor,
    sm_max_flat: ms.Tensor, sm_sum_flat: ms.Tensor,
    dq_buf: ms.Tensor, dqr_buf: ms.Tensor,
    dk_buf: ms.Tensor, dkr_buf: ms.Tensor, dv_buf: ms.Tensor,
    act_q: ms.Tensor, act_k: ms.Tensor,
    B_S1: int, S1: int, S2: int, N1: int, topK: int,
    D: int, D_ROPE: int,
    scale_value: float,
    sparse_mode: int,
) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
    return (ms.mint.empty_like(dq_buf), ms.mint.empty_like(dqr_buf),
            ms.mint.empty_like(dk_buf), ms.mint.empty_like(dkr_buf),
            ms.mint.empty_like(dv_buf))


@ms.ops._ms_pyfunc(infer_func=_infer_sfa_grad)
def _sfa_grad_core(
    q_flat: ms.Tensor, qr_flat: ms.Tensor,
    k_flat: ms.Tensor, kr_flat: ms.Tensor, v_flat: ms.Tensor,
    sparse_flat: ms.Tensor,
    do_flat: ms.Tensor, o_flat: ms.Tensor,
    sm_max_flat: ms.Tensor, sm_sum_flat: ms.Tensor,
    dq_buf: ms.Tensor, dqr_buf: ms.Tensor,
    dk_buf: ms.Tensor, dkr_buf: ms.Tensor, dv_buf: ms.Tensor,
    act_q: ms.Tensor, act_k: ms.Tensor,
    B_S1: int, S1: int, S2: int, N1: int, topK: int,
    D: int, D_ROPE: int,
    scale_value: float,
    sparse_mode: int,
) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
    cfg = _select_block_config(D, N1, topK)
    block_g = cfg["BLOCK_G"]
    num_hc = triton.cdiv(N1, block_g)
    need_clamp = (N1 % block_g) != 0

    grid = (_next_pow2(B_S1),)

    B = B_S1 // S1
    ds_buf = ms.mint.zeros((B_S1, N1, topK), dtype=ms.float32).to('Ascend')
    dp_buf = ms.mint.zeros((B_S1, N1, topK), dtype=ms.float32).to('Ascend')

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

    batch_offsets = ms.ops.arange(B, dtype=ms.int32) * S2
    sparse_global = sparse_flat.reshape(B, S1, topK) + batch_offsets.reshape(B, 1, 1)
    sparse_1d = sparse_global.reshape(-1).clamp(min=0).astype(ms.int64)

    dk_contrib = ms.ops.bmm(ds_buf.reshape(B_S1, N1, topK).transpose(1, 2),
                            q_flat.reshape(B_S1, N1, D).astype(ms.float32))
    dk_buf = ms.ops.index_add(dk_buf, sparse_1d, dk_contrib.reshape(-1, D), 0)

    dv_contrib = ms.ops.bmm(dp_buf.reshape(B_S1, N1, topK).transpose(1, 2),
                            do_flat.reshape(B_S1, N1, D).astype(ms.float32))
    dv_buf = ms.ops.index_add(dv_buf, sparse_1d, dv_contrib.reshape(-1, D), 0)

    dkr_contrib = ms.ops.bmm(ds_buf.reshape(B_S1, N1, topK).transpose(1, 2),
                             qr_flat.reshape(B_S1, N1, D_ROPE).astype(ms.float32))
    dkr_buf = ms.ops.index_add(dkr_buf, sparse_1d, dkr_contrib.reshape(-1, D_ROPE), 0)

    return dq_buf, dqr_buf, dk_buf, dkr_buf, dv_buf


# ---------------------------------------------------------------------------
# public API — aligned with CANN aclnnSparseFlashAttentionGrad signature
# ---------------------------------------------------------------------------
def sparse_flash_attention_grad_triton(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    sparse_indices: ms.Tensor,
    d_out: ms.Tensor,
    out: ms.Tensor,
    softmax_max: ms.Tensor,
    softmax_sum: ms.Tensor,
    scale_value: float = 1.0,
    query_rope=None,
    key_rope=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    sparse_block_size: int = 1,
    layout: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = INT64_MAX,
    next_tokens: int = INT64_MAX,
    deterministic: bool = False,
):
    """SparseFlashAttentionGrad (MLA-absorb, MQA). Backward of SFA forward.

    Args:
        query: [B,S1,N1,D] BSND or [T1,N1,D] TND, fp16/bf16 (D in 128/256/512)
        key:   [B,S2,1,D] BSND / [T2,1,D] TND
        value: same layout/shape as key; IGNORED in MLA-absorb (value=key[..,:D]),
               but its gradient d_value IS returned separately (CANN parity).
        sparse_indices: [B,S1,1,sparse_count] (or TND) int32, block ids, -1 invalid
        d_out: [B,S1,N1,D] gradient of the attention output
        out:   [B,S1,N1,D] forward attention output
        softmax_max/sum: [B,1,S1,N1] BSND / [1,T1,N1] TND fp32 — forward stats
        scale_value: softmax scale (1/sqrt(d_k))
        query_rope/key_rope: [.,N,Dr] (Dr=64), required (no empty rope)
        actual_seq_lengths_query/kv: [B] int32 / list / None (cumulative for TND)
        sparse_block_size: 1 (token-wise) or 2^n in [1,128] (block-wise)
        layout: "BSND" / "TND"
        sparse_mode: 0 (full) / 3 (rightDownCausal)
        pre_tokens/next_tokens: only default INT64_MAX supported
        deterministic: only False (scatter-add is order-nondeterministic)

    Returns:
        (d_query, d_key, d_value, d_query_rope, d_key_rope), each matching the
        shape/dtype of its corresponding forward input. d_key (QK path) and
        d_value (P@dO path) are separate; a caller sharing key==value sums them.
    """
    if pre_tokens != INT64_MAX or next_tokens != INT64_MAX:
        raise ValueError("pre_tokens/next_tokens only support default INT64_MAX")
    if sparse_mode not in (0, 3):
        raise ValueError("Only sparse_mode 0 (full) / 3 (rightDownCausal) supported")
    if query_rope is None or key_rope is None:
        raise ValueError("query_rope and key_rope are required (no empty rope)")
    if deterministic:
        raise ValueError("deterministic=True is not supported (scatter-add path)")
    if sparse_block_size < 1 or (sparse_block_size & (sparse_block_size - 1)) != 0 \
            or sparse_block_size > 128:
        raise ValueError("sparse_block_size must be a power of 2 in [1,128]")

    is_tnd = (layout == "TND")

    # Pre-init cross-branch vars (GRAPH_MODE parser checks definition across all
    # branches before dead-branch elimination — see forward + memory note).
    act_q_pb = None
    act_k_pb = None

    # ---- normalize TND -> dense BSND (PyNative only; GRAPH_MODE caller pre-converts) ----
    if is_tnd:
        act_q_pb = _tnd_cumsum_to_per_batch(actual_seq_lengths_query)
        act_k_pb = _tnd_cumsum_to_per_batch(actual_seq_lengths_kv)
        q_bsnd = _tnd_to_bsnd(query, act_q_pb)
        qr_bsnd = _tnd_to_bsnd(query_rope, act_q_pb)
        si_bsnd = _tnd_to_bsnd(sparse_indices, act_q_pb)
        do_bsnd = _tnd_to_bsnd(d_out, act_q_pb)
        o_bsnd = _tnd_to_bsnd(out, act_q_pb)
        k_bsnd = _tnd_to_bsnd(key, act_k_pb)
        kr_bsnd = _tnd_to_bsnd(key_rope, act_k_pb)
        # softmax (1,T1,N1) -> (B,1,S1,N1): drop N2 axis, tnd-unpack S1, restore N2
        sm_max_bsnd = _tnd_to_bsnd(softmax_max.transpose(1, 0, 2), act_q_pb).transpose(0, 2, 1, 3)
        sm_sum_bsnd = _tnd_to_bsnd(softmax_sum.transpose(1, 0, 2), act_q_pb).transpose(0, 2, 1, 3)
    else:
        q_bsnd, qr_bsnd, si_bsnd = query, query_rope, sparse_indices
        do_bsnd, o_bsnd = d_out, out
        k_bsnd, kr_bsnd = key, key_rope
        sm_max_bsnd, sm_sum_bsnd = softmax_max, softmax_sum

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
        None if is_tnd else actual_seq_lengths_query, B, S1)
    act_k = _default_actual_seq_lens(
        None if is_tnd else actual_seq_lengths_kv, B, S2)
    if is_tnd:
        act_q = act_q_pb.to(ms.int32)
        act_k = act_k_pb.to(ms.int32)

    # block-wise -> token-wise indices (kernel is purely token-wise)
    si_tok = _expand_block_indices(si_bsnd, sparse_block_size)
    topK = si_tok.shape[-1]

    # flatten (N2=1 / MQA). MLA-absorb: value=key[..,:D] -> kernel v_ptr aliases
    # k_flat (passed `value` ignored); d_value still computed into its own buffer.
    q_flat = q_bsnd.contiguous()
    qr_flat = qr_bsnd.contiguous()
    do_flat = do_bsnd.contiguous()
    o_flat = o_bsnd.contiguous()
    k_flat = k_bsnd.reshape(B * S2, D).contiguous()
    kr_flat = kr_bsnd.reshape(B * S2, D_ROPE).contiguous()
    sparse_flat = si_tok.reshape(B * S1, topK).to(ms.int32).contiguous()
    sm_max_flat = sm_max_bsnd.reshape(B * S1 * N1).astype(ms.float32).contiguous()
    sm_sum_flat = sm_sum_bsnd.reshape(B * S1 * N1).astype(ms.float32).contiguous()

    batch_offsets = ms.ops.arange(B, dtype=ms.int32) * S2
    sparse_global = sparse_flat.reshape(B, S1, topK) + batch_offsets.reshape(B, 1, 1)
    sparse_1d = sparse_global.reshape(-1).clamp(min=0)
    k_gathered = ms.ops.gather(k_flat, sparse_1d, axis=0).reshape(B * S1, topK, D).contiguous()
    kr_gathered = ms.ops.gather(kr_flat, sparse_1d, axis=0).reshape(B * S1, topK, D_ROPE).contiguous()
    v_gathered = k_gathered

    dq_buf = ms.mint.zeros((B, S1, N1, D), dtype=q_bsnd.dtype).to('Ascend')
    dqr_buf = ms.mint.zeros((B, S1, N1, D_ROPE), dtype=qr_bsnd.dtype).to('Ascend')
    dk_buf = ms.mint.zeros((B * S2, D), dtype=ms.float32).to('Ascend')
    dkr_buf = ms.mint.zeros((B * S2, D_ROPE), dtype=ms.float32).to('Ascend')
    dv_buf = ms.mint.zeros((B * S2, D), dtype=ms.float32).to('Ascend')

    dq_buf, dqr_buf, dk_buf, dkr_buf, dv_buf = _sfa_grad_core(
        q_flat, qr_flat,
        k_gathered, kr_gathered, v_gathered,
        sparse_flat,
        do_flat, o_flat,
        sm_max_flat, sm_sum_flat,
        dq_buf, dqr_buf,
        dk_buf, dkr_buf, dv_buf,
        act_q.to('Ascend'), act_k.to('Ascend'),
        B * S1, S1, S2, N1, topK,
        D, D_ROPE,
        float(scale_value),
        sparse_mode,
    )

    # fp32 workspace -> key/value dtype, reshape to BSND [B,S2,1,*]
    d_key = dk_buf.reshape(B, S2, 1, D).astype(key.dtype)
    d_key_rope = dkr_buf.reshape(B, S2, 1, D_ROPE).astype(key_rope.dtype)
    d_value = dv_buf.reshape(B, S2, 1, D).astype(value.dtype)
    d_query, d_query_rope = dq_buf, dqr_buf

    if is_tnd:
        d_query = _bsnd_to_tnd(d_query, act_q_pb)
        d_query_rope = _bsnd_to_tnd(d_query_rope, act_q_pb)
        d_key = _bsnd_to_tnd(d_key, act_k_pb)
        d_key_rope = _bsnd_to_tnd(d_key_rope, act_k_pb)
        d_value = _bsnd_to_tnd(d_value, act_k_pb)

    return d_query, d_key, d_value, d_query_rope, d_key_rope


class SparseFlashAttentionGradTriton(ms.nn.Cell):
    """nn.Cell wrapper around sparse_flash_attention_grad_triton; see that function."""

    def __init__(
        self,
        scale_value=1.0,
        sparse_block_size=1,
        layout="BSND",
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        deterministic=False,
    ):
        super().__init__()
        self.scale_value = scale_value
        self.sparse_block_size = sparse_block_size
        self.layout = layout
        self.sparse_mode = sparse_mode
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.deterministic = deterministic

    def construct(
        self,
        query, key, value, sparse_indices,
        d_out, out, softmax_max, softmax_sum,
        query_rope=None, key_rope=None,
        actual_seq_lengths_query=None, actual_seq_lengths_kv=None,
    ):
        return sparse_flash_attention_grad_triton(
            query, key, value, sparse_indices,
            d_out, out, softmax_max, softmax_sum,
            scale_value=self.scale_value,
            query_rope=query_rope, key_rope=key_rope,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            sparse_block_size=self.sparse_block_size,
            layout=self.layout,
            sparse_mode=self.sparse_mode,
            pre_tokens=self.pre_tokens,
            next_tokens=self.next_tokens,
            deterministic=self.deterministic,
        )