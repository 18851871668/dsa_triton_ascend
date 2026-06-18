"""Torch helpers for sfa_core_standalone._sfa_core: input construction,
BSND->flat packing, buffer allocation, kernel launch, and numpy comparison.

MLA-absorb: value = key (the kernel's v_ptr aliases k_flat), so no separate V.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401  registers the Ascend npu backend
    DEVICE = "npu"
except ImportError:
    DEVICE = "cpu"  # kernels require Ascend; cpu only lets the module import

D_ROPE = 64


def _np(arr, dtype, device=DEVICE):
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype).to(device)


def make_inputs(B, S1, S2, N1, D, dtype=torch.float16, device=DEVICE, seed=42):
    rng = np.random.RandomState(seed)
    q = _np(rng.randn(B, S1, N1, D), dtype, device)
    k = _np(rng.randn(B, S2, 1, D), dtype, device)
    qr = _np(rng.randn(B, S1, N1, D_ROPE), dtype, device)
    kr = _np(rng.randn(B, S2, 1, D_ROPE), dtype, device)
    return q, k, qr, kr


def make_sparse_indices(B, S1, S2, sparse_count, sparse_block_size, sparse_mode,
                        device=DEVICE, seed=7):
    rng = np.random.RandomState(seed)
    si = np.full((B, S1, 1, sparse_count), -1, dtype=np.int32)
    act_q, act_k = S1, S2
    for b in range(B):
        for s1 in range(S1):
            if sparse_mode == 0:
                threshold = act_k
            else:
                threshold = act_k - act_q + s1 + 1
            if threshold <= 0:
                continue
            num_blocks = int(np.ceil(threshold / sparse_block_size))
            n = min(sparse_count, num_blocks)
            perm = rng.permutation(num_blocks)[:n]
            si[b, s1, 0, :n] = np.sort(perm).astype(np.int32)
    return torch.from_numpy(si).to(device)


def expand_block_indices(sparse_indices, sparse_block_size):
    if sparse_block_size == 1:
        return sparse_indices
    bs = sparse_block_size
    base = sparse_indices.to(torch.int32)
    lead = base.shape[:-1]
    topK = base.shape[-1]
    base = base.reshape(*lead, topK, 1)
    offs = torch.arange(bs, dtype=torch.int32, device=base.device).reshape(*([1] * len(lead)), 1, bs)
    neg1 = torch.full_like(base, -1)
    tokens = torch.where(base == -1, neg1, base * bs + offs)
    return tokens.reshape(*lead, topK * bs)


def run_sfa(q, k, qr, kr, sparse_indices, sparse_block_size, sparse_mode,
            scale_value, return_lse, device=DEVICE):
    from sfa_core_standalone import _sfa_core
    B, S1, N1, D = q.shape
    S2 = k.shape[1]
    si_tok = expand_block_indices(sparse_indices, sparse_block_size)
    topK = si_tok.shape[-1]

    q_flat = q.contiguous()
    qr_flat = qr.contiguous()
    k_flat = k.reshape(B * S2, D).contiguous()
    kr_flat = kr.reshape(B * S2, D_ROPE).contiguous()
    v_flat = k_flat
    sparse_flat = si_tok.reshape(B * S1, topK).to(torch.int32).contiguous()

    out_buf = torch.zeros((B, S1, N1, D), dtype=q.dtype, device=device)
    sm_max_buf = torch.zeros((B, 1, S1, N1), dtype=torch.float32, device=device)
    sm_sum_buf = torch.zeros((B, 1, S1, N1), dtype=torch.float32, device=device)
    fp32_acc_buf = torch.zeros((B, S1, N1, D), dtype=torch.float32, device=device)

    act_q = torch.full((B,), S1, dtype=torch.int32, device=device)
    act_k = torch.full((B,), S2, dtype=torch.int32, device=device)

    out, smax, ssum = _sfa_core(
        q_flat, qr_flat, k_flat, kr_flat, v_flat, sparse_flat,
        out_buf, sm_max_buf, sm_sum_buf, fp32_acc_buf,
        act_q, act_k,
        B * S1, S1, S2, N1, topK, D, D_ROPE,
        float(scale_value), sparse_mode, 1 if return_lse else 0,
    )
    return out, smax, ssum


def to_np_f32(t):
    return t.float().cpu().numpy()


def allclose(a, b, bf16=False, pct_thd=None, rtol=None, atol=None, max_diff_hd=10):
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
    if close.sum() / close.size * 100.0 < pct_thd:
        return False
    diff_abs = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-10
    rel_err = diff_abs / denom
    if np.any(rel_err[~close] >= max_diff_hd):
        return False
    return True
