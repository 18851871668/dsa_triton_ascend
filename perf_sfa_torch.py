"""Torch perf script for sfa_core_standalone._sfa_core — single shape from perf_sfa_triton.

Shape: B=1, S1=512, S2=4096, N1=64, topK=2048, D=512, bf16, sparse_mode=3, return_lse.
Run:
    python perf_sfa_torch.py                  # timing (median/p20/p80)
    python perf_sfa_torch.py --autotune-confirm
    python perf_sfa_torch.py --kernel-only
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from sfa_torch_utils import D_ROPE, DEVICE, make_inputs, make_sparse_indices, run_sfa

D_NOPE = 512
B, S1, S2, N1, SPARSE_COUNT = 1, 512, 4096, 64, 2048
DTYPE = torch.bfloat16
SPARSE_MODE = 3


def _synchronize():
    try:
        torch.npu.synchronize()
    except (AttributeError, RuntimeError):
        pass


def _empty_cache():
    try:
        torch.npu.empty_cache()
    except (AttributeError, RuntimeError):
        pass


def _do_bench(fn, warmup=10, rep=50):
    for _ in range(warmup):
        out = fn()
        _synchronize()
        del out
        _empty_cache()

    times = []
    for _ in range(rep):
        _synchronize()
        t0 = time.perf_counter()
        out = fn()
        _synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        del out
        _empty_cache()

    times.sort()
    n = len(times)
    return times[n // 2], times[n // 5], times[4 * n // 5]


def _build():
    q, k, qr, kr = make_inputs(B, S1, S2, N1, D_NOPE, DTYPE)
    si = make_sparse_indices(B, S1, S2, SPARSE_COUNT, 1, SPARSE_MODE)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)
    return q, k, qr, kr, si, scale


def run_autotune_confirm():
    q, k, qr, kr, si, scale = _build()
    print(f"\nShape: B={B}, S1={S1}, S2={S2}, N1={N1}, topK={SPARSE_COUNT}, D={D_NOPE}")
    print("Running 1 invocation to trigger autotune — TRITON_PRINT_AUTOTUNING=1 output in stderr")
    print("=" * 80)
    out, smax, ssum = run_sfa(q, k, qr, kr, si, 1, SPARSE_MODE, scale, return_lse=True)
    _synchronize()
    print(f"Autotune done. Output[0] shape: {tuple(out.shape)}")


def run_timing():
    q, k, qr, kr, si, scale = _build()
    print(f"\nB={B}, S1={S1}, S2={S2}, N1={N1}, topk={SPARSE_COUNT}, D={D_NOPE}, dtype={DTYPE}")
    t_med, t_p20, t_p80 = _do_bench(
        lambda: run_sfa(q, k, qr, kr, si, 1, SPARSE_MODE, scale, return_lse=True))
    print(f"triton:  median={t_med:.2f}ms, p20={t_p20:.2f}ms, p80={t_p80:.2f}ms")


def run_kernel_only():
    q, k, qr, kr, si, scale = _build()
    for _ in range(10):
        run_sfa(q, k, qr, kr, si, 1, SPARSE_MODE, scale, return_lse=True)
    _synchronize()
    print("kernel-only run finished")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--kernel-only":
        run_kernel_only()
    elif mode == "--autotune-confirm":
        run_autotune_confirm()
    else:
        run_timing()
