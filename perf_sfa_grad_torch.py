"""Torch perf script for sfa_grad_core_standalone._sfa_grad_core.

Shape: B=1, S1=512, S2=4096, N1=64, topK=2048, D=512, bf16, sparse_mode=3.
Forward stats (out/smax/ssum) are pre-computed via run_sfa; only the backward
kernel is benchmarked.

Run:
    python perf_sfa_grad_torch.py                  # timing (median/p20/p80)
    python perf_sfa_grad_torch.py --kernel-only
    python perf_sfa_grad_torch.py --prof           # torch_npu profiler + op_statistic
"""
from __future__ import annotations

import csv
import sys
import time

import numpy as np
import torch

from sfa_torch_utils import D_ROPE, DEVICE, make_inputs, make_sparse_indices, run_sfa
from sfa_grad_torch_utils import make_dout, run_sfa_grad

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
    do = make_dout(B, S1, N1, D_NOPE, DTYPE)
    si = make_sparse_indices(B, S1, S2, SPARSE_COUNT, 1, SPARSE_MODE)
    scale = 1.0 / np.sqrt(D_NOPE + D_ROPE)
    # Pre-compute forward stats (not benchmarked)
    out, smax, ssum = run_sfa(q, k, qr, kr, si, 1, SPARSE_MODE, scale, return_lse=True)
    return q, k, qr, kr, si, do, out, smax, ssum, scale


def run_timing():
    q, k, qr, kr, si, do, out, smax, ssum, scale = _build()
    print(f"\nB={B}, S1={S1}, S2={S2}, N1={N1}, topk={SPARSE_COUNT}, D={D_NOPE}, dtype={DTYPE}")
    t_med, t_p20, t_p80 = _do_bench(
        lambda: run_sfa_grad(q, k, qr, kr, si, do, out, smax, ssum,
                             1, SPARSE_MODE, scale))
    print(f"triton bwd:  median={t_med:.2f}ms, p20={t_p20:.2f}ms, p80={t_p80:.2f}ms")


def _normalize_col_name(name):
    return name.strip().lower().replace(" ", "").replace("_", "")


def _find_avg_time_us(out_dir, kernel_name):
    from pathlib import Path
    search_root = Path(out_dir)
    csv_files = sorted(search_root.rglob("op_statistic.csv"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    for csv_path in csv_files:
        with csv_path.open("r", newline="", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter=",")
            if not reader.fieldnames:
                continue
            col_map = {_normalize_col_name(c): c for c in reader.fieldnames if c}
            op_type_col = col_map.get("optype")
            avg_time_col = col_map.get("avgtime(us)")
            if op_type_col is None or avg_time_col is None:
                continue
            for row in reader:
                if row.get(op_type_col, "").strip() != kernel_name:
                    continue
                try:
                    return float(row[avg_time_col]), str(csv_path)
                except ValueError:
                    return None, str(csv_path)
    return None, None


def run_profiling():
    import torch_npu
    import torch_npu.profiler as npu_prof
    q, k, qr, kr, si, do, out, smax, ssum, scale = _build()
    out_dir = "./profiler_data_sfa_grad_torch"
    kernel_name = "_sfa_grad_kernel"

    wait, warmup_n, active, repeat, skip_first = 1, 1, 20, 1, 1
    total_steps = skip_first + repeat * (wait + warmup_n + active)

    experimental_config = npu_prof._ExperimentalConfig(
        aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
        profiler_level=npu_prof.ProfilerLevel.Level1,
        l2_cache=False,
    )

    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        schedule=npu_prof.schedule(wait=wait, warmup=warmup_n, active=active,
                                   repeat=repeat, skip_first=skip_first),
        experimental_config=experimental_config,
        on_trace_ready=npu_prof.tensorboard_trace_handler(out_dir),
    ) as prof:
        for _ in range(total_steps):
            run_sfa_grad(q, k, qr, kr, si, do, out, smax, ssum,
                         1, SPARSE_MODE, scale)
            torch_npu.npu.synchronize()
            prof.step()

    torch_npu.npu.synchronize()
    print(f"Profiler data saved to {out_dir}")
    time_us, csv_path = _find_avg_time_us(out_dir, kernel_name)
    print(f"\n{'='*80}")
    print(f"_sfa_grad_kernel avg time: {time_us}us" if time_us else "_sfa_grad_kernel not found in op_statistic.csv")
    if csv_path:
        print(f"Source: {csv_path}")
    print(f"{'='*80}")


def run_kernel_only():
    q, k, qr, kr, si, do, out, smax, ssum, scale = _build()
    for _ in range(10):
        run_sfa_grad(q, k, qr, kr, si, do, out, smax, ssum,
                     1, SPARSE_MODE, scale)
    _synchronize()
    print("kernel-only run finished")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--kernel-only":
        run_kernel_only()
    elif mode == "--prof":
        run_profiling()
    else:
        run_timing()
