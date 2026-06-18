# SFA Torch 算子使用指南

本指南面向首次使用 `sfa_core_standalone.py`（从 `sparse_flash_attention_triton.py` 提取的、无 MindSpore 依赖的纯 torch/triton 算子）的同学。介绍 4 个配套脚本如何运行、各自用途、以及常见问题。

## 环境依赖

| 组件 | 版本 | 说明 |
|------|------|------|
| CANN | 9.0.0 | Ascend 驱动 + runtime |
| triton-ascend | 3.2.1 | kernel 编译/启动后端 |
| torch | ≥ 2.1 | 张量载体 |
| torch_npu | 对应 torch 版本 | 注册 `npu` device（CPU 环境脚本可 import 但 kernel 无法运行） |
| numpy | 任意 | golden 参考与输入构造 |
| pytest | ≥ 7 | 跑 `test_sfa_torch.py` |

> 与原 MindSpore 路径不同：**不需要** mindspore / pytest-forked。

## 文件说明

| 文件 | 作用 | 是否需要 NPU |
|------|------|--------------|
| `sfa_core_standalone.py` | 核心：`_sfa_core` launcher + `_sfa_kernel`/`_sfa_scores_block` kernel 链，零 MindSpore 依赖 | 调用时需要 |
| `sfa_torch_utils.py` | 共享工具：输入构造、BSND→flat 展平、缓冲区分配、`run_sfa` 一键启动包装、`allclose` 对比 | 调用时需要 |
| `sfa_demo_torch.py` | 最小调用示例：单 shape 跑通 + 打印输出 + golden 校验 | 是 |
| `test_sfa_torch.py` | pytest 参数化测试：golden 精度 + 形状/dtype/有限性自检 | 是 |
| `perf_sfa_torch.py` | 性能脚本：固定 shape `B=1,S1=512,S2=4096,N1=64,topK=2048,D=512,bf16` 的 timing/autotune/kernel-only | 是 |

## 快速开始

### 1. 跑通最小示例

```bash
python sfa_demo_torch.py
```

预期输出（NPU 环境）：

```
out   (1, 4, 8, 512) torch.float16
smax  (1, 1, 4, 8) torch.float32
ssum  (1, 1, 4, 8) torch.float32
out finite: True
golden check passed!
```

这验证了：输入构造 → 展平 → 缓冲区分配 → kernel 启动 → numpy golden 对比 全链路打通。

### 2. 跑测试

```bash
# 全量参数化测试（golden + basic）
pytest test_sfa_torch.py -v

# 快速冒烟（直接 python 跑 __main__ 的 4 个用例）
python test_sfa_torch.py
```

预期：

```
golden (D=512, token-wise, mode3, fp16) passed!
golden (D=128, token-wise, mode3, fp16) passed!
golden (D=256, block-wise bs=2, mode3, fp16) passed!
golden (D=256, topK=2048, two-pass, mode3, fp16) passed!
```

测试覆盖矩阵：

| 维度 | 取值 |
|------|------|
| D | 128, 256, 512 |
| dtype | fp16, bf16 |
| sparse_block_size | 1（token-wise）, 2, 4（block-wise） |
| sparse_mode | 3（rightDownCausal） |
| topK | 16 ~ 2048（含 two-pass 路径） |

### 3. 跑性能

```bash
# 默认：timing（warmup=10, rep=50，报 median/p20/p80）
python perf_sfa_torch.py

# 触发 autotune 并打印选中 config（建议设环境变量 TRITON_PRINT_AUTOTUNING=1）
python perf_sfa_torch.py --autotune-confirm

# 仅跑 10 次 kernel，不计时（验证可跑通）
python perf_sfa_torch.py --kernel-only
```

预期（timing 模式）：

```
B=1, S1=512, S2=4096, N1=64, topk=2048, D=512, dtype=torch.bfloat16
triton:  median=XX.XXms, p20=XX.XXms, p80=XX.XXms
```

> `perf_sfa_torch.py` 的 shape 与原 `perf_sfa_triton.py` 启用的唯一 config 完全一致，便于横向对比 torch 路径与 MindSpore 路径的耗时。

## 调用约定（自己集成时需知）

`_sfa_core` 是**裸 launcher**，调用方负责所有预处理：

1. **输入布局**：Q/K/QR/KR 必须是 BSND 连续张量；`value` 由 kernel 内部 alias `key`（MLA-absorb），无需单独传 V。
2. **展平**：K/KR reshape 成 `[B*S2, D]`；sparse_indices 展开为 `[B*S1, topK]` int32。
3. **block-wise**：`sparse_block_size>1` 时需先调 `expand_block_indices` 把 block id 展成 token id。
4. **缓冲区**：调用方预分配 4 个 device 张量（kernel 原地写）：
   - `out_buf`: `[B,S1,N1,D]` 同输入 dtype
   - `sm_max_buf` / `sm_sum_buf`: `[B,1,S1,N1]` fp32
   - `fp32_acc_buf`: `[B,S1,N1,D]` fp32（two-pass 路径累加器）
5. **act_q/act_k**：`[B]` int32，每 batch 有效序列长度；None 时用 `torch.full((B,), S, int32)`。
6. **return_lse**：传 `1`/`0`（int，非 bool），控制是否写 smax/ssum。

`run_sfa()`（见 `sfa_torch_utils.py`）已封装上述全部步骤，**建议直接复用**，除非有特殊布局需求。

## 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `RuntimeError: ... cannot be accessed from Triton (cpu tensor?)` | 缓冲区在 CPU | 确认 `torch_npu` 已安装，`DEVICE="npu"` 生效；或显式 `.to("npu")` |
| `import torch_npu` 失败 | 未安装 torch_npu | `pip install torch-npu`（版本须与 torch 匹配） |
| autotune 卡住很久 | 首次跑 46 个 config | 用 `--autotune-confirm` 单独触发；或减小 warmup/rep |
| golden 校验失败 | bf16 容差 | `allclose` 已对 bf16 放宽（rtol=7.8e-3, atol=7e-4）；若仍失败检查输入是否对齐 |
| `DEVICE="cpu"` 但想跑 kernel | 无 NPU | kernel 必须在 Ascend 上运行，CPU 无法执行 triton-ascend kernel |

## 与原 MindSpore 脚本的对应关系

| torch 脚本 | 对应 MindSpore 脚本 | 差异 |
|------------|---------------------|------|
| `sfa_core_standalone.py` | `sparse_flash_attention_triton.py`（`_sfa_core` 部分） | 去除 `@ms.ops._ms_pyfunc`、`_infer_sfa`、`_save_sfa_inputs`、布局归一化 helper、`SparseFlashAttentionTriton` cell |
| `sfa_demo_torch.py` | （无） | 新增的最小 demo |
| `test_sfa_torch.py` | `test_sfa_triton.py` | 去除 `test_accuracy`（依赖 CANN `ops.sparse_flash_attention`，torch 路径无此基线），保留 golden + basic |
| `perf_sfa_torch.py` | `perf_sfa_triton.py` | 仅保留 `run_timing`/`run_autotune_confirm`/`run_kernel_only`，去除 CANN 对比与 ms.profiler（torch 侧可改用 `torch_npu.npu.prof`） |
