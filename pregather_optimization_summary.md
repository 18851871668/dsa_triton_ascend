# 预 Gather (Pre-Gather) 优化总结

## 1. 优化背景

### 1.1 问题分析

SFA (Sparse Flash Attention) kernel 的核心操作是根据稀疏索引 `sparse_indices` 从 KV cache 中 gather 数据。原始实现在 kernel 内部使用 `tok_clamped` 进行随机地址访问：

```python
# 原始 kernel 内部 (scattered gather)
tok = tl.load(sparse_ptr + sp_base + blk_offs, ...)
tok_clamped = tl.where(tok_valid, tok, 0)  # 离散的随机 token 索引 (0~4095)

# K/V 的加载地址是随机跳跃的
k_tile = tl.load(k_ptr + k_base + tok_clamped[:, None] * D + d_offs[None, :])
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                  64 个 token 在 S2=4096 行中随机分布 → L1/L2 cache miss 严重
```

### 1.2 仿真数据佐证

从 msprof 仿真数据 (`profiling_orig3`) 分析：

| 指标 | 值 | 说明 |
|------|-----|------|
| Cube Core `WAIT_FLAG_DEVI` | 36.53 us (51.7%) | 等待 GM 数据到达 (L1 cache miss) |
| Vec0 MTE2 | 44.86 us (22.5%) | GM→UB 加载 (含 scattered K/V gather) |
| Vec0 MTE3 | 35.90 us (18.0%) | UB→GM 写回 (fp32_acc store) |
| `MOV_OUT_TO_L1_MULTI_ND2NZ` | 8次, 2856 cycles | GM→L1 + ND2NZ 格式转换 (scattered) |

**根因**: `tok_clamped` 是完全离散的随机索引，BLOCK_K=64 个 token 在 S2=4096 行中随机分布。每个 token 行 D=512×2B=1KB，物理分散在 4096×1KB=4MB 地址空间中，L1 cache line (64B) 命中率极低。

### 1.3 优化思路

将 kernel 内部的 scattered gather 移到 host 端，预先把 K/KR 按稀疏索引 gather 到连续 buffer 中，kernel 内部改为顺序读取：

```
原始 (kernel 内 scattered gather):
  GM: K[B*S2, D] (4MB, 稀疏分布)
       ↓ MTE2 随机 gather (慢! cache miss 严重)
  UB:  k_tile[BLOCK_K, BLOCK_D]

优化后 (host 预 gather + kernel 顺序读):
  GM: K[B*S2, D] (4MB)
       ↓ torch.index_select (NPU 硬件优化, 1.83ms)
  GM: K_gathered[B*S1, topK, D] (1GB, 连续)
       ↓ MTE2 顺序读取 (快! cache friendly)
  UB:  k_tile[BLOCK_K, BLOCK_D]
```

## 2. 具体改动

### 2.1 `sfa_torch_utils.py` — Host 端预 Gather

在 `run_sfa()` 函数中，kernel 启动前用 `torch.index_select` 预 gather K/KR：

```python
# 预 gather K/KR 到连续 buffer [B*S1, topK, D]
# -1 索引 clamp 到 0 (安全索引)，kernel 内 tok_valid=False 会 mask 掉
sparse_1d = sparse_flat.clamp(min=0).reshape(-1)    # [B*S1*topK]
k_gathered = torch.index_select(k_flat, 0, sparse_1d).reshape(B * S1, topK, D).contiguous()
kr_gathered = torch.index_select(kr_flat, 0, sparse_1d).reshape(B * S1, topK, D_ROPE).contiguous()
v_gathered = k_gathered                              # MLA-absorb: V=K
```

**关键点**：
- 使用 `torch.index_select` 而非 fancy indexing (`k_flat[sparse_safe]`)，因为 NPU 上 `index_select` 比 fancy indexing **快 8.8x** (1.83ms vs 16.12ms)
- `-1` (无效索引) 被 `clamp(min=0)` 映射到索引 0，但 kernel 内 `tok_valid=False` 会将 scores 设为 `-inf`，所以 gather 到的 K[0] 数据不会被使用
- `v_gathered = k_gathered`：MLA-absorb 模式下 V=K，共享同一 buffer

### 2.2 `sfa_core_standalone.py` — Kernel 顺序读取

#### 2.2.1 Base 偏移改为 gathered 布局

```python
# 原始 (K 布局: [B*S2, D])
k_base = b * S2 * D
kr_base = b * S2 * D_ROPE
v_base = b * S2 * D

# 优化后 (K 布局: [B*S1, topK, D] 连续)
k_base = pid_bs1 * topK * D
kr_base = pid_bs1 * topK * D_ROPE
v_base = pid_bs1 * topK * D
```

#### 2.2.2 K/KR/V 寻址从 tok_clamped 改为 blk_offs

```python
# 原始 (scattered: tok_clamped 是随机索引)
k_tile = tl.load(
    k_ptr + k_base + tok_clamped[:, None] * D + d_offs[None, :],
    mask=tok_valid[:, None] & d_valid[None, :], other=0.0)

# 优化后 (sequential: blk_offs = blk_start + arange(0, BLOCK_K) 是连续的)
k_tile = tl.load(
    k_ptr + k_base + blk_offs[:, None] * D + d_offs[None, :],
    mask=tok_valid[:, None] & d_valid[None, :], other=0.0)
```

**同样改动应用于**：
- `_sfa_scores_block()` 函数（参数名 `tok_clamped` → `blk_offs`）
- SINGLE_BLOCK path 的 K/KR/V 加载
- chunked path 的 inline score computation (K/KR 加载)
- chunked path 的 V 加载 (P@V accumulation)
- last-block 的 inline score computation (K/KR 加载)
- last-block 的 V 加载 (归一化输出)

#### 2.2.3 tok_valid 保留 (仅用于 mask)

```python
# tok_valid 仍用 sparse 索引计算 (causal mask 不变)
tok = tl.load(sparse_ptr + sp_base + blk_offs, ...)
tok_valid = blk_in_count & (tok != -1) & (tok < threshold) & (tok < act_k) & row_active
# tok_valid 仅用于: scores = tl.where(tok_valid, scores, -inf)
# 不再用于 K/V 寻址!
```

### 2.3 正确性保证

| 问题 | 处理方式 |
|------|---------|
| sparse_flat 中 -1 的位置 | `clamp(min=0)` → gather K[0]，但 `tok_valid=False` → scores=-inf → P=0 → 不影响结果 |
| causal mask (tok < threshold) | 仍用 sparse_flat 计算 `tok_valid`，mask 逻辑不变 |
| V=K (MLA-absorb) | `v_gathered = k_gathered`，同一 buffer，同一寻址方式 |
| block-wise sparse | `expand_block_indices` 在 host 端已展开为 token 索引，gather 正常工作 |

## 3. Gather 方法对比

对 4 种 gather 方法进行了性能对比测试：

| 方法 | 耗时 (ms) | 说明 |
|------|----------|------|
| **fancy indexing** (`k_flat[sparse_safe]`) | 16.12 | NPU 上效率低，生成大临时 tensor |
| **torch.index_select** (1D flatten) | **1.83** | NPU 硬件优化，最快 |
| torch.gather (expand index) | 1.83 | 与 index_select 相当 |
| index_select + pre-alloc output | 1.85 | 预分配 output buffer，无额外增益 |

**结论**: 选用 `torch.index_select`，比 fancy indexing 快 **8.8x**。

## 4. 性能结果

### 4.1 端到端性能

| 指标 | 基线 (baseline-orig) | 优化后 (pregather-seq-kv) | 改善 |
|------|---------------------|--------------------------|------|
| **test** | 4/4 pass | **4/4 pass** | ✓ 正确性保持 |
| **prof (kernel only)** | 19.66 ms | **13.79 ms** | **-29.9%** |
| **timing (end-to-end)** | 20.53 ms | **17.89 ms** | **-12.9%** |
| gather 开销 | 0 ms | 1.83 ms | — |

### 4.2 性能拆解

```
端到端 = host gather + kernel execution

基线:    0      +  19.66ms  = 19.66ms (prof) / 20.53ms (timing)
优化后:  1.83ms +  13.79ms  = 15.62ms (prof+gather) / 17.89ms (timing)

Kernel 加速: 19.66 → 13.79 = -29.9% (scattered → sequential access)
Gather 开销: +1.83ms (index_select)
净加速:      20.53 → 17.89 = -12.9%
```

### 4.3 Kernel 加速来源

| 优化点 | 估计节省 | 说明 |
|--------|---------|------|
| K/KR scattered→sequential load | ~4 ms | L1 cache miss 减少，MTE2 顺序读取 |
| V scattered→sequential load | ~1.5 ms | 同上 (V=K，同一 buffer) |
| ND2NZ 格式转换消除 | ~0.5 ms | 顺序访问无需重排 |
| **总计** | **~6 ms** | 19.66 → 13.79 |

## 5. 内存开销

| Buffer | Shape | 大小 (bf16) | 说明 |
|--------|-------|------------|------|
| K_gathered | [B*S1, topK, D] = [512, 2048, 512] | 1024 MB | 预 gather 的 K |
| KR_gathered | [B*S1, topK, D_ROPE] = [512, 2048, 64] | 128 MB | 预 gather 的 KR |
| V_gathered | = K_gathered | 0 (共享) | MLA-absorb: V=K |
| **总计额外** | | **~1.1 GB** | NPU 32GB HBM 可接受 |

## 6. Git Tags

| Tag | Commit | 说明 |
|-----|--------|------|
| `baseline-orig` | `f208cb4` | 基线 (orig kernel + K layout fix + test 增强) |
| `plan-a-bg64` | `4793ce3` | 强制 BG=64 (无增益) |
| `plan-b-1d-grid` | `d052da2` | 1D grid + cap (Triton 编译超时，不可行) |
| `pregather-seq-kv` | `3c425eb` | **预 gather + 顺序访问 (当前最优)** |

## 7. 适用条件与限制

### 7.1 适用条件

- **MLA-absorb 模式** (attention_mode=2): V=K，只需 gather 一个 buffer
- **MQA** (N2=1): 所有 query head 共享同一 KV
- **NPU 内存充足**: 额外 ~1.1 GB (对 32GB HBM 无压力)

### 7.2 限制

- **sparse_indices 每次推理可能变化**: gather 必须每次 forward pass 都做，无法缓存
  - 但 gather 开销 (1.83ms) 远小于 kernel 加速 (~6ms)，净收益明显
- **额外内存**: 1.1 GB (K_gathered + KR_gathered)
- **不适用于 batch size 极大的场景**: K_gathered 大小 = B × S1 × topK × D × 2B，随 B 线性增长

## 8. 后续优化方向

1. **重叠 gather 和 kernel**: 用 CUDA stream / NPU stream 让 gather 和前一个 kernel 重叠执行
2. **分批 gather**: 将 [B*S1, topK, D] 分成多批，每批 gather 后立即启动对应 kernel，减少峰值内存
3. **in-place gather**: 如果 K buffer 后续不再使用，可以原地覆盖 (需确保安全性)
4. **kernel 内 double buffer**: 在 kernel 内用 software pipeline 重叠 K gather 和 Q@K 计算

---

> **文档版本**: v1.0
> **基于**: pregather-seq-kv commit (3c425eb)
> **日期**: 2026-06-24
