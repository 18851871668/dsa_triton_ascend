# SFA Kernel 优化总结

## 1. 优化历程总览

| 阶段 | Tag | prof (ms) | timing (ms) | vs 基线 | 状态 |
|------|-----|-----------|-------------|---------|------|
| 基线 | `baseline-orig` | 19.66 | 20.53 | — | — |
| Grid BG=64 | `plan-a-bg64` | 19.64 | 20.84 | 0% | 无增益 |
| Grid 1D+Cap | `plan-b-1d-grid` | — | — | — | 编译超时 |
| 预 gather + 顺序访问 | `pregather-seq-kv` | 13.79 | 17.89 | -29.9% | ✓ |
| 预转置 K | `pregather-transpose` | 11.74 | 18.83 | 端到端回退 | ✗ |
| **去 tok_valid mask** | **`p1-remove-mask`** | **4.22** | **6.83** | **-69.4%** | **✓ 最优** |

**总加速: 20.53ms → 6.83ms = 3.0x**

## 2. 核心优化方案

### 2.1 预 Gather + 顺序访问 (pregather-seq-kv)

**问题**: kernel 内 `tok_clamped` 是离散随机索引，K/V gather 导致 L1 cache miss 严重 (Cube WAIT_FLAG_DEVI 占 51.7%)

**方案**: host 端用 `torch.index_select` 预 gather K/KR 到连续 buffer，kernel 内用 `blk_offs` 顺序读取

**改动**:
- `sfa_torch_utils.py`: `torch.index_select(k_flat, 0, sparse_1d).reshape(B*S1, topK, D).contiguous()`
- `sfa_core_standalone.py`: K/KR/V 寻址从 `tok_clamped[:, None] * D` 改为 `blk_offs[:, None] * D`，base 从 `b * S2 * D` 改为 `pid_bs1 * topK * D`
- gather 方法对比: fancy indexing 16.12ms → index_select 1.83ms (8.8x 加速)

**收益**: kernel -29.9% (19.66→13.79ms)，端到端 -12.9% (20.53→17.89ms)

### 2.2 去 tok_valid Mask (p1-remove-mask)

**问题**: K/KR/V 的 `tl.load` 带 `mask=tok_valid[:, None] & d_valid[None, :]`，每个 masked load 前编译器生成 `MOV_SPR_XN` 设置 pad 值 (1528 次, 7.21us)

**方案**: 预 gather 后 K/V 在 `blk_offs` 位置数据总是有效，`tok_valid` 仅需用于 `scores = tl.where(tok_valid, scores, -inf)`，不需要在 load 中 mask

**改动**: 将 K/KR/V load 中的 `mask=tok_valid[:, None] & d_valid[None, :]` 改为 `mask=d_valid[None, :]`

**收益**: kernel -69.4% (13.79→4.22ms)，端到端 -61.8% (17.89→6.83ms)

### 2.3 尝试但未采用的方案

| 方案 | 原因 |
|------|------|
| BG=64 强制 | FFTS 开销仅占 3.5%，被 BG 增加的计算量抵消 |
| 1D Grid + Cap | Triton JIT 无法编译 for loop 包裹大 kernel body |
| 预转置 K 消除 tl.trans | kernel -14.9% 但 host 转置 1GB 数据 +6.62ms，净回退 |
| 去 dv_valid mask | NPU vector core 超时 |
| 预计算 g_offs*D | 编译器已自动优化 loop-invariant，无增益 |
| fp32_acc 保留 UB | autotune 不选 BDV=512，if 分支从未生效 |
| alpha 融合 | fp16 精度问题 |

## 3. 最终代码改动 (p1-remove-mask)

### 3.1 sfa_torch_utils.py

```python
# 预 gather K/KR 到连续 buffer
sparse_1d = sparse_flat.clamp(min=0).reshape(-1)
k_gathered = torch.index_select(k_flat, 0, sparse_1d).reshape(B*S1, topK, D).contiguous()
kr_gathered = torch.index_select(kr_flat, 0, sparse_1d).reshape(B*S1, topK, D_ROPE).contiguous()
v_gathered = k_gathered  # MLA-absorb: V=K
```

### 3.2 sfa_core_standalone.py

**base 偏移**:
```python
k_base = pid_bs1 * topK * D      # gathered layout [B*S1, topK, D]
kr_base = pid_bs1 * topK * D_ROPE
v_base = pid_bs1 * topK * D
```

**K/KR/V load (去 tok_valid mask)**:
```python
# 原始: mask=tok_valid[:, None] & d_valid[None, :]
# 优化: mask=d_valid[None, :]  (tok_valid 仅用于 scores masking)
k_tile = tl.load(k_ptr + k_base + blk_offs[:, None] * D + d_offs[None, :],
                 mask=d_valid[None, :], other=0.0)
```

**tok_valid 保留 (仅用于 scores)**:
```python
tok = tl.load(sparse_ptr + sp_base + blk_offs, mask=blk_in_count, other=-1)
tok_valid = blk_in_count & (tok != -1) & (tok < threshold) & (tok < act_k) & row_active
scores = tl.where(tok_valid[None, :], scores, float('-inf'))  # ← 这里仍用 tok_valid
```

## 4. 仿真数据基线

### 4.1 硬件参数 (Ascend910B3)

| 参数 | 值 |
|------|-----|
| AIC 核数 | 24 |
| AIV 核数 | 48 |
| UB 大小 | 192 KB |
| L1 大小 | 512 KB |
| L2 大小 | 720 MB |

### 4.2 优化前 Pipeline 利用率 (orig baseline)

**Vec Core 0 (关键路径, 199 us)**:

| Pipeline | 时间 (us) | 占比 |
|----------|----------|------|
| VECTOR | 86.23 | 43.3% |
| MTE2 | 44.86 | 22.5% |
| MTE3 | 35.90 | 18.0% |
| SCALAR | 24.48 | 12.3% |
| FLOWCTRL | 7.57 | 3.8% |

**Cube Core (70.65 us)**:

| Pipeline | 时间 (us) | 占比 |
|----------|----------|------|
| FLOWCTRL (WAIT_FLAG_DEVI) | 36.53 | 51.7% |
| MTE2 | 9.84 | 13.9% |
| CUBE (MMAD) | 8.29 | 11.7% |
| MTE1 | 7.71 | 10.9% |
| FIXP | 5.40 | 7.6% |
| SCALAR | 2.88 | 4.1% |

### 4.3 Cache Miss 分析

| 来源 | 类型 | 耗时 (us) | 可优化? |
|------|------|----------|---------|
| K/V 稀疏 gather | L1/L2 miss | 36.53 | **是 (预 gather)** |
| fp32_acc GM 往返 | UB "miss" | ~14 | 部分 |
| Q 重复加载 | L0A miss | ~5 | 否 |
| L2 thrashing | L2 miss | ~10 | 否 |

## 5. Git Tags

| Tag | 说明 |
|-----|------|
| `baseline-orig` | 基线 (orig kernel + K layout fix + test 增强) |
| `plan-a-bg64` | BG=64 (无增益) |
| `plan-b-1d-grid` | 1D grid (不可行) |
| `pregather-seq-kv` | 预 gather + 顺序访问 (-29.9%) |
| `pregather-transpose` | 预转置 (端到端回退) |
| **`p1-remove-mask`** | **去 tok_valid mask (-69.4%, 当前最优)** |

## 6. 内存开销

| Buffer | Shape | 大小 (bf16) | 说明 |
|--------|-------|------------|------|
| K_gathered | [B*S1, topK, D] = [512, 2048, 512] | 1024 MB | 预 gather 的 K |
| KR_gathered | [B*S1, topK, D_ROPE] = [512, 2048, 64] | 128 MB | 预 gather 的 KR |
| V_gathered | = K_gathered | 0 (共享) | MLA-absorb: V=K |
| **总计额外** | | **~1.1 GB** | NPU 32GB HBM 可接受 |

## 7. 后续优化方向

1. **重叠 gather 和 kernel**: 用 NPU stream 让 gather 和前一个 kernel 重叠执行
2. **分批 gather**: 将 [B*S1, topK, D] 分成多批，每批 gather 后立即启动对应 kernel
3. **减少 fp32_acc GM 往返**: 需解决 Triton 编译器的 UB spill 问题
4. **仿真验证**: 对 p1-remove-mask 版本运行 msprof 仿真，确认 pipeline 利用率改善

---

> **文档版本**: v1.0
> **基于**: p1-remove-mask tag (bf7b48b)
> **日期**: 2026-06-24
