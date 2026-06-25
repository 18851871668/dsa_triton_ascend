# Sparse Flash Attention Gradient（SFA Backward）算子逻辑详解

> 基于 `sparse_flash_attention_grad_triton.py` 源码分析

---

## 一、算子概述

### 1.1 功能

这是 Sparse Flash Attention 前向算子的**反向传播（梯度计算）**。给定前向输出的梯度 `d_out`，计算对 `query`、`key`、`value`（及对应的 rope）的梯度。

### 1.2 数学公式

每个 `(b, s1)` 位置，N1 个 query head 共享一组 gathered KV（MQA, N2=1）：

```
# 前向（回顾）
score[h,p] = (q_nope[h] · k_nope[p] + q_rope[h] · k_rope[p]) * scale
P[h,p]     = exp(score[h,p] - softmax_max[h]) / softmax_sum[h]
O[h,d]     = sum_p P[h,p] * V[p,d]

# 反向（本算子）
delta[h]   = sum_d dO[h,d] · O[h,d]                    # rowsum(dO * O)
dS[h,p]    = P[h,p] · (dO[h] · k_nope[p] - delta[h]) · scale   # score 的梯度

# 对 query 的梯度（每个 head 独立累加，无竞争）
d_query[h]      = sum_p dS[h,p] · k_nope[p]
d_query_rope[h] = sum_p dS[h,p] · k_rope[p]

# 对 key 的梯度（多个 s1 可能选同一个 token，需要 scatter-add）
d_key[u]       += sum_h dS[h, p->u] · q_nope[h]
d_key_rope[u]  += sum_h dS[h, p->u] · q_rope[h]

# 对 value 的梯度（P @ dO 路径，scatter-add）
d_value[u]     += sum_h P[h, p->u] · dO[h]
```

其中 `p->u` 表示 sparse 索引 `p` 指向的实际 token 位置 `u`。

### 1.3 MLA-absorb 特性

- `value = key[...,:D]`（V 和 K 共享压缩隐空间）
- kernel 中 `v_ptr` 直接别名 `k_ptr`
- 但 `d_key`（QK 路径）和 `d_value`（P@dO 路径）**分别返回**，调用方自行相加

### 1.4 输入输出

**输入（12 个）**：

| 输入 | 形状 | 说明 |
|------|------|------|
| `query` | `[B,S1,N1,D]` | 前向的 query |
| `key` | `[B,S2,1,D]` | 前向的 key |
| `value` | `[B,S2,1,D]` | 前向的 value（MLA-absorb 下被忽略，V=K） |
| `sparse_indices` | `[B,S1,1,sparse_count]` | 稀疏 token 索引 |
| `d_out` | `[B,S1,N1,D]` | 前向输出的梯度 |
| `out` | `[B,S1,N1,D]` | 前向输出 |
| `softmax_max` | `[B,1,S1,N1]` | 前向 softmax max（fp32） |
| `softmax_sum` | `[B,1,S1,N1]` | 前向 softmax sum（fp32） |
| `query_rope` | `[B,S1,N1,D_ROPE]` | query 的 rope 部分 |
| `key_rope` | `[B,S2,1,D_ROPE]` | key 的 rope 部分 |
| `actual_seq_lengths_query` | `[B]` | 每个batch的实际query长度 |
| `actual_seq_lengths_kv` | `[B]` | 每个batch的实际key长度 |

**输出（5 个）**：

| 输出 | 形状 | 说明 |
|------|------|------|
| `d_query` | `[B,S1,N1,D]` | query 的梯度 |
| `d_key` | `[B,S2,1,D]` | key 的梯度（QK 路径） |
| `d_value` | `[B,S2,1,D]` | value 的梯度（P@dO 路径） |
| `d_query_rope` | `[B,S1,N1,D_ROPE]` | query_rope 的梯度 |
| `d_key_rope` | `[B,S2,1,D_ROPE]` | key_rope 的梯度 |

---

## 二、Kernel 结构

### 2.1 Grid 设计

```python
grid = (_next_pow2(B_S1),)   # 1D grid，每个 program 处理一个 (b, s1) 位置
```

- **1D grid**：每个 program 负责一个 query 行 `(b, s1)`
- pow2 padding 满足 Ascend 要求
- 越界的 program 通过 `bs1_in_range` 掩码空转

### 2.2 Block 配置

```python
def _select_block_config(D, N1):
    block_g = min(16, max(8, N1))   # head block：8 或 16
    return {"BLOCK_G": block_g, "BLOCK_K": 32, "BLOCK_D": 128}
```

- **BLOCK_G**：head 维 block 大小（8 或 16）
- **BLOCK_K**：token 维 block 大小（固定 32）
- **BLOCK_D**：D 维不分块（一次加载完整 D）
- **无 autotune**：固定配置，避免 autotune 重复执行导致 atomic_add 重复累加

### 2.3 三阶段架构

kernel 分为 **3 个阶段**，顺序执行：

```
Stage A: 计算 d_query / d_query_rope（无 atomic_add，每行独占）
    ↓
Stage B1: 计算 d_key / d_key_rope（atomic_add scatter）
    ↓
Stage B2: 计算 d_value（atomic_add scatter）
```

**分阶段的原因**：降低峰值 UB 占用。`dk_acc`/`dkr_acc` 和 `dv_acc` 不需要同时存在。

---

## 三、Stage A：计算 d_query / d_query_rope

### 3.1 目标

```
d_query[h]      = sum_p dS[h,p] · k_nope[p]     # [N1, D]
d_query_rope[h] = sum_p dS[h,p] · k_rope[p]     # [N1, D_ROPE]
```

### 3.2 循环结构

```
for hc in range(HC_LOOP):            # 外层：head chunk 循环
    加载 q_nope, q_rope, do_tile, o_tile, sm_max, sm_sum
    计算 delta = rowsum(dO * O)
    
    for blk_start in range(0, topK, BLOCK_K):  # 内层：token block 循环
        加载 k_full, kr_full
        重算 scores = (q·k + qr·kr) * scale
        重建 P = exp(scores - sm_max) / sm_sum
        计算 dPv = dO · k
        计算 dS = P · (dPv - delta) · scale
        累加 acc_dq  += dS · k
        累加 acc_dqr += dS · kr
    
    store acc_dq → dq_ptr     # 无 atomic（每行独占）
    store acc_dqr → dqr_ptr
```

### 3.3 关键细节

**delta 计算**：
```python
delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
# delta[h] = sum_d dO[h,d] * O[h,d]
# 这是 softmax 反向的核心项，表示"输出梯度对 logit 的总贡献"
```

**dS 计算**：
```python
scores = tl.dot(q_nope, tl.trans(k_full)).to(tl.float32)
scores += tl.dot(q_rope, tl.trans(kr_full)).to(tl.float32)
scores = scores * scale_value

P = tl.exp(scores - sm_max[:, None]) / sm_sum[:, None]
# P 是从保存的 softmax_max/sum 重建的注意力概率

dPv = tl.dot(do_tile, tl.trans(k_full)).to(tl.float32)
# dPv[h,p] = dO[h] · k[p]，即"输出梯度在 key 方向的投影"

dS = P * (dPv - delta[:, None]) * scale_value
# dS[h,p] = P[h,p] * (dPv[h,p] - delta[h]) * scale
# 这是 softmax 反向的标准公式：dScore = P * (dPv - rowsum(dPv*P)) * scale
```

**dq 累加**：
```python
acc_dq += tl.dot(dS.to(k_full.dtype), k_full).to(tl.float32)
# acc_dq[h,d] += sum_p dS[h,p] * k[p,d]
# dS: [BLOCK_G, BLOCK_K], k_full: [BLOCK_K, D]
# dot: [BLOCK_G, D]
```

**为什么无 atomic_add**：
- 每个 `(b, s1)` 由唯一一个 program 处理
- `d_query[b, s1, h, :]` 只被这一个 program 写入
- 不存在跨 program 竞争，直接 `tl.store` 即可

### 3.4 HC_LOOP 技巧

```python
if NUM_HC == 1:
    HC_LOOP: tl.constexpr = 2    # 强制 2 次循环
else:
    HC_LOOP: tl.constexpr = NUM_HC
```

当 `NUM_HC == 1`（N1 <= BLOCK_G）时，Triton 编译器在 `for hc in range(1)` 上会 crash（scf.For assertion failure）。强制设为 2，第 2 次迭代 `g_valid` 全 False，贡献为零，绕过编译器 bug。

---

## 四、Stage B1：计算 d_key / d_key_rope

### 4.1 目标

```
d_key[u]       += sum_h dS[h, p->u] · q_nope[h]    # scatter-add
d_key_rope[u]  += sum_h dS[h, p->u] · q_rope[h]    # scatter-add
```

### 4.2 循环结构

```
for blk_start in range(0, topK, BLOCK_K):   # 外层：token block 循环
    加载 k_full, kr_full（一次，跨所有 head chunk 共享）
    
    dk_acc  = zeros([BLOCK_K, D])
    dkr_acc = zeros([BLOCK_K, D_ROPE])
    
    for hc in range(HC_LOOP):               # 内层：head chunk 循环
        加载 q_nope, q_rope, do_tile, o_tile, sm_max, sm_sum
        计算 delta
        重算 scores, P, dPv, dS（同 Stage A）
        
        dk_acc  += tl.dot(tl.trans(dS), q_nope)    # dK = dS^T · Q
        dkr_acc += tl.dot(tl.trans(dS), q_rope)    # dKR = dS^T · QR
    
    atomic_add(dk_ptr, dk_acc)     # 所有 head chunk 累加后，一次 atomic_add
    atomic_add(dkr_ptr, dkr_acc)
```

### 4.3 与 Stage A 的关键区别

| 特性 | Stage A (dq) | Stage B1 (dk) |
|------|-------------|---------------|
| 循环顺序 | hc 外层，blk_start 内层 | blk_start 外层，hc 内层 |
| 原因 | dq 按 head 独立累加 | dk 需要跨 head 累加后写同一 token |
| 写入方式 | `tl.store`（无竞争） | `tl.atomic_add`（多 s1 竞争） |
| K 加载 | 每个 blk_start 内层加载 | 每个 blk_start 外层加载一次（跨 hc 共享） |

**为什么 blk_start 在外层**：
- `dk_acc` 对应 `BLOCK_K` 个 token
- 需要先累加所有 head chunk 对这批 token 的贡献
- 累加完成后一次 `atomic_add` 写入，减少 atomic 操作次数

**为什么需要 atomic_add**：
- 多个 `(b, s1)` 位置的 sparse 索引可能指向同一个 token `u`
- 例如 `s1=0` 选了 token 5，`s1=1` 也选了 token 5
- 两个 program 都要写 `d_key[token=5]`，必须原子操作

### 4.4 dk 计算公式

```python
dk_acc += tl.dot(tl.trans(dS).to(q_nope.dtype), q_nope).to(tl.float32)
# dS:    [BLOCK_G, BLOCK_K]  — score 的梯度
# trans: [BLOCK_K, BLOCK_G]
# q:     [BLOCK_G, D]
# dot:   [BLOCK_K, D]         — 对 key 的梯度
# 
# dk_acc[p, d] += sum_h dS[h, p] * q[h, d]
# 即 d_key[u] += sum_h dS[h, p->u] * q[h]
```

---

## 五、Stage B2：计算 d_value

### 5.1 目标

```
d_value[u] += sum_h P[h, p->u] · dO[h]    # scatter-add
```

### 5.2 循环结构

```
for blk_start in range(0, topK, BLOCK_K):   # 外层：token block 循环
    加载 k_full, kr_full
    
    dv_acc = zeros([BLOCK_K, D])
    
    for hc in range(HC_LOOP):               # 内层：head chunk 循环
        加载 q_nope, q_rope, do_tile, sm_max, sm_sum
        重算 scores, P（同前）
        
        dv_acc += tl.dot(tl.trans(P), do_tile)    # dV = P^T · dO
    
    atomic_add(dv_ptr, dv_acc)
```

### 5.3 与 Stage B1 的区别

| 特性 | Stage B1 (dk) | Stage B2 (dv) |
|------|-------------|---------------|
| 公式 | `dK = dS^T · Q` | `dV = P^T · dO` |
| 需要 dS | 是 | 否（直接用 P） |
| 需要 delta | 是（算 dS 需要） | 否 |
| 需要 o_tile | 是（算 delta 需要） | 否 |
| UB 占用 | dk_acc + dkr_acc | dv_acc（更小） |

**为什么单独分阶段**：
- `dk_acc [BLOCK_K, D]` + `dkr_acc [BLOCK_K, D_ROPE]` 和 `dv_acc [BLOCK_K, D]` 不需要同时存在
- 分阶段降低峰值 UB 占用

### 5.4 dv 计算公式

```python
dv_acc += tl.dot(tl.trans(P).to(do_tile.dtype), do_tile).to(tl.float32)
# P:      [BLOCK_G, BLOCK_K]  — 注意力概率
# trans:  [BLOCK_K, BLOCK_G]
# dO:     [BLOCK_G, D]
# dot:    [BLOCK_K, D]         — 对 value 的梯度
#
# dv_acc[p, d] += sum_h P[h, p] * dO[h, d]
# 即 d_value[u] += sum_h P[h, p->u] * dO[h]
```

---

## 六、Host 侧处理

### 6.1 布局归一化

```
TND → BSND（PyNative only）
PA_BSND → BSND
block-wise indices → token-wise（_expand_block_indices）
```

### 6.2 数据展平

```python
q_flat      = q_bsnd.contiguous()                          # [B,S1,N1,D]
k_flat      = k_bsnd.reshape(B * S2, D).contiguous()       # [B*S2, D]
v_flat      = k_flat                                        # V=K (MLA-absorb)
sparse_flat = si_tok.reshape(B * S1, topK).to(ms.int32)    # [B*S1, topK]
sm_max_flat = sm_max_bsnd.reshape(B * S1 * N1)             # [B*S1*N1]
sm_sum_flat = sm_sum_bsnd.reshape(B * S1 * N1)             # [B*S1*N1]
```

### 6.3 输出 buffer

```python
dq_buf   = zeros((B, S1, N1, D),       dtype=q.dtype)   # bf16/fp16
dqr_buf  = zeros((B, S1, N1, D_ROPE),  dtype=qr.dtype)  # bf16/fp16
dk_buf   = zeros((B * S2, D),          dtype=fp32)       # fp32 workspace（atomic_add 累加）
dkr_buf  = zeros((B * S2, D_ROPE),     dtype=fp32)       # fp32 workspace
dv_buf   = zeros((B * S2, D),          dtype=fp32)       # fp32 workspace
```

**dk/dkr/dv 用 fp32**：因为 `atomic_add` 多次累加需要高精度防止溢出。最终输出时再转回 bf16/fp16。

### 6.4 无 autotune 的原因

```python
# Fixed block config + single launch (NO autotune)
```

autotune 会多次运行 kernel 做基准测试，但 `dk/dkr/dv` 用 `atomic_add` 累加，多次运行会导致**重复累加**，结果错误。所以使用固定配置单次启动。

---

## 七、数据流总览

```
输入:
  Q [B,S1,N1,D]  ──┐
  QR[B,S1,N1,Dr] ──┤
  K [B*S2,D]     ──┤   ┌─────────────────────────────────┐
  KR[B*S2,Dr]   ──┼──→│         _sfa_grad_kernel          │
  dO[B,S1,N1,D] ──┤   │                                   │
  O [B,S1,N1,D] ──┤   │  Stage A: dq  = Σ dS·k           │
  sm_max[B*S1*N1]─┤   │           dqr = Σ dS·kr          │
  sm_sum[B*S1*N1]─┤   │                                   │
  sparse[B*S1,topK]┘   │  Stage B1: dk  = Σ dS^T·q (atomic)│
                       │           dkr = Σ dS^T·qr(atomic)│
                       │                                   │
                       │  Stage B2: dv  = Σ P^T·dO(atomic)│
                       └─────────────────────────────────┘
                                       │
输出:                                  ↓
  d_query  [B,S1,N1,D]    ← Stage A (tl.store)
  d_qr     [B,S1,N1,Dr]   ← Stage A (tl.store)
  d_key    [B,S2,1,D]     ← Stage B1 (atomic_add, fp32→bf16)
  d_kr     [B,S2,1,Dr]    ← Stage B1 (atomic_add, fp32→bf16)
  d_value  [B,S2,1,D]     ← Stage B2 (atomic_add, fp32→bf16)
```

---

## 八、性能特征

### 8.1 重复计算

scores/P/dS 在三个阶段**各算一遍**：
- Stage A：算 dS 用于 dq
- Stage B1：算 dS 用于 dk
- Stage B2：算 P 用于 dv（不需要 dS）

这是**以计算换 UB** 的设计：不缓存 dS/P（太大会爆 UB），改为重算。

### 8.2 atomic_add 瓶颈

`dk/dkr/dv` 的 `atomic_add` 是性能瓶颈：
- 多个 program 竞争同一 token 位置
- atomic 操作无法并行，串行化严重
- 但通过"先跨 head chunk 累加，再一次 atomic"减少了 atomic 次数

### 8.3 tl.trans 开销

kernel 中有 **10 处** `tl.trans`：
- Stage A：2 处（`trans(k_full)`, `trans(kr_full)`）
- Stage B1：4 处（同上 + `trans(dS)` 两次）
- Stage B2：4 处（同 Stage A + `trans(P)`）

每处 `tl.trans` 在 UB 中产生额外 buffer，增加 UB 压力。

---

## 九、速查表

| 阶段 | 计算 | 循环顺序 | 写入方式 | atomic_add | 重算内容 |
|------|------|---------|---------|:-:|---------|
| **A** | dq, dqr | hc 外, blk 内 | `tl.store` | 否 | scores, P, dS |
| **B1** | dk, dkr | blk 外, hc 内 | `tl.atomic_add` | 是 | scores, P, dS |
| **B2** | dv | blk 外, hc 内 | `tl.atomic_add` | 是 | scores, P |
