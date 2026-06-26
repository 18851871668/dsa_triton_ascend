# SFA 反向 Kernel 优化详解（新手版）

> 本文档逐行解释 `sfa_grad_core_standalone.py` 的每处修改，适合 Triton 新手阅读。
> 配合 `sfa_grad_kernel_logic.md`（算子逻辑详解）一起看效果更佳。

---

## 一、优化前的架构（3 阶段串行）

### 1.1 旧架构流程图

```
┌───────────────────────────────────────────────────────────┐
│  Stage A: 计算 dq / dqr                                     │
│  外层循环: for hc (head chunk)                              │
│    内层循环: for blk_start (token block)                    │
│      ① 重算 scores = (q·k^T + qr·kr^T) * scale             │
│      ② 重算 P = softmax(scores)                             │
│      ③ 重算 dS = P * (dPv - delta) * scale                  │
│      ④ 累加 acc_dq  += dS · k                               │
│      ⑤ 累加 acc_dqr += dS · kr                              │
│    写入 dq / dqr（tl.store，无竞争）                         │
├───────────────────────────────────────────────────────────┤
│  Stage B1: 计算 dk / dkr  ← 第 2 遍重算 scores/P/dS         │
│  外层循环: for blk_start                                     │
│    内层循环: for hc                                          │
│      ①~③ 同 Stage A（重复计算！）                            │
│      ⑥ 累加 dk_acc  += dS^T · q                             │
│      ⑦ 累加 dkr_acc += dS^T · qr                            │
│    写入 dk / dkr（tl.atomic_add，有竞争）                    │
├───────────────────────────────────────────────────────────┤
│  Stage B2: 计算 dv  ← 第 3 遍重算 scores/P                  │
│  外层循环: for blk_start                                     │
│    内层循环: for hc                                          │
│      ①② 同 Stage A（重复计算！）                            │
│      ⑧ 累加 dv_acc += P^T · dO                              │
│    写入 dv（tl.atomic_add，有竞争）                          │
└───────────────────────────────────────────────────────────┘
```

### 1.2 旧架构的三个核心问题

**问题 1：scores / P / dS 重复计算 3 遍**

| 阶段 | 重新计算 | 用于 |
|------|---------|------|
| Stage A | scores, P, dS | dq, dqr |
| Stage B1 | scores, P, dS | dk, dkr |
| Stage B2 | scores, P | dv |

每个 `blk_start` 迭代要算 5 个 `tl.dot`（2 个算 scores + 1 个算 dPv + 2 个算 dq/dqr），
3 个阶段共 15 个 dot，其中 10 个是重复的。

**问题 2：atomic_add 硬件级串行**

`dk/dkr/dv` 用 `tl.atomic_add` 写入 GM。当多个 program（不同 `(b,s1)` 位置）
的 sparse 索引指向同一个 token 时，它们竞争写同一地址，硬件必须串行化。

以测试 shape `B=1, S1=512, topK=2048, BLOCK_K=32` 为例：
- Stage B1: 64 个 blk × 2 次 atomic = 128 次
- Stage B2: 64 个 blk × 1 次 atomic = 64 次
- 合计 192 次 atomic_add

**问题 3：三阶段串行执行**

Stage A → B1 → B2 在同一个 program 内顺序执行，前一个必须完全完成后
才能开始下一个，无法重叠。

---

## 二、优化后的架构（1 阶段 kernel + host 侧 bmm）

### 2.1 新架构流程图

```
┌───────────────────────────────────────────────────────────┐
│  Kernel（只跑 1 遍，原来的 Stage A）                        │
│  外层循环: for hc (head chunk)                              │
│    内层循环: for blk_start (token block)                    │
│      ① 算 scores = (q·k^T + qr·kr^T) * scale  （只算1次）  │
│      ② 算 P = softmax(scores)                  （只算1次）  │
│      ③ 算 dS = P * (dPv - delta) * scale       （只算1次）  │
│      ④ 累加 acc_dq  += dS · k                               │
│      ⑤ 累加 acc_dqr += dS · kr                              │
│      ⑥ 写 dS 到 GM workspace（tl.store，无竞争）            │  ← 新增
│      ⑦ 写 P  到 GM workspace（tl.store，无竞争）            │  ← 新增
│    写入 dq / dqr                                            │
└──────────────────┬────────────────────────────────────────┘
                   │
                   │ dS workspace: [B*S1, N1, topK] fp32
                   │ P  workspace: [B*S1, N1, topK] fp32
                   │
                   ▼  kernel 结束后，host 侧用 PyTorch 算子计算
┌───────────────────────────────────────────────────────────┐
│  Host 侧（_sfa_grad_core 函数内）                           │
│                                                             │
│  dk  = bmm(dS^T, Q)   → index_add_ 到 [B*S2, D]           │
│  dkr = bmm(dS^T, QR)  → index_add_ 到 [B*S2, D_ROPE]      │
│  dv  = bmm(P^T,  dO)  → index_add_ 到 [B*S2, D]           │
│                                                             │
│  bmm = 批量矩阵乘，Cube 单元并行计算                         │
│  index_add_ = PyTorch 的 scatter-add，比 atomic_add 高效    │
└───────────────────────────────────────────────────────────┘
```

### 2.2 核心思路

**关键洞察**：dS 和 P 在 Stage A 已经算出来了，只需要存下来，Stage B1/B2 的工作可以
用 host 侧的 `bmm`（矩阵乘）等价完成，不需要在 kernel 里重算。

| 原来 kernel 里做的事 | 现在怎么做的 |
|---------------------|-------------|
| Stage A: dq = Σ dS·k | 不变（kernel 内累加） |
| Stage B1: dk = Σ dS^T·q（atomic_add） | host: `bmm(dS^T, Q)` + `index_add_` |
| Stage B2: dv = Σ P^T·dO（atomic_add） | host: `bmm(P^T, dO)` + `index_add_` |

---

## 三、逐行代码修改详解

### 3.1 `_select_block_config` — BLOCK_K 增大

**改前**：
```python
def _select_block_config(D, N1):
    block_g = min(16, max(8, N1))
    return {"BLOCK_G": block_g, "BLOCK_K": 32}
# topK=2048 → 2048/32 = 64 次循环
```

**改后**：
```python
def _select_block_config(D, N1, topK):           # ← 新增 topK 参数
    block_g = min(16, max(8, N1))
    block_k = 256                                   # ← 从 32 改为 256
    if topK < block_k:                              # ← topK 太小时降级
        block_k = 1 << (topK - 1).bit_length()      # ← 取 topK 的下一个 2 的幂
    return {"BLOCK_G": block_g, "BLOCK_K": block_k}
# topK=2048 → 2048/256 = 8 次循环（原来 64 次）
```

**逐行解释**：

| 代码 | 解释 |
|------|------|
| `def _select_block_config(D, N1, topK):` | 新增 `topK` 参数，因为 BLOCK_K 需要根据 topK 动态选择 |
| `block_k = 256` | 从 32 增大到 256。每次循环处理 256 个 token（原来 32 个） |
| `if topK < block_k:` | 如果 topK 本身比 256 小（如 topK=64），不能用 256 |
| `block_k = 1 << (topK - 1).bit_length()` | 取 topK 的下一个 2 的幂。如 topK=64 → 64，topK=50 → 64 |
| `return {"BLOCK_G": block_g, "BLOCK_K": block_k}` | 去掉了无用的 `"BLOCK_D": 128`（kernel 中 D 不分块） |

**为什么可以增大 BLOCK_K**：
- 改前有 3 个阶段，UB 同时存在 `dk_acc [BK, D]` + `dkr_acc [BK, Dr]` + `dv_acc [BK, D]`，BK=32 时已经接近 UB 上限
- 改后只有 1 个阶段，UB 只有 `acc_dq [BG, D]` + `acc_dqr [BG, Dr]`，BK=256 时 UB 仍然够用
- BLOCK_K 增大 8 倍 → 循环次数减少 8 倍 → 循环开销减少 8 倍

---

### 3.2 kernel 签名修改

**改前**：
```python
def _sfa_grad_kernel(
    q_ptr, qr_ptr,                       # query[B,S1,N1,D], query_rope[B,S1,N1,Dr]
    k_ptr, kr_ptr, v_ptr,                # key/key_rope/value, all [B,S2,1,*] (v aliases k)
    sparse_ptr,                          # token indices [B,S1,1,topK] int32 (block pre-expanded)
    do_ptr, o_ptr,                       # d_out[B,S1,N1,D], out[B,S1,N1,D]
    sm_max_ptr, sm_sum_ptr,              # forward softmax stats, flat (b*S1+s1)*N1 + g
    dq_ptr, dqr_ptr,                     # outputs: d_query, d_query_rope
    dk_ptr, dkr_ptr, dv_ptr,             # outputs (fp32 workspace): d_key, d_key_rope, d_value
    ...
)
```

**改后**：
```python
def _sfa_grad_kernel(
    q_ptr, qr_ptr,                       # 不变
    k_ptr, kr_ptr, v_ptr,                # 不变
    sparse_ptr,                          # 不变
    do_ptr, o_ptr,                       # 不变
    sm_max_ptr, sm_sum_ptr,              # 不变
    dq_ptr, dqr_ptr,                     # 不变
    ds_ptr, dp_ptr,                      # ← 改：dk_ptr/dkr_ptr/dv_ptr → ds_ptr/dp_ptr
    ...
)
```

**逐行解释**：

| 改前参数 | 改后参数 | 解释 |
|---------|---------|------|
| `dk_ptr` | 删除 | dk 不再在 kernel 内计算（改用 host bmm） |
| `dkr_ptr` | 删除 | dkr 不再在 kernel 内计算 |
| `dv_ptr` | 删除 | dv 不再在 kernel 内计算 |
| 无 | `ds_ptr` | 新增：dS 的 GM workspace，`[B*S1, N1, topK]` fp32 |
| 无 | `dp_ptr` | 新增：P 的 GM workspace，`[B*S1, N1, topK]` fp32 |

**为什么**：kernel 不再直接写 dk/dkr/dv（需要 atomic_add），而是写中间结果 dS 和 P（普通 tl.store 无竞争），dk/dkr/dv 在 host 侧用 bmm 算。

---

### 3.3 base offset 修改

**改前**：
```python
sp_base = pid_bs1 * topK
k_rd_base = pid_bs1 * topK * D            # K 读取基址（预 gather 后）
kr_rd_base = pid_bs1 * topK * D_ROPE      # KR 读取基址
dk_wr_base = b * S2 * D                   # dk 写入基址（原始 K 空间）
dkr_wr_base = b * S2 * D_ROPE             # dkr 写入基址
dq_row_base = pid_bs1 * N1 * D            # dq 行基址
dqr_row_base = pid_bs1 * N1 * D_ROPE      # dqr 行基址
```

**改后**：
```python
sp_base = pid_bs1 * topK
k_rd_base = pid_bs1 * topK * D            # 不变
kr_rd_base = pid_bs1 * topK * D_ROPE      # 不变
dq_row_base = pid_bs1 * N1 * D            # 不变
dqr_row_base = pid_bs1 * N1 * D_ROPE      # 不变
ds_base = pid_bs1 * N1 * topK             # ← 新增：dS/P workspace 的行基址
```

**逐行解释**：

| 改前 | 改后 | 解释 |
|------|------|------|
| `dk_wr_base = b * S2 * D` | 删除 | 不再在 kernel 内写 dk |
| `dkr_wr_base = b * S2 * D_ROPE` | 删除 | 不再在 kernel 内写 dkr |
| 无 | `ds_base = pid_bs1 * N1 * topK` | dS/P workspace 形状是 `[B*S1, N1, topK]`，每个 program 写自己那行 |

---

### 3.4 Stage A 内循环：新增 dS/P 写入 GM

**改前**（Stage A 内循环末尾）：
```python
            acc_dq += tl.dot(dS.to(k_full.dtype), k_full).to(tl.float32)
            acc_dqr += tl.dot(dS.to(kr_full.dtype), kr_full).to(tl.float32)
            # ← dS/P 用完就扔，没有保存

    # Store dq/dqr for this head chunk
    tl.store(dq_ptr + ..., acc_dq...)
    tl.store(dqr_ptr + ..., acc_dqr...)
```

**改后**（Stage A 内循环末尾）：
```python
            acc_dq += tl.dot(dS.to(k_full.dtype), k_full).to(tl.float32)
            acc_dqr += tl.dot(dS.to(kr_full.dtype), kr_full).to(tl.float32)

            ds_offs = ds_base + g_offs_s[:, None] * topK + blk_offs[None, :]
            store_mask = g_valid[:, None] & blk_in_count[None, :]
            tl.store(ds_ptr + ds_offs, dS, mask=store_mask)        # ← 新增：存 dS
            tl.store(dp_ptr + ds_offs, P, mask=store_mask)         # ← 新增：存 P

    # Store dq/dqr for this head chunk
    tl.store(dq_ptr + ..., acc_dq...)
    tl.store(dqr_ptr + ..., acc_dqr...)
```

**逐行解释**：

```python
ds_offs = ds_base + g_offs_s[:, None] * topK + blk_offs[None, :]
```
- `ds_base = pid_bs1 * N1 * topK`：当前 program 的行起始偏移
- `g_offs_s[:, None] * topK`：head 维偏移，每个 head 跨 topK 个元素
- `blk_offs[None, :]`：token 维偏移，连续
- 组合得到 2D 地址 `[BLOCK_G, BLOCK_K]`，对应 dS/P 矩阵的一个 tile

```python
store_mask = g_valid[:, None] & blk_in_count[None, :]
```
- `g_valid[:, None]`：head 不越界
- `blk_in_count[None, :]`：token block 不越界（topK 不是 BLOCK_K 整数倍时尾部 padding）

```python
tl.store(ds_ptr + ds_offs, dS, mask=store_mask)
```
- 把 dS `[BLOCK_G, BLOCK_K]`（fp32）写入 GM workspace
- 每个程序写自己独占的行（`pid_bs1 * N1 * topK`），无竞争
- 对比原来的 `tl.atomic_add(dk_ptr + ...)`：无竞争，普通 store 即可

```python
tl.store(dp_ptr + ds_offs, P, mask=store_mask)
```
- 同理，把 P 也存到 GM，供 host 侧算 dv 用

**新手理解**：原来 dS 算完只用一次就扔了，现在把它存到 GM 的一个"暂存区"，kernel 结束后用 PyTorch 的矩阵乘从这个暂存区读出来算 dk/dv/dkr。

---

### 3.5 删除 Stage B1（~60 行）

**改前**（Stage A 之后）：
```python
    # Stage B1: Accumulate dk/dkr across head chunks.
    for blk_start in range(0, topK, BLOCK_K):
        blk_offs = blk_start + blk_k_offs
        blk_in_count = blk_offs < topK
        tok = tl.load(sparse_ptr + sp_base + blk_offs, ...)
        tok_valid = ...
        tok_clamped = tl.where(tok_valid, tok, 0)

        k_full = tl.load(k_ptr + k_rd_base + blk_offs[:, None] * D + ...)
        kr_full = tl.load(kr_ptr + kr_rd_base + blk_offs[:, None] * D_ROPE + ...)

        dk_acc = tl.zeros([BLOCK_K, D], dtype=tl.float32)
        dkr_acc = tl.zeros([BLOCK_K, D_ROPE], dtype=tl.float32)

        for hc in range(HC_LOOP):
            # ... 重新加载 q/qr/do/o/sm_max/sm_sum ...
            # ... 重新计算 delta, scores, P, dPv, dS ...

            dk_acc += tl.dot(tl.trans(dS).to(q_nope.dtype), q_nope).to(tl.float32)
            dkr_acc += tl.dot(tl.trans(dS).to(q_rope.dtype), q_rope).to(tl.float32)

        # atomic_add 写入 dk/dkr
        dk_offs = dk_wr_base + tok_clamped[:, None] * D + d_offs[None, :]
        tl.atomic_add(dk_ptr + dk_offs, dk_acc, mask=tok_valid[:, None])
        dkr_offs = dkr_wr_base + tok_clamped[:, None] * D_ROPE + dr_offs[None, :]
        tl.atomic_add(dkr_ptr + dkr_offs, dkr_acc, mask=tok_valid[:, None])
```

**改后**：**全部删除**。

**解释**：
- Stage B1 做的事：`dk = Σ_h dS^T · q`，`dkr = Σ_h dS^T · qr`
- dS 已经在 Stage A 存到 GM 了
- host 侧用 `torch.bmm(dS^T, Q)` 等价完成，不需要 kernel 重算
- 消除了 64×2 = 128 次 `tl.atomic_add`

---

### 3.6 删除 Stage B2（~50 行）

**改前**（Stage B1 之后）：
```python
    # Stage B2: Accumulate dv across head chunks.
    for blk_start in range(0, topK, BLOCK_K):
        # ... 重新加载 k/kr ...
        # ... 重新计算 scores, P ...

        dv_acc = tl.zeros([BLOCK_K, D], dtype=tl.float32)
        for hc in range(HC_LOOP):
            # ... 重新加载 q/qr/do/sm_max/sm_sum ...
            # ... 重新计算 scores, P ...
            dv_acc += tl.dot(tl.trans(P).to(do_tile.dtype), do_tile).to(tl.float32)

        # atomic_add 写入 dv
        dv_offs = dk_wr_base + tok_clamped[:, None] * D + d_offs[None, :]
        tl.atomic_add(dv_ptr + dv_offs, dv_acc, mask=tok_valid[:, None])
```

**改后**：**全部删除**。

**解释**：
- Stage B2 做的事：`dv = Σ_h P^T · dO`
- P 已经在 Stage A 存到 GM 了
- host 侧用 `torch.bmm(P^T, dO)` 等价完成
- 消除了 64 次 `tl.atomic_add`

---

### 3.7 `_sfa_grad_core` — 新增 host 侧 bmm 计算

**改前**：
```python
def _sfa_grad_core(...):
    cfg = _select_block_config(D, N1)
    block_g = cfg["BLOCK_G"]
    num_hc = triton.cdiv(N1, block_g)
    need_clamp = (N1 % block_g) != 0

    grid = (_next_pow2(B_S1),)

    _sfa_grad_kernel[grid](
        q_flat, qr_flat,
        k_flat, kr_flat, v_flat,
        sparse_flat,
        do_flat, o_flat,
        sm_max_flat, sm_sum_flat,
        dq_buf, dqr_buf,
        dk_buf, dkr_buf, dv_buf,         # ← 传 dk/dkr/dv 给 kernel
        act_q, act_k,
        ...
    )
    return dq_buf, dqr_buf, dk_buf, dkr_buf, dv_buf
```

**改后**：
```python
def _sfa_grad_core(...):
    import torch
    cfg = _select_block_config(D, N1, topK)     # ← 新增 topK 参数
    block_g = cfg["BLOCK_G"]
    num_hc = triton.cdiv(N1, block_g)
    need_clamp = (N1 % block_g) != 0

    grid = (_next_pow2(B_S1),)

    device = q_flat.device
    B = B_S1 // S1

    # ← 新增：分配 dS/P 的 GM workspace
    ds_buf = torch.zeros((B_S1, N1, topK), dtype=torch.float32, device=device)
    dp_buf = torch.zeros((B_S1, N1, topK), dtype=torch.float32, device=device)

    _sfa_grad_kernel[grid](
        q_flat, qr_flat,
        k_flat, kr_flat, v_flat,
        sparse_flat,
        do_flat, o_flat,
        sm_max_flat, sm_sum_flat,
        dq_buf, dqr_buf,
        ds_buf, dp_buf,                   # ← 改：传 ds/dp 而非 dk/dkr/dv
        act_q, act_k,
        ...
    )

    # ← 新增：host 侧用 bmm 计算 dk/dkr/dv
    batch_offsets = torch.arange(B, dtype=torch.int32, device=device) * S2
    sparse_global = sparse_flat.reshape(B, S1, topK) + batch_offsets.reshape(B, 1, 1)
    sparse_1d = sparse_global.reshape(-1).long()

    dk_contrib = torch.bmm(ds_buf.reshape(B_S1, N1, topK).transpose(1, 2),
                           q_flat.reshape(B_S1, N1, D).to(torch.float32))
    dk_buf.index_add_(0, sparse_1d, dk_contrib.reshape(-1, D))

    dv_contrib = torch.bmm(dp_buf.reshape(B_S1, N1, topK).transpose(1, 2),
                           do_flat.reshape(B_S1, N1, D).to(torch.float32))
    dv_buf.index_add_(0, sparse_1d, dv_contrib.reshape(-1, D))

    dkr_contrib = torch.bmm(ds_buf.reshape(B_S1, N1, topK).transpose(1, 2),
                            qr_flat.reshape(B_S1, N1, D_ROPE).to(torch.float32))
    dkr_buf.index_add_(0, sparse_1d, dkr_contrib.reshape(-1, D_ROPE))

    return dq_buf, dqr_buf, dk_buf, dkr_buf, dv_buf
```

**逐行解释**：

```python
import torch
```
- 函数内 import torch（因为模块顶部用了 `from __future__ import annotations`，类型注解不触发 import）

```python
cfg = _select_block_config(D, N1, topK)
```
- 新增 `topK` 参数，用于动态选择 BLOCK_K

```python
ds_buf = torch.zeros((B_S1, N1, topK), dtype=torch.float32, device=device)
dp_buf = torch.zeros((B_S1, N1, topK), dtype=torch.float32, device=device)
```
- 分配 dS 和 P 的 GM workspace
- 形状 `[B*S1, N1, topK]`：每个 (b,s1) 位置有 N1 个 head，每个 head 对 topK 个 token 的 dS/P
- fp32 精度：因为 dS/P 参与后续 bmm 累加，需要高精度

```python
_sfa_grad_kernel[grid](
    ...,
    dq_buf, dqr_buf,
    ds_buf, dp_buf,       # 传 workspace 给 kernel
    ...,
)
```
- kernel 写入 dq/dqr（最终结果）和 ds/dp（中间结果）

```python
batch_offsets = torch.arange(B, dtype=torch.int32, device=device) * S2
sparse_global = sparse_flat.reshape(B, S1, topK) + batch_offsets.reshape(B, 1, 1)
sparse_1d = sparse_global.reshape(-1).long()
```
- 计算 sparse 索引的全局偏移：
  - `sparse_flat` 的值是 `[0, S2-1]`（每个 batch 内的 token 位置）
  - `batch_offsets = [0, S2, 2*S2, ...]`（每个 batch 的起始偏移）
  - `sparse_global = sparse_flat + batch_offsets`：全局 token 位置 `[0, B*S2-1]`
  - `sparse_1d`：展平为 1D，`.long()` 转 int64（index_add_ 要求）

```python
dk_contrib = torch.bmm(ds_buf.reshape(B_S1, N1, topK).transpose(1, 2),
                       q_flat.reshape(B_S1, N1, D).to(torch.float32))
```
- `ds_buf.reshape(B_S1, N1, topK)`：`[B*S1, N1, topK]`
- `.transpose(1, 2)`：`[B*S1, topK, N1]`（dS^T）
- `q_flat.reshape(B_S1, N1, D)`：`[B*S1, N1, D]`
- `.to(torch.float32)`：转 fp32 匹配 dS 精度
- `torch.bmm(dS^T, Q)`：`[B*S1, topK, D]` — 每个 (b,s1) 的 dk 贡献
- 数学等价：`dk_contrib[p, d] = Σ_h dS[h, p] * Q[h, d]`

```python
dk_buf.index_add_(0, sparse_1d, dk_contrib.reshape(-1, D))
```
- `dk_contrib.reshape(-1, D)`：`[B*S1*topK, D]`
- `sparse_1d`：`[B*S1*topK]` — 每行对应的全局 token 位置
- `index_add_(0, sparse_1d, ...)`：按 sparse_1d 索引累加到 `dk_buf [B*S2, D]`
- 数学等价：`dk_buf[u] += Σ_{(b,s1) 选了 u} dk_contrib[(b,s1), p, :]`
- **这就是原来 Stage B1 的 atomic_add 做的事**，但用 PyTorch 高效算子实现

```python
dv_contrib = torch.bmm(dp_buf.reshape(B_S1, N1, topK).transpose(1, 2),
                       do_flat.reshape(B_S1, N1, D).to(torch.float32))
dv_buf.index_add_(0, sparse_1d, dv_contrib.reshape(-1, D))
```
- 同理，`bmm(P^T, dO)` 计算 dv 贡献，`index_add_` scatter 到 dv_buf
- **这就是原来 Stage B2 的 atomic_add 做的事**

```python
dkr_contrib = torch.bmm(ds_buf.reshape(B_S1, N1, topK).transpose(1, 2),
                        qr_flat.reshape(B_S1, N1, D_ROPE).to(torch.float32))
dkr_buf.index_add_(0, sparse_1d, dkr_contrib.reshape(-1, D_ROPE))
```
- 同理，`bmm(dS^T, QR)` 计算 dkr 贡献
- **这就是原来 Stage B1 的 dkr atomic_add 做的事**

---

## 四、效果对比

### 4.1 计算量对比

| 指标 | 改前 | 改后 | 变化 |
|------|------|------|------|
| kernel 内循环次数 | 3 轮 × 64 = 192 | 1 轮 × 8 = 8 | **24 倍↓** |
| kernel 内 dot 数 | ~704 | ~40 | **17 倍↓** |
| kernel 内 atomic_add | 192 次 | 0 | **消除** |
| scores/P/dS 重复计算 | 3 次 | 1 次 | **2/3↓** |
| 额外 GM 写入 | 0 | dS/P workspace | 新增 |
| host 侧 bmm | 0 | 3 个 bmm + 3 个 index_add_ | 新增 |

### 4.2 为什么 host 侧 bmm 比 kernel 内 atomic_add 快

| 特性 | kernel 内 atomic_add | host 侧 bmm + index_add_ |
|------|---------------------|-------------------------|
| 计算方式 | 逐元素原子加，硬件串行 | 矩阵乘，Cube 单元并行 |
| 内存访问 | 随机 scatter（按 sparse 索引） | 连续读 dS/P + 连续写 dk |
| 竞争 | 多 program 写同一地址 | 无竞争（index_add_ 内部优化） |
| 并行度 | 受 atomic 串行限制 | 全核 Cube 并行 |

### 4.3 UB 占用对比

以 `BLOCK_G=16, BLOCK_K=256, D=512, D_ROPE=64` 为例：

| Buffer | 改前（3 阶段） | 改后（1 阶段） |
|--------|-------------|-------------|
| acc_dq `[BG, D]` | 32KB | 32KB |
| acc_dqr `[BG, Dr]` | 4KB | 4KB |
| dk_acc `[BK, D]` | 512KB ← 超限！ | 不存在 |
| dkr_acc `[BK, Dr]` | 64KB | 不存在 |
| dv_acc `[BK, D]` | 512KB ← 超限！ | 不存在 |
| **峰值** | **~600KB**（超 180KB 限制） | **~36KB** |

改前 BLOCK_K=256 会导致 UB 溢出（这就是为什么改前只能用 BLOCK_K=32）。
改后删除了 dk_acc/dkr_acc/dv_acc，UB 峰值仅 36KB，BLOCK_K=256 完全安全。

---

## 五、总结：改了什么、为什么快

| 改动 | 做了什么 | 为什么更快 |
|------|---------|-----------|
| 删除 Stage B1/B2 | 去掉 2/3 的重复计算 | 不再重算 scores/P/dS |
| dS/P 写 GM workspace | 存中间结果供 host 侧使用 | 只需 tl.store（无竞争） |
| host 侧 bmm | 用矩阵乘替代 kernel 内 dot | Cube 全核并行 vs 单核串行 |
| host 侧 index_add_ | 用 PyTorch scatter 替代 atomic_add | 无原子竞争 |
| BLOCK_K 32→256 | 每次循环处理更多 token | 循环次数 64→8，开销减少 8 倍 |

**核心思路一句话**：kernel 只做不可并行的工作（dq/dqr 累加），可并行的工作（dk/dv/dkr 的矩阵乘 + scatter）挪到 host 侧用高效 PyTorch 算子完成。
