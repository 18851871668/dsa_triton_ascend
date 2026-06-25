# Triton 语言内置操作详解

> 本文档基于 `dsa_triton_ascend` 项目实际代码编写，每个操作附带真实示例。

---

## 一、内存操作

### 1. `tl.load` — 从全局内存（GM）加载到统一缓冲区（UB）

**语法**：
```python
tl.load(ptr + offsets, mask=None, other=0.0, care_padding=True)
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ptr + offsets` | tensor | 必填 | 访问地址，支持 1D/2D 广播索引 |
| `mask` | bool tensor | None | `False` 的位置不读取，填 `other` 值 |
| `other` | scalar | 0.0 | mask 为 False 时的填充值 |
| `care_padding` | bool | True | `False` 时允许硬件使用 `index_select_simd`（无 mask 的快速 gather），但要求地址不越界 |

**示例 1：1D 加载**
```python
# 从 GM 加载一段连续数据
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < N
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
# offs = [0, 1, 2, ..., BLOCK-1]（int32）
# mask  = [True, True, ..., False]（尾部越界为 False）
# x     = [x[0], x[1], ..., 0.0]（越界位置填 0）
```

**示例 2：2D 加载（项目中实际用法）**
```python
# 加载 Q tile: [BLOCK_G, BLOCK_D]
q_tile = tl.load(
    q_ptr + q_base + g_offs[:, None] * D + d_offs[None, :],
    mask=g_valid[:, None] & d_valid[None, :],
    other=0.0,
    care_padding=False)

# 地址计算解析：
#   q_ptr        — 基地址指针
#   q_base       — 当前 (b, s1) 的偏移量
#   g_offs[:, None] * D  — 行偏移（每个 head 跨 D 个元素）
#   d_offs[None, :]      — 列偏移（D 维内连续）
# mask 解析：
#   g_valid[:, None]     — [BLOCK_G, 1]，padding head 为 False
#   d_valid[None, :]     — [1, BLOCK_D]，尾部 d 为 False
#   & 运算广播为 [BLOCK_G, BLOCK_D]
```

**示例 3：稀疏 gather 加载**
```python
# tok_clamped 是稀疏索引（每个值指向 GM 中不同的 token 行）
k_tile = tl.load(
    k_ptr + k_base + tok_clamped[:, None] * D + d_offs[None, :],
    mask=tok_valid[:, None] & d_valid[None, :], other=0.0)
# tok_clamped = [3, 17, 5, 0, 42, ...]  — 每行指向不同位置
# 这是随机访问（gather），无空间局部性，可能退化到 scalar 逐元素读
```

**关键点**：
- `care_padding=False` + innermost 维连续 → 硬件用 `index_select_simd`（向量 gather，最快）
- `care_padding=True`（默认）→ 逐元素检查 mask，可能回退 scalar
- innermost 维是稀疏索引时，`index_select_simd` 也无法使用

---

### 2. `tl.store` — 从 UB 写回全局内存

**语法**：
```python
tl.store(ptr + offsets, value, mask=None)
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `ptr + offsets` | tensor | 写入地址 |
| `value` | tensor | 要写入的数据（需匹配 dtype） |
| `mask` | bool tensor | `False` 的位置不写入 |

**示例 1：1D 存储**
```python
# 计算结果写回 GM
y = x * 2.0
tl.store(y_ptr + offs, y, mask=mask)
# 只有 mask=True 的位置才会写入
```

**示例 2：2D 存储（项目中实际用法）**
```python
# P@V 结果写回 out_ptr
out_tile = tl.dot(p_norm.to(v_tile.dtype), v_tile)
tl.store(
    out_ptr + out_base + g_offs[:, None] * D + dv_offs[None, :],
    out_tile.to(out_ptr.dtype.element_ty),
    mask=g_valid[:, None] & dv_valid[None, :] & row_active)

# out_tile.to(out_ptr.dtype.element_ty)
#   — out_ptr.dtype.element_ty 获取指针指向的数据类型（如 tl.bfloat16）
#   — .to() 将 fp32 计算结果转成 bf16 再写入
```

**示例 3：带复合 mask 的存储**
```python
# fp32_acc 写回，需要同时检查 head 有效性和行有效性
tl.store(
    fp32_acc_ptr + fp32_base + g_offs[:, None] * D + dv_offs[None, :],
    acc_dv,
    mask=g_valid[:, None] & dv_valid[None, :] & row_active)
# 三个条件 AND：
#   g_valid   — head 不越界
#   dv_valid  — D 维不越界
#   row_active — 当前行有有效数据（causal mask）
```

---

### 3. `tl.atomic_add` — 原子加

**语法**：
```python
tl.atomic_add(ptr + offsets, value, mask=None)
```

**说明**：多个 program 同时写同一地址时，用原子加保证正确累加。性能比普通 store 慢。

**示例（项目中反向传播用法）**：
```python
# dK 梯度累加：多个 query 位置对同一 key 有梯度贡献
dk_value = tl.dot(tl.trans(dS), q_nope)  # 计算梯度贡献
tl.atomic_add(
    dk_ptr + dk_base + tok_clamped[:, None] * D + d_offs[None, :],
    dk_value.to(dk_ptr.dtype.element_ty),
    mask=tok_valid[:, None] & d_valid[None, :])

# 场景：b=0, s1=0 选了 token 5，b=0, s1=1 也选了 token 5
# 两个 program 都写 dk_ptr[token=5]，必须用 atomic_add
```

---

### 4. `tl.multiple_of` — 对齐提示

**语法**：
```python
ptr = tl.multiple_of(ptr, N)
```

**说明**：告诉编译器指针 `ptr` 是 `N` 字节对齐的。不是真的修改指针，而是编译器 hint，使其生成对齐的向量 load/store 指令。

**示例（项目中连续 10 处对齐声明）**：
```python
# 所有输入/输出指针都声明 128 字节对齐
q_ptr = tl.multiple_of(q_ptr, 128)       # query
qr_ptr = tl.multiple_of(qr_ptr, 128)     # query_rope
k_ptr = tl.multiple_of(k_ptr, 128)       # key
kr_ptr = tl.multiple_of(kr_ptr, 128)     # key_rope
v_ptr = tl.multiple_of(v_ptr, 128)       # value
out_ptr = tl.multiple_of(out_ptr, 128)   # output
fp32_acc_ptr = tl.multiple_of(fp32_acc_ptr, 128)  # fp32 accumulator
sparse_ptr = tl.multiple_of(sparse_ptr, 128)      # sparse indices
sm_max_ptr = tl.multiple_of(sm_max_ptr, 128)      # softmax max
sm_sum_ptr = tl.multiple_of(sm_sum_ptr, 128)      # softmax sum

# 128 字节 = 64 个 bf16 元素
# 对齐后硬件可以使用 128-byte vector load（一次读 64 个 bf16）
```

**关键点**：Ascend 上 128 字节对齐可以使用 `index_select_simd` 等高效指令。如果实际不对齐但声明了对齐，会导致读错数据。

---

## 二、张量构造

### 5. `tl.arange` — 等差数列

**语法**：
```python
tl.arange(start, end)  # 返回 [start, start+1, ..., end-1]
```

**说明**：生成 int32 类型的 1D tensor。是构造 tile 偏移量的基础操作。

**示例 1：基础偏移**
```python
# 生成 [0, 1, 2, ..., 127]
blk_offs = tl.arange(0, BLOCK_K)  # BLOCK_K = 128

# 配合 block 偏移使用
blk_offs = blk_start + tl.arange(0, BLOCK_K)
# blk_start=128, BLOCK_K=128 → [128, 129, ..., 255]
```

**示例 2：构造 2D 索引**
```python
# 行偏移和列偏移
g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)  # [BLOCK_G]
d_offs = d_start + tl.arange(0, BLOCK_D)           # [BLOCK_D]

# 组合成 2D 地址（广播）
addr = g_offs[:, None] * D + d_offs[None, :]       # [BLOCK_G, BLOCK_D]

# g_offs[:, None] = [[0], [1], [2], ...]  — 列向量
# d_offs[None, :] = [[0, 1, 2, ..., BLOCK_D-1]]  — 行向量
# 广播后得到 [BLOCK_G, BLOCK_D] 的 2D 地址矩阵
```

**示例 3：循环内偏移**
```python
# d_start 循环：每次处理 BLOCK_D 个元素
for d_start in range(0, D, BLOCK_D):
    d_offs = d_start + tl.arange(0, BLOCK_D)  # 当前迭代的 D 偏移
    d_valid = d_offs < D                        # 边界 mask
    # d_start=0:   d_offs=[0,1,...,127],   d_valid=[T,T,...,T]
    # d_start=128: d_offs=[128,129,...,255], d_valid=[T,T,...,T]
    # d_start=384: d_offs=[384,385,...,511], d_valid=[T,T,...,T]  (D=512)
```

---

### 6. `tl.zeros` — 零矩阵

**语法**：
```python
tl.zeros(shape, dtype)
```

**示例 1：1D 零向量**
```python
# online softmax 的 sum 初始化
l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
# l_i = [0.0, 0.0, ..., 0.0]  — BLOCK_G 个零
```

**示例 2：2D 零矩阵**
```python
# score 矩阵初始化
scores = tl.zeros([BLOCK_G, BLOCK_K], dtype=tl.float32)
# scores = [[0.0, 0.0, ..., 0.0],
#           [0.0, 0.0, ..., 0.0],
#           ...]
# 后续通过 scores += tl.dot(...) 累加
```

**示例 3：不同 dtype**
```python
# fp32 累加器（高精度）
acc = tl.zeros([BLOCK_G, BLOCK_DV], dtype=tl.float32)

# 注意：tl.zeros 分配的是 UB 空间，不是 GM
# 大 tile 会占用大量 UB，需要配合 _prune_configs 检查
```

---

### 7. `tl.full` — 填充矩阵

**语法**：
```python
tl.full(shape, value, dtype)
```

**示例 1：填充 -inf（attention max 初始化）**
```python
# Flash Attention 的 running max 初始化为 -inf
m_i = tl.full([BLOCK_G], float('-inf'), dtype=tl.float32)
# m_i = [-inf, -inf, ..., -inf]

# 为什么用 -inf：
# 后续 m_new = tl.maximum(m_i, m_blk)
# -inf 确保首次比较时，任何有效的 m_blk 都能胜出
# m_new = maximum(-inf, 3.5) = 3.5
```

**示例 2：填充其他值**
```python
# 填充 1.0（乘法单位元）
scale = tl.full([BLOCK_G, BLOCK_K], 1.0, dtype=tl.float32)

# 填充 -1（无效标记）
neg_inf = tl.full([BLOCK_K], -1, dtype=tl.int32)
```

**与 `tl.zeros` 的区别**：
```python
tl.zeros([M], dtype=tl.float32)      # 只能填 0
tl.full([M], -1.0, dtype=tl.float32) # 可以填任意值
# tl.full(shape, 0, dtype) 等价于 tl.zeros(shape, dtype)
```

---

### 8. `tl.static_range` — 编译期展开循环

**语法**：
```python
for i in tl.static_range(start, end, step):
    ...
```

**说明**：和 Python `range()` 用法相同，但编译器会在编译期完全展开循环体。每个迭代生成独立的指令，无循环开销。

**示例（项目中唯一用法）**：
```python
# 编译期展开，消除循环判断开销
for h_start in tl.static_range(0, N1, BLOCK_H):
    h_offs = h_start + h_local
    h_mask = h_offs < N1
    # ... 处理 BLOCK_H 个 head
# 编译后等价于：
# h_start=0:  处理 head 0..BLOCK_H-1
# h_start=BH: 处理 head BLOCK_H..2*BH-1
# ...（无循环判断，直接顺序执行）
```

**与 `range` 的区别**：

| 特性 | `range()` | `tl.static_range()` |
|------|-----------|---------------------|
| 展开时机 | 运行时循环 | 编译期完全展开 |
| 循环开销 | 有（条件判断+跳转） | 无 |
| 代码体积 | 小 | 大（展开后指令翻倍） |
| 适用场景 | 循环次数多/不确定 | 循环次数少且固定 |

---

## 三、数学运算

### 9. `tl.dot` — 矩阵乘

**语法**：
```python
tl.dot(a, b)               # C = A @ B
tl.dot(a, b, allow_tf32=True)  # 允许 tf32 精度
```

**维度规则**：
```
a: [M, K]
b: [K, N]
result: [M, N]
```

**示例 1：直接矩阵乘**
```python
# Q @ K^T 的 score 计算
# q_tile: [BLOCK_G, BLOCK_D]   — G 个 head, D 维
# k_tile: [BLOCK_D, BLOCK_K]   — D 维, K 个 token（已转置）
scores = tl.dot(q_tile, k_tile)
# result: [BLOCK_G, BLOCK_K]   — G 个 head 对 K 个 token 的 score
```

**示例 2：带 trans 的矩阵乘**
```python
# K 原始布局 [BLOCK_K, BLOCK_D]，需要转置
k_tile = tl.load(k_ptr + ..., ...)  # [BLOCK_K, BLOCK_D]
scores += tl.dot(q_tile, tl.trans(k_tile))
#         q_tile:  [BLOCK_G, BLOCK_D]
#         k_tile:  [BLOCK_K, BLOCK_D]
#         trans:   [BLOCK_D, BLOCK_K]
#         dot:     [BLOCK_G, BLOCK_D] @ [BLOCK_D, BLOCK_K] = [BLOCK_G, BLOCK_K]
```

**示例 3：累加矩阵乘**
```python
# 多轮 d_tile 累加
scores = tl.zeros([BLOCK_G, BLOCK_K], dtype=tl.float32)
for d_start in range(0, D, BLOCK_D):
    q_tile = tl.load(...)  # [BLOCK_G, BLOCK_D]
    k_tile = tl.load(...)  # [BLOCK_K, BLOCK_D]
    scores += tl.dot(q_tile, tl.trans(k_tile))
# 最终 scores = Q @ K^T 的完整结果
```

**Ascend 硬件要点**：
- `tl.dot` 在 Cube 单元执行
- 输入需要从 row-major 转成 fractal 格式（MTE1 单元完成）
- `tl.trans` 会产生额外 UB buffer，增加 UB 压力
- `scores +=` 的累加依赖阻碍跨迭代流水线重叠

---

### 10. `tl.trans` — 转置

**语法**：
```python
tl.trans(x)  # 2D 转置：[M, N] -> [N, M]
```

**示例 1：K 转置用于 Q@K^T**
```python
# K 原始布局：每行一个 token，D 维连续
k_tile = tl.load(
    k_ptr + k_base + tok_clamped[:, None] * D + d_offs[None, :],
    ...)
# k_tile shape: [BLOCK_K, BLOCK_D]
#   行 = token（BLOCK_K 个）
#   列 = D 维（连续）

# 转置后用于 dot
k_t = tl.trans(k_tile)
# k_t shape: [BLOCK_D, BLOCK_K]
#   行 = D 维
#   列 = token（连续）

scores = tl.dot(q_tile, k_t)
#         [BLOCK_G, BLOCK_D] @ [BLOCK_D, BLOCK_K] = [BLOCK_G, BLOCK_K]
```

**示例 2：梯度反向传播中的转置**
```python
# dK = dS^T @ Q
# dS: [BLOCK_G, BLOCK_K]  — 对 score 的梯度
# Q:  [BLOCK_G, BLOCK_D]  — query
# dK: [BLOCK_K, BLOCK_D]  — 对 key 的梯度
dk_acc += tl.dot(tl.trans(dS).to(q_nope.dtype), q_nope)
#          trans(dS): [BLOCK_K, BLOCK_G]
#          Q:          [BLOCK_G, BLOCK_D]
#          dot:        [BLOCK_K, BLOCK_D]
```

**性能影响**：
```
tl.trans 在 UB 中产生额外 buffer：
  原 tile:   BLOCK_K x BLOCK_D x 2 bytes (bf16)
  转置 tile: BLOCK_K x BLOCK_D x 2 bytes (bf16)  <- 额外占用
  合计：     2 x BLOCK_K x BLOCK_D x 2

示例：BLOCK_K=128, BLOCK_D=128
  原 tile:   32KB
  转置 tile: 32KB
  额外占用:  32KB（接近 UB 上限的 1/6）
```

---

### 11. `tl.exp` — 指数

**语法**：
```python
tl.exp(x)  # 逐元素 e^x
```

**示例 1：Softmax 分子**
```python
# Flash Attention online softmax
m_blk_safe = tl.where(m_blk == float('-inf'), 0.0, m_blk)
p_raw = tl.exp(scores - m_blk_safe[:, None])
# scores:    [BLOCK_G, BLOCK_K]  — 原始 score
# m_blk_safe: [BLOCK_G]          — 当前 chunk 的 max
# 减去 max 防止 exp 溢出：
#   exp(100) = 2.7e43  <- 溢出
#   exp(100 - 100) = 1.0  <- 安全
```

**示例 2：Online softmax 缩放因子**
```python
# 旧结果的缩放因子
alpha_old = tl.exp(m_i - m_new_safe)
# m_i:       之前所有 chunk 的 max
# m_new_safe: 合并后的新 max
# alpha_old = exp(old_max - new_max)
#   如果 old_max < new_max -> alpha_old < 1（旧结果缩小）
#   如果 old_max == new_max -> alpha_old = 1（不变）

# 新 chunk 的缩放因子
alpha_new = tl.exp(m_blk - m_new_safe)
# 新 chunk 相对于全局 max 的缩放
```

**示例 3：完整 softmax**
```python
# 标准 softmax
m_i = tl.max(scores, axis=1)           # 求 max
p = tl.exp(scores - m_i[:, None])      # 减 max 后 exp
l_i = tl.sum(p, axis=1)                # 求和
p_norm = p / l_i[:, None]              # 归一化
# p_norm 即 softmax(scores)
```

---

### 12. `tl.log` — 对数

**语法**：
```python
tl.log(x)  # 逐元素自然对数 ln(x)
```

**示例（KL 散度计算）**：
```python
# KL loss = sum(p * log(p/q))
log_p = tl.log(p)          # log(softmax 输出)
log_q = tl.log(q)          # log(teacher distribution)
loss = p * (log_p - log_q) # 逐元素 KL 贡献
loss_total = tl.sum(loss, axis=1)  # 沿 token 维求和
```

**注意**：
- `tl.log(0) = -inf`，需要确保输入 > 0
- 通常配合 `tl.where` 处理零值：`safe_p = tl.where(p > 0, p, 1.0)`

---

### 13. `tl.maximum` — 逐元素取大

**语法**：
```python
tl.maximum(a, b)  # 逐元素取最大值
```

**说明**：**逐元素操作**，输入和输出形状相同（或可广播）。注意与 `tl.max`（归约）区分。

**示例 1：Online softmax 更新 max**
```python
# 合并旧 max 和新 chunk max
m_i = tl.full([BLOCK_G], float('-inf'), dtype=tl.float32)  # 初始化

for blk_start in range(0, topK, BLOCK_K):
    # ... 计算 scores ...
    m_blk = tl.max(scores, axis=1)  # 当前 chunk 的 max（归约）
    m_new = tl.maximum(m_i, m_blk)  # 逐元素取大（合并）
    # m_i:   [BLOCK_G]  — 之前所有 chunk 的全局 max
    # m_blk: [BLOCK_G]  — 当前 chunk 的 max
    # m_new: [BLOCK_G]  — 合并后的新全局 max
    m_i = m_new
```

**示例 2：边界钳位**
```python
# 确保 threshold 不为负
threshold = act_k - act_q + s1 + 1
threshold = tl.maximum(threshold, 0)
# 如果 threshold < 0（query 比 key 长），钳为 0
```

**示例 3：广播**
```python
# 标量和 tensor
m_safe = tl.where(m_i == float('-inf'), 0.0, m_i)
# 等价于：
m_safe = tl.maximum(m_i, float('-inf'))  # 不改变值，但确保类型一致
```

---

### 14. `tl.minimum` — 逐元素取小

**语法**：
```python
tl.minimum(a, b)  # 逐元素取最小值
```

**示例 1：索引钳位**
```python
# 确保 sparse count 不超过实际可用 token 数
s2_real = tl.minimum(topK, tl.maximum(act_k - act_q + s1 + 1, 0))
# act_k - act_q + s1 + 1: causal 窗口大小
# maximum(..., 0):        确保非负
# minimum(topK, ...):     不超过 topK
```

**示例 2：有效范围计算**
```python
s2_bound = tl.minimum(s2_real, VALID_K)
# VALID_K 是编译期常量
# s2_bound 是实际要处理的 token 数
```

---

## 四、归约操作

### 15. `tl.sum` — 沿轴求和

**语法**：
```python
tl.sum(x, axis=N)  # 沿 axis=N 归约求和
```

**说明**：归约后该维度被消除。

**示例 1：Softmax 分母**
```python
# p: [BLOCK_G, BLOCK_K]  — exp 后的概率
l_i = tl.sum(p, axis=1)  # 沿 axis=1（token 维）求和
# p:     [[0.1, 0.2, 0.3, 0.4],    — head 0
#         [0.5, 0.1, 0.2, 0.2]]    — head 1
# l_i:   [1.0, 1.0]                — 每个 head 的 sum
# shape: [BLOCK_G, BLOCK_K] -> [BLOCK_G]
```

**示例 2：梯度 chunk 贡献求和**
```python
# p_raw: [BLOCK_G, BLOCK_K]  — 当前 chunk 的 exp(score - m_blk)
l_chunk = tl.sum(p_raw, axis=1)
# l_chunk: [BLOCK_G]  — 当前 chunk 各 head 的 sum

# 合并到全局 sum
l_i = l_i * alpha_old + l_chunk * alpha_new
# l_i:       之前的全局 sum
# alpha_old: 旧结果缩放因子
# l_chunk:   新 chunk 的 sum
# alpha_new: 新 chunk 缩放因子
```

**示例 3：2D 加权求和**
```python
# v[None, :] * m[:, 1] 逐元素乘，再沿 axis=1 求和
acc += tl.sum(v[None, :] * m, axis=1)
# v: [BLOCK_D]           — 1D 向量
# m: [BLOCK_K, BLOCK_D]  — 2D 矩阵
# v[None, :] 广播为 [1, BLOCK_D]
# 乘积: [BLOCK_K, BLOCK_D]
# sum(axis=1): [BLOCK_K]
```

---

### 16. `tl.max` — 沿轴取最大

**语法**：
```python
tl.max(x, axis=N)  # 沿 axis=N 归约取最大值
```

**说明**：**归约操作**，该维度被消除。注意与 `tl.maximum`（逐元素）区分。

**示例 1：Score 矩阵求 max**
```python
# scores: [BLOCK_G, BLOCK_K]  — Q @ K^T 的结果
m_blk = tl.max(scores, axis=1)  # 每个 head 的最大 score
# scores: [[1.2, 3.5, 0.8, 2.1],   — head 0
#          [0.5, 1.1, 4.3, 0.9]]   — head 1
# m_blk:  [3.5, 4.3]               — 每个 head 的 max
# shape:  [BLOCK_G, BLOCK_K] -> [BLOCK_G]
```

**示例 2：与 -inf mask 配合**
```python
# 无效 token 的 score 设为 -inf
scores = tl.where(tok_valid[None, :], scores, float('-inf'))
m_blk = tl.max(scores, axis=1)
# -inf 位置不会影响 max（任何有效值 > -inf）
# 如果整行都是 -inf（无有效 token），m_blk = -inf
```

**`tl.max` vs `tl.maximum` 对照**：

| 操作 | 类型 | 输入 | 输出 | 示例 |
|------|------|------|------|------|
| `tl.max(x, axis=1)` | 归约 | `[M, N]` | `[M]` | `max([[1,3,2],[4,1,5]], axis=1)` -> `[3, 5]` |
| `tl.maximum(a, b)` | 逐元素 | `[M, N]` + `[M, N]` | `[M, N]` | `maximum([1,3,2], [2,1,4])` -> `[2,3,4]` |

---

## 五、控制流

### 17. `tl.where` — 条件选择

**语法**：
```python
tl.where(condition, a, b)
# condition=True -> 取 a，condition=False -> 取 b
```

**说明**：类似 PyTorch 的 `torch.where`。支持广播。是 Triton 中最常用的掩码操作。

**示例 1：标量条件**
```python
# 越界 program 钳位到 0
pid_bs1 = tl.program_id(0)
bs1_in_range = pid_bs1 < B_S1
pid_bs1 = tl.where(bs1_in_range, pid_bs1, 0)
# 如果 pid 越界（padding program），设为 0
# 后续用 bs1_in_range mask 确保不写出结果
```

**示例 2：1D 条件（处理 -inf）**
```python
# max 为 -inf 时（整行无效），用 0 代替避免 NaN
m_safe = tl.where(m_i == float('-inf'), 0.0, m_i)
# m_i:    [-inf, 3.5, 2.1, -inf]
# m_safe: [0.0,  3.5, 2.1, 0.0]
# 用途：exp(score - m_safe) 时，-inf 会导致 exp(-inf - (-inf)) = exp(0) = 1（错误）
#       改用 0 后：exp(score - 0) = exp(score)，score 也是 -inf -> exp(-inf) = 0
```

**示例 3：2D 广播条件（mask 无效 token）**
```python
# tok_valid: [BLOCK_K]          — 哪些 token 有效
# scores:    [BLOCK_G, BLOCK_K] — score 矩阵

# 方式 1：无效 token score 设为 -inf（softmax 自动忽略）
scores = tl.where(tok_valid[None, :], scores, float('-inf'))
# tok_valid[None, :]: [1, BLOCK_K] -> 广播为 [BLOCK_G, BLOCK_K]
# 无效列全部设为 -inf

# 方式 2：无效 token 概率设为 0（P@V 中贡献为 0）
p = tl.where(tok_valid[None, :], p, 0.0)
# 无效列的概率为 0，dot(p, v) 时不贡献
```

**示例 4：复合条件**
```python
# 多条件组合
store_mask = g_valid & row_active & (l_i > 0.0)
tl.store(sm_max_ptr + sm_base + g_offs, m_i, mask=store_mask)
# 三个条件 AND：
#   g_valid    — head 不越界
#   row_active — 行有有效数据
#   l_i > 0    — sum 大于 0（有有效 token）
```

**示例 5：索引钳位**
```python
# 无效 token 索引钳为 0（防止负地址越界）
tok_clamped = tl.where(tok_valid, tok, 0)
# tok:        [3, 17, -1, 5, -1]  — -1 是无效标记
# tok_valid:  [T,  T,   F, T,  F]
# tok_clamped:[3, 17,   0, 5,   0]
# 地址 = tok_clamped * D -> 0 是合法地址（读到 token 0 的数据）
# 但 tok_valid=False，后续 mask 会清零影响，读到什么无所谓
```

---

## 六、索引与程序标识

### 18. `tl.program_id` — 获取 program ID

**语法**：
```python
tl.program_id(axis)  # 返回当前 program 在 axis 维的索引
```

**说明**：类似 CUDA 的 `blockIdx`。返回值类型为 int32。用于确定当前 program 处理哪个数据块。

**示例 1：1D Grid**
```python
# Grid = (num_programs,)
pid = tl.program_id(0)
# pid = 0, 1, 2, ..., num_programs-1

# 每个 program 处理一段数据
offs = pid * BLOCK + tl.arange(0, BLOCK)
# pid=0: offs=[0, 1, ..., BLOCK-1]
# pid=1: offs=[BLOCK, BLOCK+1, ..., 2*BLOCK-1]
```

**示例 2：2D Grid（项目中用法）**
```python
# Grid = (grid_bs1, grid_g)
pid_bs1 = tl.program_id(0)   # 第 0 维：batch * seq1 索引
pid_g = tl.program_id(1)     # 第 1 维：head group 索引

# 解码 (b, s1)
b = pid_bs1 // S1    # batch 索引
s1 = pid_bs1 % S1    # seq1 索引
# pid_bs1=0: b=0, s1=0
# pid_bs1=1: b=0, s1=1
# pid_bs1=511: b=0, s1=511 (S1=512)

# 计算 head 偏移
g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
# pid_g=0: g_offs=[0, 1, ..., BLOCK_G-1]
# pid_g=1: g_offs=[BLOCK_G, BLOCK_G+1, ..., 2*BLOCK_G-1]
```

**示例 3：越界处理**
```python
pid_bs1 = tl.program_id(0)
bs1_in_range = pid_bs1 < B_S1
# Ascend 要求 grid 每维是 2 的幂
# 如果 B_S1=600，grid padding 到 1024
# pid 600~1023 是空转 program，bs1_in_range=False
pid_bs1 = tl.where(bs1_in_range, pid_bs1, 0)
# 钳为 0 防止地址越界
```

---

## 七、类型与常量

### 19. `tl.constexpr` — 编译期常量

**语法**：
```python
def kernel(..., param: tl.constexpr, ...):
```

**说明**：标记参数在编译期确定值。Triton 为每个不同的 constexpr 组合生成一份编译后的 kernel。`autotune` 的 config 参数必须是 `tl.constexpr`。

**示例 1：Tile 大小参数**
```python
@triton.jit
def _sfa_kernel(
    ...,
    BLOCK_G: tl.constexpr,    # head block 大小
    BLOCK_K: tl.constexpr,    # token block 大小
    BLOCK_D: tl.constexpr,    # D 维 block 大小
    BLOCK_DV: tl.constexpr,   # DV 维 block 大小
    SINGLE_BLOCK: tl.constexpr,  # 是否单 block 路径
):
    # BLOCK_G=8, BLOCK_K=128 -> 编译出一版 kernel
    # BLOCK_G=16, BLOCK_K=64 -> 编译出另一版 kernel
    ...
```

**示例 2：形状参数**
```python
@triton.jit
def _sfa_kernel(
    ...,
    B_S1: tl.constexpr,    # B * S1
    S1: tl.constexpr,      # seq1 长度
    D: tl.constexpr,       # nope 维度
    D_ROPE: tl.constexpr,  # rope 维度
):
    # 这些值在编译期已知，编译器可以做：
    # - 循环展开（range(0, D, BLOCK_D) 中 D 已知）
    # - 常量折叠（D * 2 在编译期计算）
    # - 分支消除（if SINGLE_BLOCK: 在编译期决定）
```

**示例 3：autotune 的 key**
```python
@triton.autotune(
    configs=[...],
    key=["B_S1", "N1", "S2", "topK", "D", "D_ROPE"],
)
@triton.jit
def _sfa_kernel(...):
    ...
# 当 key 中的任意值变化时，触发重新 autotune
# 不同 shape 可能选中不同的最优 config
```

**关键点**：
- `constexpr` 参数变化 -> 重新编译（首次有编译开销）
- 循环边界用 `constexpr` -> 编译器可以展开循环
- `if` 条件用 `constexpr` -> 编译期分支消除（不执行的分支不生成代码）
- autotune 的 config 参数必须是 `constexpr`

---

### 20. `tl.float32` — 浮点类型

**语法**：
```python
tl.float32   # 32 位浮点（单精度）
tl.float16   # 16 位浮点（半精度）
tl.bfloat16  # Brain Float 16
tl.int32     # 32 位有符号整数
tl.int64     # 64 位有符号整数
tl.uint32    # 32 位无符号整数
tl.int8      # 8 位有符号整数
tl.uint8     # 8 位无符号整数
tl.bool_     # 布尔
```

**说明**：项目中主要用 `tl.float32` 做累加精度，输入/输出用 bf16/fp16。

**示例 1：指定 tensor dtype**
```python
# fp32 累加器（高精度，防止溢出）
scores = tl.zeros([BLOCK_G, BLOCK_K], dtype=tl.float32)
m_i = tl.full([BLOCK_G], float('-inf'), dtype=tl.float32)
l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
```

**示例 2：类型转换**
```python
# bf16 -> fp32（计算前提升精度）
q_tile = tl.load(...)  # bf16
scores += tl.dot(q_tile, k_tile)  # dot 内部自动提升到 fp32

# fp32 -> bf16（写回时降低精度）
out_tile = tl.dot(p_norm.to(v_tile.dtype), v_tile)  # 计算时用 bf16
tl.store(out_ptr + ..., out_tile.to(out_ptr.dtype.element_ty))
#     out_ptr.dtype.element_ty -> 获取目标 dtype（如 tl.bfloat16）
#     .to() 将 fp32 结果转成 bf16 写入
```

**示例 3：int32 索引**
```python
# tl.arange 返回 int32
offs = tl.arange(0, BLOCK)  # dtype = tl.int32

# 地址计算自动用 int32
addr = offs * D  # int32 * int32 = int32

# 注意：大地址可能需要 int64
# q_base = (b * S1 + s1) * N1 * D
# 如果 N1=64, D=512, S1=4096, B=8:
#   q_base = (8*4096 + 4095) * 64 * 512 = 1,073,741,824
#   int32 max = 2,147,483,647 <- 仍然安全
# 但 B=16 时会溢出，需要 .to(tl.int64)
```

---

## 八、速查表

| 类别 | 操作 | 语法 | 一句话说明 |
|------|------|------|-----------|
| **内存** | `tl.load` | `tl.load(ptr+off, mask, other)` | GM -> UB，带 mask |
| | `tl.store` | `tl.store(ptr+off, val, mask)` | UB -> GM，带 mask |
| | `tl.atomic_add` | `tl.atomic_add(ptr+off, val, mask)` | 原子加（scatter-add） |
| | `tl.multiple_of` | `ptr = tl.multiple_of(ptr, 128)` | 对齐提示 |
| **构造** | `tl.arange` | `tl.arange(0, N)` | 等差数列 [0..N-1] |
| | `tl.zeros` | `tl.zeros([M,N], dtype)` | 零矩阵 |
| | `tl.full` | `tl.full([M], val, dtype)` | 填充矩阵 |
| | `tl.static_range` | `for i in tl.static_range(0,N,S)` | 编译期展开循环 |
| **数学** | `tl.dot` | `tl.dot(a, b)` | 矩阵乘 |
| | `tl.trans` | `tl.trans(x)` | 转置 [M,N]->[N,M] |
| | `tl.exp` | `tl.exp(x)` | 指数 e^x |
| | `tl.log` | `tl.log(x)` | 自然对数 ln(x) |
| | `tl.maximum` | `tl.maximum(a, b)` | 逐元素取大 |
| | `tl.minimum` | `tl.minimum(a, b)` | 逐元素取小 |
| **归约** | `tl.sum` | `tl.sum(x, axis=1)` | 沿轴求和 |
| | `tl.max` | `tl.max(x, axis=1)` | 沿轴取最大 |
| **控制** | `tl.where` | `tl.where(cond, a, b)` | 条件选择 |
| **索引** | `tl.program_id` | `tl.program_id(0)` | 获取 program ID |
| **类型** | `tl.constexpr` | `param: tl.constexpr` | 编译期常量标注 |
| | `tl.float32` | `dtype=tl.float32` | 32位浮点类型 |
