# MMSE-SIC 接收机集成设计

## 背景与动机

当前系统使用 matched filter 接收（`rate_calc.py` 中 `r_k = h_eff / ||h_eff||`）。在 K=N_r=16 配置下，matched filter 无法有效抑制多用户干扰，导致 PPO-RSMA 在投递率和溢出率上劣于 SDMA 启发式基线。

MMSE-SIC（最小均方误差逐次干扰消除）是 MAC 信道的容量最优线性接收机。集成 MMSE-SIC 将：
1. 大幅提升所有方案的绝对速率（matched filter → MMSE 在 K=N_r 时增益可达 5-10 dB）
2. 放大 RSMA 相对 SDMA 的优势（SIC 增益在 MMSE 接收下更明显）
3. 为发表顶刊提供有竞争力的物理层性能

同时，公共流发射预编码从正交方向（H_k 第 2 奇异向量）改为 MRT（第 1 奇异向量），最大化信道增益。上行 SIC 通过时域分离公共/私有流，不需要空间正交。

## 系统模型修正

### 发射端（用户侧）

用户 k 的发射信号：
```
x_k = w_k * (sqrt(p_c_k) * s_k^c + sqrt(p_p_k) * s_k^p)
```

其中：
- `w_k` = MRT 方向 = H_k 的最大右奇异向量 ∈ C^(N_t,)
- `p_c_k = alpha_k * p_total_k * split_mask_k`（公共流功率）
- `p_p_k = (1 - alpha_k * split_mask_k) * p_total_k`（私有流功率）
- 公共和私有流共享同一空间方向（统一 MRT）

有效信道：
```
h_k = H_k @ w_k ∈ C^(N_r,)    # 所有流共享此有效信道方向
```

### 接收端（BS 侧）— MMSE-SIC

BS 接收信号：
```
y = Σ_k h_k (sqrt(p_c_k) s_k^c + sqrt(p_p_k) s_k^p) + n
```

SIC 解码分两阶段：

**Phase 1: 公共流解码**（按 decoding_order 从小到大）

对第 i 个被解码的公共流（用户 k = π(i)）：

```
R_int_c = Σ_{同组未解码公共流 j} p_c_j h_j h_j^H
        + Σ_{不同组公共流 j} p_c_j h_j h_j^H          # 跨组公共流始终为干扰
        + Σ_{所有用户 j} p_p_j h_j h_j^H               # 所有私有流（Phase 1 未解码）
        + Σ_{同组已SIC公共流 j} β² p_c_j h_j h_j^H     # SIC 硬件残余
        + Σ_{同组已SIC公共流 j} ε_j² p_c_j I_{N_r}     # SIC CSI 误差残余
        + σ² I_{N_r}

v_k^c = R_int_c^{-1} h_k        # MMSE 组合向量（未归一化）
SINR_c_k = p_c_k * |v_k^{cH} h_k|² / (v_k^{cH} R_int_c v_k^c)
R_c_k = log2(1 + SINR_c_k)
```

**Phase 2: 私有流解码**（所有公共流已 SIC 消除后）

对第 i 个被解码的私有流（用户 k = π(i)）：

```
R_int_p = Σ_{未解码私有流 j} p_p_j h_j h_j^H
        + Σ_{已SIC私有流 j} β² p_p_j h_j h_j^H         # 私有流 SIC 残余
        + Σ_{已SIC私有流 j} ε_j² p_p_j I_{N_r}          # CSI 误差残余
        + Σ_{所有公共流 j} β² p_c_j h_j h_j^H            # 公共流 SIC 残余
        + Σ_{所有公共流 j} ε_j² p_c_j I_{N_r}
        + σ² I_{N_r}

v_k^p = R_int_p^{-1} h_k
SINR_p_k = p_p_k * |v_k^{pH} h_k|² / (v_k^{pH} R_int_p v_k^p)
R_p_k = log2(1 + SINR_p_k)
```

**鲁棒模式（imperfect CSI）**：
- MMSE 组合向量使用估计信道 H_hat 计算
- 干扰协方差额外加入 CSI 误差项：`R_int += Σ_k ε_k² p_k I_{N_r}`（worst-case robust MMSE，参考 Vorobyov et al. 2003）
- 实际 SINR 使用真实信道 H_true 验算（用于 outage 判定）

### SINR 公式简化

利用矩阵求逆引理，MMSE SINR 可等价表达为：
```
SINR_k = p_k * h_k^H R_int^{-1} h_k
```
其中 `R_int` 不含信号流 k 本身。这避免了显式计算 `v_k`。

## 代码变更详细设计

### 1. beamforming.py — 新增 `compute_precoding_v2()`

```python
def compute_precoding_v2(self, H, alpha, p_total, split_mask):
    """统一 MRT 发射预编码
    
    Args:
        H: (K, N_r, N_t) complex
        alpha: (K,) float ∈[0,1] - 公共/私有功率分割比
        p_total: (K,) float - 每用户总功率
        split_mask: (K,) float - 分裂掩码 (1=启用公共流)
    
    Returns:
        W_c: (K, N_t) complex - 公共流预编码 (||W_c[k]||² = alpha[k]*p[k]*split[k])
        W_p: (K, N_t) complex - 私有流预编码 (||W_p[k]||² = (1-alpha[k]*split[k])*p[k])
    """
    # 每用户 MRT 方向（最大右奇异向量）
    for k in range(K):
        dir_k = self.mrt_direction(H[k])
        p_c = alpha[k] * p_total[k] * split_mask[k]
        p_p = (1.0 - alpha[k] * split_mask[k]) * p_total[k]
        W_c[k] = dir_k * sqrt(max(p_c, 0))
        W_p[k] = dir_k * sqrt(max(p_p, 0))
    return W_c, W_p
```

### 2. rate_calc.py — 新增 `compute_rates_mmse_sic()`

```python
def compute_rates_mmse_sic(self, H, W_c, W_p,
                            decoding_order=None, group_ids=None,
                            sic_imperfection=0.0, error_bound=0.0):
    """MMSE-SIC 速率计算
    
    Args:
        H: (K, N_r, N_t) complex - 信道矩阵
        W_c: (K, N_t) complex - 公共流预编码
        W_p: (K, N_t) complex - 私有流预编码
        decoding_order: (K,) - 解码优先级（值越小越先解码）
        group_ids: (K,) int - 用户分组
        sic_imperfection: float - SIC 硬件不完美因子 β
        error_bound: float - CSI 误差上界 ε（用于 robust MMSE）
    
    Returns: dict {
        'R_c': (K,), 'R_p': (K,), 'R_total': (K,), 'R_sum': float,
        'SINR_c': (K,), 'SINR_p': (K,)
    }
    """
```

核心实现逻辑：

```python
# Step 1: 计算有效信道
h_c = [H[k] @ W_c[k] for k in range(K)]  # 公共流有效信道
h_p = [H[k] @ W_p[k] for k in range(K)]  # 私有流有效信道
# 提取功率: p_c_k = ||W_c[k]||², p_p_k = ||W_p[k]||²
# 归一化方向: d_c_k = h_c[k] / ||h_c[k]||, d_p_k = h_p[k] / ||h_p[k]||

# Step 2: Phase 1 — 公共流 MMSE-SIC
for i, k in enumerate(sic_order):
    R_int = build_interference_covariance_common(
        k, decoded_common, sic_order, group_ids,
        h_c, h_p, p_c, p_p, sic_imperfection, error_bound, sigma2
    )
    SINR_c[k] = p_c[k] * h_c_dir[k]^H @ inv(R_int) @ h_c_dir[k]
    decoded_common.add(k)

# Step 3: Phase 2 — 私有流 MMSE-SIC
for i, k in enumerate(sic_order):
    R_int = build_interference_covariance_private(
        k, decoded_private, sic_order,
        h_c, h_p, p_c, p_p, sic_imperfection, error_bound, sigma2
    )
    SINR_p[k] = p_p[k] * h_p_dir[k]^H @ inv(R_int) @ h_p_dir[k]
    decoded_private.add(k)
```

**性能优化**：
- 使用 `np.linalg.solve` 替代显式 `inv`（更稳定、更快）
- `R_int` 通过 rank-1 update 增量计算（SIC 每步仅移除一个干扰源）

### 3. rate_calc.py — 鲁棒模式 `compute_outage_rates_mmse_sic()`

```python
def compute_outage_rates_mmse_sic(self, H_hat, H_true, W_c, W_p,
                                    error_bound, ...):
    """鲁棒 MMSE-SIC（imperfect CSI 模式）
    
    1. 用 H_hat 计算 robust MMSE 组合向量（R_int 中加入 ε²·p·I 项）
    2. 用 H_true 计算实际 SINR（outage 判定）
    3. 取 min(declared, actual) 作为可用速率
    """
```

### 4. rsma_env.py — 切换到新路径

```python
# 在 step() 方法中：
# 旧:
# W_c, W_p = self.beamformer.compute_directions(H_hat, alpha, p_total)
# 新:
W_c, W_p = self.beamformer.compute_precoding_v2(
    H_hat, alpha, p_total, self.split_mask
)

# 速率计算同步切换
if self.csi_mode == "imperfect":
    rate_result = self.rate_calc.compute_outage_rates_mmse_sic(
        H_hat, H_true, W_c, W_p,
        error_bound=self.config.system.epsilon_csi,
        decoding_order=self.decoding_order,
        group_ids=self.group_ids,
        sic_imperfection=sic_imp,
    )
else:
    rate_result = self.rate_calc.compute_rates_mmse_sic(
        H_hat, W_c, W_p,
        decoding_order=self.decoding_order,
        group_ids=self.group_ids,
        sic_imperfection=sic_imp,
    )
```

### 5. 不变部分

- `Config` / `SystemConfig`: 无需新增字段（β, ε 已有）
- `LyapunovController`: 不变（只消费 R_sum / served_bits）
- `PPO / Meta / Buffer / Trainer`: 不变（动作空间/观测空间不变）
- Baselines: 通过 env.step() 自动使用新 rate_calc

## 测试计划

| 测试项 | 验证内容 | 方法 |
|--------|---------|------|
| MMSE ≥ MF | MMSE-SIC SINR ≥ matched filter SINR for all users | 同一场景两种方法对比 |
| K=1 退化 | K=1 时 MMSE = MRC = matched filter | 单用户场景精确比对 |
| SIC 因果性 | 先解码用户面临更多干扰 | 检查 SINR 随解码顺序变化 |
| 分组正确性 | 同组 SIC vs 不同组无 SIC | group_ids=[0,0] vs [0,1] 对比 |
| 鲁棒 ≤ 标称 | ε>0 时 robust SINR ≤ nominal SINR | 带/不带 error_bound |
| SIC 不完美 | β>0 → SINR 降低 | 对比 β=0 vs β=0.05 |
| 维度一致 | 返回 dict 与旧接口格式兼容 | 现有 test_rate_calc.py 适配 |
| 数值稳定 | 退化信道、零功率等边界条件无 NaN/Inf | 构造极端场景 |
| 回归测试 | `pytest tests/ -v` 全部通过 | CI |

## 性能预估

- 每时隙计算量：2K 次 `np.linalg.solve(N_r, N_r)` ≈ 32 × O(N_r³)
- K=N_r=16: 32 × 4096 = 131K FLOP/步 → ~0.5ms（numpy vectorized）
- 200 步/episode → ~100ms/episode（当前 matched filter ~30ms/episode）
- 训练速度影响：FPS 从 ~415 降至 ~200-300，可接受

## 不在此次范围

- RZF 接收端集成（`compute_rzf_precoding` 暂不接入，保留后续优化空间）
- 组内公共流预编码方向优化（`compute_common_precoding` 暂不接入）
- Reward 重设计（本次只改物理层，reward 作为独立后续 PR）
- Meta-Controller 调优（依赖 reward 修复后的训练结果）
