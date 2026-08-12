# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

上行 RSMA-MU-MIMO **多时间尺度在线学习与资源分配**研究，采用分层强化学习（Hierarchical RL）框架：
- **慢时间尺度**（Meta-Controller, PPO + Self-Attention）：用户分组 g_k ∈ {0,...,G-1}（G=4, Categorical 分布）、分裂用户选取 δ（Bernoulli）
- **快时间尺度**（Worker, PPO）：功率分配、速率分割（d_ratio）、解码顺序优化（Plackett-Luce）

基于 Lyapunov 队列感知混合奖励，在长期约束下最大化能量效率（EE）同时保证队列稳定性。

### 核心研究方向：多时间尺度分层 RSMA 架构

**背景**：大规模 MIMO 系统中，频繁改变用户分组和解码顺序带来极高信令开销，适宜采用多时间尺度控制。

**两层在线决策**：

| 层级 | 决策周期 | 输入 | 输出 | Agent |
|------|---------|------|------|-------|
| 慢尺度 | 每 T_s 个时隙 | per-user 特征 (K×d_user=7) | split_mask δ (K维 Bernoulli) + group_ids g (K维 Categorical, G=4) | Meta-Controller (Self-Attention PPO) |
| 快尺度 | 每个时隙 | obs (10K+1=161 维) | alpha + p_total + d_ratio (3K) + decoding z_scores (K) → env 拼 4K 维 | Worker (MLP PPO) |

**长期目标**：最大化长期平均能量效率（或和速率），同时保证队列稳定。

**RSMA 价值**：
- 用户分组直接影响公共流的复用增益与干扰模式
- 慢变分组可利用 RSMA 的灵活层结构，减少分组改变时的性能震荡
- 建模为分层 MDP：上层动作（分组）影响下层 MDP 的转移概率和奖励

**求解路径**：
- 分层强化学习：上层 Agent 学习最优分组策略，下层 Agent 在每个分组下学习快时间尺度功率与分割策略
- 当前实现：双层 Actor-Critic（Meta = Self-Attention PPO；Worker = MLP PPO + Plackett-Luce 排列）

### 核心问题
- **场景**: 上行多用户 MIMO，采用 RSMA（Rate-Splitting Multiple Access）
- **约束**: 不完美 CSI，worst-case 有界误差模型 ‖Δh_k‖ ≤ ε
- **目标**: 多时间尺度长期资源优化（能量效率最大化、队列稳定性、信息年龄最小化）
- **方法**: 分层 RL（Meta Self-Attention PPO + Worker MLP PPO）+ Lyapunov drift-plus-penalty 保证长期约束

### 参考文献
1. J. Xu and B. Clerckx, "Max-Min Fairness and PHY-Layer Design of Uplink MIMO Rate-Splitting Multiple Access with Finite Blocklength," arXiv:2405.11996, 2024.
2. S. Bi, L. Huang, H. Wang, and Y.-J. A. Zhang, "Lyapunov-guided Deep Reinforcement Learning for Stable Online Computation Offloading in Mobile-Edge Computing Networks," IEEE Trans. Wireless Commun. (TWC), 2021.
3. Q. An et al., "A Deep Reinforcement Learning-Based Resource Scheduler for Massive MIMO Networks," IEEE Trans. Machine Learning in Communications and Networking (TMLCN), 2023.
4. "Distributed Age-of-Information Scheduling With NOMA via Deep Reinforcement Learning," IEEE Trans. Mobile Computing (TMC), vol. 24, no. 1, pp. 30-44, 2025.
5. "Resource Allocation in Multi-User Cellular Networks: A Transformer-based Deep Reinforcement Learning Approach," China Communications, 2024.
6. M. J. Neely, "Stochastic Network Optimization with Application to Communication and Queueing Systems," Morgan & Claypool, 2010.
7. S. Kaul, R. Yates, and M. Gruteser, "Real-time status: How often should one update?," IEEE INFOCOM, 2012.
8. S. A. Vorobyov, A. B. Gershman, and Z.-Q. Luo, "Robust adaptive beamforming using worst-case performance optimization: a solution to the signal mismatch problem," IEEE Trans. Signal Processing, vol. 51, no. 2, pp. 313-324, 2003.

---

## 系统模型 (System Model)

### I. 网络拓扑与基本假设

考虑一个上行多用户 MIMO-RSMA 系统，运行于时间分槽(time-slotted)框架 t ∈ {1, 2, ..., T}：

- **用户数**: K 个用户（用户集合 𝒦 = {1, 2, ..., K}）
- **基站**: 配备 N_r 根接收天线
- **用户发射天线**: 每用户 N_t 根天线（默认 N_t = 2）
- **信道模型**: 准静态块衰落，每时隙内信道恒定，时隙间按 AR(1) 模型演化
- **CSI 不完美**: 基站仅获得估计信道 Ĥ，真实信道 H = Ĥ + ΔH，‖Δh_k‖ ≤ ε_k

### II. 信道模型

#### 时间相关性 (Temporal Correlation)
采用 AR(1) 模型刻画时隙间信道演化：
```
H[t] = ρ · H[t-1] + √(1 - ρ²) · N[t]
```
其中：
- ρ = J₀(2πf_d T_s) 为时间相关系数（Jake's model）
- f_d 为最大多普勒频移，T_s 为时隙间隔
- N[t] ~ CN(0, I) 为独立新息(innovation)过程
- 典型取值 ρ ∈ [0.8, 0.99]

#### CSI 误差模型 (Worst-case Bounded Error)
```
H_true = H_est + ΔH
‖Δh_k‖ ≤ ε_k,  ∀k ∈ 𝒦
```
其中 ε_k 为用户 k 的 CSI 误差上界。误差分布假设：
- ΔH 中每个元素满足有界球约束（deterministic uncertainty set）
- 支持 episode 级随机采样模式：ε ~ Uniform(ε_list) 增强策略泛化能力

#### SNR Domain Randomization（默认开启）
`SystemConfig.snr_domain_randomization=True` 时，env 每个 episode 在 `[snr_min_db, snr_max_db]=[0, 30] dB` 区间重采 σ²，强迫策略学到对 SNR 鲁棒的功率/速率分配规律（关键泛化手段）。

### III. RSMA 消息分割与信号模型

#### 消息分割策略
对于用户 k ∈ 𝒦，其消息 W_k 分为：
- **公共消息 (Common message)**: W_k^c，编码为公共流 s_k^c
- **私有消息 (Private message)**: W_k^p，编码为私有流 s_k^p

分割决策由二值向量 δ ∈ {0,1}^K 控制：
- δ_k = 1: 用户 k 启用 RSMA 分割（同时发送 s_k^c 和 s_k^p）
- δ_k = 0: 用户 k 仅发送私有流 s_k^p（退化为 SDMA）

#### 用户分组 ✅ 已实现（Meta Categorical）

Meta-Controller 同时输出用户分组决策 g_k ∈ {0, ..., G-1}（G=4）：
- **输出方式**: K 维 Categorical 分布（G=4 类），通过 Self-Attention 捕捉用户间信道相关性
- **分组语义**: 同组内公共流互相执行 SIC（干扰消除），不同组公共流视为干扰
- **设计动机**: 信道相关性高的用户倾向分入同组，最大化 SIC 增益
- **Meta 完整输出**: split_mask δ (K维 Bernoulli) + group_ids g (K维 Categorical, G=4)

#### 上行接收信号
```
y[t] = Σ_{k∈𝒦} H_k[t] (√p_k^c · s_k^c + √p_k^p · s_k^p) + n[t]
```
其中：
- H_k ∈ C^(N_r × N_t) 为用户 k 的 MIMO 信道矩阵
- p_k^c, p_k^p 分别为公共流和私有流的发射功率
- 功率约束: p_k^c + p_k^p ≤ P_max,  ∀k（单用户瞬时功率上限）
- n[t] ~ CN(0, σ² I_{N_r}) 为 AWGN 噪声

### IV. SIC 解码与 SINR 模型

#### 解码顺序 ✅ 已实现（Plackett-Luce）
基站使用逐次干扰消除(SIC)解码。解码顺序 π 由 Worker 通过 **Plackett-Luce (PL) 分布** 采样确定：

**Worker 输出**: z ∈ R^K（经 softplus 保证正值）

**采样方式**: Gumbel-Top-K trick
```
π = argsort(-(log z + Gumbel))
```
其中 Gumbel ~ -log(-log(Uniform(0,1)))

**对数概率**（用于PPO策略梯度计算）：
```
log P(π|z) = Σ_{i=1}^{K} [log z_{π(i)} - log(Σ_{j≥i} z_{π(j)})]
```

**解码顺序作用**: π 传入 rate_calc，决定各用户的干扰集合和 SINR 计算中的 SIC 顺序

**关键性质**：
- log P(π|z) 对 z 完全可微 → PPO 策略梯度正常反向传播
- Gumbel-Top-K 等价于逐步 Categorical 采样，但计算更高效
- 温度参数 pl_temperature (默认1.0) 控制探索强度
- 退化特性：当温度→0 时，等价于 argsort(-z)（确定性贪心）
- 同组内用户公共流按 π 执行 SIC，不同组公共流视为干扰

#### MMSE-鲁棒组合矩阵 (Robust Combiner)
```
W = (Ĥ Ĥ^H + Σ_error + σ² I_{N_r})^{-1} Ĥ
其中：
Σ_error = Σ_{k=1}^K ε_k² · (ĥ_k ĥ_k^H / ‖ĥ_k‖²)  ∈ C^(N_r × N_r)

当误差界相同(ε_k = ε, ∀k)且信道独立同分布时，近似为：
Σ_error ≈ ε² · I_{N_r}
即 W = (Ĥ Ĥ^H + ε² I_{N_r} + σ² I_{N_r})^{-1} Ĥ

注：Σ_error 维度为 (N_r × N_r)，与 ĤĤ^H 维度一致，确保矩阵求逆运算合法。鲁棒 MMSE 设计参考 Vorobyov et al. (2003)。
W ∈ C^(N_r × K) 为组合矩阵（当 N_t=1 时每用户一个组合向量）
```

#### 流级 SINR 表达式
对于解码顺序中第 m 个流 s_π(m)：
```
Γ_{π(m)} = |w_{π(m)}^H ĥ_{π(m)}|² · p_{π(m)} / (I_undecoded + I_SIC + σ²)
```
其中：
- I_undecoded = Σ_{j: π(j)>π(m)} p_j |w_{π(m)}^H ĥ_j|² （未解码流干扰，基于估计信道）
- I_SIC = Σ_{j: π(j)<π(m)} (β² |w_{π(m)}^H ĥ_j|² + ε_j² ‖w_{π(m)}‖²) p_j （SIC 残余干扰）
  - 第一项 β²|w^H ĥ_j|²p_j: 硬件/处理不完美导致的残余（即使 CSI 完美也存在）
  - 第二项 ε_j²‖w‖²p_j: CSI 误差导致信号重建不准确的残余（worst-case 上界 |w^H Δh_j|² ≤ ε²‖w‖²）
- σ² 为热噪声功率

**设计要点**：
- CSI 误差对 SINR 的影响通过两个途径体现：(1) SIC 残余中的信号重建误差；(2) 鲁棒 MMSE 组合器设计（W 的计算已纳入 Σ_error）
- 不单独设立 I_CSI 项，避免重复计算
- 当 CSI 完美（ε=0）时，I_SIC 退化为纯硬件残余 β²Σp_j|w^H h_j|²

**重要物理参数区分**：
- β (sic_imperfection): SIC 硬件/处理不完美因子，典型 β ∈ [0, 0.1]，即使 CSI 完美也存在
- ε (error_bound): CSI 估计误差上界，典型 ε ∈ [0.05, 0.5]，影响 SIC 信号重建精度和组合器设计
- 两者为独立物理机制：β 反映接收机硬件限制，ε 反映信道估计精度
- 两者在 SIC 残余中叠加但物理含义不同，不可合并为单一参数

#### 用户可达速率
用户 k 的总速率为其所有流速率之和：
```
R_k = R_k^c + R_k^p = log₂(1 + Γ_k^c) + log₂(1 + Γ_k^p)
```

系统和速率：
```
R_sum = Σ_{k∈𝒦} R_k
```

### V. 流量模型与队列动态

#### 数据到达模型
用户 k 在时隙 t 的数据到达量：
```
A_k[t] ~ Poisson(λ_k)    [默认: 泊松到达]
```

**可选扩展 - ON/OFF MMPP (突发流量)**:
```
State_k[t] ∈ {OFF, ON}
P(OFF → ON) = p_on,  P(ON → OFF) = p_off
A_k[t] ~ Poisson(λ_high)  if State_k = ON
A_k[t] ~ Poisson(λ_low)   if State_k = OFF
稳态平均: π_ON · λ_high + (1 - π_ON) · λ_low = λ_k
```

#### 队列演化方程 (Queue Dynamics)
```
Q_k[t+1] = min{max{Q_k[t] - R_k[t], 0} + A_k[t], Q_max}
```
其中：
- Q_k[t] 为用户 k 在时隙 t 的队列积压（单位: bits，`packet_size=1.0` 归一化，数值上 1 packet ≡ 1 bit）
- R_k[t] 为时隙 t 的服务速率
- Q_max=100.0 为有限缓冲区大小（bits），超出部分溢出(overflow)
- 默认 λ=2.5 packets/slot/user，与 Q_max 匹配下属"中等负载"工况

队列变化量（用于 Lyapunov 分析）：
```
ΔQ_k[t] = Q_k[t+1] - Q_k[t]
```

#### 信息年龄 (AoI) [可选扩展，当前系统由队列积压指标覆盖]
> AoI 与队列积压高度相关：队列稳定自动保证数据新鲜度。当前实现中不单独追踪 AoI，队列指标已充分反映时效性。如需独立 AoI 约束，可参考 Kaul et al. (INFOCOM 2012) 标准定义。

### VI. Lyapunov 优化框架

#### 长期优化问题
```
max   lim_{T→∞} (1/T) Σ_{t=1}^T E[R_sum[t]]      (最大化长期平均吞吐)
s.t.  lim_{T→∞} (1/T) Σ_{t=1}^T E[Q_k[t]] < ∞,  ∀k   (队列稳定性 - Strong stability)
      lim_{T→∞} (1/T) Σ_{t=1}^T E[p_k[t]] ≤ P_avg, ∀k  (长期功率约束)
      p_k^c[t] + p_k^p[t] ≤ P_max, ∀k,t                 (瞬时功率约束)
```

#### Lyapunov 函数与漂移
定义加权 Lyapunov 函数：
```
L[t] = (1/2) Σ_k (Q_k[t])² + (1/2) Σ_k (Z_k[t])²
```
其中 Z_k[t] 为能量虚拟队列：
```
Z_k[t+1] = max{Z_k[t] + p_k[t] - P_avg, 0}
```

Lyapunov 条件漂移(drift)：
```
Δ_L[t] = E[L[t+1] - L[t] | Q[t], Z[t]]
```

#### Drift-Plus-Penalty (DPP)
每时隙最小化 DPP 上界：
```
min  Δ_L[t] - V · E[R_sum[t]]
```
其中 V > 0 控制 "吞吐最优性 vs 队列稳定性" 权衡。

**自适应 Lyapunov 权重**:
```
V_dyn[t] = max_k{Q_k[t]} / Q_max
```
队列越满 → V_dyn 越大 → 稳定性约束权重加强。

---

## MDP 建模 (Markov Decision Process)

> 注：以下为**当前代码实际**实现。Env 的 `action_dim` 在 `RSMAEnergyEnv.__init__` 中固定为 `3K=48`，但 `step()` 同时接受 `3K` 与 `4K` 两种输入；`scripts/train.py → src/training/main.py` 中 Worker 强制走 `4K=64`（拼上 K 维 decoding_order）。

### 状态空间 S（obs_dim = 10K + 1 = 161）

env 把 obs 拼成 10 个 K 维特征块 + 1 维时间比（实现见 `src/env/rsma_env.py`）：

```
obs[t] = [
  h_gains_norm,        # (K,)  归一化信道增益
  queue_norm,          # (K,)  Q_k / Q_max
  delta_queue_norm,    # (K,)  ΔQ_k / Q_max
  arrivals_norm,       # (K,)  到达率 (归一化)
  power_history,       # (K,)  上一时隙归一化总功率
  rate_history,        # (K,)  上一时隙归一化速率
  interference_feat,   # (K,)  干扰特征
  fairness_feat,       # (K,)  公平性特征
  split_mask,          # (K,)  Meta 注入的分裂决策
  group_ids_norm,      # (K,)  Meta 注入的分组 (g/(G-1))
  time_ratio,          # (1,)  t / T
]
# 总维度: 10K + 1 = 161 (K=16)
# 注: 不含原始复信道 H 矩阵；信道信息只用聚合统计 h_gains_norm + interference_feat
# 注: 启用 power_budget 通道时再 +1 (见 rsma_env.py:86-93)
```

**Meta per-user 特征** (`env.get_user_features()` 返回 `(K, d_user=7)`)：包含 h_gain、队列、到达、上一段动作历史、归一化分组 ID 等。

### 动作空间 A（env 默认 3K，训练入口拼 4K）

**Env 接受的 action**:
```python
action_3K = [alpha (K), p_total (K), d_ratio (K)]       # 默认 action_dim
action_4K = [alpha (K), p_total (K), d_ratio (K), decoding_order (K)]  # 训练时使用
```

- env 在 `step()` 中按 `action.shape[0]` 自动判别 3K / 4K；4K 时尾部 K 维被解析为整数解码序
- `src/training/main.py:75` 中 `worker_action_dim = 4 * K` 硬编码，**当前训练入口不支持回退到纯 3K 模式**

**Worker (ActorCritic with `use_plackett_luce=True`) 实际输出**:
```
alpha_raw    ∈ R^K  → sigmoid                                  # 公/私功率比
p_total_raw  ∈ R^K  → sigmoid × P_max                          # 总功率
d_ratio_raw  ∈ R^K  → sigmoid                                  # drain ratio
z_scores     ∈ R^K  → softplus → Gumbel-Top-K → π             # 解码序排列
```

**动作后处理**:
1. 功率裁剪: p_k^c = alpha_k × p_total_k × δ_k; p_k^p = (1 - alpha_k) × p_total_k; 总功率超 P_max 时按比例缩放
2. 解码顺序: z 经 softplus 保证正值，通过 Gumbel-Top-K trick 采样排列 π；log P(π|z) 完全可微
3. 推理（确定性）可取 argsort(-z) 贪心解码

### 奖励函数 R（Worker）

`LyapunovController.compute_reward` 实际计算（见 `src/optimization/lyapunov.py:199-201`），分层混合 reward (paper_ee_v2):

```
w = (avg_queue_ratio)^blend_alpha                      # 队列紧迫度权重 (0~1)
reward = reward_scale × [ w · drift_signal + (1-w) · ee_bonus - overflow_penalty ]
其中 overflow_penalty = w_overflow · Σ OV
```

- **drift_signal**: Lyapunov drift-plus-penalty 项（队列高时主导，强清队列）
- **ee_bonus**: η_EE = R_sum / max(P_total, ee_power_floor)（队列空时主导，省电）
- **w_overflow=12.0 (RewardConfig)** / 5.0 (LyapunovController 默认)：两套默认值并存，最终以 `Config().reward.w_overflow` 经 `rsma_env.py:165` 注入 LyapunovController 为准
- **reward_scale=0.01**: 全局缩放（稳定 PPO value）
- ⚠️ **`RewardConfig.w_outage=3.0` 与 `w_power_exceed=10.0` 是死字段**：`LyapunovController.__init__` 接收 `w_outage` 但 `compute_reward` 中未使用；`w_power_exceed` 在 lyapunov.py 中根本未出现。若需启用 outage / power_exceed 惩罚，需要先把对应项加回 `compute_reward`。

设计直觉：高 SNR → 队列快清空 → w→0 → 强 EE 优化；低 SNR → 队列积压 → w→1 → 加大功率清队列。

### 折扣因子与回合设计
- **折扣因子**: γ = 0.99（长期视野）
- **回合长度**: T 个时隙（默认 T = 200）
- **回合终止**: 达到 T 时隙 或 所有用户队列溢出

---

## 分层快慢尺度训练流程（src/training/main.py）

### 控制流概览

```
reset env → 立即 _meta_decide（注入 δ + g 到 env）
loop rollout_steps 次:
    Worker.act(obs) → (3K 连续, K decoding_order)
    env.step(4K) → reward, info
    worker_buffer.add(...)
    累积 meta_stats: ee_sum += info['ee_bonus'], overflow_sum += info['overflow_penalty']

    if done:                                     # episode 终止
        闭合本段 → meta_buffer.store(...)
        env.reset(); 立即 _meta_decide(...)
    elif env.step_in_episode % T_s == 0:        # T_s 段边界
        闭合本段 → meta_buffer.store(...)
        立即 _meta_decide(...)

Worker.update()                                  # 每个 rollout 后立即触发 PPO
if meta_buffer.size >= meta_update_threshold:    # 默认 50
    Meta.update(); meta_buffer.reset()
```

### Meta-Controller（Self-Attention PPO）

- **网络**（`MetaActorCritic`）：
  - `embed(d_user=7 → hidden=64) + LN + ReLU`
  - `MultiHeadAttention(heads=4)` + 残差 + LN
  - FFN + 残差 + LN
  - **三个 head**：split (per-user 1 logit → Bernoulli) / group (per-user G logits → Categorical) / critic (`mean pool → 1`)
- **采样**：`split_mask = Bernoulli.sample()`、`group_ids = Categorical.sample()`；`log_prob = split.log_prob.sum() + group.log_prob.sum()`
- **决策注入**：`env.set_split_mask(δ)` + `env.set_group_ids(g)`
- **更新**：`MetaPPOTrainer` 走 PPO clip + GAE；split / group **两路熵正则独立加权**（`entropy_coef=0.05`, `group_entropy_coef=0.12`），梯度裁剪 0.5
- **探索**：**无 ε-greedy**，靠两路熵正则；旧版 CLAUDE.md 中提到的 `meta_epsilon_start/end` 字段已删除
- **Buffer**：`MetaRolloutBuffer` 是 FIFO（`max_size=500`），积累到 `meta_update_threshold=50` 触发更新，更新后 `reset()`

### Meta 奖励（src/optimization/meta_reward.py）

```
avg_ee       = meta_stats['ee_sum']       / steps
avg_overflow = meta_stats['overflow_sum'] / steps
group_div    = n_unique(group_ids) / n_groups   ∈ [1/G, 1]

meta_reward  = avg_ee - 2.0 · avg_overflow + 0.5 · group_div
```

最后一项 `0.5 · group_div` 是**分组多样性奖励**，防止策略坍缩到所有用户都用同一组。如需关闭，把 `compute_meta_reward` 中那一项注掉即可。

### Worker (MLP PPO + PL)

- `ActorCritic(use_plackett_luce=True)`，hidden_dim=256, n_layers=3
- buffer: 连续 3K + 额外 K 维 decoding_order
- 标准 PPO clip + GAE + 熵正则 + KL 早停 + LR 衰减（见 `PPOConfig.target_kl` 等）

---

## 优化变量总结

> 注: 用户分裂决策 δ ∈ {0,1}^K 和用户分组 g_k ∈ {0,...,G-1} 由慢时间尺度 Meta-Controller (Self-Attention PPO) 决定，不属于 Worker 优化变量。

| 变量类型 | 符号 | 描述 | 性质 | 决策方式 |
|---------|------|------|------|--------|
| 总功率 | p_total ∈ [0,P_max]^K | 各用户总发射功率 | 连续 | Worker sigmoid + 缩放 |
| 功率分割比 | α ∈ [0,1]^K | 公共/私有功率比 | 连续 | Worker sigmoid, 受 split_mask 约束 |
| Drain 比例 | d_ratio ∈ [0,1]^K | 队列排派比例 | 连续 | Worker sigmoid |
| 解码顺序 | π ∈ Perm(K) | SIC 解码排列 | 离散排序 | Worker z-scores → Plackett-Luce (Gumbel-Top-K) |
| 用户分组 | g ∈ {0,...,G-1}^K | 用户分组分配 (G=4) | 离散 | Meta Categorical |
| 分裂决策 | δ ∈ {0,1}^K | RSMA 分裂选择 | 离散 | Meta Bernoulli |
| 预编码 | W ∈ C^(N_r × K) | env 实际调用 `compute_directions` (MRT 路径) | 解析解（非 RL） | RZF + 分组公共流已实现于 beamforming.py 但**尚未集成到 env** |

---

## 技术路线

### 算法框架: 多时间尺度分层 RL + Lyapunov

```
整体架构:
┌─────────────────────────────────────────────────────┐
│  慢时间尺度 Meta-Controller (每 T_s=20 时隙)         │
│  ├── Self-Attention PPO (embed+MHA+FFN, hidden=64)   │
│  ├── 输入: per-user 特征 (K × d_user=7)             │
│  ├── 输出: split_mask (Bernoulli) + group_ids (Cat) │
│  ├── 探索: 双路熵正则 (split 0.05 + group 0.12)     │
│  └── 奖励: avg_ee - 2·overflow + 0.5·group_div      │
├─────────────────────────────────────────────────────┤
│  快时间尺度 Worker (每时隙, 4K=64 维 action)         │
│  ├── PPO Actor-Critic (MLP, hidden_dim=256)          │
│  ├── Actor: alpha + p_total + d_ratio + z_scores     │
│  ├── z_scores → Plackett-Luce (Gumbel-Top-K) → π    │
│  ├── Critic: V(s) 状态价值估计                       │
│  └── 在 (δ, g) 约束下优化功率/速率/解码序            │
├─────────────────────────────────────────────────────┤
│  Lyapunov 优化层 (paper_ee_v2 分层混合 reward)       │
│  ├── 能量虚拟队列: Z_k[t+1] = max(Z+p-P_avg, 0)     │
│  ├── 队列感知混合: w·drift + (1-w)·ee_bonus         │
│  ├── 自适应权重: w = (avg_queue_ratio)^blend_alpha   │
│  └── 自适应 V: V_dyn ∈ [V_min, V_max], exponent=1.5  │
├─────────────────────────────────────────────────────┤
│  预编码/波束赋形模块                                │
│  ├── 当前 env 走 compute_directions (MRT)            │
│  ├── compute_rzf_precoding 已实现 (待集成)           │
│  ├── compute_common_precoding (分组 MRT) 已实现      │
│  └── 功率约束裁剪 & 归一化                           │
└─────────────────────────────────────────────────────┘
```

### 关键设计决策
1. **多时间尺度分层**: 慢尺度决定分组框架（低信令开销），快尺度在约束下即时优化
2. **Lyapunov + RL 混合**: Lyapunov 保证长期约束满足（队列稳定、功率预算），RL 学习即时策略
3. **Plackett-Luce 解码序**: ✅ K维 z-scores + Gumbel-Top-K，可微且高效
4. **用户分组**: ✅ Meta Categorical (G=4)，Self-Attention 捕捉信道相关性
5. **预编码**: ⚠️ env 当前用旧版 MRT (`compute_directions`)；RZF / 分组公共流方法已写但**未接入 env**
6. **闭式解降维**: 鲁棒 MMSE 组合矩阵走解析解，不经过 RL 决策
7. **自适应权重**: 队列感知混合权重随系统状态动态调整
8. **SNR Domain Randomization**: 每 episode 随机 σ²，泛化训练
9. **分层 MDP 建模**: 上层动作影响下层转移概率和奖励

### 预编码集成状态（待办）

**已实现于 `src/env/beamforming.py`**:
```
# RZF (Regularized Zero-Forcing)
W_RZF = Ĥ^H (ĤĤ^H + α I)^{-1},  α = K σ² / P_avg

# 分组公共流 (per-group MRT)
w_g = normalize(Σ_{k ∈ group_g} ĥ_k)
```

**当前 env 实际调用**：`self.beamformer.compute_directions(H_hat, alpha, p_total)`（旧 MRT 路径，见 `rsma_env.py:371`）

**TODO**：在 `RSMAEnergyEnv._compute_rates` 中切换为 `compute_rzf_precoding` + `compute_common_precoding(group_ids)`，配套修改 baseline 评估。

---

## 配置参数（src/config.py）

### SystemConfig (系统物理参数)
| 参数 | 符号 | 默认值 | 说明 |
|------|------|--------|------|
| K | K | 16 | 用户数 |
| N_r | N_r | 16 | BS 接收天线数 |
| N_t | N_t | 2 | 每用户发射天线数 |
| T | T | 200 | 每回合时隙数 |
| tau | tau | 1.0 | 时隙时长 (ms, 归一化) |
| P_max | P_k_max | 1.0 | 单用户瞬时最大功率 (W) |
| σ² | sigma2 | 0.01 | 噪声功率（启用 SNR DR 时由 env 重采） |
| ρ | rho | 0.9 | AR(1) 信道时间相关系数 |
| channel_model | channel_model | "rayleigh" | "rayleigh" / "rician" |
| csi_mode | csi_mode | "imperfect" | "perfect" / "imperfect" |
| ε | epsilon_csi | 0.05 | CSI 误差范数上界 |
| csi_delay | csi_delay | 1 | CSI 反馈延迟（时隙） |
| outage_prob | outage_prob | 0.05 | 中断概率约束 |
| λ | lambda_rate | 2.5 | 泊松到达率 (packets/slot/user) |
| Q_max | Q_max | 100.0 | 最大队列长度 (bits) |
| packet_size | packet_size | 1.0 | 每包大小 (bits, 归一化) |
| SNR DR | snr_domain_randomization | True | episode 级 σ² 随机 |
| SNR DR 下界 | snr_min_db | 0.0 | dB |
| SNR DR 上界 | snr_max_db | 30.0 | dB |

> 注：`sic_imperfection`（β）目前**没有**单独的 SystemConfig 字段。`rate_calc.py` 函数签名默认 `0.0`，`rsma_env.py:374` 走 `getattr(config.system, 'sic_imperfection', 0.0)` 兜底——即**默认完美 SIC**。如需引入残余干扰，请在 SystemConfig 中新增字段或直接修改 `rate_calc.py` 默认值。

### LyapunovConfig (paper_ee_v2 混合奖励)
| 参数 | 默认值 | 说明 |
|------|--------|------|
| V / V_min / V_max | 5.0 / 2.0 / 20.0 | Lyapunov 权重区间 |
| V_exponent | 1.5 | 自适应 V 指数 |
| adaptive_V | True | 启用自适应 V |
| use_virtual_queue | True | 启用能量虚拟队列 Z_k |
| vq_step_size | 0.1 | Z_k 更新步长 |
| snr_gamma / snr_ref_db | 0.3 / 30.0 | SNR 感知 V 缩放（辅助） |
| v_eff_cap_factor | 3.0 | V_eff 上限因子 |
| P_min_floor | 0.5 | 最低功率地板（防"几乎关机"） |
| ee_mode | True | 启用 EE = R_sum / P_total 激励 |
| ee_weight | 6.5 | EE 激励权重 |
| ee_power_floor | 1.5 | EE 计算中的功率下限 |
| blend_alpha | 1.5 | 队列紧迫度指数（>1 加速切换到 EE 模式） |

### RewardConfig (硬约束惩罚权重)
| 参数 | 默认值 | 说明 |
|------|--------|------|
| w_overflow | 12.0 | 队列溢出硬惩罚（Lyapunov 内部默认 5.0，以 Config 注入为准） |
| w_outage | 3.0 | ⚠️ **死字段**：当前 `LyapunovController.compute_reward` 未使用 |
| w_power_exceed | 10.0 | ⚠️ **死字段**：在 lyapunov.py 中未出现 |
| reward_scale | 0.01 | 全局 reward 缩放（稳定 PPO value 头） |

### PPOConfig (Worker 算法超参)
| 参数 | 默认值 | 说明 |
|------|--------|------|
| lr_actor / lr_critic | 3e-4 / 1e-3 | 学习率 |
| gamma / gae_lambda | 0.99 / 0.95 | 折扣 / GAE |
| clip_ratio | 0.2 | PPO 裁剪范围 |
| entropy_coef / value_coef | 0.005 / 0.5 | 损失系数 |
| max_grad_norm | 0.5 | 梯度裁剪 |
| n_epochs / batch_size | 8 / 64 | PPO 内层迭代 |
| rollout_steps | 2048 | rollout buffer 容量 |
| total_episodes | 8000 | （CLI 默认覆盖，见下） |
| target_kl | 0.02 | KL 早停阈值 |
| lr_decay_final_ratio | 0.2 | LR 线性衰减末端比例 |
| kl_stop_lr_decay | 0.9 | 触发 KL 早停后 LR 衰减乘子 |
| kl_stop_streak_threshold | 3 | 连续 KL 早停几次后强制衰减 |
| hidden_dim / n_layers | 256 / 3 | MLP 结构 |
| action_dim | 48 | env 默认 3K；训练入口强制 4K，此字段仅供 ActorCritic 初始化参考 |
| log_interval / save_interval / eval_interval | 20 / 100 / 500 | 训练循环周期 |

### HierarchicalConfig (Meta 架构参数)
| 参数 | 默认值 | 说明 |
|------|--------|------|
| T_s | 20 | 慢尺度决策间隔 (slots) |
| meta_lr | 1e-3 | Meta 学习率 |
| meta_gamma / meta_gae_lambda | 0.95 / 0.95 | Meta 折扣 / GAE |
| meta_clip_ratio | 0.2 | Meta PPO 裁剪 |
| meta_entropy_coef | 0.05 | split Bernoulli 熵系数 |
| meta_group_entropy_coef | 0.12 | group Categorical 熵系数 |
| meta_hidden_dim | 64 | Meta hidden |
| meta_n_heads | 4 | Multi-head Attention 头数 |
| meta_d_user | 7 | per-user 特征维度（含归一化 group_id） |
| meta_n_epochs | 4 | Meta PPO 内层迭代 |
| meta_max_grad_norm | 0.5 | Meta 梯度裁剪 |
| meta_update_threshold | 50 | 积累多少 meta transitions 触发更新 |
| n_groups | 4 | 用户分组数 G |
| pl_temperature | 1.0 | Plackett-Luce 温度（Worker 侧） |

### Config (顶层)
| 字段 | 默认值 | 说明 |
|------|--------|------|
| seed | 42 | 全局随机种子 |
| device | "cpu" | "cpu" / "cuda" |
| exp_name | "robust_snr" | 实验名（决定 `logs/{exp_name}/` 子目录） |

---

## 评估指标

| 指标 | 公式 | 意义 |
|------|------|------|
| 累计吞吐 | Σ_t R_sum[t] | 总传输比特数 |
| 投递率 | served_packets / arrived_packets | 数据包服务完成率 |
| 溢出率 | overflow / total_arrivals | 队列溢出比例 |
| 平均队列 | (1/KT) Σ Σ Q_k[t] | 平均队列积压 |
| Jain 公平性 | (Σ R_k)² / (K·Σ R_k²) | 用户间公平性 |
| 能量效率 | R_sum / P_total | bits/Joule |
| 功率违约 | Σ 1{p_avg_k > P_avg} / K | 长期功率超预算比例 |

---

## 项目结构

```
src/
├── config.py                # SystemConfig / LyapunovConfig / RewardConfig / PPOConfig / HierarchicalConfig / Config
├── env/
│   ├── channel.py           # 信道模型 (AR(1) + CSI 误差)
│   ├── traffic.py           # 流量与队列管理 (Poisson, Q_max 截断)
│   ├── beamforming.py       # MRT 旧路径 (env 当前用); RZF + 分组公共流已实现 (待集成)
│   ├── rate_calc.py         # SIC 速率计算 (含 Plackett-Luce 解码序)
│   └── rsma_env.py          # Gymnasium env (obs=10K+1, action=3K 默认, 支持 4K)
├── models/
│   ├── actor_critic.py      # Worker MLP PPO Actor-Critic (use_plackett_luce 开关)
│   └── meta_controller.py   # Meta Self-Attention PPO (MetaActorCritic / Buffer / Trainer)
├── optimization/
│   ├── lyapunov.py          # paper_ee_v2 混合奖励 (drift + ee_bonus + 自适应权重)
│   ├── meta_reward.py       # Meta 奖励 = avg_ee - 2·overflow + 0.5·group_diversity
│   └── power_control.py     # 功率约束工具 (部分占位)
└── training/
    ├── main.py              # 双层 PPO 训练入口 (Worker + Meta 协同)
    ├── ppo.py               # PPOTrainer (clip + GAE + KL 早停 + LR 衰减)
    └── buffer.py            # RolloutBuffer (含 decoding_order 额外槽位)

scripts/
├── train.py                 # 训练 CLI（完整参数：消融/Lagrangian/resume/课程/--smoke 冒烟）
├── evaluate_snr_sweep.py    # SNR 扫描评估（PPO + baseline 对比，输出宽表）
├── optimize_loop.py         # optimize-loop 自动优化循环驱动
├── cloud_agents_api.py      # Cloud Agents API 客户端（辅助）
└── baselines/
    ├── sdma_baseline.py     # SDMA (alpha=0)
    ├── noma_baseline.py     # NOMA (alpha=1)
    ├── equal_power.py       # EqualPower
    └── adaptive_power_baseline.py

tests/
├── test_channel.py / test_env.py / test_traffic.py / test_rate_calc.py
├── test_meta_controller.py  # Meta 分组 + 分裂测试
├── test_plackett_luce.py    # PL 解码序测试
└── test_training.py         # 训练流程测试

# 项目根附属（历史/临时分析脚本，不在主流程中）
analyze_rewards.py
detailed_action_analysis.py
v_parameter_recommendation.py
test_imperfect_csi.py
Makefile                     # 统一命令入口（setup/test/smoke/eval，要求 Python 3.14）
MODEL.md                     # 模型说明（与本 CLAUDE.md 互补）
INVESTIGATION_REPORT.md      # 历史问题调查
IMPLEMENTATION_GUIDE.md      # 实现指南
DIAGNOSTIC_SUMMARY.txt       # 诊断纪要
```

> 根目录散落脚本（`analyze_rewards.py` 等）建议在新需求中逐步整理到 `scripts/`。

### optimize-loop 跨轮知识库（版本跟踪）

`logs/opt_knowledge.json` 是 `scripts/optimize_loop.py` 的跨轮累积知识库（失败方向库 + 成功方向库），通过 `.gitignore` 的 `logs/*` + `!logs/opt_knowledge.json` 豁免纳入版本跟踪，与 `logs/` 下的运行产物忽略规则区分。路径、读写约定及与 Trellis 学习通道（`.trellis/workflow.md` Phase 3.3）的分工见 `AGENTS.md` §「optimize-loop 跨轮知识库」。提出新优化方向前必须先查阅本文件，避免重复已失败方向。

---

## 开发命令

> **统一命令入口：项目根目录 `Makefile`**。**Python 版本要求固定为 3.14**（`make check-python` 会强校验，不匹配即中止）。
> 冷环境一条命令即可跑通冒烟训练：`make smoke`（自动先执行 setup）。

```bash
make help            # 查看全部命令
make setup           # 创建 .venv 并安装依赖（幂等，requirements.txt 变更时自动重装）
make test            # 快速单元测试（改完代码必须跑，等价 pytest tests/ -v）
make smoke           # 冒烟训练（等价 python scripts/train.py --smoke --exp_name make_smoke）
make eval            # SNR 扫描评估（可覆盖：make eval CKPT=logs/xxx/checkpoints/final.pt N_EPISODES=20）
make clean           # 删除 .venv
```

以下为底层直调命令（Makefile 内部等价形式，日常优先用 make）：

```bash
# 安装依赖
pip install -r requirements.txt

# 运行单元测试（改完代码必须跑）
pytest tests/ -v

# 单个测试文件
pytest tests/test_env.py -v

# ---- 训练入口 ----
# (A) 推荐：直接走 src.training.main 的完整 CLI
python -m src.training.main \
    --exp_name robust_snr \
    --total_episodes 8000 \
    --seed 42 \
    --device cpu \
    --csi_mode imperfect \
    --snr_randomize true --snr_min_db 0 --snr_max_db 30 \
    --T_s 20

# (B) scripts/train.py（参数更全：消融/Lagrangian/resume/课程等；默认 --episodes 5000，需按需指定）
python scripts/train.py --episodes 8000 --csi_mode imperfect

# 冒烟训练（极小配置，快速验证可运行性）
python scripts/train.py --smoke

# ---- 评估 ----
# SNR 扫描评估（参数名为 --n_episodes；输出为宽表，解析按列位置提取）
python scripts/evaluate_snr_sweep.py --n_episodes 50 --checkpoint logs/snr_robust/checkpoints/final.pt
```

`src.training.main` 主要 CLI 参数（仅在 `--xxx` 显式指定时才覆盖 Config 默认）：
- `--exp_name` / `--total_episodes` / `--seed` / `--device`
- `--csi_mode {perfect, imperfect}`
- `--snr_randomize {true, false}` / `--snr_min_db` / `--snr_max_db`
- `--T_s`（慢尺度决策间隔）

---

## 编码规范 (JSAC/TWC 期刊标准)

### 数学正确性
- **β (sic_imperfection) ≠ ε (error_bound)**：两者为独立物理参数
- SINR 分母三部分：未解码干扰 + SIC残余（含硬件β项和CSI误差ε项） + 热噪声
- 所有矩阵运算添加维度注释：`(N_r, K)`, `(K,)`, `(batch, K, 2)`
- 关键公式在 docstring 中以 LaTeX 注明

### 测试纪律
- **改完主动跑测试**: `pytest tests/ -v`
- 新增功能必须添加对应测试用例
- 测试覆盖: 形状正确性 / 数值正确性 / 边界条件 / 可复现性

### 可复现性
- 随机种子通过 `np.random.seed()` + `torch.manual_seed()` + `env.reset(seed=X)` 三重设置
- 实验配置完整记录（config hash, git commit, seeds）
- 多种子实验报告 95% 置信区间

### 统计显著性
- Benchmark 对比报告: 均值 ± 95% CI, p-value (Welch's t-test), Cohen's d
- 多重比较使用 Holm-Bonferroni 校正
- 收敛判定: 变异系数 + 趋势斜率双标准

### 命名约定
- 物理量语义命名: `sic_imperfection` 非 `beta`, `error_bound` 非 `eps`
- 矩阵大写: `H`, `V`, `W`; 向量小写: `h_k`, `v_s`
- 流级变量后缀: `power_streams`, `z_scores_streams`
- 类名 CamelCase, 函数/变量 snake_case, 常量 UPPER_SNAKE_CASE
