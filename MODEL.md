# 上行 RSMA 多时间尺度在线学习系统 — 数学建模文档

> **Task #10** · 对应代码仓库：`rsma-energy-efficient`
> 本文档为系统提供完整、严谨的数学建模，覆盖优化变量、约束、目标函数与分层强化学习算法。
> 标记说明：✅ = 已在代码中实现；🎯 = 目标设计（待实现 / 部分实现）。

---

## 第 1 章 系统模型

### 1.1 网络拓扑

考虑一个**单小区上行多用户 RSMA 系统**：

- 用户数：$K = 16$，每用户配备 $N_t = 2$ 根虚拟发射天线（即 2 路数据流：公共流 + 私有流）
- 基站：$N_r = 16$ 根接收天线
- 时隙模型：离散时隙 $t \in \{0, 1, \ldots, T-1\}$，每 episode 长度 $T = 200$ 时隙
- 双时间尺度：慢尺度周期 $T_s = 20$ 个时隙（Meta 决策周期）

### 1.2 信道模型

#### 1.2.1 AR(1) 时变信道 ✅

用户 $k$ 在时隙 $t$ 到基站的信道矩阵 $H_k[t] \in \mathbb{C}^{N_r \times 1}$ 服从一阶自回归（AR(1)）模型：

$$
H_k[t] = \rho \, H_k[t-1] + \sqrt{1-\rho^2}\, \Delta H_k[t]
$$

- $\rho = 0.9$：相邻时隙信道相关系数
- $\Delta H_k[t] \sim \mathcal{CN}(0, \sigma_h^2 I_{N_r})$：独立同分布的复高斯增量
- 物理含义：信道呈缓变 Rayleigh 衰落，相关性 $\rho$ 越大表示信道变化越慢

#### 1.2.2 不完美 CSI 模型 ✅

基站获得的信道估计 $\hat{H}_k$ 与真实信道 $H_k$ 之间存在估计误差：

$$
\hat{H}_k = H_k + E_k, \qquad E_k \sim \mathcal{CN}\!\left(0,\; \epsilon_{\rm CSI}^2 \, I_{N_r}\right)
$$

- $\epsilon_{\rm CSI} = 0.05$：CSI 误差标准差（相对信道功率）
- 设计要求：预编码与 SIC 解码均基于 $\hat{H}$，速率计算基于真实 $H$（评估真实性能）

### 1.3 RSMA 信号模型 ✅

每个用户根据 Meta-Controller 给出的分裂决策 $\delta_k \in \{0, 1\}$ 选择信号结构。

**(a) 启用分裂（$\delta_k = 1$）**：用户 $k$ 将本地消息分裂为公共部分 $s_k^c$ 与私有部分 $s_k^p$：

$$
x_k = \sqrt{p_k^c}\, s_k^c + \sqrt{p_k^p}\, s_k^p
$$

其中：

$$
p_k^c = \alpha_k \, p_k, \qquad p_k^p = (1 - \alpha_k)\, p_k, \qquad \alpha_k \in [0,1]
$$

- $p_k$：用户 $k$ 的总发射功率
- $\alpha_k$：公共流功率占比

**(b) 不分裂（$\delta_k = 0$）**：用户仅发送私有流，强制 $\alpha_k = 0$：

$$
x_k = \sqrt{p_k}\, s_k^p
$$

约定符号能量 $\mathbb{E}[|s_k^c|^2] = \mathbb{E}[|s_k^p|^2] = 1$。

### 1.4 接收信号模型 ✅

基站接收到的叠加信号为：

$$
y[t] = \sum_{k=1}^{K} H_k[t]\, x_k[t] + n[t], \qquad n[t] \sim \mathcal{CN}(0, \sigma^2 I_{N_r})
$$

- $\sigma^2$：每根接收天线的噪声功率
- 该模型为标准多用户 MAC（Multiple Access Channel）信号模型

### 1.5 预编码 / 组合矩阵设计

#### 1.5.1 目标方案：RZF（正则化迫零） 🎯

为抑制多用户干扰，基站采用 **RZF 接收组合**：

$$
W_{\rm RZF} = \hat{H}^{\sf H} \left( \hat{H}\hat{H}^{\sf H} + \alpha_{\rm reg} I_{N_r} \right)^{-1}
$$

其中正则化系数：

$$
\alpha_{\rm reg} = \frac{K \sigma^2}{P_{\rm total}}
$$

物理含义：当 SNR 较低时 $\alpha_{\rm reg}$ 较大，组合矩阵退化为 MRC（噪声主导）；当 SNR 较高时退化为 ZF（干扰主导）。RZF 在两者之间取得最优折中。

**RZF 的优势**：
1. **近最优**：在大维度极限下接近 MMSE 容量上界
2. **闭式解**：无需迭代或学习，计算复杂度 $\mathcal{O}(K^3)$
3. **鲁棒性**：天然处理 CSI 误差（正则项吸收估计噪声）

#### 1.5.2 当前实现：MRT + 正交补 ✅

代码中目前使用基于 MRT（最大比发送）方向并通过 Gram-Schmidt 正交补抑制其他用户干扰的简化实现，详见 [src/env/beamforming.py](file:///Users/libo/PycharmProjects/rsma-energy-efficient/src/env/beamforming.py)。

#### 1.5.3 列归一化

无论 RZF 还是 MRT，组合向量均做单位归一化：

$$
w_k = \frac{W[:, k]}{\| W[:, k] \|_2}
$$

功率完全由 $p_k$ 控制，与方向解耦。

### 1.6 SIC 解码模型 ✅

设解码顺序 $\pi$ 是关于 $2K$ 个流（$K$ 个公共流 + $K$ 个私有流）的一个排列：$\pi: \{1, \ldots, 2K\} \to \{c_1, \ldots, c_K, p_1, \ldots, p_K\}$。

#### 1.6.1 公共流 SINR

在解码位置 $\pi^{-1}(c_k)$ 处，用户 $k$ 的公共流 SINR 为：

$$
\mathrm{SINR}_k^c = \frac{|w_k^{\sf H} H_k|^2 \, p_k^c}{\displaystyle\sum_{j:\, \pi(j) > \pi(c_k)} |w_k^{\sf H} H_j|^2 \, p_j \;+\; \sigma^2}
$$

分母仅包含**尚未解码**的流的干扰加噪声（已解码流通过 SIC 减去）。

#### 1.6.2 私有流 SINR

在解码位置 $\pi^{-1}(p_k)$ 处（公共流通常已被解码消除）：

$$
\mathrm{SINR}_k^p = \frac{|w_k^{\sf H} H_k|^2\, p_k^p}{\displaystyle\sum_{j:\, \pi(j) > \pi(p_k)} \!\!|w_k^{\sf H} H_j|^2 p_j \;+\; \beta\!\!\sum_{j:\, \pi(j) < \pi(p_k)} \!\!|w_k^{\sf H} H_j|^2 p_j \;+\; \sigma^2}
$$

- $\beta \in [0, 1]$：SIC 不完美因子（残余干扰系数）
- 第一项：未解码流干扰；第二项：SIC 残留；第三项：噪声

### 1.7 可达速率 ✅

用户 $k$ 在时隙 $t$ 的瞬时可达速率：

$$
R_k[t] = \begin{cases}
\log_2\!\big(1 + \mathrm{SINR}_k^c\big) + \log_2\!\big(1 + \mathrm{SINR}_k^p\big), & \delta_k = 1 \\[4pt]
\log_2\!\big(1 + \mathrm{SINR}_k^p\big), & \delta_k = 0
\end{cases}
$$

单位为 bits/s/Hz（归一化带宽）。

### 1.8 用户分组模型 🎯

引入分组变量 $g_k \in \{1, 2, \ldots, G\}$，将用户划分为 $G$ 个互斥子集 $\mathcal{K}_g = \{k : g_k = g\}$。

- **同组用户**：共享一个公共消息流，进行组内 SIC 解码
- **异组用户**：公共流互不干扰（基于正交资源或子空间分离的简化假设）
- **预编码影响**：每组拥有独立的公共流方向 $w_c^g$（见 §6.3）
- **复杂度影响**：SIC 复杂度从 $\mathcal{O}(K)$ 降为 $\mathcal{O}(\max_g |\mathcal{K}_g|)$

---

## 第 2 章 流量与队列模型

### 2.1 Poisson 到达 ✅

每用户在每时隙的数据到达量服从 Poisson 分布：

$$
A_k[t] \sim \mathrm{Poisson}(\lambda_k), \qquad \lambda_k = 2.5 \;\text{(bits/slot)}
$$

各用户、各时隙间相互独立。

### 2.2 队列动态 ✅

$$
Q_k[t+1] = \max\!\big(Q_k[t] - D_k[t],\; 0\big) + A_k[t]
$$

实际服务比特数：

$$
D_k[t] = \min\!\big(Q_k[t],\; d_{{\rm ratio},k}[t] \cdot R_k[t]\big)
$$

其中 $d_{{\rm ratio},k} \in [0, 1]$ 为 Worker 输出的服务比例。

### 2.3 队列稳定性约束

要求所有用户队列均值有界：

$$
\lim_{T \to \infty} \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}\!\left[Q_k[t]\right] < \infty, \qquad \forall k
$$

通过 Lyapunov 漂移+加惩罚框架强制满足（详见第 8 章）。

---

## 第 3 章 优化问题建模

### 3.1 优化变量汇总表

| 变量 | 符号 | 维度 | 域 | 时间尺度 | 决策层 |
|------|------|------|-----|---------|--------|
| 用户分组 | $g_k$ | $K$ | $\{1, \ldots, G\}$ | 慢（每 $T_s$ 步） | Meta 🎯 |
| 分裂决策 | $\delta_k$ | $K$ | $\{0, 1\}$ | 慢（每 $T_s$ 步） | Meta ✅ |
| 总功率 | $p_k$ | $K$ | $[0, P_{\max}]$ | 快（每步） | Worker ✅ |
| 功率分割比 | $\alpha_k$ | $K$ | $[0, 1]$ | 快（每步） | Worker ✅ |
| 服务比例 | $d_{{\rm ratio},k}$ | $K$ | $[0, 1]$ | 快（每步） | Worker ✅ |
| 解码顺序 | $\pi$ | 排列 | $\mathcal{S}_{2K}$ | 快（每步） | Worker 🎯 |
| 预编码矩阵 | $W$ | $N_r \times K$ | $\mathbb{C}$ | 快（解析） | RZF 闭式 🎯 |

### 3.2 目标函数

最大化长期平均能量效率（EE）：

$$
\boxed{\;\max_{\{g, \delta, p, \alpha, \pi\}} \quad \bar{\eta}_{\rm EE} = \lim_{T \to \infty} \frac{1}{T} \sum_{t=0}^{T-1} \frac{\sum_{k=1}^{K} R_k[t]}{\sum_{k=1}^{K} p_k[t] + P_{\rm circuit}}\;}
$$

- 分子：总和速率（bits/s/Hz）
- 分母：总发射功率 + 电路功耗 $P_{\rm circuit}$
- 单位：bits / Joule

### 3.3 约束条件

| # | 约束 | 数学表达 |
|---|------|---------|
| C1 | 功率约束 | $0 \le p_k[t] \le P_{k,\max}, \;\forall k, t$ |
| C2 | 队列稳定性 | $\bar{Q}_k < \infty,\;\forall k$（Lyapunov 保证） |
| C3 | 分裂一致性 | $\alpha_k \cdot (1 - \delta_k) = 0$ |
| C4 | 解码可行性 | $\pi$ 必须为 $\{1, \ldots, 2K\}$ 上的合法排列 |
| C5 | 分组非空 | $|\mathcal{K}_g| \ge 1, \;\forall g \in \{1, \ldots, G\}$ |
| C6 | 长期平均功率 | $\bar{p}_k \le P_{\rm avg}$（虚拟队列保证） |

---

## 第 4 章 分层强化学习架构

### 4.1 架构总览

```
┌─────────────────────────────────────────────────────────┐
│       Meta-Controller (慢 PPO + Self-Attention)         │
│  输入: 用户特征矩阵 (K, d_user)                          │
│  输出: 分组 g ∈ {1,...,G}^K  +  分裂 δ ∈ {0,1}^K        │
│  决策周期: 每 T_s = 20 个时隙                            │
└──────────────────────────┬──────────────────────────────┘
                           │ g, δ
                           ▼
┌─────────────────────────────────────────────────────────┐
│             Worker (快 PPO + MLP)                        │
│  输入: 环境状态 (含 split_mask, group_id)                │
│  输出: 功率 p + 分割比 α + 服务比 d + 顺序分数 z         │
│  决策周期: 每个时隙                                      │
└──────────────────────────┬──────────────────────────────┘
                           │ p, α, d, z → π (Plackett-Luce)
                           ▼
┌─────────────────────────────────────────────────────────┐
│      环境 (RZF 预编码 + SIC 解码 + 队列更新)             │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Meta-Controller MDP（慢时间尺度）

设第 $n$ 个慢尺度决策点对应时隙 $t = n \cdot T_s$。

#### 4.2.1 状态空间 $s^{\rm meta}[n]$

Per-user 特征矩阵 $\mathbf{S}^{\rm meta} \in \mathbb{R}^{K \times d_{\rm user}}$，$d_{\rm user} = 6$：

| 维度 | 符号 | 含义 |
|------|------|------|
| 1 | $h_k$ | 归一化信道增益 $\|H_k\|_F / \max_j \|H_j\|_F$ |
| 2 | $q_k$ | 队列比率 $Q_k / Q_{\max}$ |
| 3 | $\delta_k^{\rm prev}$ | 上次分裂决策 |
| 4 | $\bar{R}_k$ | 近期平均速率（EMA） |
| 5 | $o_k$ | 近期溢出次数 |
| 6 | $u_k$ | 功率利用率 $\bar{p}_k / P_{\max}$ |

#### 4.2.2 动作空间 $a^{\rm meta}[n]$

- **分裂决策**：$\delta_k \in \{0, 1\}^K$，每用户独立 Bernoulli 采样：
$$\delta_k \sim \mathrm{Bernoulli}\!\big(\sigma(\theta_k^\delta)\big)$$
- **用户分组**：$g_k \in \{1, \ldots, G\}^K$，每用户独立 Categorical 采样：
$$g_k \sim \mathrm{Categorical}\!\big(\mathrm{softmax}(\theta_k^g)\big), \quad \theta_k^g \in \mathbb{R}^G$$
- 推荐 $G = 4$（分组数过多增加复杂度而收益有限）。

#### 4.2.3 自注意力网络结构 ✅

```
用户嵌入: Linear(d_user → hidden_dim)
    ↓
Self-Attention: MultiHeadAttention(hidden_dim, n_heads=4)
    [捕捉用户间信道相关性 / 干扰关系]
    ↓
Split Actor: Linear(hidden_dim → 1)   → K 个 Bernoulli logits
Group Actor: Linear(hidden_dim → G)   → K × G Categorical logits  🎯
Critic     : MeanPool → Linear(hidden_dim → 1)
```

注意力权重 $A_{ij} = \mathrm{softmax}\!\big(Q_i K_j^{\sf T} / \sqrt{d_k}\big)$ 可解释为用户 $i$ 与 $j$ 之间的"耦合强度"。

#### 4.2.4 奖励 $r^{\rm meta}[n]$

$$
r^{\rm meta}[n] = \frac{1}{T_s} \sum_{t = n T_s}^{(n+1) T_s - 1} \eta_{\rm EE}[t] \;-\; \lambda_{\rm overflow} \cdot \bar{o}[n]
$$

- 第一项：本慢周期内平均能效
- 第二项：溢出惩罚（$\bar{o}[n]$ 为周期内平均溢出次数）

### 4.3 Worker MDP（快时间尺度）

#### 4.3.1 状态空间 $s^{\rm worker}[t]$

总维度 $9K + 1 + d_{\rm group}$（$K = 16$ 时约 145 + 分组维度）：

| 分量 | 维度 | 描述 |
|------|------|------|
| 信道增益 | $K$ | $\|\hat{H}_k\|^2$ |
| 队列长度 | $K$ | $Q_k / Q_{\max}$ |
| 队列变化 | $K$ | $\Delta Q_k$ |
| 到达量 | $K$ | $A_k$ |
| 功率历史 | $K$ | $\bar{p}_k$ |
| 速率历史 | $K$ | $\bar{R}_k$ |
| 干扰水平 | $K$ | $\sum_{j \ne k} |w_k^{\sf H} H_j|^2$ |
| 公平性 | $K$ | Jain 指数局部分量 |
| 分裂掩码 | $K$ | $\delta_k$ |
| 分组 ID | $K$ 或 $K \cdot G$ | $g_k$ 或 one-hot |
| 时间比例 | $1$ | $t / T$ |

#### 4.3.2 动作空间 $a^{\rm worker}[t]$

总维度 $4K$：

$$
a^{\rm worker} = \big[p_{\rm raw},\; \alpha_{\rm raw},\; d_{\rm raw},\; z\big] \in \mathbb{R}^{4K}
$$

经如下变换映射到合法域：

$$
\begin{aligned}
p_k &= \sigma(p_{{\rm raw},k}) \cdot P_{\max} \\
\alpha_k &= \sigma(\alpha_{{\rm raw},k}) \cdot \delta_k \quad\text{（分裂一致性约束）} \\
d_{{\rm ratio},k} &= \sigma(d_{{\rm raw},k}) \\
z_k &= \mathrm{softplus}(z_{{\rm raw},k}) > 0 \quad\text{（Plackett-Luce 分数）}
\end{aligned}
$$

#### 4.3.3 Plackett-Luce 解码顺序采样 🎯

给定正分数向量 $z = (z_1, \ldots, z_K)$，从 Plackett-Luce 分布采样排列 $\pi$：

$$
P(\pi \mid z) = \prod_{i=1}^{K} \frac{z_{\pi(i)}}{\sum_{j=i}^{K} z_{\pi(j)}}
$$

**采样方法（Gumbel-Top-K trick）**：

$$
\pi = \mathrm{argsort}\!\big(-(\log z + G)\big), \qquad G_i \sim \mathrm{Gumbel}(0, 1)
$$

**对数概率**（用于 PPO 策略梯度）：

$$
\log P(\pi \mid z) = \sum_{i=1}^{K}\left[\log z_{\pi(i)} - \log\!\left(\sum_{j=i}^{K} z_{\pi(j)}\right)\right]
$$

#### 4.3.4 奖励 $r^{\rm worker}[t]$ ✅

$$
r^{\rm worker}[t] = w(\bar{Q}) \cdot \mathrm{drift}[t] \;+\; \big(1 - w(\bar{Q})\big) \cdot \eta_{\rm EE}[t] \;-\; c_{\rm overflow} \cdot \mathbb{1}\!\left[\text{overflow}\right]
$$

各项定义：
- 自适应权重：$w(\bar{Q}) = (\bar{Q} / Q_{\max})^{\beta_{\rm blend}}$
- Lyapunov 漂移：$\mathrm{drift}[t] = -\sum_k Q_k \Delta Q_k / Q_{\max}^2$
- 能量效率：$\eta_{\rm EE}[t] = \sum_k R_k / \max\!\big(\sum_k p_k,\; P_{\rm floor}\big)$

队列高时偏向稳定性（drift 主导），队列低时偏向能效（$\eta_{\rm EE}$ 主导）。

---

## 第 5 章 Plackett-Luce 解码顺序优化详解

### 5.1 动机

- SIC 解码顺序 $\pi$ 直接决定每个用户的干扰环境与可达速率
- 寻找最优顺序是 NP-hard 组合问题（共 $K!$ 种排列，$K=16$ 时约 $2 \times 10^{13}$）
- **核心思路**：将组合搜索连续化 — Plackett-Luce 通过实值分数 $z \in \mathbb{R}_+^K$ 参数化排列分布，使梯度方法可用

### 5.2 与 PPO 的集成 🎯

| 阶段 | 操作 |
|------|------|
| 网络输出 | Worker 额外输出 $z = \mathrm{softplus}(z_{\rm raw}) \in \mathbb{R}_+^K$ |
| 训练 | 用 Plackett-Luce 采样 $\pi$，记录 $\log P(\pi \mid z)$，PPO 重要性采样优化 |
| 推理 | 贪心解码 $\pi = \mathrm{argsort}(-z)$（确定性最优） |

策略梯度形式（与高斯策略合并）：

$$
\nabla_\theta J = \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta(a^{\rm cont}, \pi^{\rm perm} \mid s) \cdot \hat{A}\right]
$$

其中 $\log \pi_\theta = \log \pi_\theta^{\rm Gauss}(p, \alpha, d) + \log P(\pi \mid z)$。

### 5.3 简化策略

**完整问题**：$2K$ 个流（$K$ 公共 + $K$ 私有）的全排列，搜索空间 $(2K)!$。

**简化 1（标准 RSMA 假设）**：所有公共流先于所有私有流解码。

**简化 2（同类内部共享顺序）**：公共流解码顺序与私有流解码顺序使用同一 $K$-排列 $\pi$。

最终学习目标：仅优化 1 个 $K$-排列，对应 Plackett-Luce 输出 $z \in \mathbb{R}_+^K$。

---

## 第 6 章 RZF 预编码设计 🎯

### 6.1 公式

$$
W = \hat{H}^{\sf H} \left( \hat{H} \hat{H}^{\sf H} + \alpha_{\rm reg} I_{N_r} \right)^{-1}
$$

- $\hat{H} \in \mathbb{C}^{N_r \times K}$：估计信道矩阵（列为各用户信道）
- $\alpha_{\rm reg} = K \sigma^2 / P_{\rm avg}$：正则化系数

### 6.2 列归一化

$$
w_k = \frac{W[:, k]}{\| W[:, k] \|_2}
$$

确保 $w_k$ 仅作为方向向量，功率由 $p_k$ 独立控制。

### 6.3 公共流预编码

对启用分裂的同组用户集合 $\mathcal{K}_g = \{k : \delta_k = 1, g_k = g\}$，公共流方向取组内信道之和的归一化（MRT 方向）：

$$
w_c^g = \frac{\sum_{k \in \mathcal{K}_g} \hat{h}_k}{\big\| \sum_{k \in \mathcal{K}_g} \hat{h}_k \big\|_2}
$$

物理含义：将公共流能量集中在组内用户共同信道方向，最大化组内最弱用户的接收信号。

### 6.4 与学习方法的对比

| 方法 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| **RZF** | 近最优、零训练成本、可解释 | 需要 CSI | CSI 可获得时（本系统） |
| 神经网络预编码 | 可学习非线性映射 | 训练困难、动作维度高 | CSI 严重退化 / 完全黑箱 |
| 迭代优化（WMMSE） | 全局最优 | 计算昂贵、不适合在线 | 离线基准 |

**本系统选择 RZF** 的核心理由：将 $\mathcal{O}(N_r K)$ 维连续动作从 RL 搜索空间中剥离，让 RL 聚焦于不可微的组合决策（分组、分裂、顺序）。

---

## 第 7 章 用户分组策略 🎯

### 7.1 分组目标

- **信道相似的用户分为一组**：公共流增益大，组内 SIC 高效
- **信道正交的用户分到不同组**：私有流互不干扰，提升空间复用

形式化：理想分组应最大化

$$
J_{\rm group} = \sum_g \sum_{k, k' \in \mathcal{K}_g} |\hat{h}_k^{\sf H} \hat{h}_{k'}|^2 - \mu \sum_{g \ne g'} \sum_{k \in \mathcal{K}_g, k' \in \mathcal{K}_{g'}} |\hat{h}_k^{\sf H} \hat{h}_{k'}|^2
$$

但该问题 NP-hard，因此交由 Meta-Controller 学习。

### 7.2 Meta-Controller 分组输出

- 每用户 token 经 Self-Attention 后映射为 $G$ 维 logits
- Categorical 采样得到 $g_k$
- Self-Attention 的 Q/K 内积天然刻画用户间相似度，与分组目标契合

### 7.3 分组数 $G$ 的选择

| $G$ | 行为 | 评价 |
|-----|------|------|
| $G = 1$ | 退化为标准 RSMA（全用户共享一个公共流） | 性能上界基准 |
| $G = 2 \sim 4$ | **本系统推荐**（$G = 4$） | 复杂度与性能的最佳折中 |
| $G = K$ | 退化为 NOMA（每用户独立公共流） | 失去 RSMA 优势 |

---

## 第 8 章 Lyapunov 在线优化框架 ✅

### 8.1 虚拟队列

引入虚拟队列追踪长期平均功率约束：

$$
Z_k[t+1] = \max\!\big(Z_k[t] + p_k[t] - P_{\rm avg},\; 0\big)
$$

$Z_k$ 有界 $\Leftrightarrow$ 长期平均功率 $\bar{p}_k \le P_{\rm avg}$。

### 8.2 漂移-加-惩罚

定义 Lyapunov 函数 $L(t) = \frac{1}{2}\sum_k Q_k^2[t]$，单步漂移：

$$
\Delta L[t] \;\le\; \mathrm{const} + \sum_k Q_k[t]\!\left(A_k[t] - D_k[t]\right)
$$

漂移-加-惩罚目标：

$$
\min_{a[t]} \quad \Delta L[t] - V \cdot \eta_{\rm EE}[t]
$$

- $V$ 越大：越偏向能效优化
- $V$ 越小：越偏向队列稳定

### 8.3 自适应 $V$ 参数 ✅

$$
V[t] = V_{\min} + (V_{\max} - V_{\min}) \cdot \left(1 - \frac{\bar{Q}[t]}{Q_{\max}}\right)
$$

- 队列高时减小 $V$（优先稳定）
- 队列低时增大 $V$（优先能效）

---

## 第 9 章 训练算法

### 9.1 双层 PPO 训练循环

```text
对每个 episode (T = 200 时隙):
  t = 0:
    Meta 决策 (g, δ)  → 设置 env
  
  for t in 0..T-1:
    Worker 决策 (p, α, d, z → π)
    env.step()  → 计算 R_k, 更新 Q_k, 得到 r^worker
    累积 worker_buffer
    
    if t % T_s == 0 and t > 0:
      计算 r^meta  ← 上一周期累计
      存入 meta_buffer
      Meta 重新决策 (g, δ)
  
  Episode 结束:
    Worker PPO 更新（使用 worker_buffer）
    if len(meta_buffer) >= meta_update_threshold:
      Meta PPO 更新（使用 meta_buffer）
```

### 9.2 超参数表

| 参数 | Worker | Meta |
|------|--------|------|
| clip ratio $\epsilon$ | 0.2 | 0.2 |
| entropy coef | 0.01 | 0.05（更高探索） |
| 学习率 | $3 \times 10^{-4}$ | $1 \times 10^{-3}$ |
| 折扣因子 $\gamma$ | 0.99 | 0.95 |
| GAE $\lambda$ | 0.95 | 0.9 |
| 更新频率 | 每 episode | 每 50 meta-transitions |
| Mini-batch 大小 | 64 | 32 |
| Epochs / 更新 | 4 | 4 |

### 9.3 PPO 损失函数

对每层（Meta 或 Worker）独立优化：

$$
\mathcal{L}^{\rm PPO}(\theta) = \mathbb{E}_t\!\left[\min\!\big(r_t(\theta)\hat{A}_t,\; \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\big)\right] - c_v \mathcal{L}^{\rm VF} + c_e \mathcal{H}[\pi_\theta]
$$

- 重要性采样比：$r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\rm old}}(a_t \mid s_t)$
- $\hat{A}_t$：GAE 优势估计
- $\mathcal{H}[\pi_\theta]$：策略熵（鼓励探索）

---

## 第 10 章 实现状态对照表

| 组件 | 目标设计 | 当前实现 | 差距 |
|------|---------|---------|------|
| 预编码 | 🎯 RZF | MRT + 正交补 | 需替换为 RZF 闭式 |
| 解码顺序 | 🎯 Plackett-Luce Worker 输出 | 固定顺序 | 需新增 $z$ 输出与采样模块 |
| 用户分组 | 🎯 Meta Categorical 输出 | 无（隐式 $G=1$） | 需在 Meta 增加 Group Actor |
| 分裂决策 | Meta Bernoulli 输出 | ✅ 已实现 | — |
| 功率分配 | Worker 连续输出 | ✅ 已实现 | — |
| 速率分割 $\alpha$ | Worker 连续输出 | ✅ 已实现 | — |
| 服务比例 $d_{\rm ratio}$ | Worker 连续输出 | ✅ 已实现 | — |
| Self-Attention Meta | Meta 网络主体 | ✅ 已实现 | — |
| Lyapunov 漂移 | drift + 自适应 $V$ | ✅ 已实现 | — |
| 不完美 CSI | $\hat{H} = H + E$ 高斯模型 | ✅ 已实现 | — |
| AR(1) 时变信道 | $\rho = 0.9$ | ✅ 已实现 | — |
| 双层 PPO | 慢/快时间尺度 | ✅ 已实现 | — |

### 推荐实现优先级

1. **P0 — RZF 预编码**：闭式替换，立即提升性能上限
2. **P1 — Plackett-Luce 解码顺序**：解锁 SIC 顺序优化收益
3. **P2 — 用户分组**：扩展 Meta 输出头与 Group Actor 模块
4. **P3 — 联合调优**：三者集成后整体超参重训

---

## 附录 A · 关键符号表

| 符号 | 含义 |
|------|------|
| $K, N_r, N_t$ | 用户数、接收天线、虚拟发射天线 |
| $T, T_s$ | episode 长度、慢尺度周期 |
| $H_k, \hat{H}_k, E_k$ | 真实/估计信道、CSI 误差 |
| $\rho, \epsilon_{\rm CSI}$ | AR(1) 系数、CSI 误差标准差 |
| $\delta_k, \alpha_k, p_k$ | 分裂决策、功率分割比、总功率 |
| $g_k, G$ | 用户分组、最大分组数 |
| $\pi, z$ | 解码顺序、Plackett-Luce 分数 |
| $W, w_k, w_c^g$ | 组合矩阵、私有流方向、公共流方向 |
| $R_k, \eta_{\rm EE}$ | 用户速率、能量效率 |
| $A_k, D_k, Q_k, Z_k$ | 到达、服务、数据队列、虚拟队列 |
| $V[t], \beta_{\rm blend}$ | Lyapunov V 参数、混合系数 |
| $\beta$ | SIC 不完美因子 |

---

*文档版本：v1.0 · 对应 Task #10*
