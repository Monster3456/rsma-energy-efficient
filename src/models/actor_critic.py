"""Actor-Critic网络

Actor输出64维动作（K=16用户 × 4维: alpha, p_total, D_k, z_score）。
- 前3K=48维：连续动作（高斯策略 + tanh squashing）
- 后K=16维：Plackett-Luce解码顺序的z-scores（softplus，不经tanh）

Critic估计状态价值函数V(s)。

使用独立的actor和critic网络（不共享特征），避免value loss梯度干扰actor。
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


def orthogonal_init(layer, gain=np.sqrt(2)):
    """正交初始化"""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


def build_mlp(input_dim, hidden_dim, n_layers):
    """构建MLP特征提取器：LayerNorm + ReLU"""
    layers = []
    in_dim = input_dim
    for _ in range(n_layers):
        linear = nn.Linear(in_dim, hidden_dim)
        orthogonal_init(linear)
        layers.append(linear)
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    return nn.Sequential(*layers)


def plackett_luce_log_prob(scores: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """计算 Plackett-Luce 分布下排列的对数概率。

    Args:
        scores: (batch, K) 正值分数（softplus 输出）
        perm: (batch, K) 排列索引（每行是 0..K-1 的一个排列）

    Returns:
        log_prob: (batch,) 每个样本的对数概率

    公式: log P(π|z) = Σ_{i=1}^{K} [log z_{π(i)} - log(Σ_{j=i}^{K} z_{π(j)})]
    """
    # 按排列顺序重排分数
    ordered_scores = torch.gather(scores, 1, perm)  # (batch, K)
    log_ordered = torch.log(ordered_scores + 1e-8)

    # 计算后缀和的 log: log(sum_{j=i}^K z_{π(j)})
    # 使用 cumsum 从后往前
    reverse_cumsum = torch.flip(
        torch.cumsum(torch.flip(ordered_scores, [1]), dim=1), [1]
    )
    log_suffix_sum = torch.log(reverse_cumsum + 1e-8)

    # log_prob = sum(log_z_pi_i - log_suffix_sum_i)
    log_prob = (log_ordered - log_suffix_sum).sum(dim=1)
    return log_prob


def plackett_luce_sample(scores: torch.Tensor) -> torch.Tensor:
    """使用 Gumbel-Top-K trick 从 Plackett-Luce 分布采样排列。

    Args:
        scores: (batch, K) 正值分数

    Returns:
        perm: (batch, K) 采样的排列（每行是 0..K-1 的排列）
    """
    # Gumbel-Top-K: π = argsort(-(log(z) + Gumbel(0,1)))
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
    perturbed = torch.log(scores + 1e-8) + gumbel_noise
    perm = torch.argsort(perturbed, dim=1, descending=True)
    return perm


class MeanFieldCriticBias(nn.Module):
    """均值场 critic 偏置：从 per-user token 中提取 QoS 类统计量并输出价值偏置。

    输入维度不变：模块在 critic 内部从原始 per-user 特征计算各 QoS 类的
    平均信道增益/队列/功率，以及全局均值，经小 MLP 输出标量 bias 加到 V(s)。
    """

    def __init__(self, d_user, hidden_dim, n_classes=3, qos_drift_weights=None):
        super().__init__()
        self.n_classes = n_classes
        if qos_drift_weights is not None:
            w = torch.tensor(qos_drift_weights, dtype=torch.float32)
            self.register_buffer('norm_weights', w / w.max())
        else:
            self.register_buffer('norm_weights', torch.linspace(0.0, 1.0, n_classes))

        # 输入：d_user(全局 mean-pooling) + 每类均值(3*n_classes) + 全局均值(3)
        self.mlp = nn.Sequential(
            nn.Linear(d_user + 3 * n_classes + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _class_ids(self, per_user):
        """用归一化 priority 特征近似 QoS 类别 (per_user[..., 12])。"""
        priority = per_user[..., 12].unsqueeze(-1)  # (B, K, 1)
        nw = self.norm_weights.view(1, 1, -1)       # (1, 1, C)
        return torch.abs(priority - nw).argmin(dim=-1)  # (B, K)

    def forward(self, per_user):
        """per_user: (B, K, d_user) -> bias: (B,)"""
        feat = per_user[:, :, [0, 1, 4]]  # 信道增益、队列、功率历史
        class_ids = self._class_ids(per_user)
        means = []
        for c in range(self.n_classes):
            mask = (class_ids == c).float().unsqueeze(-1)
            cnt = mask.sum(dim=1).clamp(min=1.0)
            means.append((feat * mask).sum(dim=1) / cnt)
        mean_vec = torch.cat(means, dim=-1)          # (B, 3*C)
        global_mean = feat.mean(dim=1)               # (B, 3)
        user_mean = per_user.mean(dim=1)             # (B, d_user)
        inp = torch.cat([user_mean, mean_vec, global_mean], dim=-1)
        return self.mlp(inp).squeeze(-1)


class ActorCritic(nn.Module):
    """PPO Actor-Critic网络（独立网络架构）

    Actor: obs_dim维观测 → 4K维动作（前3K连续高斯 + 后K维PL解码顺序）
    Critic: obs_dim维观测 → 1维状态价值

    Actor和Critic使用完全独立的网络，避免value梯度干扰policy特征。

    Args:
        obs_dim: 观测维度
        action_dim: 动作维度（4K=64 when use_plackett_luce=True, 3K=48 otherwise）
        hidden_dim: 隐藏层维度
        n_layers: MLP层数
        use_plackett_luce: 是否启用Plackett-Luce解码顺序输出
    """

    def __init__(self, obs_dim=145, action_dim=64, hidden_dim=256, n_layers=3,
                 use_plackett_luce=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_plackett_luce = use_plackett_luce

        if use_plackett_luce:
            self.K = action_dim // 4  # K=16 when action_dim=64
            self.continuous_dim = 3 * self.K  # 连续动作维度=48
        else:
            self.K = action_dim // 3  # 兼容旧模式: action_dim=48, K=16
            self.continuous_dim = action_dim

        # Actor网络（独立）— 输出完整 action_dim 维
        self.actor_net = build_mlp(obs_dim, hidden_dim, n_layers)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        orthogonal_init(self.actor_head, gain=0.01)

        # Critic网络（独立）
        self.critic_net = build_mlp(obs_dim, hidden_dim, n_layers)
        self.critic_head = nn.Linear(hidden_dim, 1)
        orthogonal_init(self.critic_head, gain=1.0)

        # 动作标准差：仅针对连续动作部分（3K维）
        self.log_std = nn.Parameter(torch.full((self.continuous_dim,), -0.5))

    def forward(self, obs):
        """前向传播"""
        actor_features = self.actor_net(obs)
        action_mean = self.actor_head(actor_features)
        critic_features = self.critic_net(obs)
        value = self.critic_head(critic_features)
        return action_mean, value

    def act(self, obs, deterministic=False):
        """给定观测，采样动作

        Args:
            obs: (batch, obs_dim) 或 (obs_dim,)
            deterministic: 评估时用均值
        Returns:
            如果 use_plackett_luce=True:
                action_continuous: (batch, 3K) 连续动作 在[-1,1]范围
                decoding_order: (batch, K) 解码顺序排列
                log_prob: (batch,) 总对数概率（连续 + PL，用于 PPO ratio）
                value: (batch,) 状态价值
                cont_log_prob: (batch,) 仅连续部分对数概率（用于 KL 估计，避免 PL 排列分布污染）
            如果 use_plackett_luce=False:
                action: (batch, 3K) 在[-1,1]范围（tanh squash）
                log_prob: (batch,) 动作对数概率
                value: (batch,) 状态价值
        """
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)

        actor_features = self.actor_net(obs)
        raw_output = self.actor_head(actor_features)  # (batch, action_dim)
        critic_features = self.critic_net(obs)
        value = self.critic_head(critic_features).squeeze(-1)

        # 分割连续动作部分和PL部分
        continuous_mean = raw_output[:, :self.continuous_dim]  # (batch, 3K)
        std = self.log_std.clamp(-3.0, 0.5).exp()

        if deterministic:
            action_squashed = torch.tanh(continuous_mean)
            continuous_log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(continuous_mean, std)
            action_raw = dist.rsample()  # reparameterization trick

            # log probability before squashing
            continuous_log_prob = dist.log_prob(action_raw).sum(dim=-1)

            # tanh squashing
            action_squashed = torch.tanh(action_raw)

            # Jacobian correction for tanh
            continuous_log_prob = continuous_log_prob - torch.sum(
                torch.log(1 - action_squashed.pow(2) + 1e-6), dim=-1
            )

        if not self.use_plackett_luce:
            # 兼容模式：与之前行为一致
            if squeeze:
                action_squashed = action_squashed.squeeze(0)
                continuous_log_prob = continuous_log_prob.squeeze(0)
                value = value.squeeze(0)
            return action_squashed, continuous_log_prob, value

        # Plackett-Luce 解码顺序部分
        z_raw = raw_output[:, self.continuous_dim:]  # (batch, K)
        z_scores = F.softplus(z_raw) + 1e-6  # 保证正值

        if deterministic:
            decoding_order = torch.argsort(z_scores, dim=-1, descending=True)
            pl_log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            decoding_order = plackett_luce_sample(z_scores)
            pl_log_prob = plackett_luce_log_prob(z_scores, decoding_order)

        # 总 log_prob = 连续动作 log_prob + PL log_prob
        total_log_prob = continuous_log_prob + pl_log_prob
        cont_log_prob = continuous_log_prob

        if squeeze:
            action_squashed = action_squashed.squeeze(0)
            decoding_order = decoding_order.squeeze(0)
            total_log_prob = total_log_prob.squeeze(0)
            cont_log_prob = cont_log_prob.squeeze(0)
            value = value.squeeze(0)

        return action_squashed, decoding_order, total_log_prob, value, cont_log_prob

    def evaluate_actions(self, obs, actions, decoding_orders=None):
        """评估给定(obs, action)对的log_prob和value（PPO更新用）

        Args:
            obs: (batch, obs_dim)
            actions: (batch, 3K) squashed连续动作（tanh后的值，在[-1,1]范围内）
            decoding_orders: (batch, K) 排列索引（仅 use_plackett_luce=True 时需要）
        Returns:
            log_prob: (batch,) 总对数概率
            value: (batch,) 状态价值
            entropy: (batch,) 策略熵
        """
        actor_features = self.actor_net(obs)
        raw_output = self.actor_head(actor_features)  # (batch, action_dim)
        critic_features = self.critic_net(obs)
        value = self.critic_head(critic_features).squeeze(-1)

        # 连续动作部分
        continuous_mean = raw_output[:, :self.continuous_dim]
        std = self.log_std.clamp(-3.0, 0.5).exp()
        dist = Normal(continuous_mean, std)

        # 从squashed动作恢复原始动作（atanh）
        actions_clipped = actions.clamp(-0.999, 0.999)
        action_raw = torch.atanh(actions_clipped)

        # log probability
        continuous_log_prob = dist.log_prob(action_raw).sum(dim=-1)

        # Jacobian correction for tanh
        continuous_log_prob = continuous_log_prob - torch.sum(
            torch.log(1 - actions_clipped.pow(2) + 1e-6), dim=-1
        )

        # Entropy (Gaussian entropy, 不含tanh修正的近似)
        continuous_entropy = dist.entropy().sum(dim=-1)

        if not self.use_plackett_luce or decoding_orders is None:
            # 兼容模式：no PL, cont_log_prob = total_log_prob
            return continuous_log_prob, value, continuous_entropy, continuous_log_prob

        # Plackett-Luce 部分
        z_raw = raw_output[:, self.continuous_dim:]  # (batch, K)
        z_scores = F.softplus(z_raw) + 1e-6
        pl_log_prob = plackett_luce_log_prob(z_scores, decoding_orders)

        # PL entropy 近似（无闭式解）
        pl_entropy = torch.zeros_like(pl_log_prob)

        total_log_prob = continuous_log_prob + pl_log_prob
        total_entropy = continuous_entropy + pl_entropy

        # 额外返回 cont_log_prob 供 PPO 计算 KL（避免 PL 排列分布污染 KL 估计）
        return total_log_prob, value, total_entropy, continuous_log_prob

    def actor_params(self):
        """返回Actor参数（用于分离学习率）— 不含 log_std"""
        return list(self.actor_net.parameters()) + \
               list(self.actor_head.parameters())

    def critic_params(self):
        """返回Critic参数（用于分离学习率）"""
        return list(self.critic_net.parameters()) + \
               list(self.critic_head.parameters())

    def log_std_params(self):
        """返回 log_std 参数（独立 param group，可用更高 lr 加速探索学习）"""
        return [self.log_std]


class ActorCriticAttn(nn.Module):
    """Self-Attention Worker (替代 MLP ActorCritic, 解决 LN 稀释 SNR/ε 信号问题).

    输入 obs (10K+3) 解构为：
      - per-user tokens: (K, d_user=10)  — h_gain, queue, Δqueue, arrivals, power_hist,
                                              rate_hist, interference, fairness, split_mask, group_id
      - global features: (d_global=3)    — time_ratio, snr_norm, eps_norm

    架构：
      1. global features → FiLM 生成器 → (γ, β) 对 per-user token embedding 做仿射调制
      2. 调制后的 per-user tokens → MultiHeadSelfAttention → FFN → 残差 + LN
      3. per-user output heads → (alpha, p_total, d_ratio, z_score) → reshape 成 4K
      4. critic: 全局 mean pooling + global features 残差 → V(s) 标量

    Args:
        obs_dim:   观测总维度 = 10K + 3
        action_dim: 4K (PL 模式)
        K:         用户数（=16）
        d_user:    每用户特征维度（=10）
        d_global:  全局特征维度（=3）
        hidden_dim: token embedding 维度（=128 比 MLP 256 小，因为有 K 个 token 并行）
        n_heads:   注意力头数
        n_attn_layers: attention 层数
    """

    def __init__(self, obs_dim=163, action_dim=64, K=16, d_user=10, d_global=3,
                 hidden_dim=128, n_heads=4, n_attn_layers=2,
                 use_plackett_luce=True,
                 inject_global_to_per_user=True,
                 n_global_inject=2,
                 use_mean_field_critic=False,
                 mean_field_hidden_dim=64,
                 qos_drift_weights=None,
                 use_bucket_experts=False,
                 n_experts=4,
                 bucket_snr_min=-5.0,
                 bucket_snr_max=35.0,
                 use_c3po_cost_head=False,
                 c3po_n_points=7,
                 backpressure_beta=0.0,
                 backpressure_mode="queue",
                 simplified_action=False,
                 fix_drain: bool = False,
                 fix_order: bool = False):
        """
        Args:
            inject_global_to_per_user: 把 global 中后 n_global_inject 维（snr_norm, eps_norm）
                复制到每个 user token 与 per-user 特征拼接。让 attention 直接看到
                per-user × SNR/ε 交互，避免 LN 稀释。
            n_global_inject: 复制到 per-user 的 global 维度数（默认 2 = snr+eps，
                不复制 time_ratio 因为它对所有用户一样）
            use_mean_field_critic: critic 是否加 QoS 类均值场偏置
            mean_field_hidden_dim: 均值场 MLP 隐藏维度
            qos_drift_weights: 用于近似 QoS 类别的 drift 权重元组
            use_bucket_experts: 是否启用 SNR 桶专家多头（冻结共享主干，仅训练 G 个头）
            n_experts: 专家头数量（默认 4，对齐 Lagrangian/RolloutBuffer 的 SNR 桶数）
            bucket_snr_min/max: SNR 桶边界，用于 episode 级硬路由
            backpressure_beta: 队列反压先验强度。p_logit_k += beta * (Q_k/Q̄ - 1)，
                把队列压力直接注入功率份额 logits（max-weight/backpressure 结构先验）。
                注册为 buffer 随 checkpoint 传输：evaluate_snr_sweep.load_ppo_model 重建
                模型时不传行为开关，靠 load_state_dict(strict=False) 加载此值；β=0 精确无操作。
            backpressure_mode: 先验形式。"queue" = 纯 Q 加权（Run 10）；
                "queue_channel" = Q×ĥ 乘积（Run 10b，Tassiulas-Ephremides max-weight
                完整形式——Run 10 证明纯 Q 加权丢失信道转换效率导致 R_sum -5~6）；
                "mw_residual" = 闭式 max-weight 残差基座（Run 12）：p_raw =
                atanh(clip(p_mw,0.01,0.95)) + head_out，p_mw 对齐 sdma_baseline.py
                的 p∝Q 归一化规则——head 正交初始化(gain 0.01)下训练第 0 步功率
                即 SDMA 平价，PPO 只学围绕基座的残差修正。
                "mw_gated" = 负载感知门控基座（Run 13）：基座乘可学习标量门
                g(s)∈(0,1)，p_raw = g·atanh(clip(p_mw)) + head_out。门输入
                [mean_q, max_q, mean_h]（per_user 通道 1/1/0，均已归一化），
                零权重 + bias=4 初始化（g≈0.98 ≈ Run 12 平价起步）——深过载
                （队列高）学 g→1 保留 SDMA 平价基座，欠载（队列低）学 g→0
                让残差 head 自由省电（Run 11 已证高 SNR 深度省电可达）。
                mw_gate 层无条件构造（mode≠3 时不参与前向、无梯度）：评估侧
                load_ppo_model 用默认参数重建模型，只有模块结构恒在，
                load_state_dict(strict=False) 才能带上门权重 + mode buffer=3
                激活路径。四种 mode 均注册为 buffer 随 checkpoint 传输。
        """
        super().__init__()
        if not use_plackett_luce:
            raise NotImplementedError("ActorCriticAttn 仅支持 PL 模式")
        # P2 头裁剪实验臂（可解释性取证：d≡1、解码序近特征平坦；默认关=官方口径零变化）：
        # fix_drain: d 块恒为 1.0（env 口径），d head 的损失贡献清零（不学习）；
        # fix_order: 解码序恒为信道增益降序（gain-desc），PL z 头损失清零。
        self.fix_drain = bool(fix_drain)
        self.fix_order = bool(fix_order)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.K = K
        self.d_user = d_user
        self.d_global = d_global
        self.hidden_dim = hidden_dim
        self.use_plackett_luce = True
        # Arm A 精简动作：只学 (p, z) 两块。α 由 Meta split_mask 给出、d_k≡1
        # （可解释性：两块 σ≈1e-5 属死维/饱和），动作 4K→2K。
        self.simplified_action = bool(simplified_action)
        self.continuous_dim = 1 * K if self.simplified_action else 3 * K
        self.inject_global_to_per_user = bool(inject_global_to_per_user)
        self.n_global_inject = int(n_global_inject) if self.inject_global_to_per_user else 0

        # ---- 桶专家多头参数 ----
        self.use_bucket_experts = bool(use_bucket_experts)
        self.n_experts = int(n_experts) if self.use_bucket_experts else 1
        self.bucket_snr_min = float(bucket_snr_min)
        self.bucket_snr_max = float(bucket_snr_max)
        self.active_bucket = 0

        # ---- Backpressure 先验（非训练 buffer，值随 checkpoint 传输）----
        _mode_map = {"queue": 0, "queue_channel": 1, "mw_residual": 2, "mw_gated": 3}
        if backpressure_mode not in _mode_map:
            raise ValueError(
                f"backpressure_mode 必须是 {sorted(_mode_map)} 之一，got {backpressure_mode!r}"
            )
        self.register_buffer(
            "backpressure_beta_buf", torch.tensor(float(backpressure_beta))
        )
        self.register_buffer(
            "backpressure_mode_buf",
            torch.tensor(_mode_map[backpressure_mode], dtype=torch.long),
        )

        # ---- mw_gated 门控层（Run 13）----
        # 无条件构造：评估侧重建（默认 mode="queue"）也必须有此模块，
        # 否则 strict=False 加载不上门权重、mode=3 前向会缺模块。
        # 具体初始化在 _init_weights() 之后重设（见下方）。
        self.mw_gate = nn.Linear(3, 1)

        # ---- C3PO 7维逐SNR点 cost head ----
        self.use_c3po_cost_head = bool(use_c3po_cost_head)
        self.c3po_n_points = int(c3po_n_points) if self.use_c3po_cost_head else 0

        assert obs_dim == K * d_user + d_global, (
            f"obs_dim {obs_dim} != K*d_user + d_global = {K*d_user + d_global}"
        )

        # 实际输入到 user_embed 的维度（含 P2 注入的 global 信号）
        d_user_effective = d_user + self.n_global_inject

        # ---- Actor 主干 ----
        self.user_embed = nn.Sequential(
            nn.Linear(d_user_effective, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        # FiLM: global → (γ, β)，逐 channel 仿射调制 per-user embedding
        # 让 SNR/ε 信号成为 hidden_dim 维的 broadcast scale/shift，无法被后续 LN 稀释
        self.film = nn.Sequential(
            nn.Linear(d_global, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        # global concat 到每个 token 的额外路径（备份信号）
        self.global_proj = nn.Linear(d_global, hidden_dim)

        # 多层 Self-Attention
        self.attn_layers = nn.ModuleList()
        for _ in range(n_attn_layers):
            self.attn_layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True),
                "ln1": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                "ln2": nn.LayerNorm(hidden_dim),
            }))

        # per-user output head
        # 常规模式: 4 维 (alpha_raw, p_raw, d_raw, z_raw)
        # 精简模式: 2 维 (p_raw, z_raw) —— α 由 Meta 给、d_k≡1，动作 4K→2K
        self.head_dim = 2 if self.simplified_action else 4
        self.user_head = nn.Linear(hidden_dim, self.head_dim)
        orthogonal_init(self.user_head, gain=0.01)

        # 桶专家多头：从参考头深拷贝 G 个专家头，step-0 行为逐 bit 等价于 best。
        # 参考头冻结作为行为锚点；共享主干也冻结，仅专家头参与训练。
        if self.use_bucket_experts:
            self.expert_heads = nn.ModuleList([
                copy.deepcopy(self.user_head) for _ in range(self.n_experts)
            ])
            for p in self.user_head.parameters():
                p.requires_grad = False
        else:
            self.expert_heads = None

        # P3: SNR/ε 直通 actor head 的 bias 路径（让全局信号直接控制 α 和 p_total
        # 输出，绕过所有 LayerNorm，强化 α 跨 SNR/ε 的响应）
        # 用 zeros 初始化让训练初期等价于 P2 行为，逐步学习如何利用此路径
        # 精简模式不输出 α，故只建 p 的 bias 路径（保持参数集最小）
        self.global_to_alpha_bias = nn.Linear(d_global, 1) if not self.simplified_action else None
        self.global_to_p_bias = nn.Linear(d_global, 1)
        if self.global_to_alpha_bias is not None:
            nn.init.zeros_(self.global_to_alpha_bias.weight)
            nn.init.zeros_(self.global_to_alpha_bias.bias)
        nn.init.zeros_(self.global_to_p_bias.weight)
        nn.init.zeros_(self.global_to_p_bias.bias)

        # ---- Critic：独立路径 ----
        self.critic_user_embed = nn.Sequential(
            nn.Linear(d_user_effective, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.critic_global_proj = nn.Linear(d_global, hidden_dim)
        self.critic_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.critic_ln = nn.LayerNorm(hidden_dim)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        orthogonal_init(self.critic_head[-1], gain=1.0)

        # C3PO: 7维逐SNR点 cost head，与主 value head 共享 attention 特征
        self.critic_cost_head = None
        if self.use_c3po_cost_head:
            self.critic_cost_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, self.c3po_n_points),
            )
            orthogonal_init(self.critic_cost_head[-1], gain=1.0)

        # 连续动作 std（仅 3K 维：alpha, p_total, d_ratio）
        self.log_std = nn.Parameter(torch.full((self.continuous_dim,), -0.5))

        # ---- Mean-Field Critic 偏置（可选）----
        self.use_mean_field_critic = use_mean_field_critic
        self.mean_field_bias = None
        if self.use_mean_field_critic:
            self.mean_field_bias = MeanFieldCriticBias(
                d_user=d_user_effective,
                hidden_dim=mean_field_hidden_dim,
                n_classes=3,
                qos_drift_weights=qos_drift_weights,
            )

        self._init_weights()
        # mw_gate 初始化须在 _init_weights 之后（否则被正交 init 覆盖成输入依赖）：
        # 零权重 + bias=4 → g=sigmoid(4)≈0.982，初始 ≈ Run 12（mw_residual）平价
        nn.init.zeros_(self.mw_gate.weight)
        nn.init.constant_(self.mw_gate.bias, 4.0)

        # 桶专家模式：冻结共享主干（含 P3 bias 路径），仅专家头可训练
        if self.use_bucket_experts:
            for p in self.shared_trunk_params():
                p.requires_grad = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.user_head and m is not self.critic_head[-1]:
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _split_obs(self, obs):
        """obs (batch, 10K+3) → per_user (batch, K, d_user_effective), global (batch, d_global).

        env._get_observation 拼接顺序：
            [h_gains(K), queue(K), delta_queue(K), arrivals(K), power_hist(K),
             rate_hist(K), interference(K), fairness(K), split_mask(K),
             group_ids(K), time_ratio(1), snr_norm(1), eps_norm(1)]
        即前 K*d_user 是按特征块拼接（不是按用户）, reshape (10, K) 后转置 (K, 10)

        P2: 若 inject_global_to_per_user=True，把 global 中的 snr_norm 和 eps_norm
        （global 的最后 n_global_inject 维）复制 K 次拼到每个 user token，让 attention
        直接看到 per-user × global 交互，避免 LN 稀释 global 信号。
        """
        batch = obs.shape[0]
        K, d_user, d_global = self.K, self.d_user, self.d_global
        per_user_flat = obs[:, :K * d_user]  # (batch, K*d_user)
        per_user = per_user_flat.view(batch, d_user, K).transpose(1, 2).contiguous()  # (batch, K, d_user)
        global_feat = obs[:, K * d_user:K * d_user + d_global]  # (batch, d_global)

        if self.inject_global_to_per_user and self.n_global_inject > 0:
            # global 末尾 n_global_inject 维（snr_norm, eps_norm）复制广播到每个 user
            inject = global_feat[:, -self.n_global_inject:]  # (batch, n_inject)
            inject_expanded = inject.unsqueeze(1).expand(batch, K, -1)  # (batch, K, n_inject)
            per_user = torch.cat([per_user, inject_expanded], dim=-1)  # (batch, K, d_user + n_inject)

        return per_user, global_feat

    def _backpressure_shift(self, per_user):
        """Backpressure 先验：返回 (batch, K) 的功率 logit / p_raw 偏移。

        mode="queue"（Run 10）: beta * (Q_k/Q̄ - 1)
        mode="queue_channel"（Run 10b）: beta * (Q_k·ĥ_k/mean(Q·ĥ) - 1)
            Tassiulas-Ephremides max-weight 的完整乘积形式；ĥ 取 per_user[..., 0]
            = h_gains_norm（与策略所见信道同源），按乘积均值归一化后尺度无关。
        mode="mw_residual"（Run 12）: atanh(clip(p_mw, 0.01, 0.95))，
            p_mw = clip(K·Q_k/ΣQ, 0, 1) —— 对齐 sdma_baseline.py:50-54 的 p∝Q
            归一化规则（份额语义：队列相等时每用户拿满 P_k_max，深过载即 SDMA
            的全开行为）。注意这是「基座替换」而非加性偏移：调用方把返回值作为
            p_raw 的基座（head 输出为残差）。Q 全零时 p_mw 均匀 1/K。
            clamp 上限 0.95：atanh(1.0)=inf 必须有界，0.95 → env p=0.975W
            （与 SDMA 满功率 1.0W 差 2.5%，残差可修）。
        mode="mw_gated"（Run 13）: g·atanh(clip(p_mw, 0.01, 0.95))——同 Run 12
            基座但乘可学习标量门 g(s)∈(0,1)。门输入 [mean_q, max_q, mean_h]
            （q 取 per_user[...,1]=queue_norm，h 取 [...,0]=h_gains_norm）。
            g 初始 sigmoid(4)≈0.98（≈Run 12 平价）；欠载（队列低）时学 g→0
            释放 head 自由降功率，深过载时 g→1 保 SDMA 平价。

        queue / queue_channel 均中心化（batch 内对 K 用户均值为 0）：softmax 份额
        只取决于相对 logit 差，公共模偏移无信息却会被 tanh 饱和压缩——队列相等时
        反而压平 logit 差、抹平分配（Run 10 实现期由单调性测试暴露）。
        均值≈0（全空队列/全零信道）时守卫为全零；beta_buf=0 时精确无操作，
        官方口径逐 bit 不变（mw_residual / mw_gated 不看 beta_buf，mode 本身即开关）。
        """
        q = per_user[:, :, 1]
        mode = int(self.backpressure_mode_buf)
        if mode in (2, 3):
            q_sum = q.sum(dim=1, keepdim=True)
            p_mw = torch.where(
                q_sum > 1e-8,
                q * self.K / (q_sum + 1e-8),
                torch.full_like(q, 1.0 / self.K),
            )
            p_mw = p_mw.clamp(0.0, 1.0).clamp(0.01, 0.95)
            base = torch.atanh(p_mw)
            if mode == 2:
                return base
            h = per_user[:, :, 0]
            gate_x = torch.stack(
                [q.mean(dim=1), q.max(dim=1).values, h.mean(dim=1)], dim=1
            )  # (batch, 3)
            g = torch.sigmoid(self.mw_gate(gate_x)).squeeze(-1)  # (batch,)
            return g.unsqueeze(-1) * base
        if mode == 1:
            h = per_user[:, :, 0]
            w = q * h
            w_mean = w.mean(dim=1, keepdim=True)
            ratio = w / (w_mean + 1e-6) - 1.0
            ratio = torch.where(w_mean > 1e-6, ratio, torch.zeros_like(ratio))
        else:
            q_mean = q.mean(dim=1, keepdim=True)
            ratio = q / (q_mean + 1e-6) - 1.0
            ratio = torch.where(q_mean > 1e-6, ratio, torch.zeros_like(ratio))
        return self.backpressure_beta_buf * ratio

    def _actor_forward(self, obs):
        """Actor 主干：返回 (batch, K, hidden) per-user representations."""
        per_user, global_feat = self._split_obs(obs)
        batch = per_user.shape[0]

        # 1. embed per-user
        h = self.user_embed(per_user)  # (batch, K, hidden)

        # 2. FiLM modulation: γ, β = global → 2*hidden
        film = self.film(global_feat)  # (batch, 2*hidden)
        gamma, beta = film.chunk(2, dim=-1)  # (batch, hidden)
        # broadcast over K tokens: (batch, 1, hidden)
        h = h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # 3. 把 global 投影后加到每个 token（备份信号路径）
        g_proj = self.global_proj(global_feat).unsqueeze(1)  # (batch, 1, hidden)
        h = h + g_proj  # broadcast add

        # 4. Self-Attention layers
        for layer in self.attn_layers:
            attn_out, _ = layer["attn"](h, h, h)
            h = layer["ln1"](h + attn_out)
            ffn_out = layer["ffn"](h)
            h = layer["ln2"](h + ffn_out)

        return h  # (batch, K, hidden)

    def _snr_to_bucket(self, snr_db):
        """将 SNR(dB) 线性映射到 [0, n_experts-1] 的硬路由桶索引。"""
        if not self.use_bucket_experts:
            return 0
        span = self.bucket_snr_max - self.bucket_snr_min
        if span <= 0:
            return 0
        idx = int(np.clip(
            (float(snr_db) - self.bucket_snr_min) / span * self.n_experts,
            0, self.n_experts - 1,
        ))
        return idx

    def set_active_bucket(self, bucket_or_snr):
        """设置 episode 级硬路由桶。接受桶索引或 snr_db 标量。"""
        if isinstance(bucket_or_snr, (int, np.integer)):
            self.active_bucket = int(np.clip(bucket_or_snr, 0, self.n_experts - 1))
        else:
            self.active_bucket = self._snr_to_bucket(float(bucket_or_snr))
        return self.active_bucket

    def _head_forward(self, h, bucket_indices=None):
        """桶专家硬路由 head 选择。

        Args:
            h: (batch, K, hidden)
            bucket_indices: (batch,) 长整型桶索引；None 时使用 self.active_bucket
        Returns:
            head_out: (batch, K, 4)
        """
        if not self.use_bucket_experts or self.expert_heads is None:
            return self.user_head(h)

        batch = h.shape[0]
        device = h.device
        if bucket_indices is None:
            bucket_indices = torch.full(
                (batch,), self.active_bucket, dtype=torch.long, device=device
            )
        else:
            bucket_indices = torch.as_tensor(bucket_indices, dtype=torch.long, device=device)
            bucket_indices = bucket_indices.clamp(0, self.n_experts - 1)

        # 计算全部专家输出后按桶索引 gather；未选中的专家不产生梯度。
        expert_outputs = torch.stack([head(h) for head in self.expert_heads], dim=0)  # (G, B, K, head_dim)
        K = h.shape[1]
        # head_dim 随模式变（常规 4 / 精简 2），不能硬编码
        idx = bucket_indices.view(1, -1, 1, 1).expand(1, batch, K, self.head_dim)
        output = torch.gather(expert_outputs, 0, idx).squeeze(0)  # (B, K, head_dim)
        return output

    def sync_expert_heads_from_reference(self):
        """从冻结的参考 user_head 同步权重到所有专家头。

        用于从旧 checkpoint（无 expert_heads）恢复时保证 step-0 行为等价。
        """
        if not self.use_bucket_experts or self.expert_heads is None:
            return
        with torch.no_grad():
            for head in self.expert_heads:
                head.weight.copy_(self.user_head.weight)
                if head.bias is not None and self.user_head.bias is not None:
                    head.bias.copy_(self.user_head.bias)
        self._assert_step0_equivalence()

    def _assert_step0_equivalence(self, tol=1e-6):
        """step-0 等值断言：所有专家头在相同输入下输出逐 bit 等于参考头。"""
        if not self.use_bucket_experts or self.expert_heads is None:
            return
        self.eval()
        with torch.no_grad():
            obs = torch.randn(1, self.obs_dim, device=next(self.parameters()).device)
            h = self._actor_forward(obs)
            ref = self.user_head(h)
            for head in self.expert_heads:
                out = head(h)
                if not torch.allclose(ref, out, atol=tol, rtol=tol):
                    raise RuntimeError(
                        "Bucket-expert step-0 equivalence failed: "
                        "expert head output differs from reference head"
                    )

    def act(self, obs, deterministic=False, bucket_indices=None):
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)

        # Actor
        h = self._actor_forward(obs)  # (batch, K, hidden)
        head_out = self._head_forward(h, bucket_indices=bucket_indices)  # (batch, K, head_dim)
        # P3: SNR/ε 直通 head bias（broadcast 到所有 K 用户）
        per_user_act, global_feat_act = self._split_obs(obs)
        p_bias = self.global_to_p_bias(global_feat_act).unsqueeze(1)
        bp = self._backpressure_shift(per_user_act)

        if self.simplified_action:
            # 精简模式：head 输出 (p_raw, z_raw)，动作只含 p 块；α 由 Meta 给、d_k≡1
            p_raw = head_out[..., 0] + p_bias.squeeze(-1)   # (batch, K)
            if int(self.backpressure_mode_buf) in (2, 3):
                p_raw = bp + head_out[..., 0]
            else:
                p_raw = p_raw + bp
            z_raw = head_out[..., 1]
            continuous_mean = p_raw                          # (batch, 1K)
        else:
            a_bias = self.global_to_alpha_bias(global_feat_act).unsqueeze(1)
            # 拆 4 维
            alpha_raw = head_out[..., 0] + a_bias.squeeze(-1)   # (batch, K)
            p_raw     = head_out[..., 1] + p_bias.squeeze(-1)   # (batch, K)
            if int(self.backpressure_mode_buf) in (2, 3):
                # mw_residual / mw_gated：基座替换——p 块均值 = 闭式 max-weight(×门) + head 残差
                p_raw = bp + head_out[..., 1]
            else:
                p_raw = p_raw + bp
            d_raw     = head_out[..., 2]    # (batch, K)
            z_raw     = head_out[..., 3]    # (batch, K)

            # 拼成 3K 连续动作
            continuous_mean = torch.cat([alpha_raw, p_raw, d_raw], dim=-1)  # (batch, 3K)
        std = self.log_std.clamp(-3.0, 0.5).exp()

        if deterministic:
            action_squashed = torch.tanh(continuous_mean)
            continuous_log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(continuous_mean, std)
            action_raw = dist.rsample()
            continuous_log_prob = dist.log_prob(action_raw).sum(dim=-1)
            action_squashed = torch.tanh(action_raw)
            continuous_log_prob = continuous_log_prob - torch.sum(
                torch.log(1 - action_squashed.pow(2) + 1e-6), dim=-1
            )
        if self.fix_drain and not self.simplified_action:
            # P2a：d 块恒为 +1（env 口径 d=(a+1)/2=1.0），行为与"始终全排空"一致
            # 精简模式本就不含 d 块（d_k≡1 由 env 侧保证），无需此步
            action_squashed[..., 2 * self.K : 3 * self.K] = 1.0

        # PL 解码序
        z_scores = F.softplus(z_raw) + 1e-6  # (batch, K)
        if self.fix_order:
            # P2b：解码序恒为信道增益降序（per_user 通道 0 = h_gains_norm），PL 头不学习
            decoding_order = torch.argsort(
                per_user_act[:, :, 0], dim=-1, descending=True
            )
            pl_log_prob = torch.zeros(obs.shape[0], device=obs.device)
        elif deterministic:
            decoding_order = torch.argsort(z_scores, dim=-1, descending=True)
            pl_log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            decoding_order = plackett_luce_sample(z_scores)
            pl_log_prob = plackett_luce_log_prob(z_scores, decoding_order)

        total_log_prob = continuous_log_prob + pl_log_prob
        cont_log_prob = continuous_log_prob

        # Critic
        per_user_c, global_c = self._split_obs(obs)
        h_c = self.critic_user_embed(per_user_c)  # (batch, K, hidden)
        g_c = self.critic_global_proj(global_c).unsqueeze(1)
        h_c = h_c + g_c
        attn_c, _ = self.critic_attn(h_c, h_c, h_c)
        h_c = self.critic_ln(h_c + attn_c)
        pooled = h_c.mean(dim=1)  # (batch, hidden)
        value = self.critic_head(pooled).squeeze(-1)  # (batch,)
        if self.use_mean_field_critic and self.mean_field_bias is not None:
            value = value + self.mean_field_bias(per_user_c)

        if squeeze:
            action_squashed = action_squashed.squeeze(0)
            decoding_order = decoding_order.squeeze(0)
            total_log_prob = total_log_prob.squeeze(0)
            cont_log_prob = cont_log_prob.squeeze(0)
            value = value.squeeze(0)

        return action_squashed, decoding_order, total_log_prob, value, cont_log_prob

    def evaluate_actions(self, obs, actions, decoding_orders=None, bucket_indices=None):
        # Actor 主干
        h = self._actor_forward(obs)
        head_out = self._head_forward(h, bucket_indices=bucket_indices)
        # P3: SNR/ε 直通 head bias
        per_user_eval, global_feat_eval = self._split_obs(obs)
        p_bias = self.global_to_p_bias(global_feat_eval).unsqueeze(1)
        bp = self._backpressure_shift(per_user_eval)

        if self.simplified_action:
            p_raw = head_out[..., 0] + p_bias.squeeze(-1)
            if int(self.backpressure_mode_buf) in (2, 3):
                p_raw = bp + head_out[..., 0]
            else:
                p_raw = p_raw + bp
            z_raw = head_out[..., 1]
            continuous_mean = p_raw                      # (batch, 1K)
        else:
            a_bias = self.global_to_alpha_bias(global_feat_eval).unsqueeze(1)
            alpha_raw = head_out[..., 0] + a_bias.squeeze(-1)
            p_raw = head_out[..., 1] + p_bias.squeeze(-1)
            if int(self.backpressure_mode_buf) in (2, 3):
                p_raw = bp + head_out[..., 1]
            else:
                p_raw = p_raw + bp
            d_raw = head_out[..., 2]
            z_raw = head_out[..., 3]
            continuous_mean = torch.cat([alpha_raw, p_raw, d_raw], dim=-1)

        std = self.log_std.clamp(-3.0, 0.5).exp()
        dist = Normal(continuous_mean, std)
        actions_clipped = actions.clamp(-0.999, 0.999)
        action_raw = torch.atanh(actions_clipped)
        ls = dist.log_prob(action_raw)
        log1 = torch.log(1 - actions_clipped.pow(2) + 1e-6)
        if self.fix_drain and not self.simplified_action:
            # P2a：d 块 log 密度/雅可比贡献清零 → 该头不接收梯度（行为已由 act 固定）
            ls = ls.clone()
            log1 = log1.clone()
            ls[..., 2 * self.K : 3 * self.K] = 0.0
            log1[..., 2 * self.K : 3 * self.K] = 0.0
        continuous_log_prob = ls.sum(dim=-1) - log1.sum(dim=-1)
        ent = dist.entropy()
        if self.fix_drain and not self.simplified_action:
            ent = ent.clone()
            ent[..., 2 * self.K : 3 * self.K] = 0.0
        continuous_entropy = ent.sum(dim=-1)

        # Critic
        per_user_c, global_c = self._split_obs(obs)
        h_c = self.critic_user_embed(per_user_c)
        g_c = self.critic_global_proj(global_c).unsqueeze(1)
        h_c = h_c + g_c
        attn_c, _ = self.critic_attn(h_c, h_c, h_c)
        h_c = self.critic_ln(h_c + attn_c)
        pooled = h_c.mean(dim=1)
        value = self.critic_head(pooled).squeeze(-1)
        if self.use_mean_field_critic and self.mean_field_bias is not None:
            value = value + self.mean_field_bias(per_user_c)

        if decoding_orders is None:
            return continuous_log_prob, value, continuous_entropy, continuous_log_prob

        z_scores = F.softplus(z_raw) + 1e-6
        if self.fix_order:
            # P2b：PL 头不学习（解码序已固定 gain-desc）
            pl_log_prob = torch.zeros(decoding_orders.shape[0], device=decoding_orders.device)
        else:
            pl_log_prob = plackett_luce_log_prob(z_scores, decoding_orders)
        pl_entropy = torch.zeros_like(pl_log_prob)
        total_log_prob = continuous_log_prob + pl_log_prob
        total_entropy = continuous_entropy + pl_entropy

        return total_log_prob, value, total_entropy, continuous_log_prob

    def evaluate_cost_values(self, obs):
        """C3PO: 返回 7 维逐 SNR 点 cost 价值估计 V_c(s) ∈ (B, n_points).

        若未启用 C3PO cost head，返回 None（调用方需处理）。
        """
        if not self.use_c3po_cost_head or self.critic_cost_head is None:
            return None
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)
        per_user_c, global_c = self._split_obs(obs)
        h_c = self.critic_user_embed(per_user_c)
        g_c = self.critic_global_proj(global_c).unsqueeze(1)
        h_c = h_c + g_c
        attn_c, _ = self.critic_attn(h_c, h_c, h_c)
        h_c = self.critic_ln(h_c + attn_c)
        pooled = h_c.mean(dim=1)
        cost_values = self.critic_cost_head(pooled)  # (B, n_points)
        if squeeze:
            cost_values = cost_values.squeeze(0)
        return cost_values

    def shared_trunk_params(self):
        """返回 Actor 共享主干参数（桶专家模式下冻结）。

        mw_gate 刻意不在冻结集内：Run 13 的门控基座是**要参与学习**的负载感知
        标量（深过载学 g→1、欠载学 g→0），桶专家模式下它应处于可训练组
        （见 ppo.py 优化器组构造），而不是被 lr=0 冻结成恒 sigmoid(4)≈0.982。
        """
        shared_names = {
            "user_embed.", "film.", "global_proj.", "attn_layers.",
            "global_to_alpha_bias.", "global_to_p_bias.", "user_head.",
        }
        params = []
        for name, p in self.named_parameters():
            if any(name.startswith(prefix) for prefix in shared_names):
                params.append(p)
        return params

    def mw_gate_params(self):
        """mw_gated 门控层参数（mode≠3 时无梯度，Adam 自动跳过；始终可构造）。"""
        if not hasattr(self, "mw_gate"):
            return []
        return [self.mw_gate.weight, self.mw_gate.bias]

    def expert_head_params(self):
        """返回桶专家头参数（桶专家模式下可训练）。"""
        if not self.use_bucket_experts or self.expert_heads is None:
            return []
        return list(self.expert_heads.parameters())

    def actor_params(self):
        """返回 Actor 可训练参数。桶专家模式下仅专家头参与训练。"""
        if self.use_bucket_experts:
            return self.expert_head_params()
        params = []
        for name, p in self.named_parameters():
            if name.startswith("critic_") or name.startswith("mean_field_") or name == "log_std":
                continue
            params.append(p)
        return params

    def critic_params(self):
        return [p for n, p in self.named_parameters()
                if n.startswith("critic_") or n.startswith("mean_field_")]

    def log_std_params(self):
        return [self.log_std]


__all__ = ["ActorCritic", "ActorCriticAttn",
           "plackett_luce_log_prob", "plackett_luce_sample"]
