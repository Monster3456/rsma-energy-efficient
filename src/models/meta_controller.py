"""慢时间尺度 Meta-Controller: Self-Attention PPO

使用多头自注意力机制捕捉用户间干扰关系和分组结构。

Task#38 Group-level α 架构:
  group_level_alpha=False (默认以保证向后兼容):
    - 输出逐用户分裂决策 δ_k ∈ {0,1}^K (Bernoulli) + 逐用户分组 g_k ∈ {0,...,G-1}^K
  group_level_alpha=True (Task#38 新架构):
    - 输出 per-group 连续 α ∈ [α_min, α_max]^G (Gaussian) + 逐用户分组 g_k
    - per-user α[k] = group_alphas[g_k]
    - 搜索空间从 ~K·G 降至 ~G + K（指数级下降）

决策周期: 每 T_s 个时隙执行一次。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


class MetaActorCritic(nn.Module):
    """Self-Attention PPO 网络用于慢时间尺度分裂决策与用户分组。
    
    架构:
        用户嵌入 (d_user → hidden_dim)
        → Self-Attention (MultiHeadAttention, n_heads)
        → Split Actor: per-user logits (hidden_dim → 1) → Bernoulli
        → Group Actor: per-user logits (hidden_dim → G) → Categorical
        → Critic: mean pooling → value (hidden_dim → 1)
    
    Args:
        K: 用户数
        d_user: 每用户特征维度 (默认6)
        hidden_dim: 隐藏层维度 (默认64)
        n_heads: 注意力头数 (默认4)
        n_groups: 最大分组数 G (默认4)
    """
    
    def __init__(self, K: int, d_user: int = 6, hidden_dim: int = 64,
                 n_heads: int = 4, n_groups: int = 4,
                 group_tau_init: float = 2.0, group_tau_min: float = 0.5,
                 balance_coef: float = 0.1,
                 group_level_alpha: bool = False,
                 group_alpha_min: float = 0.1,
                 group_alpha_max: float = 0.95,
                 group_alpha_log_std_init: float = -0.5):
        super().__init__()
        self.K = K
        self.d_user = d_user
        self.hidden_dim = hidden_dim
        self.n_groups = n_groups

        # ---- Task#38: Group-level α 架构开关与映射参数 ----
        self.group_level_alpha = bool(group_level_alpha)
        self.group_alpha_min = float(group_alpha_min)
        self.group_alpha_max = float(group_alpha_max)
        self._alpha_span = self.group_alpha_max - self.group_alpha_min

        # ---- Group head 温度退火参数 ----
        # tau 大 → 高温 → 高熵高探索；tau 小 → 低温 → 决策更尖锐
        # 仅作用于 group head（分裂 head 不受影响）
        self.group_tau_init = float(group_tau_init)
        self.group_tau_min = float(group_tau_min)
        self.group_tau = float(group_tau_init)

        # ---- 分组平衡正则系数 ----
        # 0 → 关闭；>0 → 鼓励 batch 中各组被选中的边缘概率均匀
        self.balance_coef = float(balance_coef)
        
        # 用户特征嵌入
        self.embed = nn.Sequential(
            nn.Linear(d_user, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Self-Attention: 捕捉用户间干扰关系
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # 前馈层 (attention 后)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # Split Actor head:
        #   - group_level_alpha=False: 逐用户分裂 logit (Bernoulli)
        #   - group_level_alpha=True : pooled → per-group raw alpha logits (Gaussian)
        if self.group_level_alpha:
            # Pooled global → per-group raw alpha mean
            self.alpha_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_groups),
            )
            # 可学习 log_std（各组独立）
            self.alpha_log_std = nn.Parameter(
                torch.full((n_groups,), float(group_alpha_log_std_init))
            )
            # 保留名为 actor_head 以便老 ckpt 能加载；本架构中不往该 head 走
            self.actor_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.actor_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),  # 输出 1 个 logit per user
            )
        
        # Group Actor head: 每用户输出 G 个 logit → Categorical
        self.group_head = nn.Linear(hidden_dim, n_groups)  # (hidden_dim → G)
        
        # Critic head: 全局状态价值
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def anneal_temperature(self, progress: float) -> float:
        """根据训练进度退火 group head 的温度。

        progress 从 0 (训练开始) → 1 (训练结束)。
        tau = max(tau_min, tau_init * (1 - progress))

        返回当前 tau（便于训练日志记录）。
        仅影响分组决策，不影响分裂决策。
        """
        p = float(max(0.0, min(1.0, progress)))
        self.group_tau = max(self.group_tau_min, self.group_tau_init * (1.0 - p))
        return self.group_tau
    
    def forward(self, user_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播。
        
        Args:
            user_features: (batch, K, d_user) 每用户特征矩阵
            
        Returns:
            split_or_alpha_mean:
                - group_level_alpha=False: (batch, K) per-user split logits
                - group_level_alpha=True : (batch, n_groups) per-group raw alpha mean
            group_logits: (batch, K, G) 每用户分组 logits
            value: (batch, 1) 状态价值
        """
        # 嵌入
        x = self.embed(user_features)  # (batch, K, hidden)
        
        # Self-Attention (残差连接)
        attn_out, _ = self.attention(x, x, x)  # (batch, K, hidden)
        x = self.attn_norm(x + attn_out)
        
        # FFN (残差连接)
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)  # (batch, K, hidden)
        
        # Split / Alpha head
        if self.group_level_alpha:
            # Pool over users → per-group raw alpha logits (mean of Gaussian)
            pooled_alpha = x.mean(dim=1)              # (batch, hidden)
            split_or_alpha = self.alpha_head(pooled_alpha)  # (batch, n_groups)
        else:
            split_or_alpha = self.actor_head(x).squeeze(-1)  # (batch, K)
        
        # Group Actor: per-user group logits（除以温度 tau，温度退火仅作用于分组 head）
        raw_group_logits = self.group_head(x)  # (batch, K, G)
        tau = max(float(self.group_tau), 1e-6)
        group_logits = raw_group_logits / tau
        
        # Critic: global value via mean pooling
        pooled = x.mean(dim=1)  # (batch, hidden)
        value = self.critic_head(pooled)  # (batch, 1)
        
        return split_or_alpha, group_logits, value
    
    def _map_alpha(self, raw_alpha: torch.Tensor) -> torch.Tensor:
        """将原始 Gaussian 采样 raw_alpha 映射到 [α_min, α_max]。

        α = α_min + (α_max - α_min) * sigmoid(raw_alpha)
        """
        return self.group_alpha_min + self._alpha_span * torch.sigmoid(raw_alpha)

    def expand_user_alphas(self, group_alphas: torch.Tensor,
                           group_ids: torch.Tensor) -> torch.Tensor:
        """将 (..., n_groups) 的 group α 展开为 (..., K) 的 per-user α。

        Args:
            group_alphas: (..., n_groups) 已映射后的 α 值
            group_ids: (..., K) long tensor 列出每用户所属组
        Returns:
            user_alphas: (..., K)
        """
        return torch.gather(group_alphas, dim=-1, index=group_ids.long())

    def act(self, user_features: torch.Tensor, deterministic: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作（split/α + group）。

        Args:
            user_features: (K, d_user) 或 (1, K, d_user)
            deterministic: 是否确定性决策 (用于评估)

        Returns:
            在 group_level_alpha=False (默认) 下：
                split_mask: (K,) 二值分裂决策
                group_ids:  (K,) 每用户分组 ID (0 ~ G-1)
                log_prob:   scalar 总对数概率 (split + group)
                value:      scalar 状态价值
            在 group_level_alpha=True 下：
                user_alphas: (K,) per-user 连续 α 值 (已映射到 [α_min, α_max])
                group_ids:   (K,)
                log_prob:    scalar (alpha Gaussian + group Categorical)
                value:       scalar
            另将原始采样以供 buffer 存储使用，存于 self.last_raw_alpha。
        """
        if user_features.dim() == 2:
            user_features = user_features.unsqueeze(0)  # (1, K, d_user)

        split_or_alpha, group_logits, value = self.forward(user_features)
        group_logits = group_logits.squeeze(0)  # (K, G)
        value = value.squeeze()  # scalar

        # ---- Group: Categorical per user (两种架构均需) ----
        group_dist = Categorical(logits=group_logits)
        if deterministic:
            group_ids = group_logits.argmax(dim=-1)  # (K,)
        else:
            group_ids = group_dist.sample()  # (K,)
        group_log_prob = group_dist.log_prob(group_ids).sum()  # sum over K users

        if self.group_level_alpha:
            # ---- Per-group Gaussian α ----
            alpha_mean = split_or_alpha.squeeze(0)  # (n_groups,)
            alpha_std = torch.exp(self.alpha_log_std).clamp(min=1e-3, max=2.0)
            alpha_dist = Normal(alpha_mean, alpha_std)
            if deterministic:
                raw_alpha = alpha_mean
            else:
                raw_alpha = alpha_dist.sample()  # (n_groups,) 不求 grad，与 Bernoulli/Categorical 一致
            alpha_log_prob = alpha_dist.log_prob(raw_alpha).sum()
            mapped_group_alphas = self._map_alpha(raw_alpha)  # (n_groups,)
            user_alphas = self.expand_user_alphas(mapped_group_alphas, group_ids)  # (K,)

            total_log_prob = alpha_log_prob + group_log_prob
            # 缓存原始采样以供外部存储 (numpy)
            self.last_raw_alpha = raw_alpha.detach()
            return user_alphas, group_ids, total_log_prob, value

        # ---- Legacy per-user Bernoulli split ----
        split_logits = split_or_alpha.squeeze(0)  # (K,)
        split_dist = Bernoulli(logits=split_logits)
        if deterministic:
            split_mask = (split_logits > 0).float()
        else:
            split_mask = split_dist.sample()  # (K,)
        split_log_prob = split_dist.log_prob(split_mask).sum()
        total_log_prob = split_log_prob + group_log_prob
        return split_mask, group_ids, total_log_prob, value
    
    def evaluate_actions(self, user_features: torch.Tensor,
                         split_actions: torch.Tensor,
                         group_actions: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估已有动作的 log_prob, value, entropy (用于 PPO 更新)。

        Args:
            user_features: (batch, K, d_user)
            split_actions:
                - group_level_alpha=False: (batch, K) 历史分裂决策 (0/1 float)
                - group_level_alpha=True : (batch, n_groups) 历史原始 α 采样 (raw)
            group_actions: (batch, K) 历史分组决策 (可选，向后兼容)

        Returns:
            total_log_prob: (batch,) 总对数概率 (alpha/split + group)
            values: (batch,)
            split_entropy: (batch,) (Bernoulli entropy or Gaussian entropy of α)
            group_entropy: (batch,)
            total_entropy: (batch,)
        """
        split_or_alpha, group_logits, values = self.forward(user_features)

        # Group
        group_dist = Categorical(logits=group_logits)
        if group_actions is not None:
            group_log_prob = group_dist.log_prob(group_actions.long()).sum(dim=-1)
        else:
            # 向后兼容：无分组动作时 log_prob = 0
            group_log_prob = torch.zeros(values.shape[0], device=values.device)
        group_entropy = group_dist.entropy().sum(dim=-1)  # (batch,)

        if self.group_level_alpha:
            # split_actions 实际为 raw_group_alphas: (batch, n_groups)
            alpha_mean = split_or_alpha  # (batch, n_groups)
            alpha_std = torch.exp(self.alpha_log_std).clamp(min=1e-3, max=2.0)
            alpha_dist = Normal(alpha_mean, alpha_std)
            alpha_log_prob = alpha_dist.log_prob(split_actions).sum(dim=-1)  # (batch,)
            split_entropy = alpha_dist.entropy().sum(dim=-1)
            total_log_prob = alpha_log_prob + group_log_prob
        else:
            split_logits = split_or_alpha  # (batch, K)
            split_dist = Bernoulli(logits=split_logits)
            split_log_prob = split_dist.log_prob(split_actions).sum(dim=-1)  # (batch,)
            split_entropy = split_dist.entropy().sum(dim=-1)
            total_log_prob = split_log_prob + group_log_prob

        total_entropy = split_entropy + group_entropy
        return total_log_prob, values.squeeze(-1), split_entropy, group_entropy, total_entropy

    def compute_balance_reg(self, user_features: torch.Tensor) -> torch.Tensor:
        """分组平衡正则项。

        计算 batch 中各组被选中的边缘概率均值 \bar p_g，并鼓励其足够均匀：
            balance_reg = -balance_coef * H(\bar p)
        其中 H(·) 为香农熵；balance_coef <= 0 时返回 0（关闭）。

        该项可作为额外 loss 加到 PPO policy loss 上。
        """
        if self.balance_coef <= 0.0:
            return torch.zeros((), device=user_features.device)
        _, group_logits, _ = self.forward(user_features)  # (B, K, G)
        probs = F.softmax(group_logits, dim=-1)            # (B, K, G)
        # 边缘分组概率：对 batch 和用户取平均
        mean_group_probs = probs.mean(dim=(0, 1))          # (G,)
        mean_group_probs = mean_group_probs.clamp(min=1e-8)
        entropy = -(mean_group_probs * mean_group_probs.log()).sum()
        return -self.balance_coef * entropy


class MetaRolloutBuffer:
    """Meta-Controller 的 Rollout Buffer。
    
    存储慢时间尺度 transitions: (user_features, split_mask, group_ids, log_prob, reward, value, done)
    """
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.reset()
    
    def reset(self):
        self.user_features = []
        self.actions = []           # split actions
        self.group_actions = []     # group actions
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    @property
    def size(self) -> int:
        return len(self.rewards)
    
    def store(self, user_features: np.ndarray, action: np.ndarray,
              log_prob: float, reward: float, value: float, done: bool = False,
              group_action: Optional[np.ndarray] = None):
        """存储一个 meta transition。"""
        self.user_features.append(user_features)
        self.actions.append(action)
        self.group_actions.append(group_action if group_action is not None
                                  else np.zeros_like(action, dtype=np.int64))
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
        # 超过容量时丢弃最旧的
        if len(self.rewards) > self.max_size:
            self.user_features.pop(0)
            self.actions.pop(0)
            self.group_actions.pop(0)
            self.log_probs.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.dones.pop(0)
    
    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float,
                                        last_value: float = 0.0):
        """计算 GAE advantages 和 returns。"""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones, dtype=float)
        
        n = len(rewards)
        advantages = np.zeros(n)
        last_gae = 0.0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return returns, advantages
    
    def get_batches(self, batch_size: int, device: str = 'cpu',
                    gamma: float = 0.95, gae_lambda: float = 0.95,
                    last_value: float = 0.0):
        """生成 mini-batch 用于 PPO 更新。
        
        Args:
            batch_size: mini-batch 大小
            device: 计算设备
            gamma: 折扣因子
            gae_lambda: GAE lambda
            last_value: 最后状态的 value 估计
        """
        n = self.size
        returns, advantages = self.compute_returns_and_advantages(
            gamma=gamma, gae_lambda=gae_lambda, last_value=last_value
        )
        
        # 转换为 tensor
        features_t = torch.FloatTensor(np.array(self.user_features)).to(device)
        actions_t = torch.FloatTensor(np.array(self.actions)).to(device)
        group_actions_t = torch.LongTensor(np.array(self.group_actions)).to(device)
        old_log_probs_t = torch.FloatTensor(np.array(self.log_probs)).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)
        
        # 标准化 advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # 生成随机 mini-batch 索引
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield {
                'user_features': features_t[idx],
                'split_actions': actions_t[idx],
                'group_actions': group_actions_t[idx],
                'old_log_probs': old_log_probs_t[idx],
                'returns': returns_t[idx],
                'advantages': advantages_t[idx],
            }


class MetaPPOTrainer:
    """Meta-Controller 的 PPO 训练器。"""
    
    def __init__(self, meta_ac: MetaActorCritic, config):
        """
        Args:
            meta_ac: MetaActorCritic 网络
            config: HierarchicalConfig
        """
        self.meta_ac = meta_ac
        self.config = config
        
        self.optimizer = torch.optim.Adam(
            meta_ac.parameters(), lr=config.meta_lr
        )
        self.clip_ratio = config.meta_clip_ratio
        self.entropy_coef = config.meta_entropy_coef
        self.group_entropy_coef = getattr(config, 'meta_group_entropy_coef', 0.05)
        self.n_epochs = config.meta_n_epochs
        self.max_grad_norm = config.meta_max_grad_norm
        self.batch_size = max(16, config.meta_update_threshold // 4)
    
    def update(self, buffer: MetaRolloutBuffer, last_value: float = 0.0) -> Dict:
        """执行 Meta PPO 更新。
        
        Args:
            buffer: MetaRolloutBuffer
            last_value: 最后一个状态的 value (用于 GAE 计算)
            
        Returns:
            训练统计信息 dict
        """
        device = next(self.meta_ac.parameters()).device
        total_loss_sum = 0.0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            for batch in buffer.get_batches(
                self.batch_size, device,
                gamma=self.config.meta_gamma,
                gae_lambda=self.config.meta_gae_lambda,
                last_value=last_value,
            ):
                features = batch['user_features']
                split_actions = batch['split_actions']
                group_actions = batch['group_actions']
                old_log_probs = batch['old_log_probs']
                returns = batch['returns']
                advantages = batch['advantages']
                
                # 评估当前策略下的 log_prob, value, entropy
                new_log_probs, values, split_entropy, group_entropy, total_entropy = (
                    self.meta_ac.evaluate_actions(features, split_actions, group_actions)
                )
                
                # PPO clip loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns)
                
                # Entropy bonus: split 和 group 分别加权
                split_entropy_loss = -split_entropy.mean()
                group_entropy_loss = -group_entropy.mean()

                # 分组平衡正则 (可选，balance_coef=0 时为 0)
                balance_reg = self.meta_ac.compute_balance_reg(features)

                # Total loss
                loss = (policy_loss
                        + 0.5 * value_loss
                        + self.entropy_coef * split_entropy_loss
                        + self.group_entropy_coef * group_entropy_loss
                        + balance_reg)
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.meta_ac.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss_sum += loss.item()
                n_updates += 1
        
        # 清空 buffer
        buffer.reset()

        return {
            'meta_loss': total_loss_sum / max(n_updates, 1),
            'meta_updates': n_updates,
            'meta_group_tau': float(getattr(self.meta_ac, 'group_tau', 0.0)),
        }


__all__ = ["MetaActorCritic", "MetaRolloutBuffer", "MetaPPOTrainer"]
