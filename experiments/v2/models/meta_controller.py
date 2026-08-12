"""v2 Meta-Controller: 新增 per-bucket alpha 缩放（SNR 条件化门控）

核心升级：在原始 per-group alpha 上叠加 per-bucket 缩放因子。
Meta 新增 bucket_alpha_head 头，输出 7 个 SNR 桶（0/5/10/15/20/25/30 dB）的缩放因子。
当前 SNR 所在桶的缩放因子乘以 per-group alpha，实现"低 SNR 不同行为"的学习先验。

与原始 MetaActorCritic 的 diff：
- 新增 bucket_alpha_head（Linear: hidden_dim → n_alpha_buckets）
- forward 多返回 bucket_raw（4th 元素）
- act 中计算 bucket_scale 并缩放 user_alphas，log_prob 做相应调整
- evaluate_actions 提供 bucket_scale 用于 log_prob 反调整
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal
import numpy as np
from typing import Optional, Tuple, Dict


class MetaActorCritic(nn.Module):
    """v2 Meta-Controller: Self-Attention PPO + per-bucket alpha 缩放"""

    def __init__(self, K: int, d_user: int = 6, hidden_dim: int = 64,
                 n_heads: int = 4, n_groups: int = 4,
                 group_tau_init: float = 2.0, group_tau_min: float = 0.5,
                 balance_coef: float = 0.1,
                 group_level_alpha: bool = False,
                 group_alpha_min: float = 0.1,
                 group_alpha_max: float = 0.95,
                 group_alpha_log_std_init: float = -0.5,
                 use_bucket_meta_alpha: bool = True,
                 n_alpha_buckets: int = 7):
        super().__init__()
        self.K = K
        self.d_user = d_user
        self.hidden_dim = hidden_dim
        self.n_groups = n_groups
        self.use_bucket_meta_alpha = bool(use_bucket_meta_alpha)
        self.n_alpha_buckets = int(n_alpha_buckets)

        self.group_level_alpha = bool(group_level_alpha)
        self.group_alpha_min = float(group_alpha_min)
        self.group_alpha_max = float(group_alpha_max)
        self._alpha_span = self.group_alpha_max - self.group_alpha_min

        self.group_tau_init = float(group_tau_init)
        self.group_tau_min = float(group_tau_min)
        self.group_tau = float(group_tau_init)

        self.balance_coef = float(balance_coef)

        # 用户特征嵌入
        self.embed = nn.Sequential(
            nn.Linear(d_user, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # Alpha head（per-group）
        if self.group_level_alpha:
            self.alpha_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_groups),
            )
            self.alpha_log_std = nn.Parameter(
                torch.full((n_groups,), float(group_alpha_log_std_init)))
            self.actor_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.actor_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        # Group head
        self.group_head = nn.Linear(hidden_dim, n_groups)

        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ---- v2 升级：per-bucket alpha 缩放头 ----
        # 输出 n_alpha_buckets 个原始值，sigmoid 后乘 2 得 [0, 2] 缩放因子
        # 初始化偏置为 0（sigmoid(0)=0.5 → 缩放 1.0，无操作）
        self.bucket_alpha_head = nn.Linear(hidden_dim, n_alpha_buckets)
        nn.init.zeros_(self.bucket_alpha_head.weight)
        nn.init.zeros_(self.bucket_alpha_head.bias)

        # 缓存
        self.last_bucket_scale = 1.0
        self.last_bucket_raw = None
        self.last_bucket_idx = 0
        self.last_raw_alpha = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _snr_to_bucket_idx(self, snr_norm: float) -> int:
        """将归一化 SNR 映射到桶索引(0~6, 对应 0/5/10/15/20/25/30 dB)。"""
        snr_db = snr_norm * 40.0 - 5.0  # 反归一化：[-5,35] → [-5,35]
        idx = round((snr_db - 0.0) / 5.0)
        return max(0, min(idx, self.n_alpha_buckets - 1))

    def _compute_bucket_scale(self, snr_norm: float) -> torch.Tensor:
        """计算当前 SNR 桶的 alpha 缩放因子。

        Args:
            snr_norm: 当前 step 的归一化 SNR
        Returns:
            标量缩放因子 tensor
        """
        if not self.use_bucket_meta_alpha:
            return torch.ones(1, device=next(self.parameters()).device)
        pooled = self._last_pooled  # (batch, hidden_dim)，来自最近一次 forward
        bucket_raw = self.bucket_alpha_head(pooled)  # (batch, n_buckets)
        self.last_bucket_raw = bucket_raw.detach()
        self.last_bucket_idx = self._snr_to_bucket_idx(snr_norm)
        raw = bucket_raw[0, self.last_bucket_idx]
        scale = raw.sigmoid() * 2.0  # (0, 2)，初始化时 ~1.0
        self.last_bucket_scale = scale.item()
        return scale

    def anneal_temperature(self, progress: float) -> float:
        p = float(max(0.0, min(1.0, progress)))
        self.group_tau = max(self.group_tau_min, self.group_tau_init * (1.0 - p))
        return self.group_tau

    def forward(self, user_features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """前向传播，返回 4 元素（新增 bucket_raw）。"""
        x = self.embed(user_features)
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        if self.group_level_alpha:
            pooled_alpha = x.mean(dim=1)
            split_or_alpha = self.alpha_head(pooled_alpha)
        else:
            split_or_alpha = self.actor_head(x).squeeze(-1)

        raw_group_logits = self.group_head(x)
        tau = max(float(self.group_tau), 1e-6)
        group_logits = raw_group_logits / tau

        pooled = x.mean(dim=1)
        value = self.critic_head(pooled)

        # v2: bucket_raw
        if self.use_bucket_meta_alpha:
            bucket_raw = self.bucket_alpha_head(pooled)  # (batch, n_buckets)
        else:
            bucket_raw = None

        self._last_pooled = pooled  # 缓存供 act/evaluate_actions 中的 bucket_scale 计算

        return split_or_alpha, group_logits, value, bucket_raw

    def _map_alpha(self, raw_alpha: torch.Tensor) -> torch.Tensor:
        return self.group_alpha_min + self._alpha_span * torch.sigmoid(raw_alpha)

    def expand_user_alphas(self, group_alphas: torch.Tensor,
                           group_ids: torch.Tensor) -> torch.Tensor:
        return torch.gather(group_alphas, dim=-1, index=group_ids.long())

    def act(self, user_features: torch.Tensor, deterministic: bool = False
            ) -> Tuple[torch.Tensor, ...]:
        """采样动作，返回 (scaled_alphas, group_ids, adjusted_log_prob, value)。"""
        if user_features.dim() == 2:
            user_features = user_features.unsqueeze(0)

        split_or_alpha, group_logits, value, bucket_raw = self.forward(user_features)

        # 获取当前 SNR 归一化值（从用户特征 dim 7）
        snr_norm = float(user_features[0, :, 7].mean().item())

        group_logits = group_logits.squeeze(0)
        value = value.squeeze()

        # Group
        group_dist = Categorical(logits=group_logits)
        group_ids = group_logits.argmax(dim=-1) if deterministic else group_dist.sample()
        group_log_prob = group_dist.log_prob(group_ids).sum()

        if self.group_level_alpha:
            # Per-group Gaussian alpha
            alpha_mean = split_or_alpha.squeeze(0)
            alpha_std = torch.exp(self.alpha_log_std).clamp(min=1e-3, max=2.0)
            alpha_dist = Normal(alpha_mean, alpha_std)
            if deterministic:
                raw_alpha = alpha_mean
            else:
                raw_alpha = alpha_dist.sample()
            raw_alpha_log_prob = alpha_dist.log_prob(raw_alpha).sum()
            mapped_group_alphas = self._map_alpha(raw_alpha)
            user_alphas = self.expand_user_alphas(mapped_group_alphas, group_ids)

            # v2: per-bucket 缩放
            bucket_scale = self._compute_bucket_scale(snr_norm)
            user_alphas = user_alphas * bucket_scale

            # 调整 log_prob：p(scaled) = p(raw) / product(scale per group)
            # 缩放是 per-user 但采样是 per-group（n_groups=4），
            # 调整量 = -n_groups * log(scale)
            adjusted_log_prob = raw_alpha_log_prob - self.n_groups * math.log(
                max(self.last_bucket_scale, 1e-8))

            self.last_raw_alpha = raw_alpha.detach()
            total_log_prob = adjusted_log_prob + group_log_prob
            return user_alphas, group_ids, total_log_prob, value
        else:
            # Legacy Bernoulli split（无 bucket 缩放）
            split_logits = split_or_alpha.squeeze(0)
            split_dist = Bernoulli(logits=split_logits)
            if deterministic:
                split_mask = (split_logits > 0).float()
            else:
                split_mask = split_dist.sample()
            split_log_prob = split_dist.log_prob(split_mask).sum()
            total_log_prob = split_log_prob + group_log_prob
            return split_mask, group_ids, total_log_prob, value

    def evaluate_actions(self, user_features: torch.Tensor,
                         split_actions: torch.Tensor,
                         group_actions: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, ...]:
        """评估已有动作的 log_prob, value, entropy。"""
        split_or_alpha, group_logits, values, bucket_raw = self.forward(user_features)

        # Group
        group_dist = Categorical(logits=group_logits)
        if group_actions is not None:
            group_log_prob = group_dist.log_prob(group_actions.long()).sum(dim=-1)
        else:
            group_log_prob = torch.zeros(values.shape[0], device=values.device)
        group_entropy = group_dist.entropy().sum(dim=-1)

        if self.group_level_alpha:
            # split_actions 是 raw_group_alphas (batch, n_groups)，来自 buffer
            # （_meta_decide 存储的是 meta_ac.last_raw_alpha，不是 scaled user_alphas）
            alpha_mean = split_or_alpha  # (batch, n_groups)
            alpha_std = torch.exp(self.alpha_log_std).clamp(min=1e-3, max=2.0)
            alpha_dist = Normal(alpha_mean, alpha_std)
            alpha_log_prob = alpha_dist.log_prob(split_actions).sum(dim=-1)

            # v2: 调整 log_prob 匹配 act() 的调整
            # act() 中 user_alphas 被 bucket_scale 缩放，log_prob 减去了 n_groups*log(scale)
            # evaluate_actions 评估的是 raw_group_alphas 的 log_prob，需做同样调整
            if self.use_bucket_meta_alpha:
                snr_norm = float(user_features[0, :, 7].mean().item())
                bucket_scale = self._compute_bucket_scale(snr_norm)
                scale = bucket_scale.expand(alpha_log_prob.shape[0])
                alpha_log_prob = alpha_log_prob - self.n_groups * torch.log(scale.clamp(min=1e-8))

            split_entropy = alpha_dist.entropy().sum(dim=-1)
            total_log_prob = alpha_log_prob + group_log_prob
        else:
            split_logits = split_or_alpha
            split_dist = Bernoulli(logits=split_logits)
            split_log_prob = split_dist.log_prob(split_actions).sum(dim=-1)
            split_entropy = split_dist.entropy().sum(dim=-1)
            total_log_prob = split_log_prob + group_log_prob

        total_entropy = split_entropy + group_entropy
        return total_log_prob, values.squeeze(-1), split_entropy, group_entropy, total_entropy

    def compute_balance_reg(self, user_features: torch.Tensor) -> torch.Tensor:
        if self.balance_coef <= 0.0:
            return torch.zeros((), device=user_features.device)
        _, group_logits, _, _ = self.forward(user_features)
        probs = F.softmax(group_logits, dim=-1)
        mean_group_probs = probs.mean(dim=(0, 1))
        mean_group_probs = mean_group_probs.clamp(min=1e-8)
        entropy = -(mean_group_probs * mean_group_probs.log()).sum()
        return -self.balance_coef * entropy