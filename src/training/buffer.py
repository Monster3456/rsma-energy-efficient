"""经验回放缓冲区

存储PPO rollout数据，支持GAE优势函数计算。
"""
import numpy as np
import torch


class RolloutBuffer:
    """PPO Rollout Buffer

    存储一个rollout周期的(obs, action, reward, value, log_prob, done)
    支持GAE计算和mini-batch采样。
    当启用 Plackett-Luce 模式时，额外存储解码顺序 decoding_orders。
    """

    def __init__(self, rollout_steps, obs_dim, action_dim, gamma=0.99, gae_lambda=0.95,
                 device="cpu", decoding_order_dim=0,
                 n_snr_buckets=4, snr_min=-5.0, snr_max=35.0):
        self.rollout_steps = rollout_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.decoding_order_dim = decoding_order_dim  # K (0=禁用PL)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # per-SNR-bucket advantage normalization 参数
        # 纠正域随机化下低SNR段梯度被高SNR段压制导致的 delivery 欠学习
        self.n_snr_buckets = n_snr_buckets
        self.snr_min = snr_min
        self.snr_max = snr_max

        # 预分配存储
        self.observations = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, action_dim), dtype=np.float32)
        self.rewards = np.zeros(rollout_steps, dtype=np.float32)
        self.values = np.zeros(rollout_steps, dtype=np.float32)
        self.log_probs = np.zeros(rollout_steps, dtype=np.float32)
        # 连续动作部分的 log_prob（不含 PL 排列项）— 供 PPO 估计 KL 时使用
        self.cont_log_probs = np.zeros(rollout_steps, dtype=np.float32)
        self.dones = np.zeros(rollout_steps, dtype=np.float32)
        # 每一步的 SNR (dB)，用于 per-SNR-bucket advantage normalization
        self.snr_dbs = np.zeros(rollout_steps, dtype=np.float32)
        # 每一步的条件熵系数（如未达标 SNR 点放大 entropy bonus）
        self.entropy_scales = np.ones(rollout_steps, dtype=np.float32)

        # Plackett-Luce 解码顺序 (可选)
        if decoding_order_dim > 0:
            self.decoding_orders = np.zeros((rollout_steps, decoding_order_dim), dtype=np.int64)
        else:
            self.decoding_orders = None

        # 每用户 delivery 违约项（用于 per-user advantage）
        if decoding_order_dim > 0:
            self.per_user_deficits = np.zeros((rollout_steps, decoding_order_dim), dtype=np.float32)
            self.per_user_advantages = np.zeros((rollout_steps, decoding_order_dim), dtype=np.float32)
        else:
            self.per_user_deficits = None
            self.per_user_advantages = None

        # GAE计算后填充
        self.advantages = np.zeros(rollout_steps, dtype=np.float32)
        self.returns = np.zeros(rollout_steps, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, value, log_prob, done, decoding_order=None,
            cont_log_prob=None, snr_db=0.0, per_user_deficit=None, entropy_scale=1.0):
        """添加一步transition

        Args:
            decoding_order: (K,) 整数排列索引 (仅 PL 模式需要)
            cont_log_prob: 连续动作部分的 log_prob (若 None 则 fallback 用 log_prob)
            snr_db: 当前步的 SNR (dB)，用于 per-SNR-bucket advantage normalization
            per_user_deficit: (K,) 每用户 delivery 违约量 [0, 1]
            entropy_scale: 本步 entropy bonus 缩放系数（未达标点 >1）
        """
        if self.pos >= self.rollout_steps:
            raise RuntimeError("Buffer is full, call clear() or compute_returns first")

        self.observations[self.pos] = np.asarray(obs, dtype=np.float32).flatten()
        self.actions[self.pos] = np.asarray(action, dtype=np.float32).flatten()
        self.rewards[self.pos] = float(reward)
        self.values[self.pos] = float(value)
        self.log_probs[self.pos] = float(log_prob)
        self.cont_log_probs[self.pos] = float(cont_log_prob) if cont_log_prob is not None else float(log_prob)
        self.dones[self.pos] = float(done)
        self.snr_dbs[self.pos] = float(snr_db)
        self.entropy_scales[self.pos] = float(entropy_scale)

        if self.decoding_orders is not None and decoding_order is not None:
            self.decoding_orders[self.pos] = np.asarray(decoding_order, dtype=np.int64).flatten()

        if self.per_user_deficits is not None and per_user_deficit is not None:
            self.per_user_deficits[self.pos] = np.asarray(per_user_deficit, dtype=np.float32).flatten()

        self.pos += 1
        if self.pos == self.rollout_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_value, per_user_adv_weight=1.0):
        """计算GAE优势和returns

        advantages = GAE(rewards, values, gamma, lambda)
        returns = advantages + values
        归一化advantages
        """
        last_value = float(last_value)
        gae = 0.0

        for t in reversed(range(self.pos)):
            if t == self.pos - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns[:self.pos] = self.advantages[:self.pos] + self.values[:self.pos]

        # per-SNR-bucket advantage normalization
        # 全局单一 mean/std 会让高SNR段的大优势主导，压制低SNR段梯度，导致 delivery 欠学习。
        # 按 SNR 分桶分别归一化，使各 SNR 段的优势尺度对齐、梯度贡献均衡。
        adv = self.advantages[:self.pos]
        global_mean = adv.mean()
        global_std = adv.std() + 1e-8

        snr = self.snr_dbs[:self.pos]
        idx = np.clip(
            ((snr - self.snr_min) / (self.snr_max - self.snr_min) * self.n_snr_buckets).astype(int),
            0, self.n_snr_buckets - 1,
        )
        for b in range(self.n_snr_buckets):
            mask = (idx == b)
            if mask.sum() >= 8:
                b_mean = adv[mask].mean()
                b_std = adv[mask].std() + 1e-8
                adv[mask] = (adv[mask] - b_mean) / b_std
            elif mask.any():
                # 样本过少的桶回退到全局归一化，避免小样本噪声
                adv[mask] = (adv[mask] - global_mean) / global_std

        self.advantages[:self.pos] = adv

        # per-user advantage = 全局优势 + 本用户 delivery 违约项
        if self.per_user_advantages is not None:
            pu_adv = adv[:, np.newaxis] + per_user_adv_weight * self.per_user_deficits[:self.pos]
            # 对每个用户单独做轻量标准化，防止尾部用户大违约项压过全局信号
            pu_mean = pu_adv.mean(axis=0, keepdims=True)
            pu_std = pu_adv.std(axis=0, keepdims=True) + 1e-8
            self.per_user_advantages[:self.pos] = (pu_adv - pu_mean) / pu_std

    def get_batches(self, batch_size):
        """随机打乱并分batch返回

        Yields: dict with keys (obs, actions, old_log_probs, returns, advantages)
                如果启用 PL 模式还包含 decoding_orders
        """
        assert self.pos > 0, "Buffer is empty"
        indices = np.random.permutation(self.pos)

        for start in range(0, self.pos, batch_size):
            end = min(start + batch_size, self.pos)
            batch_indices = indices[start:end]

            batch = {
                "obs": torch.tensor(self.observations[batch_indices], dtype=torch.float32, device=self.device),
                "actions": torch.tensor(self.actions[batch_indices], dtype=torch.float32, device=self.device),
                "old_log_probs": torch.tensor(self.log_probs[batch_indices], dtype=torch.float32, device=self.device),
                "old_cont_log_probs": torch.tensor(self.cont_log_probs[batch_indices], dtype=torch.float32, device=self.device),
                "returns": torch.tensor(self.returns[batch_indices], dtype=torch.float32, device=self.device),
                "advantages": torch.tensor(self.advantages[batch_indices], dtype=torch.float32, device=self.device),
            }
            if self.decoding_orders is not None:
                batch["decoding_orders"] = torch.tensor(
                    self.decoding_orders[batch_indices], dtype=torch.long, device=self.device
                )
            if self.per_user_advantages is not None:
                batch["per_user_advantages"] = torch.tensor(
                    self.per_user_advantages[batch_indices], dtype=torch.float32, device=self.device
                )
            yield batch

    def clear(self):
        """清空buffer"""
        self.pos = 0
        self.full = False


__all__ = ["RolloutBuffer"]
