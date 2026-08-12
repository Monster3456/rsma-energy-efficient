"""PPO-Lagrangian 硬约束模块（Task#17）

将 Delivery >= threshold 作为硬约束，引入自适应对偶乘子 λ：
    reward_aug = reward - λ · max(0, threshold - delivery_running)

λ 通过对偶上升每个 episode 更新一次：
    λ <- clip( λ + lr · (threshold - delivery_episode), 0, λ_max )

关键设计：
- 该惩罚 **不被任何缩放因子削弱**（与 ee_bonus 平等竞争梯度）
- warmup 阶段不施加约束，避免初期乱策略导致 λ 瞬间爆炸
- λ_max 限制约束强度，防止纯追求 Delivery 而 EE 崩溃
"""
from __future__ import annotations

from typing import List

import numpy as np


class DeliveryConstraint:
    """PPO-Lagrangian: 自适应对偶乘子确保 Delivery >= threshold."""

    def __init__(
        self,
        threshold: float = 0.95,
        lr: float = 0.05,
        init_lambda: float = 5.0,
        lambda_max: float = 50.0,
        warmup_episodes: int = 500,
        lambda_decay: float = 0.999,
        kp: float = 2.0,
        kd: float = 4.0,
        cvar_alpha: float = 0.25,
        cvar_window: int = 100,
        cvar_warmup_episodes: int = 50,
        n_snr_buckets: int = 4,
        snr_min: float = -5.0,
        snr_max: float = 35.0,
        success_ema_decay: float = 0.99,
    ) -> None:
        self.threshold = float(threshold)
        self.lr = float(lr)
        self.lambda_max = float(lambda_max)
        self.warmup_episodes = int(warmup_episodes)
        # Task#17 fix: λ 每个 episode 衰减系数。当 Delivery 接近阈值时，
        # 对偶上升的增长近 0，乘以 (<1) 会让 λ 自然介減，避免无限增长。
        self.lambda_decay = float(lambda_decay)
        self.success_ema_decay = float(success_ema_decay)
        # PID-Lagrangian (Stooke 2020): 在纯积分对偶上升之上叠加比例项 (kp) 与
        # 微分项 (kd)。比例项加速对当前违约的响应，微分项抑制 delivery 单调漂移
        # 与振荡。integral 保留积分对偶变量，prev_violation 用于计算微分项。
        self.kp = float(kp)
        self.kd = float(kd)
        # CVaR-Lagrangian: 使用最近 cvar_window 个 episode delivery 中最差
        # cvar_alpha 分位的均值作为 PID 控制器的『被控输出』，压制尾部风险。
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_window = int(cvar_window)
        self.cvar_warmup_episodes = int(cvar_warmup_episodes)

        # ---- SNR-分桶上下文对偶变量 ----
        # 将全局标量 λ 升级为按 SNR 分桶的向量：每桶维护独立的 PID+CVaR 对偶变量，
        # 精准惩罚低 SNR 违约段，而不误伤已可行的高 SNR 段。episode_count 保持全局。
        self.n_snr_buckets = int(n_snr_buckets)
        self.snr_min = float(snr_min)
        self.snr_max = float(snr_max)
        n = self.n_snr_buckets
        self.lambda_val = np.full(n, float(init_lambda), dtype=np.float64)
        self.integral = np.full(n, float(init_lambda), dtype=np.float64)
        self.prev_violation = np.zeros(n, dtype=np.float64)
        self.cvar_delivery = np.ones(n, dtype=np.float64)
        # SFL: 每桶 delivery 成功率 EMA，初始 0.5 表示未知
        self.bucket_success_ema = np.full(n, 0.5, dtype=np.float64)
        self.episode_count: int = 0
        self.delivery_history: List[List[float]] = [[] for _ in range(n)]

    # ------------------------------------------------------------------ #
    def _bucket(self, snr_db: float) -> int:
        """将 snr_db 线性映射到 [0, n_snr_buckets-1] 的桶索引。"""
        span = self.snr_max - self.snr_min
        if span <= 0:
            return 0
        idx = int(
            np.clip(
                (float(snr_db) - self.snr_min) / span * self.n_snr_buckets,
                0,
                self.n_snr_buckets - 1,
            )
        )
        return idx

    # ------------------------------------------------------------------ #
    def compute_penalty(self, delivery_ratio: float, snr_db: float | None = None) -> float:
        """计算每步约束惩罚（直接从 reward 中减去）.

        warmup 期间返回 0，让 PPO 先学到基本策略。
        warmup 之后：penalty = λ_b * max(0, threshold - delivery)，
        其中 b 为 snr_db 对应的 SNR 桶。delivery 越低于阈值，惩罚越大；
        超阈值则惩罚为 0（不奖励）.

        Args:
            delivery_ratio: 当前 episode 累计交付率（cumulative served/arrived）
            snr_db: 当前段 SNR（dB）。None 时用所有桶 λ 的均值兜底。

        Returns:
            penalty: 非负标量
        """
        if self.episode_count < self.warmup_episodes:
            return 0.0

        # 内点法超线性指数障碍 (interior-point barrier)：
        # 深违约区 (violation 大) 时惩罚随 exp 超线性放大，冲破 delivery 平台；
        # 近阈值区 (violation → 0) 时一阶泰勒展开退化为 lam*violation，避免 EE 崩溃。
        kappa = 8.0
        violation = max(0.0, self.threshold - float(delivery_ratio))
        if snr_db is None:
            lam = float(np.mean(self.lambda_val))
        else:
            lam = float(self.lambda_val[self._bucket(snr_db)])
        return lam * (float(np.exp(kappa * violation)) - 1.0) / kappa

    # ------------------------------------------------------------------ #
    def _update_bucket(self, b: int, episode_delivery: float) -> None:
        """对第 b 桶执行 CVaR+PID 对偶更新（使用该桶自己的 delivery_history）。"""
        history = self.delivery_history[b]
        history.append(float(episode_delivery))

        # SFL: 更新每桶 delivery 成功率 EMA（indicator(episode_delivery >= threshold)）
        success = 1.0 if float(episode_delivery) >= self.threshold else 0.0
        self.bucket_success_ema[b] = (
            self.success_ema_decay * self.bucket_success_ema[b]
            + (1.0 - self.success_ema_decay) * success
        )

        if self.episode_count < self.warmup_episodes:
            return

        # ---- CVaR (Conditional Value-at-Risk) 尾部信号（桶内独立窗口）----
        window = min(self.cvar_window, len(history))
        if window >= self.cvar_warmup_episodes:
            recent = np.asarray(history[-window:], dtype=np.float64)
            k = max(1, int(np.ceil(self.cvar_alpha * window)))
            worst_k = np.partition(recent, k - 1)[:k]
            self.cvar_delivery[b] = float(np.mean(worst_k))
            control_signal = self.cvar_delivery[b]
        else:
            self.cvar_delivery[b] = float(episode_delivery)
            control_signal = float(episode_delivery)

        # PID-Lagrangian (Stooke 2020) 控制律（仅作用于第 b 桶）。
        violation = self.threshold - control_signal

        # 积分项 (I)：等价于原对偶上升，累积违约。
        self.integral[b] = float(
            np.clip(self.integral[b] + self.lr * violation, 0.0, self.lambda_max)
        )
        # Task#25 fix v3: 使用该桶最近 100 个 episode 的滚动平均 delivery 作为衰减条件。
        # 只有当该桶整体表现真正达标时才衰减其积分项。
        w = min(100, len(history))
        avg_delivery = float(np.mean(history[-w:]))
        if avg_delivery >= self.threshold:
            self.integral[b] *= self.lambda_decay

        # 微分项 (D)：仅在违约恶化（上升）时施加，抑制单调漂移与振荡。
        deriv = max(0.0, violation - self.prev_violation[b])

        # 合成 λ = kp * P + I + kd * D，比例项只对正违约响应。
        self.lambda_val[b] = float(
            np.clip(
                self.kp * max(0.0, violation) + self.integral[b] + self.kd * deriv,
                0.0,
                self.lambda_max,
            )
        )

        self.prev_violation[b] = violation

    # ------------------------------------------------------------------ #
    def update(self, episode_delivery: float, snr_db: float | None = None) -> float:
        """每个 episode 结束后更新对偶变量 λ.

        delivery < threshold → λ 增大（加重惩罚）
        delivery > threshold → λ 减小（放松惩罚）

        Args:
            episode_delivery: 当前 episode 终态累计交付率
            snr_db: 当前 episode SNR（dB）。指定时仅更新对应 SNR 桶；
                None 时兜底更新所有桶。

        Returns:
            更新后的 λ 均值
        """
        self.episode_count += 1

        if snr_db is None:
            for b in range(self.n_snr_buckets):
                self._update_bucket(b, episode_delivery)
        else:
            self._update_bucket(self._bucket(snr_db), episode_delivery)

        return self.get_lambda()

    # ------------------------------------------------------------------ #
    def get_lambda(self) -> float:
        return float(np.mean(self.lambda_val))

    def get_bucket_success_rates(self) -> np.ndarray:
        """返回每桶 delivery 成功率 EMA（shape n_snr_buckets）。"""
        return self.bucket_success_ema.copy()

    def get_bucket_learnability_scores(self) -> np.ndarray:
        """返回 SFL 可学习性打分 p(1-p)，越接近 0.25 表示学得越快。"""
        p = self.bucket_success_ema
        return p * (1.0 - p)

    def get_bucket_deficits(self) -> np.ndarray:
        """返回每桶 delivery 赤字 max(0, threshold - 平均 delivery)。"""
        deficits = np.zeros(self.n_snr_buckets, dtype=np.float64)
        for b, history in enumerate(self.delivery_history):
            if not history:
                deficits[b] = self.threshold
            else:
                avg = float(np.mean(history[-100:]))
                deficits[b] = max(0.0, self.threshold - avg)
        return deficits

    def get_stats(self) -> dict:
        """返回统计信息用于日志输出。"""
        all_recent = [x for bucket in self.delivery_history for x in bucket[-100:]]
        recent = all_recent if all_recent else [0.0]
        return {
            "lambda": float(np.mean(self.lambda_val)),
            "avg_delivery_100ep": float(np.mean(recent)),
            "episode_count": self.episode_count,
            "in_warmup": self.episode_count < self.warmup_episodes,
            "integral": float(np.mean(self.integral)),
            "kp": self.kp,
            "kd": self.kd,
            "cvar_delivery": float(np.mean(self.cvar_delivery)),
            "cvar_alpha": self.cvar_alpha,
            "lambda_per_bucket": [float(x) for x in self.lambda_val],
            "delivery_per_bucket": [
                float(np.mean(bucket[-100:])) if bucket else 0.0
                for bucket in self.delivery_history
            ],
            "bucket_success_ema": [float(x) for x in self.bucket_success_ema],
        }


class DeliveryVirtualQueue:
    """Neely per-user Delivery Virtual Queue (DVQ).

    为 delivery 硬约束提供逐步逐用户的 credit assignment，与全局 Lagrangian
    对偶乘子互补：Lagrangian 用单一 λ 惩罚 episode 级聚合违约，DVQ 则为每个用户
    维护一个虚拟队列 Z_k，追踪其累计交付赤字，惩罚偏向长期欠交付的用户。

    虚拟队列动力学 (Lyapunov drift-plus-penalty)：
        deficit_k = threshold * arrivals_k - served_k
        Z_k <- clip(Z_k + deficit_k, 0, z_max)
    每步惩罚：
        penalty = dvq_weight * Σ_k (Z_k * deficit_k) / K
    warmup 期间不施加惩罚，避免初期乱策略导致 Z 爆炸。
    """

    def __init__(
        self,
        K: int,
        threshold: float = 0.98,
        dvq_weight: float = 0.5,
        z_max: float = 200.0,
        warmup_episodes: int = 200,
        cvar_alpha: float = 0.25,
    ) -> None:
        self.K = int(K)
        self.threshold = float(threshold)
        self.dvq_weight = float(dvq_weight)
        self.z_max = float(z_max)
        self.warmup_episodes = int(warmup_episodes)
        # CVaR 尾部聚焦：仅对虚拟积压最大的 worst-α 分位用户施加惩罚，
        # 把约束信用精准投到长期饿死的尾部用户，突破聚合 delivery 平台同时保护 EE。
        self.cvar_alpha = float(cvar_alpha)
        self.Z = np.zeros(self.K)
        self.episode_count = 0

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """清零虚拟队列（每 episode 开始时调用）。"""
        self.Z = np.zeros(self.K)

    # ------------------------------------------------------------------ #
    def update_and_penalty(self, arrivals: np.ndarray, served: np.ndarray) -> tuple[float, dict]:
        """更新虚拟队列并返回本步惩罚。

        Args:
            arrivals: 本步各用户到达量 (shape K)
            served: 本步各用户交付量 (shape K)

        Returns:
            (penalty, info_dict)。warmup 期间 penalty=0。
        """
        if self.episode_count < self.warmup_episodes:
            return 0.0, {"dvq_mean": 0.0}

        deficit = self.threshold * arrivals - served
        pos_deficit = np.maximum(deficit, 0.0)
        # CVaR 尾部聚焦：按虚拟积压 Z 降序取 worst-α 分位用户，仅惩罚这些
        # 长期饿死的尾部用户，把约束信用精准投放到尾部而非均匀摊薄到全体。
        m = max(1, int(np.ceil(self.cvar_alpha * self.K)))
        tail_idx = np.argsort(-self.Z)[:m]
        penalty = self.dvq_weight * float(
            np.mean(self.Z[tail_idx] * pos_deficit[tail_idx])
        )
        # 虚拟队列动力学对全体用户更新不变（保持 Lyapunov drift 一致性）。
        self.Z = np.clip(self.Z + deficit, 0.0, self.z_max)
        return penalty, {
            "dvq_mean": float(np.mean(self.Z)),
            "dvq_max": float(np.max(self.Z)),
            "dvq_tail_mean": float(np.mean(self.Z[tail_idx])),
        }

    # ------------------------------------------------------------------ #
    def end_episode(self) -> None:
        """episode 结束：递增计数并清零虚拟队列。"""
        self.episode_count += 1
        self.reset()

    # ------------------------------------------------------------------ #
    def get_stats(self) -> dict:
        return {
            "dvq_mean": float(np.mean(self.Z)),
            "dvq_max": float(np.max(self.Z)),
            "episode_count": self.episode_count,
        }
