"""Meta-Controller 奖励计算

慢时间尺度（T_s 步一段）累积统计 → 标量 Meta 奖励。
平衡能效（正向）与队列溢出（负向），并通过分组多样性 + 负载均衡引导分组决策。

Arm B（10-23-hierarchical-opt）修两处：
  1. ``group_balance`` 改为统计**全部 G 组**（空组计 0）。原实现只统计被使用组，
     恒单组时 counts=[24]、std=0、balance 取**最大值 0.5**，坍缩侧无任何反向梯度。
  2. 多样性/均衡项**不参与 ``meta_reward_scale`` 整体缩放**。原实现整条奖励乘 0.02，
     结构项有效幅动仅 0.015，是 |MetaR|≈19 的 0.08%，被 ``avg_ee - 2·avg_overflow``
     （unscaled ≈ −950）完全淹没。改用独立的 ``structure_scale``。
"""

import numpy as np


def compute_meta_reward(meta_stats: dict, reward_scale: float = 1.0,
                        structure_scale: float = 1.0) -> float:
    """计算慢时间尺度 Meta 奖励。

    基于 T_s 步内的累积统计，平衡能效与队列稳定性，并加入分组结构化激励：
        reward = reward_scale * (avg_ee - 2.0*avg_overflow - alpha_guidance)
                 + structure_scale * (1.0*group_diversity + 0.5*group_balance)

    Args:
        meta_stats: dict，包含键 'ee_sum'、'overflow_sum'、'steps'。
                   可选: 'group_ids', 'n_groups'。
                   Task#36 可选: 'alpha_sum', 'alpha_count', 'snr_db',
                                 'alpha_guidance_scale', 'alpha_target'。
        reward_scale: 主项（EE/溢出/α 引导）缩放，对应 ``meta_reward_scale``。
        structure_scale: 分组结构项（多样性/均衡）缩放，**独立于 reward_scale**。

    Returns:
        标量奖励
    """
    steps = max(int(meta_stats.get("steps", 1)), 1)
    avg_ee = float(meta_stats.get("ee_sum", 0.0)) / steps
    avg_overflow = float(meta_stats.get("overflow_sum", 0.0)) / steps

    # 主项：EE - overflow 惩罚（随 reward_scale 缩放）
    main = avg_ee - 2.0 * avg_overflow

    # Task#36: SNR 条件 α 软引导项（属主项侧）
    guidance_scale = float(meta_stats.get('alpha_guidance_scale', 0.0))
    alpha_count = int(meta_stats.get('alpha_count', 0))
    if guidance_scale > 0.0 and alpha_count > 0:
        alpha_mean = float(meta_stats.get('alpha_sum', 0.0)) / max(alpha_count, 1)
        target_alpha = float(meta_stats.get('alpha_target', 0.7))
        main += -guidance_scale * (alpha_mean - target_alpha) ** 2

    # 分组结构项（独立缩放，不被 0.02 压死）
    structure = 0.0
    if 'group_ids' in meta_stats:
        group_ids = np.asarray(meta_stats['group_ids']).reshape(-1)
        n_groups = max(int(meta_stats.get('n_groups', 4)), 1)

        # 1) 分组多样性：激励使用更多分组
        n_unique = len(np.unique(group_ids))
        group_diversity = n_unique / n_groups            # 0..1
        structure += 1.0 * group_diversity

        # 2) 分组均衡：对**全部 G 组**统计（空组计 0），消除"恒单组拿满分"。
        #    counts = [n_0, ..., n_{G-1}]，含空组；std/mean 越小越均衡。
        counts = np.zeros(n_groups, dtype=np.float64)
        if group_ids.size:
            idx = np.clip(group_ids.astype(np.int64), 0, n_groups - 1)
            counts = np.bincount(idx, minlength=n_groups).astype(np.float64)
        mean_size = float(counts.mean())
        std_size = float(counts.std())
        if mean_size > 1e-8:
            balance = max(0.0, 1.0 - std_size / mean_size)
        else:
            balance = 0.0
        structure += 0.5 * balance

    return float(reward_scale * main + structure_scale * structure)


__all__ = ["compute_meta_reward"]
