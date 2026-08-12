"""Meta-Controller 奖励计算

慢时间尺度（T_s 步一段）累积统计 → 标量 Meta 奖励。
平衡能效（正向）与队列溢出（负向），并通过分组多样性 + 负载均衡引导分组决策。
"""

import numpy as np


def compute_meta_reward(meta_stats: dict) -> float:
    """计算慢时间尺度 Meta 奖励。

    基于 T_s 步内的累积统计，平衡能效与队列稳定性，并加入分组结构化激励：
        reward = avg_ee
                 - 2.0 * avg_overflow
                 + 1.0 * group_diversity        # 鼓励使用多个分组
                 + 0.5 * group_balance          # 鼓励组间均匀
                 - alpha_guidance_scale * (alpha_mean - target_alpha) ** 2  # Task#36 软引导

    Args:
        meta_stats: dict，包含键 'ee_sum'、'overflow_sum'、'steps'。
                   可选: 'group_ids', 'n_groups'。
                   Task#36 可选: 'alpha_sum', 'alpha_count', 'snr_db',
                                 'alpha_guidance_scale', 'alpha_target'。

    Returns:
        标量奖励
    """
    steps = max(int(meta_stats.get("steps", 1)), 1)
    avg_ee = float(meta_stats.get("ee_sum", 0.0)) / steps
    avg_overflow = float(meta_stats.get("overflow_sum", 0.0)) / steps

    # 基础奖励：EE - overflow 惩罚
    reward = avg_ee - 2.0 * avg_overflow

    # 分组结构化奖励
    if 'group_ids' in meta_stats:
        group_ids = np.asarray(meta_stats['group_ids']).reshape(-1)
        n_groups = int(meta_stats.get('n_groups', 4))

        # 1) 分组多样性奖励：激励使用更多分组（权重 0.5 → 1.0）
        n_unique = len(np.unique(group_ids))
        group_diversity = n_unique / max(n_groups, 1)  # 0..1
        reward += 1.0 * group_diversity

        # 2) 分组间负载均衡奖励：bonus = balance_weight * (1 - std/mean)
        #    所有组大小相同 → bonus = balance_weight；越不平衡越接近 0
        balance_weight = 0.5
        # 仅统计实际被使用的分组（避免空组使 std 极大主导信号）
        if n_unique > 0:
            _, counts = np.unique(group_ids, return_counts=True)
            mean_size = float(counts.mean())
            std_size = float(counts.std())
            if mean_size > 1e-8:
                balance = max(0.0, 1.0 - std_size / mean_size)
            else:
                balance = 0.0
            balance_bonus = balance_weight * balance
            reward += balance_bonus

    # Task#36: SNR 条件 α 软引导项
    # alpha_guidance_scale 在 main.py 根据 config + warmup 状态置为 0 或 正值；
    # 在这里根据其是否为正决定是否加引导项。
    guidance_scale = float(meta_stats.get('alpha_guidance_scale', 0.0))
    alpha_count = int(meta_stats.get('alpha_count', 0))
    if guidance_scale > 0.0 and alpha_count > 0:
        alpha_mean = float(meta_stats.get('alpha_sum', 0.0)) / max(alpha_count, 1)
        target_alpha = float(meta_stats.get('alpha_target', 0.7))
        guidance = -guidance_scale * (alpha_mean - target_alpha) ** 2
        reward += guidance

    return float(reward)


__all__ = ["compute_meta_reward"]
