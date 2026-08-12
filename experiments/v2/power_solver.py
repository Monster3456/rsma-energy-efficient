"""v2 功率求解器：ASOPA 式解耦——给定 (α, SIC order, H, q) 求最优功率分配

核心思想：
- 固定 SIC 解码顺序和 α 后，功率分配子问题 max R_sum - q·P_sum 是凸的
- beamforming 矩阵 W_c/W_p 的列范数编码了功率（||W_c[k]||² = alpha[k]·p[k]·split[k]），
  因此**每次评估速率时必须用当前功率重算 W**——否则功率变化不影响速率（v2.4 修复）
- 用 round-robin 梯度上升（每步微分一个用户）在 K 步内收敛

用法：
    from experiments.v2.power_solver import solve_optimal_power
    p_opt = solve_optimal_power(env, alpha, decoding_order, dinkelbach_q, p_init, n_steps=6)
"""
from __future__ import annotations

import numpy as np


def solve_optimal_power(
    env: "RSMAEnergyEnv",          # noqa: F821
    alpha: np.ndarray,
    decoding_order: np.ndarray,
    dinkelbach_q: float,
    p_init: np.ndarray,
    n_steps: int = 6,
    step_size: float = 0.08,
    eps: float = 0.02,
    n_dinkelbach_outer: int = 2,
    power_budget: float = 14.0,
    budget_lambda: float = 10.0,
) -> np.ndarray:
    """给定 (α, SIC order, H)，求解功率预算约束下的最优功率分配。

    v2.5：目标 = max R_sum - λ·max(0, Σp - P_budget)
    - 纯 EE 最大化会把功率压到 0（R 对数增长 vs P 线性增长，P→0 时 EE→∞），
      但 delivery 硬约束不允许——必须用总功率预算锚定功率水平。
    - P_budget 对标 SDMA 基线总功率（评估口径，~14W），solver 在预算内
      最大化速率，预算外线性惩罚。
    - 外层 Dinkelbach 迭代：q ← R(p)/Σp 仅作功率分配的相对引导。

    Args:
        env: RSMAEnergyEnv 实例（提供信道、beamforming、rate_calc）
        alpha: 当前公共流功率占比 (K,)，已映射到 [0, 1]
        decoding_order: 当前 SIC 解码顺序 (K,)
        dinkelbach_q: 初始影子价格（仅作起点）
        p_init: 初始功率 (K,)，通常为 RL 原始输出
        n_steps: 每轮内层梯度上升步数（每步微分一个用户）
        step_size: 梯度上升步长
        eps: 数值微分扰动量
        n_dinkelbach_outer: 外 Dinkelbach 迭代轮数
        power_budget: 总功率预算 (W)，对标 SDMA 基线
        budget_lambda: 预算超限惩罚系数

    Returns:
        p_opt: 优化后的功率 (K,)，已 clip 到 [0, P_k_max]
    """
    # 获取当前信道信息（缓存，确保一致性）
    H_hat = env._channel_obs["H_hat"]
    H_true = env._channel_obs["H_true"]
    sic_imp = float(getattr(env.config.system, "sic_imperfection", 0.0))
    group_ids = env.group_ids
    P_k_max = float(env.P_k_max)
    csi_mode = str(env.csi_mode)
    split_mask = env.split_mask

    # 速率计算包装函数：每次用当前功率重算 beamforming（W 编码功率！）
    # v2.5c 性能优化：solver 是近似优化器（只需功率梯度方向），用 perfect 版速率
    # （compute_rates_mmse_sic ~4ms）替代 outage 版（~11ms），提速 ~2.7 倍；
    # 实际交付仍由 env 用 imperfect 口径计算，solver 的乐观近似仅影响功率分配方向。
    # v2.5d 队列加权：目标 = Σ(backlog_w_k × R_k) − q·Σp。纯速率目标（EE）会把
    # 功率注水到少数好信道用户（实测总功率降到 3-4W），队列全空用户 0 功率 →
    # 交付崩溃。队列加权让功率直接服务交付（长队列用户多功率）。
    def _compute_R_sum(power: np.ndarray) -> float:
        p = np.asarray(power, dtype=np.float64).reshape(-1)
        W_c, W_p = env.beamformer.compute_precoding_v2(H_hat, alpha, p, split_mask)
        result = env.rate_calc.compute_rates_mmse_sic(
            H_hat, W_c, W_p,
            sic_imperfection=sic_imp,
            decoding_order=decoding_order,
            group_ids=group_ids,
        )
        R_per_user = np.asarray(result["R_total"], dtype=np.float64).reshape(-1)
        # 队列权重：Q/Q_max ∈ [0,1]，下限 0.05 保证空队列也有基础服务
        Q = np.asarray(env.traffic.queue_lengths, dtype=np.float64).reshape(-1)
        Q_max = float(getattr(env, 'Q_max', 200.0))
        backlog_w = np.maximum(Q / max(Q_max, 1e-8), 0.05)
        return float(np.sum(backlog_w * R_per_user))

    # ---- 外层 Dinkelbach 迭代 + 内层 round-robin 梯度上升（前向差分）----
    p = np.asarray(p_init, dtype=np.float64).copy()
    p = np.clip(p, 0.0, P_k_max)
    # 每用户功率下限（1% P_k_max）：防止数值微分在极低功率处失效
    # （eps=1e-4 时 R(0)≈R(1e-4)≈0 → dR≈0 → marginal=-q<0 → 卡在 0W 归零）。
    # 实测评估链路 PPO 输出低功率起点时 solver 卡在 0.21W → 交付崩溃。
    p_floor = 0.01 * P_k_max
    p = np.maximum(p, p_floor)
    # 初始投影：起点超预算时先压回预算球面。否则从全功率（~23W）出发时
    # 内层单轮把每个用户大幅降功率（marginal 深度负），没有回升轮 → 过冲归零。
    if float(np.sum(p)) > power_budget:
        p = p * (power_budget / float(np.sum(p)))
        p = np.maximum(p, p_floor)  # 投影后仍保下限
    q = float(dinkelbach_q)

    for _ in range(n_dinkelbach_outer):
        R_current = _compute_R_sum(p)

        for step in range(n_steps):
            # round-robin 选择本次修正的用户
            k = step % len(p)

            # 前向差分（1 次 rate_calc/步；上界处向下差分，下界处向上差分）
            if p[k] >= P_k_max - 1e-9:
                p_pert = p.copy()
                p_pert[k] = max(p[k] - eps, 0.0)
                R_pert = _compute_R_sum(p_pert)
                dR = (R_current - R_pert) / max(p[k] - p_pert[k], 1e-12)
            else:
                p_pert = p.copy()
                p_pert[k] = min(p[k] + eps, P_k_max)
                R_pert = _compute_R_sum(p_pert)
                dR = (R_pert - R_current) / max(p_pert[k] - p[k], 1e-12)

            # 边际目标：d/dp_k [R - q·Σp]，预算由每步投影硬约束（投影梯度法）
            # 线性惩罚（budget_lambda）会导致过冲/振荡（实测 0.96→0.02W 或 0.2→23W），
            # 投影梯度法保证功率总量恒 ≤ budget 且保留梯度方向。
            marginal = dR - q

            # 梯度上升更新（仅该用户）；R_current 用扰动点近似（误差 eps 级）
            p_new_k = p[k] + step_size * marginal
            p_new_k = float(np.clip(p_new_k, p_floor, P_k_max))  # 保下限防归零
            p[k] = p_new_k
            R_current = R_pert

            # 预算投影（每步）：Σp > budget 时按比例缩放回预算球面
            # 投影后 R_current 未同步（近似），但每外层轮开头会重算基准。
            if float(np.sum(p)) > power_budget:
                p = p * (power_budget / float(np.sum(p)))

        # 外层 Dinkelbach 更新：q ← R(p)/Σp（当前功率的 EE）
        P_total = float(np.sum(p))
        if P_total > 1e-8:
            q = R_current / P_total

    return p