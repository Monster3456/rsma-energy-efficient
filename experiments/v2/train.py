"""v2 训练入口：ASOPA 式功率解耦后处理层 + 训练循环

核心机制：
1. 在 env.step 前插入动作后处理替换层
2. 解析 RL 动作中的 (α, SIC order, d_ratio)
3. 调用 power_solver 计算给定 (α, order, H, q) 下的最优功率分配
4. 替换动作的功率分量为求解器输出
5. 调用原始 env.step
"""
import argparse
import sys
sys.path.insert(0, '/Users/libo/PycharmProjects/rsma-energy-efficient')

import numpy as np

from experiments.v2.config import make_v2_config
from experiments.v2.power_solver import solve_optimal_power
from src.env.rsma_env import RSMAEnergyEnv


def _patch_env_step(gate_threshold: float = 3.0, gate_scale: float = 1.0,
                    n_solver_steps: int = 24, solver_step_size: float = 0.08,
                    solver_interval: int = 20, power_budget: float = 14.0,
                    n_solver_outer: int = 1):
    """Monkey-patch RSMAEnergyEnv.step：ASOPA 式功率解耦后处理层。

    v2.3 (ASOPA on-policy 修复)：
    - gate_scale=1.0：替换所有用户的功率，不留 RL 原始功率
    - 保存替换后动作到 env._v2_modified_action，供主训练循环
      用 evaluate_actions 重算 log_prob，修复 on-policy 违背
    - 评估链路也挂同样 patch，保证训练/评估行为一致

    v2.5 (性能与预算约束)：
    - solver_interval>1：每 N 步调用一次 solver，中间步复用最近结果
      （功率分段恒定，训练/评估一致，solver 耗时降 N 倍）
    - power_budget：solver 的功率预算约束（对标 SDMA ~14W）
    - n_solver_steps=24 全覆盖：round-robin k=step%K，n_steps<K 时
      永远只优化前 n_steps 个用户（v2.3 功率不降的根因之一）
    - n_solver_outer=1：外层 Dinkelbach 只跑 1 轮（提速，质量足够）

    Args:
        gate_threshold: 保留接口兼容性（gate_scale=1.0 时无效）
        gate_scale: 替换强度（1.0=完全替换，ASOPA 模式）
        n_solver_steps: 功率求解器梯度上升步数（=K 时全覆盖）
        solver_step_size: 功率求解器步长
        solver_interval: 每 N 步调用一次 solver（1=每步）
        power_budget: solver 总功率预算 (W)
        n_solver_outer: 外层 Dinkelbach 迭代轮数
    """
    _original_step = RSMAEnergyEnv.step
    RSMAEnergyEnv._v2_original_step = _original_step
    _solver_cache = {"step_count": 0, "modified_continuous": None, "decoding_order": None}

    def _patched_step(self, action):
        # ---- 0. 同步当前 SNR 与累计交付率到 Lyapunov（供方案C主目标重写使用）----
        # delivery_ratio 是 episode 累计交付率（traffic 统计），wrapper 用它作为
        # 正奖励项；SNR 同步保留（观测/日志需要）。
        if hasattr(self, 'lyapunov'):
            self.lyapunov._v2_snr_db = float(getattr(self, 'current_snr_db', 15.0))
            self.lyapunov._v2_delivery_ratio = float(
                self.traffic.get_delivery_ratio()
            )

        # ---- 1. 解析动作 ----
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] == 4 * self.K:
            continuous_action = action[:3 * self.K]
            self.decoding_order = action[3 * self.K:].astype(np.int64)
            decoding_order = self.decoding_order
        else:
            continuous_action = action
            decoding_order = getattr(self, 'decoding_order', None)

        ca = np.clip(continuous_action, -1.0, 1.0)
        # 动作布局: [alpha(K), p(K), d(K)] —— 功率块在 [K:2K]
        p_raw = ca[self.K:2 * self.K]

        # ---- 2. 获取 alpha（group_level_alpha 模式下来自 Meta 的 split_mask）----
        # 注意：不能依赖 self.hierarchical_enabled（v2 config 未设置 hierarchical.enabled），
        # 与 _parse_action 的 group_level_alpha 分支保持一致——split_mask 非 None 即可用。
        alpha = (self.split_mask.copy()
                 if self.split_mask is not None
                 else None)

        # ---- 3. 功率求解器（ASOPA 式：给定 α, order, H 解预算内最优功率）----
        # gate_scale<=0 时完全跳过 solver（纯 PPO 控制，重建基线用）
        solver_ready = (
            gate_scale > 0
            and alpha is not None
            and decoding_order is not None
            and self.dinkelbach_eta is not None
        )
        if solver_ready:
            _solver_cache["step_count"] += 1
            run_solver = _solver_cache["step_count"] % max(solver_interval, 1) == 1
            if run_solver or _solver_cache["modified_continuous"] is None:
                p_ratio = (p_raw + 1.0) / 2.0
                p_current = p_ratio * self.P_k_max

                p_opt = solve_optimal_power(
                    self, alpha, decoding_order,
                    float(self.dinkelbach_eta),
                    p_current,
                    n_steps=n_solver_steps,
                    step_size=solver_step_size,
                    power_budget=power_budget,
                    n_dinkelbach_outer=n_solver_outer,
                )

                # 完全替换（gate_scale=1.0）
                p_final = p_opt

                # 反映射回 p_raw 空间
                p_opt_ratio = np.clip(p_final / max(self.P_k_max, 1e-8), 0.0, 1.0)
                p_opt_raw = 2.0 * p_opt_ratio - 1.0
                p_opt_raw = np.clip(p_opt_raw, -1.0, 1.0)

                # 替换动作的功率分量（布局 [alpha(K), p(K), d(K)] → 功率块在 [K:2K]）
                modified_continuous = continuous_action.copy()
                modified_continuous[self.K:2 * self.K] = p_opt_raw

                # 缓存供中间步复用
                _solver_cache["modified_continuous"] = modified_continuous.copy()
                _solver_cache["decoding_order"] = decoding_order
            else:
                # 中间步：复用最近 solver 的功率块（alpha/d 块用当前 RL 输出）
                modified_continuous = continuous_action.copy()
                modified_continuous[self.K:2 * self.K] = \
                    _solver_cache["modified_continuous"][self.K:2 * self.K]

            # 重组合动作
            modified_action = np.concatenate([
                modified_continuous,
                action[3 * self.K:],
            ]) if action.shape[0] == 4 * self.K else modified_continuous

            # ---- 保存替换后动作（供主训练循环重算 log_prob）----
            self._v2_modified_action = (modified_continuous, decoding_order)
        else:
            modified_action = action
            self._v2_modified_action = None

        # ---- 6. 调用原始 step ----
        result = _original_step(self, modified_action)
        # 同步 delivery_ratio 到 Lyapunov（供 _patch_power_primary 使用）
        if hasattr(self, 'lyapunov'):
            self.lyapunov._v2_delivery_ratio = float(result[4].get("delivery_ratio", 0.5))
        return result

    RSMAEnergyEnv.step = _patched_step


def _restore_env_step():
    """恢复原始 env.step。

    训练结束后调用，确保后续评估使用未修改的 env。
    """
    original = getattr(RSMAEnergyEnv, '_v2_original_step', None)
    if original is not None:
        RSMAEnergyEnv.step = original
        del RSMAEnergyEnv._v2_original_step


def _patch_power_primary(w_deliv: float = 5.0, w_power: float = 3.0,
                         power_budget: float = 24.0):
    """Monkey-patch LyapunovController.compute_reward：方案C 奖励主目标反转。

    reward = scale × (w_deliv×delivery − w_power×P_norm − q_penalty − overflow_penalty)
    其中 P_norm = total_power / power_budget ∈ [0,1]

    设计逻辑（v2.3，方案C）：
    - 主目标从“交付最大化 + 功率税”反转为“功率最小化 + 交付奖励”：
      功率是持续线性代价（-w_power×P_norm），交付是正奖励（+w_deliv×delivery）
    - 关闭 Lagrangian 指数惩罚（config.use_lagrangian=False）——线性功率税正是
      被其指数增长压倒才失效（v2.1/v2.2 教训：delivery 未达标时 penalty 指数增长，
      PPO 优先避免惩罚尖峰而非降功率）
    - 无显式 per-SNR λ：SNR 自适应从约束自然涌现——高 SNR 低功率即可满足交付
      （省电划算），低 SNR 需高功率换交付（值得），PPO 自动权衡

    DGMW panic 模式（纯吞吐量逃生通道）豁免。
    """
    from src.optimization.lyapunov import LyapunovController
    _orig_compute_reward = LyapunovController.compute_reward

    def _patched_compute_reward(self, *args, **kwargs):
        info = _orig_compute_reward(self, *args, **kwargs)
        if not info.get("dgmw_panic", False):
            delivery = float(getattr(self, '_v2_delivery_ratio', 0.5))
            p_norm = info["total_power"] / max(power_budget, 1e-8)
            new_reward = self.reward_scale * (
                w_deliv * delivery
                - w_power * p_norm
                - info["q_penalty"]
                - info["overflow_penalty"]
            )
            info["reward"] = float(new_reward)
            info["power_primary"] = -w_power * p_norm
            info["delivery_primary"] = w_deliv * delivery
        return info

    LyapunovController.compute_reward = _patched_compute_reward


def main():
    parser = argparse.ArgumentParser(description="v2: RSMA-EE ASOPA 功率解耦训练")
    parser.add_argument('--episodes', type=int, default=1500,
                        help='训练轮数')
    parser.add_argument('--resume', type=str, default='logs/opt_best/best.pt',
                        help='从指定 checkpoint 恢复训练（默认 best.pt）')
    parser.add_argument('--exp_name', type=str, default='v2_opt_iter',
                        help='实验名称')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--gate-threshold', type=float, default=3.0,
                        help='队列门控阈值（bits），低于此值保留 RL 原始功率')
    # v2.3 (ASOPA on-policy 修复)：gate_scale=1.0 完全替换功率，与 _patch_env_step 签名一致
    parser.add_argument('--solver-steps', type=int, default=24,
                        help='功率求解器梯度上升步数（=K 全覆盖，修复只优化前 n_steps 用户）')
    parser.add_argument('--solver-step-size', type=float, default=0.08,
                        help='功率求解器步长')
    parser.add_argument('--solver-interval', type=int, default=20,
                        help='每 N 步调用一次 solver（1=每步；默认 20 降耗时）')
    parser.add_argument('--power-budget', type=float, default=14.0,
                        help='solver 总功率预算 (W)，对标 SDMA 基线')
    parser.add_argument('--solver-outer', type=int, default=1,
                        help='外层 Dinkelbach 迭代轮数（默认 1 提速）')

    # 冒烟测试
    parser.add_argument('--smoke', action='store_true',
                        help='冒烟测试模式：极小配置快速验证')
    # 优化循环兼容
    parser.add_argument('--curltrac', action='store_true', default=True,
                        help='启用 CuRLTraC 课程（默认启用）')
    parser.add_argument('--no-curltrac', action='store_true',
                        help='禁用 CuRLTraC 课程')
    parser.add_argument('--delivery-threshold', type=float, default=None,
                        help='Delivery 约束阈值')
    parser.add_argument('--lagrangian-lr', type=float, default=None,
                        help='对偶乘子 λ 学习率')
    parser.add_argument('--lagrangian-max', type=float, default=None,
                        help='λ 上限')
    parser.add_argument('--w_power_exceed', type=float, default=None,
                        help='超功率惩罚权重')
    parser.add_argument('--V_min', type=float, default=None,
                        help='自适应 V 下限')
    parser.add_argument('--V_max', type=float, default=None,
                        help='自适应 V 上限')
    parser.add_argument('--blend_alpha', type=float, default=None,
                        help='队列感知混合权重指数')
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                        help='Actor learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--use-bucket-meta-alpha', type=lambda v: v.lower() in ('1', 'true', 'yes'),
                        default=None, help='启用 per-bucket alpha 缩放（默认启用）')
    parser.add_argument('--w-deliv', type=float, default=5.0,
                        help='方案C交付奖励权重（reward 主目标：+w_deliv×delivery）')
    parser.add_argument('--w-power', type=float, default=3.0,
                        help='方案C功率惩罚权重（reward 主目标：-w_power×P_norm, P_norm∈[0,1]）')
    parser.add_argument('--gate-scale', type=float, default=1.0,
                        help='功率替换强度（1.0=完全替换，ASOPA 模式；0=停用 solver）')

    args = parser.parse_args()

    # ---- 构建 v2 配置 ----
    config = make_v2_config()
    config.exp_name = args.exp_name
    config.device = args.device
    config.ppo.total_episodes = args.episodes
    config.ppo.lr_actor = args.lr_actor
    config.seed = args.seed

    # 续训
    config.resume_ckpt = args.resume if args.resume else None
    if config.resume_ckpt:
        print(f"[v2] 从 {config.resume_ckpt} 恢复训练")

    # CuRLTraC 课程
    if args.no_curltrac:
        config.ppo.use_curriculum = False
    else:
        config.ppo.use_curriculum = True

    # 超参覆盖
    if args.delivery_threshold is not None:
        config.constraint.delivery_threshold = args.delivery_threshold
    if args.lagrangian_lr is not None:
        config.constraint.lagrangian_lr = args.lagrangian_lr
    if args.lagrangian_max is not None:
        config.constraint.lagrangian_max = args.lagrangian_max
    if args.w_power_exceed is not None:
        config.reward.w_power_exceed = args.w_power_exceed
    if args.V_min is not None:
        config.lyapunov.V_min = args.V_min
    if args.V_max is not None:
        config.lyapunov.V_max = args.V_max
    if args.blend_alpha is not None:
        config.lyapunov.blend_alpha = args.blend_alpha

    # 冒烟测试
    if args.smoke:
        config.ppo.total_episodes = 1
        config.ppo.rollout_steps = 64
        config.ppo.batch_size = 16

    # ---- [v2 升级] 挂载 Meta-Controller per-bucket alpha 缩放 ----
    if args.use_bucket_meta_alpha is not None:
        config.hierarchical.use_bucket_meta_alpha = args.use_bucket_meta_alpha
    if config.hierarchical.use_bucket_meta_alpha:
        from experiments.v2.models.meta_controller import MetaActorCritic as V2MetaAC
        import src.models.meta_controller as mm
        mm.MetaActorCritic = V2MetaAC
        print(f"[v2] Meta-Controller 已升级：per-bucket alpha 缩放 (n_buckets={config.hierarchical.n_alpha_buckets})")
        # 兼容性：新版 Meta 有 bucket_alpha_head 参数，旧版 checkpoint 没有 → strict=False
        orig_load = mm.MetaActorCritic.load_state_dict
        mm.MetaActorCritic.load_state_dict = lambda self, sd, strict=True, assign=False: orig_load(self, sd, strict=False, assign=assign)

    # ---- 处理旧版 checkpoint 兼容性：去除 meta_optimizer（参数组不匹配）----
    if config.resume_ckpt and config.hierarchical.use_bucket_meta_alpha:
        import torch
        ckpt = torch.load(config.resume_ckpt, map_location='cpu', weights_only=False)
        if 'meta_optimizer' in ckpt:
            # 预创建 v2 Meta 以获取优化器参数组结构，使 checkpoint 的 meta_optimizer 与之匹配
            v2_meta = V2MetaAC(
                K=config.system.K,
                d_user=config.hierarchical.meta_d_user,
                hidden_dim=config.hierarchical.meta_hidden_dim,
                n_heads=config.hierarchical.meta_n_heads,
                n_groups=config.hierarchical.n_groups,
                group_level_alpha=config.hierarchical.group_level_alpha,
                group_alpha_min=config.hierarchical.group_alpha_min,
                group_alpha_max=config.hierarchical.group_alpha_max,
                group_alpha_log_std_init=config.hierarchical.group_alpha_log_std_init,
                group_tau_init=config.hierarchical.meta_group_tau_init,
                group_tau_min=config.hierarchical.meta_group_tau_min,
                balance_coef=config.hierarchical.meta_balance_coef,
                use_bucket_meta_alpha=True,
                n_alpha_buckets=config.hierarchical.n_alpha_buckets,
            )
            temp_opt = torch.optim.Adam(v2_meta.parameters(), lr=config.hierarchical.meta_lr)
            opt_sd = temp_opt.state_dict()
            # 保留结构（参数组字段），清除 state（重置优化器步数/动量）
            ckpt['meta_optimizer'] = {'state': {}, 'param_groups': opt_sd['param_groups']}
        temp_path = config.resume_ckpt.replace('.pt', '_v2.pt')
        torch.save(ckpt, temp_path)
        config.resume_ckpt = temp_path
        print(f"[v2] 已适配 checkpoint 至 v2 Meta-Controller：{temp_path}")

    # ---- 挂载功率后处理层 ----
    _patch_env_step(
        gate_threshold=args.gate_threshold,
        gate_scale=args.gate_scale,
        n_solver_steps=args.solver_steps,
        solver_step_size=args.solver_step_size,
        solver_interval=args.solver_interval,
        power_budget=args.power_budget,
        n_solver_outer=args.solver_outer,
    )

    # ---- 挂载方案C 奖励主目标反转（v2.3：最小功率 + 交付奖励，已关 Lagrangian）----
    _patch_power_primary(w_deliv=args.w_deliv, w_power=args.w_power)
    print(f"[v2.3] 方案C 奖励主目标已挂载：reward = scale×(w_deliv×delivery − w_power×P_norm − q − ovf) "
          f"w_deliv={args.w_deliv} w_power={args.w_power} "
          f"(softmax={config.system.use_softmax_power_share}, gate_scale={args.gate_scale}, "
          f"solver_interval={args.solver_interval}, power_budget={args.power_budget})")

    # ---- 启动训练 ----
    from src.training.main import train
    try:
        train(config)
    finally:
        _restore_env_step()


if __name__ == "__main__":
    main()