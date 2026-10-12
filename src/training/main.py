"""训练主入口（双层 PPO）

慢时间尺度 Meta-Controller（Self-Attention PPO） + 快时间尺度 Worker（Clip-PPO + Plackett-Luce）。
Meta 每 T_s 步决策一次 split_mask + group_ids；Worker 每步决策连续动作 (alpha, p_total, d_ratio) + 解码顺序。
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from src.config import Config
from src.env.rsma_env import RSMAEnergyEnv
from src.models.actor_critic import ActorCritic, ActorCriticAttn
from src.models.meta_controller import (
    MetaActorCritic,
    MetaPPOTrainer,
    MetaRolloutBuffer,
)
from src.optimization.meta_reward import compute_meta_reward
from src.optimization.solver_expert import SolverExpert
from src.training.buffer import RolloutBuffer
from src.training.lagrangian import DeliveryConstraint, DeliveryVirtualQueue
from src.training.ppo import PPOTrainer


# ---- CuRLTraC-ALP / C3PO: 与 SDMA 基线评估点一一对应的 7 个 SNR 评估点 ----
_HM_TARGETS = np.array([0.856, 0.982, 0.960, 0.894, 0.805, 0.730, 0.668])
_HM_EDGES = np.array([-2.5, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5])


def _hm_bucket(snr_db: float) -> int:
    """将 SNR 映射到评估点索引。"""
    idx = int(np.searchsorted(_HM_EDGES, snr_db, side='right')) - 1
    return int(np.clip(idx, 0, len(_HM_TARGETS) - 1))


def _hm_init_state(target_scale: float = 1.0, point_targets=None) -> dict:
    """初始化逐点 delivery EMA、达标标志、连续未达标 streak 与采样计数。

    M5.3: point_targets 提供 7 个逐点目标时直接采用（忽略 target_scale）。
    默认目标 = SDMA 逐点 delivery，对"已全面超越 SDMA"的策略 shortfall 恒 0、
    机制失明；逐点覆盖用于把挖掘压力定向到策略自身的弱点（如 0/30dB）。
    """
    if point_targets and len(point_targets) == len(_HM_TARGETS):
        targets = np.asarray(point_targets, dtype=float).copy()
    else:
        targets = _HM_TARGETS * float(target_scale)
    return {
        'targets': targets,
        'ema': targets.copy(),
        'achieved': np.zeros(len(_HM_TARGETS), dtype=bool),
        'streak': np.zeros(len(_HM_TARGETS), dtype=np.int32),
        'sample_count': np.zeros(len(_HM_TARGETS), dtype=np.int64),
    }


def _c3po_init_state(target_scale: float = 1.0, ema_decay: float = 0.99) -> dict:
    """C3PO: 初始化逐 SNR 点 delivery EMA（用于移动预算）。"""
    targets = _HM_TARGETS * float(target_scale)
    return {
        'targets': targets,
        'ema': targets.copy(),
        'decay': float(ema_decay),
    }


def _c3po_update(state: dict, snr_db: float, delivery: float) -> None:
    """C3PO: 用 episode 结束时的 delivery 更新对应点的 EMA。"""
    b = _hm_bucket(snr_db)
    state['ema'][b] = state['decay'] * state['ema'][b] + (1.0 - state['decay']) * delivery


def _c3po_compute_costs(state: dict, snr_db: float, step_delivery: float) -> np.ndarray:
    """C3PO: 构造 7 维逐点 cost 向量（仅当前 SNR 点有非零 cost）。"""
    targets = state['targets']
    costs = np.zeros(len(targets), dtype=np.float32)
    b = _hm_bucket(snr_db)
    costs[b] = float(max(0.0, targets[b] - step_delivery))
    return costs


def _c3po_budgets(state: dict, kappa: float, margin_floor: float) -> np.ndarray:
    """C3PO: 移动预算 budget_j = kappa * max(0, target_j - ema_j) + margin_floor。"""
    targets = state['targets']
    ema = state['ema']
    return np.clip(
        kappa * np.maximum(targets - ema, 0.0) + margin_floor,
        margin_floor, 1.0
    ).astype(np.float32)


def _hm_update(state: dict, snr_db: float, delivery: float, decay: float) -> None:
    """用 episode 结束时的 delivery 更新对应点的慢 EMA 与 streak。"""
    b = _hm_bucket(snr_db)
    state['ema'][b] = decay * state['ema'][b] + (1.0 - decay) * delivery
    achieved = state['ema'][b] >= state['targets'][b]
    if achieved:
        state['streak'][b] = 0
    else:
        state['streak'][b] += 1
    state['achieved'][b] = achieved


def _hm_sample_snr(state: dict, w_hard: float, floor_frac: float,
                   streak_mult: float, streak_cap: float,
                   rng: np.random.Generator,
                   fail_entropy_scale: float = 1.5,
                   performance_linked: bool = False,
                   link_coef: float = 1.0) -> tuple[float, int, bool, float]:
    """按 shortfall + streak 加权采样下一 episode 的 SNR，并返回条件熵缩放。

    权重 = base_floor + w_hard * shortfall * min(streak_mult^streak, streak_cap)，
    连续未达标点获得倍增曝光；已达标点 streak=0，权重仅含 shortfall。

    当 performance_linked=True 时，未达标点的熵缩放按
        ent_p = fail_entropy_scale * (1 + c * shortfall_p / shortfall_max)
    动态调节：缺口越大熵注入越强，缺口缩小自动收敛，防止过注入。
    """
    targets = state['targets']
    n = len(targets)
    base = floor_frac / n
    margin = 0.005
    # 护栏：已达标的点若 EMA 跌破 target - margin，则权重减半，防已达标点崩溃。
    w_eff = np.where(
        state['achieved'] & (state['ema'] < targets - margin),
        w_hard * 0.5, w_hard
    )
    shortfall = np.maximum(targets - state['ema'], 0.0)
    # M5.3 修复：重加权后未达标桶 streak 可达数千，streak_mult**streak 在
    # ~1.5^1450 处溢出 float64（RuntimeWarning: overflow encountered in power）。
    # 指数截断到 64 再幂——结果本就被 streak_cap 封顶，截断不改变任何取值。
    streak_boost = np.minimum(
        streak_mult ** np.minimum(state['streak'], 64), streak_cap)
    raw = base + w_eff * shortfall * streak_boost
    probs = raw / max(float(raw.sum()), 1e-12)
    b = int(rng.choice(n, p=probs))
    state['sample_count'][b] += 1
    sampled_snr = float(rng.uniform(_HM_EDGES[b], _HM_EDGES[b + 1]))
    is_failing = not state['achieved'][b]

    # 性能挂钩定向熵激励
    if performance_linked and is_failing:
        shortfall_max = max(float(shortfall.max()), 1e-8)
        entropy_scale = fail_entropy_scale * (1.0 + link_coef * shortfall[b] / shortfall_max)
    else:
        entropy_scale = fail_entropy_scale if is_failing else 1.0

    return sampled_snr, b, is_failing, entropy_scale


def _load_state_dict_with_padding(model: torch.nn.Module, state_dict: dict,
                                   model_name: str = "model") -> bool:
    """加载 state_dict 并自动 zero-padding 处理维度不匹配的参数。

    适用于跨版本 resume（如 v12→v13 obs_dim/d_user 维度增加）。
    对于形状完全匹配的参数直接拷贝；对于维度不同但可 padding 的参数
    （旧 tensor 各维均 <= 新 tensor 对应维），用零填充扩展后加载。

    Returns:
        True 如果发生了 padding（意味着 optimizer state 不兼容），False 否则。
    """
    model_state = model.state_dict()
    padded_keys = []
    skipped_keys = []

    for key in list(state_dict.keys()):
        if key not in model_state:
            skipped_keys.append(key)
            continue
        if state_dict[key].shape == model_state[key].shape:
            continue  # 形状匹配，无需处理
        old_shape = state_dict[key].shape
        new_shape = model_state[key].shape
        if len(old_shape) != len(new_shape):
            skipped_keys.append(key)
            del state_dict[key]
            continue
        # 检查旧 tensor 各维是否均 <= 新 tensor
        if all(o <= n for o, n in zip(old_shape, new_shape)):
            new_tensor = model_state[key].clone()  # 零初始化（继承新模型初始权重）
            slices = tuple(slice(0, s) for s in old_shape)
            new_tensor[slices] = state_dict[key]
            state_dict[key] = new_tensor
            padded_keys.append(f"{key}: {list(old_shape)}->{list(new_shape)}")
        else:
            skipped_keys.append(key)
            del state_dict[key]

    if padded_keys:
        print(f"[Resume] {model_name} padded params: {padded_keys}")
    if skipped_keys:
        print(f"[Resume] {model_name} skipped params: {skipped_keys}")

    model.load_state_dict(state_dict, strict=False)
    return len(padded_keys) > 0 or len(skipped_keys) > 0


def _optimizer_state_compatible(optimizer, ckpt_state) -> bool:
    """resume 时判断 checkpoint 的 optimizer state 与本模型 param 组是否兼容。

    结构变化（新增 mw_gate 组/参数、obs 维度变化）会使 torch 的
    load_state_dict 抛 "parameter group that doesn't match"；据此返回 False，
    调用方降级为跳过 optimizer state（模型权重仍完整加载）。
    """
    if not isinstance(ckpt_state, dict) or "param_groups" not in ckpt_state:
        return False
    try:
        # ckpt 侧 param_groups 里只有参数 id（int），无法取 numel；按每组的参数个数
        # 比较即可捕获"新增/移除参数入组"（如 mw_gate）与组数变化。形状 padding 类
        # 维度变化仍由 worker_padded/meta_padded 判据兜底（调用方已叠加）。
        counts_new = [len(g["params"]) for g in optimizer.param_groups]
        counts_old = [len(g["params"]) for g in ckpt_state["param_groups"]]
        return counts_new == counts_old
    except Exception:
        return False


def _empty_meta_stats() -> dict:
    return {
        "ee_sum": 0.0,
        "overflow_sum": 0.0,
        "steps": 0,
        "alpha_sum": 0.0,
        "alpha_count": 0,
        "snr_db": 0.0,
    }


def _set_worker_bucket(worker_ac, env):
    """根据 env.current_snr_db 设置 Worker 的 episode 级桶专家硬路由。"""
    if getattr(worker_ac, "use_bucket_experts", False):
        bucket = worker_ac.set_active_bucket(float(env.current_snr_db))
        if not hasattr(worker_ac, "_route_counts"):
            worker_ac._route_counts = [0] * int(worker_ac.n_experts)
        worker_ac._route_counts[bucket] += 1


def _inject_alpha_guidance(meta_stats: dict, config, episode_count: int) -> None:
    """Task#36: 将 α 引导 scale 与 SNR 依赖的 target_alpha 注入 meta_stats。

    仅在 alpha_guidance_enabled 且 episode_count >= alpha_guidance_warmup 时启用引导。
    target_alpha = clip(intercept - snr_db * slope, target_min, target_max)
        默认：0dB→0.9, 30dB→0.6, 夹紧到 [0.5, 0.95]。
    """
    h_cfg = getattr(config, "hierarchical", None)
    if h_cfg is None:
        meta_stats["alpha_guidance_scale"] = 0.0
        return

    enabled = bool(getattr(h_cfg, "alpha_guidance_enabled", False))
    warmup = int(getattr(h_cfg, "alpha_guidance_warmup", 0))
    if not enabled or episode_count < warmup:
        meta_stats["alpha_guidance_scale"] = 0.0
        return

    snr_db = float(meta_stats.get("snr_db", 15.0))
    intercept = float(getattr(h_cfg, "alpha_guidance_target_intercept", 0.9))
    slope = float(getattr(h_cfg, "alpha_guidance_target_slope", 0.01))
    target_min = float(getattr(h_cfg, "alpha_guidance_target_min", 0.5))
    target_max = float(getattr(h_cfg, "alpha_guidance_target_max", 0.95))
    target = float(np.clip(intercept - slope * snr_db, target_min, target_max))

    meta_stats["alpha_guidance_scale"] = float(
        getattr(h_cfg, "alpha_guidance_scale", 0.3)
    )
    meta_stats["alpha_target"] = target


def _meta_decide(meta_ac: MetaActorCritic, env: RSMAEnergyEnv, device: torch.device):
    """运行一次慢尺度 Meta 决策并把 split_mask + group_ids 注入环境。

    Returns:
        (user_features_np, action_np, group_ids_np, log_prob_float, value_float)
        其中 action_np 在 group_level_alpha=False 时为二值 split_mask (K,)；
        在 group_level_alpha=True 时为原始采样 raw_group_alphas (n_groups,)，
        env 仅看到展开后的 per-user α (存于 env.split_mask)。
    """
    user_features = env.get_user_features()  # (K, d_user)
    user_features_tensor = torch.as_tensor(user_features, dtype=torch.float32, device=device)
    with torch.no_grad():
        primary, group_ids, log_prob, value = meta_ac.act(user_features_tensor)
    primary_np = primary.detach().cpu().numpy().astype(np.float32)
    group_ids_np = group_ids.detach().cpu().numpy().astype(np.int64)

    if getattr(meta_ac, 'group_level_alpha', False):
        # primary = per-user α (K,) (已映射到 [α_min, α_max])
        env.set_split_mask(primary_np)  # 覆盖为连续 α
        env.set_group_ids(group_ids_np)
        # Buffer 存储原始采样 (n_groups,)，供 PPO log_prob 重计算
        action_for_buffer = meta_ac.last_raw_alpha.detach().cpu().numpy().astype(np.float32)
    else:
        # Legacy: primary = 二值 split_mask (K,)
        env.set_split_mask(primary_np)
        env.set_group_ids(group_ids_np)
        action_for_buffer = primary_np

    return user_features, action_for_buffer, group_ids_np, float(log_prob.item()), float(value.item())


def _apply_delivery_safety_layer(
    worker_ac,
    obs_tensor: torch.Tensor,
    action_cont_np: np.ndarray,
    decoding_order: torch.Tensor,
    log_prob_val: float,
    cont_log_prob_val: float,
    env: RSMAEnergyEnv,
    constraint_cfg,
    device: torch.device,
):
    """Delivery-Barrier 安全层：running delivery 不可行时的动作层干预。

    (a) run_deliv = D_cum.sum()/max(A_cum.sum(), eps)；仅当 run_deliv 低于
        delivery_threshold 时执行投影，可行轨迹零干预（原样返回）。
    (b) 逐用户地板（策略输出的 [0,1] 归一化动作空间）：
        f_p_k = clip(base + queue_gain*(Q_k/Q_max)
                     + deficit_gain*relu(thr*A_cum_k - D_cum_k)/max(A_cum_k, eps),
                     0, floor_cap)
        f_d_k = clip(d_base + d_queue_gain*(Q_k/Q_max), 0, 1)，仅 Q_k>0 时施加。
    (c) 投影与网络输出同一坐标系（tanh 后 [-1,1]）：地板反映射 f_tanh=2f-1
        后 elementwise 取 max；alpha 块与离散解码顺序不动。
    (d) 用投影后的完整动作调 evaluate_actions 重算 log_prob（连续 + PL），
        替换原采样 log_prob 存入 buffer，保证 PPO ratio 的 on-policy 一致；
        value 与 done 不变。

    Returns:
        (action_np, log_prob_val, cont_log_prob_val, triggered, mean_lift)
        mean_lift 为 [0,1] 空间内被抬升分量的平均抬升幅度（诊断用）。
    """
    eps = 1e-8
    A_cum = env.traffic.total_arrived.astype(np.float64)
    D_cum = env.traffic.total_served.astype(np.float64)
    run_deliv = float(D_cum.sum() / max(float(A_cum.sum()), eps))
    threshold = float(constraint_cfg.delivery_threshold)
    if run_deliv >= threshold:
        return action_cont_np, log_prob_val, cont_log_prob_val, False, 0.0

    K = env.K
    Q = env.traffic.queue_lengths.astype(np.float64)
    q_ratio = Q / max(float(env.Q_max), eps)
    deficit = np.maximum(threshold * A_cum - D_cum, 0.0) / np.maximum(A_cum, eps)

    f_p = np.clip(
        float(constraint_cfg.dsl_floor_base)
        + float(constraint_cfg.dsl_queue_gain) * q_ratio
        + float(constraint_cfg.dsl_deficit_gain) * deficit,
        0.0, float(constraint_cfg.dsl_floor_cap),
    )
    f_d = np.clip(
        float(constraint_cfg.dsl_d_base)
        + float(constraint_cfg.dsl_d_queue_gain) * q_ratio,
        0.0, 1.0,
    )

    # tanh 空间上投影：地板反映射 f_tanh = 2f - 1
    projected = action_cont_np.astype(np.float64).copy()
    p_floor = 2.0 * f_p - 1.0
    d_floor = 2.0 * f_d - 1.0
    p_block = projected[K:2 * K]
    d_block = projected[2 * K:3 * K]
    lift_p = p_block < p_floor
    p_block[lift_p] = p_floor[lift_p]
    lift_d = (Q > 0.0) & (d_block < d_floor)  # Q_k==0 时不投影 d_ratio
    d_block[lift_d] = d_floor[lift_d]

    if not (lift_p.any() or lift_d.any()):
        return action_cont_np, log_prob_val, cont_log_prob_val, False, 0.0

    # 平均抬升幅度（换算回 [0,1] 空间，仅统计被抬升分量）
    delta01 = 0.5 * (projected - action_cont_np.astype(np.float64))
    lifted_delta = np.concatenate([delta01[K:2 * K][lift_p], delta01[2 * K:3 * K][lift_d]])
    mean_lift = float(lifted_delta.mean()) if lifted_delta.size > 0 else 0.0

    # on-policy 校正：重算投影后动作的 log_prob（连续 + PL 解码顺序）
    projected = projected.astype(np.float32)
    with torch.no_grad():
        action_t = torch.as_tensor(projected, dtype=torch.float32, device=device).unsqueeze(0)
        new_log_prob, _, _, new_cont_log_prob = worker_ac.evaluate_actions(
            obs_tensor, action_t, decoding_order
        )
    return (
        projected,
        float(new_log_prob.item()),
        float(new_cont_log_prob.item()),
        True,
        mean_lift,
    )


def _hek_collect_expert(env: RSMAEnergyEnv, config, n_episodes: int):
    """HEK-Lite: 队列感知规则专家在真实 env 中收集演示。

    每步按队列状态构造目标连续动作（3K，tanh 空间 [-1,1]，env 内部做
    (a+1)/2 映射）：
      frac_p = clip(base + 0.4*(Q/Q_max), 0.02, 1.0)   # 目标物理功率占比
      frac_d = Q / (Q.sum()+eps)                        # 目标交付比例
      alpha 块置中性 0（group_level_alpha 下会被 meta 覆盖）
    只收集连续动作；排序块复用 env.decoding_order（reset 后为 None 时按
    信道增益降序构造，与 ablation 路径同一启发式），不触碰 worker 网络
    与 torch RNG。

    Returns:
        (expert_obs, expert_act): 两个 (N, obs_dim) / (N, 3K) float32 数组。
    """
    K = env.K
    base_p = float(getattr(config.constraint, "hek_base_power", 0.5))
    max_steps_per_ep = int(getattr(config.system, "T", 200))
    expert_obs: list = []
    expert_act: list = []

    for _ in range(int(n_episodes)):
        obs, _ = env.reset()
        for _ in range(max_steps_per_ep):
            Q = np.asarray(env.traffic.queue_lengths, dtype=np.float64)
            Qmax = max(float(env.Q_max), 1e-6)
            hnorm = np.asarray(
                env._channel_obs.get("channel_gains", np.ones(K)), dtype=np.float64
            )
            frac_p = np.clip(base_p + 0.4 * (Q / Qmax), 0.02, 1.0)
            frac_d = Q / (Q.sum() + 1e-6)
            a_alpha = np.zeros(K, np.float32)
            a_p = (2.0 * frac_p - 1.0).astype(np.float32)
            a_d = (2.0 * frac_d - 1.0).astype(np.float32)
            cont_act = np.concatenate([a_alpha, a_p, a_d]).astype(np.float32)

            expert_obs.append(np.asarray(obs, dtype=np.float32).copy())
            expert_act.append(cont_act)

            # 排序块：复用 env 当前 decoding_order，否则按增益降序构造
            order = env.decoding_order
            if order is None:
                order = np.argsort(np.argsort(-hnorm)).astype(np.int64)
            full_action = np.concatenate([
                cont_act.astype(np.float64), np.asarray(order, dtype=np.float64)
            ])
            obs, _, terminated, truncated, _ = env.step(full_action)
            if terminated or truncated:
                break

    if not expert_obs:
        return (
            np.zeros((0, env.obs_dim), dtype=np.float32),
            np.zeros((0, 3 * K), dtype=np.float32),
        )
    return (
        np.stack(expert_obs).astype(np.float32),
        np.stack(expert_act).astype(np.float32),
    )


def _solver_expert_collect(env: RSMAEnergyEnv, config, meta_ac, device,
                           n_episodes: int, n_steps: int = 25, lr: float = 0.1,
                           mode: str = "teacher", student_ac=None):
    """Solver-Expert: 单槽 DGLCR surrogate 精修专家在真实 env 收集演示。

    mode="teacher"（默认，M4 口径）: 轨迹由教师动作推进（教师状态分布）。
    mode="relabel"（M5-R1, ExIt-2）: 轨迹由学生策略（student_ac 确定性 act）
    推进，标签仍为教师从学生状态逐槽 solve 的动作——DAgger 式协变量偏移校正。

    每时隙 expert.solve() 以冻结对偶变量在线精修 p_logits 并从 order 候选
    集择优，产生 (obs, 3K 连续动作, K 优先级) 三元组（order 为 env 约定的
    user→priority 值，经 main 的 4K 组装约定等价于 PL 头的 BC 目标排列）。
    与 HEK 规则专家不同，演示来自可微 surrogate 的逐槽最优解。

    Meta 决策按训练同款节奏运行（episode 起点 + T_s 段边界，transition
    丢弃），使 BC 数据覆盖真实 split/group/α 分布；worker 网络与
    其参数不被触碰。

    Returns:
        (expert_obs, expert_act, expert_orders):
        (N, obs_dim) / (N, 3K) float32 与 (N, K) int64 数组。
    """
    if mode == "relabel" and student_ac is None:
        raise ValueError("mode='relabel' 需要 student_ac（学生策略网络）")
    K = env.K
    T_s = int(getattr(config.hierarchical, "T_s", 20))
    max_steps_per_ep = int(getattr(config.system, "T", 200))
    expert = SolverExpert(env, n_steps=n_steps, lr=lr)
    expert_obs: list = []
    expert_act: list = []
    expert_orders: list = []

    for _ in range(int(n_episodes)):
        obs, _ = env.reset()
        # episode 起点注入 Meta split/groups（与主循环同款，transition 丢弃）
        _meta_decide(meta_ac, env, device)
        for _ in range(max_steps_per_ep):
            result = expert.solve(env)
            expert_obs.append(np.asarray(obs, dtype=np.float32).copy())
            expert_act.append(result["cont_action"].astype(np.float32))
            expert_orders.append(np.asarray(result["order"], dtype=np.int64))
            if mode == "relabel":
                # M5-R1: 轨迹按学生策略推进（on-policy 状态分布），标签=教师
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    s_cont, s_order, _, _, _ = student_ac.act(
                        obs_t, deterministic=True)
                step_action = np.concatenate([
                    s_cont.squeeze(0).cpu().numpy(),
                    s_order.squeeze(0).cpu().numpy().astype(np.float64),
                ])
            else:
                step_action = result["action"]
            obs, _, terminated, truncated, _ = env.step(step_action)
            if terminated or truncated:
                break
            if env.step_in_episode % T_s == 0:
                _meta_decide(meta_ac, env, device)

    if not expert_obs:
        return (
            np.zeros((0, env.obs_dim), dtype=np.float32),
            np.zeros((0, 3 * K), dtype=np.float32),
            np.zeros((0, K), dtype=np.int64),
        )
    return (
        np.stack(expert_obs).astype(np.float32),
        np.stack(expert_act).astype(np.float32),
        np.stack(expert_orders).astype(np.int64),
    )


# ---- M5-F5: 训练中评估守卫参考值 ----
# 出处: logs/_eval_m4_bc_init.txt（bc_init.pt 官方口径 7 点 × 50ep, 2026-09-07）
# 守卫仅用 {0,5,30} 三点: 0/5dB = Run 3 历史缺口区, 30dB = M4 微调 queue 门首先失守点
_GUARD_POINTS = (0.0, 5.0, 30.0)
# M5.2-F5b: 七点全量参考表（单一事实源）。M5.1 终审证明旧 ok 判据只看 avg_deliv、
# 不看逐点门与队列，A2 策略爆炸（KL=3e4）仍 19/19 全绿；v2 判据补齐这两个洞。
# 队列参考取 bc_init 逐点值（比 SDMA×1.1 的 37.92 上限紧得多，能分辨深过载区漂移）。
_GUARD_REF = {
    0.0:  dict(bc_deliv=0.882, bc_queue=2.92,  sdma_deliv=0.856, sdma_queue=3.46),
    5.0:  dict(bc_deliv=0.989, bc_queue=0.43,  sdma_deliv=0.979, sdma_queue=0.68),
    10.0: dict(bc_deliv=0.993, bc_queue=0.74,  sdma_deliv=0.961, sdma_queue=2.28),
    15.0: dict(bc_deliv=0.986, bc_queue=2.47,  sdma_deliv=0.887, sdma_queue=11.42),
    20.0: dict(bc_deliv=0.964, bc_queue=8.70,  sdma_deliv=0.795, sdma_queue=22.39),
    25.0: dict(bc_deliv=0.924, bc_queue=18.85, sdma_deliv=0.724, sdma_queue=28.76),
    30.0: dict(bc_deliv=0.871, bc_queue=31.64, sdma_deliv=0.664, sdma_queue=34.47),
}


def guard_plateau_stop(stall_count: int, patience: int, episode_count: int,
                        min_episodes: int, enabled: bool) -> bool:
    """guard-plateau 早停判定（P1，纯函数便于单测）。

    同时满足才停：开关开 + 已训够 min_episodes + 连续 patience 次 guard 评估
    未刷新 best_avg_deliv。
    """
    return bool(enabled) and int(episode_count) >= int(min_episodes)         and int(stall_count) >= max(int(patience), 1)


def _guard_ref_avg(points) -> tuple:
    """给定守卫点集 → (bc_init delivery 均值, bc_init queue 均值)。"""
    return (
        float(np.mean([_GUARD_REF[float(s)]["bc_deliv"] for s in points])),
        float(np.mean([_GUARD_REF[float(s)]["bc_queue"] for s in points])),
    )


def _guard_run_point(env_g, worker_ac, meta_ac, device, config,
                     snr_db: float, n_eps: int, seed0: int):
    """单 SNR 点跑 n_eps 个确定性评估 episode → (avg_delivery, avg_mean_queue).

    口径对齐 evaluate_snr_sweep.py（不可变）: reset(options={'snr_db'}) 固定 SNR，
    delivery_ratio 取末步 info 的 episode 累计值，mean_queue 取逐步平均。
    """
    T = int(getattr(config.system, "T", 200))
    T_s = int(getattr(config.hierarchical, "T_s", 20))
    delivs, queues = [], []
    for i in range(int(n_eps)):
        obs, _ = env_g.reset(seed=seed0 + i, options={"snr_db": float(snr_db)})
        # 与主循环其余 reset 处一致：按新 SNR 重设桶专家路由，避免守卫点
        # 沿用上个 episode 的 stale active_bucket（默认 use_bucket_experts=True 时）
        _set_worker_bucket(worker_ac, env_g)
        _meta_decide(meta_ac, env_g, device)
        last_deliv, q_sum, n_step = 0.0, 0.0, 0
        for _ in range(T):
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                cont, order, _, _, _ = worker_ac.act(obs_t, deterministic=True)
            act = np.concatenate([
                cont.squeeze(0).cpu().numpy(),
                order.squeeze(0).cpu().numpy().astype(np.float64),
            ])
            obs, _, term, trunc, info = env_g.step(act)
            q_sum += float(info.get("mean_queue", 0.0))
            n_step += 1
            last_deliv = float(info.get("delivery_ratio", last_deliv))
            if term or trunc:
                break
            if env_g.step_in_episode % T_s == 0:
                _meta_decide(meta_ac, env_g, device)
        delivs.append(last_deliv)
        queues.append(q_sum / max(n_step, 1))
    return float(np.mean(delivs)), float(np.mean(queues))


def _guard_check(guard_env, worker_ac, meta_ac, device, config) -> dict:
    """守卫 mini-eval: 配置点集 × eval_guard_episodes，官方逐点门 + bc 参考对照。

    M5.2-F5b: v2 判据补上旧版的两个洞（A2 策略爆炸仍全绿的根因）——
    ok 不再只看 avg_deliv，还要求逐点门全过、且平均队列对照 bc 参考。
    """
    n_eps = int(getattr(config.ppo, "eval_guard_episodes", 4))
    pts_cfg = tuple(getattr(config.ppo, "eval_guard_snr_points", ()) or ())
    points = tuple(float(s) for s in pts_cfg) if pts_cfg else _GUARD_POINTS
    v2 = bool(getattr(config.ppo, "eval_guard_v2", False))
    q_ratio = float(getattr(config.ppo, "eval_guard_queue_ratio", 1.1))

    service = sorted(_GUARD_REF)
    per_point, points_met = [], 0
    for snr in points:
        try:
            ref = _GUARD_REF[float(snr)]
        except KeyError:
            raise ValueError(
                f"eval_guard_snr_points 含非参考网格点 {snr!r}（参考表仅支持 "
                f"{service}）；请在 scripts/train.py 层校验口径"
            ) from None
        j = points.index(snr)
        dv, q = _guard_run_point(
            guard_env, worker_ac, meta_ac, device, config,
            snr, n_eps, seed0=2000 + 100 * j)
        ok_pt = bool(dv >= ref["sdma_deliv"] and q <= 1.1 * ref["sdma_queue"])
        points_met += int(ok_pt)
        per_point.append(
            {"snr_db": snr, "deliv": dv, "queue": q, "met": ok_pt})
    avg_deliv = float(np.mean([p["deliv"] for p in per_point]))
    avg_queue = float(np.mean([p["queue"] for p in per_point]))
    margin = float(getattr(config.ppo, "eval_guard_margin", 0.02))
    bc_ref_avg, bc_queue_avg = _guard_ref_avg(points)

    deliv_ok = bool(avg_deliv >= bc_ref_avg - margin)
    if v2:
        ok = bool(deliv_ok
                  and points_met == len(points)
                  and avg_queue <= bc_queue_avg * q_ratio)
    else:
        ok = deliv_ok
    return {
        "avg_deliv": avg_deliv,
        "avg_queue": avg_queue,
        "points_met": points_met,
        "n_points": len(points),
        "ok": ok,
        "per_point": per_point,
    }


def _guard_rollback(worker_ac, meta_ac, worker_trainer, last_good,
                    halvings: int, lr_groups: list, config) -> int:
    """M5-F5 劣化处置：恢复 last-good 权重 + actor lr 减半（超限后冻结 actor）。

    _base_lrs 同步减半，使后续的线性衰减 / KL 衰减调度在新基准上生效。
    Returns: 更新后的回滚计数。
    """
    halvings = int(halvings) + 1
    worker_ac.load_state_dict(last_good[0])
    meta_ac.load_state_dict(last_good[1])
    max_halv = int(getattr(config.ppo, "eval_guard_max_halvings", 3))
    if halvings <= max_halv and lr_groups:
        lr_floor = float(getattr(config.ppo, "eval_guard_lr_floor", 1e-5))
        for gi in lr_groups:
            grp = worker_trainer.optimizer.param_groups[gi]
            grp["lr"] = max(float(grp["lr"]) * 0.5, lr_floor)
            worker_trainer._base_lrs[gi] = max(
                float(worker_trainer._base_lrs[gi]) * 0.5, lr_floor)
        print(f"[Guard] ROLLBACK #{halvings}: 恢复 last-good 权重, "
              f"actor lr 减半 -> "
              f"{worker_trainer.optimizer.param_groups[lr_groups[0]]['lr']:.2e}")
    else:
        for gi in lr_groups:
            worker_trainer.optimizer.param_groups[gi]["lr"] = 0.0
            worker_trainer._base_lrs[gi] = 0.0
        print(f"[Guard] ROLLBACK #{halvings} 超过上限({max_halv}): "
              f"actor 冻结至训练结束（critic/Meta 继续）")
    return halvings


def train(config=None):
    """双层 PPO 训练主循环。

    1. 初始化环境、Worker（PPO）、Meta（Self-Attention PPO）。
    2. 按 rollout_steps 收集 Worker 数据；每 T_s 步执行 Meta 决策并累积 Meta transition。
    3. Worker：每完成一次 rollout 立即 PPO 更新；
       Meta：积累到 meta_update_threshold 后执行 PPO 更新并清空 Meta buffer。
    4. 周期性保存 checkpoint 与 best.pt（包含 worker_ac 与 meta_ac 权重）。
    """
    if config is None:
        config = Config()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device)

    # ------------------------------ 环境 ------------------------------ #
    env = RSMAEnergyEnv(config)
    obs_dim = env.obs_dim

    # ----------------------------- Worker ----------------------------- #
    K = config.system.K
    continuous_dim = 3 * K    # alpha, p_total, d_ratio
    worker_action_dim = 4 * K  # 3K连续 + K解码顺序

    use_attention = bool(getattr(config.ppo, "use_attention", False))
    c3po_enabled = bool(
        getattr(config, "constraint", None) is not None
        and getattr(config.constraint, "c3po_enabled", False)
    )
    if use_attention:
        # P1: Self-Attention Worker（让 SNR/ε 全局信号通过 FiLM 注入避免 LN 稀释）
        # 时序历史窗口开启时，env 在 per-user 特征区末尾追加了最近 H 槽的
        # 逐用户特征快照，构造 Worker 的 d_user 相应扩展 H·F（d_global 仍为
        # 3 维尾 time_ratio/snr/eps，n_global_inject=2 的 (snr,eps) 注入不变）
        worker_d_user_base = int(getattr(config.ppo, "worker_d_user", 10))
        hist_window = int(getattr(config.system, "obs_history_window", 0))
        hist_feats = tuple(getattr(config.system, "obs_history_features", ()))
        worker_d_user_eff = worker_d_user_base
        if hist_window > 0:
            worker_d_user_eff += hist_window * len(hist_feats)
            print(
                f"[Train] obs_history_window={hist_window}, "
                f"features={list(hist_feats)}, "
                f"worker_d_user: {worker_d_user_base} -> {worker_d_user_eff}"
            )
        if bool(getattr(config.system, "obs_prune_constant_blocks", False)):
            # A-prune：观测移除 3 个常数原生块 → Worker d_user 相应 -3（与 env 对齐）
            worker_d_user_eff -= 3
            print(
                f"[Train] obs_prune_constant_blocks=True: "
                f"worker_d_user -> {worker_d_user_eff}"
            )
        worker_ac = ActorCriticAttn(
            obs_dim=obs_dim,
            action_dim=worker_action_dim,
            K=K,
            d_user=worker_d_user_eff,
            d_global=obs_dim - K * worker_d_user_eff,
            hidden_dim=getattr(config.ppo, "worker_attn_hidden", 128),
            n_heads=getattr(config.ppo, "worker_attn_heads", 4),
            n_attn_layers=getattr(config.ppo, "worker_attn_layers", 2),
            use_plackett_luce=True,
            use_mean_field_critic=getattr(config.ppo, "use_mean_field_critic", False),
            mean_field_hidden_dim=getattr(config.ppo, "mean_field_hidden_dim", 64),
            qos_drift_weights=tuple(getattr(config.system, "qos_drift_weights", ()))
            if getattr(config.ppo, "use_mean_field_critic", False) else None,
            use_bucket_experts=getattr(config.ppo, "use_bucket_experts", False),
            n_experts=getattr(config.ppo, "n_bucket_experts", 4),
            bucket_snr_min=getattr(config.ppo, "bucket_snr_min", -5.0),
            bucket_snr_max=getattr(config.ppo, "bucket_snr_max", 35.0),
            use_c3po_cost_head=c3po_enabled,
            c3po_n_points=getattr(config.constraint, "c3po_n_points", 7) if c3po_enabled else 7,
            backpressure_beta=getattr(config.ppo, "backpressure_beta", 0.0),
            backpressure_mode=getattr(config.ppo, "backpressure_mode", "queue"),
            fix_drain=getattr(config.ppo, "worker_fix_drain", False),
            fix_order=getattr(config.ppo, "worker_fix_order", False),
        ).to(device)
        _bp_modes = ['queue', 'queue_channel', 'mw_residual', 'mw_gated']
        if int(worker_ac.backpressure_mode_buf) == 3:
            print("[Train] Worker backpressure prior ON: mode=mw_gated "
                  "(闭式 max-weight 基座 × 可学习负载门 g(s) + head 残差, "
                  f"初始 g={torch.sigmoid(worker_ac.mw_gate.bias).item():.3f})")
        elif int(worker_ac.backpressure_mode_buf) == 2:
            print("[Train] Worker backpressure prior ON: mode=mw_residual "
                  "(闭式 max-weight 基座 + head 残差, beta 不适用)")
        elif float(worker_ac.backpressure_beta_buf) > 0.0:
            print(f"[Train] Worker backpressure prior ON: mode="
                  f"{_bp_modes[int(worker_ac.backpressure_mode_buf)]}, "
                  f"beta={float(worker_ac.backpressure_beta_buf):.3f}")
        print(f"[Train] Worker = Self-Attention (hidden={getattr(config.ppo, 'worker_attn_hidden', 128)}, "
              f"heads={getattr(config.ppo, 'worker_attn_heads', 4)}, "
              f"layers={getattr(config.ppo, 'worker_attn_layers', 2)})")
    else:
        worker_ac = ActorCritic(
            obs_dim=obs_dim,
            action_dim=worker_action_dim,
            hidden_dim=config.ppo.hidden_dim,
            n_layers=config.ppo.n_layers,
            use_plackett_luce=True,
        ).to(device)
    worker_trainer = PPOTrainer(worker_ac, config.ppo)
    worker_buffer = RolloutBuffer(
        rollout_steps=config.ppo.rollout_steps,
        obs_dim=obs_dim,
        action_dim=continuous_dim,  # buffer仅存连续动作(3K)
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        device=config.device,
        decoding_order_dim=K,  # 额外存储K维解码顺序
    )

    # ------------------------------ Meta ------------------------------ #
    meta_ac = MetaActorCritic(
        K=K,
        d_user=config.hierarchical.meta_d_user,
        hidden_dim=config.hierarchical.meta_hidden_dim,
        n_heads=config.hierarchical.meta_n_heads,
        n_groups=config.hierarchical.n_groups,
        group_tau_init=getattr(config.hierarchical, 'meta_group_tau_init', 2.0),
        group_tau_min=getattr(config.hierarchical, 'meta_group_tau_min', 0.5),
        balance_coef=getattr(config.hierarchical, 'meta_balance_coef', 0.1),
        group_level_alpha=getattr(config.hierarchical, 'group_level_alpha', False),
        group_alpha_min=getattr(config.hierarchical, 'group_alpha_min', 0.1),
        group_alpha_max=getattr(config.hierarchical, 'group_alpha_max', 0.95),
        group_alpha_log_std_init=getattr(
            config.hierarchical, 'group_alpha_log_std_init', -0.5
        ),
    ).to(device)
    meta_buffer = MetaRolloutBuffer(max_size=500)
    meta_trainer = MetaPPOTrainer(meta_ac, config.hierarchical)

    T = config.system.T
    T_s = max(int(config.hierarchical.T_s), 1)

    # Task#11 消融标志
    ablation = getattr(config, 'ablation', None)
    disable_meta = bool(ablation is not None and ablation.disable_meta)
    if ablation is not None:
        print(
            f"[Ablation] disable_split={ablation.disable_split}, "
            f"disable_group={ablation.disable_group}, "
            f"disable_sic_order={ablation.disable_sic_order}, "
            f"disable_meta={ablation.disable_meta}"
        )

    # Task#17: PPO-Lagrangian 硬约束
    constraint_cfg = getattr(config, 'constraint', None)
    use_lagrangian = bool(
        constraint_cfg is not None and getattr(constraint_cfg, 'use_lagrangian', False)
    )
    delivery_constraint: 'DeliveryConstraint | None' = None
    delivery_threshold = float(getattr(constraint_cfg, 'delivery_threshold', 0.98)) if constraint_cfg is not None else 0.98
    if use_lagrangian:
        if c3po_enabled:
            # C3PO 替代 PID-Lagrangian：冻结 delivery 对偶通道
            constraint_cfg.use_lagrangian = False
            use_lagrangian = False
            print(
                f"[C3PO] freezing Lagrangian delivery channel (c3po_enabled=True)"
            )
        else:
            delivery_constraint = DeliveryConstraint(
                threshold=constraint_cfg.delivery_threshold,
                lr=constraint_cfg.lagrangian_lr,
                init_lambda=constraint_cfg.lagrangian_init,
                lambda_max=constraint_cfg.lagrangian_max,
                warmup_episodes=constraint_cfg.lagrangian_warmup,
                lambda_decay=getattr(constraint_cfg, 'lagrangian_decay', 0.999),
                success_ema_decay=getattr(constraint_cfg, 'sfl_success_ema_decay', 0.99),
            )
            delivery_threshold = float(getattr(constraint_cfg, 'delivery_threshold', 0.98))
            print(
                f"[Lagrangian] enabled: threshold={constraint_cfg.delivery_threshold}, "
                f"lr={constraint_cfg.lagrangian_lr}, init={constraint_cfg.lagrangian_init}, "
                f"max={constraint_cfg.lagrangian_max}, warmup={constraint_cfg.lagrangian_warmup} ep, "
                f"decay={getattr(constraint_cfg, 'lagrangian_decay', 0.999)}"
            )

    # Neely per-user Delivery Virtual Queue (DVQ)
    dvq: 'DeliveryVirtualQueue | None' = None
    if constraint_cfg is not None and getattr(constraint_cfg, 'use_dvq', False):
        dvq = DeliveryVirtualQueue(
            K=config.system.K,
            threshold=constraint_cfg.delivery_threshold,
            dvq_weight=constraint_cfg.dvq_weight,
            z_max=constraint_cfg.dvq_z_max,
            warmup_episodes=constraint_cfg.dvq_warmup,
            cvar_alpha=config.system.dvq_cvar_alpha,
        )
        print(
            f"[DVQ] enabled: K={config.system.K}, threshold={constraint_cfg.delivery_threshold}, "
            f"weight={constraint_cfg.dvq_weight}, z_max={constraint_cfg.dvq_z_max}, "
            f"warmup={constraint_cfg.dvq_warmup} ep, cvar_alpha={config.system.dvq_cvar_alpha}"
        )

    # RCPO cost-critic 优势分解：delivery 约束从 reward 减法迁移到独立 cost 通道，
    # reward 恢复纯 EE 平稳；λ 仍由 delivery_constraint 每 episode 更新（逻辑不变）。
    # 词典序梯度投影依赖 RCPO 提供独立的 delivery 约束梯度，故强制开启。
    use_lex_proj = bool(
        constraint_cfg is not None
        and getattr(constraint_cfg, 'use_lexicographic_projection', False)
    )
    use_rcpo = bool(
        use_lagrangian
        and constraint_cfg is not None
        and (getattr(constraint_cfg, 'use_rcpo_cost_critic', False) or use_lex_proj)
    )
    rcpo_lambda_scale = (
        float(getattr(constraint_cfg, 'lex_proj_lambda_scale', 0.2))
        if use_lex_proj and constraint_cfg is not None
        else float(getattr(constraint_cfg, 'rcpo_lambda_scale', 0.2))
        if constraint_cfg is not None else 0.2
    )
    if use_lex_proj:
        print(
            f"[LexProj] enabled: per-SNR-bucket gradient projection | "
            f"lambda_scale={rcpo_lambda_scale}"
        )
    if use_rcpo:
        worker_trainer.cost_critic_coef = float(getattr(constraint_cfg, 'cost_critic_coef', 0.5))
        worker_trainer.cost_gamma = float(getattr(constraint_cfg, 'cost_gamma', 0.99))
        print(
            f"[RCPO] enabled: reward 保持纯 EE（不减 penalty），约束经 "
            f"A_eff=A_r-λ_scaled·A_c 施加 | lambda_scale={rcpo_lambda_scale}, "
            f"cost_critic_coef={worker_trainer.cost_critic_coef}, "
            f"cost_gamma={worker_trainer.cost_gamma}"
        )

    # Delivery-Barrier 安全层（动作层干预）：仅在 worker rollout 采样处生效，
    # 不改 reward/advantage/critic 及任何网络参数；False 时走原始采样路径。
    dsl_enabled = bool(
        constraint_cfg is not None
        and getattr(constraint_cfg, 'use_delivery_safety_layer', False)
    )
    if dsl_enabled:
        print(
            f"[DSL] enabled: p_floor={constraint_cfg.dsl_floor_base}"
            f"+{constraint_cfg.dsl_queue_gain}*(Q/Q_max)"
            f"+{constraint_cfg.dsl_deficit_gain}*deficit (cap={constraint_cfg.dsl_floor_cap}) | "
            f"d_floor={constraint_cfg.dsl_d_base}+{constraint_cfg.dsl_d_queue_gain}*(Q/Q_max) | "
            f"gate: run_deliv<{constraint_cfg.delivery_threshold}"
        )

    # CRPO-S: 按SNR分桶的episode级原始目标切换。桶不可行时整段episode改用
    # 有界排水盈余奖励(纯吞吐/排队列目标)，桶可行时保持EE+Lagrangian现状，
    # 迟滞带防止目标在相邻episode间抖振；crpo_on=False时行为与现状完全一致。
    crpo_on = bool(getattr(constraint_cfg, 'use_crpo_switch', False))
    crpo_mode = {}   # dict: snr_bucket -> bool, True=可行性模式(排水盈余奖励), False=EE模式(现状)

    # ----------------------- SNR 课程调度器 ----------------------- #
    curriculum_scheduler = None
    if getattr(config.ppo, 'curriculum_enabled', False):
        from src.training.curriculum_scheduler import (
            SNRCurriculumScheduler,
            CurriculumStage,
        )
        stages_config = getattr(config.ppo, 'curriculum_stages', None)
        if stages_config:
            stages = [CurriculumStage(**s) for s in stages_config]
        else:
            total_ep = config.ppo.total_episodes
            stages = [
                CurriculumStage(0, total_ep // 5, 0.0, 10.0),
                CurriculumStage(total_ep // 5, total_ep * 3 // 5, 0.0, 20.0),
                CurriculumStage(total_ep * 3 // 5, total_ep, 0.0, 30.0),
            ]
        curriculum_scheduler = SNRCurriculumScheduler(stages)
        print(f"[Curriculum] SNR curriculum enabled, {len(stages)} stages")
        for i, s in enumerate(stages):
            print(
                f"  Stage {i}: ep [{s.start_episode}, {s.end_episode}) "
                f"SNR [{s.snr_min_db}, {s.snr_max_db}] dB"
            )

    # 日志目录
    log_dir = os.path.join("logs", config.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 保存配置
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        cfg_dict = {
            "seed": config.seed,
            "device": config.device,
            "exp_name": config.exp_name,
            "ppo": vars(config.ppo),
            "system": vars(config.system),
            "lyapunov": vars(config.lyapunov),
            "reward": vars(config.reward),
            "hierarchical": vars(config.hierarchical),
        }
        if ablation is not None:
            cfg_dict["ablation"] = vars(ablation)
        if constraint_cfg is not None:
            cfg_dict["constraint"] = vars(constraint_cfg)
        json.dump(cfg_dict, f, indent=2, default=str)

    # ----------------------------- 训练循环 ----------------------------- #
    # SPLC: Self-Paced Load Curriculum（任务分布层干预）
    # 每次 env.reset() 前按窗口化 running delivery 动态缩放 env.load_ratio_min/max；
    # flag 关闭时整段逻辑跳过，env.load_ratio_min/max 保持配置原值。
    lc_on = constraint_cfg is not None and getattr(constraint_cfg, 'use_load_curriculum', False)
    if lc_on:
        lc_base_min = float(env.load_ratio_min)   # 记录配置原值
        lc_base_max = float(env.load_ratio_max)
        lc_scale = float(constraint_cfg.lc_init_scale)
        lc_final = float(constraint_cfg.lc_final_scale)
        lc_deliv_window = []          # 存最近每 episode 的 running delivery
        lc_pass_count = 0             # 连续达标计数
        lc_level_ep = 0               # 当前档已停留 episode 数

        def _apply_lc_scale():
            # 缩放两端点后写回 env 实例属性；下次 reset() 会据此现算到达率
            env.load_ratio_min = lc_base_min * lc_scale
            env.load_ratio_max = lc_base_max * lc_scale

        print(
            f"[SPLC] enabled: scale {lc_scale:.2f}->{lc_final:.2f}, "
            f"target={constraint_cfg.lc_delivery_target}, step={constraint_cfg.lc_step}, "
            f"window={constraint_cfg.lc_window}, patience={constraint_cfg.lc_patience}, "
            f"min_ep={constraint_cfg.lc_min_episodes_per_level} | "
            f"base load_ratio=[{lc_base_min}, {lc_base_max}]"
        )

    # 课程调度器：首次 reset 前应用 episode 0 的 SNR 范围
    if curriculum_scheduler is not None:
        curriculum_scheduler.update_env(env, 0)

    # SPLC: 叠加在 curriculum_scheduler 设定的基线上，episode 0 生效
    if lc_on:
        _apply_lc_scale()

    # CuRLTraC: 性能触发式难度退火课程（MVP）
    curl_on = bool(getattr(config.ppo, 'use_curriculum', False))
    if curl_on:
        curl_stage = 0
        curl_window = []
        curl_low_streak = 0
        curl_stage_eps = 0
        curl_n = int(getattr(config.ppo, 'curriculum_n_stages', 5))
        curl_s0 = float(getattr(config.ppo, 'curriculum_snr_min_start', 5.0))
        curl_s1 = float(getattr(config.ppo, 'curriculum_snr_min_end', -5.0))
        curl_l0 = float(getattr(config.ppo, 'curriculum_load_ratio_max_start', 0.8))
        curl_l1 = float(getattr(config.ppo, 'curriculum_load_ratio_max_end', 1.3))
        curl_min_eps = int(getattr(config.ppo, 'curriculum_min_stage_episodes', 50))

        def _curl_apply(st):
            a = st / max(curl_n - 1, 1)
            env.snr_min_db = curl_s0 + a * (curl_s1 - curl_s0)
            if getattr(config.ppo, 'curriculum_apply_snr_max', True):
                env.snr_max_db = float(getattr(config.system, 'snr_max_db', 35.0))
            env.load_ratio_max = curl_l0 + a * (curl_l1 - curl_l0)
            config.system.snr_min_db = env.snr_min_db
            config.system.load_ratio_max = env.load_ratio_max

        _curl_apply(curl_stage)
        print(
            f"[CuRLTraC] enabled: stages={curl_n}, beta={getattr(config.ppo, 'curriculum_beta', 0.5)}, "
            f"fallback={getattr(config.ppo, 'curriculum_fallback_beta', 0.3)}, "
            f"window={getattr(config.ppo, 'curriculum_window', 50)}"
        )

    # C3PO: 初始化逐 SNR 点 delivery EMA 状态
    c3po_state = None
    if c3po_enabled:
        c3po_state = _c3po_init_state(
            target_scale=float(getattr(config.ppo, 'hard_mining_target', 1.0)),
            ema_decay=0.99,
        )
        print(
            f"[C3PO] enabled: n_points={getattr(config.constraint, 'c3po_n_points', 7)}, "
            f"nu=[{getattr(config.constraint, 'c3po_nu_init', 0.05):.3f}, "
            f"{getattr(config.constraint, 'c3po_nu_final', 0.5):.3f}], "
            f"kappa={getattr(config.constraint, 'c3po_kappa', 1.0):.3f}"
        )

    # CuRLTraC-ALP: 定向硬例挖掘状态初始化
    hm_on = bool(getattr(config.ppo, 'hard_mining_enable', False))
    # C3PO 模式下关闭硬例挖掘加权与未达标点熵激励，切断正反馈回退环
    if c3po_enabled:
        hm_on = False
        print("[HardMining] disabled due to C3PO")
    # v2: CuRLTraC 课程与硬例挖掘可以共存——课程在早期阶段推达标，
    # 硬例挖掘在课程冻结后继续推未达标点。移除旧冲突逻辑。
    if hm_on and (curl_on or curriculum_scheduler is not None):
        pass
    hm_point_targets = tuple(getattr(config.ppo, 'hm_point_targets', ()) or ())
    hm_state = (
        _hm_init_state(float(getattr(config.ppo, 'hard_mining_target', 1.0)),
                       hm_point_targets)
        if hm_on else None
    )
    hm_w_hard = float(getattr(config.ppo, 'point_w_hard', getattr(config.ppo, 'hard_mining_w_hard', 2.0)))
    hm_decay = float(getattr(config.ppo, 'hard_mining_ema_decay', 0.99))
    hm_floor = float(getattr(config.ppo, 'point_base_floor', getattr(config.ppo, 'hard_mining_floor_frac', 0.5)))
    hm_streak_mult = float(getattr(config.ppo, 'streak_mult', 1.5))
    hm_streak_cap = float(getattr(config.ppo, 'streak_cap', 4.0))
    hm_fail_entropy_scale = float(getattr(config.ppo, 'fail_entropy_scale', 1.5))
    hm_rng = np.random.default_rng(config.seed) if hm_on else None
    hm_entropy_scale = 1.0
    if hm_on:
        pl_str = (
            f"perf_linked(c={getattr(config.ppo, 'entropy_link_coef', 1.0)})"
            if getattr(config.ppo, 'entropy_performance_linked', False)
            else f"fixed_fail_scale={hm_fail_entropy_scale}"
        )
        tgt_str = (
            f"targets=[{', '.join(f'{t:.3f}' for t in hm_state['targets'])}](override)"
            if hm_point_targets else "targets=SDMA(default)"
        )
        print(
            f"[HardMining] enabled: granularity=point, w_hard={hm_w_hard}, ema_decay={hm_decay}, "
            f"floor={hm_floor}/{len(_HM_TARGETS)}, streak_mult={hm_streak_mult}, "
            f"streak_cap={hm_streak_cap}, entropy={pl_str} | {tgt_str}"
        )

    reset_options = None
    if hm_on:
        sampled_snr, _, is_failing, hm_entropy_scale = _hm_sample_snr(
            hm_state, hm_w_hard, hm_floor, hm_streak_mult, hm_streak_cap, hm_rng,
            fail_entropy_scale=hm_fail_entropy_scale,
            performance_linked=getattr(config.ppo, 'entropy_performance_linked', False),
            link_coef=getattr(config.ppo, 'entropy_link_coef', 1.0),
        )
        reset_options = {'snr_db': sampled_snr}
    obs, _ = env.reset(seed=config.seed, options=reset_options)
    _set_worker_bucket(worker_ac, env)

    # 首段 Meta 决策（t=0 之前必须先注入 split_mask + group_ids）
    prev_meta_obs, prev_split, prev_group_ids, prev_log_prob, prev_value = _meta_decide(
        meta_ac, env, device
    )
    meta_stats = _empty_meta_stats()
    step_delivery_ema = None  # SLCP-D: per-step 即时交付率 EMA，每 episode 重置

    total_steps = 0
    episode_count = 0
    episode_reward = 0.0
    episode_length = 0
    episode_rewards: list = []
    meta_segment_rewards: list = []

    # 外层 Dinkelbach eta 更新：每 episode 累计 Σrate / Σpower，每 10 episode 更新一次
    dinkelbach_ep_rate_sum = 0.0
    dinkelbach_ep_power_sum = 0.0
    dinkelbach_episode_ee_buffer: list = []
    dinkelbach_first_anchor_done = False

    start_time = time.time()
    max_total_steps = config.ppo.total_episodes * config.system.T

    # ---------- Resume from checkpoint ---------- #
    resume_ckpt_path = getattr(config, "resume_ckpt", None)
    if resume_ckpt_path:
        print(f"[Resume] Loading checkpoint: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        # 使用 padding-aware 加载（处理 v12→v13 obs_dim/d_user 维度变化）
        worker_padded = _load_state_dict_with_padding(worker_ac, ckpt["worker_state_dict"], "Worker")
        # 桶专家：旧 checkpoint 无 expert_heads，需从参考头同步保持 step-0 等价
        if getattr(worker_ac, "use_bucket_experts", False):
            if not any(k.startswith("expert_heads.") for k in ckpt.get("worker_state_dict", {}).keys()):
                worker_ac.sync_expert_heads_from_reference()
                print("[Resume] Bucket experts synced from reference user_head")
        meta_padded = _load_state_dict_with_padding(meta_ac, ckpt["meta_state_dict"], "Meta")
        # Optimizer state: 维度变化后 momentum buffer 形状不兼容，必须跳过；
        # param 组结构变化（如新加 mw_gate 组/参数）同样会使 load_state_dict 抛
        # "parameter group that doesn't match"，一律降级为跳过（模型权重仍完整加载）。
        opt_ok = (
            not (worker_padded or meta_padded)
            and _optimizer_state_compatible(worker_trainer.optimizer, ckpt.get("worker_optimizer"))
            and _optimizer_state_compatible(meta_trainer.optimizer, ckpt.get("meta_optimizer"))
        )
        if opt_ok:
            worker_trainer.optimizer.load_state_dict(ckpt["worker_optimizer"])
            meta_trainer.optimizer.load_state_dict(ckpt["meta_optimizer"])
            print("[Resume] Optimizer states loaded successfully.")
        else:
            print("[Resume] Model weights loaded; optimizer state skipped "
                  "(dimension or param-group change) - re-init from scratch.")

        # Resume 时可选降低 actor lr、收紧 clip 信任域
        resume_lr_scale = float(getattr(config.ppo, 'resume_lr_actor_scale', 1.0))
        if resume_lr_scale > 0.0 and resume_lr_scale != 1.0:
            for g in [worker_trainer.optimizer.param_groups[0], worker_trainer.optimizer.param_groups[2]]:
                g['lr'] = float(g['lr']) * resume_lr_scale
            print(f"[Resume] actor/log_std lr scaled by {resume_lr_scale}")
        resume_clip = float(getattr(config.ppo, 'resume_clip_ratio', -1.0))
        if resume_clip > 0.0:
            worker_trainer.clip_ratio = resume_clip
            print(f"[Resume] clip_ratio overridden to {resume_clip}")

        total_steps = int(ckpt.get("total_steps", 0))
        episode_count = int(ckpt.get("episode", 0))
        update_count_resume = int(ckpt.get("update_count", 0))
        meta_update_count_resume = int(ckpt.get("meta_update_count", 0))
        # Lagrangian state
        if delivery_constraint is not None:
            lag_state = ckpt.get("lagrangian_state")
            n_buckets = delivery_constraint.n_snr_buckets

            def _to_bucket_array(val, default_arr):
                # 兼容标量(旧 ckpt) 与数组(新 ckpt): 标量广播到各桶；
                # 数组长度匹配则直接用，否则用其均值广播。
                if val is None:
                    return default_arr
                arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
                if arr.size == n_buckets:
                    return arr.astype(np.float64)
                return np.full(n_buckets, float(np.mean(arr)), dtype=np.float64)

            if lag_state is not None:
                delivery_constraint.lambda_val = _to_bucket_array(
                    lag_state.get("lambda"), delivery_constraint.lambda_val
                )
                delivery_constraint.integral = _to_bucket_array(
                    lag_state.get("integral"), delivery_constraint.integral
                )
                delivery_constraint.episode_count = int(lag_state.get("episode_count", 0))
                hist = lag_state.get("delivery_history")
                if hist:
                    # 新格式为 list-of-lists（每桶一条）；旧格式为 flat list（广播到桶0）。
                    if len(hist) > 0 and isinstance(hist[0], (list, tuple, np.ndarray)):
                        delivery_constraint.delivery_history = [
                            list(hist[b]) if b < len(hist) else []
                            for b in range(n_buckets)
                        ]
                    else:
                        delivery_constraint.delivery_history = [
                            list(hist) if b == 0 else []
                            for b in range(n_buckets)
                        ]
            # 允许 CLI 覆盖（适用于旧 ckpt 没保存 lagrangian 状态的场景）
            override_lambda = getattr(config, "resume_lambda", None)
            override_ep = getattr(config, "resume_episode_count", None)
            if override_lambda is not None:
                delivery_constraint.lambda_val = np.full(
                    n_buckets, float(override_lambda), dtype=np.float64
                )
            if override_ep is not None:
                delivery_constraint.episode_count = int(override_ep)
            print(
                f"[Resume] Lagrangian: λ={delivery_constraint.get_lambda():.3f}, "
                f"episode_count={delivery_constraint.episode_count}"
            )
        if dvq is not None:
            dvq_state = ckpt.get("dvq_state")
            if dvq_state is not None:
                z_saved = dvq_state.get("Z")
                if z_saved is not None:
                    z_arr = np.asarray(z_saved, dtype=float)
                    if z_arr.shape == dvq.Z.shape:
                        dvq.Z = z_arr
                dvq.episode_count = int(dvq_state.get("episode_count", 0))
                print(
                    f"[Resume] DVQ: episode_count={dvq.episode_count}, "
                    f"Z_mean={float(np.mean(dvq.Z)):.3f}"
                )
            else:
                print("[Resume] DVQ: no saved state in checkpoint (starting fresh).")
        # Resume 时重新计算 max_total_steps：从当前位置再训练 total_episodes 轮
        max_total_steps = total_steps + config.ppo.total_episodes * config.system.T
        print(
            f"[Resume] total_steps={total_steps}, episodes={episode_count}, "
            f"update_count={update_count_resume}, meta_update_count={meta_update_count_resume}"
        )
        print(f"[Resume] Adjusted max_total_steps={max_total_steps} (+{config.ppo.total_episodes} episodes)")
        # 课程调度器：resume 后以实际 episode_count 重新应用 SNR 范围
        # 并重置 env，使接下来采样的 SNR 处于课程区间内
        if curriculum_scheduler is not None:
            info_cur = curriculum_scheduler.update_env(env, episode_count)
            print(
                f"[Curriculum] resume @ ep={episode_count} -> stage {info_cur['stage_idx']} "
                f"SNR [{info_cur['snr_min_db']}, {info_cur['snr_max_db']}] dB"
            )
            obs, _ = env.reset()
            _set_worker_bucket(worker_ac, env)
            prev_meta_obs, prev_split, prev_group_ids, prev_log_prob, prev_value = (
                _meta_decide(meta_ac, env, device)
            )
            meta_stats = _empty_meta_stats()
    else:
        update_count_resume = 0
        meta_update_count_resume = 0

    # ---------- Partial resume (Task#33): 仅加载 Worker 权重 ---------- #
    # 适用场景: 跨版本迁移 (如 v3 d_user=7 → v5 d_user=9)。
    # Worker (actor + critic) 维度未变，可直接加载；Meta 随机初始化；
    # 优化器 / Lagrangian 状态均从零开始。与 --resume 互斥。
    partial_ckpt_path = getattr(config, "partial_resume_ckpt", None)
    if partial_ckpt_path and not resume_ckpt_path:
        print(f"[PartialResume] Loading Worker weights from: {partial_ckpt_path}")
        ckpt_p = torch.load(partial_ckpt_path, map_location=device, weights_only=False)
        worker_sd = ckpt_p.get("worker_state_dict")
        if worker_sd is None:
            raise KeyError("partial_resume checkpoint missing 'worker_state_dict'")
        _load_state_dict_with_padding(worker_ac, worker_sd, "Worker-PartialResume")
        # 桶专家：旧 checkpoint 无 expert_heads 时从参考头同步
        if getattr(worker_ac, "use_bucket_experts", False):
            if not any(k.startswith("expert_heads.") for k in worker_sd.keys()):
                worker_ac.sync_expert_heads_from_reference()
                print("[PartialResume] Bucket experts synced from reference user_head")
        # 记录源 ckpt 的 meta 维度供调试
        old_meta_d = None
        old_meta_sd = ckpt_p.get("meta_state_dict")
        if old_meta_sd is not None and "embed.0.weight" in old_meta_sd:
            old_meta_d = int(old_meta_sd["embed.0.weight"].shape[1])
        print(
            f"[PartialResume] Worker loaded with dimension padding."
        )
        print(
            f"[PartialResume] Meta re-initialized (current d_user={config.hierarchical.meta_d_user}, "
            f"source ckpt d_user={old_meta_d})"
        )
        print("[PartialResume] Optimizers / Lagrangian state start from scratch.")

    print(f"[Train] Starting training: {config.exp_name}")
    print(f"[Train] obs_dim={obs_dim}, worker_action_dim={worker_action_dim} (3K={continuous_dim}+K={K}), device={device}")
    print(f"[Train] rollout_steps={config.ppo.rollout_steps}, batch_size={config.ppo.batch_size}")
    print(f"[Train] max_total_steps={max_total_steps}")
    print(
        f"[Hierarchical] K={K}, T_s={T_s}, "
        f"d_user={config.hierarchical.meta_d_user}, n_heads={config.hierarchical.meta_n_heads}, "
        f"hidden={config.hierarchical.meta_hidden_dim}, n_groups={config.hierarchical.n_groups}, "
        f"meta_update_threshold={config.hierarchical.meta_update_threshold}"
    )
    print(
        f"[Train] adaptive_V={config.lyapunov.adaptive_V}, V_min={config.lyapunov.V_min}, "
        f"V_max={config.lyapunov.V_max}, exponent={config.lyapunov.V_exponent}"
    )
    print(f"[Train] csi_mode={config.system.csi_mode}, epsilon_csi={config.system.epsilon_csi}")
    if getattr(worker_ac, "use_bucket_experts", False):
        print(
            f"[BucketExperts] enabled: n_experts={worker_ac.n_experts}, "
            f"SNR=[{worker_ac.bucket_snr_min:.1f},{worker_ac.bucket_snr_max:.1f}]dB, "
            f"lr_expert={getattr(config.ppo, 'lr_bucket_expert', config.ppo.lr_actor):.2e}"
        )

    update_count = update_count_resume
    meta_update_count = meta_update_count_resume
    best_mean_reward = -float("inf")
    best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")
    ckpt_meta_path = os.path.join(log_dir, "checkpoint_meta.jsonl")

    last_meta_info: dict = {}

    # -------------------- HEK-Lite: 启发式专家热启动 -------------------- #
    # 初始化层干预：规则专家演示 + BC 预训练把 actor 送到高功率/高交付盆地，
    # 再以衰减锚定损失微调（ppo.py update 内生效）。预热期临时挂起
    # load curriculum 防拽回；完成 hek_anchor_updates 次 PPO update 后恢复。
    # hek_enabled=False 时整段跳过，行为与现状完全一致。
    # resume / partial_resume 时跳过：初始化已由 checkpoint 给定，BC 会覆盖加载权重。
    hek_on = bool(
        constraint_cfg is not None
        and getattr(constraint_cfg, 'hek_enabled', False)
        and not resume_ckpt_path
        and not partial_ckpt_path
    )
    hek_lc_suspended = False
    hek_updates_done = 0
    hek_anchor_updates = (
        int(getattr(constraint_cfg, 'hek_anchor_updates', 40)) if constraint_cfg else 0
    )
    hek_orig_lc = None
    if hek_on:
        try:
            # 抗干扰：预热期临时关闭 load curriculum（记录原值，预热结束恢复）
            hek_orig_lc = bool(getattr(constraint_cfg, 'use_load_curriculum', False))
            constraint_cfg.use_load_curriculum = False
            if lc_on and hek_anchor_updates > 0:
                hek_lc_suspended = True
                env.load_ratio_min = lc_base_min
                env.load_ratio_max = lc_base_max

            # 注入 HEK 超参到 trainer（ConstraintConfig 与 PPOConfig 解耦，
            # 仿照 cost_critic_coef 注入模式）
            worker_trainer.hek_enabled = True
            worker_trainer.hek_bc_epochs = int(getattr(constraint_cfg, 'hek_bc_epochs', 150))
            worker_trainer.hek_bc_lr = float(getattr(constraint_cfg, 'hek_bc_lr', 3e-4))
            worker_trainer.hek_anchor_updates = hek_anchor_updates
            worker_trainer.hek_anchor_coef = float(getattr(constraint_cfg, 'hek_anchor_coef', 0.2))
            worker_trainer.hek_anchor_decay = float(getattr(constraint_cfg, 'hek_anchor_decay', 0.95))

            exp_obs, exp_act = _hek_collect_expert(
                env, config, int(getattr(constraint_cfg, 'hek_expert_episodes', 15))
            )
            bc_info = worker_trainer.bc_pretrain(exp_obs, exp_act)
            print(
                f"[HEK] expert demos={exp_obs.shape[0]} steps | "
                f"BC pretrain: loss={bc_info['bc_loss']:.4f} epochs={bc_info['bc_epochs']} | "
                f"anchor: coef={worker_trainer.hek_anchor_coef} "
                f"decay={worker_trainer.hek_anchor_decay} "
                f"updates={hek_anchor_updates} | "
                f"lc_suspended={hek_lc_suspended}"
            )
            # 演示收集打乱了 env 状态：重新 reset 并重做首段 Meta 决策，
            # 保证主循环从干净初始状态起步
            obs, _ = env.reset()
            _set_worker_bucket(worker_ac, env)
            prev_meta_obs, prev_split, prev_group_ids, prev_log_prob, prev_value = (
                _meta_decide(meta_ac, env, device)
            )
            meta_stats = _empty_meta_stats()
        except Exception as hek_err:
            print(f"[HEK] disabled due to error: {hek_err}")
            worker_trainer.hek_enabled = False
            worker_trainer.hek_obs = None
            worker_trainer.hek_act = None
            hek_lc_suspended = False
            if hek_orig_lc is not None:
                constraint_cfg.use_load_curriculum = hek_orig_lc
            if lc_on:
                _apply_lc_scale()
            obs, _ = env.reset()
            _set_worker_bucket(worker_ac, env)
            prev_meta_obs, prev_split, prev_group_ids, prev_log_prob, prev_value = (
                _meta_decide(meta_ac, env, device)
            )
            meta_stats = _empty_meta_stats()

    # -------------------- Solver-Expert Iteration: 单槽 DGLCR 精修专家热启动 --------------------
    # Expert-Iteration（ExIt/AlphaZero 模式，初始化层干预）：可微 surrogate 在
    # 真实 env 逐时隙在线求解（p_logits Adam 精修 + order 候选择优），演示经
    # bc_pretrain 联合训练连续头 + PL 排序头，再以衰减锚定微调。solver_expert_
    # enabled=False 时整段跳过，行为与现状完全一致。
    # resume / partial_resume 时跳过（初始化已由 checkpoint 给定，BC 会覆盖）。
    # 需 Worker 为 PL 模式（BC order 目标依赖 PL 头）；与 HEK 互斥（共用 BC 通道）。
    se_on = bool(
        constraint_cfg is not None
        and getattr(constraint_cfg, 'solver_expert_enabled', False)
        and not hek_on
        and not resume_ckpt_path
        and not partial_ckpt_path
    )
    if se_on and not bool(getattr(worker_ac, "use_plackett_luce", False)):
        print("[SolverExpert] disabled: worker is not in Plackett-Luce mode")
        se_on = False
    if se_on and bool(getattr(worker_ac, "use_bucket_experts", False)):
        # BC/anchor 通道无逐样本桶路由（evaluate_actions bucket_indices=None
        # 时全部样本落入单一 active head），桶专家组合下演示会污染单头
        print("[SolverExpert] disabled: incompatible with bucket experts "
              "(use --no-bucket-experts)")
        se_on = False
    if se_on:
        try:
            worker_trainer.hek_enabled = True
            worker_trainer.hek_bc_epochs = int(
                getattr(constraint_cfg, 'solver_expert_bc_epochs', 150))
            worker_trainer.hek_bc_lr = float(
                getattr(constraint_cfg, 'solver_expert_bc_lr', 3e-4))
            worker_trainer.hek_anchor_updates = int(
                getattr(constraint_cfg, 'solver_expert_anchor_updates', 40))
            worker_trainer.hek_anchor_coef = float(
                getattr(constraint_cfg, 'solver_expert_anchor_coef', 0.2))
            worker_trainer.hek_anchor_decay = float(
                getattr(constraint_cfg, 'solver_expert_anchor_decay', 0.95))

            # 评估口径收集：CuRLTraC stage-0（snr_min=5dB）覆盖不到 0-5dB 缺口区，
            # 临时切到课程终态/eval 配置（snr[-5,35]、load[0.75,1.3]）收集，
            # 结束后恢复当前课程档位，主训练不受影响
            se_saved_env = (
                env.snr_min_db, env.snr_max_db,
                env.load_ratio_min, env.load_ratio_max,
            )
            env.snr_min_db = float(getattr(
                config.ppo, 'curriculum_snr_min_end', config.system.snr_min_db))
            env.snr_max_db = float(config.system.snr_max_db)
            env.load_ratio_min = float(config.system.load_ratio_min)
            env.load_ratio_max = float(getattr(
                config.ppo, 'curriculum_load_ratio_max_end',
                config.system.load_ratio_max))

            t0 = time.time()
            exp_obs, exp_act, exp_orders = _solver_expert_collect(
                env, config, meta_ac, device,
                n_episodes=int(getattr(constraint_cfg, 'solver_expert_episodes', 6)),
                n_steps=int(getattr(constraint_cfg, 'solver_expert_steps', 25)),
                lr=float(getattr(constraint_cfg, 'solver_expert_lr', 0.1)),
            )
            t_collect = time.time() - t0
            (
                env.snr_min_db, env.snr_max_db,
                env.load_ratio_min, env.load_ratio_max,
            ) = se_saved_env
            bc_info = worker_trainer.bc_pretrain(exp_obs, exp_act, orders_np=exp_orders)
            # 纯 BC 检查点：专家迭代对照组（BC 后、PPO 微调前的策略快照）
            bc_ckpt_path = os.path.join(checkpoint_dir, "bc_init.pt")
            torch.save(
                {
                    "episode": 0,
                    "total_steps": 0,
                    "update_count": 0,
                    "meta_update_count": 0,
                    "worker_state_dict": worker_ac.state_dict(),
                    "meta_state_dict": meta_ac.state_dict(),
                    "worker_optimizer": worker_trainer.optimizer.state_dict(),
                    "meta_optimizer": meta_trainer.optimizer.state_dict(),
                    "config": vars(config.ppo),
                    "bc_stage": True,
                },
                bc_ckpt_path,
            )
            print(
                f"[SolverExpert] demos={exp_obs.shape[0]} steps "
                f"({t_collect:.0f}s collect) | "
                f"BC pretrain: loss={bc_info['bc_loss']:.4f} "
                f"epochs={bc_info['bc_epochs']} (joint cont+PL) | "
                f"anchor: coef={worker_trainer.hek_anchor_coef} "
                f"decay={worker_trainer.hek_anchor_decay} "
                f"updates={worker_trainer.hek_anchor_updates}"
            )

            # ---- M5-R1: ExIt 第二迭代 — on-policy relabel 收集 + 聚合二次 BC ----
            # 学生（首轮 BC 后）走轨迹、教师从学生状态打标 → 校正 BC 协变量偏移。
            # 收集口径与首轮一致（eval 配置 snr/load），完成后同样恢复课程档位。
            # bc_pretrain 会重置 _hek_updates 并用聚合演示替换锚定缓存。
            if bool(getattr(constraint_cfg, "solver_expert_relabel", False)):
                env.snr_min_db = float(getattr(
                    config.ppo, 'curriculum_snr_min_end', config.system.snr_min_db))
                env.snr_max_db = float(config.system.snr_max_db)
                env.load_ratio_min = float(config.system.load_ratio_min)
                env.load_ratio_max = float(getattr(
                    config.ppo, 'curriculum_load_ratio_max_end',
                    config.system.load_ratio_max))
                t0r = time.time()
                r_obs, r_act, r_orders = _solver_expert_collect(
                    env, config, meta_ac, device,
                    n_episodes=int(getattr(
                        constraint_cfg, 'solver_expert_episodes', 6)),
                    n_steps=int(getattr(constraint_cfg, 'solver_expert_steps', 25)),
                    lr=float(getattr(constraint_cfg, 'solver_expert_lr', 0.1)),
                    mode="relabel", student_ac=worker_ac,
                )
                t_relabel = time.time() - t0r
                (
                    env.snr_min_db, env.snr_max_db,
                    env.load_ratio_min, env.load_ratio_max,
                ) = se_saved_env
                if bool(getattr(constraint_cfg, "solver_expert_mix_original", True)):
                    all_obs = np.concatenate([exp_obs, r_obs])
                    all_act = np.concatenate([exp_act, r_act])
                    all_orders = np.concatenate([exp_orders, r_orders])
                else:
                    all_obs, all_act, all_orders = r_obs, r_act, r_orders
                bc_info2 = worker_trainer.bc_pretrain(
                    all_obs, all_act, orders_np=all_orders)
                relabel_ckpt_path = os.path.join(
                    checkpoint_dir, "bc_relabel.pt")
                torch.save(
                    {
                        "episode": 0,
                        "total_steps": 0,
                        "update_count": 0,
                        "meta_update_count": 0,
                        "worker_state_dict": worker_ac.state_dict(),
                        "meta_state_dict": meta_ac.state_dict(),
                        "worker_optimizer": worker_trainer.optimizer.state_dict(),
                        "meta_optimizer": meta_trainer.optimizer.state_dict(),
                        "config": vars(config.ppo),
                        "bc_stage": True,
                        "relabel_stage": True,
                    },
                    relabel_ckpt_path,
                )
                print(
                    f"[SolverExpert] relabel(ExIt-2): +{r_obs.shape[0]} "
                    f"on-policy demos ({t_relabel:.0f}s) | aggregate BC: "
                    f"n={all_obs.shape[0]} loss={bc_info2['bc_loss']:.4f} "
                    f"epochs={bc_info2['bc_epochs']} | saved {relabel_ckpt_path}"
                )

            # ---- M5.2-F6: KL-to-BC 信任域参考快照 ----
            # 参考中心 = 全部 BC 阶段之后、PPO 微调之前的策略。M5.1 终审证明 Arm A
            # 的收益来自被 lr 下限冻在 BC 盆地附近（隐式信任域）；F6 把该约束显式化，
            # 使 lr 得以走正常退火而不丢深过载区行为。load_state_dict 内部拷贝数据，
            # 无需调用方 deepcopy。
            if float(getattr(config.ppo, "kl_bc_coef", 0.0)) > 0.0:
                ok_ref = worker_trainer.set_bc_reference(worker_ac.state_dict())
                print(f"[KLBC] reference injected={ok_ref} "
                      f"coef={config.ppo.kl_bc_coef}")

            # 演示收集打乱了 env 状态：重新 reset 并重做首段 Meta 决策，
            # 保证主循环从干净初始状态起步
            obs, _ = env.reset()
            _set_worker_bucket(worker_ac, env)
            prev_meta_obs, prev_split, prev_group_ids, prev_log_prob, prev_value = (
                _meta_decide(meta_ac, env, device)
            )
            meta_stats = _empty_meta_stats()
        except Exception as se_err:
            print(f"[SolverExpert] disabled due to error: {se_err}")
            worker_trainer.hek_enabled = False
            worker_trainer.hek_obs = None
            worker_trainer.hek_act = None
            worker_trainer.hek_orders = None
            try:
                (
                    env.snr_min_db, env.snr_max_db,
                    env.load_ratio_min, env.load_ratio_max,
                ) = se_saved_env
            except NameError:
                pass  # 收集前失败（超参注入段），env 未被改写
            obs, _ = env.reset()
            _set_worker_bucket(worker_ac, env)
            prev_meta_obs, prev_split, prev_group_ids, prev_log_prob, prev_value = (
                _meta_decide(meta_ac, env, device)
            )
            meta_stats = _empty_meta_stats()

    # ---- M5.2-F6: KL-to-BC 信任域参考快照（非 se 路径兜底）----
    # se 路径在全部 BC 阶段后已注入；resume / 非 PL / 桶专家等 se_on=False 的
    # 流程此前从不注入（_bc_ac 恒 None → _klbc_term 静默 no-op），导致文档化的
    # "best_guard + --kl-bc-coef 0.x" 微调流程实际不生效。兜底：PPO 主循环开始
    # 前对当前（resume 加载后的）策略取一次快照，作为微调盆地中心。
    if float(getattr(config.ppo, "kl_bc_coef", 0.0)) > 0.0:
        if getattr(worker_trainer, "_bc_ac", None) is None:
            ok_ref = worker_trainer.set_bc_reference(worker_ac.state_dict())
            print(f"[KLBC] reference injected={ok_ref} "
                  f"coef={config.ppo.kl_bc_coef} (non-se fallback)")
        else:
            print(f"[KLBC] reference already injected (se path) coef={config.ppo.kl_bc_coef}")

    # ---- M5-F4: Meta 冻结（采样照常、梯度更新暂停至指定 episode）----
    meta_freeze_eps = int(getattr(config.hierarchical, "meta_freeze_episodes", 0))
    if meta_freeze_eps > 0:
        print(f"[MetaFreeze] Meta 更新冻结至 ep={meta_freeze_eps}"
              f"（Worker 在 BC 演示同款 (α,g) 条件分布上稳定）")

    # ---- M5-F5: 评估守卫初始化（eval_guard_enable=False 时零介入）----
    guard_on = bool(getattr(config.ppo, "eval_guard_enable", False))
    guard_env = None
    guard_last_good = None
    guard_halvings = 0
    guard_last_ep = 0
    guard_best_deliv = -1.0
    guard_stall_count = 0
    early_stopped = False
    guard_lr_groups: list = []
    if guard_on:
        guard_env = RSMAEnergyEnv(config)
        guard_env.snr_domain_randomization = False
        # actor+log_std 所在 param_group 索引（桶专家模式下组结构不同，按参数集合定位）
        _actor_ids = ({id(p) for p in worker_ac.actor_params()}
                      | {id(p) for p in worker_ac.log_std_params()})
        guard_lr_groups = [
            i for i, g in enumerate(worker_trainer.optimizer.param_groups)
            if any(id(p) in _actor_ids for p in g["params"])
        ]
        guard_last_good = (
            {k: v.detach().cpu().clone()
             for k, v in worker_ac.state_dict().items()},
            {k: v.detach().cpu().clone()
             for k, v in meta_ac.state_dict().items()},
        )
        guard_pts_cfg = tuple(getattr(config.ppo, "eval_guard_snr_points", ()) or ())
        guard_points = (tuple(float(s) for s in guard_pts_cfg)
                        if guard_pts_cfg else _GUARD_POINTS)
        g_ref_avg, g_q_ref = _guard_ref_avg(guard_points)
        print(f"[Guard] enabled: every={config.ppo.eval_guard_every}ep "
              f"points={guard_points} n_eps={config.ppo.eval_guard_episodes} "
              f"v2={config.ppo.eval_guard_v2} "
              f"ref={g_ref_avg:.3f}−{config.ppo.eval_guard_margin} "
              f"q_ref={g_q_ref:.2f}×{config.ppo.eval_guard_queue_ratio} "
              f"lr_groups={guard_lr_groups}")

    while total_steps < max_total_steps:
        worker_buffer.clear()
        # RCPO: 稠密 per-slot cost 与 cost value 暂存（普通 python 数组，不改 buffer.py）
        cost_buf: list = []
        cost_val_buf: list = []
        lambda_buf: list = []
        # C3PO: 7维逐SNR点 cost 与 cost value 暂存
        c3po_cost_buf: list = []
        c3po_cost_val_buf: list = []
        # PLPI: per-step running delivery ratio（用于 PPO loss 级功率激励）
        pi_delivery_buf: list = []
        # DSL 诊断（本 rollout 累计）：投影触发步数与平均地板抬升幅度
        dsl_trigger_count = 0
        dsl_lift_sum = 0.0

        for _ in range(config.ppo.rollout_steps):
            # ---- Worker 决策 (PL模式: 连续动作 + 解码顺序) ----
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_continuous, decoding_order, log_prob, value, cont_log_prob = worker_ac.act(obs_tensor)

            action_continuous_np = action_continuous.squeeze(0).cpu().numpy()
            decoding_order_np = decoding_order.squeeze(0).cpu().numpy().astype(np.int64)
            log_prob_val = float(log_prob.item())
            cont_log_prob_val = float(cont_log_prob.item())
            value_val = float(value.item())

            # ---- Delivery-Barrier 安全层（动作层干预）----
            # running delivery 不可行时，把 p_total/d_ratio 向上投影到状态相关地板，
            # 并用 evaluate_actions 重算 log_prob 保证 PPO on-policy 一致；
            # 标志关闭或轨迹已可行时完全走原路径（零干预）。
            if dsl_enabled:
                (action_continuous_np, log_prob_val, cont_log_prob_val,
                 dsl_triggered, dsl_lift) = _apply_delivery_safety_layer(
                    worker_ac, obs_tensor, action_continuous_np, decoding_order,
                    log_prob_val, cont_log_prob_val, env, constraint_cfg, device,
                )
                if dsl_triggered:
                    dsl_trigger_count += 1
                    dsl_lift_sum += dsl_lift

            # 合并为4K维action传给env（env自动检测4K维并提取解码顺序）
            full_action = np.concatenate([
                action_continuous_np,
                decoding_order_np.astype(np.float64)
            ])
            next_obs, reward, terminated, truncated, info = env.step(full_action)
            done = bool(terminated or truncated)

            # ---- ASOPA on-policy 修复：重算被 solver 替换动作的 log_prob ----
            if hasattr(env, '_v2_modified_action') and env._v2_modified_action is not None:
                v2_cont, v2_order = env._v2_modified_action
                env._v2_modified_action = None  # 消费后清除
                v2_cont_t = torch.from_numpy(v2_cont).float().unsqueeze(0).to(device)
                v2_order_t = torch.from_numpy(v2_order.astype(np.int64)).unsqueeze(0).to(device)
                with torch.no_grad():
                    new_log_prob, _, new_value, new_cont_log_prob = worker_ac.evaluate_actions(
                        obs_tensor, v2_cont_t, v2_order_t
                    )
                log_prob_val = float(new_log_prob.item())
                cont_log_prob_val = float(new_cont_log_prob.item())
                value_val = float(new_value.item())
                action_continuous_np = v2_cont.astype(np.float32)
                decoding_order_np = v2_order.astype(np.int64)

            # 外层 Dinkelbach：累计本 episode 的 Σrate / Σpower
            dinkelbach_ep_rate_sum += float(info.get("R_sum", 0.0))
            dinkelbach_ep_power_sum += float(
                info.get("P_consume", info.get("total_power", 0.0))
            )

            # State-Augmented 观测：注入当前 SNR 桶的 Lagrangian λ
            if delivery_constraint is not None:
                _snr = float(info.get("snr_db", 0.0))
                _bucket = delivery_constraint._bucket(_snr)
                env.set_lambda_bucket(float(delivery_constraint.lambda_val[_bucket]))
            else:
                env.set_lambda_bucket(0.0)

            # Task#17 + v14s: PPO-Lagrangian 约束惩罚（从 reward 中减去）
            #   penalty = λ * max(0, threshold - delivery_running)
            #   warmup 期间 penalty=0；不被 reward_scale 削弱
            #   v14s: 除以 reward_scale 使 penalty 在未缩放空间竞争，
            #         否则 reward_scale=0.01 将 penalty 稀释到可忽略
            crpo_feas_active = False
            if crpo_on and delivery_constraint is not None:
                _b = delivery_constraint._bucket(float(info.get('snr_db', 0.0)))
                crpo_feas_active = crpo_mode.get(_b, True)   # 未见过的桶默认可行性模式(冷启动先学吞吐)
            if crpo_feas_active:
                # CRPO-S 可行性模式: 整段episode用有界排水盈余奖励替代env reward
                _arr = np.asarray(info.get('arrivals', np.zeros(config.system.K)), dtype=np.float64)
                _srv = np.asarray(info.get('served', np.zeros(config.system.K)), dtype=np.float64)
                _denom = max(float(_arr.sum()), 1.0)
                reward = float(constraint_cfg.crpo_feas_scale) * float(np.clip((float(_srv.sum()) - float(_arr.sum())) / _denom, -1.0, 1.0))
                info['crpo_mode'] = 1.0
                info['lagrangian_penalty'] = 0.0
            else:
                info['crpo_mode'] = 0.0
            constraint_penalty = 0.0
            if not crpo_feas_active and delivery_constraint is not None:
                if getattr(constraint_cfg, 'use_step_penalty', False):
                    # SLCP-D: per-step 即时交付率(served/arrivals)，arrivals=0 视为满足(=1.0)，clip[0,1]
                    _arr = np.asarray(info.get('arrivals', np.zeros(config.system.K)), dtype=np.float64)
                    _srv = np.asarray(info.get('served', np.zeros(config.system.K)), dtype=np.float64)
                    _arr_sum = float(_arr.sum())
                    if _arr_sum > 1e-8:
                        inst_delivery = min(1.0, max(0.0, float(_srv.sum()) / _arr_sum))
                    else:
                        inst_delivery = 1.0
                    # EMA 平滑抗 Poisson 噪声
                    _beta = float(getattr(constraint_cfg, 'slcp_ema_beta', 0.85))
                    if step_delivery_ema is None:
                        step_delivery_ema = inst_delivery
                    else:
                        step_delivery_ema = _beta * step_delivery_ema + (1.0 - _beta) * inst_delivery
                    delivery_for_penalty = step_delivery_ema
                else:
                    delivery_for_penalty = float(info.get('delivery_ratio', 0.0))
                constraint_penalty = delivery_constraint.compute_penalty(
                    delivery_for_penalty, snr_db=float(info.get("snr_db", 0.0))
                )
                if not use_rcpo:
                    # v15: 显式 delivery 约束 —— penalty 直接减去（与 reward 同量级）
                    # RCPO 模式下跳过：reward 保持环境原值（纯 EE 平稳），
                    # 约束改由 PPO update 中 A_eff = A_r - λ_scaled·A_c 施加。
                    reward = float(reward) - constraint_penalty
                info["lagrangian_penalty"] = constraint_penalty
                info["lagrangian_lambda"] = delivery_constraint.get_lambda()

            # RCPO: 稠密 per-slot cost（归一化交付赤字），提供逐步因果信用
            if use_rcpo:
                arrivals_step = np.asarray(
                    info.get("arrivals", np.zeros(config.system.K)), dtype=np.float64
                )
                served_step = np.asarray(
                    info.get("served", np.zeros(config.system.K)), dtype=np.float64
                )
                arr_sum = float(arrivals_step.sum())
                if delivery_constraint.episode_count < delivery_constraint.warmup_episodes:
                    cost_t = 0.0  # warmup 期间不施加约束
                else:
                    deficit = float(delivery_constraint.threshold) * arr_sum - float(served_step.sum())
                    cost_t = max(0.0, deficit) / max(arr_sum, 1e-8)  # 归一化到 [0,1]
                cost_buf.append(cost_t)
                # cost value 用步前 obs（与 worker_buffer 中 value_val 对齐）
                cost_val_buf.append(float(worker_trainer.predict_cost_value(obs)))
                lambda_buf.append(float(
                    delivery_constraint.lambda_val[
                        delivery_constraint._bucket(float(info.get("snr_db", 0.0)))
                    ]
                ))
                info["rcpo_cost"] = cost_t

            # PLPI: 记录每步 running delivery，供 PPO update 计算功率激励
            pi_delivery_buf.append(float(info.get("delivery_ratio", 0.0)))

            # DVQ: per-user 虚拟队列惩罚（逐步逐用户 credit assignment）
            # CRPO-S 可行性模式下跳过：保持排水盈余目标纯净，不叠加约束惩罚
            if dvq is not None and not crpo_feas_active:
                arrivals_step = np.asarray(info.get("arrivals", np.zeros(config.system.K)))
                served_step = np.asarray(info.get("served", np.zeros(config.system.K)))
                dvq_pen, dvq_info = dvq.update_and_penalty(arrivals_step, served_step)
                reward = float(reward) - dvq_pen
                info["dvq_penalty"] = dvq_pen
                info.update(dvq_info)

            # Mean-Field MAPPO: 每用户 delivery 违约项
            per_user_deficit = None
            if getattr(config.ppo, 'use_per_user_advantage', False):
                _arr = np.asarray(info.get("arrivals", np.zeros(config.system.K)), dtype=np.float64)
                _srv = np.asarray(info.get("served", np.zeros(config.system.K)), dtype=np.float64)
                _denom = np.maximum(_arr, 1e-8)
                per_user_deficit = np.maximum(delivery_threshold * _arr - _srv, 0.0) / _denom
                per_user_deficit = per_user_deficit.astype(np.float32)

            worker_buffer.add(
                obs, action_continuous_np, reward, value_val, log_prob_val,
                done, decoding_order=decoding_order_np,
                cont_log_prob=cont_log_prob_val,
                snr_db=float(info.get('snr_db', 0.0)),
                per_user_deficit=per_user_deficit,
                entropy_scale=float(hm_entropy_scale),
            )

            # ---- Meta 段统计累积 ----
            meta_stats["ee_sum"] += float(info.get("ee_bonus", 0.0))
            meta_stats["overflow_sum"] += float(info.get("overflow_penalty", 0.0))
            meta_stats["steps"] += 1
            # Task#36: 累积平均 α 与当前段 SNR（同一 episode 内 SNR 不变）
            meta_stats["alpha_sum"] += float(info.get("mean_alpha", 0.0))
            meta_stats["alpha_count"] += 1
            meta_stats["snr_db"] = float(info.get("snr_db", meta_stats["snr_db"]))

            episode_reward += float(reward)
            episode_length += 1
            total_steps += 1
            obs = next_obs

            # ---- 处理 Meta 段切换 / Episode 结束 ----
            if done:
                # Task#17: episode 结束，用最终累计 delivery 更新对偶乘子 λ
                episode_delivery = float(info.get("delivery_ratio", 0.0))
                _ep_snr_db = float(info.get("snr_db", 0.0))
                if delivery_constraint is not None:
                    delivery_constraint.update(
                        episode_delivery, snr_db=_ep_snr_db
                    )
                # CuRLTraC-ALP: 更新对应点的 delivery EMA 与 streak
                if hm_on:
                    _hm_update(hm_state, _ep_snr_db, episode_delivery, hm_decay)
                # CRPO-S: episode级模式切换(迟滞带防振荡)。
                # 注意: update 照常调用(EE模式桶仍需λ)，仅reward写入路径按模式分流。
                if crpo_on and delivery_constraint is not None:
                    _b = delivery_constraint._bucket(float(info.get("snr_db", 0.0)))
                    _thr = float(delivery_constraint.threshold)
                    # 确保桶有默认值（与 L863 的 .get(_b, True) 一致）
                    if _b not in crpo_mode:
                        crpo_mode[_b] = True  # 冷启动默认可行性模式
                    if episode_delivery < _thr:
                        crpo_mode[_b] = True
                    elif episode_delivery > _thr + float(getattr(constraint_cfg, 'crpo_hysteresis', 0.03)):
                        crpo_mode[_b] = False
                    info['crpo_feas_mode_episode'] = 1.0 if crpo_mode[_b] else 0.0
                if dvq is not None:
                    dvq.end_episode()

                # 外层 Dinkelbach eta 更新：每 N episode 按实测 EE 更新一次
                ep_ee = (
                    dinkelbach_ep_rate_sum / max(dinkelbach_ep_power_sum, 1e-9)
                    if dinkelbach_ep_power_sum > 1e-9
                    else 0.0
                )
                dinkelbach_episode_ee_buffer.append(float(ep_ee))
                dinkelbach_ep_rate_sum = 0.0
                dinkelbach_ep_power_sum = 0.0
                dinkelbach_interval = int(getattr(config.lyapunov, "dinkelbach_update_interval", 10))
                dinkelbach_alpha = float(getattr(config.lyapunov, "dinkelbach_anchor_alpha", 0.2))
                if len(dinkelbach_episode_ee_buffer) >= dinkelbach_interval:
                    measured_ee = float(np.mean(dinkelbach_episode_ee_buffer))
                    q_min = float(config.lyapunov.dinkelbach_q_min)
                    q_max = float(config.lyapunov.dinkelbach_q_max)
                    if not dinkelbach_first_anchor_done:
                        env.dinkelbach_eta = float(np.clip(measured_ee, q_min, q_max))
                        dinkelbach_first_anchor_done = True
                    else:
                        env.dinkelbach_eta = float(np.clip(
                            (1.0 - dinkelbach_alpha) * env.dinkelbach_eta
                            + dinkelbach_alpha * measured_ee,
                            q_min, q_max,
                        ))
                    env.lyapunov.dinkelbach_q = env.dinkelbach_eta
                    dinkelbach_episode_ee_buffer = []

                # Episode 结束: 关闭最后一段 Meta transition
                meta_stats['group_ids'] = prev_group_ids
                meta_stats['n_groups'] = config.hierarchical.n_groups
                _inject_alpha_guidance(meta_stats, config, episode_count)
                meta_reward = compute_meta_reward(meta_stats) * config.hierarchical.meta_reward_scale
                meta_buffer.store(
                    prev_meta_obs, prev_split,
                    prev_log_prob, meta_reward, prev_value,
                    done=True, group_action=prev_group_ids
                )
                meta_segment_rewards.append(meta_reward)
                meta_stats = _empty_meta_stats()
                step_delivery_ema = None  # SLCP-D: per-step 即时交付率 EMA，每 episode 重置

                episode_rewards.append(episode_reward)
                episode_count += 1
                episode_reward = 0.0
                episode_length = 0

                # SPLC: 窗口化 running delivery 自定步升档
                if lc_on:
                    cur_deliv = float(env.traffic.get_delivery_ratio())
                    lc_deliv_window.append(cur_deliv)
                    if len(lc_deliv_window) > constraint_cfg.lc_window:
                        lc_deliv_window.pop(0)
                    lc_level_ep += 1
                    if (lc_scale < lc_final
                            and lc_level_ep >= constraint_cfg.lc_min_episodes_per_level
                            and len(lc_deliv_window) >= constraint_cfg.lc_window):
                        win_mean = sum(lc_deliv_window) / len(lc_deliv_window)
                        if win_mean >= constraint_cfg.lc_delivery_target:
                            lc_pass_count += 1
                        else:
                            lc_pass_count = 0
                        if lc_pass_count >= constraint_cfg.lc_patience:
                            lc_scale = min(lc_final, lc_scale + constraint_cfg.lc_step)
                            lc_pass_count = 0
                            lc_level_ep = 0
                            lc_deliv_window = []
                            print(f'[SPLC] ep={episode_count} 升档 load_scale->{lc_scale:.3f}')

                # CuRLTraC: 性能触发式难度退火（MVP）
                if curl_on:
                    curl_window.append(episode_delivery)
                    curl_wsize = int(getattr(config.ppo, 'curriculum_window', 50))
                    if len(curl_window) > curl_wsize:
                        curl_window.pop(0)
                    curl_stage_eps += 1
                    beta = float(getattr(config.ppo, 'curriculum_beta', 0.5))
                    fallback = float(getattr(config.ppo, 'curriculum_fallback_beta', 0.3))
                    patience = int(getattr(config.ppo, 'curriculum_patience', 20))
                    win_mean = sum(curl_window) / len(curl_window) if curl_window else 0.0
                    changed = False
                    if curl_stage < curl_n - 1 and len(curl_window) >= curl_wsize and curl_stage_eps >= curl_min_eps and win_mean >= beta:
                        curl_stage += 1
                        changed = True
                    elif 0 < curl_stage < curl_n - 1 and win_mean < fallback:
                        curl_low_streak += 1
                        if curl_low_streak >= patience:
                            curl_stage -= 1
                            changed = True
                    else:
                        curl_low_streak = 0
                    if changed:
                        curl_stage_eps = 0
                        curl_low_streak = 0
                        curl_window = []
                        _curl_apply(curl_stage)
                        print(f'[CuRLTraC] ep={episode_count} stage={curl_stage} snr_min={env.snr_min_db:.1f}dB load_max={env.load_ratio_max:.2f}')

                # 课程调度器：下一个 episode 前更新 env 的 SNR 范围
                if curriculum_scheduler is not None:
                    curriculum_scheduler.update_env(env, episode_count)

                # SPLC: 叠加最新 lc_scale（在 curriculum_scheduler 基线之上）
                if lc_on:
                    _apply_lc_scale()

                # CuRLTraC-ALP: 按点缺口+streak重采样下一 episode SNR，并刷新条件熵系数
                reset_options = None
                if hm_on:
                    sampled_snr, _, is_failing, hm_entropy_scale = _hm_sample_snr(
                        hm_state, hm_w_hard, hm_floor, hm_streak_mult, hm_streak_cap, hm_rng,
                        fail_entropy_scale=hm_fail_entropy_scale,
                        performance_linked=getattr(config.ppo, 'entropy_performance_linked', False),
                        link_coef=getattr(config.ppo, 'entropy_link_coef', 1.0),
                    )
                    reset_options = {'snr_db': sampled_snr}
                obs, _ = env.reset(options=reset_options)
                _set_worker_bucket(worker_ac, env)

                # 新 episode 起点立即做一次 Meta 决策
                prev_meta_obs, prev_split, prev_group_ids, prev_log_prob, prev_value = (
                    _meta_decide(meta_ac, env, device)
                )
            elif env.step_in_episode % T_s == 0:
                # T_s 段边界（非 episode 结束）：闭合上一段 + 新决策
                meta_stats['group_ids'] = prev_group_ids
                meta_stats['n_groups'] = config.hierarchical.n_groups
                _inject_alpha_guidance(meta_stats, config, episode_count)
                meta_reward = compute_meta_reward(meta_stats) * config.hierarchical.meta_reward_scale
                meta_buffer.store(
                    prev_meta_obs, prev_split,
                    prev_log_prob, meta_reward, prev_value,
                    done=False, group_action=prev_group_ids
                )
                meta_segment_rewards.append(meta_reward)
                meta_stats = _empty_meta_stats()

                prev_meta_obs, prev_split, prev_group_ids, prev_log_prob, prev_value = (
                    _meta_decide(meta_ac, env, device)
                )

            if total_steps >= max_total_steps:
                break

        # ---- Bootstrap 最后一步 value ----
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, _, last_value, _ = worker_ac.act(obs_tensor, deterministic=True)
            last_value = float(last_value.item())

        worker_buffer.compute_returns_and_advantages(
            last_value,
            per_user_adv_weight=float(getattr(config.ppo, 'per_user_adv_weight', 1.0))
            if getattr(config.ppo, 'use_per_user_advantage', False) else 0.0,
        )

        # RCPO: bootstrap cost value + 组装 cost_data（λ_bucket 为 episode 内常量，
        # rollout 跨 episode 时取均值作为标量传入 update）
        cost_data = None
        lambda_scaled = 0.0
        if use_rcpo:
            cost_data = {
                "costs": np.asarray(cost_buf, dtype=np.float32),
                "cost_values": np.asarray(cost_val_buf, dtype=np.float32),
                "last_cost_value": float(worker_trainer.predict_cost_value(obs)),
            }
            lambda_scaled = rcpo_lambda_scale * (
                float(np.mean(lambda_buf)) if lambda_buf else 0.0
            )

        # 进度感知 LR 衰减
        if hasattr(worker_trainer, "set_progress"):
            worker_trainer.set_progress(total_steps / max(max_total_steps, 1))

        # ---- Meta group head 温度退火 ----
        # progress = current_episode / total_episodes，仅影响分组决策。
        if hasattr(meta_ac, "anneal_temperature"):
            meta_progress = (
                episode_count / max(config.ppo.total_episodes, 1)
            )
            meta_ac.anneal_temperature(meta_progress)

        # PLPI: power incentive coefficient（每 episode 衰减）
        pi_coef = 0.0
        pi_threshold = float(getattr(constraint_cfg, 'delivery_threshold', 0.98)) if constraint_cfg else 0.98
        if constraint_cfg is not None and getattr(constraint_cfg, 'use_power_incentive', False):
            pi_coef = float(getattr(constraint_cfg, 'pi_power_coef', 1.0))
            pi_decay = float(getattr(constraint_cfg, 'pi_power_decay', 0.999))
            pi_coef = pi_coef * (pi_decay ** episode_count)

        # ---- SFL 可学习性增强采样权重 ----
        sample_weights = None
        bucket_deficits = None
        use_sfl = bool(
            constraint_cfg is not None
            and getattr(constraint_cfg, 'use_sfl_sampling', False)
        )
        if use_sfl and delivery_constraint is not None:
            deficits = delivery_constraint.get_bucket_deficits()
            learnability = delivery_constraint.get_bucket_learnability_scores()
            a = float(getattr(constraint_cfg, 'sfl_deficit_exponent', 1.0))
            b = float(getattr(constraint_cfg, 'sfl_learnability_exponent', 1.0))
            uniform_mix = float(getattr(constraint_cfg, 'sfl_uniform_mix', 0.1))
            n_buckets = int(delivery_constraint.n_snr_buckets)

            raw = (
                np.power(np.maximum(deficits, 0.0) + 1e-8, a)
                * np.power(np.maximum(learnability, 0.0) + 1e-8, b)
            )
            raw_sum = float(raw.sum())
            if raw_sum > 1e-12:
                bucket_probs = raw / raw_sum
            else:
                bucket_probs = np.ones(n_buckets, dtype=np.float64) / max(n_buckets, 1)
            bucket_weights = (1.0 - uniform_mix) * bucket_probs + uniform_mix / max(n_buckets, 1)
            bucket_weights = bucket_weights / max(float(bucket_weights.sum()), 1e-12)

            # 映射到每样本权重
            snr_dbs = worker_buffer.snr_dbs[:worker_buffer.pos]
            span = max(delivery_constraint.snr_max - delivery_constraint.snr_min, 1e-8)
            bucket_idx = np.clip(
                ((snr_dbs - delivery_constraint.snr_min) / span * n_buckets).astype(int),
                0, n_buckets - 1,
            )
            sample_weights = bucket_weights[bucket_idx].astype(np.float64)
            bucket_deficits = deficits

        # ---- Worker PPO 更新 ----
        entropy_scales = (
            worker_buffer.entropy_scales[:worker_buffer.pos].copy()
            if hm_on else None
        )
        update_info = worker_trainer.update(
            worker_buffer, cost_data=cost_data, lambda_scaled=lambda_scaled,
            pi_delivery=pi_delivery_buf, pi_coef=pi_coef, pi_threshold=pi_threshold,
            sample_weights=sample_weights, bucket_deficits=bucket_deficits,
            entropy_scale=entropy_scales,
        )
        if use_rcpo:
            update_info["rcpo_cost_mean"] = float(np.mean(cost_buf)) if cost_buf else 0.0
            update_info["rcpo_lambda_scaled"] = float(lambda_scaled)
        update_count += 1

        # ---- Meta PPO 更新 (积累足够后) ----
        if disable_meta:
            # Worker-Only 变体：完全跳过 Meta 更新，并清空累计 buffer
            if meta_buffer.size > 0:
                meta_buffer.reset()
            meta_train_info = None
        elif meta_buffer.size >= config.hierarchical.meta_update_threshold:
            if episode_count < meta_freeze_eps:
                # M5-F4: 冻结期——累积段丢弃，不做 Meta 梯度更新
                meta_buffer.reset()
                meta_train_info = None
            else:
                meta_train_info = meta_trainer.update(meta_buffer)
                meta_update_count += 1
                last_meta_info = meta_train_info
        else:
            meta_train_info = None

        # ---- 日志输出 ----
        if update_count % config.ppo.log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_steps / max(elapsed, 1e-8)

            recent_rewards = episode_rewards[-100:] if episode_rewards else [0.0]
            mean_reward = float(np.mean(recent_rewards))
            recent_meta = (meta_segment_rewards[-100:]
                           if meta_segment_rewards else [0.0])
            mean_meta_reward = float(np.mean(recent_meta))

            meta_loss_str = "N/A"
            if last_meta_info:
                meta_loss_str = f"{last_meta_info.get('meta_loss', 0.0):.4f}"

            # Task#17: 拼接 Lagrangian 状态信息
            lag_str = ""
            if delivery_constraint is not None:
                lag_stats = delivery_constraint.get_stats()
                warmup_tag = "W" if lag_stats["in_warmup"] else "A"
                lag_str = (
                    f" | Lambda={lag_stats['lambda']:.2f}[{warmup_tag}] | "
                    f"AvgDeliv={lag_stats['avg_delivery_100ep']:.3f}"
                )

            # RCPO 状态信息（cost 均值 / λ_scaled / cost value loss）
            rcpo_str = ""
            if use_rcpo:
                rcpo_str = (
                    f" | RCPO: cost={update_info.get('rcpo_cost_mean', 0.0):.3f} "
                    f"lam_s={update_info.get('rcpo_lambda_scaled', 0.0):.3f} "
                    f"cvL={update_info.get('cost_value_loss', 0.0):.4f}"
                )

            # DSL 状态信息（本 rollout 触发率与平均地板抬升幅度）
            dsl_str = ""
            if dsl_enabled:
                dsl_str = (
                    f" | DSL: trig={dsl_trigger_count / max(config.ppo.rollout_steps, 1):.2f} "
                    f"lift={dsl_lift_sum / max(dsl_trigger_count, 1):.3f}"
                )

            # PLPI 状态信息（本 rollout 平均 delivery 与当前功率激励系数）
            plpi_str = ""
            if pi_coef > 0:
                mean_deliv = float(np.mean(pi_delivery_buf)) if pi_delivery_buf else 0.0
                plpi_str = f" | PLPI: dr={mean_deliv:.3f} coef={pi_coef:.3f}"

            # M5.2-F6: KL-to-BC 信任域量级（未启用时不打印）
            klbc_str = ""
            if float(update_info.get("kl_bc_coef", 0.0)) > 0.0:
                klbc_str = (
                    f" | KLBC={update_info.get('kl_bc_loss', 0.0):.4f} "
                    f"c={update_info['kl_bc_coef']:.2f}"
                )

            # M5.4 诊断：band 熵 + log_std（分布无关的真坍缩判据）+ 冻结位。
            # M5.3 教训：重加权采样后批熵跨臂不可比（混入构成偏移），必须拆分。
            diag_str = (
                f" | ENT=[{update_info.get('entropy_lo', 0.0):.1f},"
                f"{update_info.get('entropy_mid', 0.0):.1f},"
                f"{update_info.get('entropy_hi', 0.0):.1f}]"
                f" LS={update_info.get('log_std_mean', 0.0):.2f}"
                + (" FZ" if update_info.get("lr_frozen", 0.0) > 0 else "")
            )

            # SNR 课程状态信息
            cur_str = ""
            if curriculum_scheduler is not None:
                stage_idx = curriculum_scheduler.get_stage_index(episode_count)
                cur_str = (
                    f" | SNR=[{env.snr_min_db:.1f},{env.snr_max_db:.1f}]dB "
                    f"S{stage_idx}"
                )

            # CuRLTraC 状态信息
            curl_str = ""
            if curl_on:
                curl_str = (
                    f" | CuRLTraC=S{curl_stage}/{curl_n - 1} "
                    f"SNR=[{env.snr_min_db:.1f},{env.snr_max_db:.1f}]dB "
                    f"load_max={env.load_ratio_max:.2f}"
                )

            # 硬例挖掘指纹：逐点采样计数、条件熵系数、streak
            hm_str = ""
            if hm_on and hm_state is not None:
                mean_ent_scale = float(np.mean(worker_buffer.entropy_scales[:worker_buffer.pos])) if worker_buffer.pos > 0 else 1.0
                counts = hm_state['sample_count'].tolist()
                streaks = hm_state['streak'].tolist()
                hm_str = (
                    f" | HM: counts={counts} ent={mean_ent_scale:.2f} "
                    f"streak={streaks}"
                )

            # 桶专家路由计数指纹与 per-head 梯度范数
            bucket_routes_str = ""
            bucket_grad_str = ""
            if getattr(worker_ac, "use_bucket_experts", False):
                counts = getattr(worker_ac, "_route_counts", [0] * int(worker_ac.n_experts))
                bucket_routes_str = f" | Buckets={counts} active={worker_ac.active_bucket}"
                grad_parts = []
                for i in range(int(worker_ac.n_experts)):
                    key = f"expert_{i}_grad_norm"
                    if key in update_info:
                        grad_parts.append(f"{update_info[key]:.4f}")
                if grad_parts:
                    bucket_grad_str = f" | EGrad=[{', '.join(grad_parts)}]"

            print(
                f"[Step {total_steps:>8d}] "
                f"Updates={update_count:>4d} | "
                f"Episodes={episode_count:>5d} | "
                f"MeanReward={mean_reward:.3f} | "
                f"PolicyLoss={update_info['policy_loss']:.4f} | "
                f"ValueLoss={update_info['value_loss']:.4f} | "
                f"Entropy={update_info['entropy']:.4f} | "
                f"KL={update_info['approx_kl']:.5f} | "
                f"LR={update_info.get('lr_actor', 0.0):.2e} | "
                f"KLstop={int(update_info.get('early_stopped', 0))} | "
                f"MetaR={mean_meta_reward:+.3f} | "
                f"MetaUpd={meta_update_count} | "
                f"MetaLoss={meta_loss_str} | "
                f"FPS={fps:.0f}"
                f"{lag_str}"
                f"{rcpo_str}"
                f"{dsl_str}"
                f"{plpi_str}"
                f"{klbc_str}"
                f"{diag_str}"
                f"{cur_str}"
                f"{curl_str}"
                f"{hm_str}"
                f"{bucket_routes_str}"
                f"{bucket_grad_str}"
            )

        # ---- Checkpoint 保存 ----
        if update_count % config.ppo.save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{update_count}.pt")
            ckpt_data = {
                "episode": episode_count,
                "total_steps": total_steps,
                "update_count": update_count,
                "meta_update_count": meta_update_count,
                "worker_state_dict": worker_ac.state_dict(),
                "meta_state_dict": meta_ac.state_dict(),
                "worker_optimizer": worker_trainer.optimizer.state_dict(),
                "meta_optimizer": meta_trainer.optimizer.state_dict(),
                "config": vars(config.ppo),
            }
            if delivery_constraint is not None:
                ckpt_data["lagrangian_state"] = {
                    "lambda": [float(x) for x in delivery_constraint.lambda_val],
                    "integral": [float(x) for x in delivery_constraint.integral],
                    "episode_count": int(delivery_constraint.episode_count),
                    "delivery_history": [
                        list(bucket[-1000:]) for bucket in delivery_constraint.delivery_history
                    ],
                }
            if dvq is not None:
                ckpt_data["dvq_state"] = {
                    "Z": dvq.Z.tolist(),
                    "episode_count": int(dvq.episode_count),
                }
            torch.save(ckpt_data, ckpt_path)
            print(f"[Checkpoint] Saved: {ckpt_path}")

            # best.pt: 基于最近 100 episode 的 MeanReward
            recent_rewards_eval = episode_rewards[-100:] if episode_rewards else []
            if recent_rewards_eval:
                cur_mean = float(np.mean(recent_rewards_eval))
                try:
                    with open(ckpt_meta_path, "a") as mf:
                        mf.write(json.dumps({
                            "update": update_count,
                            "total_steps": total_steps,
                            "mean_reward": cur_mean,
                            "approx_kl": float(update_info.get("approx_kl", 0.0)),
                            "lr_actor": float(update_info.get("lr_actor", 0.0)),
                            "early_stopped": int(update_info.get("early_stopped", 0)),
                            "meta_update_count": meta_update_count,
                            "ckpt": os.path.basename(ckpt_path),
                        }) + "\n")
                except Exception:
                    pass
                if cur_mean > best_mean_reward:
                    best_mean_reward = cur_mean
                    best_ckpt = dict(ckpt_data)
                    best_ckpt["mean_reward"] = cur_mean
                    torch.save(best_ckpt, best_ckpt_path)
                    print(
                        f"[BestCkpt] Updated best.pt @ update={update_count} "
                        f"MeanReward={cur_mean:.3f}"
                    )

        # ---- M5-F5: 评估守卫（episode 节奏 mini-eval；劣化 → 回滚 + lr 减半）----
        # 位于 update 边界：当前 buffer 已被消费，回滚后下一轮 rollout 从恢复的
        # 策略全新采集，无 off-policy 污染
        if guard_on and episode_count - guard_last_ep >= int(config.ppo.eval_guard_every):
            guard_last_ep = episode_count
            g_res = _guard_check(guard_env, worker_ac, meta_ac, device, config)
            pp_str = " ".join(
                f"{p['snr_db']:.0f}dB:{p['deliv']:.3f}/Q{p['queue']:.1f}"
                f"{'√' if p['met'] else '×'}"
                for p in g_res["per_point"])
            print(f"[Guard] ep={episode_count} avg_deliv={g_res['avg_deliv']:.4f} "
                  f"avg_queue={g_res['avg_queue']:.2f} "
                  f"points_met={g_res['points_met']}/{g_res['n_points']} "
                  f"ok={g_res['ok']} | {pp_str}")
            es = guard_plateau_stop(
                stall_count=guard_stall_count,
                patience=int(getattr(config.ppo, "eval_guard_early_stop_patience", 2)),
                episode_count=episode_count,
                min_episodes=int(getattr(config.ppo, "eval_guard_early_stop_min_episodes", 3000)),
                enabled=bool(getattr(config.ppo, "eval_guard_early_stop", False)),
            )
            if g_res["avg_deliv"] > guard_best_deliv:
                guard_best_deliv = g_res["avg_deliv"]
                guard_stall_count = 0
                torch.save(
                    {
                        "episode": episode_count,
                        "total_steps": total_steps,
                        "update_count": update_count,
                        "meta_update_count": meta_update_count,
                        "worker_state_dict": worker_ac.state_dict(),
                        "meta_state_dict": meta_ac.state_dict(),
                        "worker_optimizer": worker_trainer.optimizer.state_dict(),
                        "meta_optimizer": meta_trainer.optimizer.state_dict(),
                        "config": vars(config.ppo),
                        "guard_avg_deliv": guard_best_deliv,
                    },
                    os.path.join(checkpoint_dir, "best_guard.pt"),
                )
                print(f"[Guard] best_guard.pt updated @ ep={episode_count} "
                      f"avg_deliv={guard_best_deliv:.4f}")
            if g_res["ok"]:
                guard_last_good = (
                    {k: v.detach().cpu().clone()
                     for k, v in worker_ac.state_dict().items()},
                    {k: v.detach().cpu().clone()
                     for k, v in meta_ac.state_dict().items()},
                )
            else:
                guard_halvings = _guard_rollback(
                    worker_ac, meta_ac, worker_trainer, guard_last_good,
                    guard_halvings, guard_lr_groups, config)
            if g_res["avg_deliv"] <= guard_best_deliv:
                guard_stall_count += 1
                if es:
                    print(
                        f"[Guard] EARLY-STOP: guard 连续 {guard_stall_count} 次评估未刷新 "
                        f"best（best={guard_best_deliv:.4f} @ ep≈上次更新）→ 提前终止",
                        flush=True,
                    )
                    early_stopped = True
                    break

    # ------------------------- 最终 checkpoint ------------------------- #
    final_ckpt = os.path.join(checkpoint_dir, "final.pt")
    final_ckpt_data = {
        "episode": episode_count,
        "total_steps": total_steps,
        "update_count": update_count,
        "meta_update_count": meta_update_count,
        "early_stopped": bool(early_stopped),
        "worker_state_dict": worker_ac.state_dict(),
        "meta_state_dict": meta_ac.state_dict(),
        "worker_optimizer": worker_trainer.optimizer.state_dict(),
        "meta_optimizer": meta_trainer.optimizer.state_dict(),
        "config": vars(config.ppo),
    }
    if delivery_constraint is not None:
        final_ckpt_data["lagrangian_state"] = {
            "lambda": [float(x) for x in delivery_constraint.lambda_val],
            "integral": [float(x) for x in delivery_constraint.integral],
            "episode_count": int(delivery_constraint.episode_count),
            "delivery_history": [
                list(bucket[-1000:]) for bucket in delivery_constraint.delivery_history
            ],
        }
    if dvq is not None:
        final_ckpt_data["dvq_state"] = {
            "Z": dvq.Z.tolist(),
            "episode_count": int(dvq.episode_count),
        }
    torch.save(final_ckpt_data, final_ckpt)
    print(
        f"[Train] Training complete. Total steps={total_steps}, "
        f"Episodes={episode_count}, MetaUpdates={meta_update_count}"
    )
    print(f"[Train] Final checkpoint: {final_ckpt}")

    return worker_ac, meta_ac


def _build_config_from_args() -> Config:
    """从命令行参数构造 Config，仅覆盖显式传入的字段。"""
    parser = argparse.ArgumentParser(description="RSMA Hierarchical PPO training entry")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="实验名 (覆盖 Config.exp_name)")
    parser.add_argument("--total_episodes", type=int, default=None,
                        help="总训练 episode 数 (覆盖 ppo.total_episodes)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (覆盖 Config.seed)")
    parser.add_argument("--device", type=str, default=None,
                        help="设备: cpu/cuda (覆盖 Config.device)")
    parser.add_argument("--csi_mode", type=str, default=None,
                        choices=["perfect", "imperfect"],
                        help="CSI 模式 (覆盖 system.csi_mode)")
    parser.add_argument("--snr_randomize", type=lambda v: v.lower() in ("1", "true", "yes"),
                        default=None,
                        help="是否启用 SNR domain randomization")
    parser.add_argument("--snr_min_db", type=float, default=None)
    parser.add_argument("--snr_max_db", type=float, default=None)
    parser.add_argument("--T_s", type=int, default=None,
                        help="慢尺度决策间隔 (覆盖 hierarchical.T_s)")
    parser.add_argument("--use_attention", type=lambda v: v.lower() in ("1", "true", "yes"),
                        default=None,
                        help="Worker 改用 Self-Attention + FiLM (替代 MLP)")
    parser.add_argument("--obs_history_window", type=int, default=None,
                        help="时序历史窗口长度 H (覆盖 system.obs_history_window, 0=关闭)")
    args = parser.parse_args()

    cfg = Config()
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name
    if args.total_episodes is not None:
        cfg.ppo.total_episodes = args.total_episodes
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.csi_mode is not None:
        cfg.system.csi_mode = args.csi_mode
    if args.snr_randomize is not None:
        cfg.system.snr_domain_randomization = args.snr_randomize
    if args.snr_min_db is not None:
        cfg.system.snr_min_db = args.snr_min_db
    if args.snr_max_db is not None:
        cfg.system.snr_max_db = args.snr_max_db
    if args.T_s is not None:
        cfg.hierarchical.T_s = args.T_s
    if args.use_attention is not None:
        cfg.ppo.use_attention = args.use_attention
    if args.obs_history_window is not None:
        cfg.system.obs_history_window = args.obs_history_window
    return cfg


if __name__ == "__main__":
    cfg = _build_config_from_args()
    train(cfg)
