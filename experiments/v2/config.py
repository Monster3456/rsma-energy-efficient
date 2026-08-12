"""v2 配置：继承基类 Config，覆盖关键配置

v2 相对 v1 的改动：
1. 训练 EE 口径对齐到评估口径 R/P_tx（pa_efficiency=1, circuit_power=0）
2. 关闭所有已验证无效的机制（RCPO/DSL/PLPI/SPLC/HEK/DGMW/LDPS 等）
3. 保留已验证有效的 4 条成功方向
4. 保留 best.pt 实际结构（4 bucket experts, FiLM, group-alpha）
"""
import sys
sys.path.insert(0, '/Users/libo/PycharmProjects/rsma-energy-efficient')

from src.config import Config


def make_v2_config() -> Config:
    """构建 v2 配置，覆盖 v1 关键参数。"""
    cfg = Config()
    lya = cfg.lyapunov
    rew = cfg.reward
    ppo = cfg.ppo
    hier = cfg.hierarchical
    con = cfg.constraint
    sys = cfg.system

    # ================================================================
    # 1. [核心] 训练 EE 口径对齐到评估口径 R/P_tx
    #    v1: P_consume = P_tx / 0.35 + 5.0（含电路功耗+PA效率）
    #    v2: P_consume = P_tx / 1.0 + 0.0 = P_tx（纯发射功率，与评估一致）
    # ================================================================
    sys.pa_efficiency = 1.0
    sys.circuit_power = 0.0
    lya.ee_power_floor = 0.01      # 仅防除零，不扭曲口径

    # ================================================================
    # 2. 关闭所有已验证无效的机制（保留代码但通过配置禁用）
    # ================================================================
    # RCPO cost-critic（iter-30/33/35 验证失败）
    con.use_rcpo_cost_critic = False
    # Delivery-Barrier 安全层（iter-xx 验证失败）
    con.use_delivery_safety_layer = False
    # Policy-Level Power Incentive（验证失败）
    con.use_power_incentive = False
    # Self-Paced Load Curriculum（验证失败）
    con.use_load_curriculum = False
    # Heuristic Expert Kickstart（验证失败）
    con.hek_enabled = False
    # DGMW panic mode（验证失败，关闭以消除死分支噪声）
    lya.dgmw_enable = False
    # LDPS 势函数整形（验证失败）
    lya.ldps_enable = False
    # 词典序梯度投影（配置矛盾 + 验证失败，关闭）
    con.use_lexicographic_projection = False
    # CRPO 开关（保留但关闭，避免与 ASOPA 解耦冲突）
    con.use_crpo_switch = False
    # 目标反转（关闭，恢复 EE 为主目标）
    lya.w_deliv = 0.0
    lya.w_ee_floor = 0.0

    # ================================================================
    # 3. 保留已验证有效的成功方向
    # ================================================================
    # CuRLTraC 性能触发式难度退火课程（iter-22 成功）
    ppo.use_curriculum = True
    ppo.curriculum_n_stages = 5
    ppo.curriculum_beta = 0.5
    ppo.curriculum_fallback_beta = 0.3
    ppo.curriculum_window = 50
    ppo.curriculum_patience = 20
    ppo.curriculum_snr_min_start = 5.0
    ppo.curriculum_snr_min_end = -5.0
    ppo.curriculum_load_ratio_max_start = 0.8
    ppo.curriculum_load_ratio_max_end = 1.3
    ppo.curriculum_min_stage_episodes = 50
    ppo.curriculum_freeze_episodes = 500
    ppo.curriculum_apply_snr_max = True

    # 逐桶/逐点硬例挖掘（iter-4/5/8 成功）
    ppo.hard_mining_enable = True
    ppo.hard_mining_w_hard = 2.0
    ppo.hard_mining_ema_decay = 0.99
    ppo.hard_mining_floor_frac = 0.5
    ppo.hard_mining_target = 1.0
    ppo.hm_granularity = "point"     # 逐评估点粒度（iter-8 成功）
    ppo.point_w_hard = 2.0
    ppo.streak_mult = 1.5
    ppo.streak_cap = 4.0
    ppo.fail_entropy_scale = 1.5
    ppo.point_base_floor = 0.5

    # 性能挂钩定向熵激励（iter-8 成功）
    ppo.entropy_performance_linked = True
    ppo.entropy_link_coef = 1.0

    # SFL 可学习性采样（辅助硬例挖掘）
    con.use_sfl_sampling = True

    # ================================================================
    # 4. 保留 best.pt 实际结构
    # ================================================================
    # 4 桶专家多头 Actor（best.pt 中实际生效）
    ppo.use_bucket_experts = True
    ppo.n_bucket_experts = 4
    ppo.lr_bucket_expert = 1e-4

    # Group-level alpha（Meta 输出 4 组 α）
    hier.group_level_alpha = True
    hier.group_alpha_min = 0.1
    hier.group_alpha_max = 0.95

    # State-Augmented 观测（λ/DVQ/η 注入观测）
    lya.state_aug_enabled = True

    # ================================================================
    # 5. [v2.3 方案C] Lagrangian 通道关闭（主目标反转的前提）
    # ================================================================
    # 教训（v2.1/v2.2）：线性功率税被 Lagrangian 指数惩罚压倒——delivery 未达标时
    # penalty 指数增长，PPO 优先避免惩罚尖峰而非降功率。方案 C 把主目标反转为
    # “最小功率 + 交付奖励”，delivery 约束改由 w_deliv×delivery 线性奖励体现
    # （见 train.py _patch_power_primary），不再需要 Lagrangian 对偶惩罚。
    con.use_lagrangian = False
    con.use_step_penalty = False

    # ================================================================
    # 6. Lagrangian 机制参数保留（v2.3 已关闭开关，参数保留便于回退）
    # ================================================================
    con.use_lagrangian = False      # v2.3 关闭（方案C主目标反转）
    con.delivery_threshold = 0.98
    con.lagrangian_lr = 0.05
    con.lagrangian_init = 5.0
    con.lagrangian_max = 100.0
    con.lagrangian_warmup = 200
    con.lagrangian_decay = 0.999
    con.use_step_penalty = False    # v2.3 关闭（同 Lagrangian）
    con.slcp_ema_beta = 0.85
    con.use_dvq = True
    con.dvq_weight = 0.5
    con.dvq_z_max = 200.0
    con.dvq_warmup = 200

    # ================================================================
    # 6. 保留 Dinkelbach 分式规划（EE 线性化，理论正确）
    # ================================================================
    lya.use_dinkelbach = True
    lya.dinkelbach_q_init = 1.0
    lya.dinkelbach_decay = 0.99
    lya.dinkelbach_q_min = 0.01
    lya.dinkelbach_q_max = 50.0
    lya.dinkelbach_outer_update = True
    lya.dinkelbach_update_interval = 10
    lya.dinkelbach_anchor_alpha = 0.2

    # ================================================================
    # 7. 保留 Lyapunov/LDPTRLQ 核心奖励结构
    # ================================================================
    lya.ldptrlq_q_weight = 2.0
    lya.drift_weight = 60.0
    lya.blend_alpha = 1.5
    lya.ee_weight = 8.0
    lya.V = 1.0
    lya.V_min = 0.3
    lya.V_max = 5.0
    lya.V_exponent = 1.5
    lya.adaptive_V = True
    lya.w_backpressure = 25.0
    lya.overflow_warn_ratio = 0.6
    lya.w_early_overflow = 5.0

    rew.w_overflow = 20.0
    rew.w_outage = 3.0
    rew.w_power_exceed = 10.0
    rew.reward_scale = 0.01
    rew.ee_sat_cap = 8.0

    # ================================================================
    # 8. PPO 训练超参（保留 v1 best 配置）
    # ================================================================
    ppo.lr_actor = 3e-4
    ppo.lr_critic = 1e-3
    ppo.lr_log_std = 1e-3
    ppo.gamma = 0.99
    ppo.gae_lambda = 0.95
    ppo.clip_ratio = 0.2
    ppo.entropy_coef = 0.008
    ppo.value_coef = 0.5
    ppo.max_grad_norm = 0.5
    ppo.n_epochs = 8
    ppo.batch_size = 64
    ppo.rollout_steps = 2048
    ppo.total_episodes = 1500
    ppo.hidden_dim = 256
    ppo.n_layers = 3
    ppo.action_dim = 96
    ppo.target_kl = 0.02
    ppo.lr_decay_final_ratio = 0.2

    # Attention Worker 结构
    ppo.use_attention = True
    ppo.worker_d_user = 13
    ppo.worker_attn_hidden = 128
    ppo.worker_attn_heads = 4
    ppo.worker_attn_layers = 2

    # Mean-Field MAPPO
    ppo.use_mean_field_critic = True
    ppo.use_per_user_advantage = True
    ppo.per_user_adv_weight = 1.0

    # 续训参数
    ppo.resume_lr_actor_scale = 1.0
    ppo.resume_clip_ratio = -1.0

    # ================================================================
    # 9. Meta-Controller 配置
    # ================================================================
    hier.T_s = 20
    hier.meta_lr = 5e-4
    hier.meta_gamma = 0.95
    hier.meta_hidden_dim = 64
    hier.meta_clip_ratio = 0.2
    hier.meta_entropy_coef = 0.05
    hier.meta_n_heads = 4
    hier.meta_d_user = 12
    hier.meta_n_epochs = 4
    hier.meta_update_threshold = 30
    hier.meta_gae_lambda = 0.95
    hier.meta_max_grad_norm = 0.5
    hier.n_groups = 4
    hier.meta_group_entropy_coef = 0.12
    hier.meta_group_tau_init = 2.0
    hier.meta_group_tau_min = 0.5
    hier.meta_balance_coef = 0.1
    hier.meta_reward_scale = 0.02
    hier.alpha_guidance_enabled = True
    hier.alpha_guidance_scale = 0.3
    hier.alpha_guidance_warmup = 300
    hier.pl_temperature = 1.0

    # ================================================================
    # 10. [v2 升级] Meta-Controller per-bucket alpha 缩放
    # ================================================================
    hier.use_bucket_meta_alpha = True
    hier.n_alpha_buckets = 7
    hier.bucket_meta_alpha_init = 1.0

    # ================================================================
    # 11. 系统配置
    # ================================================================
    sys.K = 24
    sys.N_r = 16
    sys.N_t = 2
    sys.P_k_max = 1.0
    sys.sigma2 = 0.01
    sys.T = 200
    sys.tau = 1.0
    sys.rho = 0.9
    sys.csi_mode = "imperfect"
    sys.snr_domain_randomization = True
    sys.snr_min_db = -5.0
    sys.snr_max_db = 35.0
    sys.precoding_type = "rzf"
    # [v2 功率税改造] 关闭 softmax 总功率归一化：softmax 会把总功率钉在
    # 预算附近（PPO 23W 恒定功率根因之一），且使线性功率税失效。
    # 改为直接映射 p_total∈[0,1]，PPO 可整体降功率，功率税才有意义。
    sys.use_softmax_power_share = False
    sys.adaptive_lambda = True
    sys.target_load_ratio = 0.8
    sys.lambda_min = 0.2
    sys.lambda_max = 10.0
    sys.load_ratio_min = 0.75
    sys.load_ratio_max = 1.3
    sys.load_ratio_snr_mid = 15.0
    sys.load_ratio_steepness = 0.3
    sys.enable_hetero_qos = True
    sys.qos_classes = 3
    sys.qos_lambda_multipliers = (2.0, 1.0, 0.3)
    sys.qos_drift_weights = (8.0, 1.0, 0.2)
    sys.qos_overflow_weights = (8.0, 1.0, 0.2)
    sys.use_dvq = True
    sys.dvq_threshold = 0.98
    sys.dvq_weight = 0.6
    sys.dvq_cvar_alpha = 0.25
    sys.dvq_warmup_episodes = 200
    sys.dvq_z_max = 200.0

    return cfg