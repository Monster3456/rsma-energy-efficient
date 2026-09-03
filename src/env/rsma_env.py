"""RSMA上行多用户Gymnasium环境

队列稳定上行MIMO-RSMA长期能效最大化环境。
每个时隙，智能体决策各用户的功率分割比、总发射功率和服务速率，
基于Lyapunov漂移加惩罚框架计算奖励，在长期约束下最小化总发射功率。
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import OBS_HISTORY_ALLOWED_FEATURES, Config
from src.env.beamforming import BeamformingDesigner
from src.env.channel import (
    ChannelGenerator,
    compute_path_loss_db,
    compute_path_loss_linear,
    generate_user_distances,
)
from src.env.rate_calc import RSMARateCalculator
from src.env.traffic import TrafficManager, estimate_capacity, snr_adaptive_load_ratio
from src.optimization.lyapunov import LyapunovController


class RSMAEnergyEnv(gym.Env):
    """队列稳定上行MIMO-RSMA长期能效最大化环境.

    状态空间 (按块拼接, 共 10K+1 维, K=16 时 161 维):
        - h_gains_norm:       (K,)   归一化信道增益
        - queue_norm:         (K,)   归一化队列积压
        - delta_queue_norm:   (K,)   队列变化 (归一化)
        - arrivals_norm:      (K,)   到达率 (归一化)
        - power_history:      (K,)   上一时隙归一化功率
        - rate_history:       (K,)   上一时隙归一化速率
        - interference_feat:  (K,)   干扰特征
        - fairness_feat:      (K,)   公平性特征
        - split_mask:         (K,)   分裂掩码 (Meta-Controller 决策)
        - group_ids_norm:     (K,)   归一化用户组ID
        - time_ratio:         (1,)   时间进度 t/T

    动作空间 (3K 或 4K 维, 连续, 区间 [-1, 1]):
        - alpha_k:    公共/私有功率分割比 (映射到 [0, 1])
        - p_total_k:  归一化总功率        (映射到 [0, 1])
        - d_ratio_k:  归一化发送比特数    (映射到 [0, 1])
        - [可选] decoding_order_k: Plackett-Luce 解码优先级 (值越小越先解码)

    奖励: Lyapunov 漂移加惩罚 (由 LyapunovController.compute_reward 计算).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Config] = None):
        super().__init__()

        self.config: Config = config if config is not None else Config()

        sys_cfg = self.config.system

        self.K: int = sys_cfg.K
        self.N_r: int = sys_cfg.N_r
        self.N_t: int = sys_cfg.N_t
        self.P_k_max: float = sys_cfg.P_k_max
        self.sigma2: float = sys_cfg.sigma2
        self.T: int = sys_cfg.T
        self.tau: float = sys_cfg.tau
        self.Q_max: float = sys_cfg.Q_max
        self.lambda_rate: float = sys_cfg.lambda_rate
        self.csi_mode: str = sys_cfg.csi_mode
        self.outage_prob: float = sys_cfg.outage_prob
        # 预编码/接收机方案: "rzf" (统一MRT+MMSE-SIC, 默认) 或 "mrt" (正交补+matched filter, 向后兼容)
        self.precoding_type: str = str(getattr(sys_cfg, "precoding_type", "rzf")).lower()
        if self.precoding_type not in ("rzf", "mrt"):
            raise ValueError(
                f"precoding_type must be 'rzf' or 'mrt', got {self.precoding_type!r}"
            )

        # SNR Domain Randomization 参数
        self.snr_domain_randomization: bool = bool(
            getattr(sys_cfg, "snr_domain_randomization", False)
        )
        self.snr_min_db: float = float(getattr(sys_cfg, "snr_min_db", 0.0))
        self.snr_max_db: float = float(getattr(sys_cfg, "snr_max_db", 30.0))
        # Task#9: SNR 自适应到达率参数
        self.adaptive_lambda_enabled: bool = bool(
            getattr(sys_cfg, "adaptive_lambda", False)
        )
        self.target_load_ratio: float = float(
            getattr(sys_cfg, "target_load_ratio", 0.75)
        )
        self.lambda_min: float = float(getattr(sys_cfg, "lambda_min", 1.5))
        self.lambda_max: float = float(getattr(sys_cfg, "lambda_max", 7.0))
        # Task#44: SNR 自适应 load ratio 参数
        self.load_ratio_min: float = float(getattr(sys_cfg, "load_ratio_min", 0.3))
        self.load_ratio_max: float = float(getattr(sys_cfg, "load_ratio_max", 0.9))
        self.load_ratio_snr_mid: float = float(
            getattr(sys_cfg, "load_ratio_snr_mid", 12.0)
        )
        self.load_ratio_steepness: float = float(
            getattr(sys_cfg, "load_ratio_steepness", 0.15)
        )
        # 当前 episode 实际使用的 λ（供观测与记录使用）
        self.current_lambda_rate: float = float(sys_cfg.lambda_rate)
        self.current_snr_db: float = float(
            -10.0 * np.log10(max(sys_cfg.sigma2, 1e-12))
        )
        # 内部 RNG, 与 channel/traffic 子模块的 RNG 解耦, 便于复现
        self._snr_rng: np.random.Generator = np.random.default_rng(self.config.seed)

        # Task#13: 路径损耗参数与 per-episode 用户距离 / PL 缓存
        self.use_path_loss: bool = bool(getattr(sys_cfg, "use_path_loss", False))
        self.cell_radius_min: float = float(getattr(sys_cfg, "cell_radius_min", 50.0))
        self.cell_radius_max: float = float(getattr(sys_cfg, "cell_radius_max", 300.0))
        self.path_loss_exponent: float = float(getattr(sys_cfg, "path_loss_exponent", 3.76))
        self.path_loss_ref_db: float = float(getattr(sys_cfg, "path_loss_ref_db", 128.1))
        self.path_loss_ref_distance: float = float(
            getattr(sys_cfg, "path_loss_ref_distance", 100.0)
        )
        self._pl_rng: np.random.Generator = np.random.default_rng(self.config.seed)
        # 在 reset() 中重新采样 (距离在 episode 内保持不变 - 大尺度慢变)
        self.user_distances: np.ndarray = np.full(self.K, self.path_loss_ref_distance,
                                                  dtype=np.float64)
        self.path_loss_db_relative: np.ndarray = np.zeros(self.K, dtype=np.float64)
        self.path_loss_amp: np.ndarray = np.ones(self.K, dtype=np.float64)

        # 子模块实例（seed 通过重新构造完成）
        self._seed: Optional[int] = self.config.seed
        self._build_modules(self._seed)

        # Task#20 P3 （可选）: obs 中是否追加 per-user CSI 估计不确定度特征
        self.obs_include_csi_quality: bool = bool(
            getattr(sys_cfg, "obs_include_csi_quality", False)
        )

        # ---- 时序策略容量：历史窗口观测（默认关，obs_dim 不变）----
        # window=H>0 时在 per-user 特征区末尾追加最近 H 个时隙的逐用户特征
        # 快照（最旧在前），策略保持无状态。窗口在 reset() 清空，由首次
        # _get_observation 用 reset 时刻快照预填充（避免零瞬态）。
        self.obs_history_window: int = int(getattr(sys_cfg, "obs_history_window", 0))
        self.obs_history_features: tuple = tuple(
            getattr(sys_cfg, "obs_history_features", OBS_HISTORY_ALLOWED_FEATURES)
        )
        # 构造后修改 config 属性的常见用法走不到 SystemConfig.__post_init__，
        # 在消费端（env）入口再次校验
        unknown_feats = [
            f for f in self.obs_history_features
            if f not in OBS_HISTORY_ALLOWED_FEATURES
        ]
        if unknown_feats:
            raise ValueError(
                f"obs_history_features 含未知特征 {unknown_feats}，"
                f"允许值: {list(OBS_HISTORY_ALLOWED_FEATURES)}"
            )
        if self.obs_history_window < 0:
            raise ValueError(
                f"obs_history_window 必须 >= 0，收到 {self.obs_history_window}"
            )
        self._obs_history: deque = deque(
            maxlen=max(self.obs_history_window, 1)
        )

        # ---- v13: 异构 QoS 配置 (必须在 obs_dim 计算之前初始化) ----
        self.enable_hetero_qos = bool(getattr(sys_cfg, 'enable_hetero_qos', False))
        self.qos_classes = int(getattr(sys_cfg, 'qos_classes', 3))
        self.qos_lambda_multipliers = tuple(getattr(sys_cfg, 'qos_lambda_multipliers', (1.5, 1.0, 0.5)))
        self.qos_drift_weights = tuple(getattr(sys_cfg, 'qos_drift_weights', (3.0, 1.0, 0.5)))
        self.qos_overflow_weights = tuple(getattr(sys_cfg, 'qos_overflow_weights', (3.0, 1.0, 0.5)))
        self.qos_class_ids = np.zeros(self.K, dtype=np.int32)
        self.qos_user_drift_weights = np.ones(self.K, dtype=np.float64)
        self.qos_user_overflow_weights = np.ones(self.K, dtype=np.float64)

        # 观测维度: 11K+3 (Task#13 起新增 per-user normalized_distance 一块)
        # h_gains(K) + queue(K) + delta_queue(K) + arrivals(K)
        # + power_history(K) + rate_history(K) + interference(K)
        # + fairness(K) + split_mask(K) + group_ids_norm(K)
        # + distance_norm(K) + time_ratio(1) + snr_db_norm(1) + epsilon_csi_norm(1)
        self._obs_dim: int = self.K * 11 + 3
        # v14 QoS: 启用时新增 2K 维度 (lambda_load + priority)
        if self.enable_hetero_qos:
            self._obs_dim += 2 * self.K
        if self.obs_include_csi_quality:
            self._obs_dim += self.K
        # 时序历史窗口: 每个追加 lag 快照贡献 len(features) 个 K 维块
        # （拼接在既有 per-user 特征块之后、global 尾之前）
        if self.obs_history_window > 0:
            self._obs_dim += self.obs_history_window * len(self.obs_history_features) * self.K

        # 分层架构支持
        self.hierarchical_enabled: bool = getattr(
            getattr(self.config, 'hierarchical', None), 'enabled', False
        )
        if self.hierarchical_enabled:
            self._obs_dim += 1  # 追加 power_budget_normalized 维度

        # Task#38: Group-level α 架构开关。开启后 split_mask 被作为
        # per-user 连续 α 值【覆盖】Worker 的 α 输出（而非仅作二值 gate）。
        h_cfg = getattr(self.config, 'hierarchical', None)
        self.group_level_alpha: bool = bool(
            getattr(h_cfg, 'group_level_alpha', False)
        )

        self.P_target = None  # 当前功率预算 (由 Meta-Controller 设置)
        self.step_in_episode: int = 0
        self._meta_stats_accum = {}
        self._meta_stats_count = 0
        self._reset_meta_stats()

        # ---- 分层 PPO 追踪变量 ----
        self.split_mask: np.ndarray = np.ones(self.K, dtype=np.float32)
        self.group_ids: np.ndarray = np.zeros(self.K, dtype=np.int64)
        self.decoding_order: Optional[np.ndarray] = None
        self._prev_rates: np.ndarray = np.zeros(self.K, dtype=np.float64)
        self.avg_rates: np.ndarray = np.zeros(self.K, dtype=np.float64)
        self.overflow_counts: np.ndarray = np.zeros(self.K, dtype=np.float64)
        self.power_utilization: np.ndarray = np.zeros(self.K, dtype=np.float64)
        self.channel_gains_norm: np.ndarray = np.zeros(self.K, dtype=np.float64)

        # 空间定义
        self.action_dim: int = 3 * self.K
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        # 观测以归一化形式输出, 使用宽松上下界保证健壮性
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self._obs_dim,), dtype=np.float32
        )

        # ---- State-Augmented 观测：外部可注入的对偶/全局状态 ----
        lya_cfg = self.config.lyapunov
        self.state_aug_enabled: bool = bool(getattr(lya_cfg, "state_aug_enabled", False))
        self.dinkelbach_eta: float = float(getattr(lya_cfg, "dinkelbach_q_init", 1.0))
        self.lambda_bucket_val: float = 0.0

        # 将 Dinkelbach q 管理权交给训练循环（外层每 10 episode 更新）
        self.lyapunov.dinkelbach_q = self.dinkelbach_eta
        self.lyapunov.dinkelbach_outer_update = bool(
            getattr(lya_cfg, "dinkelbach_outer_update", False)
        )

        # 时隙计数与运行时缓存
        self.t: int = 0
        self._prev_power: np.ndarray = np.zeros(self.K, dtype=np.float64)
        self._channel_obs: dict = {}

    # ------------------------------------------------------------------ #
    # 内部辅助
    # ------------------------------------------------------------------ #
    def _build_modules(self, seed: Optional[int]) -> None:
        sys_cfg = self.config.system
        lya_cfg = self.config.lyapunov
        rew_cfg = self.config.reward
        self.ee_sat_cap: float = float(getattr(rew_cfg, "ee_sat_cap", 8.0))

        self.channel = ChannelGenerator(
            K=sys_cfg.K, N_r=sys_cfg.N_r, N_t=sys_cfg.N_t,
            rho=sys_cfg.rho,
            csi_mode=sys_cfg.csi_mode,
            epsilon_csi=sys_cfg.epsilon_csi,
            csi_delay=sys_cfg.csi_delay,
            seed=seed,
            sigma2=sys_cfg.sigma2,
            snr_dependent_csi=bool(getattr(sys_cfg, "snr_dependent_csi", False)),
            pilot_power=float(getattr(sys_cfg, "pilot_power", 1.0)),
            epsilon_floor=float(getattr(sys_cfg, "epsilon_floor", 0.01)),
        )
        self.rate_calc = RSMARateCalculator(
            K=sys_cfg.K, N_r=sys_cfg.N_r, N_t=sys_cfg.N_t,
            sigma2=sys_cfg.sigma2,
        )
        self.beamformer = BeamformingDesigner(
            K=sys_cfg.K, N_r=sys_cfg.N_r, N_t=sys_cfg.N_t,
        )
        self.traffic = TrafficManager(
            K=sys_cfg.K,
            lambda_rate=sys_cfg.lambda_rate,
            Q_max=sys_cfg.Q_max,
            packet_size=sys_cfg.packet_size,
            seed=seed,
        )
        self.lyapunov = LyapunovController(
            K=sys_cfg.K, V=lya_cfg.V,
            V_min=lya_cfg.V_min, V_max=lya_cfg.V_max,
            V_exponent=lya_cfg.V_exponent,
            adaptive_V=lya_cfg.adaptive_V,
            Q_max=sys_cfg.Q_max,
            use_virtual_queue=lya_cfg.use_virtual_queue,
            vq_step_size=lya_cfg.vq_step_size,
            reward_scale=rew_cfg.reward_scale,
            w_overflow=rew_cfg.w_overflow,
            w_outage=rew_cfg.w_outage,
            snr_gamma=getattr(lya_cfg, "snr_gamma", 0.0),
            snr_ref_db=getattr(lya_cfg, "snr_ref_db", 30.0),
            v_eff_cap_factor=getattr(lya_cfg, "v_eff_cap_factor", 3.0),
            snr_floor=getattr(lya_cfg, "snr_floor", 1.0),
            snr_power_exponent=getattr(lya_cfg, "snr_power_exponent", 1.0),
            P_min_floor=getattr(lya_cfg, "P_min_floor", 0.0),
            ee_mode=getattr(lya_cfg, "ee_mode", False),
            ee_weight=getattr(lya_cfg, "ee_weight", 10.0),
            ee_power_floor=getattr(lya_cfg, "ee_power_floor", 0.5),
            blend_alpha=getattr(lya_cfg, "blend_alpha", 2.0),
            # P0 修复参数：drift 放大 + 早期溢出渐进惩罚
            drift_weight=getattr(lya_cfg, "drift_weight", 1.0),
            overflow_warn_ratio=getattr(lya_cfg, "overflow_warn_ratio", 1.0),
            w_early_overflow=getattr(lya_cfg, "w_early_overflow", 0.0),
            # Task#15: PA + 电路功耗模型
            pa_efficiency=getattr(sys_cfg, "pa_efficiency", 0.35),
            circuit_power=getattr(sys_cfg, "circuit_power", 5.0),
            # v22: Lyapunov-DRL 权衡参数
            lyapunov_V=getattr(lya_cfg, "lyapunov_V", 5.0),
            # v13s: LDPTRLQ 队列二次惩罚权重
            ldptrlq_q_weight=getattr(lya_cfg, "ldptrlq_q_weight", 20.0),
            # CRPO 风格可行性门控
            ee_gate_enable=getattr(lya_cfg, "ee_gate_enable", True),
            ee_gate_threshold=getattr(lya_cfg, "ee_gate_threshold", 0.98),
            ee_gate_band=getattr(lya_cfg, "ee_gate_band", 0.15),
            ee_gate_min=getattr(lya_cfg, "ee_gate_min", 0.05),
            # backpressure / max-weight drift reward
            w_backpressure=getattr(lya_cfg, "w_backpressure", 25.0),
            # 目标反转：交付凹效用主目标 + EE 单侧下界软约束
            w_deliv=getattr(lya_cfg, "w_deliv", 3.0),
            deliv_ref=getattr(lya_cfg, "deliv_ref", 2.0),
            ee_floor=getattr(lya_cfg, "ee_floor", 30.0),
            w_ee_floor=getattr(lya_cfg, "w_ee_floor", 0.5),
            # DGMW: Delivery-Gated Max-Weight Pure Mode
            dgmw_enable=getattr(lya_cfg, "dgmw_enable", False),
            dgmw_panic_threshold=getattr(lya_cfg, "dgmw_panic_threshold", 0.55),
            dgmw_recovery_threshold=getattr(lya_cfg, "dgmw_recovery_threshold", 0.85),
            dgmw_mw_weight=getattr(lya_cfg, "dgmw_mw_weight", 100.0),
            # LDPS: Lyapunov Drift Potential Shaping (PBRS)
            ldps_enable=getattr(lya_cfg, "ldps_enable", False),
            ldps_eta=getattr(lya_cfg, "ldps_eta", 30.0),
            ldps_gamma=getattr(lya_cfg, "ldps_gamma", 0.99),
            # Dinkelbach 变换：EE 比率→线性目标
            use_dinkelbach=getattr(lya_cfg, "use_dinkelbach", False),
            dinkelbach_q_init=getattr(lya_cfg, "dinkelbach_q_init", 1.0),
            dinkelbach_decay=getattr(lya_cfg, "dinkelbach_decay", 0.99),
            dinkelbach_q_min=getattr(lya_cfg, "dinkelbach_q_min", 0.01),
            dinkelbach_q_max=getattr(lya_cfg, "dinkelbach_q_max", 50.0),
            # reward_v2: 逐用户 served 加权 + q* 冻结（默认关 = 原行为）
            reward_v2_per_user_weight=getattr(lya_cfg, "reward_v2_per_user_weight", False),
            reward_v2_fixed_q=getattr(lya_cfg, "reward_v2_fixed_q", 0.0),
            reward_v2_weight_lo=getattr(lya_cfg, "reward_v2_weight_lo", 0.5),
            reward_v2_weight_hi=getattr(lya_cfg, "reward_v2_weight_hi", 2.0),
        )

    def _parse_action(self, action) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """解析 3K 维动作为 (alpha, p_total, d_ratio).

        PPO 输出通常位于 (-1, 1) 区间, 通过 (x+1)/2 线性映射到 [0,1].

        Task#38: 当 group_level_alpha=True 时，split_mask 已包含 Meta 決定的
        per-user α 值 (连续)，此时覆盖 Worker 的 α 输出。
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] != self.action_dim:
            raise ValueError(f"action shape {a.shape} != ({self.action_dim},)")

        a = np.clip(a, -1.0, 1.0)
        mapped = (a + 1.0) / 2.0  # → [0, 1]

        K = self.K
        if self.group_level_alpha:
            # split_mask = per-user α (从 Meta group α 展开)，覆盖 Worker 的 α
            alpha = np.clip(self.split_mask.astype(np.float64), 0.0, 1.0)
        else:
            alpha = np.clip(mapped[0:K], 0.0, 1.0)
            # split_mask 约束: non-split 用户 alpha=0, 全部功率归私有流
            alpha = alpha * self.split_mask
        if bool(getattr(self.config.system, 'use_softmax_power_share', False)):
            # Mean-Field MAPPO: p 块是 K 个 logits，softmax 后得到总功率预算份额
            p_logits = a[K:2 * K]
            p_logits = p_logits - np.max(p_logits)
            p_exp = np.exp(p_logits)
            p_ratio = self.K * p_exp / (np.sum(p_exp) + 1e-12)
            p_total = np.clip(p_ratio, 0.0, 1.0)
        else:
            p_total = np.clip(mapped[K:2 * K], 0.0, 1.0)
        d_ratio = np.clip(mapped[2 * K:3 * K], 0.0, 1.0)
        return alpha, p_total, d_ratio

    def _get_observation(self) -> np.ndarray:
        """构造观测向量 (10K+1 维), 按设计顺序拼接.

        组成: h_gains(K) + queue(K) + delta_queue(K) + arrivals(K)
              + power_history(K) + rate_history(K) + interference(K)
              + fairness(K) + split_mask(K) + group_ids_norm(K) + time_ratio(1)
        """
        K = self.K
        lya_cfg = self.config.lyapunov
        Q = self.traffic.queue_lengths
        gains = self._channel_obs.get("channel_gains", np.zeros(K))

        # 归一化信道增益
        norm_gain = float(self.N_r * self.N_t)
        h_gains_norm = gains / max(norm_gain, 1e-8)
        self.channel_gains_norm = h_gains_norm.copy()  # 缓存供 get_user_features 使用

        # 归一化队列
        queue_norm = Q / max(self.Q_max, 1e-8)

        # 队列变化 (delta_queue)
        if self._prev_queue is not None:
            delta_queue = (Q - self._prev_queue) / max(self.Q_max, 1e-8)
        else:
            delta_queue = np.zeros(K, dtype=np.float64)
        delta_queue_norm = np.clip(delta_queue, -1.0, 1.0)

        # 到达率归一化（Task#9: 使用当前 episode 的自适应 λ）
        arrivals_norm = np.full(K, self.current_lambda_rate / max(self.Q_max, 1e-8),
                                dtype=np.float64)

        # 功率历史
        power_history = self._prev_power / max(self.P_k_max, 1e-8)

        # 速率历史
        rate_history = self._prev_rates / max(np.max(self._prev_rates), 1e-8) \
            if np.max(self._prev_rates) > 1e-8 else np.zeros(K, dtype=np.float64)

        # 干扰特征: 各用户信道增益占总增益的比例 (越小 -> 受干扰越大)
        total_gain = np.sum(gains) + 1e-8
        interference_feat = 1.0 - gains / total_gain

        # 公平性特征: 队列偏差 (越大 -> 越不公平, 该用户欠服务)
        mean_q = np.mean(Q) + 1e-8
        fairness_feat = np.clip(Q / mean_q - 1.0, -1.0, 2.0) / 2.0

        # 时间进度
        time_ratio = np.array([self.t / max(self.T, 1)], dtype=np.float64)

        # 归一化 group_ids: [0, G-1] -> [0, 1]
        n_groups = getattr(
            getattr(self.config, 'hierarchical', None), 'n_groups', 4
        )
        group_ids_norm = self.group_ids.astype(np.float64) / max(n_groups - 1, 1)

        # Task#13: 归一化距离特征 distance_norm ∈ [0, 1]
        d_range = max(self.cell_radius_max - self.cell_radius_min, 1e-8)
        distance_norm = (self.user_distances - self.cell_radius_min) / d_range
        distance_norm = np.clip(distance_norm, 0.0, 1.0).astype(np.float64)

        # v14 QoS: per-user QoS需求特征（2维，比单一类别ID信息更丰富）
        # dim1: lambda_load - 归一化到达率倍率，反映"谁流量大"
        # dim2: priority - 归一化服务优先级权重，反映"谁更重要"
        qos_lambda_feat = np.zeros(self.K, dtype=np.float64)
        qos_priority_feat = np.zeros(self.K, dtype=np.float64)
        if self.enable_hetero_qos:
            max_lambda_mult = max(self.qos_lambda_multipliers)
            max_drift_wt = max(self.qos_drift_weights)
            for i in range(self.qos_classes):
                mask = self.qos_class_ids == i
                qos_lambda_feat[mask] = self.qos_lambda_multipliers[i] / max(max_lambda_mult, 1e-8)
                qos_priority_feat[mask] = self.qos_drift_weights[i] / max(max_drift_wt, 1e-8)

        # P0 修复：暴露信道质量信号给 policy（之前完全缺失，导致 SNR DR 无效）
        #   snr_db = 10·log10(P_max / sigma2)
        #   归一化到 [0, 1]：以当前 env.snr_min_db..snr_max_db 区间归一（课程调度会动态修改）
        snr_db = 10.0 * np.log10(max(self.P_k_max, 1e-12) / max(self.sigma2, 1e-12))
        snr_min = float(getattr(self, "snr_min_db", getattr(self.config.system, "snr_min_db", 0.0)))
        snr_max = float(getattr(self, "snr_max_db", getattr(self.config.system, "snr_max_db", 30.0)))
        snr_norm = (snr_db - snr_min) / max(snr_max - snr_min, 1e-8)
        snr_norm = float(np.clip(snr_norm, -0.5, 1.5))  # 留点缓冲不强裁
        # Task#20: Îµ 归一化优先使用 channel 动态 sigma_est，回退到 epsilon_csi
        sigma_est_eff = float(
            self._channel_obs.get("sigma_est", float(self.config.system.epsilon_csi))
        )
        eps_norm = sigma_est_eff / 0.5

        # State-Augmented 观测：λ桶值 / DVQ积压 / Dinkelbach-eta 归一化后广播到每用户
        if self.state_aug_enabled:
            lagrangian_max = float(
                getattr(getattr(self.config, "constraint", None), "lagrangian_max", 100.0)
            )
            lambda_bucket_norm = np.full(
                K,
                float(np.clip(self.lambda_bucket_val / max(lagrangian_max, 1e-8), 0.0, 1.0)),
                dtype=np.float64,
            )
            dvq_y_max = float(getattr(self.lyapunov, "dvq_y_max", 5000.0))
            dvq_backlog_norm = np.full(
                K,
                float(np.clip(self.lyapunov._Y / max(dvq_y_max, 1e-8), 0.0, 1.0)),
                dtype=np.float64,
            )
            eta_max = float(getattr(lya_cfg, "dinkelbach_q_max", 50.0))
            eta_norm = np.full(
                K,
                float(np.clip(self.dinkelbach_eta / max(eta_max, 1e-8), 0.0, 1.0)),
                dtype=np.float64,
            )
            interference_feat = lambda_bucket_norm
            fairness_feat = dvq_backlog_norm
            distance_norm = eta_norm

        obs_blocks = [
            h_gains_norm.astype(np.float64),        # K: 信道增益
            queue_norm.astype(np.float64),          # K: 归一化队列
            delta_queue_norm.astype(np.float64),    # K: 队列变化
            arrivals_norm,                          # K: 到达率
            power_history.astype(np.float64),       # K: 功率历史
            rate_history.astype(np.float64),        # K: 速率历史
            interference_feat.astype(np.float64),   # K: 干扰特征 / λ桶值
            fairness_feat.astype(np.float64),       # K: 公平性特征 / DVQ积压
            self.split_mask.astype(np.float64),     # K: 分裂掩码
            group_ids_norm,                         # K: 归一化组ID
            distance_norm,                          # K: 距离 / Dinkelbach-eta
        ]

        # v14 QoS: 启用时追加 2K 维特征
        if self.enable_hetero_qos:
            obs_blocks.append(qos_lambda_feat)    # K维: 到达率负载
            obs_blocks.append(qos_priority_feat)  # K维: 服务优先级

        # 时序历史窗口：追加最近 H 个时隙的逐用户特征快照（最旧在前），
        # 每个 lag 贡献 len(features) 个 K 维块（按配置的特征顺序），
        # 位于既有 per-user 特征块之后、global 尾 (time_ratio, snr, eps) 之前。
        if self.obs_history_window > 0:
            hist_feat_blocks = {
                "h_gains": h_gains_norm,
                "queue": queue_norm,
                "power": power_history,
            }
            snapshot = np.concatenate(
                [np.asarray(hist_feat_blocks[f], dtype=np.float64)
                 for f in self.obs_history_features]
            )
            if len(self._obs_history) == 0:
                # reset 后首次调用：用 reset 时刻快照预填充窗口（避免零瞬态），
                # 本槽快照已在窗口内，无需再追加
                self._obs_history.extend(
                    snapshot.copy() for _ in range(self.obs_history_window)
                )
                history_snapshots = list(self._obs_history)
            else:
                # 先取出 t-H..t-1 的历史区，再把当前槽快照推入窗口
                history_snapshots = list(self._obs_history)
                self._obs_history.append(snapshot)
            for snap in history_snapshots:
                obs_blocks.append(snap)  # (len(features)*K,)：lag 内按特征顺序

        obs_blocks.extend([
            time_ratio,                             # 1: 时间进度
            np.array([snr_norm, eps_norm], dtype=np.float64),  # 2: 信道质量信号
        ])

        # Task#20 P3: 可选 per-user 估计不确定度特征
        if self.obs_include_csi_quality:
            error_var = self._channel_obs.get("error_variance", np.zeros(K))
            ref = max(self.N_r * self.N_t * 0.1, 1e-8)
            est_uncertainty = np.clip(error_var / ref, 0.0, 1.0).astype(np.float64)
            obs_blocks.append(est_uncertainty)

        obs = np.concatenate(obs_blocks)

        # 分层模式: 追加 power_budget_normalized
        if self.hierarchical_enabled:
            if self.P_target is not None:
                pb_norm = self.P_target / max(self.K * self.P_k_max, 1e-8)
            else:
                pb_norm = 0.5  # 默认中间值
            obs = np.append(obs, np.float64(pb_norm))

        return obs.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境, 返回 (observation, info).

        若启用 SNR domain randomization, 每个 episode 随机采样
        snr_db ~ U[snr_min_db, snr_max_db], sigma2 = 10^(-snr_db/10).
        """
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed
            self._build_modules(seed)
            self._snr_rng = np.random.default_rng(seed)
            self._pl_rng = np.random.default_rng(seed)

        # SNR domain randomization: 每个 episode 重新采样 sigma2
        # 若调用方通过 options['snr_db'] 显式指定（如定向硬例挖掘），则优先使用该值。
        if options is not None and 'snr_db' in options:
            snr_db = float(options['snr_db'])
            snr_db = float(np.clip(snr_db, self.snr_min_db, self.snr_max_db))
            self.sigma2 = float(10.0 ** (-snr_db / 10.0))
        elif self.snr_domain_randomization:
            snr_db = float(self._snr_rng.uniform(self.snr_min_db, self.snr_max_db))
            self.sigma2 = float(10.0 ** (-snr_db / 10.0))
        else:
            self.sigma2 = float(self.config.system.sigma2)
            snr_db = float(-10.0 * np.log10(max(self.sigma2, 1e-12)))
        # 同步给 rate_calc, step() 中的速率计算依赖当前 sigma2
        self.rate_calc.sigma2 = self.sigma2
        # Task#20: 同步给 channel，使估计噪声 sigma_est 随 SNR 自适应
        self.channel.set_sigma2(self.sigma2)
        self.current_snr_db = snr_db

        # Task#9 / Task#44: 根据当前 SNR 计算自适应 λ，负载比随 SNR 动态变化
        if self.adaptive_lambda_enabled:
            sys_cfg = self.config.system
            cap = estimate_capacity(
                snr_db,
                K=self.K,
                N_r=self.N_r,
                use_path_loss=bool(getattr(sys_cfg, "use_path_loss", False)),
                cell_radius_min=float(getattr(sys_cfg, "cell_radius_min", 50.0)),
                cell_radius_max=float(getattr(sys_cfg, "cell_radius_max", 300.0)),
                path_loss_exponent=float(getattr(sys_cfg, "path_loss_exponent", 3.76)),
                path_loss_ref_distance=float(getattr(sys_cfg, "path_loss_ref_distance", 100.0)),
            )
            # Task#44: SNR-adaptive ratio 替代常数 target_load_ratio
            ratio = snr_adaptive_load_ratio(
                snr_db,
                r_min=self.load_ratio_min,
                r_max=self.load_ratio_max,
                snr_mid=self.load_ratio_snr_mid,
                steepness=self.load_ratio_steepness,
            )
            adaptive_lam = ratio * cap / max(self.K, 1)
            adaptive_lam = float(np.clip(adaptive_lam, self.lambda_min, self.lambda_max))
            self.current_lambda_rate = adaptive_lam
            self.traffic.lambda_rate = adaptive_lam
        else:
            self.current_lambda_rate = float(self.config.system.lambda_rate)
            self.traffic.lambda_rate = self.current_lambda_rate

        # v13 QoS: 分配用户类别和 per-user lambda
        if self.enable_hetero_qos:
            n_per_class = self.K // self.qos_classes
            for i in range(self.qos_classes):
                start = i * n_per_class
                end = start + n_per_class if i < self.qos_classes - 1 else self.K
                self.qos_class_ids[start:end] = i
            # per-user lambda = base_lambda * class_multiplier
            base_lambda = self.current_lambda_rate
            lambda_per_user = np.ones(self.K, dtype=np.float64) * base_lambda
            for i in range(self.qos_classes):
                mask = self.qos_class_ids == i
                lambda_per_user[mask] *= self.qos_lambda_multipliers[i]
            self.traffic.set_lambda_per_user(lambda_per_user)
            # per-user reward weights
            for i in range(self.qos_classes):
                mask = self.qos_class_ids == i
                self.qos_user_drift_weights[mask] = self.qos_drift_weights[i]
                self.qos_user_overflow_weights[mask] = self.qos_overflow_weights[i]
        else:
            self.qos_class_ids[:] = 0
            self.qos_user_drift_weights[:] = 1.0
            self.qos_user_overflow_weights[:] = 1.0

        self.t = 0
        self.step_in_episode = 0
        self._prev_power = np.zeros(self.K, dtype=np.float64)
        self._prev_queue = None
        # 时序历史窗口清空：首次 _get_observation 用 reset 时刻快照预填充
        self._obs_history.clear()
        self._reset_meta_stats()

        # 分层 PPO 追踪变量重置
        # Bug fix: 初始化为 0.5 而非 1.0，避免 Meta 首次决策前 alpha 全为 1.0（纯公共流）
        self.split_mask = np.full(self.K, 0.5, dtype=np.float32)
        self.group_ids = np.zeros(self.K, dtype=np.int64)
        self.decoding_order = None
        self._prev_rates = np.zeros(self.K, dtype=np.float64)
        self.avg_rates = np.zeros(self.K, dtype=np.float64)
        self.overflow_counts = np.zeros(self.K, dtype=np.float64)
        self.power_utilization = np.zeros(self.K, dtype=np.float64)
        self.channel_gains_norm = np.zeros(self.K, dtype=np.float64)

        # Task#13: 重新采样用户距离与路径损耗 (per-episode 大尺度慢变)
        if self.use_path_loss:
            self.user_distances = generate_user_distances(
                self.K, self.cell_radius_min, self.cell_radius_max, rng=self._pl_rng,
            )
            pl_db_abs = compute_path_loss_db(
                self.user_distances,
                pl_ref=self.path_loss_ref_db,
                pl_exp=self.path_loss_exponent,
            )
            pl_db_at_ref = float(
                self.path_loss_ref_db
                + 10.0 * self.path_loss_exponent
                * np.log10(self.path_loss_ref_distance / 1000.0)
            )
            self.path_loss_db_relative = pl_db_abs - pl_db_at_ref
            self.path_loss_amp = compute_path_loss_linear(
                self.user_distances,
                pl_ref=self.path_loss_ref_db,
                pl_exp=self.path_loss_exponent,
                ref_distance=self.path_loss_ref_distance,
            )
            self.channel.set_path_loss(self.path_loss_amp)
        else:
            self.user_distances = np.full(
                self.K, self.path_loss_ref_distance, dtype=np.float64
            )
            self.path_loss_db_relative = np.zeros(self.K, dtype=np.float64)
            self.path_loss_amp = np.ones(self.K, dtype=np.float64)
            self.channel.set_path_loss(None)

        self._channel_obs = self.channel.reset()
        self.traffic.reset()
        self.lyapunov.reset()

        obs = self._get_observation()
        info = {
            "t": self.t,
            "channel_gains": self._channel_obs["channel_gains"].copy(),
            "queue_lengths": self.traffic.queue_lengths.copy(),
            "current_V": self.lyapunov.V,
            "sigma2": self.sigma2,
            "snr_db": float(-10.0 * np.log10(max(self.sigma2, 1e-12))),
            "lambda_rate": self.current_lambda_rate,
            "user_distances": self.user_distances.copy(),
            "path_loss_db_relative": self.path_loss_db_relative.copy(),
        }
        return obs, info

    def step(self, action):
        """执行一个时隙, 返回 (observation, reward, terminated, truncated, info).

        Args:
            action: (3K,) 或 (4K,) 连续动作。
                - 3K维: [alpha, p_total, d_ratio] (向后兼容)
                - 4K维: [alpha, p_total, d_ratio, decoding_order]
                  其中 decoding_order 为K维整数优先级，值越小越先解码。
        """
        # 0. 解析 decoding_order (如果 action 为 4K 维)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] == 4 * self.K:
            continuous_action = action[:3 * self.K]
            self.decoding_order = action[3 * self.K:].astype(np.int64)
        elif action.shape[0] == 3 * self.K:
            continuous_action = action
            # 保留之前设置的 decoding_order (可能由外部设置)
        else:
            raise ValueError(
                f"action shape {action.shape} must be ({3*self.K},) or ({4*self.K},)"
            )

        # Task#11 消融：disable_sic_order 时，强制采用信道增益降序（强用户先解码）
        ab = self._ablation()
        if ab is not None and (ab.disable_sic_order or ab.disable_meta):
            gains = self._channel_obs.get("channel_gains", None)
            if gains is not None:
                # rank: 增益最大的用户 priority=0 (先解码)
                self.decoding_order = np.argsort(np.argsort(-gains)).astype(np.int64)

        # 1. 解析动作
        alpha, p_total_ratio, d_ratio = self._parse_action(continuous_action)

        # 2. 实际总功率 = p_total_ratio * P_k_max
        p_total = p_total_ratio * self.P_k_max

        # 分层模式: 按 P_target 裁剪每用户功率
        if self.hierarchical_enabled and self.P_target is not None:
            p_per_user_cap = self.P_target / max(self.K, 1)
            p_total = np.minimum(p_total, p_per_user_cap)

        # 3. 预编码 + 速率计算：根据 precoding_type 切换路径
        H_true = self._channel_obs["H_true"]
        H_hat = self._channel_obs["H_hat"]
        sic_imp = getattr(self.config.system, 'sic_imperfection', 0.0)

        if self.precoding_type == "rzf":
            # 新路径：统一 MRT 发射 + MMSE-SIC 接收
            W_c, W_p = self.beamformer.compute_precoding_v2(
                H_hat, alpha, p_total, self.split_mask
            )
            if self.csi_mode == "imperfect":
                # Task#20: 使用 channel 动态估计的 sigma_est 作为 robust error_bound
                error_bound_dyn = float(
                    self._channel_obs.get(
                        "sigma_est", float(self.config.system.epsilon_csi)
                    )
                )
                rate_result = self.rate_calc.compute_outage_rates_mmse_sic(
                    H_hat, H_true, W_c, W_p,
                    error_bound=error_bound_dyn,
                    sic_imperfection=sic_imp,
                    decoding_order=self.decoding_order,
                    group_ids=self.group_ids,
                )
            else:
                rate_result = self.rate_calc.compute_rates_mmse_sic(
                    H_hat, W_c, W_p,
                    sic_imperfection=sic_imp,
                    decoding_order=self.decoding_order,
                    group_ids=self.group_ids,
                )
        else:
            # 旧路径：正交补 MRT 发射 + matched filter 接收（向后兼容）
            # split_mask 在 _parse_action 中已作用于 alpha（alpha *= split_mask）
            W_c, W_p = self.beamformer.compute_directions(H_hat, alpha, p_total)
            if self.csi_mode == "imperfect":
                # Task#20: error_variance 使用 channel 动态返回的 per-user 值
                error_var = self._channel_obs.get(
                    "error_variance",
                    np.full(
                        self.K,
                        float(self.config.system.epsilon_csi) ** 2,
                        dtype=np.float64,
                    ),
                ).astype(np.float64)
                rate_result = self.rate_calc.compute_outage_rates(
                    H_hat, W_c, W_p, error_var,
                    outage_prob=self.outage_prob,
                    sic_imperfection=sic_imp,
                    decoding_order=self.decoding_order,
                    group_ids=self.group_ids,
                )
            else:
                rate_result = self.rate_calc.compute_rates(
                    H_hat, W_c, W_p,
                    sic_imperfection=sic_imp,
                    decoding_order=self.decoding_order,
                    group_ids=self.group_ids,
                )

        usable_rates = rate_result["R_total"]
        outage_flags = np.zeros(self.K, dtype=np.float64)

        # 5. 实际发送比特 D_k = d_ratio * min(R*tau, Q_k[t-1])
        queue_before = self.traffic.queue_lengths.copy()
        max_sendable = np.maximum(np.minimum(usable_rates * self.tau, queue_before), 0.0)
        D_k = d_ratio * max_sendable

        # 6. 队列更新
        traffic_out = self.traffic.step(D_k)
        served = traffic_out["served"]
        overflow = traffic_out["overflow"]
        arrivals = traffic_out["arrivals"]

        # 7. Lyapunov 奖励 (使用服务前队列长度作为权重, 自适应V在内部计算)
        power_per_user = p_total  # p_c + p_p = alpha*p + (1-alpha)*p = p_total
        # R_sum: 所有用户的可达速率之和 (bits/Hz/slot)
        R_sum_val = float(rate_result["R_sum"])  # 保留用于info监控
        # 外层 Dinkelbach：将当前 eta 注入 Lyapunov，保持 step 级奖励为 R - eta*P
        self.lyapunov.dinkelbach_q = self.dinkelbach_eta
        rew_info = self.lyapunov.compute_reward(
            queue_lengths=queue_before,
            served_bits=served,
            power_consumed=power_per_user,
            overflow=overflow,
            prev_queue=getattr(self, '_prev_queue', None),
            outage_flags=outage_flags if self.csi_mode == "imperfect" else None,
            sigma2=self.sigma2,
            R_sum=None,  # v16: EE分子改用served_bits，不再传R_sum
            P_target=self.P_target if self.hierarchical_enabled else None,
            user_drift_weights=self.qos_user_drift_weights if self.enable_hetero_qos else None,
            user_overflow_weights=self.qos_user_overflow_weights if self.enable_hetero_qos else None,
            delivery_running=self.traffic.get_delivery_ratio(),
            arrivals=arrivals,
        )
        self._prev_queue = queue_before.copy()

        # 目标反转后 EE 不再是主奖励项，直接使用 compute_reward 的 reward。
        reward = float(rew_info["reward"])
        if not np.isfinite(reward):
            reward = -1e3  # 数值兜底, 避免 NaN/Inf 污染训练

        # 8. 信道演化
        self._channel_obs = self.channel.step()

        # 9. 更新缓存与时隙计数
        self._prev_power = power_per_user.copy()
        self._prev_rates = usable_rates.copy()
        # 指数移动平均速率 (EMA, beta=0.1)
        ema_beta = 0.1
        self.avg_rates = (1 - ema_beta) * self.avg_rates + ema_beta * usable_rates
        # 累计溢出次数 (per-user)
        self.overflow_counts += (overflow > 0).astype(np.float64)
        # 功率利用率 = p_k / P_k_max
        self.power_utilization = power_per_user / max(self.P_k_max, 1e-8)
        self.t += 1

        # 10. 终止判定
        terminated = False
        truncated = bool(self.t >= self.T)

        # 11. 构造观测与 info
        obs = self._get_observation()

        total_power = float(np.sum(power_per_user))
        mean_queue = float(np.mean(self.traffic.queue_lengths))
        total_arrived = float(np.sum(arrivals))
        overflow_rate = (
            float(np.sum(overflow) / total_arrived) if total_arrived > 1e-9 else 0.0
        )

        # 分层模式统计累计
        self.step_in_episode += 1
        if self.hierarchical_enabled:
            self._accumulate_meta_stats(
                total_power=total_power,
                mean_queue=mean_queue,
                delivery_ratio=self.traffic.get_delivery_ratio(),
                overflow_rate=overflow_rate,
                R_sum=float(rate_result["R_sum"]),
            )

        info = {
            "t": self.t,
            "total_power": total_power,
            "P_consume": rew_info.get("P_consume", total_power),
            "mean_queue": mean_queue,
            "delivery_ratio": self.traffic.get_delivery_ratio(),
            "R_sum": float(rate_result["R_sum"]),
            "R_total": rate_result["R_total"].copy(),
            "overflow_rate": overflow_rate,
            "outage_count": int(np.sum(outage_flags)) if self.csi_mode == "imperfect" else 0,
            "mean_alpha": float(np.mean(alpha)),
            "served": served.copy(),
            "arrivals": arrivals.copy(),
            "drift_signal": rew_info["drift_signal"],
            "ee_bonus": rew_info["ee_bonus"],
            "deliv_utility": rew_info.get("deliv_utility", 0.0),
            "ee_floor_penalty": rew_info.get("ee_floor_penalty", 0.0),
            "overflow_penalty": rew_info["overflow_penalty"],
            "queue_lengths": self.traffic.queue_lengths.copy(),
            "current_V": rew_info["current_V"],
            "EE": rew_info.get("EE", 0.0),
            "dinkelbach_q": rew_info.get("dinkelbach_q", 0.0),
            "avg_queue_ratio": rew_info.get("avg_queue_ratio", 0.0),
            "queue_weight": rew_info.get("queue_weight", 0.0),
            "lambda_rate": self.current_lambda_rate,
            "snr_db": self.current_snr_db,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Hierarchical interface
    # ------------------------------------------------------------------ #
    def _ablation(self):
        """获取 ablation 配置（不存在时返回 None）。"""
        return getattr(self.config, 'ablation', None)

    def set_split_mask(self, split_mask: np.ndarray) -> None:
        """设置慢时间尺度分裂决策（由 Meta-Controller 调用）。

        Args:
            split_mask: shape (K,), binary. 1=该用户启用公共+私有流分裂
        """
        sm = np.array(split_mask, dtype=np.float32)
        ab = self._ablation()
        if ab is not None and (ab.disable_split or ab.disable_meta):
            sm = np.zeros(self.K, dtype=np.float32)
        self.split_mask = sm

    def set_group_ids(self, group_ids: np.ndarray) -> None:
        """设置慢时间尺度用户分组（由 Meta-Controller 调用）。

        Args:
            group_ids: shape (K,), int. 每用户所属组 {0,...,G-1}
        """
        gids = np.array(group_ids, dtype=np.int64)
        ab = self._ablation()
        if ab is not None and (ab.disable_group or ab.disable_meta):
            gids = np.zeros(self.K, dtype=np.int64)
        self.group_ids = gids

    def set_power_target(self, P_target: float) -> None:
        self.P_target = float(P_target)

    def set_lambda_bucket(self, lambda_val: float) -> None:
        """注入当前 SNR 桶的 Lagrangian λ 值（用于 State-Augmented 观测）。"""
        self.lambda_bucket_val = float(lambda_val)

    def get_user_features(self) -> np.ndarray:
        """返回 Meta-Controller 所需的 per-user 特征矩阵。

        Returns:
            features: shape (K, d_user), d_user=9 (v12) 或 10 (v13 QoS)
            包含: channel_gain, queue_ratio, prev_split, avg_rate, overflow_count,
                  power_util, group_id_norm, snr_norm, capacity_norm[, qos_class_norm]
        """
        K = self.K
        Q_max = self.Q_max
        n_groups = getattr(
            getattr(self.config, 'hierarchical', None), 'n_groups', 4
        )

        d_user = 12 if self.enable_hetero_qos else 9
        features = np.zeros((K, d_user), dtype=np.float32)
        features[:, 0] = self.channel_gains_norm                        # 归一化信道增益
        features[:, 1] = self.traffic.queue_lengths / max(Q_max, 1e-8)  # 队列比率
        features[:, 2] = self.split_mask                                # 上次分裂决策
        features[:, 3] = self.avg_rates                                 # 平均速率 (EMA)
        features[:, 4] = self.overflow_counts / max(self.t, 1)          # 溢出频率
        features[:, 5] = self.power_utilization                         # 功率利用率
        features[:, 6] = self.group_ids / max(n_groups - 1, 1)          # 归一化组ID

        # dim 7: snr_norm - 归一化 SNR（全局值，广播到所有用户）
        snr_norm = float(np.clip((self.current_snr_db - 0.0) / 30.0, 0.0, 1.0))
        features[:, 7] = snr_norm

        # dim 8: capacity_norm - 归一化系统容量（反映当前 SNR 下的负载能力）
        cap = estimate_capacity(self.current_snr_db, K=self.K, N_r=self.N_r)
        cap_norm = float(np.clip(cap / 100.0, 0.0, 1.0))
        features[:, 8] = cap_norm

        # v14 QoS: dim 9-11 — 类别 + lambda负载 + 优先级权重
        if self.enable_hetero_qos:
            max_lambda_mult = max(self.qos_lambda_multipliers)
            max_drift_wt = max(self.qos_drift_weights)
            # dim 9: QoS类别归一化（保留原有）
            features[:, 9] = self.qos_class_ids / max(self.qos_classes - 1, 1)
            # dim 10: lambda负载归一化
            for i in range(self.qos_classes):
                mask = self.qos_class_ids == i
                features[mask, 10] = self.qos_lambda_multipliers[i] / max(max_lambda_mult, 1e-8)
            # dim 11: 优先级权重归一化
            for i in range(self.qos_classes):
                mask = self.qos_class_ids == i
                features[mask, 11] = self.qos_drift_weights[i] / max(max_drift_wt, 1e-8)

        return features

    def get_meta_stats(self) -> dict:
        stats = {}
        n = max(self._meta_stats_count, 1)
        for k, v in self._meta_stats_accum.items():
            stats[k] = v / n
        # EE (Task#15: P_consume = P_tx/η + P_circuit)
        avg_power = stats.get('avg_power_ratio', 0.5) * self.K * self.P_k_max
        avg_r_sum = stats.get('avg_R_sum', 0.0)
        sys_cfg = self.config.system
        eta = float(getattr(sys_cfg, 'pa_efficiency', 0.35))
        Pc = float(getattr(sys_cfg, 'circuit_power', 5.0))
        avg_p_consume = avg_power / max(eta, 1e-3) + Pc
        stats['avg_EE'] = avg_r_sum / max(avg_p_consume, 0.1)
        self._reset_meta_stats()
        return stats

    def _reset_meta_stats(self) -> None:
        self._meta_stats_accum = {
            'avg_queue_ratio': 0.0,
            'avg_power_ratio': 0.0,
            'avg_delivery': 0.0,
            'avg_overflow': 0.0,
            'avg_R_sum': 0.0,
        }
        self._meta_stats_count = 0

    def _accumulate_meta_stats(self, total_power, mean_queue, delivery_ratio,
                               overflow_rate, R_sum) -> None:
        self._meta_stats_accum['avg_queue_ratio'] += mean_queue / max(self.Q_max, 1e-8)
        self._meta_stats_accum['avg_power_ratio'] += total_power / max(self.K * self.P_k_max, 1e-8)
        self._meta_stats_accum['avg_delivery'] += delivery_ratio
        self._meta_stats_accum['avg_overflow'] += overflow_rate
        self._meta_stats_accum['avg_R_sum'] += R_sum
        self._meta_stats_count += 1

    @property
    def obs_dim(self) -> int:
        return self._obs_dim


