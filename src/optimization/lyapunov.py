"""Lyapunov漂移加惩罚框架（v13s: LDPTRLQ 奖励）

核心奖励项（LDPTRLQ）：
  reward = scale * (V * ee_bonus - q_weight * ΣQ²/Q_max² - overflow_penalty)

与旧版 drift 信号（仅惩罚队列增长 ΣQ·ΔQ）不同，
LDPTRLQ 直接惩罚队列存在性 ΣQ²，从根本上压低稳态队列长度。
"""
from typing import Dict, Optional

import numpy as np


def compute_adaptive_V(
    queue_states: np.ndarray,
    Q_max: float,
    V_min: float = 5.0,
    V_max: float = 100.0,
    exponent: float = 2.0,
) -> float:
    """根据当前队列状态计算自适应V值。

    队列低时V大（更省电），队列高时V小（优先清队列）。
        V(t) = V_max * (1 - avg_queue_ratio)^exponent + V_min
    """
    Q = np.asarray(queue_states, dtype=np.float64).reshape(-1)
    avg_ratio = float(np.mean(Q)) / max(Q_max, 1e-8)
    avg_ratio = float(np.clip(avg_ratio, 0.0, 1.0))
    V = V_max * (1.0 - avg_ratio) ** exponent + V_min
    return float(V)


class LyapunovController:
    """Lyapunov 控制器（精简3项奖励版）。

    主要参数:
        blend_alpha: 队列紧迫度指数, w = (Q_avg/Q_max) ** blend_alpha
                     >1 使低队列时快速切换到 EE 主导模式
        ee_weight:   EE 激励权重, ee_bonus = ee_weight * R_sum / max(P_total, P_floor)
        w_overflow:  溢出硬约束惩罚权重
        reward_scale: 全局奖励缩放因子
    """

    def __init__(
        self,
        K: int,
        V: float = 3.0,
        V_min: float = 1.0,
        V_max: float = 10.0,
        V_exponent: float = 1.5,
        adaptive_V: bool = True,
        Q_max: float = 100.0,
        use_virtual_queue: bool = True,
        vq_step_size: float = 0.1,
        reward_scale: float = 0.01,
        w_overflow: float = 15.0,
        ee_weight: float = 8.0,
        ee_power_floor: float = 1.0,
        blend_alpha: float = 2.5,
        # 以下参数保留接口兼容性但不再使用
        w_outage: float = 0.0,
        snr_gamma: float = 0.0,
        snr_ref_db: float = 30.0,
        v_eff_cap_factor: float = 3.0,
        snr_floor: float = 1.0,
        snr_power_exponent: float = 1.0,
        P_min_floor: float = 0.0,
        ee_mode: bool = True,
        # P0 修复参数
        drift_weight: float = 35.0,           # drift 信号放大系数（与 config 默认值对齐）
        overflow_warn_ratio: float = 0.6,     # 早期溢出阈值（与 config 默认值对齐）
        w_early_overflow: float = 5.0,        # 早期溢出惩罚权重
        # Task#15: PA + 电路功耗模型
        pa_efficiency: float = 0.35,          # η: PA 效率（Class-B）
        circuit_power: float = 5.0,           # P_circuit: 静态电路功耗 (W)
        # v22: Lyapunov-DRL 权衡参数
        lyapunov_V: float = 5.0,              # V 越大越偏 EE，V 越小越保守保队列
        # v13s: LDPTRLQ 队列二次惩罚权重
        ldptrlq_q_weight: float = 20.0,
        # CRPO 风格可行性门控（delivery 违约时抑制 EE 项）
        ee_gate_enable: bool = True,
        ee_gate_threshold: float = 0.98,
        ee_gate_band: float = 0.15,
        ee_gate_min: float = 0.05,
        # backpressure / max-weight drift reward 权重
        w_backpressure: float = 0.0,
        # 目标反转：交付凹效用主目标 + EE 单侧下界软约束
        w_deliv: float = 3.0,
        deliv_ref: float = 2.0,
        ee_floor: float = 30.0,
        w_ee_floor: float = 0.5,
        # DGMW: Delivery-Gated Max-Weight Pure Mode
        dgmw_enable: bool = False,
        dgmw_panic_threshold: float = 0.55,
        dgmw_recovery_threshold: float = 0.85,
        dgmw_mw_weight: float = 100.0,
        # LDPS: Lyapunov Drift Potential Shaping (Ng 1999 PBRS)
        ldps_enable: bool = False,
        ldps_eta: float = 30.0,
        ldps_gamma: float = 0.99,
        # Dinkelbach 变换：EE 比率 R/P → 线性 R - q*·P
        use_dinkelbach: bool = False,
        dinkelbach_q_init: float = 1.0,
        dinkelbach_decay: float = 0.99,
        dinkelbach_q_min: float = 0.01,
        dinkelbach_q_max: float = 50.0,
        dinkelbach_outer_update: bool = False,
    ) -> None:
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if V_min <= 0 or V_max < V_min:
            raise ValueError(f"Invalid V range: [{V_min}, {V_max}]")

        self.K = int(K)
        self.V_fixed = float(V)
        self.V_min = float(V_min)
        self.V_max = float(V_max)
        self.V_exponent = float(V_exponent)
        self.adaptive_V = bool(adaptive_V)
        self.Q_max = float(Q_max)

        self.V = float(np.clip(V, self.V_min, self.V_max + self.V_min))

        self.use_virtual_queue = bool(use_virtual_queue)
        self.vq_step_size = float(vq_step_size)
        self.reward_scale = float(reward_scale)
        self.w_overflow = float(w_overflow)

        # EE 激励参数
        self.ee_weight = float(ee_weight)
        self.ee_power_floor = float(max(ee_power_floor, 0.01))

        # 分层混合权重指数
        self.blend_alpha = float(max(blend_alpha, 1e-3))

        # P0 修复参数
        self.drift_weight = float(max(drift_weight, 1e-6))
        self.overflow_warn_ratio = float(np.clip(overflow_warn_ratio, 0.0, 1.0))
        self.w_early_overflow = float(max(w_early_overflow, 0.0))

        # Task#15: PA + 电路功耗模型
        # P_consume = P_tx / pa_efficiency + circuit_power
        # 即使 P_tx → 0，P_consume 仍为 circuit_power，EE 上限有限
        self.pa_efficiency = float(max(pa_efficiency, 1e-3))
        self.circuit_power = float(max(circuit_power, 0.0))

        # v22: Lyapunov-DRL 固定权衡参数
        self.lyapunov_V = float(lyapunov_V)

        # v13s: LDPTRLQ 队列二次惩罚权重
        self.ldptrlq_q_weight = float(max(ldptrlq_q_weight, 0.0))

        # CRPO 风格可行性门控参数
        self.ee_gate_enable = bool(ee_gate_enable)
        self.ee_gate_threshold = float(ee_gate_threshold)
        self.ee_gate_band = float(max(ee_gate_band, 1e-6))
        self.ee_gate_min = float(np.clip(ee_gate_min, 0.0, 1.0))

        # backpressure / max-weight drift reward 权重
        self.w_backpressure = float(max(w_backpressure, 0.0))

        # 目标反转：交付凹效用主目标 + EE 单侧下界软约束
        self.w_deliv = float(max(w_deliv, 0.0))
        self.deliv_ref = float(max(deliv_ref, 1e-8))
        self.ee_floor = float(ee_floor)
        self.w_ee_floor = float(max(w_ee_floor, 0.0))

        # 兼容性别名
        self.scale = self.reward_scale
        self.V_effective: float = self.V

        self._virtual_queue = np.zeros(self.K, dtype=np.float64)

        # per-user Neely 交付欠账虚拟队列: Z_k = max(Z_k + β·A_k - D_k, 0)
        # 用欠账加权奖励定向抬升饿死尾部用户的 delivery、压低队列
        # 已验证 deficit-queue 无效，关闭之
        self.enable_deficit_queue = False
        self._deficit = np.zeros(self.K, dtype=np.float64)
        self.deficit_beta = 0.98
        self.w_deficit = 6.0
        self.deficit_scale = self.Q_max

        # 比例公平(PF)凹效用重塑 delivery 目标:
        # 以累计平均交付率倒数加权当前交付, 放大尾部饥饿用户的边际奖励
        # (≈∇Σlog R̄_k), 攻克 delivery 停滞
        self.enable_fair_utility = True
        self.w_fair = 6.0
        self.fair_eps = 1.0
        self.fair_ema = 0.05
        self._avg_served = np.full(self.K, self.fair_eps, dtype=np.float64)

        # ---- CRPO 硬双模切换 (Xu et al., ICML 2021) + Max-Weight 吞吐 ----
        # 可行 (delivery >= thr): 最大化 EE (V*ee_bonus)，利用 EE 裕度省电；
        # 不可行 (delivery < thr): 切换为纯线性 backlog 加权吞吐 Σ(Q_k/Q_max·w_k)·D_k，
        #   线性边际价值不饱和，迫使策略抬功率清空最大积压 → 直接抬 delivery、降 queue。
        # 迟滞带 crpo_band 防止在阈值附近抖动；两模式均保留队列二次惩罚与溢出惩罚。
        self.crpo_enable = True
        self.crpo_threshold = float(self.ee_gate_threshold)   # 复用 delivery 可行阈值
        self.crpo_band = 0.02                                 # 迟滞带宽（独立小带，避免过宽致可行态不可达）
        self.crpo_bp_weight = 14.0    # 不可行模式线性 max-weight 吞吐权重
        self.crpo_ee_weight = 1.0     # 可行模式 EE 项相对权重 (乘在 V*ee_bonus 上)
        self._crpo_feasible = False   # 迟滞状态：上一时刻是否判为可行

        # ---- Neely 聚合交付欠账虚拟队列 (drift-plus-penalty, 稠密自适应对偶) ----
        self.enable_delivery_vq = True
        self.dvq_target = 0.95            # 交付流量目标比例(A 的目标交付分数)
        self.dvq_y_scale = 300.0          # Y 归一化尺度
        self.dvq_y_max = 5000.0           # Y 上界防爆
        self.w_vq_throughput = 0.08       # Y 加权吞吐项系数
        self.dvq_backlog_mix = 0.3        # 吞吐项中 backlog 加权占比(其余为均匀 ΣD_k)
        self._Y = 0.0

        # ---- 能量 Lyapunov 虚拟队列 X (drift-plus-penalty 强制平均功率约束) ----
        # X 随功率超预算无界增长, 作为功率的自适应对偶价格; 把EE短板(功率>SDMA)拉回
        self.enable_power_vq = True
        self.power_budget = 8.0      # 目标平均总功率(W), 设在SDMA(8.82)之下以使EE_ratio>1
        self.pvq_eta = 0.1           # 虚拟队列步长
        self.pvq_scale = 100.0       # X 归一化尺度
        self.pvq_max = 1000.0        # X 上界防爆
        self.w_power_vq = 1.0        # 功率对偶惩罚权重(温和斜率, 避免与q*·P软项双重惩罚)
        self._X = 0.0

        # ---- DGLCR: Delivery-Gated Lyapunov Composite Reward ----
        # 核心创新 (arXiv:2506.04291 + CRPO): sigmoid 门控 EE 奖励项,
        # delivery 低于阈值时关闭 EE 引力(∂EE/∂P<0)并放大 throughput 推力,
        # delivery 达标后平滑恢复 EE+功率约束, 解决低功率→低 delivery 陷阱。
        self.ee_gate_kappa = 40.0       # sigmoid 陡度: 越大门控越硬
        self.ee_gate_mid = 0.935       # delivery 门控中点 (0.90~0.97 过渡区)
        self.w_tp_dglcr = 3.0          # 基吞吐量权重 (max-weight backlog 加权)
        self.tp_boost_max = 4.0        # delivery→0 时 throughput 额外放大倍数
        self.w_y_tp = 0.3              # Y 虚拟队列吞吐量加成系数

        # DGPS: Delivery-Gated Penalty Suppression ----
        # delivery 低于阈值时用 steep sigmoid 抑制 q/overflow/power_vq 三大惩罚,
        # 释放纯吞吐量梯度高速公路; delivery 恢复后自动重开惩罚, 无参过渡回完整 DGLCR。
        self.penalty_gate_enable = True   # 是否启用惩罚门控
        self.penalty_gate_kappa = 50.0    # 惩罚门控 sigmoid 陡度（比 EE gate 更陡，阈值附近快速切换）
        self.penalty_gate_mid = 0.96      # 惩罚门控中点（略高于 EE gate 的 0.935，保证惩罚先于 EE 恢复）
        self.penalty_gate_min = 0.0       # 门控下限（delivery 极低时惩罚零抑制）

        # ---- DGMW: Delivery-Gated Max-Weight Pure Mode ----
        # 当 running delivery 低于 panic_threshold 时，硬切换到纯 max-weight
        # 吞吐量最大化模式（无 scale、无 EE、无任何惩罚稀释），创建压倒性梯度
        # 逃离低功率吸引子。迟滞带：进入 panic 后需 delivery 持续高于
        # recovery_threshold 才退出。
        self.dgmw_enable = bool(dgmw_enable)
        self.dgmw_panic_threshold = float(dgmw_panic_threshold)
        self.dgmw_recovery_threshold = float(dgmw_recovery_threshold)
        self.dgmw_mw_weight = float(dgmw_mw_weight)
        self._dgmw_panic = False    # 迟滞状态：当前是否处于 panic 模式

        # ---- LDPS: Lyapunov Drift Potential Shaping (PBRS, Ng 1999) ----
        # F(s) = -ldps_eta·ΣQ²/Q_max²；整形项 η·(Σprev_Q² - γ·ΣQ²)/Q_max²
        # 策略不变（PBRS 定理），提供逐步稠密队列排水信用，穿透优势标准化
        self.ldps_enable = bool(ldps_enable)
        self.ldps_eta = float(max(ldps_eta, 0.0))
        self.ldps_gamma = float(np.clip(ldps_gamma, 0.0, 1.0))

        # ---- Dinkelbach 变换：EE 比率 R/P → 线性 R - q*·P ----
        # 消除 P→0 奇点吸引子。q* 通过指数加权累积比率在线跟踪当前 EE 水平。
        self.use_dinkelbach = bool(use_dinkelbach)
        self.dinkelbach_outer_update = bool(dinkelbach_outer_update)
        self.dinkelbach_q = float(dinkelbach_q_init)
        self.dinkelbach_decay = float(np.clip(dinkelbach_decay, 0.0, 1.0))
        self.dinkelbach_q_min = float(dinkelbach_q_min)
        self.dinkelbach_q_max = float(dinkelbach_q_max)
        self._dkl_served_ema = 0.0
        self._dkl_power_ema = 0.0

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Episode 开始：重置虚拟队列。"""
        self._virtual_queue = np.zeros(self.K, dtype=np.float64)
        self._deficit = np.zeros(self.K, dtype=np.float64)
        self._avg_served = np.full(self.K, self.fair_eps, dtype=np.float64)
        self._crpo_feasible = False
        self._Y = 0.0
        self._X = 0.0
        self._dgmw_panic = False

    # ------------------------------------------------------------------ #
    # Reward computation (精简3项)
    # ------------------------------------------------------------------ #
    def compute_reward(
        self,
        queue_lengths: np.ndarray,
        served_bits: np.ndarray,
        power_consumed: np.ndarray,
        overflow: np.ndarray,
        prev_queue: Optional[np.ndarray] = None,
        # 以下参数保留接口兼容性
        outage_flags: Optional[np.ndarray] = None,
        sigma2: Optional[float] = None,
        R_sum: Optional[float] = None,
        P_target: Optional[float] = None,
        # v13 QoS: per-user 加权
        user_drift_weights: Optional[np.ndarray] = None,
        user_overflow_weights: Optional[np.ndarray] = None,
        # CRPO 可行性门控：本 episode 累计 running delivery ratio
        delivery_running: Optional[float] = None,
        # per-user 到达量, 用于更新 Neely 交付欠账虚拟队列 Z_k
        arrivals: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """计算 LDPTRLQ 奖励。

        reward = scale * (V * ee_bonus - q_weight * ΣQ²/Q_max² - overflow_penalty)

        与旧版 drift 信号（-ΣQ·ΔQ/Q_max²）不同，直接惩罚 ΣQ²/Q_max²，
        使稳态队列长度被压到最低。
        """
        power_vq_penalty = 0.0
        penalty_gate = 1.0  # DGPS 门控默认值（非 DGLCR 分支不抑制惩罚）
        Q = np.asarray(queue_lengths, dtype=np.float64).reshape(-1)
        D = np.asarray(served_bits, dtype=np.float64).reshape(-1)
        P = np.asarray(power_consumed, dtype=np.float64).reshape(-1)
        OV = np.asarray(overflow, dtype=np.float64).reshape(-1)

        if not (Q.shape[0] == D.shape[0] == P.shape[0] == OV.shape[0] == self.K):
            raise ValueError(
                f"All inputs must have shape (K={self.K},); got "
                f"Q={Q.shape}, D={D.shape}, P={P.shape}, OV={OV.shape}"
            )

        # LDPS 势函数整形: F(s) = -η·ΣQ²/Q_max², shaping = η·(Σprev_Q² - γ·ΣQ²)/Q_max²
        # PBRS 定理保证最优策略不变；逐步稠密的队列排水差分信号改善信用分配
        ldps_shaping = 0.0
        if self.ldps_enable and prev_queue is not None:
            prev_Q = np.asarray(prev_queue, dtype=np.float64).reshape(-1)
            ldps_shaping = float(
                self.ldps_eta
                * (np.sum(prev_Q ** 2) - self.ldps_gamma * np.sum(Q ** 2))
                / (self.Q_max ** 2)
            )

        # ------------------------------------------------------------ #
        # 1. 能量效率 EE (主优化目标)
        # Task#15: 引入现实功耗模型 P_consume = P_tx/η + P_circuit
        # 避免 PPO 学到“几乎关机”的极低功率捷径（P→0 时 EE→∞）
        # ------------------------------------------------------------ #
        total_power = float(np.sum(P))
        # v16: EE 分子使用实际交付数据量(served_bits)，而非物理层和速率(R_sum)
        # 这避免了空队列时虚高EE的激励偏差
        served_total = float(np.sum(D))  # D = served_bits

        # 实际消耗功率：发射功率 / PA 效率 + 静态电路功耗
        P_consume = total_power / self.pa_efficiency + self.circuit_power
        P_eff = max(P_consume, self.ee_power_floor)
        if self.use_dinkelbach:
            if not self.dinkelbach_outer_update:
                # Dinkelbach 在线 q* 更新：指数加权累积比率（~100 步窗口）
                beta = 1.0 - self.dinkelbach_decay
                self._dkl_served_ema = self.dinkelbach_decay * self._dkl_served_ema + beta * served_total
                self._dkl_power_ema = self.dinkelbach_decay * self._dkl_power_ema + beta * P_eff
                q_new = self._dkl_served_ema / max(self._dkl_power_ema, 1e-8)
                self.dinkelbach_q = float(np.clip(q_new, self.dinkelbach_q_min, self.dinkelbach_q_max))
            # 线性目标 R - q*·P：梯度 dR/dP - q* 在低功率为正（log 陡峭段），推高功率
            dkl_value = served_total - self.dinkelbach_q * P_eff
            ee_value = served_total / P_eff  # 保留比率用于日志/评估兼容
            ee_bonus = float(self.ee_weight * dkl_value)
        else:
            ee_value = served_total / P_eff
            ee_bonus = float(self.ee_weight * ee_value)

        # 目标反转：交付凹效用（主目标）——凹性促使功率摊向尾部饥饿用户，直接最大化吞吐
        D_clipped = np.maximum(D, 0.0)
        deliv_utility = float(self.w_deliv * np.sum(np.sqrt(D_clipped / max(self.deliv_ref, 1e-8))))
        # EE 单侧下界软约束（目标反转：EE 由最大化目标降为下界约束）
        ee_floor_penalty = float(self.w_ee_floor * max(0.0, self.ee_floor - ee_value))

        # ------------------------------------------------------------ #
        # 2. LDPTRLQ 队列二次惩罚 (v13s)
        # 直接惩罚 ΣQ²/Q_max²，而非旧版 drift 信号 -ΣQ·ΔQ/Q_max²
        # 这从根本上使策略有动力将 Q 压到 0，而非仅减缓增长
        # ------------------------------------------------------------ #
        q_sq_sum = float(np.sum(Q ** 2))
        q_penalty = self.ldptrlq_q_weight * q_sq_sum / (self.Q_max ** 2)
        # backpressure / max-weight drift reward:
        # 按队列积压加权实际服务量，将调度导向被饿死用户以抬升尾部 delivery、降队列
        backlog_w = Q / max(self.Q_max, 1e-8)
        if user_drift_weights is not None:
            backlog_w = backlog_w * np.asarray(user_drift_weights, dtype=np.float64).reshape(-1)
        backpressure = float(np.sum(backlog_w * D))
        bp_reward = float(self.w_backpressure * backpressure)
        # 保留 drift_signal 用于日志（向后兼容）
        if prev_queue is not None:
            prev_Q = np.asarray(prev_queue, dtype=np.float64).reshape(-1)
            delta_Q = Q - prev_Q
        else:
            delta_Q = -D
        drift_term = float(np.sum(Q * delta_Q)) / (self.Q_max ** 2)
        if user_drift_weights is not None:
            drift_term = float(np.sum(user_drift_weights * Q * delta_Q)) / (self.Q_max ** 2)
        drift_signal = -drift_term * self.drift_weight  # 仅用于日志

        # ------------------------------------------------------------ #
        # 3. 溢出惩罚 (硬约束 + 早期渐进警告)
        # ------------------------------------------------------------ #
        # 硬约束：实际溢出（队列已满还在加包）
        # v13 QoS: per-user overflow 加权
        if user_overflow_weights is not None:
            overflow_penalty = float(self.w_overflow * np.sum(user_overflow_weights * OV))
        else:
            overflow_penalty = float(self.w_overflow * np.sum(OV))
        # P0 修复：早期渐进警告（队列接近满时就开始平滑惩罚，避免阶跃突变）
        # early_pen = w_early × Σ_k ReLU(Q_k/Q_max - threshold)²
        if self.w_early_overflow > 0.0 and self.overflow_warn_ratio < 1.0:
            q_ratio = Q / max(self.Q_max, 1e-8)
            excess = np.maximum(q_ratio - self.overflow_warn_ratio, 0.0)
            early_overflow = float(self.w_early_overflow * np.sum(excess ** 2))
            overflow_penalty += early_overflow
        else:
            early_overflow = 0.0

        # ------------------------------------------------------------ #
        # 3.5 per-user Neely 交付欠账虚拟队列
        # deficit_reward 用当前欠账 Z_k 加权实际交付 D_k, 定向抬升饿死尾部
        # 用户的 delivery, 随后更新 Z_k = max(Z_k + β·A_k - D_k, 0)
        # ------------------------------------------------------------ #
        if self.enable_deficit_queue and arrivals is not None:
            A = np.asarray(arrivals, dtype=np.float64).reshape(-1)
            deficit_reward = float(
                self.w_deficit * np.sum((self._deficit / max(self.deficit_scale, 1e-8)) * D)
            )
            self._deficit = np.maximum(self._deficit + self.deficit_beta * A - D, 0.0)
        else:
            deficit_reward = 0.0

        # ------------------------------------------------------------ #
        # 3.6 比例公平(PF)凹效用重塑 delivery
        # 以累计平均交付率倒数加权当前交付 D_k, 放大尾部饥饿用户的边际奖励,
        # 等价于 ∇Σlog R̄_k 的一阶近似, 攻克 delivery 停滞。
        # 随后 EMA 更新累计平均交付率 R̄_k。
        # ------------------------------------------------------------ #
        if self.enable_fair_utility:
            pf_reward = float(self.w_fair * np.sum(D / (self._avg_served + self.fair_eps)))
            self._avg_served = (1.0 - self.fair_ema) * self._avg_served + self.fair_ema * D
        else:
            pf_reward = 0.0

        # ------------------------------------------------------------ #
        # 4. 最终奖励（目标反转）:
        #   reward = scale * (deliv_utility - q_penalty - overflow_penalty
        #                     - ee_floor_penalty + bp_reward + deficit_reward + pf_reward)
        # EE 不再作为被 V 加权的最大化项进入 reward，仅通过 ee_floor_penalty 作为下界约束出现。
        # 保留 ee_gate 计算与返回（向后兼容日志），但令 ee_gate=1.0 且不再乘入 reward。
        # ------------------------------------------------------------ #
        if self.ee_gate_enable and delivery_running is not None:
            d_low = self.ee_gate_threshold - self.ee_gate_band
            ramp = (float(delivery_running) - d_low) / max(self.ee_gate_threshold - d_low, 1e-8)
            _ = float(np.clip(ramp, self.ee_gate_min, 1.0))  # 仅计算，不再使用
        ee_gate = 1.0

        # ------------------------------------------------------------ #
        # 4. DGLCR: Delivery-Gated Lyapunov Composite Reward
        #   核心创新: sigmoid 门控 EE 项, 根治 ∂EE/∂P<0 导致的低功率陷阱。
        #   delivery 低 → ee_gate≈0 (关闭 EE 引力) + tp_boost 放大 throughput 推力
        #   delivery 高 → ee_gate≈1 (恢复 EE+功率约束) + 正常 throughput
        #   Y 虚拟队列提供自适应对偶变量, 累积欠交付→自动抬升 throughput 权重。
        # ------------------------------------------------------------ #
        if self.enable_delivery_vq and delivery_running is not None:
            dr = float(delivery_running)

            # ---- DGMW Panic Mode: 硬切纯 max-weight 吞吐量 ----
            # delivery 极低时（<panic_threshold），所有混合奖励机制均失效：
            # ee_gate≈0、penalty_gate≈0，但 reward_scale=0.01 压缩后吞吐量
            # 梯度仅 ~1.26/step，不足以推动策略逃离低功率吸引子。
            # DGMW 提供「未稀释的纯吞吐量梯度信道」：无 scale、无 EE、
            # 无惩罚，仅 max-weight 吞吐量，权重 100×，创建压倒性梯度信号。
            # 迟滞带：panic→recovery 需 delivery 持续达标方退出，防抖。
            if self.dgmw_enable:
                if not self._dgmw_panic and dr < self.dgmw_panic_threshold:
                    self._dgmw_panic = True
                elif self._dgmw_panic and dr >= self.dgmw_recovery_threshold:
                    self._dgmw_panic = False

                if self._dgmw_panic:
                    # 纯 max-weight 吞吐量：Σ backlog_w·D_k，权重 dgmw_mw_weight
                    mw_bl = Q / max(self.Q_max, 1e-8)
                    if user_drift_weights is not None:
                        mw_bl = mw_bl * np.asarray(user_drift_weights, dtype=np.float64).reshape(-1)
                    mw_tp = float(np.sum(mw_bl * D))

                    # 仍更新 Y/X 虚拟队列（保持状态连续性），但不影响当前 reward
                    if arrivals is not None:
                        A_arr = np.asarray(arrivals, dtype=np.float64).reshape(-1)
                        self._Y = max(self._Y + (self.dvq_target * float(np.sum(np.maximum(A_arr, 0.0))) - float(np.sum(np.maximum(D, 0.0)))), 0.0)
                        self._Y = min(self._Y, self.dvq_y_max)
                    if self.enable_power_vq:
                        self._X = max(self._X + self.pvq_eta * (total_power - self.power_budget), 0.0)
                        self._X = min(self._X, self.pvq_max)

                    reward = self.dgmw_mw_weight * mw_tp  # 无 scale！

                    # 为日志计算 w / avg_queue_ratio（panic 模式不依赖这些值）
                    _avg_q = float(np.mean(Q)) / max(self.Q_max, 1e-8)
                    _avg_q = float(np.clip(_avg_q, 0.0, 1.0))
                    _w = float(_avg_q ** self.blend_alpha)
                    _w = float(np.clip(_w, 0.0, 1.0))

                    return {
                        "reward": float(reward),
                        "ee_bonus": ee_bonus,
                        "V_ee_bonus": 0.0,
                        "ee_gate": 0.0,
                        "crpo_feasible": 0.0,
                        "vq_Y": float(self._Y),
                        "vq_y_weight": 0.0,
                        "power_vq_X": float(self._X),
                        "power_vq_penalty": 0.0,
                        "penalty_gate": 0.0,
                        "penalty_suppressed": True,
                        "ee_value": ee_value,
                        "deliv_utility": deliv_utility,
                        "ee_floor_penalty": ee_floor_penalty,
                        "drift_signal": drift_signal,
                        "q_penalty": 0.0,
                        "q_sq_sum": q_sq_sum,
                        "bp_reward": bp_reward,
                        "deficit_reward": deficit_reward,
                        "deficit_sum": float(self._deficit.sum()),
                        "pf_reward": pf_reward,
                        "overflow_penalty": 0.0,
                        "early_overflow": early_overflow,
                        "queue_weight": _w,
                        "avg_queue_ratio": _avg_q,
                        "total_power": total_power,
                        "P_consume": P_consume,
                        "R_sum": served_total,
                        "EE": ee_value,
                        "V": 0.0,
                        "current_V": self.V,
                        "dgmw_panic": True,
                    }

            # (a) Sigmoid 交付门: 平滑控制 EE-vs-throughput 主导权
            ee_gate = 1.0 / (1.0 + np.exp(-self.ee_gate_kappa * (dr - self.ee_gate_mid)))
            ee_gate = float(np.clip(ee_gate, 0.0, 1.0))

            # DGPS: 惩罚门控 — delivery 低时抑制所有惩罚，释放纯吞吐量梯度
            if self.penalty_gate_enable:
                penalty_gate = 1.0 / (1.0 + np.exp(-self.penalty_gate_kappa * (dr - self.penalty_gate_mid)))
                penalty_gate = float(np.clip(penalty_gate, self.penalty_gate_min, 1.0))
            else:
                penalty_gate = 1.0

            # Dinkelbach: 绕过 ee_gate 和 penalty_gate
            # 线性目标 R-q*P 在所有 delivery 水平提供正确梯度方向（低功率时为正），
            # 无需门控抑制；队列/溢出惩罚始终生效以提供排水信号
            if self.use_dinkelbach:
                ee_gate = 1.0
                penalty_gate = 1.0

            # (b) EE 分量: 门控到近零 (delivery 低时移除 ∂EE/∂P<0 的引力陷阱)
            ee_component = ee_gate * self.lyapunov_V * ee_bonus

            # (c) 吞吐量分量: max-weight backlog 加权, delivery 低时放大
            backlog_w = Q / max(self.Q_max, 1e-8)
            if user_drift_weights is not None:
                backlog_w = backlog_w * np.asarray(user_drift_weights, dtype=np.float64).reshape(-1)
            mw_tp = float(np.sum(backlog_w * D))
            tp_boost = 1.0 + self.tp_boost_max * (1.0 - ee_gate)
            tp_component = tp_boost * self.w_tp_dglcr * mw_tp

            # (d) 功率对偶惩罚: 仅当 EE 门开放时才生效
            # Dinkelbach 下同样生效: q*·P 软项给方向, X 硬约束做兜底, 双通道压功率
            if self.enable_power_vq:
                x_price = self._X / max(self.pvq_scale, 1e-8)
                power_vq_penalty = ee_gate * self.w_power_vq * x_price * total_power
            else:
                power_vq_penalty = 0.0

            # (e) Y 虚拟队列: 自适应对偶, 欠交付累积→自动抬升 throughput 权重
            D_total = float(np.sum(np.maximum(D, 0.0)))
            if arrivals is not None:
                A_arr = np.asarray(arrivals, dtype=np.float64).reshape(-1)
                A_total = float(np.sum(np.maximum(A_arr, 0.0)))
                self._Y = max(self._Y + (self.dvq_target * A_total - D_total), 0.0)
                self._Y = min(self._Y, self.dvq_y_max)
                y_weight = self._Y / max(self.dvq_y_scale, 1e-8)
                y_component = self.w_y_tp * y_weight * D_total
            else:
                y_component = 0.0

            # (f) X 能量虚拟队列: 仅当 EE 门开放时累积功率超预算
            # Dinkelbach 下持续累积: 功率超预算→x_price 上升→对 total_power 施加渐强下行压力
            if self.enable_power_vq:
                self._X = max(self._X + ee_gate * self.pvq_eta * (total_power - self.power_budget), 0.0)
                self._X = min(self._X, self.pvq_max)

            objective = ee_component + tp_component + y_component
            # DGPS: 惩罚项被 delivery 门控抑制；delivery 低时 penalty_gate≈0→纯吞吐量目标
            reward = self.reward_scale * (objective - penalty_gate * q_penalty - penalty_gate * overflow_penalty - penalty_gate * power_vq_penalty)
            reward = float(reward)
        elif self.crpo_enable and delivery_running is not None:
            dr = float(delivery_running)
            # 迟滞：dr>=thr 判可行，dr<thr-band 判不可行，带内维持上一状态
            if dr >= self.crpo_threshold:
                self._crpo_feasible = True
            elif dr < self.crpo_threshold - self.crpo_band:
                self._crpo_feasible = False

            # 线性 max-weight backlog 加权吞吐（不饱和，驱动清积压）
            mw_backlog = Q / max(self.Q_max, 1e-8)
            if user_drift_weights is not None:
                mw_backlog = mw_backlog * np.asarray(
                    user_drift_weights, dtype=np.float64
                ).reshape(-1)
            maxweight_throughput = float(np.sum(mw_backlog * D))

            if self._crpo_feasible:
                # 可行模式：EE 最大化 - 队列/溢出惩罚
                objective = self.crpo_ee_weight * self.lyapunov_V * ee_bonus
            else:
                # 不可行模式：纯线性 max-weight 吞吐 - 队列/溢出惩罚
                objective = self.crpo_bp_weight * maxweight_throughput

            reward = self.reward_scale * (
                objective - q_penalty - overflow_penalty
            )
        else:
            # 回退：原目标反转奖励
            reward = self.reward_scale * (
                deliv_utility
                - q_penalty
                - overflow_penalty
                - ee_floor_penalty
                + bp_reward + deficit_reward + pf_reward
            )
        # LDPS: 势函数整形项经 reward_scale 缩放后注入最终 reward（PBRS 策略不变）
        reward = reward + self.reward_scale * ldps_shaping
        reward = float(reward)

        # 仍然计算 w 用于日志监控（但不影响 reward）
        avg_queue_ratio = float(np.mean(Q)) / max(self.Q_max, 1e-8)
        avg_queue_ratio = float(np.clip(avg_queue_ratio, 0.0, 1.0))
        w = float(avg_queue_ratio ** self.blend_alpha)
        w = float(np.clip(w, 0.0, 1.0))

        # 更新自适应 V (保留用于外部查询)
        if self.adaptive_V:
            self.V = compute_adaptive_V(
                Q, self.Q_max, self.V_min, self.V_max, self.V_exponent
            )
        self.V_effective = self.V

        return {
            "reward": reward,
            "ee_bonus": ee_bonus,
            "V_ee_bonus": self.lyapunov_V * ee_bonus,
            "ee_gate": ee_gate,
            "crpo_feasible": float(self._crpo_feasible),
            "vq_Y": float(self._Y),
            "vq_y_weight": float(self._Y / max(self.dvq_y_scale, 1e-8)),
            "power_vq_X": float(self._X),
            "power_vq_penalty": float(power_vq_penalty) if self.enable_power_vq else 0.0,
            "penalty_gate": penalty_gate,
            "penalty_suppressed": penalty_gate < 0.5,
            "ee_value": ee_value,
            "dinkelbach_q": float(self.dinkelbach_q) if self.use_dinkelbach else 0.0,
            "deliv_utility": deliv_utility,
            "ee_floor_penalty": ee_floor_penalty,
            "drift_signal": drift_signal,
            "q_penalty": q_penalty,
            "q_sq_sum": q_sq_sum,
            "bp_reward": bp_reward,
            "deficit_reward": deficit_reward,
            "deficit_sum": float(self._deficit.sum()),
            "pf_reward": pf_reward,
            "overflow_penalty": overflow_penalty,
            "early_overflow": early_overflow,
            "queue_weight": w,
            "avg_queue_ratio": avg_queue_ratio,
            "total_power": total_power,
            "P_consume": P_consume,
            "R_sum": served_total,  # v16: 现在记录的是served_total（EE分子）
            "EE": ee_value,
            "V": self.lyapunov_V,
            "current_V": self.V,
            "ldps_shaping": float(ldps_shaping),
        }

    # ------------------------------------------------------------------ #
    # Virtual queue (auxiliary constraints)
    # ------------------------------------------------------------------ #
    def update_virtual_queue(
        self,
        power_consumed: np.ndarray,
        constraint_value: np.ndarray,
    ) -> np.ndarray:
        """更新虚拟队列以强制平均约束（保持原行为）。"""
        if not self.use_virtual_queue:
            return self._virtual_queue.copy()

        g = np.asarray(constraint_value, dtype=np.float64).reshape(-1)
        if g.shape[0] != self.K:
            raise ValueError(
                f"constraint_value shape {g.shape} != K={self.K}"
            )

        self._virtual_queue = np.maximum(
            self._virtual_queue + self.vq_step_size * g, 0.0
        )
        return self._virtual_queue.copy()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def get_v_tradeoff_info(self) -> Dict[str, float]:
        return {
            "V": float(self.V),
            "V_effective": float(self.V_effective),
            "V_min": float(self.V_min),
            "V_max": float(self.V_max),
            "adaptive_V": self.adaptive_V,
            "blend_alpha": float(self.blend_alpha),
            "ee_weight": float(self.ee_weight),
            "w_overflow": float(self.w_overflow),
        }

    @property
    def virtual_queue(self) -> np.ndarray:
        return self._virtual_queue.copy()
