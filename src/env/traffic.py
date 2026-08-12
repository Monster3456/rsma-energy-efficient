"""流量模型模块

实现泊松到达的数据包队列模型。
Q_k(t) = max(Q_k(t-1) - D_k(t), 0) + A_k(t)
其中 A_k(t) ~ Poisson(lambda_rate)，单位包大小为 packet_size（归一化为 bits）。
"""
from typing import Dict, Optional

import numpy as np


def estimate_capacity(
    snr_db: float,
    K: int = 16,
    N_r: int = 16,
    N_t: int = 1,
    use_path_loss: bool = False,
    cell_radius_min: float = 50.0,
    cell_radius_max: float = 300.0,
    path_loss_exponent: float = 3.76,
    path_loss_ref_distance: float = 100.0,
) -> float:
    """估算上行 MIMO-RSMA 系统在给定 SNR 下的完美信道总容量（bits/slot/Hz）。

    Task#27: 以前版本使用 imperfect CSI 拟合点，在低 SNR 高估导致
    实际负载过高。现改用 MMSE 接收机下的香农上界作为“完美容量”参考：
        C ≈ min(K, N_r) · log2(1 + SNR · N_r / K)
    该近似反映了 MIMO 多流增益。在 K > N_r 过载场景下额外
    乘以 sqrt(N_r/K) 作为过载衰减。

    Task#43: 回滚 Task#41 修改——保留 N_t 形参以兼容现有调用，但不再
    使用 N_t 乘法（v8 微调显示加入 N_t 后 0dB 性能反而退化）。

    Task#14: 路径损耗修正（仅在 use_path_loss=True 时启用）：等效 SNR 被
    pl_eff_db = 0.6 · PL_rel 衰减，阶段后额外 ×0.85 反映 SIC/调度损失。
    """
    s = float(snr_db)

    if use_path_loss:
        d_rms = float(np.sqrt((cell_radius_max ** 2 + cell_radius_min ** 2) / 2.0))
        d_ref = float(max(path_loss_ref_distance, 1.0))
        pl_relative_db = 10.0 * float(path_loss_exponent) * float(np.log10(d_rms / d_ref))
        pl_eff_db = 0.6 * pl_relative_db
        s_eff = s - pl_eff_db
    else:
        s_eff = s

    # Task#27: Shannon-based perfect-channel capacity (MMSE upper bound)
    snr_linear = 10.0 ** (s_eff / 10.0)
    effective_streams = float(min(max(K, 1), max(N_r, 1)))
    # 多用户 MIMO：每流有效 SNR 获得 N_r/K 的接收阵列增益（均匀功率分配近似）
    # Task#43: 回滚 Task#41——不再乘 N_t
    _ = N_t  # 保留形参以兼容旧调用，但不使用
    per_stream_snr = snr_linear * float(N_r) / float(max(K, 1))
    cap_per_stream = float(np.log2(1.0 + per_stream_snr))
    cap = effective_streams * cap_per_stream

    # 过载场景（K > N_r）：多用户干扰与调度损失
    if K > N_r:
        overload_factor = float(np.sqrt(max(N_r, 1) / max(K, 1)))
        cap = cap * overload_factor

    # 路径损耗场景的额外 SIC/调度损失
    if use_path_loss:
        cap = cap * 0.85

    # Clip 范围：低 SNR 不下穿 5 bits/slot，避免极端下路；上限 200
    cap = float(np.clip(cap, 5.0, 200.0))
    return cap


def snr_adaptive_load_ratio(
    snr_db: float,
    r_min: float = 0.3,
    r_max: float = 0.9,
    snr_mid: float = 12.0,
    steepness: float = 0.15,
) -> float:
    """SNR 自适应的目标负载比 (sigmoid 形式)。

    Task#44: 替代常数 target_load_ratio=0.75。低 SNR 下 SIC 不可靠 + path_loss
    使有效容量远小于理论 Shannon 上界，需保守负载；高 SNR 下推高负载以提升 EE。

    ratio(snr_db) = r_min + (r_max - r_min) * sigmoid(steepness * (snr_db - snr_mid))

    典型取值（默认参数）：
        0dB → ~0.35,  5dB → ~0.49,  10dB → ~0.59,  12dB → ~0.60,
        15dB → ~0.66, 20dB → ~0.74, 25dB → ~0.81, 30dB → ~0.85

    Args:
        snr_db: 当前 episode 信噪比 (dB)
        r_min: 低 SNR 极限的目标负载比下限
        r_max: 高 SNR 极限的目标负载比上限
        snr_mid: sigmoid 中点 (dB)，对应 (r_min+r_max)/2
        steepness: sigmoid 陡度 (1/dB)

    Returns:
        目标负载比 ∈ [r_min, r_max]
    """
    x = float(steepness) * (float(snr_db) - float(snr_mid))
    # 数值稳定的 sigmoid
    if x >= 0:
        s = 1.0 / (1.0 + float(np.exp(-x)))
    else:
        ex = float(np.exp(x))
        s = ex / (1.0 + ex)
    ratio = float(r_min) + (float(r_max) - float(r_min)) * s
    return float(ratio)


class TrafficManager:
    """泊松随机到达与队列演化管理。

    队列动态：
        Q_k[t] = min( max{ Q_k[t-1] - D_k[t], 0 } + A_k[t],  Q_max )
    其中：
        - D_k[t]：本时隙发送的比特数（受到 Q_k[t-1] 上界约束）
        - A_k[t] = packet_size * Poisson(lambda_rate)
        - 超过 Q_max 的部分计入 overflow（被丢弃）
    """

    def __init__(
        self,
        K: int,
        lambda_rate: float,
        Q_max: float,
        packet_size: float,
        seed: Optional[int] = None,
        lambda_per_user: Optional[np.ndarray] = None,
    ) -> None:
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if lambda_rate < 0:
            raise ValueError(f"lambda_rate must be non-negative, got {lambda_rate}")
        if Q_max <= 0:
            raise ValueError(f"Q_max must be positive, got {Q_max}")
        if packet_size <= 0:
            raise ValueError(f"packet_size must be positive, got {packet_size}")

        self.K = int(K)
        self._lambda_rate = float(lambda_rate)
        self.Q_max = float(Q_max)
        self.packet_size = float(packet_size)
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # v13: per-user lambda 支持
        if lambda_per_user is not None:
            self.lambda_per_user = np.asarray(lambda_per_user, dtype=np.float64).reshape(self.K)
        else:
            self.lambda_per_user = np.full(self.K, self._lambda_rate, dtype=np.float64)

        # 运行期状态
        self.queue_lengths = np.zeros(self.K, dtype=np.float64)
        self._total_arrived = np.zeros(self.K, dtype=np.float64)
        self._total_served = np.zeros(self.K, dtype=np.float64)
        self._total_overflow = np.zeros(self.K, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> Dict[str, np.ndarray]:
        """Episode 开始：重置队列与累计计数（冷启动 Q[0]=0）。"""
        if self._seed is not None:
            self.rng = np.random.default_rng(self._seed)

        self.queue_lengths = np.zeros(self.K, dtype=np.float64)
        self._total_arrived = np.zeros(self.K, dtype=np.float64)
        self._total_served = np.zeros(self.K, dtype=np.float64)
        self._total_overflow = np.zeros(self.K, dtype=np.float64)

        return {
            "queue_lengths": self.queue_lengths.copy(),
            "arrivals": np.zeros(self.K, dtype=np.float64),
        }

    def step(self, served_bits: np.ndarray) -> Dict[str, np.ndarray]:
        """单时隙队列更新。

        Args:
            served_bits: (K,) 本时隙欲发送比特 D_k[t]。负值截断为 0，
                超过 Q_k[t-1] 的部分自动 clip。
        """
        served_bits = np.asarray(served_bits, dtype=np.float64).reshape(-1)
        if served_bits.shape[0] != self.K:
            raise ValueError(
                f"served_bits shape {served_bits.shape} does not match K={self.K}"
            )

        queue_before = self.queue_lengths.copy()

        # 1) 服务：D_k 不能为负且不超过队列存量
        served = np.minimum(np.maximum(served_bits, 0.0), queue_before)
        q_after_service = queue_before - served  # >= 0

        # 2) 泊松到达 (v13: 使用 per-user lambda)
        arrivals_pkt = self.rng.poisson(self.lambda_per_user, size=self.K).astype(np.float64)
        arrivals = arrivals_pkt * self.packet_size

        # 3) 入队 + 溢出截断
        q_new = q_after_service + arrivals
        overflow = np.maximum(q_new - self.Q_max, 0.0)
        self.queue_lengths = np.minimum(q_new, self.Q_max)

        # 4) 累计统计
        self._total_arrived += arrivals
        self._total_served += served
        self._total_overflow += overflow

        return {
            "queue_lengths": self.queue_lengths.copy(),
            "arrivals": arrivals,
            "served": served,
            "overflow": overflow,
            "queue_before": queue_before,
        }

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #
    def get_delivery_ratio(self) -> float:
        """累积交付率 = sum(served) / sum(arrived)。

        空队列返回 0.5（中性值），避免返回 1.0 让 Lagrangian 误判约束已满足，
        也避免返回 0.0 触发虚假违约惩罚。
        """
        total_arrived = float(self._total_arrived.sum())
        if total_arrived <= 0.0:
            return 0.5
        return float(self._total_served.sum() / total_arrived)

    @property
    def mean_queue_length(self) -> float:
        """当前所有用户的平均队列长度。"""
        return float(np.mean(self.queue_lengths))

    @property
    def total_arrived(self) -> np.ndarray:
        return self._total_arrived.copy()

    @property
    def total_served(self) -> np.ndarray:
        return self._total_served.copy()

    @property
    def lambda_rate(self) -> float:
        """当前全局 lambda_rate。"""
        return self._lambda_rate

    @lambda_rate.setter
    def lambda_rate(self, value: float) -> None:
        """设置全局 lambda_rate 并同步更新 lambda_per_user。"""
        self._lambda_rate = float(value)
        self.lambda_per_user = np.full(self.K, self._lambda_rate, dtype=np.float64)

    def set_lambda_per_user(self, lambdas: np.ndarray) -> None:
        """动态更新 per-user lambda (v13 QoS)。"""
        self.lambda_per_user = np.asarray(lambdas, dtype=np.float64).reshape(self.K)

    @property
    def total_overflow(self) -> np.ndarray:
        return self._total_overflow.copy()
