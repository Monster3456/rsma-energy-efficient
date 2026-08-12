"""信道模型模块

实现上行MIMO信道生成与时间演化：
- AR(1) 时间相关 Rayleigh 衰落: H_k[t] = rho * H_k[t-1] + sqrt(1-rho^2) * W_k[t]
- 完美 CSI: 基站直接获取真实信道
- 不完美 CSI: 基站获取的是带延迟和估计噪声的信道,
  误差被约束在椭球 ||H_k - H_hat_k||_F <= epsilon_k 内
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np


def _cn_randn(rng: np.random.Generator, shape: tuple) -> np.ndarray:
    """生成元素独立的标准复高斯 CN(0, 1) 矩阵.

    实部、虚部各为 N(0, 1/2), 故 E[|z|^2] = 1.
    """
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (real + 1j * imag) / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Task#13: 3GPP Urban Micro 路径损耗模型
# ---------------------------------------------------------------------------
def generate_user_distances(
    K: int,
    r_min: float = 50.0,
    r_max: float = 300.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """在环形区域 [r_min, r_max] 均匀采样 K 个用户距离 (m).

    采样面积均匀：d = sqrt(U[r_min^2, r_max^2]).
    """
    if rng is None:
        rng = np.random.default_rng()
    distances = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2, size=K))
    return distances.astype(np.float64)


def compute_path_loss_db(
    distances: np.ndarray,
    pl_ref: float = 128.1,
    pl_exp: float = 3.76,
) -> np.ndarray:
    """3GPP UMi 路径损耗 (dB): PL = pl_ref + 10*pl_exp*log10(d_km)."""
    d = np.maximum(np.asarray(distances, dtype=np.float64), 1e-3)
    d_km = d / 1000.0
    return pl_ref + 10.0 * pl_exp * np.log10(d_km)


def compute_path_loss_linear(
    distances: np.ndarray,
    pl_ref: float = 128.1,
    pl_exp: float = 3.76,
    ref_distance: Optional[float] = None,
) -> np.ndarray:
    """路径损耗的幅度缩放因子.

    以 ref_distance 处的 PL 为参考归一（使该处输出 = 1），仅保留近远相对差异。
    这样发射 SNR 标度不被绝对 PL（~80–110dB）冲掉，保留 ~29dB 用户间增益不平衡.

    Returns
    -------
    pl_amp : np.ndarray
        shape (K,), 幅度因子 amp = 10**(-PL_db_relative / 20)
        近用户 (50m): amp ≈ 3.7；远用户 (300m): amp ≈ 0.13。
    """
    pl_db = compute_path_loss_db(distances, pl_ref, pl_exp)
    if ref_distance is not None and ref_distance > 0:
        pl_db_ref = float(pl_ref + 10.0 * pl_exp * np.log10(ref_distance / 1000.0))
        pl_db = pl_db - pl_db_ref
    return (10.0 ** (-pl_db / 20.0)).astype(np.float64)


class ChannelGenerator:
    """上行 MIMO 信道生成与演化.

    支持两种模式:
    - perfect CSI:   H_hat = H_true
    - imperfect CSI: 基于 MMSE 预测 H_hat = rho^delay * H[t-delay] + 估计噪声,
                     总误差 ||H - H_hat||_F 被截断在 epsilon_k 椭球内.

    Parameters
    ----------
    K : int
        用户数.
    N_r : int
        基站接收天线数.
    N_t : int
        每个用户的发射天线数.
    rho : float
        AR(1) 时间相关系数, 取值 [0, 1).
    csi_mode : str
        "perfect" 或 "imperfect".
    epsilon_csi : float
        每元素估计噪声标准差 (归一化信道下).
    csi_delay : int
        反馈延迟时隙数 (>= 0).
    seed : int, optional
        随机种子, 用于可复现实验.
    """

    def __init__(
        self,
        K: int,
        N_r: int,
        N_t: int,
        rho: float = 0.9,
        csi_mode: str = "perfect",
        epsilon_csi: float = 0.1,
        csi_delay: int = 1,
        seed: Optional[int] = None,
        sigma2: float = 0.01,
        snr_dependent_csi: bool = False,
        pilot_power: float = 1.0,
        epsilon_floor: float = 0.01,
    ) -> None:
        if csi_mode not in ("perfect", "imperfect"):
            raise ValueError(f"csi_mode must be 'perfect' or 'imperfect', got {csi_mode!r}")
        if not (0.0 <= rho < 1.0):
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        if csi_delay < 0:
            raise ValueError(f"csi_delay must be >= 0, got {csi_delay}")

        self.K: int = int(K)
        self.N_r: int = int(N_r)
        self.N_t: int = int(N_t)
        self.rho: float = float(rho)
        self.csi_mode: str = csi_mode
        self.epsilon_csi: float = float(epsilon_csi)
        self.csi_delay: int = int(csi_delay)

        # Task#20: SNR 依赖的估计噪声参数
        self.sigma2: float = float(sigma2)
        self.snr_dependent_csi: bool = bool(snr_dependent_csi)
        self._pilot_power: float = float(pilot_power)
        self._epsilon_floor: float = float(epsilon_floor)

        # 椭球半径放大因子: 3-sigma 约 99.7% 覆盖
        self._chi_factor: float = 3.0

        self._rng: np.random.Generator = np.random.default_rng(seed)

        # 当前真实信道 H[t], 形状 (K, N_r, N_t)
        self._H_true: Optional[np.ndarray] = None
        # 历史缓存: 用于在 imperfect CSI 中读取 t-delay 时刻的信道
        self._history: deque = deque(maxlen=max(self.csi_delay, 0) + 1)

        # Task#13: 路径损耗幅度因子 (K,), None 表示不应用 PL.
        # 应用方式: H_observed[k,:,:] = pl_amp[k] * H_rayleigh[k,:,:].
        self._path_loss_amp: Optional[np.ndarray] = None

    def set_sigma2(self, sigma2: float) -> None:
        """同步当前 episode 的噪声功率（供 SNR 依赖估计噪声使用）."""
        self.sigma2 = float(sigma2)

    def _current_sigma_est(self) -> float:
        """返回当前等效估计噪声标准差 sigma_est.

        - snr_dependent_csi=True: sigma_est² = σ² / (K * P_pilot) + epsilon_floor²
        - snr_dependent_csi=False: sigma_est = epsilon_csi (旧行为)
        """
        if self.snr_dependent_csi:
            denom = max(self.K * self._pilot_power, 1e-12)
            sigma_est_sq = self.sigma2 / denom + self._epsilon_floor ** 2
            return float(np.sqrt(sigma_est_sq))
        return float(self.epsilon_csi)

    def set_path_loss(self, pl_amp: Optional[np.ndarray]) -> None:
        """设置路径损耗幅度因子 (K,). None 表示关闭 PL."""
        if pl_amp is None:
            self._path_loss_amp = None
            return
        arr = np.asarray(pl_amp, dtype=np.float64).reshape(-1)
        if arr.shape[0] != self.K:
            raise ValueError(
                f"path_loss_amp shape {arr.shape} != ({self.K},)"
            )
        if np.any(arr <= 0):
            raise ValueError("path_loss_amp must be strictly positive")
        self._path_loss_amp = arr

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _sample_innovation(self) -> np.ndarray:
        """采样 AR(1) 创新项 W_k[t] ~ CN(0, I), 形状 (K, N_r, N_t)."""
        return _cn_randn(self._rng, (self.K, self.N_r, self.N_t))

    def _build_observation(self) -> dict:
        """根据当前 self._H_true 与历史缓存生成观测字典."""
        H_true_ray = self._H_true
        assert H_true_ray is not None

        if self.csi_mode == "perfect":
            H_hat_ray = H_true_ray.copy()
            sigma_e2_ray = np.zeros(self.K, dtype=np.float64)
        else:
            H_hat_ray, sigma_e2_ray = self._imperfect_estimate()

        # Task#13: 在输出侧乘以路径损耗幅度 (K, 1, 1) 广播到 (K, N_r, N_t).
        # 内部仍以原始 Rayleigh 表示演化与截断 CSI 误差椭球, 仅在导出时调用 PL.
        pl_amp = self._path_loss_amp
        if pl_amp is not None:
            scale = pl_amp[:, None, None]
            H_true = H_true_ray * scale
            H_hat = H_hat_ray * scale
            # 误差方差按 |amp|^2 缩放
            sigma_e2 = sigma_e2_ray * (pl_amp ** 2)
        else:
            H_true = H_true_ray.copy()
            H_hat = H_hat_ray
            sigma_e2 = sigma_e2_ray

        gains_hat = np.sum(np.abs(H_hat) ** 2, axis=(1, 2)).astype(np.float64)
        gains_true = np.sum(np.abs(H_true) ** 2, axis=(1, 2)).astype(np.float64)

        return {
            "H_true": H_true,
            "H_hat": H_hat,
            "error_variance": sigma_e2,
            # Task#20: channel_gains 基于 H_hat（模型只能看到估计信道）
            "channel_gains": gains_hat,
            # 保留真实增益，供内部诊断/奖励计算使用
            "channel_gains_true": gains_true,
            # 当前等效估计噪声标准差（供 rate_calc 动态 error_bound 使用）
            "sigma_est": float(
                0.0 if self.csi_mode == "perfect" else self._current_sigma_est()
            ),
        }

    def _imperfect_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """构造 imperfect CSI 下的估计 H_hat 与每用户误差方差.

        H_hat[t] = rho^delay * H[t-delay] + n_est,  n_est ~ CN(0, sigma_est^2 I)
        每元素总误差方差: sigma_e^2 = (1 - rho^(2*delay)) + sigma_est^2
        将每用户的总误差范数截断在 epsilon_k = sqrt(N_r*N_t*sigma_e^2) * chi 内.
        """
        H_true = self._H_true
        assert H_true is not None

        delay = self.csi_delay
        rho_d = self.rho ** delay

        # 历史不足时退化为当前信道(只在第一帧之后, 历史长度增长到 delay+1 后稳定)
        if len(self._history) >= delay + 1:
            H_delayed = self._history[0]
        else:
            H_delayed = H_true

        sigma_est = self._current_sigma_est()
        n_est = sigma_est * _cn_randn(self._rng, (self.K, self.N_r, self.N_t))

        H_hat = rho_d * H_delayed + n_est

        # 总误差方差（用于截断椭球计算）
        per_elem_var = (1.0 - self.rho ** (2 * delay)) + sigma_est ** 2
        # 对外返回的 error_variance 仅包含估计噪声，信道老化已通过 H_hat = rho^d * H[t-d] 隐式体现
        sigma_e2 = np.full(self.K, sigma_est ** 2, dtype=np.float64)

        eps_k = float(np.sqrt(self.N_r * self.N_t * per_elem_var) * self._chi_factor)

        # 硬截断: 保证 ||H_true - H_hat||_F <= eps_k
        diff = H_true - H_hat
        for k in range(self.K):
            norm_k = float(np.linalg.norm(diff[k]))
            if norm_k > eps_k and norm_k > 0.0:
                scale = eps_k / norm_k
                diff[k] = diff[k] * scale
                H_hat[k] = H_true[k] - diff[k]

        return H_hat, sigma_e2

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def reset(self) -> dict:
        """Episode 开始, 生成初始信道 H_k[0] ~ CN(0, I).

        Returns
        -------
        dict
            键: 'H_true', 'H_hat', 'error_variance', 'channel_gains'.
        """
        self._H_true = _cn_randn(self._rng, (self.K, self.N_r, self.N_t))
        self._history.clear()
        self._history.append(self._H_true.copy())
        return self._build_observation()

    def step(self) -> dict:
        """生成下一时隙信道 (AR(1) 演化)."""
        if self._H_true is None:
            raise RuntimeError("Must call reset() before step().")

        W = self._sample_innovation()
        self._H_true = self.rho * self._H_true + np.sqrt(1.0 - self.rho ** 2) * W
        self._history.append(self._H_true.copy())
        return self._build_observation()

    def get_eigenvalues(self, H_hat: np.ndarray) -> np.ndarray:
        """返回每用户信道矩阵的 top-2 奇异值 (按降序), 形状 (K, 2).

        若 N_t < 2, 末尾用 0 填充.
        """
        if H_hat.ndim != 3 or H_hat.shape[0] != self.K:
            raise ValueError(f"Expected shape (K={self.K}, N_r, N_t), got {H_hat.shape}")

        out = np.zeros((self.K, 2), dtype=np.float64)
        for k in range(self.K):
            s = np.linalg.svd(H_hat[k], compute_uv=False)
            n = min(2, s.shape[0])
            out[k, :n] = s[:n]
        return out
