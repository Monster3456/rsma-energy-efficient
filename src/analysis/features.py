"""观测特征注册表、块级切片与参照状态构造。

可解释性分析把观测的 per-user 特征块当作合作博弈的「玩家」（与 FSS 语义一致）：

- 13 个原生 per-user 块（``state_aug_enabled=True`` 时块 6/7/10 的原位语义为
  Lagrangian λ 桶 / DVQ 积压 / Dinkelbach η，见 ``env._get_observation``）。
- 时序历史滞后块（``obs_history_window=H>0``）是 h_gains/queue/power 的滞后副本，
  语义冗余，不进入博弈（文档注明：活体特征是集合内的原生块）。
- 全局尾 ``[time_ratio, snr_norm, eps_norm]``：``snr_norm``/``eps_norm`` 作为
  第 14/15 号玩家（与 per-user 块并列）；``time_ratio`` 不参与。

KernelSHAP 联盟掩码与 Expected-Gradients 参照都以「块」为单位构造。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 13 个原生 per-user 特征块（观测块序，state_aug 替换后的语义）。
NATIVE_FEATURE_NAMES: tuple[str, ...] = (
    "h_gains",  # 0  归一化信道增益（路径损耗/CSI 质量核心）
    "queue",  # 1  归一化队列积压 Q/Q_max
    "delta_queue",  # 2  队列增量 (Q−Q_prev)/Q_max
    "arrivals",  # 3  归一化到达 λ/Q_max（SNR 自适应）
    "power_history",  # 4  上时隙功率 P_prev/P_k_max
    "rate_history",  # 5  上时隙速率 R_prev/max(R_prev)
    "lambda_bucket",  # 6  state_aug: Lagrangian λ 桶值
    "dvq_backlog",  # 7  state_aug: 交付虚拟队列积压 Y
    "split_mask_alpha",  # 8  per-user 公共/私有功率分割比 α（Meta 输出）
    "group_ids",  # 9  归一化分组 id
    "dinkelbach_eta",  # 10 state_aug: Dinkelbach q*
    "qos_lambda",  # 11 QoS λ 倍率
    "qos_priority",  # 12 QoS drift/overflow 优先级权重
)
GLOBAL_FEATURE_NAMES: tuple[str, ...] = ("time_ratio", "snr_norm", "eps_norm")

#: 局部归因（KernelSHAP / EG）的 15 个玩家：13 原生块 + snr_norm + eps_norm。
SHAP_PLAYER_NAMES: tuple[str, ...] = NATIVE_FEATURE_NAMES + ("global_snr", "global_eps")

_N_NATIVE = len(NATIVE_FEATURE_NAMES)  # 13
_N_SHAP_PLAYERS = len(SHAP_PLAYER_NAMES)  # 15
_SNR_PLAYER_IDX = _N_NATIVE  # 13 -> global tail idx 1
_EPS_PLAYER_IDX = _N_NATIVE + 1  # 14 -> global tail idx 2


@dataclass(frozen=True)
class ObsLayout:
    """观测布局描述：K 用户、13 原生块 + H·F 历史块 + 3 全局尾。

    块序 = feature-major（块 f 位于 ``obs[f*K:(f+1)*K]``），与
    ``env._get_observation`` / ``ActorCriticAttn._split_obs`` 一致。
    """

    K: int
    H: int = 0
    history_features: tuple[str, ...] = ()
    n_native: int = _N_NATIVE
    n_global: int = 3

    @property
    def d_user_eff(self) -> int:
        """per-user 总块数 = 13 原生 + H·len(history_features) 历史滞后。"""
        return self.n_native + self.H * len(self.history_features)

    @property
    def globals_offset(self) -> int:
        """全局尾起始位置 = d_user_eff · K。"""
        return self.d_user_eff * self.K

    @property
    def n_blocks_total(self) -> int:
        """总块数（per-user 全部块 + 全局）。"""
        return self.d_user_eff + self.n_global

    @property
    def n_players(self) -> int:
        """局部归因玩家数 = 13 原生 + snr + eps。"""
        return _N_SHAP_PLAYERS

    def obs_dim(self) -> int:
        return self.globals_offset + self.n_global


def resolve_layout(config) -> ObsLayout:
    """从 ``src.config.Config`` 构造观测布局。"""
    sys_cfg = config.system
    hist = tuple(getattr(sys_cfg, "obs_history_features", ()))
    return ObsLayout(
        K=int(sys_cfg.K),
        H=int(getattr(sys_cfg, "obs_history_window", 0)),
        history_features=hist,
    )


def split_blocks(obs: np.ndarray, layout: ObsLayout) -> np.ndarray:
    """把 (obs_dim,) 或 (N, obs_dim) 观测切为 (n_blocks_total, K) 块矩阵。

    block f (f < d_user_eff) 位于 ``obs[..., f*K:(f+1)*K]``；
    全局块位于电平尾，每块单元素。
    """
    arr = np.asarray(obs, dtype=np.float64)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[None, :]
    per_user = arr[:, : layout.globals_offset]
    blocks = per_user.reshape(arr.shape[0], layout.d_user_eff, layout.K).transpose(0, 2, 1)
    glob = arr[:, -layout.n_global :]
    out = np.concatenate([blocks, glob[:, None, :]], axis=2)  # (N, K, n_blocks_total)
    if squeeze:
        out = out[0]
    return out


def native_blocks(obs: np.ndarray, layout: ObsLayout) -> np.ndarray:
    """前 13 个原生 per-user 块 → (N, K, 13) 或 (K, 13)。"""
    arr = np.asarray(obs, dtype=np.float64)
    sq = arr.ndim == 1
    if sq:
        arr = arr[None, :]
    per_user = arr[:, : layout.globals_offset]
    out = per_user.reshape(arr.shape[0], layout.d_user_eff, layout.K).transpose(0, 2, 1)[
        :, :, : layout.n_native
    ]
    return out[0] if sq else out


def global_tail(obs: np.ndarray, layout: ObsLayout) -> np.ndarray:
    """全局尾 [time_ratio, snr_norm, eps_norm]，(3,) 或 (N, 3)。"""
    arr = np.asarray(obs, dtype=np.float64)
    squeeze = arr.ndim == 1
    out = arr[..., -layout.n_global :] if not squeeze else arr[-layout.n_global :]
    return out


def native_feature_vector(obs: np.ndarray, layout: ObsLayout, user_idx: int) -> np.ndarray:
    """第 user_idx 个用户的 13 维原生特征向量。"""
    return native_blocks(obs, layout)[user_idx]


def mask_player_obs(obs: np.ndarray, layout: ObsLayout, active: np.ndarray, ref_obs: np.ndarray) -> np.ndarray:
    """按玩家联盟掩码构造掩码观测（KernelSHAP 联盟求值）。

    Args:
        obs: 待解释状态 (obs_dim,)。
        active: (n_players=15,) 布尔；True=该玩家取 obs 值，False=取参照值。
        ref_obs: 参照状态 (obs_dim,)；非玩家块（历史滞后、time_ratio）恒取参照值，
            即非玩家特征按参照边际化（标准 fill-in 约定）。

    Returns:
        (obs_dim,) 掩码后的状态。
    """
    active = np.asarray(active, dtype=bool)
    assert active.shape == (layout.n_players,), f"active 形状 {active.shape} != (15,)"
    o = np.array(obs, dtype=np.float64)
    r = np.array(ref_obs, dtype=np.float64)
    masked = r.copy()
    for j in range(layout.n_native):
        if active[j]:
            masked[j * layout.K : (j + 1) * layout.K] = o[j * layout.K : (j + 1) * layout.K]
    g0 = layout.globals_offset
    if active[_SNR_PLAYER_IDX]:
        masked[g0 + 1] = o[g0 + 1]
    if active[_EPS_PLAYER_IDX]:
        masked[g0 + 2] = o[g0 + 2]
    return masked


# ---------------------------------------------------------------- 参照构造 --


def ref_zero(layout: ObsLayout) -> np.ndarray:
    """R_zero：全零参照（类比 DeepSHAP 文献的「黑图」）。"""
    return np.zeros(layout.obs_dim(), dtype=np.float64)


def ref_means(obs_stack: np.ndarray, layout: ObsLayout) -> np.ndarray:
    """R_means：逐块边际均值参照（每个 K 块取 N×K 均值；全局尾取 N 均值）。"""
    arr = np.asarray(obs_stack, dtype=np.float64)
    m = np.zeros(layout.obs_dim(), dtype=np.float64)
    blocks = arr[:, : layout.globals_offset].reshape(arr.shape[0], layout.d_user_eff, layout.K)
    # 每个 per-user 块取 N×K 样本均值（块内所有用户同一参照值）
    block_means = blocks.mean(axis=(0, 2))  # (d_user_eff,)
    m[: layout.d_user_eff * layout.K] = block_means.repeat(layout.K)
    m[layout.globals_offset :] = arr[:, -layout.n_global :].mean(axis=0)
    return m


def ref_bootstrap(obs_stack: np.ndarray, layout: ObsLayout, rng: np.random.Generator, n: int) -> np.ndarray:
    """R_boot：从收集状态中随机有放回抽取 n 行作为参照分布 (n, obs_dim)。"""
    arr = np.asarray(obs_stack, dtype=np.float64)
    idx = rng.integers(0, arr.shape[0], size=n)
    return arr[idx]


def build_neutral_obs(layout: ObsLayout, global_obs: np.ndarray | None = None) -> np.ndarray:
    """R_neutral：无信息的中性状态。

    per-user 块取语义中性值（队列半满、信道中性、无约束压力、无历史等），
    全局尾取 ``global_obs`` 的真实尾（若给定，即「中性用户态 + 真实运行点」），
    否则取默认 (time_ratio=0.5, snr_norm=0.5, eps_norm=0.02)。
    """
    K = layout.K
    vec = np.zeros(layout.obs_dim(), dtype=np.float64)
    vec[0 * K : 1 * K] = 0.5  # h_gains：中性信道
    vec[1 * K : 2 * K] = 0.5  # queue：半满
    # 2 delta_queue / 3 arrivals(0.05≈中等负载)：
    vec[3 * K : 4 * K] = 0.05
    # 4 power_history / 5 rate_history / 6 lambda_bucket / 7 dvq_backlog = 0
    vec[8 * K : 9 * K] = 0.5  # split_mask α：公共/私有对半分割
    vec[9 * K : 10 * K] = 0.5  # group_ids：组间中性
    # 10 dinkelbach_eta = 0（无 Dinkelbach 补贴）
    vec[11 * K : 12 * K] = 1.0  # qos_lambda：优先类（乘数/max = 1）
    vec[12 * K : 13 * K] = 1.0  # qos_priority：优先类
    if layout.H > 0:
        # 历史滞后块 = 当前时隙快照的中性值（自洽的恒定中性状态）
        for lag in range(layout.H):
            base = (layout.n_native + lag * len(layout.history_features)) * K
            feats = layout.history_features
            for off, name in enumerate(feats):
                if name == "h_gains" or name == "queue":
                    vec[(base + off) * K : (base + off + 1) * K] = 0.5
                elif name == "power":
                    vec[(base + off) * K : (base + off + 1) * K] = 0.0
    g0 = layout.globals_offset
    if global_obs is not None:
        g = np.asarray(global_obs, dtype=np.float64)
        assert g.shape[-1] == layout.n_global
        vec[g0 : g0 + layout.n_global] = g
    else:
        vec[g0] = 0.5  # time_ratio
        vec[g0 + 1] = 0.5  # snr_norm（≈15dB 中点）
        vec[g0 + 2] = 0.02  # eps_norm（小 CSI 误差）
    return vec
