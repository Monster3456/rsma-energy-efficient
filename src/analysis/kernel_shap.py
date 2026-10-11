"""KernelSHAP：模型无关的局部归因代理（自实现，无依赖）。

对块级博弈（论文口径：feature block = 玩家）做 Shapley-核加权最小二乘。
联盟 z 的解释语义 = 「激活玩家取值自被解释输入、非玩家按参照背景边际化」——
参照变体（zero/means/boot/neutral，文献 2 协议）统一用背景分布表示：
确定性参照 = 单点背景，bootstrap = 经验状态分布背景。

采样：按核权重 p(s) ∝ (n−1)/(s·(n−s)) 抽大小 s，均匀子集 + 互补配对
（Štrumbelj-Kononenko 配对降方差）；含全集/空集的 ε 权行钉住效率性。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.analysis.features import ObsLayout, mask_player_obs


def shapley_kernel_weight(n: int, s: int) -> float:
    """标准 SHAP 核权重 w(z) = (n−1)/[C(n,s)·s·(n−s)]，0<s<n；端点返回 0。"""
    from math import comb

    if s <= 0 or s >= n:
        return 0.0
    return (n - 1) / (comb(n, s) * s * (n - s))


def sample_coalitions(n: int, n_pairs: int, rng: np.random.Generator):
    """按核权重抽 n_pairs 对互补联盟。

    Returns:
        (Z (2·n_pairs+2, n) bool, weights (2·n_pairs+2,))——末两行为全集/空集 ε 权。
    """
    sizes = np.arange(1, n)
    w_raw = (n - 1) / (sizes * (n - sizes))
    p_size = w_raw / w_raw.sum()
    z_rows = []
    weights = []
    for _ in range(n_pairs):
        s = int(rng.choice(sizes, p=p_size))
        z = np.zeros(n, dtype=bool)
        z[rng.choice(n, size=s, replace=False)] = True
        z_rows.append(z)
        weights.append(shapley_kernel_weight(n, s))
        z_rows.append(~z)
        weights.append(shapley_kernel_weight(n, n - s))
    Z = np.stack(z_rows)
    n_full = np.ones(n, dtype=bool)
    Z = np.concatenate([Z, n_full[None, :], ~n_full[None, :]], axis=0)
    eps = float(min(weights)) * 1e-3 if weights else 1e-9
    weights = np.array(weights + [eps, eps], dtype=np.float64)
    return Z, weights


def kernel_shap(
    eval_fn: Callable,
    x_in: np.ndarray,
    n_players: int,
    bg_rows: np.ndarray,
    mask_fn: Callable,
    n_pairs: int = 512,
    seed: int = 0,
    batch: int = 4096,
):
    """通用 KernelSHAP：对任意输入的块级博弈算归因。

    Args:
        eval_fn: fn(tensor (B, input_dim)) -> tensor (B,) 目标（no_grad 批前向）。
        x_in: (input_dim,) 被解释输入（np.float64，展平形态）。
        n_players: 玩家数。
        bg_rows: (bg_k, input_dim) 背景分布样本（确定性参照 = 重复单点）。
        mask_fn: fn(bg_row (input_dim,), active (n_players,) bool) -> 掩码输入。
        n_pairs: 互补联盟对数（行数 = 2·n_pairs + 2）。
        seed / batch: 随机种子 / 前向批大小。

    Returns:
        (phi (n_players,), phi0, info dict)。
    """
    import torch

    n = int(n_players)
    rng = np.random.default_rng(seed)
    bg = np.asarray(bg_rows, dtype=np.float64)
    Z, weights = sample_coalitions(n, int(n_pairs), rng)

    flat, row_counts = [], []
    for z in Z:
        rows = np.stack([mask_fn(b, z) for b in bg])  # (bg_k, input_dim)
        flat.append(rows)
        row_counts.append(rows.shape[0])
    all_masked = np.concatenate(flat, axis=0)

    out = np.zeros(all_masked.shape[0], dtype=np.float64)
    with torch.no_grad():
        for s in range(0, all_masked.shape[0], batch):
            chunk = torch.from_numpy(all_masked[s : s + batch]).float()
            out[s : s + batch] = eval_fn(chunk).cpu().numpy()

    values = np.zeros(len(Z), dtype=np.float64)
    idx = 0
    for r, cnt in enumerate(row_counts):
        values[r] = out[idx : idx + cnt].mean()
        idx += cnt

    n_rows = Z.shape[0]
    Ze = np.concatenate([np.ones((n_rows, 1)), Z.astype(np.float64)], axis=1)
    W = np.sqrt(weights)[:, None]
    beta, *_ = np.linalg.lstsq(Ze * W, values * np.sqrt(weights), rcond=None)
    return (
        beta[1:],
        float(beta[0]),
        {
            "n_rows": n_rows,
            "full_value": float(values[-2]),
            "empty_value": float(values[-1]),
        },
    )


def kernel_shap_blocks(
    eval_fn,
    obs: np.ndarray,
    layout: ObsLayout,
    ref_obs: np.ndarray,
    n_pairs: int = 512,
    bg_k: int = 32,
    seed: int = 0,
    batch: int = 4096,
    include_history: bool = False,
):
    """obs 15/18 块玩家（13 原生 + [3 历史组] + snr + eps）的 KernelSHAP。

    ref_obs 为 (obs_dim,) 单参照 → bg_k 份同一背景；为 (n_bg, obs_dim)
    多行 → 取前 bg_k 行作为经验背景分布（R_boot）。
    """
    ref = np.asarray(ref_obs, dtype=np.float64)
    if ref.ndim == 2:
        bg = ref
    else:
        bg = np.repeat(ref[None, :], bg_k, axis=0)
    if len(bg) > bg_k:
        bg = bg[:bg_k]

    def mask_fn(bg_row, active):
        return mask_player_obs(obs, layout, active, bg_row, include_history=include_history)

    return kernel_shap(
        eval_fn, obs, layout.n_players_for(include_history), bg, mask_fn, n_pairs=n_pairs, seed=seed, batch=batch
    )


def kernel_shap_channels(
    eval_fn,
    x_in: np.ndarray,
    ref_in: np.ndarray,
    n_channels: int,
    n_pairs: int = 512,
    bg_k: int = 32,
    seed: int = 0,
    batch: int = 4096,
):
    """meta 通道级 (K, n_channels) 输入的 KernelSHAP。

    玩家 = 通道 c；联盟掩码 = 所有用户在通道 c 上取参照值（参照通道分布）。
    """
    x = np.asarray(x_in, dtype=np.float64)
    ref = np.asarray(ref_in, dtype=np.float64)
    if ref.ndim == 1:  # 通道均值向量 (n_channels,) → 广播到所有用户
        ref = np.broadcast_to(ref[None, :], (x.shape[0], x.shape[1]))
    bg = ref if ref.ndim == 3 else np.repeat(ref[None, :, :], bg_k, axis=0)
    if len(bg) > bg_k:
        bg = bg[:bg_k]

    def mask_fn(bg_row_flat, active):
        out = bg_row_flat.reshape(x.shape).copy()
        out[:, active] = x[:, active]
        return out.reshape(-1)

    def eval_fn_reshaped(t):
        # kernel_shap 内部展平为 (B, K·C)，meta 前向需要 (B, K, C)
        return eval_fn(t.reshape(t.shape[0], x.shape[0], x.shape[1]))

    return kernel_shap(
        eval_fn_reshaped, x.reshape(-1), n_channels, bg.reshape(len(bg), -1),
        mask_fn, n_pairs=n_pairs, seed=seed, batch=batch,
    )
