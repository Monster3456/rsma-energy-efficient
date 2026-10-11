"""Expected Gradients：DeepSHAP 族模型内省反向传播归因（自实现，无依赖）。

Expected Gradients（= SHAP GradientExplainer）把 Shapley 值的期望-梯度形式
按参照/背景 MC 估计：

    φ_i = E_{α~U(0,1), x~P_ref}[(x_i − x'_i) · ∇_i f(x' + α·(x − x'))]

- 单点参照（R_zero / R_means / R_neutral）= 固定 x'，退化为参照形式（文献 2
  的 DeepLIFT/DeepSHAP 参照口径，IG 路径）；
- 经验背景（R_boot）= x' ~ 收集状态分布（完整 Expected Gradients）。

块级聚合与 KernelSHAP 语义一致：per-user 块取 K 个用户分量之和（「整个块的联合
效应」），snr/eps 取单分量。返回 (n_players=15,) 与 SHAP_PLAYER_NAMES 对齐。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from src.analysis.features import ObsLayout, player_slices


def eg_blocks(
    fn: Callable,
    obs: np.ndarray,
    layout: ObsLayout,
    ref_obs: np.ndarray,
    n_eg: int = 256,
    seed: int = 0,
    include_history: bool = False,
) -> np.ndarray:
    """对 obs 算 15/18 块的 Expected-Gradients 归因（块 = Σ 分量）。

    Args:
        fn: 可微目标 fn(tensor (B, obs_dim)) -> tensor (B,)（输入 requires_grad）。
        obs: (obs_dim,) 被解释状态。
        layout: 观测布局。
        ref_obs: (obs_dim,) 单点参照，或 (n_bg, obs_dim) 背景分布。
        n_eg: MC 采样数（α 路径 × 背景抽取）。
        seed: 确定性随机源。
        include_history: True 时把历史滞后块按特征类型并成 3 个组玩家（共 18 块）。

    Returns:
        (n_players,) 归因向量（与 player_names(layout, include_history) 对齐）。
    """
    x = torch.from_numpy(np.asarray(obs, dtype=np.float64)).float()
    ref = np.asarray(ref_obs, dtype=np.float64)
    single_ref = ref.ndim == 1
    rng = np.random.default_rng(seed)

    acc = torch.zeros(x.shape[0], dtype=torch.float32)
    alphas = rng.random(n_eg)
    for a in alphas:
        if single_ref:
            xp = torch.from_numpy(ref).float()
        else:
            row = rng.integers(0, ref.shape[0])
            xp = torch.from_numpy(ref[row]).float()
        x_interp = (xp + a * (x - xp)).requires_grad_(True)
        val = fn(x_interp.unsqueeze(0))
        (grad,) = torch.autograd.grad(val.sum(), x_interp, create_graph=False)
        acc += (x - xp) * grad.detach()

    comp = acc.numpy() / n_eg
    slices = [inds for _, inds in player_slices(layout, include_history)]
    out = np.array([comp[sl].sum() for sl in slices], dtype=np.float64)
    return out


def eg_channels(
    fn: Callable,
    x_in: np.ndarray,
    ref_in: np.ndarray,
    n_channels: int,
    n_eg: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """meta 通道级 (K, n_channels) 输入的 Expected-Gradients 归因（玩家 = 通道）。

    per-user 展平 (K·C,)；通道 c 的分量索引 = {c + k·C : k}，归因 = 分量和。
    """
    x = torch.from_numpy(np.asarray(x_in, dtype=np.float64).reshape(-1)).float()
    ref = np.asarray(ref_in, dtype=np.float64)
    if ref.ndim == 1:  # 通道均值向量 (C,) → 广播到所有用户 (K, C)
        ref = np.broadcast_to(ref[None, :], (x_in.shape[0], x_in.shape[1]))
    single_ref = ref.ndim == 2
    rng = np.random.default_rng(seed)

    acc = torch.zeros(x.shape[0], dtype=torch.float32)
    for a in rng.random(n_eg):
        if single_ref:
            xp = torch.from_numpy(ref.reshape(-1)).float()
        else:
            xp = torch.from_numpy(ref[rng.integers(0, ref.shape[0])].reshape(-1)).float()
        x_interp = (xp + a * (x - xp)).requires_grad_(True)
        val = fn(x_interp.view(1, -1, n_channels))
        (grad,) = torch.autograd.grad(val.sum(), x_interp, create_graph=False)
        acc += (x - xp) * grad.detach()

    comp = acc.numpy() / n_eg
    C = int(n_channels)
    return np.array([comp[c::C].sum() for c in range(C)], dtype=np.float64)
