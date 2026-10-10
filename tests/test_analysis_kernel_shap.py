"""KernelSHAP 核的快速单元测试。

闭式真值：线性模型 f(x)=w·x 上，参照-边际化 Shapley = w_j(x_j − E[x_j])
（block 口径）。另测块级掩码往返与 15 玩家接线形状。
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.fast

from src.analysis.features import ObsLayout, mask_player_obs
from src.analysis.kernel_shap import (
    kernel_shap,
    kernel_shap_blocks,
    sample_coalitions,
    shapley_kernel_weight,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_shapley_kernel_weight_symmetric():
    """核权重关于 s 与 n−s 对称，端点取 0。"""
    assert shapley_kernel_weight(8, 2) == pytest.approx(shapley_kernel_weight(8, 6))
    assert shapley_kernel_weight(8, 0) == 0.0
    assert shapley_kernel_weight(8, 8) == 0.0
    assert shapley_kernel_weight(8, 4) > 0.0


def test_sample_coalitions_paired():
    """互补配对：每对 (z, ~z)；末两行为全集/空集。"""
    n = 6
    Z, w = sample_coalitions(n, n_pairs=20, rng=_rng(1))
    assert Z.shape == (2 * 20 + 2, n)
    assert Z[-2].all() and not Z[-1].any()
    for i in range(20):
        assert np.array_equal(Z[2 * i], ~Z[2 * i + 1])
    assert np.isfinite(w).all() and (w > 0).all()


def test_kernel_linear_recovery():
    """线性模型闭式真值：φ_j = w_j(x_j − E[x_j])。"""
    n = 6
    rng = _rng(2)
    w = rng.normal(size=n)
    X = rng.normal(size=(400, n))
    x = X[7]
    mean = X.mean(axis=0)
    lin = torch.nn.Linear(n, 1, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(w))
    fn = lambda t: lin(t).squeeze(-1)
    ref = mean

    def mask_fn(bg_row, active):
        out = bg_row.copy()
        out[active] = x[active]
        return out

    phi, phi0, _info = kernel_shap(fn, x, n, np.repeat(ref[None, :], 64, axis=0), mask_fn, n_pairs=512, seed=3)
    expected = w * (x - mean)
    np.testing.assert_allclose(phi, expected, atol=2e-2)
    assert phi0 == pytest.approx(float(ref @ w), abs=2e-2)  # 基线 = E[f]


def test_coalition_masking_roundtrip():
    """块级掩码：非激活块取参照、激活块取 obs；14/15 号玩家联动全局尾。"""
    layout = ObsLayout(K=4, H=0)
    obs = np.arange(layout.obs_dim(), dtype=np.float64) / 100.0
    ref = np.full(layout.obs_dim(), 7.0)
    active = np.zeros(layout.n_players, dtype=bool)
    active[[0, 13]] = True  # 仅块 0 与 snr 玩家激活
    masked = mask_player_obs(obs, layout, active, ref)
    np.testing.assert_allclose(masked[0:4], obs[0:4])  # 块 0 激活
    np.testing.assert_allclose(masked[4:8], ref[4:8])  # 块 1（queue）未激活→参照
    np.testing.assert_allclose(masked[8 : layout.globals_offset], ref[8 : layout.globals_offset])
    g0 = layout.globals_offset
    assert masked[g0 + 1] == pytest.approx(obs[g0 + 1])  # snr 玩家激活（索引 13）
    assert masked[g0 + 2] == pytest.approx(ref[g0 + 2])  # eps 玩家未激活（索引 14）


def test_kernel_shap_blocks_wiring():
    """15 玩家接线：形状、全集/空集端点值、效率性 sanity。"""
    layout = ObsLayout(K=4, H=0)
    rng = _rng(4)
    obs = rng.uniform(size=layout.obs_dim())
    ref = np.zeros(layout.obs_dim())
    fn = lambda t: t.sum(dim=-1)  # 目标 = Σ obs：线性，block 归因 = 块求和
    phi, phi0, _info = kernel_shap_blocks(fn, obs, layout, ref, n_pairs=64, bg_k=8, seed=5)
    assert phi.shape == (layout.n_players,)
    # 线性目标：φ_j = Σ 块分量 = |活跃分量数|·(obs−ref)；参照全零 → 各块分量和
    block_sums = np.concatenate(
        [obs[j * 4 : (j + 1) * 4].sum(axis=0, keepdims=True) for j in range(13)] + [obs[-2:-1], obs[-1:]]
    )
    np.testing.assert_allclose(phi, block_sums, atol=1e-2)
    np.testing.assert_allclose(phi0, 0.0, atol=1e-2)
