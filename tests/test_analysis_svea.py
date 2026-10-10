"""SVEA 分类博弈核的快速单元测试（闭式真值）。

覆盖：∅ 误差闭式、子集单调性、oracle 线性博弈 MC Shapley 效率性、
单特征支配数据的阈值 0 选择、t-CI 形状。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.fast

from src.analysis.games import (
    standardize,
    svea_game,
    t_ci,
    tr_er,
    tr_er_empty,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_tr_er_empty_closed_form():
    """截距 only 分类器误差：2·min(p, 1−p)。"""
    y = np.array([1.0] * 7 + [-1.0] * 3)
    assert tr_er_empty(y) == pytest.approx(2.0 * 0.3, abs=1e-9)
    y_all_pos = np.ones(5)
    assert tr_er_empty(y_all_pos) == pytest.approx(0.0, abs=1e-9)


def test_tr_er_monotone_nested_subsets():
    """S⊂T ⇒ tr_er(S) ≥ tr_er(T)：LP 精确性 + 假设空间包含论证。"""
    rng = _rng(1)
    X = rng.normal(size=(40, 4))
    y = np.where(X[:, 0] + 0.2 * X[:, 1] + 0.5 * rng.normal(size=40) > 0, 1.0, -1.0)
    tr_sub = tr_er(X[:, [0, 1]], y)
    tr_super = tr_er(X[:, [0, 1, 2]], y)
    assert tr_sub >= tr_super - 1e-6


def test_mc_shapley_oracle_linear():
    """oracle v(S)=Σ_{j∈S} a_j：MC 排列 Shapley 应精确恢复 a 且满足效率性。"""
    a = np.array([1.0, 2.0, 0.5, 3.0])
    n = len(a)
    (1 << n) - 1
    phi, n_solves = _mc_oracle(a, B=1200, seed=2)
    assert phi == pytest.approx(a, abs=0.03)
    assert phi.sum() == pytest.approx(a.sum(), abs=1e-6)  # 效率性（telescoping）
    assert n_solves <= 2 * 1200 * n + 2


def _mc_oracle(a, B, seed):
    from src.analysis.games import mc_permutation_shapley

    n = len(a)

    def eval_subset(mask):
        return sum(a[j] for j in range(n) if mask & (1 << j))

    return mc_permutation_shapley(eval_subset, n, B, _rng(seed))


def test_svea_dominant_feature_toy():
    """只有特征 0 携带可分离信号 → SVEA_neg == {0}，e_0 为全局最小。"""
    rng = _rng(3)
    m = 240
    x0 = np.where(rng.random(m) < 0.5, 1.0, -1.0)  # 干净信号，可分
    X = np.column_stack([x0, rng.normal(size=(m, 4))])  # 其余为纯噪声
    y = x0.copy()
    game = svea_game(X, y, _rng(4), m_sv=200, B=400)
    e = game["sv"]
    assert e[0] < 0.0
    assert e[1:].mean() > 0.0
    assert 0 in game["sv_neg"]
    assert e[0] < e[1:].min()


def test_svea_null_player_excluded():
    """零方差特征 = null player：排除后 φ=0、SVEA>0 永不支配；其他特征 Shapley 不变。"""
    rng = _rng(9)
    m = 240
    x0 = np.where(rng.random(m) < 0.5, 1.0, -1.0)
    X = np.column_stack([x0, rng.normal(size=(m, 4)), np.ones(m) * 0.5])  # 列 5 常数
    y = x0.copy()
    from src.analysis.games import constant_feature_cols

    const = constant_feature_cols(X)
    assert 5 in const
    game = svea_game(X, y, _rng(4), m_sv=200, B=300, exclude=const)
    assert 5 not in game["sv_neg"]
    assert abs(game["phi"][5]) < 1e-9          # 排除的 null player → φ=0
    assert game["sv"][5] > 0.0                 # e = c̃/n > 0（原始 n=13 口径）
    game_red = svea_game(X[:, :5], y, _rng(4), m_sv=200, B=300)
    np.testing.assert_allclose(game["phi"][:5], game_red["phi"], atol=0.05)


def test_t_ci_shapes():
    """G 组 SVEA 矩阵 → 均值/区间形状与顺序。"""
    G, n_feat = 5, 3
    e_matrix = np.stack([np.linspace(-1, 1, n_feat) + gi * 0.1 for gi in range(G)])
    mean, lo, hi = t_ci(e_matrix)
    assert mean.shape == hi.shape == lo.shape == (n_feat,)
    assert np.all(lo <= mean) and np.all(mean <= hi)


def test_standardize_matches_scipy():
    """z-score 与手动标准化一致（仿射不变性检查）。"""
    rng = _rng(5)
    X = rng.normal(size=(50, 3)) * np.array([1.0, 5.0, 0.1]) + np.array([2, -1, 0])
    Xz, mu, sd = standardize(X)
    np.testing.assert_allclose(Xz.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(Xz.std(axis=0), 1.0, atol=1e-9)
    np.testing.assert_allclose(Xz, (X - mu) / sd, atol=1e-12)
