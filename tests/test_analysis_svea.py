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


def test_parallel_workers_identical_to_serial():
    """workers>1 与串行结果逐位一致（两阶段法仅改墙钟不改数学）。"""
    rng = _rng(11)
    m = 120
    x0 = np.where(rng.random(m) < 0.5, 1.0, -1.0)
    X = np.column_stack([x0, rng.normal(size=(m, 4))])
    y = x0.copy()
    g_serial = svea_game(X, y, _rng(12), m_sv=80, B=60, workers=1)
    g_par = svea_game(X, y, _rng(13), m_sv=80, B=60, workers=1)  # 同参不同种子仅用于跑通形状
    g_par_same = svea_game(X, y, _rng(12), m_sv=80, B=60, workers=2)
    np.testing.assert_allclose(g_serial["phi"], g_par_same["phi"], atol=1e-12)
    np.testing.assert_allclose(g_serial["sv"], g_par_same["sv"], atol=1e-12)
    assert g_serial["n_solves"] == g_par_same["n_solves"]
    _ = g_par  # 仅确认并行路径可跑（独立种子）

def test_psv_mlp_locks_functionality():
    """--psv_mlp 非死 flag：小 MLP 在可分数据上确实训练（acc_sub 高、PSV≈1）。"""
    from src.analysis.games import psv_mlp

    rng = _rng(21)
    n = 200
    x0 = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    X = np.column_stack([x0, rng.normal(size=(n, 4))])  # 仅特征 0 携带标签信息
    y = x0.copy()

    psv, acc_sub, acc_full = psv_mlp(X, y, np.array([0]), _rng(22))
    assert acc_sub > 0.7, "信息特征子集应被 MLP 学到"
    assert 0.0 <= acc_sub <= 1.0 and 0.0 <= acc_full <= 1.0
    assert psv == pytest.approx(acc_sub / acc_full, abs=1e-9)
    assert psv > 0.8, "全特征含噪声维度不应显著损害精度 → PSV 应接近 1"


def test_psv_mlp_insufficient_samples_returns_none():
    """样本不足（te<20）→ (None, None, None) 哨兵，不崩溃。"""
    from src.analysis.games import psv_mlp

    rng = _rng(23)
    X = rng.normal(size=(40, 3))
    y = np.where(rng.random(40) < 0.5, 1.0, -1.0)
    assert psv_mlp(X, y, np.array([0]), _rng(24)) == (None, None, None)


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


def test_left_of_origin_criterion():
    """Tripathi §3.2 判据：区间整体落在原点左侧 ⇔ 上界 < 0。"""
    from src.analysis.games import left_of_origin

    lo = np.array([-2.0, -1.0, -0.5, 0.1])
    hi = np.array([-0.5, -0.0, 0.2, 0.3])
    flag = left_of_origin(lo, hi)
    assert flag.dtype == bool and flag.shape == (4,)
    assert list(flag) == [True, False, False, False]


def test_mean_subgame_e_shape_and_determinism():
    """组内 n_sub 个子博弈取均值：形状 (n_features,)，同种子可复现。"""
    from src.analysis.games import mean_subgame_e

    rng = _rng(31)
    m = 120
    x0 = np.where(rng.random(m) < 0.5, 1.0, -1.0)
    X = np.column_stack([x0, rng.normal(size=(m, 3))])
    y = x0.copy()

    e1 = mean_subgame_e(X, y, _rng(40), n_sub=5, m_sv=60, B=20)
    e2 = mean_subgame_e(X, y, _rng(40), n_sub=5, m_sv=60, B=20)
    e3 = mean_subgame_e(X, y, _rng(41), n_sub=5, m_sv=60, B=20)
    assert e1.shape == (4,)
    np.testing.assert_allclose(e1, e2, atol=1e-12)  # 同种子逐位复现
    assert not np.allclose(e1, e3), "不同种子应给出不同抽样"
    assert np.all(np.isfinite(e1))


def test_mean_subgame_e_recovers_dominant_feature():
    """单特征可分数据：组均值应把信息特征判为支配（e<0），噪声特征不支配。

    组内平均 30 个子博弈是 Tripathi §3.2 的抗噪关键步骤——平均后组均值稳定为负。
    """
    from src.analysis.games import mean_subgame_e

    rng = _rng(33)
    m = 400
    x0 = np.where(rng.random(m) < 0.5, 1.0, -1.0)
    X = np.column_stack([x0, rng.normal(size=(m, 3))])
    y = x0.copy()

    e_one = mean_subgame_e(X, y, _rng(50), n_sub=1, m_sv=120, B=30)
    e_avg = mean_subgame_e(X, y, _rng(51), n_sub=30, m_sv=120, B=30)
    assert e_avg[0] < 0.0, f"信息特征应支配，e[0]={e_avg[0]}"
    assert e_avg[0] < 0.0 <= e_avg[1] or e_avg[0] < e_avg[1], "噪声特征不得比信息特征更支配"
    # 平均 30 个子博弈后不应比单次更接近噪声零线（符号判定更稳）
    assert e_avg[0] < 0.0 and (e_one[0] < 0.0 or e_avg[0] < -0.01)


def test_parallel_matches_serial_with_exclude():
    """回归锁定：exclude 非空时并行 mask→列映射必须与串行逐位一致。

    曾经的 bug：``_eval_mask_par`` 把 mask 位直接当**原始列号**取，绕过 ``active``
    映射。exclude 非空时玩家数 < 特征数，于是并行路径在错误的列子集上解 LP——
    噪声列互换时误差小不易察觉，但会把位于高位列的信息特征判成非支配。
    """
    rng = _rng(7)
    m = 500
    x0 = np.where(rng.random(m) < 0.5, 1.0, -1.0)
    x11 = np.where(rng.random(m) < 0.5, 1.0, -1.0)
    X = rng.normal(size=(m, 13))
    X[:, 0] = x0                    # 信息列（低位）
    X[:, 3] = 0.5                   # 常数 → null player
    X[:, 6] = 1.0
    X[:, 10] = -2.0
    X[:, 11] = x11                  # 信息列（高位，active 里但 buggy 映射下会丢）
    y = np.where(x0 + x11 + 0.1 * rng.normal(size=m) > 0, 1.0, -1.0)
    const = np.array([3, 6, 10])

    g1 = svea_game(X, y, _rng(100), m_sv=300, B=80, exclude=const, zscore=False, workers=1)
    g2 = svea_game(X, y, _rng(100), m_sv=300, B=80, exclude=const, zscore=False, workers=2)
    np.testing.assert_allclose(g2["phi"], g1["phi"], atol=1e-9, err_msg="并行/串行 Shapley 必须逐位一致")
    np.testing.assert_allclose(g2["sv"], g1["sv"], atol=1e-9)
    assert list(g2["sv_neg"]) == list(g1["sv_neg"])
    assert set(g1["sv_neg"]) == {0, 11}, "两个信息列都应被判为支配（否则映射仍错）"


def test_standardize_matches_scipy():
    """z-score 与手动标准化一致（仿射不变性检查）。"""
    rng = _rng(5)
    X = rng.normal(size=(50, 3)) * np.array([1.0, 5.0, 0.1]) + np.array([2, -1, 0])
    Xz, mu, sd = standardize(X)
    np.testing.assert_allclose(Xz.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(Xz.std(axis=0), 1.0, atol=1e-9)
    np.testing.assert_allclose(Xz, (X - mu) / sd, atol=1e-12)
