"""A-prune 观测剪枝测试（可解释性取证驱动，默认关 = 官方口径零变化）。

`obs_prune_constant_blocks=True` 移除固定 SNR 评估点内零方差的 3 个原生块
（arrivals / λ桶 / Dinkelbach-η，块 3/6/10；后两者仅 state_aug 语义），
per-user 13→10 块；验证：默认关闭、维度降级、块对齐（保留块/历史/全局逐位
一致）、state_aug=False 时只剪 arrivals。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.fast

from src.analysis.features import ObsLayout, player_names, player_slices
from src.config import Config
from src.env.rsma_env import RSMAEnergyEnv

_NATIVE_KEEP = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12]


def _cfg(prune: bool, K: int = 8, H: int = 0, state_aug: bool = True) -> Config:
    cfg = Config()
    cfg.system.K = K
    cfg.system.T = 6
    cfg.system.obs_history_window = H
    cfg.system.obs_history_features = ("h_gains", "queue", "power")
    cfg.system.obs_prune_constant_blocks = prune
    cfg.lyapunov.state_aug_enabled = state_aug
    cfg.seed = 0
    return cfg


def test_config_default_off():
    """默认关闭：官方口径零变化。"""
    cfg = Config()
    assert cfg.system.obs_prune_constant_blocks is False
    env = RSMAEnergyEnv(cfg)
    assert env.obs_dim == 13 * 24 + 3  # K=24 默认 → 315


def test_prune_dim_no_history():
    env = RSMAEnergyEnv(_cfg(True, K=8, H=0))
    assert env.obs_dim == 10 * 8 + 3
    obs, _ = env.reset(seed=0)
    assert obs.shape == (83,)


def test_prune_block_alignment_h4():
    """保留块/历史区/全局尾与未剪布局逐位一致；剪掉的 3 块被移除。"""
    K = 8
    full = RSMAEnergyEnv(_cfg(False, K=K, H=4))
    pruned = RSMAEnergyEnv(_cfg(True, K=K, H=4))
    assert full.obs_dim == 25 * K + 3 and pruned.obs_dim == 22 * K + 3
    of, _ = full.reset(seed=7)
    op, _ = pruned.reset(seed=7)
    # 保留的原生块：pruned 行 r == full 行 KEEP[r]
    for r, j in enumerate(_NATIVE_KEEP):
        np.testing.assert_allclose(op[r * K : (r + 1) * K], of[j * K : (j + 1) * K])
    # 历史区：pruned 行 10..21 == full 行 13..24（剪枝只删原生区前部）
    np.testing.assert_allclose(op[10 * K : 22 * K], of[13 * K : 25 * K])
    # 全局尾
    np.testing.assert_allclose(op[-3:], of[-3:])


def test_prune_state_aug_off_drops_only_arrivals():
    """state_aug=False 时块 6/10 是 interference/distance（有信息），只剪 arrivals。"""
    K = 8
    full = RSMAEnergyEnv(_cfg(False, K=K, H=0, state_aug=False))
    pruned = RSMAEnergyEnv(_cfg(True, K=K, H=0, state_aug=False))
    assert full.obs_dim == 13 * K + 3
    assert pruned.obs_dim == 12 * K + 3  # 只少 1 块
    of, _ = full.reset(seed=3)
    op, _ = pruned.reset(seed=3)
    keep = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for r, j in enumerate(keep):
        np.testing.assert_allclose(op[r * K : (r + 1) * K], of[j * K : (j + 1) * K])


def test_prune_layout_and_players():
    """分析侧布局：obs_dim、玩家数、玩家名与切片对齐剪枝空间。"""
    layout = ObsLayout(K=8, H=4, history_features=("h_gains", "queue", "power"), prune_constants=True)
    assert layout.d_user_eff == 22
    assert layout.obs_dim() == 22 * 8 + 3
    assert layout.n_players == 12  # 10 原生 + snr + eps
    assert layout.n_players_for(include_history=True) == 15
    names = player_names(layout, include_history=True)
    assert "arrivals" not in names and "lambda_bucket" not in names
    assert names[0] == "h_gains" and names[9] == "qos_priority"
    sl = player_slices(layout, include_history=True)
    # 剪枝空间的 hist_queue 组行号 = native_active(10) + lag*3 + f(1)，共 H=4 个 lag
    hist_q = sl[11][1]
    expect = set()
    for lag in range(layout.H):
        row0 = (layout.native_active + lag * 3 + 1) * 8
        expect.update(range(row0, row0 + 8))
    assert set(hist_q.tolist()) == expect