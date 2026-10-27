"""Arm C：目标条件化 ζ（去分组、留分层）。

Meta 输出慢尺度交付目标 ζ，env 将 ζ_norm 作为额外全局观测维注入 Worker，
并把 ζ 作为交付虚拟队列目标，使慢层目标真正改变快层约束压力。
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.fast

from src.config import Config
from src.env.rsma_env import RSMAEnergyEnv
from src.models.meta_controller import MetaActorCritic


def _cfg(zeta: bool) -> Config:
    cfg = Config()
    cfg.system.obs_history_window = 0
    cfg.system.obs_prune_constant_blocks = False
    cfg.ppo.worker_d_user = 13
    cfg.hierarchical.enabled = True
    cfg.hierarchical.group_level_alpha = True
    cfg.hierarchical.meta_zeta_enabled = bool(zeta)
    return cfg


def _meta() -> MetaActorCritic:
    return MetaActorCritic(K=24, d_user=12, hidden_dim=32, n_heads=2,
                           n_groups=4, group_level_alpha=True)


def test_meta_emits_zeta_in_range():
    m = _meta()
    uf = torch.randn(24, 12)
    m.act(uf, deterministic=True)
    z = m.last_zeta
    assert z is not None
    assert m.zeta_min <= z <= m.zeta_max, f"ζ={z} 越界"


def test_env_obs_gains_zeta_dim():
    """zeta_enabled 时观测多一维；关闭时不变（回滚安全）。"""
    e_on = RSMAEnergyEnv(_cfg(True))
    e_off = RSMAEnergyEnv(_cfg(False))
    assert e_on.obs_dim == e_off.obs_dim + 1
    assert e_on.zeta_enabled and not e_off.zeta_enabled


def test_set_delivery_target_updates_obs_and_vq():
    e = RSMAEnergyEnv(_cfg(True))
    e.reset(seed=0)
    e.set_delivery_target(0.80)
    assert e._delivery_target == pytest.approx(0.80)
    assert e.lyapunov.dvq_target == pytest.approx(0.80)
    obs = e._get_observation()
    # ζ_norm 位于 power_budget 之前的最后一位（分层模式再追加 pb）
    assert obs[-2] == pytest.approx(0.80 / e.zeta_cap, abs=1e-6)
    e.set_delivery_target(0.60)
    assert e.lyapunov.dvq_target == pytest.approx(0.60)


def test_zeta_off_backward_compat():
    """zeta_enabled=False 时 set_delivery_target 不写观测、obs_dim 不变。"""
    e = RSMAEnergyEnv(_cfg(False))
    e.reset(seed=0)
    base = e._get_observation().copy()
    e.set_delivery_target(0.7)
    assert e.lyapunov.dvq_target == pytest.approx(0.7)  # VQ 目标仍更新
    obs2 = e._get_observation()
    assert obs2.shape == base.shape
    np.testing.assert_allclose(obs2, base)
