# -*- coding: utf-8 -*-
"""D3 逐点压力向量观测：默认关（obs_dim 不变）、开启 +7、注入 clip、错误形状拒绝。"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import Config
from src.env.rsma_env import RSMAEnergyEnv

pytestmark = pytest.mark.fast


def _make_env(pressure: bool) -> RSMAEnergyEnv:
    cfg = Config()
    cfg.system.obs_pressure_vector = pressure
    env = RSMAEnergyEnv(cfg)
    env.reset(seed=7)
    return env


def test_pressure_off_obs_dim_unchanged():
    env = _make_env(pressure=False)
    base = int(env.obs_dim)
    env_on = _make_env(pressure=True)
    assert env_on.obs_dim == base + 7
    assert env.obs_dim == np.asarray(env._get_observation()).size


def test_pressure_zeros_before_injection():
    env = _make_env(pressure=True)
    obs = env._get_observation()
    assert obs.shape == (env.obs_dim,)
    assert np.allclose(obs[-7:], 0.0)  # 未注入时为中性零


def test_pressure_injection_roundtrip():
    env = _make_env(pressure=True)
    pv = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    env.set_pressure_vector(pv)
    obs = env._get_observation()
    assert np.allclose(obs[-7:], pv)
    # 超界值被 clip 到 [0,1]
    env.set_pressure_vector(np.array([-0.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    obs2 = env._get_observation()
    assert np.allclose(obs2[-7:], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_pressure_wrong_shape_rejected():
    env = _make_env(pressure=True)
    with pytest.raises(ValueError):
        env.set_pressure_vector(np.zeros(6))
    with pytest.raises(ValueError):
        env.set_pressure_vector(np.zeros(8))