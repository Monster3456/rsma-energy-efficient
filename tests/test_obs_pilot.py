# -*- coding: utf-8 -*-
"""R1 联合导频：默认关（obs 不变）、开启 obs_dim +1、set_pilot_power 注入与边界。"""
from __future__ import annotations

import numpy as np
import torch
import pytest

from src.config import Config
from src.env.rsma_env import RSMAEnergyEnv

pytestmark = pytest.mark.fast


def _make_env(pilot: bool) -> RSMAEnergyEnv:
    cfg = Config()
    cfg.system.pilot_adaptive = pilot
    env = RSMAEnergyEnv(cfg)
    env.reset(seed=7)
    return env


def test_pilot_off_obs_unchanged():
    base = _make_env(pilot=False).obs_dim
    on = _make_env(pilot=True).obs_dim
    assert on == base + 1


def test_pilot_injection_bound():
    env = _make_env(pilot=True)
    env.set_pilot_power(4.0)
    assert env._pilot_power == 4.0
    assert env.channel._pilot_power == 4.0
    env.set_pilot_power(100.0)   # clip 到 [min, max]
    assert env._pilot_power <= env.pilot_power_max
    env.set_pilot_power(-1.0)
    assert env._pilot_power >= env.pilot_power_min


def test_pilot_obs_tail():
    env = _make_env(pilot=True)
    env.set_pilot_power(8.0)
    obs = env._get_observation()
    assert obs.shape == (env.obs_dim,)
    assert obs[-1] == pytest.approx(8.0 / env.pilot_power_max)


def test_pilot_off_ignored():
    env = _make_env(pilot=False)
    env.set_pilot_power(4.0)   # 不启用时不生效、不改变任何东西
    assert env._pilot_power is None


def test_pilot_learnable_gradient_flow():
    """pilot 是 meta 可学习动作：act→evaluate_actions→backward 梯度到 pilot 头。"""
    from src.models.meta_controller import MetaActorCritic
    m = MetaActorCritic(K=24, d_user=12, n_groups=4, group_level_alpha=True,
                        pilot_enabled=True)
    feats = torch.randn(1, 24, 12)
    _a, gids, _lp, _v = m.act(feats)
    assert 0.25 <= m.last_pilot <= 16.0
    raw_alpha = m.last_raw_alpha                    # (G,)
    raw_pilot = m.last_raw_pilot                    # (1,)
    lp2, _v2, _se, _ge, _te = m.evaluate_actions(
        feats, raw_alpha.unsqueeze(0), gids.unsqueeze(0), raw_pilot.reshape(1, 1))
    loss = -lp2.sum()
    loss.backward()
    assert m.pilot_head.weight.grad is not None
    assert m.pilot_log_std.grad is not None
    # 确定性路径不采样、仍暴露导频
    m.act(feats, deterministic=True)
    assert m.last_pilot is not None