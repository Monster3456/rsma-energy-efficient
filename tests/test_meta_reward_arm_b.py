"""Arm B：Meta 分组激励修复 + 分组进奖励（10-23-hierarchical-opt）。

回归锁定两处曾致分组坍缩的问题：
  1. ``group_balance`` 原先只统计被使用组，恒单组时 std=0 → bonus 取**最大值**，
     坍缩侧无反向梯度；
  2. 多样性/均衡项随 ``meta_reward_scale=0.02`` 一起缩放，有效幅动仅 0.015
     （|MetaR|≈19 的 0.08%），被主项淹没。
以及组级 DVQ：给定 group_ids 时 Y 按组核算，使分组改变奖励。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.fast

from src.config import Config
from src.optimization.lyapunov import LyapunovController
from src.optimization.meta_reward import compute_meta_reward


def _stats(group_ids, n_groups=4, steps=20):
    return {
        "ee_sum": 100.0, "overflow_sum": 10.0, "steps": steps,
        "group_ids": np.asarray(group_ids), "n_groups": n_groups,
    }


def test_balance_penalizes_collapse():
    """恒单组必须拿不到满格 balance（旧实现 counts=[24]、std=0、bonus=0.5 满格）。"""
    single = _stats(np.full(24, 3), n_groups=4)          # 24 用户全进组 3
    even = _stats(np.repeat([0, 1, 2, 3], 6), n_groups=4)  # 6/6/6/6
    r_single = compute_meta_reward(single, reward_scale=1.0, structure_scale=1.0)
    r_even = compute_meta_reward(even, reward_scale=1.0, structure_scale=1.0)
    # 主项相同，差异全来自结构项
    assert r_even > r_single, f"均衡分组应得分更高: {r_even} vs {r_single}"


def test_structure_scale_isolated_from_main():
    """结构项用独立 scale：主项 scale=0.02 时，结构项仍能推动分数。"""
    single = _stats(np.full(24, 0), n_groups=4)
    even = _stats(np.repeat([0, 1, 2, 3], 6), n_groups=4)
    # 只放大主项缩放：若结构项被一起缩放，两者差会被压到 0.02 倍
    a = compute_meta_reward(single, reward_scale=0.02, structure_scale=5.0)
    b = compute_meta_reward(even, reward_scale=0.02, structure_scale=5.0)
    gap = b - a
    assert gap > 1.0, f"结构项应有效推动（gap={gap}），不该被 0.02 压死"
    # 对照：结构 scale=0 时两者应几乎相同
    a0 = compute_meta_reward(single, reward_scale=0.02, structure_scale=0.0)
    b0 = compute_meta_reward(even, reward_scale=0.02, structure_scale=0.0)
    assert abs(b0 - a0) < 1e-9


def test_group_level_dvq_changes_reward():
    """给定 group_ids 时组级 Y_g 生效：分组不同 ⇒ 奖励不同（旧实现分组不影响奖励）。"""
    cfg = Config()
    ly = LyapunovController(K=24, V=cfg.lyapunov.V)
    ly.n_groups = 4
    K = 24
    q = np.full(K, 5.0)
    served = np.full(K, 1.0)
    power = np.full(K, 0.3)
    overflow = np.zeros(K)
    arrivals = np.full(K, 2.0)

    def run(gids):
        ly.reset()
        ly._Y_g = None
        out = ly.compute_reward(
            queue_lengths=q, served_bits=served, power_consumed=power,
            overflow=overflow, arrivals=arrivals, delivery_running=0.9,
            group_ids=np.asarray(gids, dtype=np.int64),
        )
        return out["reward"]

    r_even = run(np.repeat([0, 1, 2, 3], 6))
    r_single = run(np.full(K, 0))
    assert r_even != r_single, "分组不同必须给出不同奖励"
    # 退化：n_groups=1 时走聚合 Y，与 group_ids 无关
    ly.n_groups = 1
    ly.reset(); ly._Y_g = None
    r1 = run(np.repeat([0, 1, 2, 3], 6))
    ly.reset(); ly._Y_g = None
    r2 = run(np.full(K, 0))
    assert r1 == pytest.approx(r2), "n_groups=1 时应走聚合 Y，与分组无关"


def test_default_config_keeps_backward_compat():
    """默认 meta_reward_scale=0.02 不变；新增字段默认值不影响旧路径（structure_scale 仅在显式传时生效）。"""
    cfg = Config()
    assert cfg.hierarchical.meta_reward_scale == 0.02
    assert cfg.hierarchical.meta_structure_scale == 5.0
    # 不传 scale 时行为是 reward_scale=1（纯计算），与旧 API 兼容
    r = compute_meta_reward(_stats(np.full(24, 0)))
    assert np.isfinite(r)
