"""Worker α 死维语义锁定。

group_level_alpha=True（默认）时，env._parse_action 用 Meta 的 split_mask 整体覆盖
Worker 的 α 输出块——该块是死维。本测试把这一语义锁进门禁，防止未来有人误用
Worker α（或回归到非 group-alpha 行为）而不自知。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.fast

from src.config import Config
from src.env.rsma_env import RSMAEnergyEnv


def _make_env(group_level_alpha: bool, K: int = 8, T: int = 4) -> RSMAEnergyEnv:
    cfg = Config()
    cfg.system.K = K
    cfg.system.T = T
    cfg.hierarchical.group_level_alpha = group_level_alpha
    cfg.seed = 0
    return RSMAEnergyEnv(cfg)


def _action_with_alpha(env, alpha_block: float) -> np.ndarray:
    """3K 动作：α 块置 alpha_block，其余全 0.0（d=0.5 映射后、p 等随环境解释）。"""
    a = np.zeros(env.action_dim, dtype=np.float64)
    a[: env.K] = alpha_block
    return a


def test_worker_alpha_ignored_under_group_alpha():
    """group_level_alpha=True：α 块两个极端值产生完全相同的解析 α（= split_mask）。"""
    env = _make_env(group_level_alpha=True)
    env.reset(seed=0)
    env.set_split_mask(np.full(env.K, 0.3, dtype=np.float32))
    alpha_lo, p_lo, _ = env._parse_action(_action_with_alpha(env, -1.0))
    alpha_hi, p_hi, _ = env._parse_action(_action_with_alpha(env, 1.0))
    np.testing.assert_allclose(alpha_lo, alpha_hi)
    np.testing.assert_allclose(alpha_lo, np.full(env.K, 0.3), atol=1e-9)  # 取 split_mask
    np.testing.assert_allclose(p_lo, p_hi)


def test_worker_alpha_used_when_group_alpha_off():
    """group_level_alpha=False：α 来自 Worker α 块，(a+1)/2 映射后乘 split_mask。"""
    env = _make_env(group_level_alpha=False)
    env.reset(seed=0)
    env.set_split_mask(np.ones(env.K, dtype=np.float32))  # 不 gate
    alpha_lo, _, _ = env._parse_action(_action_with_alpha(env, -1.0))
    alpha_hi, _, _ = env._parse_action(_action_with_alpha(env, 1.0))
    assert alpha_lo[0] == pytest.approx(0.0, abs=1e-9)
    assert alpha_hi[0] == pytest.approx(1.0, abs=1e-9)


def test_worker_alpha_target_guarded():
    """分析侧：worker_targets 对 α 目标给出明确死维报错（防止误解释）。"""
    from src.analysis.targets import worker_targets

    with pytest.raises(ValueError, match="死维"):
        worker_targets(None, "alpha_avg")