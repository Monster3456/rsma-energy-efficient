"""P1 guard-plateau 早停判定单测（纯函数；默认关 → 永不触发 = 零影响）。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.fast

from src.training.main import guard_plateau_stop


def test_disabled_never_stops():
    """开关默认关：任何 stall/episode 组合都不触发（官方口径零变化）。"""
    assert guard_plateau_stop(stall_count=100, patience=2, episode_count=999999,
                              min_episodes=0, enabled=False) is False


def test_requires_min_episodes():
    """未训够 min_episodes 不早停（避免课程/挖掘未展开就停）。"""
    assert guard_plateau_stop(stall_count=5, patience=2, episode_count=2999,
                              min_episodes=3000, enabled=True) is False


def test_requires_patience():
    """stall 未达 patience 不早停。"""
    assert guard_plateau_stop(stall_count=1, patience=2, episode_count=4000,
                              min_episodes=3000, enabled=True) is False
    assert guard_plateau_stop(stall_count=2, patience=2, episode_count=4000,
                              min_episodes=3000, enabled=True) is True


def test_boundary_exact_min():
    """恰好到达 min_episodes 时允许停（≥ 语义）。"""
    assert guard_plateau_stop(stall_count=2, patience=2, episode_count=3000,
                              min_episodes=3000, enabled=True) is True


def test_patience_at_least_one():
    """patience 下限 1（防误配 0 导致永不触发/除零）。"""
    assert guard_plateau_stop(stall_count=0, patience=0, episode_count=4000,
                              min_episodes=3000, enabled=True) is False
    assert guard_plateau_stop(stall_count=1, patience=0, episode_count=4000,
                              min_episodes=3000, enabled=True) is True