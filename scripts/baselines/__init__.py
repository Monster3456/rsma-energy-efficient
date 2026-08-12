"""基线算法模块"""
from scripts.baselines.sdma_baseline import SDMABaseline
from scripts.baselines.noma_baseline import NOMABaseline
from scripts.baselines.equal_power import EqualPowerBaseline
from scripts.baselines.adaptive_power_baseline import (
    AdaptivePowerBaseline,
    WaterFillingBaseline,
)

__all__ = [
    "SDMABaseline",
    "NOMABaseline",
    "EqualPowerBaseline",
    "AdaptivePowerBaseline",
    "WaterFillingBaseline",
]
