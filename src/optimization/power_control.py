"""功率控制模块

实现功率约束投影和功率分割后处理。
确保每用户瞬时功率 p_k <= P_k_max，并按 split_mask 正确分割公共/私有流功率。
"""
import numpy as np
from typing import Tuple


def project_power(p_total: np.ndarray, P_max: float) -> np.ndarray:
    """将功率投影到可行域 [0, P_max]。
    
    Args:
        p_total: 各用户总功率, shape (K,)
        P_max: 单用户最大功率约束
    
    Returns:
        投影后的功率向量, shape (K,)
    """
    return np.clip(p_total, 0.0, P_max)


def split_power(
    p_total: np.ndarray,
    alpha: np.ndarray,
    split_mask: np.ndarray,
    P_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """按分裂掩码和功率分割比例，将总功率分为公共流和私有流功率。
    
    对于 split_mask[k] = 1 的用户:
        p_com_k = alpha_k * p_total_k
        p_pri_k = (1 - alpha_k) * p_total_k
    对于 split_mask[k] = 0 的用户 (非分裂):
        p_com_k = 0
        p_pri_k = p_total_k
    
    Args:
        p_total: 各用户总功率, shape (K,)
        alpha: 公共流功率比例 ∈ [0, 1], shape (K,)
        split_mask: 分裂掩码 ∈ {0, 1}^K, shape (K,)
        P_max: 单用户最大功率约束
    
    Returns:
        p_com: 公共流功率, shape (K,)
        p_pri: 私有流功率, shape (K,)
    """
    # 非分裂用户的 alpha 强制为 0
    alpha_effective = alpha * split_mask
    
    # 计算公共流和私有流功率
    p_com = alpha_effective * p_total
    p_pri = p_total - p_com
    
    # 安全裁剪（防止数值误差导致负值）
    p_com = np.clip(p_com, 0.0, P_max)
    p_pri = np.clip(p_pri, 0.0, P_max)
    
    return p_com, p_pri


def normalize_power(
    p_com: np.ndarray,
    p_pri: np.ndarray,
    P_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """若总功率超限，按比例缩放使 p_com + p_pri <= P_max。
    
    Args:
        p_com: 公共流功率, shape (K,)
        p_pri: 私有流功率, shape (K,)
        P_max: 单用户最大功率约束
    
    Returns:
        缩放后的 (p_com, p_pri)
    """
    p_total = p_com + p_pri
    exceed_mask = p_total > P_max
    
    if np.any(exceed_mask):
        scale = np.where(exceed_mask, P_max / (p_total + 1e-8), 1.0)
        p_com = p_com * scale
        p_pri = p_pri * scale
    
    return p_com, p_pri
