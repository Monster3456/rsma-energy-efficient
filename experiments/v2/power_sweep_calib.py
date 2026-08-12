"""λ 标定功率扫描：SDMA（队列加权功率）在固定 SNR 下扫描功率预算档，测"功率→交付"边际收益。

用法: python3 experiments/v2/power_sweep_calib.py [--snr 15] [--episodes 10]
输出: 各功率档的平均功率与平均 served_bits，打印 d(served)/dP 斜率（λ 候选）
"""
import argparse
import sys
sys.path.insert(0, '/Users/libo/PycharmProjects/rsma-energy-efficient')

import numpy as np
from src.env.rsma_env import RSMAEnergyEnv
from scripts.baselines.sdma_baseline import SDMABaseline


class ScaledSDMA(SDMABaseline):
    """SDMA 基线 + 功率预算缩放（用于边际收益扫描）"""

    def __init__(self, config, scale: float):
        super().__init__(config)
        self.P_budget = self.K * self.P_k_max * scale  # 功率预算 × scale


def run_sdma_power_sweep(snr_db: float, n_episodes: int, scales=(0.2, 0.4, 0.6, 0.8, 1.0)):
    from experiments.v2.config import make_v2_config
    cfg = make_v2_config()
    cfg.system.csi_mode = "imperfect"
    cfg.system.snr_domain_randomization = False  # 固定 SNR
    # SDMA 是 3K 动作（alpha+p+d），不需要 hierarchical/Meta
    cfg.hierarchical.enabled = False
    cfg.seed = 42

    results = {}
    for scale in scales:
        env = RSMAEnergyEnv(cfg)
        baseline = ScaledSDMA(cfg, scale)
        served_list, power_list = [], []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            env.current_snr_db = snr_db
            ep_served, ep_power, n_steps = 0.0, 0.0, 0
            for _ in range(cfg.system.T):
                action = baseline.select_action(obs, env)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_served += float(np.sum(info.get("served", np.zeros(cfg.system.K))))
                ep_power += float(info.get("total_power", 0.0))
                n_steps += 1
                if terminated or truncated:
                    break
            served_list.append(ep_served)
            power_list.append(ep_power / max(n_steps, 1))  # 平均每步功率
        results[scale] = {
            "avg_served": float(np.mean(served_list)),
            "avg_power": float(np.mean(power_list)),
        }
        print(f"scale={scale:.1f}: avg_power={results[scale]['avg_power']:.2f}W "
              f"avg_served={results[scale]['avg_served']:.1f} bits/ep")

    # 边际收益：相邻点斜率的中位数（稳健估计）
    xs = np.array([results[s]["avg_power"] for s in scales])
    ys = np.array([results[s]["avg_served"] for s in scales])
    slopes = np.diff(ys) / np.maximum(np.diff(xs), 1e-8)
    med_slope = float(np.median(slopes))
    print(f"\n[λ 标定] 中位边际收益 d(served)/dP = {med_slope:.3f} bits/W")
    print(f"[λ 标定] 候选 λ = {med_slope:.3f}（SDMA 工作点斜率，PPO 收敛到 ≈SDMA 功率）")
    print(f"[λ 标定] 候选 λ/2 = {med_slope / 2:.3f}（保留功率换交付空间，PPO 停在高一些的功率）")
    return med_slope


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr", type=float, default=15.0)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    run_sdma_power_sweep(args.snr, args.episodes)
