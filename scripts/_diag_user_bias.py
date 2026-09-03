"""Run 6 低 SNR 逐用户分配偏向验证：按信道增益排序，看速率/队列/served 的双峰结构。

口径：fixed λ=1.363 + N_r=32。若强用户 R_k 高且队列空、弱用户 R_k≈0 且队列满，
则证实「功率份额绑定信道而非队列」的分配失配。
"""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np

import src.config
import evaluate_snr_sweep_temporal as wrapper

temporal_cls, d_user_eff = wrapper.make_temporal_config_cls(
    4, ("h_gains", "queue", "power"), power_free=True, fixed_lambda=1.363, n_r=16
)
src.config.Config = temporal_cls

spec = importlib.util.spec_from_file_location("evs", REPO / "scripts" / "evaluate_snr_sweep.py")
evs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evs)

CKPT = str(REPO / "logs" / "run12_mw_residual" / "checkpoints" / "best.pt")

snr_db = 0
sigma2 = 10 ** (-snr_db / 10)
cfg = evs.build_config_for_snr(sigma2, "imperfect")
env = evs.RSMAEnergyEnv(cfg)
model, ckpt_obs_dim, err = evs.load_ppo_model(CKPT, env)
meta_ac, T_s, _ = evs.load_meta_model(CKPT, env)
action_fn = evs.make_ppo_action_fn(model, ckpt_obs_dim, env.obs_dim)
K = env.K

R_u = np.zeros(K)
Q_u = np.zeros(K)
D_u = np.zeros(K)
A_u = np.zeros(K)
gains = None
n_ep = 10
for ep in range(n_ep):
    obs, info = env.reset(seed=ep)
    evs._meta_decide_eval(meta_ac, env)
    steps = 0
    for _ in range(env.T):
        action = np.array(action_fn(obs, env), dtype=np.float64).reshape(-1)
        obs, r, term, trunc, info = env.step(action)
        R_u += np.asarray(info["R_total"], dtype=np.float64)
        Q_u += np.asarray(info["mean_queue_per_user"] if "mean_queue_per_user" in info else env.traffic.queue_lengths)
        D_u += np.asarray(info["served"], dtype=np.float64)
        A_u += np.asarray(info["arrivals"], dtype=np.float64)
        if gains is None:
            gains = np.asarray(env._channel_obs.get("channel_gains"), dtype=np.float64).copy()
        steps += 1
        if steps % T_s == 0 and not (term or trunc):
            evs._meta_decide_eval(meta_ac, env)
        if term or trunc:
            break

R_u /= n_ep * steps
Q_u /= n_ep * steps
D_u /= n_ep * steps
A_u /= n_ep * steps
order = np.argsort(-gains)  # 强->弱

print(f"\n=== SNR={snr_db}dB PPO 逐用户（按信道增益降序，每用户每步均值）===")
print(f"{'rank':>4} {'gain':>8} {'R_k':>7} {'Q_k':>7} {'D_k':>7} {'A_k':>7} {'R/A':>6}")
for i, u in enumerate(order):
    print(
        f"{i:>4} {gains[u]:>8.4f} {R_u[u]:>7.3f} {Q_u[u]:>7.2f} {D_u[u]:>7.3f} {A_u[u]:>7.3f} {R_u[u] / max(A_u[u], 1e-9):>6.2f}"
    )

top8, bot8 = order[:8], order[-8:]
print(
    f"\n强8用户: 均值R={R_u[top8].mean():.3f} 均值Q={Q_u[top8].mean():.2f} R/A={R_u[top8].sum() / max(A_u[top8].sum(), 1e-9):.2f}"
)
print(
    f"弱8用户: 均值R={R_u[bot8].mean():.3f} 均值Q={Q_u[bot8].mean():.2f} R/A={R_u[bot8].sum() / max(A_u[bot8].sum(), 1e-9):.2f}"
)
print(f"速率-增益相关: {np.corrcoef(R_u, gains)[0, 1]:.3f} | 速率-队列相关: {np.corrcoef(R_u, Q_u)[0, 1]:.3f}")
