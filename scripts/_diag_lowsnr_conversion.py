"""Run 6 低 SNR 速率→交付转化诊断：拆分 0dB 缺口 = 容量 vs d_ratio vs 队列绑定。

口径：fixed λ=1.363 + N_r=32（与 eval_run6_fixlam_nr32_best.txt 一致）。
每用户每步记录：usable rate R_k、d_ratio、发送前队列 Q_k、实际 served D_k。
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
    4, ("h_gains", "queue", "power"), fixed_lambda=1.363, n_r=32
)
src.config.Config = temporal_cls

spec = importlib.util.spec_from_file_location("evs", REPO / "scripts" / "evaluate_snr_sweep.py")
evs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evs)

CKPT = str(REPO / "logs" / "temporal_worker_h4_fixlam_nr32" / "checkpoints" / "best.pt")

for snr_db in (0, 5):
    sigma2 = 10 ** (-snr_db / 10)
    cfg = evs.build_config_for_snr(sigma2, "imperfect")
    env = evs.RSMAEnergyEnv(cfg)
    model, ckpt_obs_dim, err = evs.load_ppo_model(CKPT, env)
    meta_ac, T_s, _ = evs.load_meta_model(CKPT, env)
    action_fn = evs.make_ppo_action_fn(model, ckpt_obs_dim, env.obs_dim)
    K = env.K

    n_ep = 10
    for tag, act_src in (("PPO", "ppo"), ("SDMA", "sdma")):
        R_all, d_all, Qbind_all, D_all = [], [], [], []
        conv_num, conv_den, srv_tot, arr_tot = 0.0, 0.0, 0.0, 0.0
        for ep in range(n_ep):
            obs, info = env.reset(seed=ep)
            evs._meta_decide_eval(meta_ac, env)
            steps = 0
            for _ in range(env.T):
                if act_src == "ppo":
                    action = np.array(action_fn(obs, env), dtype=np.float64).reshape(-1)
                else:
                    action = evs.SDMABaseline(cfg).select_action(obs, env)
                q_before = env.traffic.queue_lengths.copy()
                R_k = np.asarray(info["R_total"], dtype=np.float64) if steps > 0 else None
                mapped = (np.clip(action, -1, 1) + 1) / 2
                d_k = mapped[2 * K : 3 * K] if act_src == "ppo" else np.ones(K)
                obs, r, term, trunc, info = env.step(action)
                if R_k is not None:
                    R_all.extend(R_k.tolist())
                    d_all.extend(d_k.tolist())
                    sendable = np.minimum(R_k, q_before)
                    Qbind_all.extend((q_before < R_k).tolist())  # True = 队列不足以吃满速率
                    D_all.extend(np.asarray(info["served"], dtype=np.float64).tolist())
                srv_tot += float(np.sum(info["served"]))
                arr_tot += float(np.sum(info["arrivals"]))
                steps += 1
                if steps % T_s == 0 and not (term or trunc):
                    evs._meta_decide_eval(meta_ac, env)
                if term or trunc:
                    break
        R_all = np.array(R_all)
        d_all = np.array(d_all)
        Qb = np.array(Qbind_all)
        D_all = np.array(D_all)
        print(f"\n=== SNR={snr_db}dB [{tag}]  served/arrived = {srv_tot / max(arr_tot, 1e-9):.3f} ===")
        print(f"  per-user-slot 均值: R_k={R_all.mean():.3f}  d={d_all.mean():.3f}  D_k={D_all.mean():.3f}")
        print(f"  队列绑定占比 (Q_k < R_k): {Qb.mean() * 100:.1f}%")
        print(f"  速率天花板 delivery = mean(R)/lambda = {R_all.mean() / 1.363:.3f}")
        print(f"  d·min(R,Q) 理论 delivery = {np.mean(np.minimum(R_all, 1e9)):.3f}")
