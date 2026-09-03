"""逐用户剖析：d_ratio 分布 / 每用户 served-arrived / Meta 预算 P_target。"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np

import src.config
import evaluate_snr_sweep_temporal as wrapper

temporal_cls, d_user_eff = wrapper.make_temporal_config_cls(4, ("h_gains", "queue", "power"))
src.config.Config = temporal_cls

spec = importlib.util.spec_from_file_location("evs", REPO / "scripts" / "evaluate_snr_sweep.py")
evs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evs)

CKPT = str(REPO / "logs" / "temporal_worker_h4_cur" / "checkpoints" / "best.pt")

for snr_db in (15, 30):
    sigma2 = 10 ** (-snr_db / 10)
    cfg = evs.build_config_for_snr(sigma2, "imperfect")
    env = evs.RSMAEnergyEnv(cfg)
    print(f"\n=== SNR={snr_db}dB  P_k_max={env.P_k_max}  hierarchical={env.hierarchical_enabled} ===")
    model, ckpt_obs_dim, err = evs.load_ppo_model(CKPT, env)
    meta_ac, T_s, _ = evs.load_meta_model(CKPT, env)
    action_fn = evs.make_ppo_action_fn(model, ckpt_obs_dim, env.obs_dim)
    K = env.K

    d_users = np.zeros((10, K))       # per-ep per-user mean d
    srv_users = np.zeros((10, K))     # per-ep per-user served total
    arr_users = np.zeros((10, K))
    p_targets = []
    all_d = []
    for ep in range(10):
        obs, info = env.reset(seed=ep)
        evs._meta_decide_eval(meta_ac, env)
        p_targets.append(env.P_target)
        du = np.zeros(K)
        srv = np.zeros(K)
        arr = np.zeros(K)
        steps = 0
        for _ in range(env.T):
            action = np.array(action_fn(obs, env), dtype=np.float64).reshape(-1)
            mapped = (np.clip(action, -1, 1) + 1) / 2
            du += mapped[2 * K:3 * K]
            all_d.extend(mapped[2 * K:3 * K].tolist())
            obs, r, term, trunc, info = env.step(action)
            srv += np.asarray(info["served"], dtype=np.float64)
            arr += np.asarray(info["arrivals"], dtype=np.float64)
            steps += 1
            if steps % T_s == 0 and not (term or trunc):
                evs._meta_decide_eval(meta_ac, env)
                p_targets.append(env.P_target)
            if term or trunc:
                break
        d_users[ep] = du / steps
        srv_users[ep] = srv
        arr_users[ep] = arr

    dm = d_users.mean(axis=0)
    sm = srv_users.mean(axis=0)
    am = arr_users.mean(axis=0)
    per_user_deliv = np.where(am > 1e-9, sm / np.maximum(am, 1e-9), np.nan)
    print(f"eval hierarchical={env.hierarchical_enabled} -> 总功率恒=K*P_k_max=24W（softmax 份额）")
    print(f"d_ratio per-user mean: min={dm.min():.2f} p25={np.percentile(dm,25):.2f} med={np.median(dm):.2f} p75={np.percentile(dm,75):.2f} max={dm.max():.2f}")
    hist, edges = np.histogram(all_d, bins=[0, .05, .2, .4, .6, .8, .95, 1.001])
    print("d histogram (all user-steps): " + " ".join(f"[{edges[i]:.1f},{edges[i+1]:.1f}):{hist[i]}" for i in range(len(hist))))
    print(f"users with mean d<0.2: {int(np.sum(dm<0.2))}/{K}   d>0.8: {int(np.sum(dm>0.8))}/{K}")
    print(f"per-user delivery: min={np.nanmin(per_user_deliv):.2f} med={np.nanmedian(per_user_deliv):.2f} max={np.nanmax(per_user_deliv):.2f}")
    starved = np.argsort(per_user_deliv)[:6]
    print("starved users (deliv, d_mean, alpha?): " + " ".join(f"u{i}({per_user_deliv[i]:.2f},d={dm[i]:.2f})" for i in starved))
