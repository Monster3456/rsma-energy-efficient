"""一次性诊断脚本：temporal_worker_h4_cur 失败根因分析（跑完可删）。

四个假设、五个变体（SNR 15/30 dB × 20 episodes）：
  H1 满功率钉死        -> orig 变体统计 p_ratio 分布
  H2 满功率有害(干扰)   -> p60/p40 强制降压，看 delivery 是否反而升
  H3 策略丢弃可服务比特 -> d100 强制 d_ratio=1，看 delivery 变化
  H4 历史窗口被忽略     -> hist0 置零 user_embed 历史列，看 delivery 是否不变
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import torch

import src.config
import evaluate_snr_sweep_temporal as wrapper

WINDOW, FEATURES = 4, ("h_gains", "queue", "power")
temporal_cls, d_user_eff = wrapper.make_temporal_config_cls(WINDOW, FEATURES)
BASE_D = d_user_eff - WINDOW * len(FEATURES)
HIST_COLS = slice(BASE_D, BASE_D + WINDOW * len(FEATURES))
src.config.Config = temporal_cls

spec = importlib.util.spec_from_file_location("evs", REPO / "scripts" / "evaluate_snr_sweep.py")
evs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evs)

CKPT = REPO / "logs" / "temporal_worker_h4_cur" / "checkpoints" / "best.pt"


def run_variant(snr_db, n_eps, zero_history=False, force_p=None, force_d=None):
    sigma2 = 10 ** (-snr_db / 10)
    cfg = evs.build_config_for_snr(sigma2, "imperfect")
    env = evs.RSMAEnergyEnv(cfg)
    model, ckpt_obs_dim, err = evs.load_ppo_model(str(CKPT), env)
    assert model is not None, err
    if zero_history:
        with torch.no_grad():
            model.user_embed[0].weight[:, HIST_COLS].zero_()
    meta_ac, T_s, _ = evs.load_meta_model(str(CKPT), env)
    action_fn = evs.make_ppo_action_fn(model, ckpt_obs_dim, env.obs_dim)

    K = env.K
    ep_final = {k: [] for k in ["deliv", "power", "R_sum", "queue", "alpha", "ovf",
                                "EE", "deliv_util", "ee_bonus", "ovf_pen"]}
    p_all, d_all, p_cap_all = [], [], []
    for ep in range(n_eps):
        obs, info = env.reset(seed=ep)
        if meta_ac is not None:
            evs._meta_decide_eval(meta_ac, env)
        for k in ep_final:
            ep_final[k] = ep_final[k] if isinstance(ep_final[k], list) else []
        acc = {k: 0.0 for k in ep_final}
        steps = 0
        for _ in range(env.T):
            action = np.array(action_fn(obs, env), dtype=np.float64).reshape(-1)
            if force_p is not None:
                action[K:2 * K] = 2.0 * force_p - 1.0
            if force_d is not None:
                action[2 * K:3 * K] = 2.0 * force_d - 1.0
            mapped = (np.clip(action, -1, 1) + 1) / 2
            p_all.append(float(np.mean(mapped[K:2 * K])))
            d_all.append(float(np.mean(mapped[2 * K:3 * K])))
            p_cap_all.append(float(np.mean(mapped[K:2 * K] > 0.97)))
            obs, r, term, trunc, info = env.step(action)
            for k in acc:
                key = {"deliv": "delivery_ratio", "power": "total_power",
                       "R_sum": "R_sum", "queue": "mean_queue", "alpha": "mean_alpha",
                       "ovf": "overflow_rate", "EE": "EE", "deliv_util": "deliv_utility",
                       "ee_bonus": "ee_bonus", "ovf_pen": "overflow_penalty"}[k]
                acc[k] += float(info.get(key, 0.0))
            steps += 1
            if meta_ac is not None and steps % T_s == 0 and not (term or trunc):
                evs._meta_decide_eval(meta_ac, env)
            if term or trunc:
                break
        for k in ep_final:
            ep_final[k].append(acc[k] / max(steps, 1))
    return {k: float(np.mean(v)) for k, v in ep_final.items()} | {
        "p_mean": float(np.mean(p_all)), "d_mean": float(np.mean(d_all)),
        "p_cap_frac": float(np.mean(p_cap_all)),
    }


VARIANTS = [
    ("orig", {}),
    ("hist0", {"zero_history": True}),
    ("p60", {"force_p": 0.6}),
    ("p40", {"force_p": 0.4}),
    ("d100", {"force_d": 1.0}),
]

print(f"ckpt={CKPT.name}  window={WINDOW}  hist_cols={HIST_COLS}  (user_embed in={d_user_eff + 2})")
hdr = f"{'SNR':>4} {'variant':>7} {'deliv':>6} {'power':>6} {'R_sum':>7} {'queue':>6} {'alpha':>5} {'ovf':>6} {'p_mean':>6} {'p_cap':>6} {'d_mean':>6} {'delivU':>7} {'eeB':>6} {'ovfPen':>7} {'EE':>6}"
print(hdr)
for snr_db in (15, 30):
    for name, kw in VARIANTS:
        s = run_variant(snr_db, 20, **kw)
        print(f"{snr_db:>4} {name:>7} {s['deliv']:>6.3f} {s['power']:>6.2f} {s['R_sum']:>7.2f} "
              f"{s['queue']:>6.2f} {s['alpha']:>5.2f} {s['ovf']:>6.3f} {s['p_mean']:>6.3f} "
              f"{s['p_cap_frac']:>6.3f} {s['d_mean']:>6.3f} {s['deliv_util']:>7.2f} {s['ee_bonus']:>6.2f} "
              f"{s['ovf_pen']:>7.2f} {s['EE']:>6.3f}", flush=True)
