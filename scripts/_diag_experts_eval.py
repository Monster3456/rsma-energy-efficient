"""专家头激活评估（全 7 SNR 点）：绕过 eval 脚本专家头丢弃 bug，得出 checkpoint 真实得分。

SDMA 参考来自 logs/eval_temporal_cur_best.txt（同 50ep 口径的不可变脚本输出）。
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
from src.models.actor_critic import ActorCriticAttn

temporal_cls, d_user_eff = wrapper.make_temporal_config_cls(4, ("h_gains", "queue", "power"))
src.config.Config = temporal_cls

spec = importlib.util.spec_from_file_location("evs", REPO / "scripts" / "evaluate_snr_sweep.py")
evs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evs)

CKPT = str(REPO / "logs" / "temporal_worker_h4_cur" / "checkpoints" / "best.pt")
SDMA = {0: (0.856, 3.46), 5: (0.979, 0.68), 10: (0.961, 2.28), 15: (0.887, 11.42),
        20: (0.795, 22.39), 25: (0.724, 28.76), 30: (0.664, 34.47)}

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
STATE = ckpt.get("worker_state_dict") or ckpt


def run_point(snr_db, n_eps):
    sigma2 = 10 ** (-snr_db / 10)
    cfg = evs.build_config_for_snr(sigma2, "imperfect")
    env = evs.RSMAEnergyEnv(cfg)
    model = ActorCriticAttn(
        obs_dim=24 * d_user_eff + 3, action_dim=4 * 24, K=24,
        d_user=d_user_eff, d_global=3, hidden_dim=128, n_heads=4, n_attn_layers=2,
        use_plackett_luce=True, inject_global_to_per_user=True, n_global_inject=2,
        use_bucket_experts=True, n_experts=4,
    )
    model.load_state_dict(STATE, strict=False)
    model.eval()
    meta_ac, T_s, _ = evs.load_meta_model(CKPT, env)
    action_fn = evs.make_ppo_action_fn(model, 24 * d_user_eff + 3, env.obs_dim)
    K = env.K
    delivs, queues, powers, ees = [], [], [], []
    for ep in range(n_eps):
        obs, info = env.reset(seed=ep)
        model.set_active_bucket(float(snr_db))
        evs._meta_decide_eval(meta_ac, env)
        acc = {"p": 0.0, "q": 0.0, "ee": 0.0}
        steps, last = 0, 0.0
        for _ in range(env.T):
            action = np.array(action_fn(obs, env), dtype=np.float64).reshape(-1)
            obs, r, term, trunc, info = env.step(action)
            acc["p"] += info["total_power"]
            acc["q"] += info["mean_queue"]
            acc["ee"] += info.get("EE", 0.0)
            last = info["delivery_ratio"]
            steps += 1
            if steps % T_s == 0 and not (term or trunc):
                evs._meta_decide_eval(meta_ac, env)
            if term or trunc:
                break
        delivs.append(last)
        queues.append(acc["q"] / steps)
        powers.append(acc["p"] / steps)
        ees.append(acc["ee"] / steps)
    return (float(np.mean(delivs)), float(np.mean(queues)),
            float(np.mean(powers)), float(np.mean(ees)))


print(f"ckpt={CKPT}")
passes, deliv_list, queue_list = 0, [], []
print(f"{'SNR':>4} {'deliv':>6} {'queue':>7} {'power':>6} {'EE':>6}  pass(deliv>=SDMA, q<=1.1*SDMA)")
for snr_db in (0, 5, 10, 15, 20, 25, 30):
    d, q, p, ee = run_point(snr_db, 20)
    sd, sq = SDMA[snr_db]
    ok = d >= sd and q <= 1.1 * sq
    passes += ok
    deliv_list.append(d)
    queue_list.append(q)
    print(f"{snr_db:>4} {d:>6.3f} {q:>7.2f} {p:>6.2f} {ee:>6.3f}  "
          f"{'PASS' if ok else 'fail'} (sdma {sd:.3f}/{sq:.2f})", flush=True)
avg_d = float(np.mean(deliv_list))
avg_q = float(np.mean(queue_list))
print(f"\nexperts-active score: [{passes}, {avg_d:.4f}, {-avg_q:.4f}]  "
      f"(vs eval-script path [0, ~0.609, ~-28.8], baseline best [5, 0.9163, -11.0186, 0.6914])")
