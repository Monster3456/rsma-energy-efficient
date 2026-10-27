"""Official-grid evaluator for trainable ActorCriticAttn baselines (TD3/DDPG/DR-SAC).

These trainers save their own checkpoint schema ({"model_state"|"worker_state_dict",
"obs_dim", "action_dim"}), which the immutable evaluate_snr_sweep.py alone cannot
rebuild (it defaults to the MLP ActorCritic). This script rebuilds the SA-GAT
ActorCriticAttn with the SAME constructor the trainers use and evaluates it on the
official 7-point x n_episodes grid through the immutable script's own
env/metric helpers (import-only, never modified).

Usage:
  PYTHONPATH=. python scripts/baselines/eval_actor_baseline.py \
      --checkpoint logs/comp_td3/checkpoints/final.pt --n_episodes 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_snr_sweep import SNR_POINTS, build_config_for_snr, evaluate_method
from src.env.rsma_env import RSMAEnergyEnv
from src.models.actor_critic import ActorCriticAttn


def load_policy_state(ckpt: dict) -> dict:
    for key in ("worker_state_dict", "model_state", "model_state_dict"):
        if key in ckpt:
            return ckpt[key]
    if "user_embed.0.weight" in ckpt:  # bare state dict
        return ckpt
    raise ValueError("checkpoint 无 worker_state_dict/model_state 键且非裸 state_dict")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--points", default="0,5,10,15,20,25,30")
    ap.add_argument("--d_user", type=int, default=13)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--csi_mode", default="imperfect")
    args = ap.parse_args(argv)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = load_policy_state(ckpt)
    obs_dim = int(ckpt.get("obs_dim", 13 * 24 + 3))
    action_dim = int(ckpt.get("action_dim", 4 * 24))
    K = (obs_dim - 3) // args.d_user  # H=0 默认口径: obs = K*d_user + 3 global
    d_global = obs_dim - K * args.d_user

    policy = ActorCriticAttn(
        obs_dim=obs_dim, action_dim=action_dim, K=K, d_user=args.d_user,
        d_global=d_global, hidden_dim=args.hidden, n_heads=args.heads,
        n_attn_layers=args.layers, use_plackett_luce=True,
    )
    policy.load_state_dict(state, strict=False)
    policy.eval()

    want = {float(s) for s in args.points.split(",")}
    print(f"### baseline eval: {Path(args.checkpoint).parent.parent.name} "
          f"(obs={obs_dim}, act={action_dim}, K={K})", flush=True)
    print("| SNR(dB) | Deliv | Q | P(W) | Rsum | Ovf |")
    print("|---:|---:|---:|---:|---:|---:|")
    # A-prune 转正后 build_config_for_snr 默认剪枝（obs 243），而 TD3/DDPG/DR-SAC
    # 等旧检查点是未剪口径（obs 315）。按 d_user 反推剪枝开关，否则 _split_obs 会崩。
    prune = bool(args.d_user == 10)
    tot_deliv = []
    for snr_db, sigma2 in SNR_POINTS:
        if snr_db not in want:
            continue
        cfg = build_config_for_snr(sigma2, args.csi_mode)
        cfg.system.obs_prune_constant_blocks = prune
        cfg.ppo.worker_d_user = int(args.d_user)
        env = RSMAEnergyEnv(cfg)
        if env.obs_dim != obs_dim:
            raise SystemExit(
                f"obs_dim 不匹配：env={env.obs_dim} vs ckpt={obs_dim}。"
                f"试 --d_user {10 if prune else 13}（A-prune 剪枝口径 10，未剪口径 13）。"
            )

        def action_fn(obs, env, _p=policy):
            o = torch.as_tensor(obs[None, :], dtype=torch.float32)
            with torch.no_grad():
                cont, order, *_ = _p.act(o, deterministic=True)
            return np.concatenate([cont.numpy()[0], order.numpy()[0].astype(np.float64)])

        r = evaluate_method(action_fn, env, args.n_episodes)
        tot_deliv.append(r["delivery_ratio"])
        print(f"| {snr_db} | {r['delivery_ratio']:.3f} | {r['mean_queue']:.2f} | "
              f"{r['total_power']:.2f} | {r['R_sum']:.2f} | {r['overflow_rate']:.4f} |",
              flush=True)
    print(f"\nmean_deliv={np.mean(tot_deliv):.4f}" if tot_deliv else "")
    print("SNR SWEEP REPORT (baseline evals)")
    return 0


if __name__ == "__main__":
    sys.exit(main())