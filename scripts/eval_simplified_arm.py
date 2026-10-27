"""Arm A 精简模式检查点的官方 7x50 评估。

为什么需要它：不可变的 ``scripts/evaluate_snr_sweep.py`` 用
``_infer_action_dim_from_state`` 找 ``actor_head.weight``，而 ActorCriticAttn 的
键名是 ``user_head``；推断失败就回退 ``env.action_dim``=3K，于是按 4 维头重建模型，
加载时形状不符被跳过 → **头是随机初始化，评估结果是垃圾**。

本脚本 import 不可变脚本的 env / 指标助手（只 import、不改），自行按检查点的
``user_head`` / ``log_std`` 形状重建 ActorCriticAttn（simplified_action=True），
并接入 Meta-Controller（分层结构保留）。

用法：
  PYTHONPATH=. python scripts/eval_simplified_arm.py \\
      --checkpoint logs/hopt_armA/checkpoints/best_guard.pt --n_episodes 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_snr_sweep import (
    SNR_POINTS,
    build_config_for_snr,
    evaluate_method,
)
from src.env.rsma_env import RSMAEnergyEnv
from src.models.actor_critic import ActorCriticAttn


def _load_states(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    worker_state = ck.get("worker_state_dict") or ck.get("model_state_dict")
    meta_state = ck.get("meta_state_dict")
    if worker_state is None:
        raise SystemExit("checkpoint 无 worker_state_dict")
    return ck, worker_state, meta_state


def _head_dim(worker_state: dict) -> int:
    w = worker_state.get("user_head.weight")
    if w is None:
        raise SystemExit("checkpoint 无 user_head.weight")
    return int(w.shape[0])


def _continuous_dim(worker_state: dict) -> int:
    ls = worker_state.get("log_std")
    if ls is None:
        raise SystemExit("checkpoint 无 log_std")
    return int(ls.shape[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--csi_mode", default="imperfect")
    ap.add_argument("--obs_history_window", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args(argv)

    ck, w_state, m_state = _load_states(args.checkpoint)
    head_dim = _head_dim(w_state)
    cont_dim = _continuous_dim(w_state)
    simplified = head_dim == 2
    if not simplified:
        raise SystemExit(f"本脚本只处理精简模式（user_head=2 维），实测 {head_dim} 维")

    # champion 口径：H=4 → d_user=25；未剪 obs_dim = 25*24+3 = 603
    d_user = 13 + args.obs_history_window * 3
    K = 24
    obs_dim = d_user * K + 3
    action_dim = cont_dim + K          # 连续块 + 解码序
    d_global = obs_dim - K * d_user

    print(f"### Arm A 精简评估: {args.checkpoint}")
    print(f"    obs_dim={obs_dim} d_user={d_user} action_dim={action_dim} "
          f"cont={cont_dim} head_dim={head_dim} simplified={simplified}")

    policy = ActorCriticAttn(
        obs_dim=obs_dim, action_dim=action_dim, K=K, d_user=d_user,
        d_global=d_global, hidden_dim=args.hidden, n_heads=args.heads,
        n_attn_layers=args.layers, use_plackett_luce=True,
        simplified_action=True,
    )
    missing, unexpected = policy.load_state_dict(w_state, strict=False)
    if missing or unexpected:
        print(f"    [WARN] missing={missing} unexpected={unexpected}")
    policy.eval()

    # Meta：分层结构保留，α 与分组仍由 Meta 决定
    meta_ac, t_s = None, 20
    if m_state is not None:
        from src.models.meta_controller import MetaActorCritic
        meta_ac = MetaActorCritic(
            K=K, d_user=12, hidden_dim=64, n_heads=4, n_groups=4,
            group_level_alpha=True,
        )
        mm, mu = meta_ac.load_state_dict(m_state, strict=False)
        if mm or mu:
            print(f"    [WARN] meta missing={mm} unexpected={mu}")
        meta_ac.eval()
        t_s = int(ck.get("t_s", 20) or 20)
    print(f"    meta={'on' if meta_ac is not None else 'off'}  T_s={t_s}")

    print("| SNR(dB) | Deliv | Q | P(W) | Rsum | Ovf |")
    print("|---:|---:|---:|---:|---:|---:|")
    delivs = []
    for snr_db, sigma2 in SNR_POINTS:
        cfg = build_config_for_snr(sigma2, args.csi_mode)
        # champion 未剪口径 + H=4
        cfg.system.obs_prune_constant_blocks = False
        cfg.system.obs_history_window = args.obs_history_window
        cfg.system.obs_history_features = ("h_gains", "queue", "power")
        cfg.ppo.worker_d_user = d_user
        env = RSMAEnergyEnv(cfg)
        assert env.obs_dim == obs_dim, f"env.obs_dim={env.obs_dim} != {obs_dim}"

        def action_fn(obs, env, _p=policy):
            o = torch.as_tensor(obs[None, :], dtype=torch.float32)
            with torch.no_grad():
                cont, order, *_ = _p.act(o, deterministic=True)
            return np.concatenate([
                cont.numpy()[0],
                order.numpy()[0].astype(np.float64),
            ])

        r = evaluate_method(action_fn, env, args.n_episodes,
                            meta_ac=meta_ac, T_s=t_s)
        delivs.append(r["delivery_ratio"])
        print(f"| {snr_db} | {r['delivery_ratio']:.3f} | {r['mean_queue']:.2f} | "
              f"{r['total_power']:.2f} | {r['R_sum']:.2f} | {r['overflow_rate']:.4f} |",
              flush=True)
    print(f"\nmean_deliv={np.mean(delivs):.4f}")
    print("SNR SWEEP REPORT (arm A simplified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
