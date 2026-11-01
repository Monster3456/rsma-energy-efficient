# -*- coding: utf-8 -*-
"""O1: 可靠性选点扫描（零重训，部署侧协议优化）。

问题：best_guard.pt 按**均值交付**选点，而冠军 0 dB 有 20% 逐回合违约。
训练轨迹的周期 checkpoint（ckpt_100..700/final）是否是"更稳"的候选——
即是否存在一个 checkpoint，均值相近但 0/5 dB 尾违约率显著更低？

做法：对给定 checkpoint 列表，在尾部焦点点（默认 0/5 dB）逐 episode 跑官方口径
（deliv >= ZETA_SDMA 为逐回合门），输出每个 ckpt 的 (mean deliv, ep-viol,
mean_queue, power)，并给出 min-tail 选点与 best_guard 的对比。

用法:
  PYTHONPATH=. python scripts/reliability_selection_sweep.py \\
      --dir logs/m5_armA_sftp/checkpoints [--points 0,5] [--n_episodes 30] [--seed 42]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_simplified_arm import (  # noqa: E402
    _continuous_dim,
    _head_dim,
    _load_states,
)
from scripts.evaluate_snr_sweep import (  # noqa: E402
    SNR_POINTS,
    _run_episode,
    build_config_for_snr,
    load_meta_model,
)
from src.env.rsma_env import RSMAEnergyEnv  # noqa: E402
from src.models.actor_critic import ActorCriticAttn  # noqa: E402

K = 24
ZETA_SDMA = {0: 0.856, 5: 0.979, 10: 0.961, 15: 0.887, 20: 0.795, 25: 0.724, 30: 0.664}
CKPT_PRIORITY = ("ckpt_700.pt", "ckpt_600.pt", "ckpt_500.pt", "ckpt_400.pt",
                 "ckpt_300.pt", "ckpt_200.pt", "ckpt_100.pt", "final.pt",
                 "best_guard.pt", "best.pt")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--points", type=str, default="0,5")
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    ck_dir = Path(args.dir)
    files = [f for f in CKPT_PRIORITY if (ck_dir / f).exists()]
    extra = sorted(glob.glob(str(ck_dir / "ckpt_*.pt")))
    files = list(dict.fromkeys([Path(f).name for f in extra] + files))
    points = [int(x) for x in args.points.split(",")]

    env_cache = {}
    rows = []
    for fname in files:
        ck_path = str(ck_dir / fname)
        ck, w_state, _m = _load_states(ck_path)
        d_user = 13 + 4 * 3
        obs_dim = d_user * K + 3
        action_dim = _continuous_dim(w_state) + K
        simplified = _head_dim(w_state) == 2
        policy = ActorCriticAttn(
            obs_dim=obs_dim, action_dim=action_dim, K=K, d_user=d_user,
            d_global=3, hidden_dim=128, n_heads=4, n_attn_layers=2,
            use_plackett_luce=True, simplified_action=simplified,
        )
        mm, uu = policy.load_state_dict(w_state, strict=False)
        if mm or uu:
            print(f"  [warn] {fname} missing={mm}")
        policy.eval()

        per_pt = {}
        for snr_db in points:
            sigma2 = dict(SNR_POINTS)[snr_db]
            if snr_db not in env_cache:
                cfg = build_config_for_snr(sigma2, "imperfect")
                cfg.system.obs_prune_constant_blocks = False
                cfg.system.obs_history_window = 4
                cfg.system.obs_history_features = ("h_gains", "queue", "power")
                cfg.ppo.worker_d_user = d_user
                env_cache[snr_db] = RSMAEnergyEnv(cfg)
            env = env_cache[snr_db]
            assert env.obs_dim == obs_dim, f"{fname}: env {env.obs_dim} != {obs_dim}"
            meta_ac, t_s, err = load_meta_model(ck_path, env)
            gate = ZETA_SDMA[snr_db]
            viol = 0
            dls, qs, pws = [], [], []

            def action_fn(obs_, env_, _p=policy):
                o_ = torch.as_tensor(obs_[None, :], dtype=torch.float32)
                with torch.no_grad():
                    c_, od_, *_ = _p.act(o_, deterministic=True)
                return np.concatenate([c_.numpy()[0], od_.numpy()[0].astype(np.float64)])

            for ep in range(args.n_episodes):
                m = _run_episode(action_fn, env, seed=ep, meta_ac=meta_ac, T_s=int(t_s))
                dls.append(float(m["delivery_ratio"]))
                qs.append(float(m["mean_queue"]))
                pws.append(float(m["total_power"]))
                if m["delivery_ratio"] < gate:
                    viol += 1
            per_pt[str(snr_db)] = {
                "mean_deliv": float(np.mean(dls)),
                "ep_viol": viol / args.n_episodes,
                "mean_queue": float(np.mean(qs)),
                "total_power": float(np.mean(pws)),
            }
        rows.append((fname, per_pt))

    print("| ckpt | " + " | ".join(f"{p}dB mean/epv" for p in points) + " | mean(epv) |", flush=True)
    print("|---|" + "---|" * len(points) + "---|", flush=True)
    evals = []
    for fname, pp in rows:
        part = " ".join(f"{pp[str(p)]['mean_deliv']:.3f}/{pp[str(p)]['ep_viol']:.2f}" for p in points)
        evals.append(np.mean([pp[str(p)]["ep_viol"] for p in points]))
        print(f"| {fname:>16} | {part} | {evals[-1]:.3f} |", flush=True)
    best_idx = int(np.argmin(evals))
    best_name = rows[best_idx][0]
    guard_row = next(r for r in rows if r[0] == "best_guard.pt")
    print(f"\nmin-tail 选点: {best_name} (mean epv={evals[best_idx]:.3f}) vs "
          f"best_guard.pt (mean epv={np.mean([guard_row[1][str(p)]['ep_viol'] for p in points]):.3f})",
          flush=True)
    out = {"dir": str(ck_dir), "points": points, "n_episodes": args.n_episodes,
           "rows": {f: pp for f, pp in rows}, "min_tail_ckpt": best_name}
    Path("logs/reliability_selection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("结果 -> logs/reliability_selection.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())