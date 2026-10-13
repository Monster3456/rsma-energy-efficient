"""JointOpt: a strictly deeper per-slot model-based optimizer as offline-style upper bound (supplement J).

DiffOpt (the paper's solver expert) runs 25 cold-start Adam steps per slot over two
decoding-order candidates. JointOpt tightens that bound with (i) more Adam steps and
(ii) warm-starting the power logits from the previous slot, while keeping the same
differentiable MMSE-SIC shadow engine and oracle CSI. No new optimization code is
introduced: it reuses src/optimization/solver_expert.py (mutable) as-is.

Usage:
  PYTHONPATH=. python scripts/baselines/offline_upperbound.py \
      --points 0,15,30 --n_episodes 20 --n_steps 60
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from scripts.evaluate_snr_sweep import SNR_POINTS, build_config_for_snr, evaluate_method
from src.env.rsma_env import RSMAEnergyEnv
from src.optimization.solver_expert import SolverExpert


def make_jointopt_fn(env, n_steps: int, lr: float):
    expert = SolverExpert(env, n_steps=n_steps, lr=lr, use_true_channel=True)
    state = {"theta": None}

    def action_fn(obs, env):
        out = expert.solve(env, warm_theta=state["theta"])
        state["theta"] = np.clip(out["theta"], -10.0, 10.0)
        return out["action"]

    return action_fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", default="0,15,30")
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--n_steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--csi_mode", default="imperfect")
    args = ap.parse_args(argv)

    want = {int(x) for x in args.points.split(",")}
    print(f"### JointOpt upper bound (n_steps={args.n_steps}, warm-start, oracle CSI)")
    print("| SNR(dB) | JointOpt Deliv | JointOpt Q | JointOpt P(W) |")
    print("|---:|---:|---:|---:|")
    for snr_db, sigma2 in SNR_POINTS:
        if snr_db not in want:
            continue
        print(f"\n=== SNR={snr_db:>2} dB (sigma2={sigma2:.4g}) ===", flush=True)
        cfg = build_config_for_snr(sigma2, args.csi_mode)
        env = RSMAEnergyEnv(cfg)
        r = evaluate_method(make_jointopt_fn(env, args.n_steps, args.lr), env, args.n_episodes)
        print(
            f"| {snr_db} | {r['delivery_ratio']:.3f} | {r['mean_queue']:.2f} | {r['total_power']:.2f} |",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
