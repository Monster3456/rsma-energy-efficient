"""Extended (novel) comparator evaluation on the official SNR grid.

Evaluates training-free / recently-proposed comparators that are NOT wired into
the immutable scripts/evaluate_snr_sweep.py, using that script's own environment
construction and metric accumulation (import-only; the immutable file is never
modified). Emits a 表1-compatible wide table so scripts/plot_benchmarks.py and
optimize_loop.parse_eval_output can consume it.

Comparators:
  diffopt        per-slot model-based online optimization: the differentiable
                 MMSE-SIC shadow engine (SolverExpert) refines power logits and
                 picks the decoding order every slot, with NO learned policy.
                 NOTE: its surrogate sees the true channel (oracle CSI) -> a
                 NON-CAUSAL upper bound, used for information-gap analysis.
  diffopt_causal same per-slot optimizer but the surrogate uses ONLY the
                 estimated channel H_hat (declared rates) -> the FAIR causal
                 model-based comparator against the learned policy.
  adaptive_power repository adaptive-power heuristic (scripts/baselines/).
  water_filling  classical water-filling power allocation (subclass of adaptive).

Usage:
  PYTHONPATH=. python scripts/evaluate_extended_baselines.py \
      --baselines diffopt,adaptive_power,water_filling --n_episodes 50 \
      > logs/_eval_extended.txt
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from scripts.baselines.adaptive_power_baseline import AdaptivePowerBaseline, WaterFillingBaseline
from scripts.evaluate_snr_sweep import (
    METRIC_KEYS,
    SNR_POINTS,
    build_config_for_snr,
    evaluate_method,
    make_baseline_action_fn,
)
from src.env.rsma_env import RSMAEnergyEnv
from src.optimization.solver_expert import SolverExpert

LABEL = {
    "diffopt": "DiffOpt",
    "diffopt_causal": "DiffOptCausal",
    "adaptive_power": "AdaptivePower",
    "water_filling": "WaterFilling",
}


def make_diffopt_fn(env, causal: bool = False):
    """Per-slot differentiable optimization; cold-start each slot (teacher mode)."""
    expert = SolverExpert(env, use_true_channel=not causal)

    def action_fn(obs, env):
        return expert.solve(env)["action"]

    return action_fn


def build_action_fn(name: str, cfg, env):
    if name == "diffopt":
        return make_diffopt_fn(env, causal=False)
    if name == "diffopt_causal":
        return make_diffopt_fn(env, causal=True)
    if name == "adaptive_power":
        return make_baseline_action_fn(AdaptivePowerBaseline(cfg))
    if name == "water_filling":
        return make_baseline_action_fn(WaterFillingBaseline(cfg))
    raise SystemExit(f"unknown baseline {name!r}; choose from {sorted(LABEL)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", default="diffopt,adaptive_power,water_filling")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--csi_mode", default="imperfect")
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.baselines.split(",") if n.strip()]
    for n in names:
        if n not in LABEL:
            raise SystemExit(f"unknown baseline {n!r}; choose from {sorted(LABEL)}")

    per_snr = {snr: {} for snr, _ in SNR_POINTS}
    for snr_db, sigma2 in SNR_POINTS:
        print(f"\n=== SNR={snr_db:>2} dB (sigma2={sigma2:.4g}) ===", flush=True)
        cfg = build_config_for_snr(sigma2, args.csi_mode)
        env = RSMAEnergyEnv(cfg)
        for name in names:
            t0 = time.time()
            action_fn = build_action_fn(name, cfg, env)
            res = evaluate_method(action_fn, env, args.n_episodes)
            per_snr[snr_db][LABEL[name]] = res
            print(
                f"  {LABEL[name]:14s} deliv={res['delivery_ratio']:.4f} "
                f"Q={res['mean_queue']:.2f} P={res['total_power']:.2f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    # 表1-compatible wide table
    hdr = ["SNR(dB)"]
    for name in names:
        lab = LABEL[name]
        hdr += [f"{lab} P(W)", f"{lab} Q", f"{lab} Deliv", f"{lab} Rsum", f"{lab} Ovf"]
    print("\n### 表1：扩展对比算法逐点（imperfect CSI, %d episodes/点）" % args.n_episodes)
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---:|" * len(hdr))
    for snr_db, _ in SNR_POINTS:
        row = [str(snr_db)]
        for name in names:
            r = per_snr[snr_db][LABEL[name]]
            row += [
                f"{r['total_power']:.3f}",
                f"{r['mean_queue']:.2f}",
                f"{r['delivery_ratio']:.3f}",
                f"{r['R_sum']:.2f}",
                f"{r['overflow_rate']:.4f}",
            ]
        print("| " + " | ".join(row) + " |")
    print("\nSNR SWEEP REPORT (extended baselines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
