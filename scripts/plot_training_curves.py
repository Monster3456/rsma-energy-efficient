"""Training-curve figure from per-update checkpoint_meta.jsonl logs (supplement H).

Champion recipe over seeds 42-46 as mean + min-max band; pre-registered campaign
arms and the LDPTRLQ-family comparator as thin reference lines. No embedded title.

Usage:
  python scripts/plot_training_curves.py [--out docs/paper/figs/training_curves.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

SEEDS = ["m5_armA_sftp", "mAdjudS43", "mAdjudS44", "mAdjudS45", "mAdjudS46"]
ARMS = {
    "m51_armA2_lr1e4": "lr x3.3",
    "m51_armA3_nokldecay": "no KL decay",
    "m52_armD1_klbc": "KL-to-BC TR",
    "m52_armD2_anchor": "strong anchor",
    "m53_armE1_hm030": "mining 0.92/0.93",
    "m53_armE2_hm030w": "mining 0.95/0.95",
}
LDP = "b_ldptrlq_style"


def load_curve(run: str) -> tuple[list[float], list[float]]:
    path = Path("logs") / run / "checkpoint_meta.jsonl"
    xs, ys = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        rec = json.loads(line)
        xs.append(rec["update"])
        ys.append(rec["mean_reward"])
    return xs, ys


def build(out: str) -> None:
    seed_curves = [load_curve(r) for r in SEEDS]
    xs = seed_curves[0][0]
    grid = np.array([np.interp(xs, cx, cy) for cx, cy in seed_curves])
    mean = grid.mean(axis=0)
    lo, hi = grid.min(axis=0), grid.max(axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.fill_between(xs, lo, hi, color="#d62728", alpha=0.15, label="champion seeds 42-46 (min-max)")
    ax.plot(xs, mean, color="#d62728", lw=2.2, label="champion (5-seed mean)")
    for run, label in ARMS.items():
        try:
            cx, cy = load_curve(run)
        except FileNotFoundError:
            continue
        ax.plot(cx, cy, lw=1.0, alpha=0.75, label=f"arm: {label}")
    try:
        cx, cy = load_curve(LDP)
        ax.plot(cx, cy, lw=1.4, ls="--", color="#8c564b", label="LDPTRLQ family")
    except FileNotFoundError:
        pass
    ax.set_xlabel("PPO update")
    ax.set_ylabel("rollout mean reward")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/paper/figs/training_curves.png")
    args = ap.parse_args(argv)
    build(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
