"""Benchmark curve plotting from evaluate_snr_sweep reports.

Parses the text report emitted by scripts/evaluate_snr_sweep.py (immutable) and draws
SNR-vs-metric curves for every algorithm the sweep evaluated. Nothing here re-runs the
simulation; feed it an existing logs/_eval_*.txt.

Three modes:
  all       6-panel figure, one curve per algorithm (needs --baselines sdma,noma,equal_power
            in the eval that produced the log, otherwise absent algorithms are skipped)
  ablation  one metric, many checkpoints overlaid, SDMA drawn as the shared reference
  seedband  6-panel figure: baselines from one report + PPO-RSMA mean/min-max envelope
            over several reports (seed-replication band; draws each member faint)

Examples:
  python scripts/plot_benchmarks.py all --log logs/_eval_bench_all.txt
  python scripts/plot_benchmarks.py ablation --metric delivery_ratio \
      --spec "A_bestguard=logs/_eval_m5_A_bestguard.txt" \
      --spec "bc_init=logs/_eval_m4_bc_init.txt"
  python scripts/plot_benchmarks.py seedband \
      --baseline-log logs/_eval_bench_all.txt \
      --ppp-report "champ=logs/_eval_m5_A_bestguard.txt" \
      --ppp-report "S43=logs/_eval_mAdjud_S43_bestguard.txt" \
      --ppp-report "S44=logs/_eval_mAdjud_S44_bestguard.txt"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Mirrors evaluate_snr_sweep.py:31/45/417-423 — keep in sync if that file ever changes.
METRIC_LABEL = {
    "total_power": "P(W)",
    "mean_queue": "Q",
    "delivery_ratio": "Deliv",
    "R_sum": "Rsum",
    "overflow_rate": "Ovf",
}
ALGO_ORDER = ["PPO-RSMA", "SDMA", "NOMA", "EqualPower"]

PANELS = [
    ("delivery_ratio", "Delivery ratio", "higher is better", False),
    ("mean_queue", "Mean queue length", "lower is better", False),
    ("total_power", "Total transmit power (W)", "lower is better", False),
    ("ee", "Energy efficiency  R_sum / P  (bps/Hz/W)", "higher is better", False),
    ("overflow_rate", "Queue overflow rate", "lower is better", False),
    ("R_sum", "Sum rate (bps/Hz)", "higher is better", False),
]

STYLE = {
    "PPO-RSMA": dict(color="#d62728", marker="o", lw=2.4, ms=7, zorder=5),
    "SDMA": dict(color="#1f77b4", marker="s", lw=1.8, ms=6, ls="--", zorder=4),
    "NOMA": dict(color="#2ca02c", marker="^", lw=1.8, ms=6, ls="-.", zorder=3),
    "EqualPower": dict(color="#7f7f7f", marker="v", lw=1.8, ms=6, ls=":", zorder=2),
    # extended (novel) comparators
    "DiffOpt": dict(color="#9467bd", marker="D", lw=2.0, ms=6, ls="-", zorder=6),
    "DiffOptCausal": dict(color="#e377c2", marker="d", lw=1.6, ms=5, ls=(0, (4, 2)), zorder=6),
    "AdaptivePower": dict(color="#ff7f0e", marker="p", lw=1.6, ms=6, ls="--", zorder=2),
    "WaterFilling": dict(color="#17becf", marker="h", lw=1.6, ms=6, ls=":", zorder=2),
}


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_overview(path: str | Path) -> tuple[list[float], dict[str, dict[str, list[float | None]]]]:
    """Parse 表1 (SNR x algorithm x metric wide table).

    Returns (snr_points, data[algo][metric_key] -> per-SNR values, None where 'n/a').
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    hdr = next((i for i, ln in enumerate(lines) if ln.startswith("| SNR(dB) |")), None)
    if hdr is None:
        raise ValueError(f"no overview table ('| SNR(dB) |' header) found in {path}")

    label_to_key = {v: k for k, v in METRIC_LABEL.items()}
    cols: list[tuple[str, str]] = []
    for cell in _split_row(lines[hdr])[1:]:
        algo, _, label = cell.rpartition(" ")
        if label not in label_to_key:
            raise ValueError(f"unrecognised column header {cell!r} in {path}")
        cols.append((algo, label_to_key[label]))

    snrs: list[float] = []
    data: dict[str, dict[str, list[float | None]]] = {}
    for ln in lines[hdr + 2 :]:  # hdr+1 is the '---:' separator
        if not ln.startswith("|"):
            break
        row = _split_row(ln)
        snrs.append(float(row[0]))
        for (algo, key), raw in zip(cols, row[1:]):
            data.setdefault(algo, {}).setdefault(key, []).append(None if raw == "n/a" else float(raw))

    for algo, metrics in data.items():
        r_sum, power = metrics.get("R_sum", []), metrics.get("total_power", [])
        metrics["ee"] = [(r / p if (r is not None and p not in (None, 0)) else None) for r, p in zip(r_sum, power)]
    return snrs, data


def _present(algos: dict[str, dict[str, list]], key: str) -> list[str]:
    """ALGO_ORDER filtered to algorithms that actually have data for `key`."""
    return [a for a in ALGO_ORDER if a in algos and any(v is not None for v in algos[a].get(key, []))]


def _plot_series(ax, snrs, values, label, style):
    xs = [s for s, v in zip(snrs, values) if v is not None]
    ys = [v for v in values if v is not None]
    if xs:
        ax.plot(xs, ys, label=label, **style)


def plot_all(log: str, out: str) -> None:
    snrs, data = parse_overview(log)
    algos = _present(data, "delivery_ratio")
    print(f"parsed {log}: {len(snrs)} SNR points, algorithms with data = {algos}")

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    for ax, (key, title, note, _log) in zip(axes.ravel(), PANELS):
        for algo in algos:
            values = data[algo].get(key)
            if values is None:
                continue
            _plot_series(ax, snrs, values, algo, STYLE.get(algo, dict(marker="x")))
        ax.set_title(f"{title}\n({note})", fontsize=11)
        ax.set_xlabel("SNR (dB)")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(snrs)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle(
        "RSMA uplink energy-efficient scheduling — benchmark across all algorithms\n"
        f"(imperfect CSI, 50 episodes/SNR point, source: {Path(log).name})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_seed_band(baseline_log: str, ppp_reports: list[str], out: str,
                   extra_reports: list[str] | None = None) -> None:
    """6-panel figure: classical baselines + PPO-RSMA seed-replication band.

    Baselines (SDMA/NOMA/EqualPower) come from `baseline_log`; every `--ppp-report`
    is parsed as one PPO-RSMA replication (TAG=PATH) and drawn as a faint member line,
    with the mean per point and a min-max shaded envelope. `ee` is recomputed from the
    report's own R_sum / P so member reports need no extra columns.
    """
    if len(ppp_reports) < 2:
        print("note: fewer than 2 PPO-RSMA reports given; band will collapse to a line")

    snrs_b, data_b = parse_overview(baseline_log)
    for extra in extra_reports or []:
        _, data_e = parse_overview(extra)
        for algo, metrics in data_e.items():
            data_b.setdefault(algo, metrics)
        print(f"merged extra report: {extra}")
    # ALGO_ORDER first, then any merged-in comparators, in insertion order
    present = [a for a in data_b if any(v is not None for v in data_b[a].get("delivery_ratio", []))]
    algos = [a for a in ALGO_ORDER if a in present] + [a for a in present if a not in ALGO_ORDER]

    members: list[tuple[str, list[float], dict[str, list[float | None]]]] = []
    for spec in ppp_reports:
        tag, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--ppp-report must be TAG=PATH, got {spec!r}")
        snrs, data = parse_overview(path)
        algo = "PPO-RSMA" if "PPO-RSMA" in data else next(iter(data))
        members.append((tag, snrs, data[algo]))
        print(f"parsed PPO member {tag:10s} <- {path}")

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    for ax, (key, title, note, _log) in zip(axes.ravel(), PANELS):
        for algo in algos:
            if algo == "PPO-RSMA":
                continue  # drawn as the band below, not as a bare line
            values = data_b[algo].get(key)
            if values is not None:
                _plot_series(ax, snrs_b, values, algo, STYLE.get(algo, dict(marker="x")))

        # per-point mean / min / max across member reports (ignore None per point)
        xs = [s for s in snrs_b if s is not None]
        grid: list[list[float | None]] = [
            [m[2].get(key, [])[i] if i < len(m[2].get(key, [])) else None for m in members]
            for i in range(len(xs))
        ]
        mean = [float(np.mean([v for v in col if v is not None])) if any(v is not None for v in col) else None for col in grid]
        lo = [min(v for v in col if v is not None) if any(v is not None for v in col) else None for col in grid]
        hi = [max(v for v in col if v is not None) if any(v is not None for v in col) else None for col in grid]
        pts = [s for s, m in zip(xs, mean) if m is not None]
        ax.plot(pts, [m for m in mean if m is not None], label=f"PPO-RSMA ({len(members)} seeds, mean)", **STYLE["PPO-RSMA"])
        # faint member lines
        for tag, _, data_m in members:
            vals = data_m.get(key, [])
            _plot_series(
                ax, xs, vals,
                f"member {tag}",
                dict(color="#d62728", marker="o", lw=1.0, ms=3, alpha=0.25, zorder=2),
            )
        # min-max envelope
        lo_pts = [s for s, v in zip(xs, lo) if v is not None]
        hi_pts = [s for s, v in zip(xs, hi) if v is not None]
        if lo_pts and hi_pts:
            ax.fill_between(xs, [v if v is not None else float("nan") for v in lo],
                            [v if v is not None else float("nan") for v in hi],
                            color="#d62728", alpha=0.15, zorder=1,
                            label="min–max seed envelope")

        ax.set_title(f"{title}\n({note})", fontsize=11)
        ax.set_xlabel("SNR (dB)")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(snrs_b)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        "RSMA uplink energy-efficient scheduling — benchmark with PPO-RSMA seed-replication band\n"
        f"(imperfect CSI, 50 episodes/SNR point, source: {Path(baseline_log).name})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_ablation(specs: list[str], metric: str, out: str) -> None:
    if metric not in {*METRIC_LABEL, "ee"}:
        raise SystemExit(f"--metric must be one of {sorted({*METRIC_LABEL, 'ee'})}")

    curves: list[tuple[str, list[float], list[float | None]]] = []
    sdma_ref: tuple[list[float], list[float | None]] | None = None
    for spec in specs:
        tag, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--spec must be TAG=PATH, got {spec!r}")
        snrs, data = parse_overview(path)
        algo = "PPO-RSMA" if "PPO-RSMA" in data else next(iter(data))
        curves.append((tag, snrs, data[algo].get(metric, [])))
        if sdma_ref is None and "SDMA" in data:
            sdma_ref = (snrs, data["SDMA"].get(metric, []))
        print(f"parsed {tag:18s} <- {path}")

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    cmap = plt.get_cmap("tab10")
    for i, (tag, snrs, values) in enumerate(curves):
        # offset by 1: cmap(0) is the same blue as the fixed SDMA reference line
        _plot_series(ax, snrs, values, tag, dict(color=cmap((i + 1) % 10), marker="o", lw=1.9, ms=6))
    if sdma_ref:
        _plot_series(
            ax,
            sdma_ref[0],
            sdma_ref[1],
            "SDMA (reference)",
            dict(color="#1f77b4", marker="s", ls="--", lw=2.6, ms=8, zorder=6),
        )
    title = "Energy efficiency  R_sum / P" if metric == "ee" else metric
    ax.set_title(f"{title} vs SNR — checkpoint ablation", fontsize=13)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best", ncol=2)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    p_all = sub.add_parser("all", help="6-panel figure, one curve per algorithm")
    p_all.add_argument("--log", required=True, help="eval report .txt to parse")
    p_all.add_argument("--out", default="logs/figs/benchmarks_all_algorithms.png")

    p_ab = sub.add_parser("ablation", help="overlay many checkpoints on one metric")
    p_ab.add_argument("--spec", action="append", required=True, help="TAG=PATH, repeatable")
    p_ab.add_argument("--metric", default="delivery_ratio")
    p_ab.add_argument("--out", default="logs/figs/ablation.png")

    p_bb = sub.add_parser("seedband", help="baselines + PPO-RSMA seed-replication band")
    p_bb.add_argument("--baseline-log", required=True, help="eval report with baseline columns (SDMA/NOMA/EqualPower)")
    p_bb.add_argument("--ppp-report", action="append", required=True, help="TAG=PATH of a PPO-RSMA eval report, repeatable")
    p_bb.add_argument("--extra-report", action="append", help="PATH of an additional eval report whose algorithm columns are merged into the figure, repeatable")
    p_bb.add_argument("--out", default="logs/figs/benchmarks_seed_band.png")

    args = ap.parse_args(argv)
    if args.mode == "all":
        plot_all(args.log, args.out)
    elif args.mode == "seedband":
        plot_seed_band(args.baseline_log, args.ppp_report, args.out, args.extra_report)
    else:
        plot_ablation(args.spec, args.metric, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
