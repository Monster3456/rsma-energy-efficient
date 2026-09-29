"""Paired per-SNR statistics vs Ours champion, for the paper's statistical section.

Unified per-point delivery comes from scripts.plot_benchmarks.parse_overview over
the official eval reports. Tests are paired at the 7-SNR design level
(df=6); family = external comparators only (Holm-Bonferroni, alpha=0.05);
our engine's own variants (DiffOpt*/JointOpt) are reported as oracle references,
outside the significance family. Also prints the 5-seed 95% CI of the champion
recipe best-guard deliveries.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

from scripts.plot_benchmarks import parse_overview

REPO = Path(".").resolve()
OURS = 0.949621  # 5-seed best-guard mean

CANDS = {
    "SDMA": (REPO / "logs/_eval_bench_all.txt", "SDMA"),
    "NOMA": (REPO / "logs/_eval_bench_all.txt", "NOMA"),
    "EqualPower": (REPO / "logs/_eval_bench_all.txt", "EqualPower"),
    "AdaptivePower": (REPO / "logs/_eval_extended.txt", "AdaptivePower"),
    "WaterFilling": (REPO / "logs/_eval_extended.txt", "WaterFilling"),
    "LDPTRLQ": (REPO / "logs/_eval_ldptrlq_best.txt", "PPO-RSMA"),
    "TD3": (REPO / "logs/_eval_td3_final.txt", "TD3"),
    "DDPG": (REPO / "logs/_eval_ddpg_final.txt", "DDPG"),
}
ORACLE = {  # our-engine variants: reported, not in the Holm family
    "DiffOpt": (REPO / "logs/_eval_extended.txt", "DiffOpt"),
    "DiffOptCausal": (REPO / "logs/_eval_extended_causal.txt", "DiffOptCausal"),
    "JointOpt": (REPO / "logs/_eval_jointopt2.txt", "JointOpt"),
}

OURS_LOG = REPO / "logs/_eval_m5_A_bestguard.txt"


def deliv(path: Path, algo: str) -> list[float]:
    try:
        _, data = parse_overview(path)
        v = data[algo]["delivery_ratio"]
        vals = [x for x in v if x is not None]
        if vals:
            return vals
    except (KeyError, ValueError):
        pass
    # fallback: bare table "| SNR | Deliv | Q | P | ..." -> delivery is field 2
    out = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ln.strip().strip("|").split("|")
        if len(m) >= 2 and m[0].strip().isdigit():
            try:
                out.append(float(m[1].strip()))
            except ValueError:
                continue
    if not out:
        raise SystemExit(f"no per-point delivery extracted for {path} / {algo}")
    return out


def report(ours: list[float], name: str, cand: list[float], n: int) -> tuple[float, float, float, float]:
    d = np.asarray(ours) - np.asarray(cand)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    t = md / (sd / np.sqrt(len(d)))
    p = 2.0 * stats.t.sf(abs(t), df=len(d) - 1)
    cohen = md / sd
    return md, t, p, cohen


def main() -> int:
    ours = deliv(OURS_LOG, "PPO-RSMA")
    print(f"Ours champion per-point delivery: {[round(x,4) for x in ours]}")
    print(f"5-seed best-guard: mean={OURS:.4f} "
          f"95% CI=[{OURS - 2.776 * 0.0047 / np.sqrt(5):.4f}, {OURS + 2.776 * 0.0047 / np.sqrt(5):.4f}]")

    def _cand(name, spec):
        path, algo = spec
        return name, *report(ours, name, deliv(path, algo), 7)

    rows = [_cand(n, s) for n, s in CANDS.items()] + [_cand(n, s) for n, s in ORACLE.items()]

    rows.sort(key=lambda r: r[3])  # by p
    fam = [r for r in rows if r[0] not in ORACLE]
    alpha = 0.05
    for i, (_, _, _, p, _) in enumerate(fam):
        sig = "SIG" if p <= alpha / (len(fam) - i) else "ns"
        print(f"  Holm[{i}] {sig}")
    # cache Holm thresholds on the family order
    th = {r[0]: alpha / (len(fam) - i) for i, r in enumerate(fam)}

    print("\n| comparator | mean deliv (ours vs) | mean diff | t (df=6) | p | Cohen's d | Holm-sig |")
    for name, md, t, p, cohen in rows:
        tag = "oracle-ref" if name in ORACLE else ("SIG" if p <= th[name] else "ns")
        print(f"| {name:13s} | {OURS:.4f} vs {OURS-md:.4f} | {md:+.4f} | {t:+.3f} | {p:.3g} | {cohen:+.2f} | {tag} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())