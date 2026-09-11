# Gap Map

_Field gaps with stable IDs. Best `[5, 0.9163, -11.0186, 0.6914]` = `[points_met/7, avg_delivery, queue_metric, ee_ratio]`, lexicographic (points_met dominates). Criterion CORRECTED from `compute_score` (optimize_loop.py:1777-1778)._

## G1 — Two low-SNR points fail the per-point delivery+queue gate
`points_met` stuck at **5/7**. The 2 low-SNR points {0,5} dB fail the per-point gate: **delivery ≥ SDMA×1.0 AND queue ≤ SDMA×1.1** (both relative, ≥/≤ so matching SDMA passes). EE>1.05 / power<SDMA are **NOT** per-point gates. TOP priority: close G1 without regressing the 5 passing points.

## G2 — Global EE ratio below termination target
`ee_ratio` = **0.6914 < 1.05** — a GLOBAL `check_termination` + 4th-lexicographic-slot concern, **not** a `points_met` blocker.

## G3 — Average delivery below SDMA
`avg_delivery` = **0.9163 < 1.0×** — a GLOBAL aggregate (2nd lexicographic slot), not the per-point gate.

## G4 — Mechanism-family exhaustion (meta-gap)
40 falsified/desk-rejected directions span: Lagrangian/PID-dual, CuRLTraC-variant curricula, gradient-manip (CRPO/primal-proj), model-soup/SWA, quantile-critic, SIC-order, global & per-bucket `group_alpha`, scalar penalty ladders, naive bucket-MoE/obs-augment, EE-axis solver theory. New work must be **structurally orthogonal**. Campaign-prioritized axes (see `idea-stage/IDEA_REPORT.md`): train/eval CSIR-distribution alignment, miner alignment to the true 2-bar gate + its [-5,-2.5]dB support hole, DVQ-drift de-saturation, cheap no-training failure diagnostics.
