---
type: idea
node_id: idea:l5-5-csir-attribution
title: "L5-5 CSIR min-branch rate-loss relative attribution"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["diagnostic", "wave0", "csir", "L5"]
---

# L5-5 CSIR min-branch rate-loss relative attribution

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
rate_calc returns min(declared@error_bound=sigma_est, actual@H_true). Split the penalty into a robust-regularization haircut vs a realization-mismatch haircut and compute the PPO-minus-SDMA DIFFERENCE in CSIR-attributable rate loss at 0/5 dB (+20 dB control). Tests whether L2's CSIR-dominance premise survives in the RELATIVE sense the delivery bar needs.

## Key risks
Jury: ACCEPT-PILOT (2nd wave). Gates L2-1/L2-3 (NOT L2-4). A near-zero PPO-minus-SDMA difference DEMOTES the whole CSIR robustness axis. MUST be an EXTERNAL dual-call harness around the public rate_calc (declare at sigma_est vs declare with perfect CSI) - NEVER instrumentation of its internals (immutable). Distinct from F/i31 probe B (which switched csi_mode=perfect, changing the trajectory + confounding re-adaptation).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

