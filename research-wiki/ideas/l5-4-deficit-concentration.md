---
type: idea
node_id: idea:l5-4-deficit-concentration
title: "L5-4 Per-user deep-fade deficit concentration"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["diagnostic", "wave0", "fairness", "L5"]
---

# L5-4 Per-user deep-fade deficit concentration

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1,G3. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
Decompose the pooled delivery deficit (SDMA_served - PPO_served) at 0/5 dB by per-user path-loss offset (28 dB spread over K=24). Is the gap carried by a small deep-fade cohort (queues saturated at Q_max, arrivals->overflow) while surplus rate is WASTED on strong users with empty queues (served=min(d_ratio*min(R*tau,Q),Q))? Also measure the wasted-capacity differential vs SDMA and the d_ratio throttle.

## Key risks
Jury: ACCEPT-PILOT. If the deficit concentrates it NAMES the mining target for L4-5/L3-4 and hands L3 a physically-meaningful soft-router key (per-user feasibility) instead of the SNR bucket id that failed in D/i3. Distinct from existing _diag_user_bias.py (which profiled PPO alone, no PPO-minus-SDMA delta, no path-loss key).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

