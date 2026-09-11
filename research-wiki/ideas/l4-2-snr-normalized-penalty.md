---
type: idea
node_id: idea:l4-2-snr-normalized-penalty
title: "L4-2 SNR-normalized drift/penalty equalizer"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["lyapunov", "wave2", "reward-structure", "L4"]
---

# L4-2 SNR-normalized drift/penalty equalizer

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
Consume the unused sigma2 argument (compute_reward receives it at rsma_env.py:851 but ignores it) to build a monotone n(sigma2) that equalizes the penalty-block RMS across SNR, making the reward magnitude (hence the value target) regime-invariant. n~1 at high SNR (protects the 5), n<1 at low SNR.

## Key risks
Jury: ACCEPT-ON-PAPER. Regime-aware reward-scale surgery DOMINATED by L4-1 (regime-agnostic bound); follow-up only if cross-regime imbalance persists after L4-1. NOT family A (no lambda/Y/X dual update change), NOT F/i21 (Dinkelbach/EE term untouched), NOT C/i2 (scales a reward penalty, alpha untouched).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

