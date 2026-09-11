---
type: idea
node_id: idea:l5-3-load-crossing
title: "L5-3 Offered-load lambda* crossing at 0/5 dB"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["diagnostic", "wave0", "overload", "L5"]
---

# L5-3 Offered-load lambda* crossing at 0/5 dB

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
Sweep fixed lambda at 0/5 dB (via the mutable temporal-config override; immutable eval called unchanged) and locate the crossing lambda* where PPO delivery overtakes SDMA. Tests whether the 2 points sit above both methods' feasibility knee (load-calibration artifact) - MUST re-run with CURRENT constants (lambda_min=0.2, load_ratio_min=0.75; the earlier '7x overload' used stale lambda_min=1.5 and is RETRACTED).

## Key risks
Jury: ACCEPT-PILOT. Distinguishes load-calibration artifact from genuine deficit. DIAGNOSTIC ONLY - becoming a lambda retune collides with the hand-tuned B.6 constants (Task#27/43/46/47/48) and drifts the benchmark. Orthogonal to B (no training curriculum) and C (sweeps exogenous arrivals, not group_alpha).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

