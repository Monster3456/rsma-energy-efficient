---
type: idea
node_id: idea:l5-1-bar-slack
title: "L5-1 Bar-slack binding-criterion decomposition"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["diagnostic", "wave0", "criterion", "L5"]
---

# L5-1 Bar-slack binding-criterion decomposition

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
Re-derive the per-point bars from immutable-eval output tables to find WHICH bar binds at 0/5 dB and the relative gap size (SDMA-PPO)/SDMA + the gate noise floor vs episode count. ALREADY surfaced the Section-A criterion correction (points_met gates on delivery+queue only, >=/<=, per compute_score:1777-1778). MVE: zero-training re-analysis of logs/eval_*.txt + one 50-ep eval of best ckpt (call evaluate_snr_sweep.py, never edit).

## Key risks
Jury: ACCEPT-PILOT. Prices every other candidate (3% gap => cheap nudges; 25% => architecture). Also pin delivery/queue aggregation semantics (sum vs per-user mean; end-state vs time-avg). Orthogonal to F/i31 (group_alpha window sweep + csi switch) and F/i44 (hyperparameter liveness probe) - perturbs nothing.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

