---
type: idea
node_id: idea:l3-4-regime-reweighted-mining
title: "L3-4 Learned regime-classifier reweights mining"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["data-shaping", "wave1", "miner", "routing", "L3"]
---

# L3-4 Learned regime-classifier reweights mining

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
A tiny MLP regime classifier (input: physical regime descriptor = interference-vs-power + queue-pressure from existing obs; label: per-state delivery deficit, already in buffer.per_user_deficits/RCPO cost) outputs a soft posterior P(failing-regime|s) that reweights the CuRLTraC replay distribution. Generalizes the mining key from eval-point-id to a LEARNED physical regime. NO architecture change, NO obs change.

## Key risks
Jury: ACCEPT-PILOT. Pure data-shaping (rides s1-s3), lowest-risk L3. Sequence after L4-5. VERIFY per-state deficit labels actually exist in the buffer. NOT D/i3 (no expert heads, no arch change - only the sampling weight changes), NOT E/i9-i11 (keys on a learned soft regime posterior, not eval-point id; reweights replay sampling, not advantage temperature).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

