---
type: idea
node_id: idea:l3-1-soft-moe-router
title: "L3-1 Soft-MoE interference-vs-power router"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["routing", "wave2", "architecture", "L3"]
---

# L3-1 Soft-MoE interference-vs-power router

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
A softmax router over G=2 expert heads CLONED from best, keyed on a dimensionless interference-vs-power regime scalar r=mean(interference share)/mean(signal-power proxy) from existing per-user obs. expert-1 dominates as r->high (interference-limited: near-SDMA, robust, low-power), expert-0 as r->low. Cloned init => uniform mixture == best at step-0.

## Key risks
Jury: ACCEPT-ON-PAPER. Cloned-init protection is real BUT the core bet is family-D's LOST bet (regime-specialized heads); cloned init protects step-0, NOT the gradient dynamics. Needs diagnostic evidence of regime collapse first. NOT D/i3 (SOFTMAX mixture, no SNR gather, no frozen trunk, all experts get grad), NOT D/i18 (no bucket-context vector added to obs).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

