---
type: idea
node_id: idea:l5-2-deliv-trajectory
title: "L5-2 Transient-vs-steady-state delivery trajectory"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["diagnostic", "wave0", "queue", "L5"]
---

# L5-2 Transient-vs-steady-state delivery trajectory

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
Log cumulative delivery/queue EVERY step on matched seeds, PPO vs SDMA at 0/5 dB (+10 dB control). The scored delivery is a whole-trajectory integral incl. the empty-queue ramp (T=200). Is the gap decided in the early fill transient (warm-up artifact) or does it diverge in steady state (genuine deficit)? Opposite answers => opposite fixes.

## Key risks
Jury: ACCEPT-PILOT. Fork between curriculum plays and service-rate plays. Validate the matched-seed replication against immutable-eval aggregates first. Per-step logging must NOT modify the immutable eval script (import via importlib, the established _diag_* pattern).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

