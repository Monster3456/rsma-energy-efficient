---
type: idea
node_id: idea:l4-1-bounded-dvq-drift
title: "L4-1 Bounded/normalized DVQ Lyapunov drift term"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["lyapunov", "wave1", "queue", "reward-structure", "L4"]
---

# L4-1 Bounded/normalized DVQ Lyapunov drift term

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
Replace the DVQ drift's fixed linear scale y_weight=Y/dvq_y_scale(=300) with a bounded/normalized per-step form tanh(Y/y_ref) (or Y/running-shortfall-RMS) in lyapunov.py compute_reward. At low SNR Y pins at cap 5000 => y_weight~16.7 (wound up, no differential gradient); with NO return normalization the shared critic must fit a regime-specific value scale. Bounding restores the queue-pressure gradient + stabilizes value targets. High-SNR Y~0 so tanh~linear = unchanged (protects the 5).

## Key risks
Jury: ACCEPT-PILOT but EVIDENCE-GATED - FAMILY-A ADJACENCY IS REAL (the bundle's 'NOT family A' was too glib; it IS a reward-function edit). What separates it from blind family-A tuning is a VERIFIED saturation pathology: gate on Y/y_weight logging - if Y isn't pinned at cap, this is family A in a tanh costume. Also verify y_ref leaves the 5/7.5/10 dB operating range uncompressed. Leaves the Y accumulator + lambda dual untouched.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

