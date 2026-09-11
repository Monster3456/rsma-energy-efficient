---
type: idea
node_id: idea:l1-4-triggered-reduction-onset
title: "L1-4 Performance-triggered ONSET of power reduction"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["solver", "wave2", "curriculum", "L1"]
---

# L1-4 Performance-triggered ONSET of power reduction

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
Let RL own power during early exploration; at a low-SNR delivery-feasibility trigger switch power to solver-fixed for the rest of training so alpha fine-tunes against a stable power manifold. Rides s0 (extends performance-triggered difficulty-annealing from the env-difficulty axis to the action-space-reduction axis).

## Key risks
Jury: NEEDS-DIAGNOSTIC-FIRST. Inherits L1-1's unvalidated premise PLUS a switching discontinuity. The s0-style trigger instinct is RIGHT - keep the sketch, gate on the L1-gate probe + L5-3. NOT banlist B (i24/i28/i7 annealed ENV/SIGNAL difficulty; this schedules an ACTION-SPACE reduction onset - a different object being timed).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

