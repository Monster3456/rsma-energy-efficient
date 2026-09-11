---
type: idea
node_id: idea:l1-gate-probe
title: "L1-gate Runtime solver-power-elimination probe (jury-added)"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["solver", "wave0", "probe", "L1"]
---

# L1-gate Runtime solver-power-elimination probe (jury-added)

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
Eval-time overwrite of the Worker power block with SolverExpert's solve on an EXISTING checkpoint at 0/5/10 dB - does solver power beat learned power ON DELIVERY? Gates the weeks-scale L1 cluster for the cost of one eval run. No training.

## Key risks
Jury: ACCEPT-PILOT (added by the jury; the bundle omitted it). Go/no-go for L1-1/L1-4. SolverExpert is currently wired ONLY into offline BC (never at decision time), so this is the cheapest test of the variable-reduction premise. Touches only mutable eval-side glue; calls the immutable eval unchanged.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

