---
type: idea
node_id: idea:l4-3-miner-support-extension
title: "L4-3 Miner low-SNR support extension (-2.5 -> -5 dB)"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["data-shaping", "wave2", "miner", "hygiene", "L4"]
---

# L4-3 Miner low-SNR support extension (-2.5 -> -5 dB)

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
Extend _HM_EDGES[0] from -2.5 to -5.0 so the currently UNREACHABLE [-5,-2.5] dB extreme-low tail (vs snr_min_db=-5) becomes minable; weight the tail by the unmet low-SNR shortfall. Rides s1-s3.

## Key risks
Jury: ACCEPT-ON-PAPER (HYGIENE). Correct fix for the B.3 support bug BUT [-5,-2.5] has ZERO eval exposure (eval points are 0,5,...) so NO measurable points_met effect is possible. Demote from candidate to hygiene - fold the edge fix into the next miner PR, do not spend an iteration on it alone.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

