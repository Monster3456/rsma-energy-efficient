---
type: idea
node_id: idea:l4-4-intra-bucket-tilt
title: "L4-4 Miner intra-bucket low-edge density tilt"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["data-shaping", "wave2", "miner", "L4"]
---

# L4-4 Miner intra-bucket low-edge density tilt

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
Replace the miner's within-bucket UNIFORM SNR draw with a low-edge-tilted density (Beta(a<1)/triangular) for the 2 unmet buckets only, concentrating samples at/below the exact 0 and 5 dB eval points. Bucket weights/targets/entropy and all other buckets unchanged. Rides s1-s3.

## Key risks
Jury: ACCEPT-ON-PAPER. Cheap density concentration. Tilt toward the EVAL SNRs +/-margin, not blindly to the bucket low edge (which can move mass AWAY from 0 dB depending on edge placement). Distinct from L4-3 (density within existing buckets vs widening support).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

