---
type: idea
node_id: idea:l4-5-miner-composite-key
title: "L4-5 Miner key = delivery + queue-shortfall composite"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["data-shaping", "wave1", "miner", "queue", "L4"]
---

# L4-5 Miner key = delivery + queue-shortfall composite

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
Extend the hard-example miner key from delivery-shortfall-ONLY to delivery + w_q*max(0,queue_ratio-1.1) (+ optional EE-margin) so it oversamples the ACTUAL binding constraint at 0/5 dB. Aligns the one winning mechanism (data shaping) to the TRUE 2-bar per-point gate. Rides s3 (per-eval-point fine-grained mining).

## Key risks
Jury: ACCEPT-PILOT. Keep delivery DOMINANT for graceful degradation - queue-shortfall may be sparse (the 1.1x slack makes most states zero-shortfall), degenerating the key back to delivery-only. Clean: extends the WINNING family, not any failed family. Mutable: src/training/main.py miner only.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

