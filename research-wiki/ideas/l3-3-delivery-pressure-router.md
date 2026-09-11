---
type: idea
node_id: idea:l3-3-delivery-pressure-router
title: "L3-3 Delivery-pressure soft router (EE-agg vs delivery-safe expert)"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["routing", "wave2", "architecture", "queue", "L3"]
---

# L3-3 Delivery-pressure soft router (EE-agg vs delivery-safe expert)

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
A softmax router keyed on queue-backlog delivery-pressure (mean queue_norm + DVQ/fairness backlog from existing obs) mixes 2 experts CLONED from best: high backlog => delivery-safe expert (clears delivery>=1.0, queue<=1.1); low backlog => EE-aggressive expert. Encodes the lexicographic delivery-first priority as ARCHITECTURE, not dual pressure. Cloned init => step-0 == best (protects the 5).

## Key risks
Jury: ACCEPT-ON-PAPER. Best L3 alignment to the corrected criterion, BUT its router key (DVQ backlog) SATURATES exactly at low SNR (B.5) - sequence AFTER L4-1 so the key isn't saturated. NOT family A (does not touch the Lagrangian/PID/CRPO or advantage), NOT D/i3 (SOFT softmax mixture over cloned experts keyed on backlog, no SNR gather, no frozen trunk). Construct unconditionally + buffer-gate so the immutable eval's strict=False loader picks it up.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

