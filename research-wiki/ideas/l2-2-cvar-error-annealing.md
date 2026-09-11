---
type: idea
node_id: idea:l2-2-cvar-error-annealing
title: "L2-2 CVaR-gated error-scale annealing (REJECTED)"
stage: archived
outcome: negative
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["rejected", "desk-reject", "csir", "curriculum", "L2"]
---

# L2-2 CVaR-gated error-scale annealing (REJECTED)

**stage:** `archived`  ·  **outcome:** `negative`

idea-creator 2026-09 campaign (jury REJECT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
Inflate the training CSI-error scale only when CVaR_0.2 of low-SNR delivery across error realizations clears the SDMA bar, else retract (a distributionally-robust curriculum trigger).

## Key risks
Jury: REJECT (desk-reject, NOT experimentally tested). A noisy CVaR trigger controller stacked on an UNVALIDATED axis; dominated by L2-1 (plain randomization). Revisit only if L2-1 succeeds and tail-specific annealing is still needed.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

