---
type: idea
node_id: idea:l3-2-csir-modulation-router
title: "L3-2 CSIR-confidence output-modulation router (REJECTED)"
stage: archived
outcome: negative
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["rejected", "desk-reject", "csir", "routing", "L3"]
---

# L3-2 CSIR-confidence output-modulation router (REJECTED)

**stage:** `archived`  ·  **outcome:** `negative`

idea-creator 2026-09 campaign (jury REJECT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
A softmax router keyed on realized CSIR-error confidence (eps_norm + per-user error_variance) mixes M=2 output-modulation vectors (gamma,beta) calibrating a shared head's outputs.

## Key risks
Jury: REJECT (desk-reject, NOT experimentally tested). Architecture-side CSIR hedge DOMINATED by the data-side L2-4/L2-1 pair at a fraction of the risk; routing complexity on a shared critic with no return-norm. Prefer L2-4/L2-1.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

