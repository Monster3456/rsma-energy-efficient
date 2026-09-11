---
type: idea
node_id: idea:l1-1-solver-power-elimination
title: "L1-1 Runtime solver power-elimination -> RL owns per-user alpha"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["solver", "wave2", "architecture", "L1"]
---

# L1-1 Runtime solver power-elimination -> RL owns per-user alpha

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
At decision time overwrite the Worker's power block with SolverExpert's per-slot gradient solve (conditioned on the current split), set group_level_alpha=False so the Worker's alpha block goes live per-user, and mask the dead power block in the PPO log_prob. Power is ELIMINATED (variable-reduction, per arXiv 2311.16895), not blended. RL's gradient then concentrates on alpha (which controls the CSIR-robust common/private tradeoff that dominates at low SNR).

## Key risks
Jury: NEEDS-DIAGNOSTIC-FIRST. BUNDLES TWO CONFOUNDED CHANGES (solver power + group_level_alpha=False reversing the current best config) => a failure would be unattributable. Zero evidence solver power beats learned power ON DELIVERY at 0/5 dB. Run the L1-gate probe first; split the two changes. NOT F/i3 (hard replace, not interpolate), NOT C (no global/tabled alpha), NOT the offline BC (removes power from RL at decision time rather than teaching RL to imitate it).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

