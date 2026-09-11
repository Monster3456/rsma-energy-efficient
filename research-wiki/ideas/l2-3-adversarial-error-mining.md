---
type: idea
node_id: idea:l2-3-adversarial-error-mining
title: "L2-3 Adversarial CSI-error-realization mining"
stage: proposed
outcome: pending
added: 2026-09-03T15:12:44Z
based_on: []
target_gaps: []
tags: ["csir", "wave1", "data-shaping", "miner", "L2"]
---

# L2-3 Adversarial CSI-error-realization mining

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1,G3. Targets points_met 5->7 at the 2 low-SNR points.

## Thesis
Extend the CuRLTraC miner key from state-difficulty to adverse per-user error_variance realizations at 0/5 dB; store + oversample the channel-RNG seeds whose upper-tail error coincides with constraint violation. Teaches the policy to hedge exactly the adversarial CSIR configs. Decisive NEGATIVE if failure is uniform across error draws (falsifies CSIR-dominance for our pipeline).

## Key risks
Jury: ACCEPT-PILOT. Cheapest CSIR training-side test on the winning family; gate on L5-5 showing a nonzero PPO-minus-SDMA difference; CAP the oversampling weight (worst-case error draws may be unlearnable and spill over via weight sharing). Literally a new mining key on s1/s2/s3 - inherits the only falsification-resistant family.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

