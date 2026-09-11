---
type: idea
node_id: idea:l2-1-csir-domain-rand
title: "L2-1 CSI-error-scale domain randomization (low-SNR weighted)"
stage: proposed
outcome: pending
added: 2026-09-03T15:58:53Z
based_on: []
target_gaps: []
tags: ["csir", "wave1", "data-distribution", "L2", "novelty-check", "proceed-with-caution"]
---

# L2-1 CSI-error-scale domain randomization (low-SNR weighted)

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1,G3. Targets points_met 5->7 at the 2 low-SNR points. novelty-check 2026-09-03 (GLM-5.3 cross-model): PROCEED WITH CAUTION, 5/10. No named paper contains the result -> ABANDON unavailable. Novelty lives in the FINDING, not the DR/alignment mechanism.

## Thesis
Per-episode randomize the CSI-estimation-error scale epsilon_csi (e.g. U[0.5x,2x] nominal), low-SNR episodes drawn from the upper half, so the policy MARGINALIZES over the estimation-error distribution instead of overfitting the nominal realization. Center the range on the audited epsilon (L2-4). Rides mining (s1-s3) unchanged.

## Key risks
Jury: ACCEPT-PILOT. Best data-side CSIR robustness play; sequence AFTER L2-4 so randomization centers on the aligned epsilon. Orthogonal to B (randomizes an ENV INPUT PARAMETER every episode, no task-difficulty schedule), D/i18 (obs untouched), A (no dual change). Mutable env-reset plumbing + curriculum hook; channel.py physics never edited.

novelty-check 2026-09-03 (GLM-5.3 cross-model): PROCEED WITH CAUTION, 5/10. No named paper contains the result -> ABANDON unavailable. Novelty lives in the FINDING, not the DR/alignment mechanism. BINDING CAVEAT (C3): the 'SNR-weighted' qualifier is DECORATION until it beats plain UNIFORM eps-randomization in an ablation - run that (aligned-eps retrain recovers {0,5}dB where a uniform-DR control recovers LESS) or drop the qualifier. Include a low-SNR SDMA / P1-style robust-optimization baseline to show the recovered policy isn't merely degenerating to SDMA. Closest prior work: P2 (2507.06602 DR), P7 (2412.14297 distributionally-robust policy learning, generic RL theory).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

