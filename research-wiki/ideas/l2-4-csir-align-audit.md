---
type: idea
node_id: idea:l2-4-csir-align-audit
title: "L2-4 Train/eval CSIR-distribution alignment audit"
stage: proposed
outcome: pending
added: 2026-09-03T15:58:53Z
based_on: []
target_gaps: []
tags: ["csir", "wave0", "data-distribution", "top-bet", "L2", "novelty-check", "proceed-with-caution"]
---

# L2-4 Train/eval CSIR-distribution alignment audit

**stage:** `proposed`  ·  **outcome:** `pending`

idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1,G3. Targets points_met 5->7 at the 2 low-SNR points. novelty-check 2026-09-03 (GLM-5.3 cross-model): PROCEED WITH CAUTION, 5/10. No named paper contains the result -> ABANDON unavailable. Novelty lives in the FINDING, not the DR/alignment mechanism.

## Thesis
Does the policy TRAIN under a fixed epsilon_csi while the EVALUATOR samples an SNR-dependent sigma_est that GROWS as SNR drops (sigma_est=0.2045@0dB vs 0.0119@30dB, 17x)? If so the policy is OOD at exactly the 2 failing points and nowhere else - uniquely explaining why 36 interventions across 6 families all died there. FIX if confirmed: flip snr_dependent_csi / widen training epsilon to cover eval-time low-SNR variance, retrain with the accepted s0/s3 recipe.

PRE-REGISTERED CAUSALITY GATE (reviewer-prescribed, run BEFORE retraining; upgrades CAUTION->PROCEED iff it lands): (1) Swap-A: eval the existing fixed-eps policy at 30dB with sigma_est inflated to ~0.2 -> if delivery COLLAPSES, error scale (not SNR) is causal. (2) Swap-B: eval the same policy at 0dB with near-perfect CSI -> if it PASSES, CSIR error is the cause. (3) Report per-point sigma_est incl. 10-15dB (smooth spread => 'OOD exactly at {0,5}' is only approximate). FALSIFICATION: if the aligned-eps retrain fails to recover {0,5}dB, the root-cause claim is falsified -> RE-SCOPE to feasibility/error-floor (is delivery capped for ANY policy at sigma_est~0.2?), not abandonment.

## Key risks
Jury: ACCEPT-PILOT, ELEVATE TO WAVE 0 - the single strongest bet. A config/log audit (hours), NOT gated by L5-5 (a distribution mismatch is a bug regardless of whether CSIR loss dominates). One-flag, benchmark-preserving fix. VERIFY the epsilon-sampling plumbing is NOT inside rsma_env.py's immutable physical-call segment. If the audit comes back aligned, that negative is itself decisive (eliminates the distribution-bug hypothesis, hands the bet to L4-5+L4-1).

novelty-check 2026-09-03 (GLM-5.3 cross-model): PROCEED WITH CAUTION, 5/10. No named paper contains the result -> ABANDON unavailable. Novelty lives in the FINDING, not the DR/alignment mechanism. Closest prior work: P2 (2507.06602 DR for RAN generalization - attributes mismatch to topology/traffic, not CSIR-error-at-low-SNR), P5 (2501.19019 uplink-RSMA low-SNR CSIR-dominance fact, non-RL), P1 (2203.03690 robust RSMA downlink, non-RL). Reviewer: the diagnosis is a novel INSTANCE of the well-worn 'train at a fixed operating point -> randomize it' meta-pattern; two confounds not yet excluded (intrinsic low-SNR hardness; error floors at sigma_est~0.2).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

