import sys
from pathlib import Path

aris = Path((Path.home() / ".aris" / "repo").read_text(encoding="utf-8").strip())
sys.path.insert(0, str(aris / "tools"))
import research_wiki as rw  # noqa: E402

WIKI = "research-wiki"
STAMP = "novelty-check 2026-09-03 (GLM-5.3 cross-model): PROCEED WITH CAUTION, 5/10. No named paper contains the result -> ABANDON unavailable. Novelty lives in the FINDING, not the DR/alignment mechanism."
GATE = (
    "PRE-REGISTERED CAUSALITY GATE (reviewer-prescribed, run BEFORE retraining; upgrades CAUTION->PROCEED iff it lands): "
    "(1) Swap-A: eval the existing fixed-eps policy at 30dB with sigma_est inflated to ~0.2 -> if delivery COLLAPSES, error scale (not SNR) is causal. "
    "(2) Swap-B: eval the same policy at 0dB with near-perfect CSI -> if it PASSES, CSIR error is the cause. "
    "(3) Report per-point sigma_est incl. 10-15dB (smooth spread => 'OOD exactly at {0,5}' is only approximate). "
    "FALSIFICATION: if the aligned-eps retrain fails to recover {0,5}dB, the root-cause claim is falsified -> RE-SCOPE to feasibility/error-floor (is delivery capped for ANY policy at sigma_est~0.2?), not abandonment."
)

# ---- L2-4: the diagnosis / Wave-0 audit ----
l24_thesis = (
    "Does the policy TRAIN under a fixed epsilon_csi while the EVALUATOR samples an SNR-dependent sigma_est that GROWS as SNR drops "
    "(sigma_est=0.2045@0dB vs 0.0119@30dB, 17x)? If so the policy is OOD at exactly the 2 failing points and nowhere else - uniquely "
    "explaining why 36 interventions across 6 families all died there. FIX if confirmed: flip snr_dependent_csi / widen training epsilon "
    "to cover eval-time low-SNR variance, retrain with the accepted s0/s3 recipe.\n\n" + GATE
)
l24_risks = (
    "Jury: ACCEPT-PILOT, ELEVATE TO WAVE 0 - the single strongest bet. A config/log audit (hours), NOT gated by L5-5 (a distribution "
    "mismatch is a bug regardless of whether CSIR loss dominates). One-flag, benchmark-preserving fix. VERIFY the epsilon-sampling "
    "plumbing is NOT inside rsma_env.py's immutable physical-call segment. If the audit comes back aligned, that negative is itself "
    "decisive (eliminates the distribution-bug hypothesis, hands the bet to L4-5+L4-1).\n\n" + STAMP + " "
    "Closest prior work: P2 (2507.06602 DR for RAN generalization - attributes mismatch to topology/traffic, not CSIR-error-at-low-SNR), "
    "P5 (2501.19019 uplink-RSMA low-SNR CSIR-dominance fact, non-RL), P1 (2203.03690 robust RSMA downlink, non-RL). Reviewer: the diagnosis "
    "is a novel INSTANCE of the well-worn 'train at a fixed operating point -> randomize it' meta-pattern; two confounds not yet excluded "
    "(intrinsic low-SNR hardness; error floors at sigma_est~0.2)."
)
rw.upsert_idea(
    WIKI,
    "l2-4-csir-align-audit",
    "L2-4 Train/eval CSIR-distribution alignment audit",
    description="idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1,G3. Targets points_met 5->7 at the 2 low-SNR points. "
    + STAMP,
    stage="proposed",
    outcome="pending",
    thesis=l24_thesis,
    risks=l24_risks,
    tags=["csir", "wave0", "data-distribution", "top-bet", "L2", "novelty-check", "proceed-with-caution"],
    update_on_exist=True,
)

# ---- L2-1: the DR fix / Wave-1 ----
l21_thesis = (
    "Per-episode randomize the CSI-estimation-error scale epsilon_csi (e.g. U[0.5x,2x] nominal), low-SNR episodes drawn from the upper "
    "half, so the policy MARGINALIZES over the estimation-error distribution instead of overfitting the nominal realization. Center the "
    "range on the audited epsilon (L2-4). Rides mining (s1-s3) unchanged."
)
l21_risks = (
    "Jury: ACCEPT-PILOT. Best data-side CSIR robustness play; sequence AFTER L2-4 so randomization centers on the aligned epsilon. "
    "Orthogonal to B (randomizes an ENV INPUT PARAMETER every episode, no task-difficulty schedule), D/i18 (obs untouched), A (no dual "
    "change). Mutable env-reset plumbing + curriculum hook; channel.py physics never edited.\n\n" + STAMP + " "
    "BINDING CAVEAT (C3): the 'SNR-weighted' qualifier is DECORATION until it beats plain UNIFORM eps-randomization in an ablation - "
    "run that (aligned-eps retrain recovers {0,5}dB where a uniform-DR control recovers LESS) or drop the qualifier. Include a low-SNR "
    "SDMA / P1-style robust-optimization baseline to show the recovered policy isn't merely degenerating to SDMA. Closest prior work: "
    "P2 (2507.06602 DR), P7 (2412.14297 distributionally-robust policy learning, generic RL theory)."
)
rw.upsert_idea(
    WIKI,
    "l2-1-csir-domain-rand",
    "L2-1 CSI-error-scale domain randomization (low-SNR weighted)",
    description="idea-creator 2026-09 campaign (jury ACCEPT). Addresses G1,G3. Targets points_met 5->7 at the 2 low-SNR points. "
    + STAMP,
    stage="proposed",
    outcome="pending",
    thesis=l21_thesis,
    risks=l21_risks,
    tags=["csir", "wave1", "data-distribution", "L2", "novelty-check", "proceed-with-caution"],
    update_on_exist=True,
)

rw.rebuild_query_pack(WIKI)
print(
    "WRITE-BACK OK: l2-4-csir-align-audit + l2-1-csir-domain-rand enriched with novelty verdict + pre-registered gate."
)
