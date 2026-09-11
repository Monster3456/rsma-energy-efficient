#!/usr/bin/env python
"""Phase-7 write-back: 21 idea-creator candidates -> research-wiki idea nodes.

17 go-forward  -> stage=proposed, outcome=pending  (untested; NOT banlisted).
4 jury rejects -> stage=archived, outcome=negative (banlisted from re-proposal,
                  labelled "desk-reject, NOT experimentally tested" for honesty).

All target G1 (points_met 5->7 at the 2 low-SNR points). Reproducible/idempotent
(stable slugs + update_on_exist=True). Run:  python idea-stage/_write_ideas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"F:\PyCharm\project\rsma-energy-efficient")
WIKI = ROOT / "research-wiki"
aris = Path((Path.home() / ".aris" / "repo").read_text(encoding="utf-8").strip())
sys.path.insert(0, str(aris / "tools"))
import research_wiki as rw  # noqa: E402

# (slug, title, thesis, risks, stage, outcome, tags, gaps)
IDEAS = [
    # ---- Wave 0 diagnostics ----
    (
        "l5-1-bar-slack",
        "L5-1 Bar-slack binding-criterion decomposition",
        "Re-derive the per-point bars from immutable-eval output tables to find WHICH bar binds at 0/5 dB and the relative gap size (SDMA-PPO)/SDMA + the gate noise floor vs episode count. ALREADY surfaced the Section-A criterion correction (points_met gates on delivery+queue only, >=/<=, per compute_score:1777-1778). MVE: zero-training re-analysis of logs/eval_*.txt + one 50-ep eval of best ckpt (call evaluate_snr_sweep.py, never edit).",
        "Jury: ACCEPT-PILOT. Prices every other candidate (3% gap => cheap nudges; 25% => architecture). Also pin delivery/queue aggregation semantics (sum vs per-user mean; end-state vs time-avg). Orthogonal to F/i31 (group_alpha window sweep + csi switch) and F/i44 (hyperparameter liveness probe) - perturbs nothing.",
        "proposed",
        "pending",
        ["diagnostic", "wave0", "criterion", "L5"],
        "G1",
    ),
    (
        "l5-2-deliv-trajectory",
        "L5-2 Transient-vs-steady-state delivery trajectory",
        "Log cumulative delivery/queue EVERY step on matched seeds, PPO vs SDMA at 0/5 dB (+10 dB control). The scored delivery is a whole-trajectory integral incl. the empty-queue ramp (T=200). Is the gap decided in the early fill transient (warm-up artifact) or does it diverge in steady state (genuine deficit)? Opposite answers => opposite fixes.",
        "Jury: ACCEPT-PILOT. Fork between curriculum plays and service-rate plays. Validate the matched-seed replication against immutable-eval aggregates first. Per-step logging must NOT modify the immutable eval script (import via importlib, the established _diag_* pattern).",
        "proposed",
        "pending",
        ["diagnostic", "wave0", "queue", "L5"],
        "G1",
    ),
    (
        "l2-4-csir-align-audit",
        "L2-4 Train/eval CSIR-distribution alignment audit",
        "Does the policy TRAIN under a fixed epsilon_csi while the EVALUATOR samples an SNR-dependent sigma_est that GROWS as SNR drops (sigma_est=0.2045@0dB vs 0.0119@30dB, 17x)? If so the policy is OOD at exactly the 2 failing points and nowhere else - uniquely explaining why 36 interventions across 6 families all died there. FIX if confirmed: flip snr_dependent_csi / widen training epsilon to cover eval-time low-SNR variance, retrain with the accepted s0/s3 recipe.",
        "Jury: ACCEPT-PILOT, ELEVATE TO WAVE 0 - the single strongest bet. A config/log audit (hours), NOT gated by L5-5 (a distribution mismatch is a bug regardless of whether CSIR loss dominates). One-flag, benchmark-preserving fix. VERIFY the epsilon-sampling plumbing is NOT inside rsma_env.py's immutable physical-call segment. If the audit comes back aligned, that negative is itself decisive (eliminates the distribution-bug hypothesis, hands the bet to L4-5+L4-1).",
        "proposed",
        "pending",
        ["csir", "wave0", "data-distribution", "top-bet", "L2"],
        "G1,G3",
    ),
    (
        "l5-3-load-crossing",
        "L5-3 Offered-load lambda* crossing at 0/5 dB",
        "Sweep fixed lambda at 0/5 dB (via the mutable temporal-config override; immutable eval called unchanged) and locate the crossing lambda* where PPO delivery overtakes SDMA. Tests whether the 2 points sit above both methods' feasibility knee (load-calibration artifact) - MUST re-run with CURRENT constants (lambda_min=0.2, load_ratio_min=0.75; the earlier '7x overload' used stale lambda_min=1.5 and is RETRACTED).",
        "Jury: ACCEPT-PILOT. Distinguishes load-calibration artifact from genuine deficit. DIAGNOSTIC ONLY - becoming a lambda retune collides with the hand-tuned B.6 constants (Task#27/43/46/47/48) and drifts the benchmark. Orthogonal to B (no training curriculum) and C (sweeps exogenous arrivals, not group_alpha).",
        "proposed",
        "pending",
        ["diagnostic", "wave0", "overload", "L5"],
        "G1",
    ),
    (
        "l5-4-deficit-concentration",
        "L5-4 Per-user deep-fade deficit concentration",
        "Decompose the pooled delivery deficit (SDMA_served - PPO_served) at 0/5 dB by per-user path-loss offset (28 dB spread over K=24). Is the gap carried by a small deep-fade cohort (queues saturated at Q_max, arrivals->overflow) while surplus rate is WASTED on strong users with empty queues (served=min(d_ratio*min(R*tau,Q),Q))? Also measure the wasted-capacity differential vs SDMA and the d_ratio throttle.",
        "Jury: ACCEPT-PILOT. If the deficit concentrates it NAMES the mining target for L4-5/L3-4 and hands L3 a physically-meaningful soft-router key (per-user feasibility) instead of the SNR bucket id that failed in D/i3. Distinct from existing _diag_user_bias.py (which profiled PPO alone, no PPO-minus-SDMA delta, no path-loss key).",
        "proposed",
        "pending",
        ["diagnostic", "wave0", "fairness", "L5"],
        "G1,G3",
    ),
    (
        "l5-5-csir-attribution",
        "L5-5 CSIR min-branch rate-loss relative attribution",
        "rate_calc returns min(declared@error_bound=sigma_est, actual@H_true). Split the penalty into a robust-regularization haircut vs a realization-mismatch haircut and compute the PPO-minus-SDMA DIFFERENCE in CSIR-attributable rate loss at 0/5 dB (+20 dB control). Tests whether L2's CSIR-dominance premise survives in the RELATIVE sense the delivery bar needs.",
        "Jury: ACCEPT-PILOT (2nd wave). Gates L2-1/L2-3 (NOT L2-4). A near-zero PPO-minus-SDMA difference DEMOTES the whole CSIR robustness axis. MUST be an EXTERNAL dual-call harness around the public rate_calc (declare at sigma_est vs declare with perfect CSI) - NEVER instrumentation of its internals (immutable). Distinct from F/i31 probe B (which switched csi_mode=perfect, changing the trajectory + confounding re-adaptation).",
        "proposed",
        "pending",
        ["diagnostic", "wave0", "csir", "L5"],
        "G1",
    ),
    (
        "l1-gate-probe",
        "L1-gate Runtime solver-power-elimination probe (jury-added)",
        "Eval-time overwrite of the Worker power block with SolverExpert's solve on an EXISTING checkpoint at 0/5/10 dB - does solver power beat learned power ON DELIVERY? Gates the weeks-scale L1 cluster for the cost of one eval run. No training.",
        "Jury: ACCEPT-PILOT (added by the jury; the bundle omitted it). Go/no-go for L1-1/L1-4. SolverExpert is currently wired ONLY into offline BC (never at decision time), so this is the cheapest test of the variable-reduction premise. Touches only mutable eval-side glue; calls the immutable eval unchanged.",
        "proposed",
        "pending",
        ["solver", "wave0", "probe", "L1"],
        "G1",
    ),
    # ---- Wave 1 interventions ----
    (
        "l4-5-miner-composite-key",
        "L4-5 Miner key = delivery + queue-shortfall composite",
        "Extend the hard-example miner key from delivery-shortfall-ONLY to delivery + w_q*max(0,queue_ratio-1.1) (+ optional EE-margin) so it oversamples the ACTUAL binding constraint at 0/5 dB. Aligns the one winning mechanism (data shaping) to the TRUE 2-bar per-point gate. Rides s3 (per-eval-point fine-grained mining).",
        "Jury: ACCEPT-PILOT. Keep delivery DOMINANT for graceful degradation - queue-shortfall may be sparse (the 1.1x slack makes most states zero-shortfall), degenerating the key back to delivery-only. Clean: extends the WINNING family, not any failed family. Mutable: src/training/main.py miner only.",
        "proposed",
        "pending",
        ["data-shaping", "wave1", "miner", "queue", "L4"],
        "G1",
    ),
    (
        "l2-1-csir-domain-rand",
        "L2-1 CSI-error-scale domain randomization (low-SNR weighted)",
        "Per-episode randomize the CSI-estimation-error scale epsilon_csi (e.g. U[0.5x,2x] nominal), low-SNR episodes drawn from the upper half, so the policy MARGINALIZES over the estimation-error distribution instead of overfitting the nominal realization. Center the range on the audited epsilon (L2-4). Rides mining (s1-s3) unchanged.",
        "Jury: ACCEPT-PILOT. Best data-side CSIR robustness play; sequence AFTER L2-4 so randomization centers on the aligned epsilon. Orthogonal to B (randomizes an ENV INPUT PARAMETER every episode, no task-difficulty schedule), D/i18 (obs untouched), A (no dual change). Mutable env-reset plumbing + curriculum hook; channel.py physics never edited.",
        "proposed",
        "pending",
        ["csir", "wave1", "data-distribution", "L2"],
        "G1,G3",
    ),
    (
        "l4-1-bounded-dvq-drift",
        "L4-1 Bounded/normalized DVQ Lyapunov drift term",
        "Replace the DVQ drift's fixed linear scale y_weight=Y/dvq_y_scale(=300) with a bounded/normalized per-step form tanh(Y/y_ref) (or Y/running-shortfall-RMS) in lyapunov.py compute_reward. At low SNR Y pins at cap 5000 => y_weight~16.7 (wound up, no differential gradient); with NO return normalization the shared critic must fit a regime-specific value scale. Bounding restores the queue-pressure gradient + stabilizes value targets. High-SNR Y~0 so tanh~linear = unchanged (protects the 5).",
        "Jury: ACCEPT-PILOT but EVIDENCE-GATED - FAMILY-A ADJACENCY IS REAL (the bundle's 'NOT family A' was too glib; it IS a reward-function edit). What separates it from blind family-A tuning is a VERIFIED saturation pathology: gate on Y/y_weight logging - if Y isn't pinned at cap, this is family A in a tanh costume. Also verify y_ref leaves the 5/7.5/10 dB operating range uncompressed. Leaves the Y accumulator + lambda dual untouched.",
        "proposed",
        "pending",
        ["lyapunov", "wave1", "queue", "reward-structure", "L4"],
        "G1",
    ),
    (
        "l2-3-adversarial-error-mining",
        "L2-3 Adversarial CSI-error-realization mining",
        "Extend the CuRLTraC miner key from state-difficulty to adverse per-user error_variance realizations at 0/5 dB; store + oversample the channel-RNG seeds whose upper-tail error coincides with constraint violation. Teaches the policy to hedge exactly the adversarial CSIR configs. Decisive NEGATIVE if failure is uniform across error draws (falsifies CSIR-dominance for our pipeline).",
        "Jury: ACCEPT-PILOT. Cheapest CSIR training-side test on the winning family; gate on L5-5 showing a nonzero PPO-minus-SDMA difference; CAP the oversampling weight (worst-case error draws may be unlearnable and spill over via weight sharing). Literally a new mining key on s1/s2/s3 - inherits the only falsification-resistant family.",
        "proposed",
        "pending",
        ["csir", "wave1", "data-shaping", "miner", "L2"],
        "G1,G3",
    ),
    (
        "l3-4-regime-reweighted-mining",
        "L3-4 Learned regime-classifier reweights mining",
        "A tiny MLP regime classifier (input: physical regime descriptor = interference-vs-power + queue-pressure from existing obs; label: per-state delivery deficit, already in buffer.per_user_deficits/RCPO cost) outputs a soft posterior P(failing-regime|s) that reweights the CuRLTraC replay distribution. Generalizes the mining key from eval-point-id to a LEARNED physical regime. NO architecture change, NO obs change.",
        "Jury: ACCEPT-PILOT. Pure data-shaping (rides s1-s3), lowest-risk L3. Sequence after L4-5. VERIFY per-state deficit labels actually exist in the buffer. NOT D/i3 (no expert heads, no arch change - only the sampling weight changes), NOT E/i9-i11 (keys on a learned soft regime posterior, not eval-point id; reweights replay sampling, not advantage temperature).",
        "proposed",
        "pending",
        ["data-shaping", "wave1", "miner", "routing", "L3"],
        "G1",
    ),
    # ---- Wave 2 lower priority ----
    (
        "l4-4-intra-bucket-tilt",
        "L4-4 Miner intra-bucket low-edge density tilt",
        "Replace the miner's within-bucket UNIFORM SNR draw with a low-edge-tilted density (Beta(a<1)/triangular) for the 2 unmet buckets only, concentrating samples at/below the exact 0 and 5 dB eval points. Bucket weights/targets/entropy and all other buckets unchanged. Rides s1-s3.",
        "Jury: ACCEPT-ON-PAPER. Cheap density concentration. Tilt toward the EVAL SNRs +/-margin, not blindly to the bucket low edge (which can move mass AWAY from 0 dB depending on edge placement). Distinct from L4-3 (density within existing buckets vs widening support).",
        "proposed",
        "pending",
        ["data-shaping", "wave2", "miner", "L4"],
        "G1",
    ),
    (
        "l4-2-snr-normalized-penalty",
        "L4-2 SNR-normalized drift/penalty equalizer",
        "Consume the unused sigma2 argument (compute_reward receives it at rsma_env.py:851 but ignores it) to build a monotone n(sigma2) that equalizes the penalty-block RMS across SNR, making the reward magnitude (hence the value target) regime-invariant. n~1 at high SNR (protects the 5), n<1 at low SNR.",
        "Jury: ACCEPT-ON-PAPER. Regime-aware reward-scale surgery DOMINATED by L4-1 (regime-agnostic bound); follow-up only if cross-regime imbalance persists after L4-1. NOT family A (no lambda/Y/X dual update change), NOT F/i21 (Dinkelbach/EE term untouched), NOT C/i2 (scales a reward penalty, alpha untouched).",
        "proposed",
        "pending",
        ["lyapunov", "wave2", "reward-structure", "L4"],
        "G1",
    ),
    (
        "l4-3-miner-support-extension",
        "L4-3 Miner low-SNR support extension (-2.5 -> -5 dB)",
        "Extend _HM_EDGES[0] from -2.5 to -5.0 so the currently UNREACHABLE [-5,-2.5] dB extreme-low tail (vs snr_min_db=-5) becomes minable; weight the tail by the unmet low-SNR shortfall. Rides s1-s3.",
        "Jury: ACCEPT-ON-PAPER (HYGIENE). Correct fix for the B.3 support bug BUT [-5,-2.5] has ZERO eval exposure (eval points are 0,5,...) so NO measurable points_met effect is possible. Demote from candidate to hygiene - fold the edge fix into the next miner PR, do not spend an iteration on it alone.",
        "proposed",
        "pending",
        ["data-shaping", "wave2", "miner", "hygiene", "L4"],
        "G1",
    ),
    (
        "l3-3-delivery-pressure-router",
        "L3-3 Delivery-pressure soft router (EE-agg vs delivery-safe expert)",
        "A softmax router keyed on queue-backlog delivery-pressure (mean queue_norm + DVQ/fairness backlog from existing obs) mixes 2 experts CLONED from best: high backlog => delivery-safe expert (clears delivery>=1.0, queue<=1.1); low backlog => EE-aggressive expert. Encodes the lexicographic delivery-first priority as ARCHITECTURE, not dual pressure. Cloned init => step-0 == best (protects the 5).",
        "Jury: ACCEPT-ON-PAPER. Best L3 alignment to the corrected criterion, BUT its router key (DVQ backlog) SATURATES exactly at low SNR (B.5) - sequence AFTER L4-1 so the key isn't saturated. NOT family A (does not touch the Lagrangian/PID/CRPO or advantage), NOT D/i3 (SOFT softmax mixture over cloned experts keyed on backlog, no SNR gather, no frozen trunk). Construct unconditionally + buffer-gate so the immutable eval's strict=False loader picks it up.",
        "proposed",
        "pending",
        ["routing", "wave2", "architecture", "queue", "L3"],
        "G1",
    ),
    (
        "l3-1-soft-moe-router",
        "L3-1 Soft-MoE interference-vs-power router",
        "A softmax router over G=2 expert heads CLONED from best, keyed on a dimensionless interference-vs-power regime scalar r=mean(interference share)/mean(signal-power proxy) from existing per-user obs. expert-1 dominates as r->high (interference-limited: near-SDMA, robust, low-power), expert-0 as r->low. Cloned init => uniform mixture == best at step-0.",
        "Jury: ACCEPT-ON-PAPER. Cloned-init protection is real BUT the core bet is family-D's LOST bet (regime-specialized heads); cloned init protects step-0, NOT the gradient dynamics. Needs diagnostic evidence of regime collapse first. NOT D/i3 (SOFTMAX mixture, no SNR gather, no frozen trunk, all experts get grad), NOT D/i18 (no bucket-context vector added to obs).",
        "proposed",
        "pending",
        ["routing", "wave2", "architecture", "L3"],
        "G1",
    ),
    (
        "l1-1-solver-power-elimination",
        "L1-1 Runtime solver power-elimination -> RL owns per-user alpha",
        "At decision time overwrite the Worker's power block with SolverExpert's per-slot gradient solve (conditioned on the current split), set group_level_alpha=False so the Worker's alpha block goes live per-user, and mask the dead power block in the PPO log_prob. Power is ELIMINATED (variable-reduction, per arXiv 2311.16895), not blended. RL's gradient then concentrates on alpha (which controls the CSIR-robust common/private tradeoff that dominates at low SNR).",
        "Jury: NEEDS-DIAGNOSTIC-FIRST. BUNDLES TWO CONFOUNDED CHANGES (solver power + group_level_alpha=False reversing the current best config) => a failure would be unattributable. Zero evidence solver power beats learned power ON DELIVERY at 0/5 dB. Run the L1-gate probe first; split the two changes. NOT F/i3 (hard replace, not interpolate), NOT C (no global/tabled alpha), NOT the offline BC (removes power from RL at decision time rather than teaching RL to imitate it).",
        "proposed",
        "pending",
        ["solver", "wave2", "architecture", "L1"],
        "G1",
    ),
    (
        "l1-4-triggered-reduction-onset",
        "L1-4 Performance-triggered ONSET of power reduction",
        "Let RL own power during early exploration; at a low-SNR delivery-feasibility trigger switch power to solver-fixed for the rest of training so alpha fine-tunes against a stable power manifold. Rides s0 (extends performance-triggered difficulty-annealing from the env-difficulty axis to the action-space-reduction axis).",
        "Jury: NEEDS-DIAGNOSTIC-FIRST. Inherits L1-1's unvalidated premise PLUS a switching discontinuity. The s0-style trigger instinct is RIGHT - keep the sketch, gate on the L1-gate probe + L5-3. NOT banlist B (i24/i28/i7 annealed ENV/SIGNAL difficulty; this schedules an ACTION-SPACE reduction onset - a different object being timed).",
        "proposed",
        "pending",
        ["solver", "wave2", "curriculum", "L1"],
        "G1",
    ),
    # ---- REJECTED (desk-reject; banlisted from re-proposal, honestly labelled) ----
    (
        "l1-2-waterfilling-law",
        "L1-2 Closed-form water-filling power law (REJECTED)",
        "Derive a closed-form/KKT water-filling power law for the low-SNR EE-DPP surrogate and use it as a zero-Adam reduction operator so solver-reduction is CPU-viable for FULL runs.",
        "Jury: REJECT (desk-reject, NOT experimentally tested). Weeks of theory aimed at the 4TH lexicographic slot (EE); no points_met path; contingent on L1-1 succeeding AND CPU being the actual blocker. Do not re-propose unless the corrected criterion changes.",
        "archived",
        "negative",
        ["rejected", "desk-reject", "solver", "L1"],
        "G2",
    ),
    (
        "l1-3-power-budget-residual",
        "L1-3 Eliminate power distribution -> RL owns total budget (REJECTED)",
        "A per-slot water-filling solve redistributes an RL-chosen total-power budget, collapsing the Worker's power block to a 1-dim budget residual while alpha stays Meta-owned.",
        "Jury: REJECT (desk-reject, NOT experimentally tested). Self-declared 4th-slot (power/EE) play; per-slot redistribution under the uplink per-user power structure is theoretically dubious for DELIVERY. Do not re-propose for points_met.",
        "archived",
        "negative",
        ["rejected", "desk-reject", "solver", "L1"],
        "G2",
    ),
    (
        "l3-2-csir-modulation-router",
        "L3-2 CSIR-confidence output-modulation router (REJECTED)",
        "A softmax router keyed on realized CSIR-error confidence (eps_norm + per-user error_variance) mixes M=2 output-modulation vectors (gamma,beta) calibrating a shared head's outputs.",
        "Jury: REJECT (desk-reject, NOT experimentally tested). Architecture-side CSIR hedge DOMINATED by the data-side L2-4/L2-1 pair at a fraction of the risk; routing complexity on a shared critic with no return-norm. Prefer L2-4/L2-1.",
        "archived",
        "negative",
        ["rejected", "desk-reject", "csir", "routing", "L3"],
        "G1",
    ),
    (
        "l2-2-cvar-error-annealing",
        "L2-2 CVaR-gated error-scale annealing (REJECTED)",
        "Inflate the training CSI-error scale only when CVaR_0.2 of low-SNR delivery across error realizations clears the SDMA bar, else retract (a distributionally-robust curriculum trigger).",
        "Jury: REJECT (desk-reject, NOT experimentally tested). A noisy CVaR trigger controller stacked on an UNVALIDATED axis; dominated by L2-1 (plain randomization). Revisit only if L2-1 succeeds and tail-specific annealing is still needed.",
        "archived",
        "negative",
        ["rejected", "desk-reject", "csir", "curriculum", "L2"],
        "G1",
    ),
]

n_prop = n_rej = 0
for slug, title, thesis, risks, stage, outcome, tags, gaps in IDEAS:
    # gaps are NOT first-class nodes here (they live only in gap_map.md), so fold
    # the ids into the description instead of target_gaps= (which iterates a
    # string char-by-char -> dangling edges gap:G, gap:1, gap:,).
    rw.upsert_idea(
        str(WIKI),
        slug,
        title,
        description=f"idea-creator 2026-09 campaign ({'jury ACCEPT' if outcome == 'pending' else 'jury REJECT'}). Addresses {gaps}. Targets points_met 5->7 at the 2 low-SNR points.",
        stage=stage,
        outcome=outcome,
        thesis=thesis,
        risks=risks,
        tags=tags,
        update_on_exist=True,
    )
    if outcome == "pending":
        n_prop += 1
    else:
        n_rej += 1

print(f"WROTE {n_prop + n_rej} idea nodes: {n_prop} proposed/pending, {n_rej} archived/negative (desk-reject).")
