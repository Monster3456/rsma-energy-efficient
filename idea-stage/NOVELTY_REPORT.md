# Novelty Check Report

**Subject:** idea-creator 2026-09 top bet — L2-4 (train/eval CSIR-distribution alignment audit) + L2-1 (low-SNR-weighted ε-domain-randomization), the paired Wave-0/Wave-1 diagnosis+fix.
**Date:** 2026-09-03 · **Executor:** Qoder CLI · **Cross-model reviewer:** qodercli `-m GLM-5.3` (Zhipu/GLM family — distinct from executor).
**Dossier:** `idea-stage/NOVELTY_DOSSIER.md` · **Trace:** `.aris/traces/novelty-check/2026-09-03_run01/` · **Prior-work verification:** `idea-stage/verified_papers.json` (VERDICT PASS, hallucination_rate 0.0).

---

### Proposed Method

Diagnose the 2 failing low-SNR points {0,5} dB of a constrained uplink-RSMA MU-MIMO energy-efficiency RL agent (hierarchical PPO + Lyapunov DPP) as a **train/eval channel-estimation-error distribution mismatch**: the policy trains at a fixed error scale ε but is evaluated under an SNR-dependent σ_est that grows ~17× toward low SNR, putting it out-of-distribution exactly at the failing points. Fix by (1) **aligning** the training CSI-error distribution to eval-time low-SNR statistics and (2) **low-SNR-weighted domain randomization** of ε.

### Core Claims

1. **C1 (diagnosis):** fixed-ε-train vs SNR-dependent-σ_est-eval mismatch is a *named root cause* of RL failure specifically at low SNR in uplink RSMA / wireless EE. — **Closest:** P2 (Demirel 2026, DR for RAN generalization) + P5 (2501.19019, uplink-RSMA low-SNR CSIR-dominance fact). — **Delta:** no prior work frames the *train-vs-eval CSIR-error-distribution gap* as the low-SNR failure cause; P2 attributes mismatch to topology/traffic, P5 is non-RL analysis. **Reviewer caveat:** novel as an *instance*, but an instance of the well-worn "train at a fixed operating point → randomize it" meta-pattern; the causal finding is currently correlation, not proof.
2. **C2 (alignment fix):** matching the training CSI-error distribution to eval-time low-SNR statistics recovers the low-SNR points. — **Closest:** P2 / textbook matched training. — **Delta:** application to this constrained-RSMA setting; untested. Verifiable by the aligned-ε retrain ablation.
3. **C3 (robustness fix):** *SNR-weighted* ε-domain-randomization yields a CSIR-robust low-SNR policy. — **Closest:** P2 (DR), P7 (2412.14297, distributionally-robust policy learning). — **Delta:** the SNR-weighting. **Reviewer caveat (binding):** "SNR-weighted" is *decoration* until it beats plain **uniform** ε-randomization in an ablation — run that or drop the qualifier.
4. **C4 (setting):** done under constrained lexicographic delivery/queue RL for uplink-RSMA MU-MIMO EE. — **Closest:** P6 (Lyapunov-DPP-for-RL), P5 (uplink RSMA). — **Delta:** exact stack (constrained + RSMA + uplink + EE + hierarchical PPO + DPP) not found in one paper; **but a setting alone is the weakest form of novelty** and carries the least weight.

### Closest Prior Work

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| **P2** Generalization in RL for RAN (2507.06602, Demirel) | 2026 | arXiv | Domain randomization for wireless-RL generalization | Mismatch = topology/traffic, **not CSIR-error at low SNR**; downlink link adaptation; no RSMA/uplink-EE/queue constraint |
| **P5** Uplink RSMA with Imperfect CSI (2501.19019) | 2025 | arXiv | Supplies the exact low-SNR CSIR-dominance fact the diagnosis leans on | **Non-RL** closed-form analysis; no training fix, no alignment/DR, no constrained-EE agent |
| **P1** Robust Design of RSMA with Imperfect CSI (2203.03690, Yu) | 2022 | arXiv | RSMA + imperfect CSI + robustness | **Conventional robust optimization, not RL**; downlink; worst-case rate bounds, no train/eval mismatch, no DR |
| P3 Model-Free Robust Beamforming, Satellite DL (2510.25393, Schröder) | 2025 | arXiv | Robust RL under imperfect CSI | AoD/phase errors, no SNR-dependent error link; satellite downlink; not RSMA/uplink/EE |
| P4 Robust Generalization for Channel Estimation (2302.02302, Luan) | 2023 | arXiv | "Robust generalization" + CSI | About the **estimator net**, not an RL control policy; no decision-loop σ_est mismatch |
| P6 Lyapunov DPP tailored for RL (2506.04291) | 2025 | arXiv | Constrained-RL + Lyapunov-DPP + queue | Drift-term form for value stability; no CSI-error robustness |
| P7 Distributionally Robust Policy Learning (2412.14297) | 2024 | arXiv | "Distributionally robust policy" framing of C3 | Generic RL theory, not wireless; no CSI/σ_est/SNR link |

*Methodological neighbors (not wireless-CSI):* Distributionally Adaptive Meta-RL (NeurIPS 2022); robust DL power control trained under randomized channel-estimation error (ICASSP 2020); SNR-range training of learned physical-layer autoencoders/receivers. None ties robustness to a **train-vs-eval CSIR-error-distribution gap at low SNR in a constrained RL loop**.

### Overall Novelty Assessment

- **Score: 5/10** — clear neighbors (P2, P5, P1) with a defensible delta worth a pilot, but the delta is *correlation not yet proof*. (Anchor: 5 = clear neighbors + defensible delta; 1–3 reserved for results a named paper already contains — not the case here.)
- **Recommendation: PROCEED WITH CAUTION.** No single published paper contains this result, so ABANDON is unavailable (verdict-limit #2). CAUTION is *named*, not a safe middle: (1) the fix is standard DR/matched-training applied to a new system; (2) the causal claim is untested with two live confounds (intrinsic low-SNR hardness; error floors at σ_est≈0.2); (3) C3's "SNR-weighted" is unproven against uniform DR.
- **Key differentiator:** the **diagnostic finding** — that a fixed-ε-train vs SNR-dependent-σ_est-eval gap is the *shared* root cause of low-SNR failure across 36 prior interventions in 6 mechanism families — not the generic DR/alignment mechanism. Novelty lives in the finding, and the finding must be *shown*.
- **Risk (what a reviewer cites as prior work):** P2 (DR for wireless-RL generalization) + P5 (low-SNR CSIR dominance in uplink RSMA) + the robust-DL-power-control practice of training under randomized CSI error. Rebuttal: none performs *this diagnosis in this constrained setting*, and P5 actually strengthens the puzzle (at low SNR RSMA→SDMA, so `delivery ≥ SDMA×1.0` should be *easier* — the failure genuinely needs an explanation).

### Cross-Model Verdict Provenance

GLM-5.3 (Zhipu) reviewer, distinct family from the Qoder executor, pure-analysis (`--tools ""`, no file-modification capability). Provenance is **asserted-by-invocation, NOT self-attested** — the responder explicitly declined to adopt the "GLM-5.3" identity, so the shim header (not the responder) stamps the model pin. This is the same trust level ARIS gives a native codex `model:` pin; `independence_verified: unverified` is recorded honestly in the trace. Full text: `.aris/traces/novelty-check/2026-09-03_run01/001-novelty-verdict.response.md`.

### Pre-Registered Causality Gate (reviewer-prescribed — run BEFORE any retraining)

The CAUTION upgrades to **PROCEED** on the strength of the finding iff these land. These are cheap (eval-only) and fold directly into the L2-4 Wave-0 audit:

1. **Swap-A (error-scale sufficiency):** evaluate the *existing fixed-ε policy* at **30 dB with σ_est artificially inflated to ≈0.2**. → If delivery **collapses**, error scale (not SNR per se) is causal.
2. **Swap-B (error-scale necessity):** evaluate the *same policy* at **0 dB with near-perfect CSI**. → If it **passes**, the CSIR error is the cause.
3. **Per-point σ_est table:** report σ_est at **10–15 dB** too — with a smooth spread, "OOD exactly at {0,5} and nowhere else" is approximate and 10 dB likely sees mild mismatch.
4. **Aligned-ε retrain (confirmation):** retrain with training ε matched to eval low-SNR variance; the {0,5} dB points must recover **where a uniform-DR control recovers less** (this is the C2 result *and* the C3 SNR-weighting ablation).
5. **Non-degeneracy baseline:** include a low-SNR SDMA (or P1-style robust-optimization) baseline to show the recovered policy isn't merely degenerating to SDMA behavior.

**Falsification branch:** if the aligned-ε retrain fails to recover {0,5} dB, the root-cause claim is falsified → **re-scope** to feasibility / error-floor investigation (is delivery capped for *any* policy at σ_est≈0.2?), **not** abandonment.

### Suggested Positioning

> No published work identifies the gap between fixed-error training and SNR-dependent eval-time CSIR-error statistics as the cause of a constrained RL agent's low-SNR failure in uplink RSMA, and closes it with matched or error-randomized training.

State the delta as the **finding**, verified by the swap ablations — not as "we applied domain randomization to RSMA" (that framing is dominated by P2 and earns nothing). Lead with the diagnosis and its two-experiment causal proof; position DR/alignment as the *consequence* of the finding, and drop "SNR-weighted" from C3 unless ablation #4 shows it beats uniform DR.
