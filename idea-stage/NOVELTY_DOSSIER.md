# NOVELTY DOSSIER — cross-model novelty verdict request

You are a **cross-model research reviewer** (your model family is DISTINCT from the author/executor).
This is **pure analysis** — you have no file-modification tools. Judge the novelty of the proposed
method below against the verified prior work. Begin your reply with `Reviewer-Model: GLM-5.3`.
Judge under the **NOVELTY VERDICT LIMITS** block at the end (copied verbatim); search exhaustively,
judge calibrated.

---

## 1. Proposed method (the idea under test)

**Setting:** uplink RSMA MU-MIMO, energy-efficiency objective, **constrained** deep RL (hierarchical
PPO: SA-GAT Worker + Meta-Controller; Lyapunov drift-plus-penalty; imperfect CSI). K=24, N_r=16,
RZF, MMSE-SIC. Evaluated at 7 SNR points {0,5,10,15,20,25,30} dB. Success = per-SNR-point
`delivery ≥ SDMA×1.0 AND queue ≤ SDMA×1.1` (lexicographic `points_met`); currently 5/7, the 2
**low-SNR** points {0,5} dB fail.

**Core method claim:** The RL policy is **trained at a FIXED channel-estimation-error scale ε**, but
**evaluated under an SNR-DEPENDENT σ_est** (estimation-error variance that GROWS as SNR drops —
measured ≈0.2045 @ 0 dB vs ≈0.0119 @ 30 dB, a ~17× spread). This is a **train/eval distribution
mismatch in the CSIR error** that makes the policy **out-of-distribution exactly at the low-SNR
points and nowhere else** — proposed as the ROOT CAUSE of why the 2 low-SNR points fail (and, by
extension, why 36 prior interventions across 6 mechanism families all failed at those same 2 points).

**Two-part fix:**
1. **Distribution ALIGNMENT:** match the training CSI-error model to the eval-time low-SNR error
   statistics (flip a `snr_dependent_csi` flag / widen the training ε range to cover eval variance),
   then retrain.
2. **Low-SNR-weighted DOMAIN RANDOMIZATION of ε:** per-episode randomize the estimation-error scale
   (e.g. U[0.5×, 2×] nominal), oversampling the upper tail at the low-SNR points, so the policy
   becomes **distributionally robust** to CSIR error where it binds.

## 2. Core claims to check (each vs the literature)

- **C1 (diagnosis):** train/eval mismatch of the *channel-estimation-error distribution* (fixed-ε
  train vs SNR-dependent σ_est eval) is a named root cause of RL failure *specifically at low SNR*,
  in uplink RSMA / wireless EE.
- **C2 (alignment fix):** aligning the training CSI-error distribution to eval-time low-SNR
  statistics recovers the low-SNR points.
- **C3 (robustness fix):** SNR-weighted domain randomization of the estimation-error scale yields a
  CSIR-robust low-SNR policy.
- **C4 (setting):** done under *constrained* RL (per-point delivery+queue, lexicographic) for
  *uplink RSMA MU-MIMO energy efficiency* with hierarchical PPO + Lyapunov DPP.

## 3. Verified prior work (all arXiv IDs passed verify_papers.py: VERDICT PASS, hallucination_rate 0.0)

| # | Paper (arXiv) | Year | What it does | Overlap with our idea | Key difference (our delta) |
|---|---------------|------|--------------|-----------------------|----------------------------|
| P1 | **Robust Design of RSMA with Imperfect CSI** (2203.03690, Yu) | 2022 | **Conventional robust optimization** (closed-form rate bounds) for **DOWNLINK** RSMA under erroneous CSI | RSMA + imperfect CSI + robustness | **Not RL**; downlink not uplink; worst-case rate bounds, **no train/eval distribution mismatch**, **no domain randomization**, no queue/delivery constraint |
| P2 | **Generalization in RL for Radio Access Networks** (2507.06602, Demirel) | 2026 | Studies RL generalization across topology/traffic/mobility/channel; **explicitly uses domain randomization** ("introduces variability into training environments") | DR for wireless-RL generalization | Mismatch attributed to **diverse environments, NOT CSI-error at low SNR**; **downlink link adaptation**; **no RSMA, no uplink EE, no queue/delivery constraint**; does not diagnose a fixed-ε-vs-σ_est train/eval gap |
| P3 | **Model-Free Robust Beamforming in Satellite Downlink using RL** (2510.25393, Schröder) | 2025 | RL (SAC) robust beamforming, LEO **satellite downlink**, imperfect CSIT | robust RL under imperfect CSI | Models **AoD/phase errors, no SNR-dependent error link**; **no train/eval distribution-mismatch diagnosis**; not RSMA/uplink/EE/constrained |
| P4 | **Achieving Robust Generalization for Wireless Channel Estimation** (2302.02302, Luan) | 2023 | Robust generalization of a **channel-estimation MODEL** (a learned estimator) across conditions | "robust generalization" + CSI + wireless | About the **estimator network**, NOT an RL control policy; no RSMA, no EE, no queue constraint, no train/eval σ_est mismatch in a decision-making loop |
| P5 | **Uplink RSMA with Imperfect CSI** (2501.19019) | 2025 | Analysis of **uplink** RSMA under imperfect CSI; finds CSIR error dominates SIC error **at low SNR**; small common fraction optimal | Uplink RSMA + imperfect CSI + the low-SNR CSIR-dominance FACT we build on | **Not RL** (closed-form outage/throughput analysis); does not propose a training fix, no distribution alignment/DR, no constrained-EE agent |
| P6 | **A Lyapunov Drift-Plus-Penalty Method Tailored for RL** (2506.04291) | 2025 | RL-tailored DPP drift term for queue stability | Our constrained-RL + Lyapunov-DPP + queue setting | About the **drift-term form for value stability**, not CSI-error distribution mismatch; no RSMA/imperfect-CSI robustness |
| P7 | **Distributionally Robust Policy Learning under Concept Drifts** (2412.14297) | 2024 | Generic **distributionally-robust RL** under concept drift | The "distributionally robust policy" framing of C3 | **Generic RL theory, not wireless**; no CSI/σ_est, no RSMA, no SNR-dependent error, no sim channel model |

**Also surfaced (methodological neighbors, not wireless-CSI):** Distributionally Adaptive Meta-RL
(NeurIPS 2022), uncertainty-aware offline-to-online RL, robust DL power control (ICASSP 2020). None
ties the robustness to a **train-vs-eval channel-estimation-error distribution gap at low SNR**.

## 4. Questions for you (the reviewer)

1. **Is this method novel?** Specifically the *combination*: (a) diagnosing a fixed-ε-train vs
   SNR-dependent-σ_est-eval **distribution mismatch** as the root cause of **low-SNR** RL failure,
   (b) fixing it by **alignment + low-SNR-weighted ε-domain-randomization**, (c) in **constrained
   uplink RSMA MU-MIMO energy efficiency**.
2. **What is the closest prior work?** Name it (P1-P7 above, or something we missed).
3. **What is the delta, in one sentence a reviewer could verify?**
4. Is the novelty in the **mechanism** (DR/alignment are generic) or in the **finding** (the
   specific train/eval σ_est mismatch as the low-SNR root cause)? Judge the idea, not each claim in
   isolation — known parts arranged to reveal something unknown are novel.
5. Does any **single named published paper already contain this result** (the only qualification for
   ABANDON)? If not, the verdict cannot be ABANDON.

## 5. Our self-assessment (for you to challenge, not accept)

We read the delta as: **domain randomization and robust-RSMA-under-imperfect-CSI both exist
separately (P2, P1), and the low-SNR CSIR-dominance FACT is known (P5), but nobody has (i) framed
the train/eval CSI-error-distribution MISMATCH as the root cause of low-SNR RL failure, nor (ii)
targeted it with SNR-weighted ε-randomization/alignment, nor (iii) done so in constrained uplink-RSMA
EE.** The novel part is likely the **diagnostic finding + its targeted fix in this setting**, not the
generic DR mechanism. We expect **PROCEED** or **PROCEED WITH CAUTION**, not ABANDON. Challenge this.

---

```
=== NOVELTY VERDICT LIMITS (these bound how you judge, never how widely you search) ===
Search exhaustively; judge calibrated. Two failures waste months equally:
passing an idea a published paper already contains, and killing a viable idea
because the territory has neighbors.
1. Proximity is information, not a verdict. Someone working nearby goes in the
   report; it is not by itself a reason to reject.
2. ABANDON has exactly one qualification: a specific published paper already
   contains this result — name that paper. No named paper, no ABANDON.
3. Crowded-but-deltaed is PROCEED: state the delta in one sentence a reviewer
   could verify. Thin or contested delta is PROCEED WITH CAUTION — say what
   would make it carry, not why it should die. CAUTION is not a safe middle:
   if you cannot name the specific thing that makes the delta thin, the
   verdict is PROCEED.
4. Concurrent or competing work is not a veto. That is a race — report it and
   let the user decide whether to run it.
5. A direct attack on a central problem is legitimate novelty when nobody has
   executed it well. "This area is hot" does not mean "this area is taken."
6. This check is an early gate, never the last one — more triage, pilots, or
   external review still stand between any idea and a paper, whatever order
   this run uses. A wrongly passed idea dies cheaply at one of them; a wrongly
   killed idea is never seen again. When torn between two verdicts, choose the
   more permissive one.
Say plainly when an idea clears the check. Do not manufacture overlap.
```
