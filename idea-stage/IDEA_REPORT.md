# IDEA REPORT — RSMA uplink: lift `points_met` 5/7 → 7/7 (the 2 low-SNR points)

_Run: `/idea-creator "RSMA 上行 EE：把 points_met 从 5/7 推到 7/7（2 个低 SNR 点）"` · 2026-09-03._
_Pipeline: Phase-0 banlist load (36 falsified) → Phase-1 landscape (WebSearch + 2 paper fetches) →
Phase-1.5/2 five-lens Agent fan-out (22 candidates: L1×4, L2×4, L3×4, L4×5, L5×5) → Phase-3 consolidation → **Phase-4 cross-model
jury (GLM-5.3, Zhipu family — DISTINCT from the Qoder executor)** → this report. Phase-5 pilots
SKIPPED (user runs experiments per project scope)._

---

## ⚠️ HEADLINE: the objective was mis-framed — corrected from source

The project brief (and this wiki's seed) stated each SNR point is "met" only if it clears **four**
bars simultaneously (delivery, queue, EE>1.05, power<SDMA). **Direct code inspection falsifies this.**
`compute_score` (`scripts/optimize_loop.py:1759-1785`, lines 1777-1778) gates each point on **two**
bars only:

```
point is "met"  ⟺  ppo.delivery ≥ sdma.delivery × 1.0   AND   ppo.queue ≤ sdma.queue × 1.1
score = [points_met, avg_delivery, −avg_queue, ee_ratio]   # lexicographic; points_met dominates
```

- Comparisons are **≥ / ≤** — *matching SDMA exactly already passes.* The required lift is the
  relative gap `(SDMA − PPO)/SDMA` at {0,5} dB, which may be only a few percent.
- **EE and power are NOT per-point gates.** `ee_ratio` is the **4th (least-significant) lexicographic
  slot** and a global aggregate; EE>1.05 / power appear only in the **global** `check_termination`
  (`optimize_loop.py:475-497`).

**Reframe:** lifting `points_met` 5→7 is a **relative per-point delivery + queue race vs SDMA at
{0,5} dB** — not "fix low-SNR energy efficiency." (EE still matters as a *termination* constraint: a
delivery play that wrecks global EE/power dies at `check_termination`.) RESEARCH_BRIEF.md,
gap_map.md and the query_pack have been corrected accordingly.

## Verified code facts the ideas stand on (from sub-agent source inspection)

1. **Action = 4K=96 = [alpha, p, d, order].** With `group_level_alpha=True` (current best) the
   Worker's `alpha` block is a **dead dim** (alpha = Meta `split_mask`).
2. **`SolverExpert` is wired ONLY into offline BC** (`_solver_expert_collect`→`bc_pretrain`); never
   used at decision time. `blend_alpha`/`_inject_alpha_guidance` merely interpolates Meta alpha.
3. **Miner support hole:** `_hm_sample_snr` uses `_HM_EDGES[0] = −2.5` while `snr_min_db = −5` →
   the **[−5, −2.5] dB tail is unreachable during mining**. Miner key = **delivery-shortfall only**
   (ignores the queue bar — i.e. it mines the wrong states if queue binds).
4. **No return/reward normalization** (no PopArt/RunningMeanStd); single shared critic, `value_loss = MSE`.
5. **DVQ drift saturates at low SNR:** `y_component = w_y_tp·(Y/300)·D_total`, `Y` pinned at cap 5000
   → `y_weight≈16.7` (wound up, no differential gradient). `compute_reward` **receives `sigma2` but
   never uses it**.
6. **Offered load** `= clip(snr_adaptive_load_ratio(snr)·cap/K, λ_min, λ_max)`; CURRENT constants
   `λ_min=0.2, λ_max=10, load_ratio_min=0.75, load_ratio_max=1.3` (config.py:89-98), **hand-tuned
   across Task#27/43/46/47/48 to reduce low-SNR overload.** The exact overload multiple at 0/5 dB is
   therefore an **open, measurable question** (an earlier draft's "7× overload" used stale λ_min=1.5).
   Any idea that merely re-tunes λ/load collides with prior manual tuning and drifts the benchmark.

---

## Ranked ideas (cross-model jury verdicts)

Jury legend: `ACCEPT-PILOT` = worth a CPU experiment · `ACCEPT-ON-PAPER` = sound, lower priority ·
`NEEDS-DIAGNOSTIC-FIRST` = gate on evidence · `REJECT` = drop.

### Wave 0 — DIAGNOSTICS FIRST (no training, CPU-minutes). These price every other idea.
| ID | Idea | Verdict | Why it gates downstream |
|----|------|---------|--------------------------|
| **L5-1** | Bar-slack decomposition: which bar binds at 0/5 dB + relative gap size + gate noise floor | **ACCEPT-PILOT** | Prices the effect size. A 3% gap ⇒ cheap nudges suffice; a 25% gap ⇒ architecture. Also pin delivery/queue aggregation semantics. |
| **L5-2** | Transient-vs-steady-state delivery trajectory (matched seeds, PPO vs SDMA) | **ACCEPT-PILOT** | Fork between "warm-up/measurement artifact" and "genuine steady-state deficit" → curriculum plays vs service-rate plays. |
| **L2-4** | **Train/eval CSIR-distribution alignment audit** (is training ε fixed while eval σ_est grows at low SNR?) | **ACCEPT-PILOT (elevate to Wave 0)** | The jury's strongest bet. A config/log audit (hours), NOT gated by L5-5. If misaligned → one-flag `snr_dependent_csi` fix. |
| **L5-3** | Offered-load λ* crossing at 0/5 dB (current constants) | **ACCEPT-PILOT** | Distinguishes load-calibration artifact from genuine deficit. **Diagnostic only — never commit a λ change (B.6).** |
| **L5-4** | Per-user deep-fade deficit concentration + wasted capacity on empty-queue users | **ACCEPT-PILOT** | If the gap concentrates in a deep-fade cohort, it names the mining target for L4-5/L3-4. |
| **L5-5** | CSIR min-branch rate-loss **relative** attribution (PPO−SDMA), external dual-call harness | **ACCEPT-PILOT (2nd wave)** | Gates L2-1/L2-3. Near-zero difference DEMOTES the whole CSIR axis. **Must be an external harness, never rate_calc instrumentation.** |
| — | **L1-gate probe** (jury-added): eval-time overwrite of the power block with SolverExpert on an existing checkpoint at 0/5/10 dB | **ACCEPT-PILOT** | Gates the weeks-scale L1 cluster for the cost of one eval run. |

### Wave 1 — INTERVENTIONS (one variable at a time; guardrail = the 5 passing points must not regress)
| ID | Idea | Verdict | Note |
|----|------|---------|------|
| **L4-5** | Miner key = delivery **+ queue-shortfall** composite (align to the TRUE 2-bar gate) | **ACCEPT-PILOT** | Rides the only winning family (data shaping). Keep delivery dominant (queue-shortfall may be sparse under the 1.1× slack). |
| **L2-4-fix** | Apply `snr_dependent_csi` alignment (if audit confirms) | **ACCEPT-PILOT** | One-flag, benchmark-preserving. Verify ε-plumbing is NOT inside the immutable physics call segment. |
| **L2-1** | Per-episode CSI-error-scale domain randomization, low-SNR-weighted | **ACCEPT-PILOT** | Center the randomization on the audited ε. Rides mining (s1-s3). |
| **L4-1** | Bounded/normalized DVQ drift `tanh(Y/y_ref)` (de-saturate y_weight at low SNR) | **ACCEPT-PILOT (evidence-gated)** | Targets the QUEUE bar + stabilizes the shared critic. **Family-A adjacency is real** — gate on verified Y-saturation logging or it is "family A in a tanh costume." Check y_ref doesn't compress the 5/7.5/10 dB range. |
| **L2-3** | Adversarial CSI-error-realization mining (new key on CuRLTraC) | **ACCEPT-PILOT** | Cheapest CSIR training test; decisive negative value. Gate on L5-5 nonzero difference; cap oversampling weight. |
| **L3-4** | Learned regime-classifier reweights the mining distribution | **ACCEPT-PILOT** | Pure data-shaping, no arch/obs change. Generalizes the mining key from eval-point-id → learned physical regime. Verify per-state deficit labels exist in the buffer. |

### Wave 2 — LOWER PRIORITY / architecture-adjacent
| ID | Idea | Verdict | Note |
|----|------|---------|------|
| **L4-4** | Miner intra-bucket low-edge density tilt | **ACCEPT-ON-PAPER** | Tilt toward the eval SNRs ±margin, not blindly to the bucket low edge. |
| **L4-2** | SNR-normalized drift/penalty equalizer (consume unused `sigma2`) | **ACCEPT-ON-PAPER** | Dominated by L4-1; follow-up if cross-regime imbalance persists. |
| **L4-3** | Extend miner support −2.5→−5 dB | **ACCEPT-ON-PAPER (hygiene)** | Correct fix for B.3 but **[−5,−2.5] has zero eval exposure ⇒ no measurable points_met effect**. Fold into the next miner PR. |
| **L3-3** | Delivery-pressure soft router (EE-aggressive vs delivery-safe expert) | **ACCEPT-ON-PAPER** | Best L3 alignment to the corrected criterion, but its router key (DVQ backlog) saturates at low SNR (B.5) — sequence AFTER L4-1. |
| **L3-1** | Soft-MoE interference-vs-power router over cloned experts | **ACCEPT-ON-PAPER** | Cloned-init protects step-0 but the core bet is family-D's lost bet; needs diagnostic evidence of regime collapse first. |
| **L1-1** | Runtime solver power-elimination → RL owns per-user alpha | **NEEDS-DIAGNOSTIC-FIRST** | **Bundles two confounded changes** (solver power + `group_level_alpha=False` reversing best config) → unattributable failure. Run the L1-gate probe first. |
| **L1-4** | Performance-triggered ONSET of power reduction | **NEEDS-DIAGNOSTIC-FIRST** | Inherits L1-1's unvalidated premise; the s0-style trigger instinct is right. Gate on the probe + L5-3. |

### REJECTED (desk-reject by the cross-model jury — recorded to the banlist so they are not re-proposed)
| ID | Idea | Why rejected |
|----|------|--------------|
| **L1-2** | Closed-form water-filling power law | Weeks of theory aimed at the **4th slot**; no `points_met` path; contingent on L1-1 AND CPU being the blocker. |
| **L1-3** | Eliminate power distribution → RL owns total-power budget | Self-declared **4th-slot** play; per-slot redistribution theoretically dubious for uplink delivery. |
| **L3-2** | CSIR-confidence output-modulation router | Architecture-side CSIR hedge **dominated by the data-side L2-4/L2-1** at a fraction of the risk; routing complexity on a shared critic with no return-norm. |
| **L2-2** | CVaR-gated error-scale annealing | Noisy CVaR trigger controller stacked on an **unvalidated axis**; dominated by L2-1. |

---

## Sequencing plan (jury-recommended, with pre-registered decision rules)

**Wave 0 (no training, run in parallel):** L5-1 · L5-3 · L2-4-audit · L5-2 · L1-gate-probe; then a
second pass: L5-4 · L5-5 · Y/`y_weight` saturation logging.

**Pre-register these decision rules NOW (else diagnostics become procrastination):**
- **L5-1 gap < eval noise floor** → cheap nudges only (L4-4/L4-5); skip architecture.
- **L2-4 misaligned** → apply `snr_dependent_csi` flag, re-eval, then L2-1.
- **L5-3: operating λ above PPO's knee AND SDMA service higher** → service-rate race → L2-3/L2-1/L4-1.
- **L5-3: both below knee yet PPO loses** → transient artifact → s3-family entropy/curriculum + L4-4.
- **L1-gate probe shows ≥ a few-% delivery gain** → L1-1/L1-4 graduate; else the L1 cluster dies.
- **Y not pinned at cap** → L4-1 dies (it was family A in disguise).
- **L5-5 PPO−SDMA CSIR-loss difference ≈ 0** → demote L2-1/L2-3; hand the bet to L4-5 + L4-1.

**Wave 1 (one variable at a time, 5-point no-regression guardrail):** L4-5 → L2-4-fix/L2-1 → L4-1 →
L2-3 → L3-4. **Wave 2:** L3-3 (post-L4-1) · L3-1 · gated-in L1 items.

## The bet

> **Top-3:** (1) **L2-4 audit → `snr_dependent_csi` flag**, (2) **L2-1 randomization** if the audit
> confirms, (3) **L4-5 composite miner key**, with **L4-1** as the evidence-gated fourth.
>
> **If betting on ONE idea to lift points_met 5→7:** *close the train/eval CSIR distribution gap*
> (L2-4's flag, hardened by L2-1). The two failing points are **precisely** where eval σ_est runs
> 2–17× its high-SNR value (0.2045 @ 0 dB vs 0.0119 @ 30 dB), so a policy trained at fixed ε is OOD
> **there and nowhere else** — which uniquely explains why 36 interventions across 6 mechanism
> families all died at the same 2 points, while state-mining (which shifts state coverage but not ε
> calibration) gave partial gains that plateaued. It is a one-flag, hours-scale, benchmark-preserving
> fix, and because the gate is ≥ (not >), even partially reclosing the min-branch haircut differential
> vs SDMA may be all the margin needed. **If the audit comes back aligned, that negative is itself
> decisive** — it eliminates the distribution-bug hypothesis and hands the bet to L4-5 + L4-1.

## Provenance & honesty notes

- **Cross-model jury:** GLM-5.3 via `qodercli`, Zhipu/GLM family — **distinct from the Qoder
  executor**, satisfying the ARIS cross-family invariant for a Type-B (idea-quality) verdict.
  Provenance is **asserted-by-invocation** (the shim stamps `Reviewer-Backend`/`Reviewer-Family`);
  the reviewer **declined to self-attest a model label** ("identity withheld — the calling harness
  stamps provenance"), which is the honest expected behavior, **not** `REVIEW_UNAVAILABLE`. A real
  verdict was returned. Generation (Phase-1.5/2 lenses) used same-family Qoder `Agent` sub-agents —
  safe for generation, not for the verdict, which went cross-family.
- **Separation of evidence vs hypothesis:** the Section-A criterion correction and the B.1–B.5 code
  facts are **confirmed by direct source reads**. The B.6 overload magnitude is an **open question**
  (the earlier "7×" figure used stale constants and is retracted). The CSIR-OOD mechanism is a
  **hypothesis** — L2-4/L5-5 are the cheap tests that confirm or kill it.
- **Phase-5 pilots SKIPPED** per project scope (the user runs all training/eval). Every ACCEPT-PILOT
  idea is flagged for the user to run; the Wave-0 diagnostics are CPU-minutes and read-only.

## Wiki write-back (Phase 7)

All 23 nodes (22 lens candidates + the jury-added L1-gate probe) written to `research-wiki/ideas/` — 19 as `stage=proposed, outcome=pending` (go-forward, not yet tested; NOT banlisted), 4 desk-rejects as `stage=archived, outcome=negative` with an explicit "jury desk-reject, NOT experimentally tested" label so they enter the banlist and are not re-proposed. Wiki total is now **63 idea nodes** (40 negative = 36 seeded failures + 4 desk-rejects, 4 positive, 19 pending). Gap ids are folded into each node's description rather than `target_gaps=` (gaps are not first-class nodes here, so those edges would dangle). The Section-A criterion correction is written back to RESEARCH_BRIEF.md + gap_map.md and propagated via `rebuild_query_pack` (Failed Ideas banlist still renders; injection gate re-scanned clean).
