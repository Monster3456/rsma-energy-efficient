# IDEA-TRIAGE BUNDLE — cross-model jury request

You are a **cross-model research reviewer** (your model family is DISTINCT from the author/executor).
This is **pure analysis** — you have no file-modification tools. Your job is a **VERDICT**: critique,
re-rank, and triage the candidate research ideas below for a real uplink-RSMA energy-efficiency RL
project. Be adversarial and mechanistic. Do not rubber-stamp. Begin your reply with
`Reviewer-Model: GLM-5.3`.

---

## A. CRITERION CORRECTION (verified from source — this overrides any prior framing)

The author's own knowledge brief WRONGLY stated that each SNR point is "met" only if it clears FOUR
bars (delivery≥1.0×, queue≤1.1×, EE>1.05×, power<1.0×) simultaneously. **Direct code inspection
falsifies this.** The operative score the loop optimizes is `compute_score` in
`scripts/optimize_loop.py:1759-1785`:

```
per-SNR-point "met" gate (lines 1777-1778):
    ppo.delivery >= sdma.delivery * d_ratio(=1.0)  AND  ppo.queue <= sdma.queue * q_ratio(=1.1)
score = [pass_count, avg_delivery, -avg_queue, ee_ratio]   # LEXICOGRAPHIC, pass_count dominates
```

- **`points_met` is gated on TWO bars only: per-point DELIVERY and per-point QUEUE, both RELATIVE to SDMA.** EE and power are **NOT** per-point gates.
- `ee_ratio` is only the **4th (least-significant) lexicographic slot**, a GLOBAL aggregate.
- EE>1.05 and power appear only in `check_termination` (`optimize_loop.py:475-497`) as GLOBAL "all-done" conditions, and EE there is a global mean — they do **not** gate `points_met`.

**Consequence for this campaign:** lifting `points_met` 5→7 means winning a **relative delivery +
queue race vs SDMA at the 2 low-SNR points {0,5} dB** — NOT "fixing low-SNR energy efficiency."
Judge every candidate by: *does it plausibly raise per-point delivery and/or lower per-point queue
at 0/5 dB relative to SDMA, without regressing the 5 passing points?* Ideas aimed purely at
aggregate EE/power are, at best, 4th-slot / termination plays — flag them as such.

## B. CONFIRMED code facts (from the generator sub-agents' source inspection — trust these)

1. **Action = 4K = 96 = [alpha(K), p(K), d(K), order(K)].** Under `group_level_alpha=True` (current
   best config) the Worker's `alpha` block is a **dead dim** (alpha = Meta's `split_mask`).
2. **`SolverExpert` (`src/optimization/solver_expert.py`) is wired ONLY into an OFFLINE
   behavior-cloning pipeline** (`_solver_expert_collect`→`bc_pretrain`) that teaches RL to *imitate*
   power. It is **never used at decision time** to remove/fix a variable. `blend_alpha`
   (`_inject_alpha_guidance`, main.py:209) merely *interpolates* Meta alpha toward an SNR-linear target.
3. **Hard-example miner floor bug:** `_hm_sample_snr` draws `uniform(_HM_EDGES[b], _HM_EDGES[b+1])`
   with `_HM_EDGES[0] = -2.5`, while `snr_min_db = -5`. The **[-5, -2.5] extreme-low tail is
   UNREACHABLE during mining.** Miner key = **delivery-shortfall only** (ignores the queue bar).
4. **No reward/return normalization exists** (no PopArt / RunningMeanStd / VecNormalize);
   `buffer.returns = raw_adv + values`, `value_loss = MSE(values, returns)` on a **single shared critic**.
5. **DVQ Lyapunov drift saturates at low SNR:** `y_component = w_y_tp·(Y/dvq_y_scale)·D_total` with
   fixed `dvq_y_scale=300`, `dvq_y_max=5000`; chronic low-SNR shortfall pins `Y` at the cap →
   `y_weight≈16.7` (wound up, no differential gradient). `compute_reward` RECEIVES `sigma2`
   (rsma_env.py:851) but **never uses it**.
6. **Offered load:** `adaptive_lam = clip(snr_adaptive_load_ratio(snr)·cap/K, lambda_min, lambda_max)`
   (rsma_env.py:616-625). **CURRENT constants: `lambda_min=0.2`, `lambda_max=10.0`,
   `load_ratio_min=0.75`, `load_ratio_max=1.3`, `snr_mid=15`, `steepness=0.3`** (config.py:89-98).
   These were HAND-TUNED across Task#27/43/46/47/48 specifically to reduce low-SNR overload.
   **The exact overload multiple at 0/5 dB is therefore an OPEN, measurable question — NOT the
   "7× overload" an earlier draft claimed (that used stale λ_min=1.5).** Any idea that merely
   re-tunes λ/load constants collides with prior manual tuning.

## C. THE WINNING FAMILY (mine this) — all 4 accepted gains are DATA-DISTRIBUTION shaping

s0 (i22) performance-triggered difficulty-annealing curriculum · s1/s2 (i4/i5) CuRLTraC targeted
hard-example mining · s3 (i8) per-eval-point fine-grained mining + targeted entropy on unmet points.
**Every accepted gain reshaped WHICH states the policy trains on.** Every reward-side, dual-side,
advantage-side, critic-side, weight-averaging, global/tabled-group_alpha, naive-bucket-MoE, and
naive-obs-augment intervention FAILED (36 falsified; families A-F in the project brief). Prefer
candidates that ride data-shaping or open a genuinely new, cheap-to-test axis.

## D. CANDIDATES (21, clustered; each field compressed). Verdict each cluster + call out standouts.

### Cluster L5 — DIAGNOSTICS (no/low training, CPU-minutes). Author leans: RUN THESE FIRST.
- **L5-1 bar-slack binding-criterion decomposition** [diagnostic, LOW, days]. Re-derive the 4 bars
  per point from immutable-eval output tables; identify which bar binds at 0/5 dB + slack size.
  **ALREADY PARTIALLY PAID OFF — it surfaced the Section-A correction.** Remaining value: measure
  per-point delivery/queue slack vs SDMA and the lexicographic tiebreak sensitivity.
- **L5-2 transient-vs-steady-state delivery trajectory** [diagnostic, LOW, days]. Log cumulative
  delivery/queue EVERY step, matched seeds, PPO vs SDMA at 0/5 dB (+10 dB control). Is the scored
  delivery (a whole-trajectory integral incl. the empty-queue ramp, T=200) decided in the early
  fill transient or does it diverge in steady state? Decides warm-up-artifact vs genuine deficit.
- **L5-3 offered-load λ-clip overload crossing** [empirical, MEDIUM, days]. Sweep fixed λ at 0/5 dB
  (via the mutable temporal-config override, immutable eval called unchanged), locate crossing λ*
  where PPO delivery overtakes SDMA. **Must re-run with CURRENT constants (Section B.6).** Tells us
  if the 2 points sit above both methods' feasibility knee (a load-calibration artifact).
- **L5-4 per-user deep-fade deficit concentration** [diagnostic, LOW, days]. Decompose pooled
  delivery deficit (SDMA_served − PPO_served) by per-user path-loss offset (28 dB spread, K=24);
  measure wasted capacity `Σ max(R_k·τ − Q_k,0)` on empty-queue users + the d_ratio throttle. Is the
  gap carried by a small deep-fade cohort while surplus rate is wasted? Names the mining target.
- **L5-5 CSIR min-branch rate-loss relative attribution** [diagnostic, MEDIUM, days]. rate_calc
  returns `min(declared@error_bound=σ_est, actual@H_true)`. Split into robust-regularization vs
  realization-mismatch haircuts; compute the **PPO-minus-SDMA DIFFERENCE** in CSIR-attributable loss.
  Tests whether L2's CSIR-dominance premise survives in the RELATIVE sense the delivery bar needs
  (σ_est=0.2045@0dB vs 0.0119@30dB, 17× spread). A near-zero difference DEMOTES the whole L2 axis.

### Cluster L2 — CSIR-ROBUSTNESS via data shaping (rides Section-C family). Aligned IF L5-5 confirms.
- **L2-1 CSI-error-scale domain randomization** [new method, MEDIUM, days]. Per-episode randomize
  `epsilon_csi` scale (e.g. U[0.5×,2×]), low-SNR episodes drawn from upper half; policy marginalizes
  over the error distribution. Rides mining (s1-s3) unchanged. Mutable env-reset plumbing + curriculum hook.
- **L2-2 CVaR-gated error-scale annealing** [new method, MEDIUM, days]. Inflate training ε only when
  CVaR_0.2 of low-SNR delivery clears SDMA, else retract. Rides s0+s3; trigger stat = CVaR over error
  realizations (NOT the existing per-bucket-delivery CVaR in lagrangian.py — different RV, different role).
- **L2-3 adversarial error-realization mining** [empirical, LOW, days]. Extend the CuRLTraC miner key
  from state-difficulty to adverse per-user `error_variance` seeds at 0/5 dB; oversample them.
  **Lowest-risk L2 idea — literally a new mining key on the one winning family.** Decisive negative
  if failure is uniform across error draws (falsifies CSIR-dominance for our pipeline).
- **L2-4 train/eval CSIR-distribution alignment audit** [diagnostic, LOW, days]. Does the policy train
  under a fixed ε while the evaluator samples an SNR-dependent σ_est that GROWS as SNR drops (OOD at
  0/5 dB)? If so, one-flag `snr_dependent_csi` alignment. Would explain why 36 directions all failed
  at exactly these points. Clear binary outcome.

### Cluster L4 — Lyapunov reward-structure + miner support (mixed; L4-1/L4-5 well-aligned to queue bar).
- **L4-1 bounded/normalized DVQ drift term** [new method, LOW, days]. Replace `Y/dvq_y_scale` with
  `tanh(Y/y_ref)` (or Y/running-shortfall-RMS) in lyapunov.py `compute_reward`. De-saturates the
  wound-up y_weight at low SNR → stabilizes the shared critic's value target (no return-norm exists,
  B.4/B.5). High-SNR Y≈0 so tanh≈linear = unchanged → protects the 5. **Directly targets the QUEUE
  bar (half the per-point gate).** NOT family A (leaves Y accumulator + λ dual untouched).
- **L4-2 SNR-normalized drift/penalty equalizer** [new method, MEDIUM, days]. Consume the unused
  `sigma2` arg (B.5) to build a monotone n(σ²) that equalizes penalty-block RMS across SNR →
  regime-invariant value target. Distinct from L4-1 (regime-AWARE equalizer vs regime-AGNOSTIC bound).
- **L4-3 miner low-SNR support extension** [empirical, MEDIUM, days]. Extend `_HM_EDGES[0]` −2.5→−5
  so the unreachable extreme-low tail (B.3) is minable; weight by low-SNR shortfall. Rides s1-s3.
- **L4-4 miner intra-bucket low-edge density tilt** [empirical, LOW, days]. Replace within-bucket
  uniform SNR draw with a low-edge-tilted density (Beta(a<1)/triangular) for the 2 unmet buckets only.
  Concentrates data at/below 0 & 5 dB. Rides s1-s3; distinct from L4-3 (density vs support).
- **L4-5 miner key = binding-constraint composite** [new method, MEDIUM, days]. Extend miner key from
  delivery-only to delivery + queue-shortfall (w_q·max(0,queue_ratio−1.1)) (+ optional EE-margin).
  **Directly aligns the miner to the TRUE 2-bar per-point gate (Section A).** Rides s3.

### Cluster L3 — SOFT learned regime routing (vs banlisted HARD bucket heads D/i3).
- **L3-1 soft-MoE interference-vs-power router** [new method, MEDIUM, days]. Softmax router over G=2
  cloned expert heads keyed on a dimensionless interference-vs-power scalar from existing obs; cloned
  init ⇒ step-0 ≡ best (protects the 5). NOT D/i3 (no SNR gather, no frozen trunk, all experts get grad).
- **L3-2 CSIR-confidence output-modulation router** [new method, MEDIUM, days]. Softmax over M=2
  output-modulation vectors keyed on realized CSI-error confidence (eps_norm + per-user error_variance).
  Zero-init ⇒ step-0 ≡ best. Architecture-side CSIR hedge (complements L2's data-side).
- **L3-3 delivery-pressure router (EE-aggressive vs delivery-safe expert)** [new method, MEDIUM, days].
  Softmax over 2 cloned experts keyed on queue-backlog delivery-pressure (mean queue_norm + DVQ backlog
  from existing obs). Encodes the lexicographic delivery-first priority as ARCHITECTURE, not dual
  pressure. **Best L3 alignment with Section-A criterion (queue/delivery-keyed).** NOT family A.
- **L3-4 regime-classifier-reweighted mining** [empirical, LOW, days]. Tiny MLP regime classifier
  (input: physical regime descriptor; label: per-state delivery deficit, already in buffer/RCPO cost)
  outputs a soft posterior that reweights the CuRLTraC replay distribution. **NO architecture change,
  NO obs change — pure data-shaping (rides s1-s3), lowest-risk L3.** Generalizes the mining key from
  eval-point-id to a learned physical regime.

### Cluster L1 — variable-REDUCTION solver-in-the-loop (NEW axis; but EE-framed — re-judge under Section A).
- **L1-1 runtime solver power-elimination → RL owns per-user alpha** [new method, MEDIUM, weeks].
  At decision time overwrite the Worker power block with SolverExpert's solve (conditioned on current
  split), set `group_level_alpha=False` so RL owns per-user alpha; power is ELIMINATED not blended.
  NOT F/i3 (hard replace, not interpolate), NOT C (no global/tabled alpha), NOT the offline BC.
- **L1-2 closed-form water-filling power law (cheap CPU reduction operator)** [theoretical, HIGH, weeks].
  Derive a closed-form/KKT power law for the low-SNR EE-DPP surrogate; use as a zero-Adam reduction
  operator so reduction is CPU-viable for FULL runs. Clean negative result if no law matches.
- **L1-3 eliminate power DISTRIBUTION only → RL owns total-power budget scalar** [new method, MEDIUM,
  weeks]. Per-slot water-filling redistributes an RL-chosen total budget; collapses power block to a
  1-dim budget residual, alpha stays Meta-owned. Targets the power/EE bars (4th-slot under Section A).
- **L1-4 performance-triggered ONSET of power reduction** [empirical, MEDIUM, weeks]. Let RL own power
  early; at a low-SNR delivery-feasibility trigger switch power to solver-fixed so alpha fine-tunes
  against a stable manifold. **Rides s0** (extends performance-triggered annealing to the reduction axis).

---

## E. YOUR VERDICT TASK (be specific and mechanistic; cite candidate IDs)

1. **Re-rank under the CORRECTED criterion (Section A).** Which candidates most directly raise
   per-point DELIVERY / lower per-point QUEUE at 0/5 dB vs SDMA without regressing the 5? Demote
   EE/power-only plays to 4th-slot/termination relevance and SAY so.
2. **Sequencing.** The author proposes running the L5 diagnostics FIRST (cheap, no training, and L5-1
   already reframed the problem). Agree or disagree? Which 1-3 diagnostics gate the most downstream
   decisions (e.g. does L5-5 correctly gate the entire L2 axis; does L5-2 gate whether delivery is even
   converged at T=200)? Give a concrete ordered plan.
3. **Banlist collision audit.** For each of your top-5 picks, stress-test the `orthogonality` claim:
   is it a reskin of a FAILED family (A-F), or of the hand-tuned λ/load constants (B.6), or of the
   offline-BC SolverExpert (B.2)? Call out any candidate that is secretly a banlisted mechanism.
4. **Falsifiability + failure modes.** For your top-3, state the cheapest decisive experiment and the
   most likely reason it fails. Flag any candidate whose MVE secretly needs immutable-physics edits.
5. **Kill list.** Which candidates should be dropped NOW (weak, redundant, mis-framed, or infeasible
   on CPU), and why.
6. **A verdict line per candidate**: `ACCEPT-PILOT` / `ACCEPT-ON-PAPER` / `NEEDS-DIAGNOSTIC-FIRST` /
   `REJECT`, one line of justification each.

End with a **top-3 recommendation** and a one-paragraph "if I had to bet on ONE idea to lift
points_met 5→7, it is X because Y."
