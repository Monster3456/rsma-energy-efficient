# Research Wiki Query Pack

_Auto-generated. Do not edit._

## Project Direction
**Problem**

Maximize **energy efficiency (EE = sum-rate / total-power)** of an **uplink RSMA MU-MIMO** link under **hard per-SNR-point delivery-ratio and queue-stability constraints**, benchmarked against SDMA. RSMA splits each user's message into a common stream (decoded by all) + private stream (decoded by the intended user) with MMSE-SIC. The agent controls power, the common/private split (`group_alpha`), precoding, and queue-aware scheduling under a Lyapunov drift-plus-penalty objective. It must hold across the whole SNR range **simultaneously**, not on average.

**Constraints**

**Immutable physics/metric — DO NOT modify (any diff, even whitespace, is force-excluded and breaks comparability):** `src/env/channel.py`, `src/env/rate_calc.py`, `scripts/evaluate_snr_sweep.py`, and the physics call sites (`self.channel`/`self.rate_calc`) in `src/env/rsma_env.py`. Only non-physics parts (reward shaping, observation, curriculum hooks) are mutable.

**Fixed params:** K=24 users, N_r=16 antennas, RZF combining, MMSE-SIC, action dim 4K=96, 7 eval SNR points {0,5,10,15,20,25,30} dB (train range [-5,35]), **imperfect CSI**.

**Success criterion (lexicographic) — CORRECTED from `compute_score` (`optimize_loop.py:1777-1778`):** score = `[points_met, avg_delivery, -avg_queue, ee_ratio]`, lexicographic, **`points_met` (of 7) dominates**. A point is "met" iff **per-point delivery ≥ SDMA×1.0 AND queue ≤ SDMA×1.1** (≥/≤, so matching SDMA passes) — **ONLY these 2 relative bars**. EE>1.05 / power<1.0 are **NOT** per-point gates (ee_ratio = 4th slot, global aggregate; EE/power only in the global `check_termination`, `optimize_loop.py:475-497`). So lifting `points_met` 5→7 = a **relative delivery+queue race vs SDMA at the 2 low-SNR points {0,5} dB**, NOT "fix low-SNR EE" — but a delivery play that wrecks global EE/power still dies at termination.

**Direction**

Structurally-**new** mechanisms to lift `points_met` 5/7 → 7/7: make the **2 low-SNR points {0,5} dB** clear the **per-point delivery+queue gate vs SDMA** without regressing the 5 that pass, keeping global EE/power enough to survive `check_termination`. Must be **orthogonal** to the 40 falsified/desk-rejected families (see Non-Goals / Failed Ideas). Campaign-prioritized axes (see `idea-stage/IDEA_REPORT.md`): (a) **train/eval CSIR-distribution alignment** — eval σ_est runs 2-17× higher at the 2 failing points, so a fixed-ε-trained policy may be OOD there and nowhere else (the jury's top bet); (b) **align the hard-example miner to the true 2-bar gate** (its key is delivery-only) + fix its `[-5,-2.5]` dB support hole; (c) **de-saturate the DVQ Lyapunov drift** at low SNR (gate on verified Y-saturation first — else it is family-A in disguise); (d) **cheap no-training diagnostics** decomposing WHY the 2 points fail before any training spend.

**Background**

Hierarchical PPO: an **SA-GAT Worker** (state-augmented graph-attention over 24 users + history window) proposes per-user actions; a **Meta-Controller** sets slow context. Constraints handled via a **Lagrangian/PID dual** on per-bucket delivery + a **DVQ** Lyapunov term + CVaR. Successful lineage = **CuRLTraC** difficulty-annealing curriculum + **targeted hard-example mining** (per-eval-point SNR resampling by delivery shortfall) + **targeted entropy incentive** on unmet points. `optimize_loop.py` runs autonomously and logs every direction to `logs/opt_knowledge.json`.

**Non-goals**

- Do **not** re-propose any banlisted direction (36 falsified; enumerated in `research-wiki/ideas/*.md` with `outcome: negative`, summarized under `## Failed Ideas`). Exhausted families: Lagrangian/PID-dual tweaks, CuRLTraC-variant curricula, gradient-manipulation (CRPO / primal null-space projection), model-soup/SWA, quantile-critic, SIC-order probes, global & per-bucket-tabled `group_alpha`, scalar `w_power_exceed`/`lagrangian_max` ladders.
- Do **not** touch the immutable physics/metric files (pure-format diffs trip `code_guard` and roll back the iteration).
- Do **not** autonomously restart `optimize_loop.py` or run training/eval — the user runs experiments. This wiki + `/idea-creator` + `/novelty-check` + `/comm-lit-review` are the **selection/knowledge layer only**.

**Domain Knowledge**

- **Low-SNR regime** (where the 2 unmet points sit) is interference/queue-limited, **not** power-limited: burning power saturates without clearing the bar (the λ-saturation / anti-windup pathology behind many failures). These points likely need *structural* help (near-SDMA operation, better SIC ordering, solver warm-start), not more RL pressure.
- **`group_alpha`** (private-stream geometric power scaling `s=v^(k-1)`) interacts with MMSE-SIC ordering. Global tightening (v=1.0, 0.9) **lost points**; any `group_alpha` lever must be adaptive & low-SNR-targeted, not a global constant.
- **Lexicographic trap:** raising aggregate `ee_ratio` while dropping `points_met` is a NET LOSS (iter-22 traded ee 0.938→0.687 but points 4→5 and was accepted). Reason about `points_met` first.

**Existing Results**

**Best: `[5, 0.9163, -11.0186, 0.6914]`** = 5/7 points cleared, avg_delivery 0.9163, queue_metric -11.0186, aggregate EE ratio 0.6914. Gaps: **G1** (TOP PRIORITY) the 2 low-SNR points {0,5} dB fail the **per-point delivery≥SDMA AND queue≤1.1×SDMA gate** → `points_met` stuck 5/7; **G2** global ee_ratio 0.6914 < 1.05 (a `check_termination`/4th-slot concern, NOT a per-point `points_met` blocker); **G3** avg delivery 0.9163 < 1.0× (a global-aggregate/2nd-slot concern); **G4** mechanism-family exhaustion (see `research-wiki/gap_map.md`). Successful lineage (live in codebase, seeded `outcome: positive`): CuRLTraC curriculum (iter 22), targeted hard-example mining (iters 4,5), per-eval-point mining + entropy incentive (iter 8).
## Open Gaps
# Gap Map

_Field gaps with stable IDs. Best `[5, 0.9163, -11.0186, 0.6914]` = `[points_met/7, avg_delivery, queue_metric, ee_ratio]`, lexicographic (points_met dominates). Criterion CORRECTED from `compute_score` (optimize_loop.py:1777-1778)._

## G1 — Two low-SNR points fail the per-point delivery+queue gate
`points_met` stuck at **5/7**. The 2 low-SNR points {0,5} dB fail the per-point gate: **delivery ≥ SDMA×1.0 AND queue ≤ SDMA×1.1** (both relative, ≥/≤ so matching SDMA passes). EE>1.05 / power<SDMA are **NOT** per-point gates. TOP priority: close G1 without regressing the 5 passing points.

## G2 — Global EE ratio below termination target
`ee_ratio` = **0.6914 < 1.05** — a GLOBAL `check_termination` + 4th-lexicographic-slot concern, **not** a `points_met` blocker.

## G3 — Average delivery below SDMA
`avg_delivery` = **0.9163 < 1.0×** — a GLOBAL aggregate (2nd lexicographic slot), not the per-point gate.

## G4 — Mechanism-family exhaustion (meta-gap)
40 falsified/desk-rejected directions span: Lagrangian/PID-dual, CuRLTraC-variant curricula, gradient-manip (CRPO/primal-proj), model-soup/SWA, quantile-critic, SIC-order, global & per-bucket `group_alpha`, scalar penalty ladder
## Failed Ideas (avoid repeating)
- **参考自适应 anti-windup PID 对偶更新 [i7]**: Failure: code_guard blocked edit to scripts/optimize_loop.py.

## Connections
- **Dinkelbach分式奖励重构+State-Augment [i21]**: Failure: code_guard: build produced no file change.

## Connections
- **第二轴课程 [i24]**: Failure: score did not beat best (2 rounds).

## Connections
- **逐桶独立状态机差异化功率惩罚调度 [i26]**: Failure: score did not beat best (2 rounds).

## Connections
- **全信号+易环境双信号课程 [i28]**: Failure: score did not beat best (2 rounds).

## Connections
- **梯度操纵Primal安全优化 [i30]**: Failure: score did not beat best (2 rounds).

## Connections
- **双探针纯评估 [i31]**: Failure: code_guard blocked edit to scripts/run_dual_probes.py.

## Connections
- **梯度操纵Primal安全优化 [i33]**: Failure: score did not beat best (2 rounds).

## Connections
- **梯度操纵Primal安全优化 [i35]**: Failure: score did not beat best (2 rounds).

## Connections
...(truncated)
