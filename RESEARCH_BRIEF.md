# RESEARCH_BRIEF — RSMA Energy-Efficient Uplink (hierarchical PPO)

_Distilled from `MODEL.md`, `CLAUDE.md`, `agents.md`, `logs/opt_knowledge.json`. Data source for `query_pack.md → ## Project Direction`. Headings match the research-wiki template. Kept terse on purpose: the query-pack assembler gives Project Direction no char cap and `break`s once the 8000-char budget is hit, so a verbose brief crowds out the Open Gaps and Failed Ideas (banlist) sections. Full detail lives in the idea nodes and `logs/opt_knowledge.json`._

## Problem Statement

Maximize **energy efficiency (EE = sum-rate / total-power)** of an **uplink RSMA MU-MIMO** link under **hard per-SNR-point delivery-ratio and queue-stability constraints**, benchmarked against SDMA. RSMA splits each user's message into a common stream (decoded by all) + private stream (decoded by the intended user) with MMSE-SIC. The agent controls power, the common/private split (`group_alpha`), precoding, and queue-aware scheduling under a Lyapunov drift-plus-penalty objective. It must hold across the whole SNR range **simultaneously**, not on average.

## Constraints

**Immutable physics/metric — DO NOT modify (any diff, even whitespace, is force-excluded and breaks comparability):** `src/env/channel.py`, `src/env/rate_calc.py`, `scripts/evaluate_snr_sweep.py`, and the physics call sites (`self.channel`/`self.rate_calc`) in `src/env/rsma_env.py`. Only non-physics parts (reward shaping, observation, curriculum hooks) are mutable.

**Fixed params:** K=24 users, N_r=16 antennas, RZF combining, MMSE-SIC, action dim 4K=96, 7 eval SNR points {0,5,10,15,20,25,30} dB (train range [-5,35]), **imperfect CSI**.

**Success criterion (lexicographic) — CORRECTED from `compute_score` (`optimize_loop.py:1777-1778`):** score = `[points_met, avg_delivery, -avg_queue, ee_ratio]`, lexicographic, **`points_met` (of 7) dominates**. A point is "met" iff **per-point delivery ≥ SDMA×1.0 AND queue ≤ SDMA×1.1** (≥/≤, so matching SDMA passes) — **ONLY these 2 relative bars**. EE>1.05 / power<1.0 are **NOT** per-point gates (ee_ratio = 4th slot, global aggregate; EE/power only in the global `check_termination`, `optimize_loop.py:475-497`). So lifting `points_met` 5→7 = a **relative delivery+queue race vs SDMA at the 2 low-SNR points {0,5} dB**, NOT "fix low-SNR EE" — but a delivery play that wrecks global EE/power still dies at termination.

## What I'm Looking For

**STATUS: core campaign closed (2026-09-11).** The 2-point gap that motivated this
campaign is closed: the solver-expert distilled hierarchy passes **7/7 evaluation
points** (§ Existing Results). Remaining open work is **statistical rigor, not new
mechanisms**: (a) multi-seed CI replication for any fine-tune component claim (all
M5.x intervention deltas fall inside the champion recipe's own ±0.005 seed band);
(b) recommended follow-on probe, if compute allows: explicit freeze-onset response
surface with `--lr-freeze-update` at 3 onsets × 3 seeds (`M5.4-M` pre-registration in
`experiment-results.md`), the only axis with a provisional (non-significant) signal.

## Background

Hierarchical PPO: an **SA-GAT Worker** (state-augmented graph-attention over 24 users + history window) proposes per-user actions; a **Meta-Controller** sets slow context. Constraints handled via a **Lagrangian/PID dual** on per-bucket delivery + a **DVQ** Lyapunov term + CVaR. Successful lineage = **CuRLTraC** difficulty-annealing curriculum + **targeted hard-example mining** (per-eval-point SNR resampling by delivery shortfall) + **targeted entropy incentive** on unmet points. `optimize_loop.py` runs autonomously and logs every direction to `logs/opt_knowledge.json`.

## Non-Goals

- Do **not** re-propose any banlisted direction (36 falsified; enumerated in `research-wiki/ideas/*.md` with `outcome: negative`, summarized under `## Failed Ideas`). Exhausted families: Lagrangian/PID-dual tweaks, CuRLTraC-variant curricula, gradient-manipulation (CRPO / primal null-space projection), model-soup/SWA, quantile-critic, SIC-order probes, global & per-bucket-tabled `group_alpha`, scalar `w_power_exceed`/`lagrangian_max` ladders.
- Do **not** touch the immutable physics/metric files (pure-format diffs trip `code_guard` and roll back the iteration).
- Do **not** autonomously restart `optimize_loop.py` or run training/eval — the user runs experiments. This wiki + `/idea-creator` + `/novelty-check` + `/comm-lit-review` are the **selection/knowledge layer only**.

## Domain Knowledge

- **Low-SNR regime** (where the 2 unmet points sit) is interference/queue-limited, **not** power-limited: burning power saturates without clearing the bar (the λ-saturation / anti-windup pathology behind many failures). These points likely need *structural* help (near-SDMA operation, better SIC ordering, solver warm-start), not more RL pressure.
- **`group_alpha`** (private-stream geometric power scaling `s=v^(k-1)`) interacts with MMSE-SIC ordering. Global tightening (v=1.0, 0.9) **lost points**; any `group_alpha` lever must be adaptive & low-SNR-targeted, not a global constant.
- **Lexicographic trap:** raising aggregate `ee_ratio` while dropping `points_met` is a NET LOSS (iter-22 traded ee 0.938→0.687 but points 4→5 and was accepted). Reason about `points_met` first.

## Existing Results (if any)

**FINAL (2026-09-11, official 7pt×50ep imperfect-CSI evaluation; full write-up `docs/paper-results.md`):**

- **Champion** `logs/m5_armA_sftp/checkpoints/best_guard.pt` = **[7, 0.9519, -8.3971, 1.0804]** — all 7 eval points clear the per-point delivery≥SDMA AND queue≤1.1×SDMA gates; EE ratio 1.0804>1.05; mean power 13.84 W < SDMA. Dominates SDMA/NOMA/EqualPower on delivery/queue/overflow/EE at every SNR (30 dB: 0.888 vs 0.664/0.581/0.601) — see `logs/figs/benchmarks_all_algorithms.png`.
- **Lineage**: M4 pure-BC (8k solver-expert demos) first 7/7 [7, 0.9441, …] → M5 SFTP fine-tune [7, 0.9519, …] → seed 43 verbatim rerun **reproduces within ±0.002** (0.9503); finals 0.9491–0.9496. **σ_seed ≈ 0.005.**
- **Campaign verdict**: five released-LR/trust-region/anchor/data-reshaping arms (M5.1–M5.3) all score 0.946–0.9474 — **inside champion seed noise → no treatment effect at current n**; the "frozen-lr-is-the-lever" narrative is unproven, re-stated as "seed-robust ~0.950±0.005 fixed point". Full per-arm pre-registered verdict tables in `experiment-results.md`.
- Guard lesson: legacy 3-point mini-eval mis-ranks checkpoints (S44 best_guard −0.9 pt, final fine); **guard v2** (6-point + queue) is mandatory going forward.
- Statistically-sound next probe, if funded: 3-onset × 3-seed explicit-freeze response surface (M5.4-M pre-registration), not single-seed arbitration.
