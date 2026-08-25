# Eval Compatibility for Obs/Architecture Variants

> Hard constraints from the immutable evaluation script. Violating these silently breaks checkpoint evaluation.

---

## The Constraint Chain

`scripts/evaluate_snr_sweep.py` is **immutable** (optimize-loop code_guard). It:

1. Loads checkpoints with `model.act(obs, deterministic=True)` — **observation only**, no hidden state, no episode-boundary reset hook. This makes recurrent policies (GRU/LSTM Workers) infeasible: act-time and update-time log-probs diverge because PPO update is stateless per shuffled sample.
2. Infers `d_user_eff` from the checkpoint's `user_embed.0.weight` shape, then computes `n_inject = d_user_eff - d_user(config)`. This inference supports only small injections (designed for +2); a large mismatch (e.g., history features growing the per-user obs by H·F) breaks shape reconstruction.
3. Reuses one model across 50 episodes × 7 SNR points with fresh env resets — any variant relying on cross-episode state is invalid.

## Rules for Any Obs-Dim or Worker-Architecture Change

- **Keep the policy stateless.** Cross-slot information must enter through the observation itself (e.g., history window), not through model hidden state.
- **Grow the per-user obs region, never the global tail.** The global tail `[time_ratio, snr_norm, eps_norm]` must stay the last 3 dims; per-user features are appended within the per-user block so the eval script's per-user-region logic keeps working.
- **Default off.** New obs features must be behind a config flag that defaults to 0/off, with a byte-equality test proving the default path is unchanged.
- **Provide an eval wrapper, not a fork.** Do not copy metric logic. Instead patch `src.config.Config` (a subclass mutating `system.obs_history_*` and `ppo.worker_d_user` after `super().__init__()`) **before** `runpy.run_path("scripts/evaluate_snr_sweep.py", run_name="__main__")` with rewritten `sys.argv`. See `scripts/evaluate_snr_sweep_temporal.py`.
- **Include a preflight check** in the wrapper: assert `user_embed` input dim == `worker_d_user_eff + 2` before launching the sweep.
- **Resume equivalence.** If loading an old checkpoint into a wider obs input, the padding-aware loader must zero the new input weights so step-0 behavior is unchanged.

## Required Tests for a Variant

- Default-off invariance: obs bytes identical with flag on=0 vs. absent.
- Eval round-trip: build a checkpoint via the same construction path as `src/training/main.py`, then run the wrapper script on it in a subprocess and assert exit 0.
- Rejection path: the wrapper must fail loudly when given a checkpoint whose dims don't match the requested variant.

## Common Mistake

Training a variant for hours, then discovering the eval script crashes with `mat1 and mat2 shapes cannot be multiplied` — or worse, **silently misreads** extra obs dims as global conditioning, producing plausible-looking but wrong metrics. Design the eval path before training, not after.
