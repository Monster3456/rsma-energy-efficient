# RSMA Energy Efficient

RSMA-MU-MIMO energy efficiency optimization with hierarchical PPO (SA-GAT Worker + Meta-Controller).

This project uses hierarchical reinforcement learning to maximize energy efficiency (EE) under delivery rate and queue stability constraints.

## Project Structure

- `src/env/` — Channel models, RSMA rate calculation, gym environment, beamforming, traffic model
- `src/models/` — Network architectures (Actor-Critic Worker, Meta-Controller)
- `src/training/` — PPO training, Lagrangian constraints, replay buffer, curriculum scheduling
- `src/optimization/` — Lyapunov optimization, power control, Meta reward formulation
- `scripts/` — Training entry point, SNR sweep evaluation, baselines, optimization loop
- `tests/` — Unit tests for physical models and training logic
- `experiments/v2/` — v2 experiment code (meta-controller training, evaluation, power solver)
- `logs/` — Experiment artifacts (gitignored except opt_knowledge.json)

