---
type: paper
node_id: paper:schröder2025_modelfree_robust_beamforming
title: "Model-Free Robust Beamforming in Satellite Downlink using Reinforcement Learning"
authors: ["Alea Schröder", "Steffen Gracla", "Carsten Bockelmann", "Dirk Wübben", "Armin Dekorsy"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2510.25393"
  doi: null
  s2: null
tags: ["robust-rl", "beamforming", "satellite", "imperfect-csi", "prior-work", "novelty-2026-09"]
added: 2026-09-03T16:04:52Z
---

# Model-Free Robust Beamforming in Satellite Downlink using Reinforcement Learning

## One-line thesis
P3. Model-free robust beamforming (SAC) in LEO satellite DOWNLINK under imperfect CSIT (AoD/phase errors). Delta: no SNR-dependent error link, no train/eval distribution-mismatch diagnosis; not RSMA / uplink / EE / constrained.

## Problem / Gap
_TODO._

## Method
_TODO._

## Key Results
_TODO._

## Assumptions
_TODO._

## Limitations / Failure Modes
_TODO._

## Reusable Ingredients
_TODO._

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
_TODO._

## Abstract (original)

> Satellite-based communications are expected to be a substantial future market in 6G networks. As satellite constellations grow denser and transmission resources remain limited, frequency reuse plays an increasingly important role in managing inter-user interference. In the multi-user downlink, precoding enables the reuse of frequencies across spatially separated users, greatly improving spectral efficiency. The analytical calculation of suitable precodings for perfect channel information is well studied, however, their performance can quickly deteriorate when faced with, e.g., outdated channel state information or, as is particularly relevant for satellite channels, when position estimates are erroneous. Deriving robust precoders under imperfect channel state information is not only analytically intractable in general but often requires substantial relaxations of the optimization problem or heuristic constraints to obtain feasible solutions. Instead, in this paper we flexibly derive robust precoding algorithms from given data using reinforcement learning. We describe how we adapt the applied Soft Actor-Critic learning algorithm to the problem of downlink satellite beamforming and show numerically that the resulting precoding algorithm adjusts to all investigated scenarios. The considered scenarios cover both single satellite and cooperative multi-satellite beamforming, using either global or local channel state information, and two error models that represent increasing levels of uncertainty. We show that the learned algorithms match or markedly outperform two analytical baselines in sum rate performance, adapting to the required level of robustness. We also analyze the mechanisms that the learned algorithms leverage to achieve robustness. The implementation is publicly available for use and reproduction of the results.

