---
type: paper
node_id: paper:demirel2025_generalization_reinforcement_learning
title: "Generalization in Reinforcement Learning for Radio Access Networks"
authors: ["Burak Demirel", "Yu Wang", "Cristian Tatino", "Pablo Soldati"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2507.06602"
  doi: null
  s2: null
tags: ["domain-randomization", "rl-generalization", "ran", "prior-work", "novelty-2026-09"]
added: 2026-09-03T16:04:31Z
---

# Generalization in Reinforcement Learning for Radio Access Networks

## One-line thesis
P2 (co-closest neighbor). Studies RL generalization across RAN topology/traffic/mobility/channel and EXPLICITLY uses domain randomization. Delta: mismatch attributed to diverse environments, NOT CSI-error-scale at low SNR; downlink link adaptation; no RSMA / uplink-EE / queue-delivery constraint; does not diagnose a fixed-eps-vs-sigma_est train/eval gap.

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

> Modern RAN operate in highly dynamic and heterogeneous environments, where hand-tuned, rule-based RRM algorithms often underperform. While RL can surpass such heuristics in constrained settings, the diversity of deployments and unpredictable radio conditions introduce major generalization challenges. Data-driven policies frequently overfit to training conditions, degrading performance in unseen scenarios. To address this, we propose a generalization-centered RL framework for RAN control that: (i) robustly reconstructs dynamically varying states from partial and noisy observations, while encoding static and semi-static information, such as radio nodes, cell attributes, and their topology, through graph representations; (ii) applies domain randomization to broaden the training distribution; and (iii) distributes data generation across multiple actors while centralizing training in a cloud-compatible architecture aligned with O-RAN principles. Although generalization increases computational and data-management complexity, our distributed design mitigates this by scaling data collection and training across diverse network conditions. Applied to downlink link adaptation in five 5G benchmarks, our policy improves average throughput and spectral efficiency by ~10% over an OLLA baseline (10% BLER target) in full-buffer MIMO/mMIMO and by >20% under high mobility. It matches specialized RL in full-buffer traffic and achieves up to 4- and 2-fold gains in eMBB and mixed-traffic benchmarks, respectively. In nine-cell deployments, GAT models offer 30% higher throughput over MLP baselines. These results, combined with our scalable architecture, offer a path toward AI-native 6G RAN using a single, generalizable RL agent.

