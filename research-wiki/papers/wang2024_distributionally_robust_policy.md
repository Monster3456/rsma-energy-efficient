---
type: paper
node_id: paper:wang2024_distributionally_robust_policy
title: "Distributionally Robust Policy Learning under Concept Drifts"
authors: ["Jingyuan Wang", "Zhimei Ren", "Ruohan Zhan", "Zhengyuan Zhou"]
year: 2024
venue: "Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025"
external_ids:
  arxiv: "2412.14297"
  doi: null
  s2: null
tags: ["distributionally-robust", "rl-theory", "concept-drift", "prior-work", "novelty-2026-09"]
added: 2026-09-03T16:05:47Z
---

# Distributionally Robust Policy Learning under Concept Drifts

## One-line thesis
P7. Distributionally robust policy learning under concept drifts (generic RL theory). Matches the 'distributionally robust policy' framing of claim C3. Delta: generic RL, not wireless; no CSI / sigma_est, no RSMA, no SNR-dependent error, no sim channel model.

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

> Distributionally robust policy learning aims to find a policy that performs well under the worst-case distributional shift, and yet most existing methods for robust policy learning consider the worst-case joint distribution of the covariate and the outcome. The joint-modeling strategy can be unnecessarily conservative when we have more information on the source of distributional shifts. This paper studies a more nuanced problem -- robust policy learning under the concept drift, when only the conditional relationship between the outcome and the covariate changes. To this end, we first provide a doubly-robust estimator for evaluating the worst-case average reward of a given policy under a set of perturbed conditional distributions. We show that the policy value estimator enjoys asymptotic normality even if the nuisance parameters are estimated with a slower-than-root-$n$ rate. We then propose a learning algorithm that outputs the policy maximizing the estimated policy value within a given policy class $Π$, and show that the sub-optimality gap of the proposed algorithm is of the order $κ(Π)n^{-1/2}$, where $κ(Π)$ is the entropy integral of $Π$ under the Hamming distance and $n$ is the sample size. A matching lower bound is provided to show the optimality of the rate. The proposed methods are implemented and evaluated in numerical studies, demonstrating substantial improvement compared with existing benchmarks.

