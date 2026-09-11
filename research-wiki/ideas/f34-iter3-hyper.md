---
type: idea
node_id: idea:f34-iter3-hyper
title: "激活既有引导通路探针 [i3]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "hyperparameter", "banlist", "iter3"]
---

# 激活既有引导通路探针 [i3]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 3, channel=hyperparameter).

## Thesis
激活既有引导通路探针：blend_alpha=0.1小步启用引导动作混合（从未被触碰的既有旋钮），若指标偏离冻结基线则证实model-based引导通路实装且有效，开启与全部失败族（纯RL机制改动）结构正交的solver-in-the-loop新族

params_tried: {"V": "V_current*0.25", "V_min": "V_min_current*0.25", "V_max": "V_max_current*0.25", "blend_alpha": 0.1}

Full verdict: score [5, 0.914, -11.7057, 0.6932] 未超过 best [5, 0.9163, -11.0186, 0.6914]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

