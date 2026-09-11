---
type: idea
node_id: idea:f14-iter48-hyper
title: "group_alpha轴v=0.9单点轻探 [i48]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "hyperparameter", "banlist", "iter48"]
---

# group_alpha轴v=0.9单点轻探 [i48]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 48, channel=hyperparameter).

## Thesis
group_alpha轴v=0.9单点轻探（收紧是否伤点数的实证边界）：group_alpha_min=group_alpha_max=0.9温和收紧private流功率几何缩放（尾部份额0.9^23≈0.089），delivery_threshold维持0.95；与v=1.0同税对照比四项delta——保点且EE↑/power↓则下轮渐进v=0.85，丢点则全局收紧证伪、下下轮转逐桶自适应v（高

params_tried: {"group_alpha_min": 0.9, "group_alpha_max": 0.9, "delivery_threshold": 0.95}

Full verdict: score [5, 0.8944, -13.2014, 0.6763] 未超过 best [5, 0.9037, -12.0071, 0.6866]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

