---
type: idea
node_id: idea:s3-iter8-code
title: "逐评估点细粒度硬例挖掘+未达标点定向熵激励 [i8 OK]"
stage: proposed
outcome: positive
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["successful-direction", "code_innovation", "curriculum", "iter8"]
---

# 逐评估点细粒度硬例挖掘+未达标点定向熵激励 [i8 OK]

**stage:** `proposed`  ·  **outcome:** `positive`

Validated SUCCESS (optimize_loop iter 8). Gain: score [5, 0.9123, -11.2, 0.6934] -> [5, 0.9163, -11.0186, 0.6914]

## Thesis
逐评估点细粒度硬例挖掘+未达标点定向熵激励：shortfall从4桶细化为7个SNR评估点（消除桶内稀释），连续未达标streak倍增采样权重（封顶），且仅对未达标点episode提高entropy bonus×1.5注入鞍点逃逸扰动；达标点熵不变+base地板保护

Gain: score [5, 0.9123, -11.2, 0.6934] -> [5, 0.9163, -11.0186, 0.6914]

## Key risks
Success lesson: score [5, 0.9123, -11.2, 0.6934] -> [5, 0.9163, -11.0186, 0.6914]. This family WORKS — build on it, do not discard.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

