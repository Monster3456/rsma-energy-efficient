---
type: idea
node_id: idea:f15-iter50-code
title: "逐桶自适应group_alpha查表调度 [i50]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter50"]
---

# 逐桶自适应group_alpha查表调度 [i50]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 50, channel=code_innovation).

## Thesis
逐桶自适应group_alpha查表调度（7评估点渐变v值，低SNR点v→1.0保delivery/高SNR点v=0.7激进省功率）：train.py episode reset按当前SNR最近邻查表setattr(env, group_alpha_min/max)（env第924行实时读则零env改动，缓存则≤2行reset重读），config加≤9字段；机制=相对分配整形器逐点差异化（低SNR

Full verdict: score [4, 0.874, -14.5086, 0.7025] 未超过 best [5, 0.9037, -12.0071, 0.6866]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

