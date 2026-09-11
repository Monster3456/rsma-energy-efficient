---
type: idea
node_id: idea:f26-iter12-code
title: "贪心字典序模型汤 [i12]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter12"]
---

# 贪心字典序模型汤 [i12]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 12, channel=code_innovation).

## Thesis
贪心字典序模型汤（SWA/权重平均）：复用逐点硬例挖掘课程末段已有checkpoint（不足则best分叉N=3短分支仅变seed/w_hard补料+末段加密存ckpt），按单模型严格字典序score降序逐个尝试加入平均、同一评估种子集下字典序不降才保留，终选{贪心汤,均匀汤,单最优}取优；贪心汤永不劣于单最优ckpt无回退风险

Full verdict: score [5, 0.913, -11.7686, 0.6918] 未超过 best [5, 0.9163, -11.0186, 0.6914]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

