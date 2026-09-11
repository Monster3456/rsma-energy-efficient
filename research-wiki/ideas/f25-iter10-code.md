---
type: idea
node_id: idea:f25-iter10-code
title: "逐点挖掘原料的贪心字典序模型汤 [i10]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter10"]
---

# 逐点挖掘原料的贪心字典序模型汤 [i10]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 10, channel=code_innovation).

## Thesis
逐点挖掘原料的贪心字典序模型汤（SWA/权重平均）：取逐点硬例挖掘课程末段5-10个同源checkpoint做贪心权重平均——按单支字典序score排序、逐支加入仅当汤的字典序score不降才保留，平坦化尖锐极小抑制达标点震荡；原料不足则从best分叉N=3短程逐点挖掘分支喂汤（仅变w_hard/种子），零训练成本主路径，soup_enable=False回退best

Full verdict: score [5, 0.913, -11.7686, 0.6918] 未超过 best [5, 0.9163, -11.0186, 0.6914]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

