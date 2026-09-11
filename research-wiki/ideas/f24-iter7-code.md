---
type: idea
node_id: idea:f24-iter7-code
title: "课程末段定向精修+受控探索注入 [i7]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter7"]
---

# 课程末段定向精修+受控探索注入 [i7]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 7, channel=code_innovation).

## Thesis
课程末段定向精修+受控探索注入（PATH式主动课程精修）：CuRLTraC冻结在评估配置后，对2个未达标SNR点的shortfall权重w_hard上调（如2→4）并延长冻结训练≥500ep，同时在actor注入小幅参数空间噪声（或entropy bonus×1.5）助其逃离当前鞍点；达标点用base地板≥1/(2K)保护防回归

Full verdict: score [5, 0.8944, -12.61, 0.6831] 未超过 best [5, 0.9123, -11.2, 0.6934]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

