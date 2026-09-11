---
type: idea
node_id: idea:f27-iter2-code
title: "分位数critic [i2]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter2"]
---

# 分位数critic [i2]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 2, channel=code_innovation).

## Thesis
分位数critic（DisPPO式分布价值表征）：models/将critic末层1→16维non-crossing分位数头（用现有value头初始化使全部分位数=当前V预测，step-0与best行为一致），training/价值损失换quantile Huber(κ=1,τ_i=(2i−1)/32)、advantage取分位数均值（GAE流程不变），消除7个SNR点多峰混合回报下均值criti

Full verdict: score [5, 0.8929, -13.4, 0.6833] 未超过 best [5, 0.9163, -11.0186, 0.6914]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

