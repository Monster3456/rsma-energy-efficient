---
type: idea
node_id: idea:f23-iter3-code
title: "桶专家多头actor [i3]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter3"]
---

# 桶专家多头actor [i3]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 3, channel=code_innovation).

## Thesis
桶专家多头actor（策略结构容量分化）：models/将actor最终线性层从best深拷贝为G=4个SNR桶专家头+one-hot硬路由（step-0逐bit==best起点零损失），冻结共享主干仅小lr微调各头（每头只在本桶数据上更新），critic共享，train.py按episode的SNR桶硬路由（episode内不切换），step-0等值断言+per-head梯度范数+逐桶路由计数入

Full verdict: qodercli 施工超时（1500s） (rounds_tried=0)

## Key risks
Failure: build timeout (1500s).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

