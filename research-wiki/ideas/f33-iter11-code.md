---
type: idea
node_id: idea:f33-iter11-code
title: "逐评估点advantage温度重加权 [i11]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter11"]
---

# 逐评估点advantage温度重加权 [i11]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 11, channel=code_innovation).

## Thesis
逐评估点advantage温度重加权：ppo.py按样本所属SNR评估点shortfall_p=max(0,target_p−delivery_EMA_p)对delivery-cost advantage分量连续重加权w_p=1+κ·shortfall_p（κ=2.5起随CuRLTraC退火、shortfall用EMA），未达标点梯度贡献放大且不减达标点曝光

Full verdict: score [5, 0.9124, -11.8686, 0.6902] 未超过 best [5, 0.9163, -11.0186, 0.6914]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

