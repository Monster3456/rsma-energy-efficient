---
type: idea
node_id: idea:f18-iter56-code
title: "桶上下文条件化观测增广 [i56]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter56"]
---

# 桶上下文条件化观测增广 [i56]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 56, channel=code_innovation).

## Thesis
桶上下文条件化观测增广（CCPO式约束条件化）：rsma_env.py观测末端替换6维冗余/常数特征为[桶索引/本桶delivery_EMA−target余量/episode delivery短缺紧迫度/桶级λ_b/功率超支比例/课程进度]，value+critic同步接收，余量特征训练期±ε抖动随课程退火，内嵌计数器指纹硬门槛

Full verdict: score [5, 0.8823, -13.2557, 0.6829] 未超过 best [5, 0.9037, -12.0071, 0.6866]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

