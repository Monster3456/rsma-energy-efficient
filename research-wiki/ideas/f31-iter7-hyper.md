---
type: idea
node_id: idea:f31-iter7-hyper
title: "lagrangian_max=100向上解锁解除λ饱和 [i7]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "hyperparameter", "banlist", "iter7"]
---

# lagrangian_max=100向上解锁解除λ饱和 [i7]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 7, channel=hyperparameter).

## Thesis
lagrangian_max=100向上解锁解除λ饱和（iter-52起拖欠6轮的#1欠账）：λ被50截断等价系统性偏低、使PID积分项失效（anti-windup病理），与2个未达标SNR点持续违约精确同构；向上=增强delivery约束压力零牺牲风险，诊断λ轨迹稳定(50,100)且未达标点delivery升则确认饱和污染历史code判定

params_tried: {"lagrangian_max": 100}

Full verdict: score [5, 0.913, -11.7686, 0.6918] 未超过 best [5, 0.9163, -11.0186, 0.6914]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

