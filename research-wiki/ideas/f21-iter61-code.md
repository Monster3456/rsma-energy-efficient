---
type: idea
node_id: idea:f21-iter61-code
title: "桶级自模仿学习 [i61]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter61"]
---

# 桶级自模仿学习 [i61]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 61, channel=code_innovation).

## Thesis
桶级自模仿学习（SIL）锚定自身高delivery轨迹：src/training/新增per-SNR桶exemplar缓冲（存delivery达标且低功率的自身历史回合），PPO update后加辅助模仿损失−logπ(a|s)·max(R_D−V(s),0)强化已达标行为防课程遗忘，与CuRLTraC联动逐阶段刷新缓冲，指纹门槛入metrics

Full verdict: score [5, 0.8861, -14.6271, 0.696] 未超过 best [5, 0.9037, -12.0071, 0.6866]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

