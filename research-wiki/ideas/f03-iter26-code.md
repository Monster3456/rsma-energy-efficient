---
type: idea
node_id: idea:f03-iter26-code
title: "逐桶独立状态机差异化功率惩罚调度 [i26]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter26"]
---

# 逐桶独立状态机差异化功率惩罚调度 [i26]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 26, channel=code_innovation).

## Thesis
逐桶独立状态机差异化功率惩罚调度：4个SNR桶各维护独立(w_b,stage_b,win_b)，env每episode按场景桶查表取标量w_power_exceed（差异化全在调度层、env零改动）；达标有裕量(≥0.01)的桶×1.2/档爬升，裕量<0.01的边际桶立即回退一档+冻结2窗（退快进慢不对称滞回），并设全局达标点数棘轮否决（低于历史最好值期间禁止任何桶推进防桶间泄漏）

Full verdict: score [4, 0.9074, -12.8971, 0.7106] 未超过 best [5, 0.9037, -12.0071, 0.6866]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

