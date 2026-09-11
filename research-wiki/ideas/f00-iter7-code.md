---
type: idea
node_id: idea:f00-iter7-code
title: "参考自适应 anti-windup PID 对偶更新 [i7]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter7"]
---

# 参考自适应 anti-windup PID 对偶更新 [i7]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 7, channel=code_innovation).

## Thesis
参考自适应 anti-windup PID 对偶更新：SNR 分桶乘子加条件积分抗饱和（λ 触顶且仍违约时冻结积分）+ dead-zone 误差死区 + 以桶 delivery EMA 为参考的自适应阈值 d_b*=min(threshold, EMA+margin) 单调递增，消除不可行桶 windup 烧功率

Full verdict: 护栏拦截: 修改了白名单外文件: scripts/optimize_loop.py (rounds_tried=0)

## Key risks
Failure: code_guard blocked edit to scripts/optimize_loop.py.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

