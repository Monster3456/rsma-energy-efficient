---
type: idea
node_id: idea:f06-iter31-code
title: "双探针纯评估 [i31]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter31"]
---

# 双探针纯评估 [i31]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 31, channel=code_innovation).

## Thesis
双探针纯评估（零训练）去三重混淆：探针A扫描5个group_alpha窗口定死alpha→功率因果方向，探针B切perfect CSI验证'功率对冲均衡'假设，附带功率头饱和诊断；一次实验区分'信号失效/策略不可达/物理不存在'，阳性轴下轮触发对应课程（alpha窗口硬约束+重训 或 CSI质量ADR退火课程）

Full verdict: 护栏拦截: 修改了白名单外文件: scripts/run_dual_probes.py (rounds_tried=0)

## Key risks
Failure: code_guard blocked edit to scripts/run_dual_probes.py.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

