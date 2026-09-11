---
type: idea
node_id: idea:f13-iter46-hyper
title: "group_alpha轴启动 [i46]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "hyperparameter", "banlist", "iter46"]
---

# group_alpha轴启动 [i46]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 46, channel=hyperparameter).

## Thesis
group_alpha轴启动——v=1.0等权对照组：group_alpha_min=group_alpha_max=1.0关闭private流几何功率缩放（s=v^(k-1)≡1），delivery_threshold恢复0.95单因子原则；一次实验回答'group_alpha当前在帮忙还是帮倒忙'并兼测从best重训的基线方差（噪声标尺），为后续收紧侧{0.85,0.7}扫描定基准；环路'未超

params_tried: {"group_alpha_min": 1.0, "group_alpha_max": 1.0, "delivery_threshold": 0.95}

Full verdict: score [5, 0.8967, -12.9271, 0.6796] 未超过 best [5, 0.9037, -12.0071, 0.6866]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

