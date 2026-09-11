---
type: idea
node_id: idea:f16-iter52-code
title: "Relaxed-IPO对数障碍+SCPO状态级交付配额信用 [i52]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter52"]
---

# Relaxed-IPO对数障碍+SCPO状态级交付配额信用 [i52]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 52, channel=code_innovation).

## Thesis
Relaxed-IPO对数障碍+SCPO状态级交付配额信用：奖励从episode末标量λ线性惩罚改为 r_t=EE_t+(1/V_b)·log(margin_t+ε)，margin_t=累计交付进度−thr·(t/T)的逐步配额余量，margin≤δ切线延拓防−inf，与CuRLTraC课程退火联动

Full verdict: score [4, 0.9074, -12.8971, 0.7106] 未超过 best [5, 0.9037, -12.0071, 0.6866]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

