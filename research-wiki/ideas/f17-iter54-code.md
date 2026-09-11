---
type: idea
node_id: idea:f17-iter54-code
title: "桶级CRPO交替更新+激活指纹硬门槛 [i54]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter54"]
---

# 桶级CRPO交替更新+激活指纹硬门槛 [i54]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 54, channel=code_innovation).

## Thesis
桶级CRPO交替更新+激活指纹硬门槛：落点移至ppo.py更新规则层，每次update前查各SNR桶delivery_EMA——全桶≥target−η走正常EE步、任一违约则advantage替换为cost-GAE且仅违约桶样本非零权重；η随CuRLTraC课程0.05→0.01收紧；内嵌update计数器+uuid nonce+src哈希manifest写入回传schema，判别改动是否真被执行

Full verdict: score [5, 0.9027, -12.2371, 0.6883] 未超过 best [5, 0.9037, -12.0071, 0.6866]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

