---
type: idea
node_id: idea:f19-iter58-code
title: "双评论家字典序PPO样本级优先级掩码 [i58]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter58"]
---

# 双评论家字典序PPO样本级优先级掩码 [i58]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 58, channel=code_innovation).

## Thesis
双评论家字典序PPO样本级优先级掩码：models/加共享encoder双value头（V_D拟合delivery塑形回报含DVQ、V_EE拟合EE回报，均全样本回归），ppo.py分别GAE各自标准化后按mask=(|A_D|>τ)逐样本合成A=where(mask,A_D,β·A_EE)，τ为delivery无差异带宽随CuRLTraC从~1.0σ退火至~0.3σ，掩码只作用policy梯度，

Full verdict: score [4, 0.8884, -13.3843, 0.68] 未超过 best [5, 0.9037, -12.0071, 0.6866]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

