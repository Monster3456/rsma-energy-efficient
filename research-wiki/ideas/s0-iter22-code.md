---
type: idea
node_id: idea:s0-iter22-code
title: "性能触发式难度退火课程 [i22 OK]"
stage: proposed
outcome: positive
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["successful-direction", "code_innovation", "curriculum", "iter22"]
---

# 性能触发式难度退火课程 [i22 OK]

**stage:** `proposed`  ·  **outcome:** `positive`

Validated SUCCESS (optimize_loop iter 22). Gain: score [4, 0.9331, -11.0257, 0.9382] -> [5, 0.9037, -12.0071, 0.6866]

## Thesis
性能触发式难度退火课程（CuRLTraC风格）：训练从易达标区（load_ratio_max=0.8、snr_min=5dB）起步，维护近50个episode逐点delivery达标率，≥β=0.5才推进一档难度（5档线性退火到评估配置load_max=1.3/snr_min=-5dB），<0.3连续20ep回退一档滞回，到达评估配置后冻结收尾训练≥500ep

Gain: score [4, 0.9331, -11.0257, 0.9382] -> [5, 0.9037, -12.0071, 0.6866]

## Key risks
Success lesson: score [4, 0.9331, -11.0257, 0.9382] -> [5, 0.9037, -12.0071, 0.6866]. This family WORKS — build on it, do not discard.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

