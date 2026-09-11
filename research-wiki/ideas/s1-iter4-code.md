---
type: idea
node_id: idea:s1-iter4-code
title: "CuRLTraC定向硬例挖掘 [i4 OK]"
stage: proposed
outcome: positive
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["successful-direction", "code_innovation", "curriculum", "iter4"]
---

# CuRLTraC定向硬例挖掘 [i4 OK]

**stage:** `proposed`  ·  **outcome:** `positive`

Validated SUCCESS (optimize_loop iter 4). Gain: score [5, 0.9037, -12.0071, 0.6866] -> [5, 0.9119, -11.3857, 0.6927]

## Thesis
CuRLTraC定向硬例挖掘（ALP-GMM式任务空间主动重采样）：train.py维护逐桶delivery缺口shortfall_b=max(0,target_b−delivery_EMA_b)，按p_b∝base+w_hard×shortfall_b重加权episode的SNR采样分布，把训练曝光导向2个未达标点；base概率地板≥1/(2K)防已达标点回归，shortfall慢EMA防震荡，

Gain: score [5, 0.9037, -12.0071, 0.6866] -> [5, 0.9119, -11.3857, 0.6927]

## Key risks
Success lesson: score [5, 0.9037, -12.0071, 0.6866] -> [5, 0.9119, -11.3857, 0.6927]. This family WORKS — build on it, do not discard.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

