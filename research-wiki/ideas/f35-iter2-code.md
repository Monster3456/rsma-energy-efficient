---
type: idea
node_id: idea:f35-iter2-code
title: "低SNR alpha天花板门控 [i2]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter2"]
---

# 低SNR alpha天花板门控 [i2]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 2, channel=code_innovation).

## Thesis
低SNR alpha天花板门控（MVP）：config加per-bucket alpha_ceiling（0/5dB桶≈0.05-0.1强制近SDMA，其余桶=1.0无操作），rsma_env._parse_action中alpha=np.minimum(alpha, ceiling(当前SNR桶))。只做这一处门控，不做采样倾斜/课程改动

Full verdict: score [4, 0.8951, -12.8386, 0.6899] 未超过 best [5, 0.9163, -11.0186, 0.6914]，2 轮验证失败 (rounds_tried=2)

## Key risks
Failure: score did not beat best (2 rounds).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

