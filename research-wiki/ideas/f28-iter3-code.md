---
type: idea
node_id: idea:f28-iter3-code
title: "分位数critic [i3]"
stage: archived
outcome: negative
added: 2026-09-03T04:33:47Z
based_on: []
target_gaps: []
tags: ["failed-direction", "code_innovation", "banlist", "iter3"]
---

# 分位数critic [i3]

**stage:** `archived`  ·  **outcome:** `negative`

Falsified optimize_loop direction (iter 3, channel=code_innovation).

## Thesis
分位数critic（DisPPO式分布价值表征）：models/将critic末层1→16维non-crossing分位数头（用现有value头初始化使全部分位数=当前V预测，step-0与best行为一致），training/价值损失换quantile Huber(κ=1,τ_i=(2i−1)/32)、advantage取分位数均值（GAE流程不变），消除7个SNR点多峰混合回报下均值criti

Full verdict: 快速测试门禁失败 (6 用例): tests/test_training.py::TestActorCritic::test_forward_shape; tests/test_training.py::TestActorCritic::test_evaluate_actions_shape; tests/test_training.py::TestActorCritic::test_log_prob_finite; tests/test_training.py::TestActorCriticPL::test_forward_shape_pl; tests/test_training.py::TestActorCriticPL::test_evaluate_actions_pl (rounds_tried=0)

## Key risks
Failure: fast-gate test failure.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

