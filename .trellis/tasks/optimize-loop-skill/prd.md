# PRD: SA-GAT PPO 自动优化循环 Skill

## 需求背景

RSMA-EE 项目的训练-评估-优化循环高度重复且手动化：调参 → 训练(数小时) → SNR sweep 评估 → 分析结果 → 再调参。项目已迭代 v12~v22 共 40+ 个 Task，每次迭代需要人工判断瓶颈、决定调参方向、手动执行命令。

本 Skill 的目标是将这一循环完全自动化：启动后无人值守运行，由 LLM（通过 Qoder CLI）驱动决策，自动完成参数调整、创新方法引入、训练、评估，直至达到性能目标。

## 核心研究约束（不可改变）

- **系统模型**: MIMO-RSMA 上行链路, K=24 用户, N_r=16 天线, RZF 预编码, MMSE-SIC 接收
- **核心架构**: 分层 PPO — SA-GAT Worker (Self-Attention + FiLM + P3 直通) + Meta-Controller (Group-level alpha)
- **动作空间**: 4K=96 维 (alpha, p_total, d_ratio, Plackett-Luce 解码顺序)
- **奖励结构**: LDPTRLQ (Lyapunov drift + penalty + EE reward)
- **环境物理**: SNR[-5,35]dB, imperfect CSI, 3 类 QoS, sigmoid 到达率
- **评估标准**: SNR sweep 7 点 (0,5,10,15,20,25,30 dB)

## 终止条件（验收标准）

全部 7 个 SNR 点同时满足：
- `delivery_ratio >= 0.98`
- `mean_queue < 10`
- `mean(EE) > mean(SDMA_EE) * 1.05` （全域平均 EE 优于 SDMA 至少 5%）
- `mean(power) < mean(SDMA_power)` （平均功率不高于 SDMA）

## 安全边界

| 参数 | 上限 | 触发动作 |
|------|------|---------|
| 最大迭代次数 | 30 轮 | 保存状态 + 生成诊断报告 + 退出 |
| 连续无改善轮数 | 10 轮 | 同上 |
| 总运行时间 | 48 小时 | 同上 |

## 设计决策摘要

| # | 决策 | 结论 |
|---|------|------|
| 1 | 决策权 | LLM 驱动 (Qoder CLI) + 约束护栏 |
| 2 | 迭代粒度 | 快速验证 loop (500~1000 episodes) |
| 3 | 代码可变范围 | 奖励函数、Lagrangian、PPO训练器、训练循环 |
| 4 | 创新引入 | 瓶颈时自动搜索 IEEE/GitHub 择优集成 |
| 5 | 版本管理 | 每轮 Git commit + 指标摘要 |
| 6 | 模式升级 | 3轮无改善 OR delivery<0.90 OR EE<SDMA → 创新搜索 |
| 7 | 运行模式 | 完全无人值守自动循环 |
| 8 | 执行引擎 | Shell 编排 + Qoder CLI |
| 9 | 起始点 | v13m_fix (src-v12/logs/snr_aware_v13m_fix/checkpoints/best.pt) |
| 10 | 每轮上下文 | ~800行 (指标+gap+趋势+历史+代码片段+训练日志) |

## 代码可修改白名单

- `src/env/rsma_env.py` — 仅奖励计算部分
- `src/training/lagrangian.py` — 完整文件
- `src/training/ppo.py` — 完整文件
- `src/training/main.py` — 训练循环
- `src/config.py` — 超参数

## 不可修改

- `src/env/rsma_env.py` — 信道模型、速率计算、step 动力学
- `src/models/actor_critic.py` — SA-GAT 架构
- `src/models/meta_controller.py` — Meta 架构
- `scripts/evaluate_snr_sweep.py` — 评估标准

## 交付物

1. `.trellis/tasks/optimize-loop-skill/prd.md` — 本文档
2. `.qoder/skills/optimize-loop/SKILL.md` — Skill 定义
3. `scripts/optimize_loop_config.yaml` — 配置文件
4. `scripts/optimize_loop.py` — 编排脚本（主引擎）
5. `logs/optimize_loop_report.md` — 最终报告（运行时生成）

## 模式升级触发条件

- **轻量模式 → 创新搜索模式**: 连续 3 轮无改善 OR delivery < 0.90 OR EE 全域低于 SDMA
- **创新搜索内容**: 通过 Qoder CLI 调用 web search 搜索 IEEE Xplore / GitHub 上的 constrained RL / safe RL / Lyapunov-based 方法
- **引入标准**: 方法与当前 SA-GAT PPO 框架兼容，不改变问题定义

## 执行流程

```
启动 → 加载v13m_fix → 评估基线 → [循环开始]
  → 训练(1000eps) → SNR sweep评估 → 检查终止条件
  → [未达标] → Qoder CLI分析 → 调参/引入创新 → Git commit → [循环继续]
  → [达标] → 生成报告 → 退出
  → [安全边界] → 生成诊断报告 → 退出
```
