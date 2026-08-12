# Handoff: 实施方案A — Lyapunov-引导 DRL

## Goal & Done-State

将当前 PPO 框架改造为 **Lyapunov-引导 DRL**（参考 Bi et al., IEEE TWC 2021），使 delivery ≥ 98% 且 EE 优于 SDMA。完成标志：SNR Sweep 评估中全 SNR 点 delivery ≥ SDMA，功率 < SDMA。

## Current State

### ✅ 已完成
- v18 基线恢复确认：SNR 5-10dB delivery 98%，20-30dB 超越 SDMA，功率低 3-24%
- 评估脚本 Meta-Controller bug 已修复（`evaluate_snr_sweep.py` 正确运行 Meta）
- split_mask 初始化已改为 0.5（避免 alpha=1.0 灾难）
- delivery_ratio 空队列返回 0.5（避免 Lagrangian 虚假满足）
- lagrangian_decay=0.999（加速衰减）
- w_max 已恢复为 1.0，ee_power_floor 已恢复为 3.0
- Alpha=0 验证完成：证明公共流不是问题，根因是 Worker 功率分配效率

### ❌ 未解决
- SNR=0dB delivery 仅 78.8%（SDMA 92.5%）——**核心瓶颈**
- Lagrangian 软约束无法突破 92-93% delivery 天花板（v19/v21_stage1 均验证失败）
- PPO 功率分配效率低于 SDMA 的队列加权启发式

### 🚫 已放弃的方案
- **w_max=0.8**：v20 证明导致 delivery 全面下降（71-90%），EE 过度主导
- **ee_power_floor=1.0**：导致低功率区间梯度过强，模型过度省电
- **纯 Lagrangian 加大 λ_max=500**：v21_stage1 证明 λ=48 仍无法突破 93%
- **CPO/TRPO 替换**：改动太大，风险高

## Decisions Made

1. **选择方案 A（Lyapunov-引导 DRL）** — 用户确认
2. **参考论文**：Bi et al., "Lyapunov-Guided Deep Reinforcement Learning for Stable Online Computation Offloading in MEC Networks", IEEE TWC 2021
3. **参考代码**：`revenol/LyDROO`（GitHub 2.2k stars）
4. **核心改造思路**：`reward = V × EE_bonus - Lyapunov_drift`，其中 V 是权衡参数
   - 不再用 blend weight w 切换 EE/drift
   - Lyapunov 漂移**直接作为约束**嵌入 reward，而非作为"队列满时的切换信号"
   - V 参数控制 EE 和队列稳定性的权衡（V 越大越偏 EE，V 越小越保守）

## Next Steps（按顺序）

### Step 1: 阅读 LyDROO 源码 + Bi et al. 论文
- Clone: `git clone https://github.com/revenol/LyDROO`
- 重点阅读论文 Section II-III（Lyapunov 理论 + DRL 融合）
- 理解 `V × utility - drift` 的 reward 结构

### Step 2: 改造 lyapunov.py 的 compute_reward

当前结构：
```python
reward = scale * (w * drift_signal + (1-w) * ee_bonus - overflow_penalty)
```

改造为 Lyapunov-引导结构：
```python
# V 是权衡参数（可训练或固定）
# drift_penalty = sum(Q_k * (Q_k_new - Q_k_old)) / Q_max^2  [正值表示队列恶化]
# EE_bonus = ee_weight * served_total / P_consume

reward = scale * (V * ee_bonus - drift_penalty - overflow_penalty)
```

关键区别：
- **去掉 blend weight w**：不再根据队列比例切换目标
- **drift 作为惩罚项**：始终存在，与 V×EE 做减法权衡
- **V 参数**：控制"多大程度追求 EE vs 保守稳定队列"
  - V 大 → 更激进追求 EE（可能牺牲 delivery）
  - V 小 → 更保守保护队列（delivery 高但 EE 低）
  - 推荐初始 V=5.0，通过实验调优

### Step 3: 配置与训练
```bash
# config.py 修改：
# ee_weight = 8.0（保持）
# 新增：lyapunov_V = 5.0（权衡参数）
# 移除或禁用 blend_alpha 机制
# 保持 Lagrangian 作为额外安全网（delivery_threshold=0.98, lagrangian_max=100）

python3 scripts/train.py --exp_name snr_aware_v22_lyapunov \
  --episodes 56384 \
  --resume logs/snr_aware_v18_from_v15/checkpoints/best.pt
```

### Step 4: 评估
```bash
python3 scripts/evaluate_snr_sweep.py \
  --checkpoint logs/snr_aware_v22_lyapunov/checkpoints/best.pt \
  --n_episodes 50
```

## Key Locations

| 文件 | 用途 |
|------|------|
| `src/optimization/lyapunov.py` | 核心 reward 计算（需改造） |
| `src/config.py` | 训练配置 |
| `src/training/main.py` | 训练循环 + Lagrangian |
| `src/env/rsma_env.py` | 环境 step |
| `scripts/evaluate_snr_sweep.py` | 评估脚本（已修复 Meta） |
| `logs/snr_aware_v18_from_v15/checkpoints/best.pt` | 最佳起点 checkpoint |
| `RESEARCH_INVESTIGATION_REPORT.md` | 完整调研报告 |
| `RESEARCH_QUICK_REFERENCE.md` | 快速实施指南 |

## Open Questions / Risks

- V 参数初始值如何选？论文建议从小值开始逐步增大
- 去掉 blend_alpha 后，低 SNR 时队列保护是否足够？（drift_penalty 需要足够大的权重）
- 是否需要同时保留 Lagrangian 作为双重保障？（建议保留，作为 safety net）
- 从 v18 resume 是否兼容新 reward 结构？（架构不变，只改 reward 计算，应该可以）

## Single Recommended First Action

**读完 RESEARCH_QUICK_REFERENCE.md 的"方案A实施"部分，然后修改 `src/optimization/lyapunov.py` 中的 `compute_reward` 方法，将 blend-weight 机制替换为 `V × EE - drift` 结构。**
