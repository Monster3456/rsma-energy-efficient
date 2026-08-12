# Handoff: RSMA PPO 能效优化 — LDPTRLQ 实施计划

## Goal & Done-State

**目标：** 将当前 PPO 奖励函数改造为 LDPTRLQ 结构（Xu et al., arXiv:2506.04291, 2025），解决 30dB Q=20-29 的问题。

**Done-state：** PPO 在 0-30dB 全范围 Delivery≥98%，Q≤SDMA 水平，EE 显著优于基线。

---

## Current State

### ✅ 已完成
- v13m_fix 确认为当前最优基线（5-20dB EE优势 30-65%）
- 评估脚本全部修复（Meta-Controller 正确加载、sys.path 修正、group_level_alpha 分离）
- 方法调研完成：确认 LDPTRLQ（arXiv:2506.04291）为最佳方案
- v13r（队列加权 EE, ee_queue_alpha=1.0）已训练完成：5-20dB Delivery↑、Q↓，但 30dB Q↑
- 失败训练产物已清理（v13m/n/o/p/q）

### ❌ 未解决
- **30dB Q=20-29**：PPO 尾集稳态队列无惩罚（仅 drift≈ΔQ≈ 0 时惩罚消失）
- **0dB Delivery=0.94**：PPO 功率效率低于 SDMA
- **Meta alpha collapse**：30dB 下 alpha=0.1（远用户无法受益于公共流）

---

## Next Steps（LDPTRLQ 实施）

### Step 1: 修改 `src-v12/src/optimization/lyapunov.py`

将 `compute_reward` 中的奖励计算从：
```python
# 当前（blend-weight 切换）
reward = scale * (w * drift_signal + (1-w) * ee_bonus - overflow)
```
改为 LDPTRLQ 结构：
```python
# LDPTRLQ: 惩罚队列存在性（而非仅增长）
Q_current = Q  # 当前队列
Q_next = Q + arrivals - served  # 下一时隙队列（可近似）
Q_penalty = 0.5 * np.sum(Q_next**2 + Q_current**2) / (Q_max**2)  # 归一化
reward = scale * (V * ee_bonus - Q_penalty - overflow)
```

关键参数：
- `V`：控制 EE vs Q 权衡（建议从 3.0 开始，小于当前 ee_weight=6.0）
- `Q_max=100`：归一化保持不变
- 去掉 blend_alpha/w_min 等动态切换逻辑

### Step 2: 配置 config.py
- 新增 `ldptrlq_V: float = 3.0`
- 保留 `ee_weight=6.0`（作为 V 的组成部分）
- 保留 Lagrangian 作为 delivery 安全网
- `ee_queue_alpha` 回退为 0（LDPTRLQ 的 Q² 已包含队列感知）

### Step 3: 从 v13m_fix resume 训练
```bash
python3 scripts/train.py --episodes 47000 --exp_name snr_aware_v14_ldptrlq \
  --resume logs/snr_aware_v13m_fix/checkpoints/best.pt \
  --resume-episode-count 4500 --delivery-threshold 0.98 --lagrangian-max 100
```

### Step 4: 评估
```bash
python3 scripts/evaluate_snr_sweep.py \
  --checkpoint logs/snr_aware_v14_ldptrlq/checkpoints/best.pt --n_episodes 50
```

---

## Decisions Made

| 决策 | 理由 |
|------|------|
| 采用 LDPTRLQ 方法 | 惩罚队列存在性（ΣQ²）而非仅增长（ΣQ·ΔQ），解决30dB Q稳态高问题 |
| V 参数初始 3.0 | 从保守开始，V越大越偏 EE，V越小越保护队列 |
| 保留 Lagrangian | 作为 delivery 补充安全网 |
| 从 v13m_fix resume | 保留已学到的好策略 |
| 去掉 blend_alpha 动态切换 | LDPTRLQ 的固定 V 替代 |

---

## Abandoned Paths

| 尝试 | 结果 | 原因 |
|------|------|------|
| blend_q_ref=10 | 训练卡死/劣化 | 奖励信号突变 |
| p_floor=0.5 | 无效 | boost×p_base > p_floor |
| SNR[-5,35] 训练 | 劣于 v13m_fix | 训练不足+分布变化 |
| 从零训 8k/20k episodes | 全劣于 v13m_fix | 无法复制40k+的策略积累 |
| ee_queue_alpha=1.0 | 5-20dB↑，30dB↓ | 高SNR所有用户Q都高时加权失效 |
| 奖励修改(v13h/i/j/k) | 全失败 | 标量奖励无法传递逐用户信息 |

---

## Key Locations

| 内容 | 路径 |
|------|------|
| 最优 checkpoint | `logs/v13m_fix_baseline.pt` |
| 奖励函数（需改造） | `src-v12/src/optimization/lyapunov.py` |
| 配置 | `src-v12/src/config.py` |
| 训练脚本 | `src-v12/scripts/train.py` |
| 评估脚本 | `src-v12/scripts/evaluate_snr_sweep.py` |
| LDPTRLQ 论文 | https://arxiv.org/html/2506.04291v1 |
| 调研报告 | `RESEARCH_EXECUTIVE_SUMMARY.md` |

---

## LDPTRLQ 核心公式（论文 Eq.38）

```
R(t) = -½ Σ(Q²ₖ(t+1) + Q²ₖ(t)) - V·p(t)
```

我们的场景（EE 最大化，p(t) = -EE）：
```
R(t) = V × EE(t) - ½ Σ(Q²ₖ(t+1) + Q²ₖ(t)) / Q_max²
```

**与当前方法的核心区别：**
- 当前：仅惩罚队列增长（Q=20但稳定 → 惩罚=0）
- LDPTRLQ：惩罚队列存在（Q=20 → 惩罚=200/用户，强迫降低）

---

## Single Starting Action

**修改 `src-v12/src/optimization/lyapunov.py` 的 `compute_reward` 方法，将 blend-weight 机制替换为 LDPTRLQ 的 `V×EE - ½Σ(Q²(t+1)+Q²(t))/Q_max²` 结构。**

## Goal & Done-State

**目标：** 在 RSMA 上行 MIMO 系统中，通过分层 PPO（Meta-Controller + Worker）实现 Delivery≥98% 约束下的能效（EE）最大化，全 SNR 范围内 Q 可控。

**当前最优模型：** `logs/v13m_fix_baseline.pt`

**Done-state：** PPO 在 0-30dB 全范围内 Delivery≥98%，Q≤SDMA 水平，EE 显著优于基线。

---

## Current State

### v13m_fix 评估结果（当前最优）

| SNR | PPO P | PPO Q | PPO Deliv | PPO EE | SDMA P | SDMA Q | SDMA Deliv | SDMA EE | PPO EE优势 |
|-----|-------|-------|-----------|--------|--------|--------|------------|---------|-----------|
| 0 | 12.40 | 1.38 | 0.948 | — | 9.42 | 0.71 | 0.979 | — | 劣势 |
| 5 | 6.21 | 0.29 | 0.989 | 0.77 | 5.37 | 0.23 | 0.994 | 0.52 | **+48%** |
| 10 | 6.71 | 0.59 | 0.990 | 1.28 | 9.00 | 0.47 | 0.995 | 0.90 | **+42%** |
| 15 | 6.99 | 1.65 | 0.990 | 2.05 | 14.98 | 1.12 | 0.995 | 1.24 | **+65%** |
| 20 | 10.49 | 3.27 | 0.990 | 2.10 | 17.37 | 2.17 | 0.995 | 1.62 | **+30%** |
| 25 | 14.69 | 8.04 | 0.986 | 2.13 | 18.56 | 3.41 | 0.995 | 1.91 | **+12%** |
| 30 | 17.78 | 20.92 | 0.951 | 2.17 | 19.34 | 5.15 | 0.994 | 2.09 | +4% |

### 核心优势区（5-20dB）
- EE 提升 30-65%
- Delivery ≥ 98.9%
- 功耗降低 16-53%

### 待解决问题
1. **0dB**: PPO Delivery=0.948（<0.98），功耗反超 SDMA
2. **25-30dB**: Q 爆炸（8.04/20.92 vs SDMA 3.41/5.15）
3. **Meta-Controller collapse**: 30dB 下 alpha 恒为 0.1（最小值），远用户无法受益于公共流

---

## Decisions Made

| 决策 | 理由 |
|------|------|
| PPO 后处理：gamma=2 阈值触发 | Lyapunov L=½Q² 数学一致 |
| Q_thresh=3, boost_max=5 | PPO 主导，安全网仅 Q>3 触发 |
| delivery_threshold=0.98 | 用户硬约束 |
| SNR 训练范围 [-5,35]dB | 解决边界效应（评估 [0,30]） |
| 对照实验：PPO 与 SDMA 完全相同到达率 | 公平性，不可调 load_ratio 偏袒 PPO |
| Worker+Meta 必须联合优化 | 用户核心偏好 |

---

## Abandoned Paths

| 尝试 | 结果 | 原因 |
|------|------|------|
| blend_q_ref=10（急剧改变 blend 归一化） | 训练卡死/从零训练劣化 | 奖励信号突变导致 value function 崩溃 |
| p_floor=0.5（功率下限） | 无效 | boost×p_base > p_floor，下限从未触发 |
| v13o 从零训 8000ep（SNR[-5,35]+blend_q_ref=10） | 全面劣于 v13m_fix | 训练不足+奖励不稳 |
| v13p resume+SNR[-5,35]（5000ep） | 劣化 | SNR 扩展破坏已有策略 |
| v13q 从零训 20000ep（SNR[-5,35]） | 仍劣于 v13m_fix | 从零训练需要更多 episode |
| v13n p_floor=0.5 重训 | Q 更高 | PPO 学会依赖后处理，道德风险 |
| 奖励修改（v13h/i/j/k 的 max_queue penalty 等） | 全失败 | 标量奖励无法传递逐用户队列信息 |

---

## Open Questions / Risks

1. **30dB Q=20.92 根因**：PPO 在 130% 过载下省电太多（P=17.78 vs SDMA 19.34，差距仅 8%），但 Q 差 4 倍。真正原因是 Meta alpha=0.1 导致远用户私有流不可用。
2. **Meta 为何 collapse**：训练 SNR 均匀分布，30dB 仅占 ~2.5%。Meta 学到低 alpha=最优 EE（对 97% 训练数据正确）。
3. **能否在不牺牲 5-20dB EE 的前提下修复 30dB**：当前架构本质局限——需要方法论创新。

---

## Next Steps（下一步：IEEE/GitHub 调研）

### 推荐调研方向

1. **Multi-timescale Lyapunov + DRL**：
   - Lyapunov-guided safe RL（constrained RL with queue stability guarantee）
   - 论文关键词：`Lyapunov drift penalty` + `deep reinforcement learning` + `energy efficiency`

2. **RSMA + DRL 功率控制**：
   - 最新 RSMA 功率分配 DRL 方法（2023-2024）
   - 关键词：`RSMA` + `deep reinforcement learning` + `power control` + `uplink`

3. **Queue-aware power allocation in wireless**：
   - 信道-队列联合感知的功率分配
   - Whittle Index + DRL、Max-Weight scheduling + RL

4. **Safe/Constrained RL for communication**：
   - Lagrangian PPO with hard constraints
   - CPO (Constrained Policy Optimization) / FOCOPS / CUP

5. **Multi-agent / Hierarchical RL for resource allocation**：
   - 分层 MARL 做 RSMA
   - Option framework / feudal RL for multi-timescale

---

## Key Locations

| 内容 | 路径 |
|------|------|
| 最优 checkpoint | `logs/v13m_fix_baseline.pt` |
| 评估结果 | `logs/eval_v13m_fix_final.log` |
| 训练脚本 | `src-v12/scripts/train.py` |
| 评估脚本（已修复） | `src-v12/scripts/evaluate_snr_sweep.py` |
| 环境核心 | `src-v12/src/env/rsma_env.py` |
| 奖励函数 | `src-v12/src/optimization/lyapunov.py` |
| 配置 | `src-v12/src/config.py` |
| Worker 模型 | `src-v12/src/models/actor_critic.py` |
| Meta 模型 | `src-v12/src/models/meta_controller.py` |

---

## Single Starting Action

开始 IEEE Xplore + GitHub 调研，搜索：
1. `"RSMA" AND "reinforcement learning" AND "energy efficiency"` (2023-2025)
2. `"queue stability" AND "deep reinforcement learning" AND "power allocation"`
3. `"Lyapunov" AND "PPO" AND "wireless"`

找到 3-5 篇最相关的方法论文献，分析其如何解决 EE-Queue 权衡问题。
