# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

上行 RSMA-MU-MIMO **多时间尺度在线学习与资源分配**研究（K=24 用户 > N_r=16 天线的**过载场景**），采用分层强化学习框架：

- **慢时间尺度**（Meta-Controller, Self-Attention PPO，每 T_s=20 时隙）：用户分组 g_k ∈ {0,...,G-1}（G=4, Categorical）+ **组级连续功率分割比 α**（Task#38 `group_level_alpha=True`：G 维 Gaussian → sigmoid 映射到 [0.1, 0.95]，按分组展开为 per-user α，**覆盖** Worker 的 α 输出块）
- **快时间尺度**（Worker, **SA-GAT Self-Attention PPO**，每时隙）：功率分配（softmax 份额）、drain 比例（d_ratio）、解码顺序（Plackett-Luce）；action = 4K = 96 维

奖励为 **DGLCR**（Delivery-Gated Lyapunov Composite Reward）+ **Dinkelbach 线性化 EE 目标**，约束层为 Lagrangian delivery 硬约束（≥0.98）+ 交付虚拟队列（DVQ）。实验由 `scripts/optimize_loop.py` 半自主驱动，跨轮知识沉淀在 `logs/opt_knowledge.json`。

**当前最优成绩（最终证据，2026-09-11）**：字典序 score `[7, 0.9519, -8.3971, 1.0804]` = **7/7 评估点全达标**，交付物 `logs/m5_armA_sftp/checkpoints/best_guard.pt`。seed 43 逐字复跑在 ±0.002 内复现（0.9503），final 0.9491–0.9496，**σ_seed≈0.005**；M5.1–M5.3 五类干预（释放 lr / KL 信任域 / 强锚定 / 采样分布）全部 0.946–0.9474，落在冠军噪声带内——无处理效应。论文证据包见 `docs/paper-results.md`；判定/预注册归档见 `.trellis/tasks/09-02-solver-expert/experiment-results.md`。

### 成功判据（评估口径，务必先理解再改代码）

- **评估点**：7 个 SNR 点 {0,5,10,15,20,25,30} dB（训练范围 [-5,35] dB），imperfect CSI
- **逐点达标门**（仅两条相对判据）：delivery ≥ SDMA×1.0 **且** queue ≤ SDMA×1.1
- **字典序 score** = `[points_met, avg_delivery, -avg_queue, ee_ratio]`，`points_met` 绝对优先（`optimize_loop.py` 的 `compute_score`）
- **全局终止判据**（`check_termination`）：mean EE > SDMA×1.05 且 mean power < SDMA——不是逐点门，但违反会终止优化轮
- **陷阱**：抬 ee_ratio 但掉 points_met 是净损失；先想清楚 points_met

## 硬性护栏（违反会回滚整轮实验或破坏可比性）

1. **不可变文件（零容忍，纯格式 diff 也会触发 optimize_loop `code_guard` 拦截并回滚）**：
   - `src/env/channel.py`、`src/env/rate_calc.py`、`scripts/evaluate_snr_sweep.py`
   - `src/env/rsma_env.py` 中的物理调用点（`self.channel` / `self.rate_calc`）；非物理部分（reward shaping、观测、课程钩子）可改
   - 权威清单 = `scripts/optimize_loop_config.yaml`（`immutable` + `code_guard.immutable_paths`）；`ruff.toml` 已 force-exclude 同样文件，**永远不要对它们跑 ruff format**
2. **提出任何新优化方向前必须先读 `logs/opt_knowledge.json`**（跨轮失败/成功方向库，版本跟踪）与 `RESEARCH_BRIEF.md` 的 Non-Goals banlist（36+ 已证伪机制族：Lagrangian/PID 微调、CRPO/梯度投影变体、模型汤、分位数 critic、全局 group_alpha 常数等）
3. **不要自主重启 `optimize_loop.py` 或发起全量训练/评估**——实验运行由用户主导；研究 wiki / idea 层只做选择与知识沉淀
4. **`scripts/train.py` 第 4 行硬编码作者 macOS 绝对路径** → 直接运行报 `ModuleNotFoundError: No module named 'src'`，必须 `PYTHONPATH=.` 前缀（`experiments/v2/` 同理）
5. **任何代码改动后先跑快速门禁**：`.venv/Scripts/python.exe -m pytest -m fast -q tests/`（fast 子集 322 个，全量 324 个测试数秒内完成，无借口跳过）

## 系统模型（当前代码实际；与旧文档冲突时以代码为准）

### 网络与信道

- K=24 用户、N_r=16 BS 天线、N_t=2 用户天线、T=200 时隙/episode、P_k_max=1.0W（总上限 24W）、Q_max=100 bits
- **3GPP UMi 路径损耗（Task#13）**：用户距离 50–300m 每 episode 采样（episode 内固定），以 100m 处相对 PL=0 归一化，保留 ~29dB 近远差驱动差异化功率/SIC
- AR(1) 时间相关 ρ=0.9；**SNR 依赖 CSI 估计噪声（Task#20）**：sigma_est² = σ²/(K·pilot_power) + epsilon_floor²（epsilon_floor=0.01）；worst-case 有界误差 H = Ĥ + ΔH, ‖Δh_k‖ ≤ ε
- **SNR Domain Randomization**：每 episode 在 [-5, 35] dB 重采 σ²；实际训练 SNR 由硬例挖掘器逐 episode 采样（见训练流程）
- **现实功耗模型（Task#15）**：P_consume = P_tx/pa_efficiency + circuit_power = P_tx/0.35 + 5.0W——根除 P→0 时 EE→∞ 的“关机捷径”

### 预编码 / 接收机（旧版 CLAUDE.md 的 “RZF 待集成” 已过时）

- `precoding_type="rzf"`（**默认**）= 统一 MRT 发射 + **MMSE-SIC 接收**：`rate_calc.compute_rates_mmse_sic / compute_outage_rates_mmse_sic`（设计文档 `docs/superpowers/specs/2026-06-05-mmse-sic-integration-design.md`）
- `"mrt"` = 旧正交补 MRT + matched filter 路径（向后兼容）
- outage 模式 = min(declared[Ĥ,ε], actual[H_true,ε=0])
- β (`sic_imperfection`，现为 SystemConfig 字段，默认 0.0) 与 ε (CSI 误差界) 是**独立物理参数**：β = SIC 硬件残余，ε = 信道估计误差；SINR 分母 = 未解码干扰 + SIC 残余（β² 项 + ε² 项）+ 热噪声。公式权威 = 不可变的 `rate_calc.py`

### RSMA 分割与动作语义

- **group_level_alpha=True（默认）**：Meta 的组级 α 展开为 per-user 连续值写入 `env.split_mask`，`_parse_action` 中直接**覆盖** Worker 的 α 块 → **Worker 的 α 输出是死维**
- **softmax 功率份额（Mean-Field MAPPO，`use_softmax_power_share=True`）**：Worker p 块是 K 个 logits，p_k = K·softmax(logits) 再 clip [0,1]——总功率由 logit 展宽控制（均匀=24W，极端展宽可压到 ~8.8W）
- d_ratio ∈ [0,1]（drain 比例）；解码顺序 π 由 z-scores 经 softplus + Gumbel-Top-K 采样（Plackett-Luce，log P(π|z) 完全可微，温度 pl_temperature=1.0）

### 流量与队列

- **SNR 自适应到达率**：λ(SNR) = load_ratio(snr_db)·capacity(SNR)/K，load_ratio 为 sigmoid（0.75@低SNR → 1.3@高SNR，中点 15dB，陡度 0.3）；`lambda_rate=5.5` 为旧回退值（adaptive_lambda=True 默认）
- **异构 QoS（v13，默认开）**：3 类用户，λ 倍率 (2.0,1.0,0.3)、drift 权重 (8.0,1.0,0.2)、overflow 权重 (8.0,1.0,0.2)
- 队列演化 Q[t+1] = min{max{Q−R,0}+A, Q_max}；**per-user DVQ CVaR 尾部聚焦**：仅惩罚虚拟积压 worst-25% 用户（dvq_threshold=0.98）

## MDP 建模

### 观测（`rsma_env._get_observation`）

```
基础 obs = 11K + 3：
  11 个 per-user K 维块: h_gains, queue, Δqueue, arrivals, power_history,
    rate_history, interference*, fairness*, split_mask(=per-user α), group_ids_norm, distance_norm*
  3 个 global: time_ratio, snr_norm, eps_norm
+2K  (enable_hetero_qos=True 默认): qos_lambda_load, qos_priority   → obs_dim = 13K+3 = 315
  **2026-10-12 A-prune 转正：`obs_prune_constant_blocks` 默认 True**——移除固定 SNR 点内
  零方差的 3 个原生块（arrivals/λ桶/Dinkelbach-η，块 3/6/10）→ 默认 obs_dim = 10K+3 = 243
  （champion H=4 未剪 603 为论文遗留口径：训练/评估显式 `obs_prune_constant_blocks=False`）
+H·3K (obs_history_window=H>0): 最近 H 槽的 (h_gains, queue, power) 快照，最旧在前，
     拼在 per-user 区之后、global 尾之前（时序 Worker 实验用 H=4，策略保持无状态）
+K   (obs_include_csi_quality=True 可选): per-user 估计不确定度
```

- **\*State-Augmented 等维替换（`state_aug_enabled=True` 默认）**：interference / fairness / distance 三块被**原地替换**为归一化的 Lagrangian λ 桶值 / DVQ 积压 Y / Dinkelbach q*——obs_dim 不变
- `hierarchical.enabled` 的 power_budget +1 维通道处于休眠（HierarchicalConfig 无该字段，getattr 恒 False）
- 观测不含原始复信道 H，只有聚合统计

### 动作

- env `action_dim = 3K = 72`，`step()` 按 shape 自动判别 3K/4K；训练入口 `src/training/main.py:499` 硬编码 `worker_action_dim = 4*K = 96`（3K 连续 + K 解码序）
- Worker 输出：α 块（group_level_alpha 下为死维）、p logits（softmax 份额）、d_ratio、z_scores（PL）

### Worker 奖励（`lyapunov.py compute_reward`，默认活跃分支 = DGLCR + Dinkelbach）

```
P_consume = ΣP_tx/0.35 + 5.0;  P_eff = max(P_consume, ee_power_floor=3.0)
Dinkelbach 线性化:  ee_bonus = ee_weight(8.0) · (ΣD − q*·P_eff)     # D=served_bits，非物理层 R_sum
  q* 由训练外层每 10 episode 更新（dinkelbach_outer_update=True），clip [0.01, 50]
  use_dinkelbach=True 强制 ee_gate = penalty_gate = 1（绕过 sigmoid 门控）
reward = reward_scale(0.01) · [ lyapunov_V(5.0)·ee_bonus
                              + tp_boost·w_tp_dglcr(3.0)·Σ(Q_k/Q_max·w_qos·D_k)   # max-weight 吞吐
                              + w_y_tp(0.3)·(Y/300)·ΣD                            # Y=聚合交付欠账 VQ (target 0.95)
                              − ldptrlq_q_weight(2.0)·ΣQ²/Q_max²                  # 队列二次惩罚
                              − overflow_penalty                                  # w_overflow(20)·Σw_qos·OV + 早期渐进警告
                              − w_power_vq(1.0)·(X/100)·P_total ]                 # X=功率预算 VQ (budget 8W)
```

- 非 Dinkelbach 时的 **DGLCR sigmoid 门控**：ee_gate(κ=40, mid=0.935) 门控 EE 项、penalty_gate(κ=50, mid=0.96) 抑制惩罚，tp_boost = 1+4·(1−ee_gate)——根治 ∂EE/∂P<0 低功率陷阱
- **备用分支**：CRPO 硬切换（可行=EE / 不可行=纯 max-weight，迟滞带 0.02）；目标反转回退（交付凹效用 Σ√(D_k/2) 主目标 + EE 下界软约束 + backpressure w=25 + PF 公平效用 w_fair=6）
- ⚠️ **死字段**：`RewardConfig.w_outage`（LyapunovController 接收但未使用）、`w_power_exceed`（lyapunov.py 中不存在）；`ee_sat_cap` 由 rsma_env 消费
- reward_v2（逐用户 served 加权 + q* 冻结）、DGMW、LDPS 等实验开关默认关，注释标明“已验证无效”

### Meta 奖励（`meta_reward.py`）

```
meta_reward = avg_ee − 2.0·avg_overflow + 1.0·group_diversity + 0.5·group_balance
              − alpha_guidance_scale·(alpha_mean − target_alpha)²    # Task#36 SNR 条件 α 引导
target_alpha = clip(0.9 − snr_db·0.01, 0.5, 0.95)；引导仅 warmup(300ep) 后激活
最终乘 meta_reward_scale=0.02
```

## 训练流程（src/training/main.py，双层 PPO）

```
episode 开始前: _hm_sample_snr 硬例挖掘采样本 episode SNR（7 评估点粒度，
    按逐点 delivery shortfall 加权 + 连续未达标 streak 倍增 + 均匀地板）
reset env → _set_worker_bucket（按 SNR 硬路由桶专家头）→ _meta_decide（注入 α + g）
loop rollout_steps(2048):
    Worker.act(obs) → 4K 动作 → env.step → buffer.add
    每 T_s=20 步 / episode 结束: 闭合 meta 段 → meta_buffer.store → _meta_decide
    CRPO-S（use_crpo_switch=True）: 桶不可行时整段 episode 切换为排水盈余可行性奖励（迟滞 0.03）
Worker: 每 rollout 后 PPO 更新（clip + GAE + KL 早停 + LR 衰减 + SFL 可学习性加权采样
    + 性能挂钩熵激励：未达标点熵系数按 shortfall 放大）
Meta: 积累 meta_update_threshold=30 段后更新
外层: 每 10 episode 更新 Dinkelbach q*；硬例挖掘状态 EMA 更新
```

### 约束栈（ConstraintConfig，默认开）

- **Lagrangian**：delivery ≥ 0.98 硬约束，对偶乘子 λ（init 5, lr 0.05, max 100, warmup 200ep）
- **SLCP-D**：约束 penalty 输入改为 per-step 即时交付率 EMA（β=0.85），提供动作级差异化梯度
- **per-user DVQ**（Neely 交付虚拟队列）+ **词典序梯度投影**（EE 策略梯度投影到 delivery 约束梯度零空间）
- **已验证无效、默认关**（代码保留，勿重复尝试）：RCPO cost-critic、Delivery Safety Layer、PLPI 功率激励、SPLC 负载课程、HEK-Lite、C3PO

### 训练前干预通道（默认全关 = 官方口径零变化）

- **Solver-Expert Iteration（ExIt，M3）**：`solver_expert_enabled` → `_solver_expert_collect` 收集演示 → `bc_pretrain` 联合训练连续头 + PL 排序头 → 衰减锚定 PPO 微调（复用 HEK 的 BC/anchor 通道）
- CuRLTraC 难度退火课程（`use_curriculum`）、backpressure 先验（`backpressure_beta`，4 种 mode：queue / queue_channel / mw_residual / mw_gated；Run 13 判定 mw_gated 门控学习率饿死，见 temporal-worker experiment-results）

## Worker 网络：SA-GAT（`models/actor_critic.py` ActorCriticAttn，use_attention=True 默认）

替代旧 MLP（LN 稀释 SNR/ε 信号问题；旧 `ActorCritic` 仍保留可回退）：

1. obs 解构为 per-user tokens (K, d_user=13：11 基础 + 2 QoS；历史窗口开启时 d_user_eff = 13 + H·3) + global (3)
2. global → FiLM 生成器对 token embedding 仿射调制；(snr_norm, eps_norm) 另复制注入每个 token（per-user × SNR/ε 交互）
3. MultiHeadSelfAttention（4 头 × 2 层，hidden 128）+ FFN + 残差/LN → per-user heads → (α, p, d, z) 拼 4K
4. critic：mean pooling + global 残差 → V(s)；**Mean-Field 偏置**（QoS 类均值场统计量，mean_field_hidden=64）
5. **桶专家多头**（use_bucket_experts=True，4 头按 SNR 桶 episode 级硬路由，专家头独立 lr=1e-4）
6. backpressure 先验直接加在功率 logits 上（β/mode 注册为 buffer 随 checkpoint 传输，评估侧 strict=False 自动加载）

## Meta 网络（`models/meta_controller.py`）

- embed(meta_d_user=12 → 64) + MHA(4 头) + FFN；三 head：
  - group：per-user G=4 logits Categorical，**温度退火** τ 2.0→0.5，分组平衡正则 0.1
  - α：group_level_alpha=True 时 mean-pool → G 维 Gaussian（log_std init −0.5）→ sigmoid 映射 [0.1,0.95]；False 时退回 per-user Bernoulli split
  - critic：mean pool → 1
- 注入：`env.set_split_mask(per-user α)` + `env.set_group_ids(g)`；Meta buffer FIFO(500)，满 30 段触发 PPO 更新

## Solver Expert + 可微引擎（09-02 任务，M1/M2 已完成）

- **`src/optimization/diff_engine.py`**：rate_calc MMSE-SIC + MRT 预编码的 torch 影子实现（float64 一致性 atol=1e-9，`tests/test_diff_engine.py` 锁定），为专家提供梯度通道
- **`src/optimization/solver_expert.py`**：单槽 DGLCR surrogate 在线精修专家——每时隙对 p_logits 跑 Adam（25 步，lr 0.1，**冷启动**，warm 已证更差）+ 解码序候选择优（gain_desc / queue_last）；α 死维、d_ratio 固定最优；surrogate 精确镜像 lyapunov DGLCR 活跃分支（含 q* EMA / Y 队列“先更新后使用”耦合），常数从 env.lyapunov 实例运行时读取
- **M2 诊断结果**（`scripts/_diag_solver_expert.py`，日志 `logs/_diag_solver_expert_full.log`）：教师在 0/5/15/30dB 全点 delivery 超 SDMA（+4.0/+1.4/+10.0/+23.9pt）且功率更低——首个全点越过 pass 线的教师上界

## 配置参数（src/config.py；完整字段以代码注释为准，大量实验旋钮默认关）

### SystemConfig（要点）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| K / N_r / N_t | 24 / 16 / 2 | 过载场景 K > N_r |
| T / Q_max / P_k_max | 200 / 100.0 / 1.0 | |
| use_path_loss | True | 3GPP UMi，50–300m，exponent 3.76 |
| snr_dependent_csi | True | MMSE 导频估计噪声 + epsilon_floor=0.01 |
| epsilon_csi | 0.05 | 已弃用语义：仅 snr_dependent_csi=False 回退值 |
| precoding_type | "rzf" | 统一 MRT 发射 + MMSE-SIC 接收 |
| sic_imperfection | 0.0 | β，SIC 硬件残余因子 |
| lambda_rate | 5.5 | 旧回退值；adaptive_lambda=True 时走 SNR 自适应 λ |
| adaptive_lambda / load_ratio_min..max | True / 0.75..1.3 | sigmoid(中点 15dB, 陡度 0.3) |
| pa_efficiency / circuit_power | 0.35 / 5.0 | P_consume = P_tx/η + P_c |
| snr_domain_randomization / min / max | True / −5 / 35 | dB |
| use_softmax_power_share | True | p 块 = softmax 份额 logits |
| enable_hetero_qos / qos_classes | True / 3 | λ 倍率 (2,1,0.3)、drift/overflow 权重 (8,1,0.2) |
| use_dvq / dvq_cvar_alpha | True / 0.25 | per-user DVQ，CVaR worst-25% |
| obs_history_window / features | 0 / (h_gains,queue,power) | H>0 启用时序历史观测 |
| state_aug_enabled | True | λ/DVQ/q* 等维替换注入 obs |
| obs_prune_constant_blocks | **True**（A-prune 转正） | 移除固定 SNR 点内零方差常数块（arrivals/λ桶/Dinkelbach-η），10 原生块；ablation 证明 Δ=−0.0001 无处理效应（7/7 门、0.9526 vs 0.9527） |

### LyapunovConfig / RewardConfig（要点）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| V / V_min / V_max | 1.0 / 0.3 / 5.0 | 自适应 V（exponent 1.5） |
| lyapunov_V | 5.0 | DGLCR 中 EE 项权衡 |
| ee_weight / ee_power_floor | 8.0 / 3.0 | Dinkelbach 线性 EE |
| use_dinkelbach / outer_update / interval | True / True / 10 | q* 外层每 10ep 更新 |
| drift_weight | 60.0 | 仅日志（drift_signal） |
| ldptrlq_q_weight | 2.0 | 队列二次惩罚 |
| w_backpressure | 25.0 | 回退分支 backpressure |
| w_deliv / deliv_ref / ee_floor / w_ee_floor | 3.0 / 2.0 / 30.0 / 0.5 | 目标反转回退分支 |
| w_overflow | 20.0 | RewardConfig；+早期渐进警告 (warn 0.6, w 5.0) |
| reward_scale | 0.01 | |
| w_outage / w_power_exceed | 3.0 / 10.0 | ⚠️ 死字段（未接入 compute_reward） |
| dgmw / ldps / reward_v2 | 全关 | 已验证无效或实验性 |

### PPOConfig（要点）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| lr_actor / lr_critic / lr_log_std | 3e-4 / 1e-3 / 1e-3 | |
| gamma / gae_lambda / clip | 0.99 / 0.95 / 0.2 | target_kl=0.02 早停 |
| entropy_coef / value_coef | 0.008 / 0.5 | |
| n_epochs / batch / rollout_steps | 8 / 64 / 2048 | |
| action_dim | 96 | 4K（K=24） |
| use_attention | True | SA-GAT Worker |
| worker_d_user / attn_hidden / heads / layers | 13 / 128 / 4 / 2 | |
| use_mean_field_critic / use_per_user_advantage | True / True | |
| use_bucket_experts / n_bucket_experts / lr | True / 4 / 1e-4 | SNR 桶 [-5,35] |
| hard_mining_enable / hm_granularity | True / "point" | 7 评估点逐点挖掘，_HM_TARGETS 对齐 SDMA |
| entropy_performance_linked | True | 未达标点熵激励 |
| backpressure_beta / mode | 0.0 / "queue" | 关 |
| use_curriculum (CuRLTraC) | False | 关 |
| hidden_dim / n_layers | 256 / 3 | 仅旧 MLP Worker 用 |

### HierarchicalConfig（要点）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| T_s / meta_lr | 20 / 5e-4 | |
| meta_d_user / meta_hidden_dim / n_heads | 12 / 64 / 4 | |
| meta_update_threshold | 30 | |
| n_groups | 4 | |
| group_level_alpha | True | 组级连续 α（α_min/max 0.1/0.95） |
| meta_group_tau_init / tau_min | 2.0 / 0.5 | group head 温度退火 |
| meta_balance_coef / meta_reward_scale | 0.1 / 0.02 | |
| alpha_guidance_enabled / scale / warmup | True / 0.3 / 300 | Task#36 |
| pl_temperature | 1.0 | Worker PL 温度 |

### ConstraintConfig / AblationConfig
- ConstraintConfig 见上文「约束栈」；AblationConfig：disable_split / disable_group / disable_sic_order / disable_meta（Task#11 消融）
- 顶层 Config：seed=42, device="cpu", exp_name="robust_snr"

## 开发命令

> **本机（Windows）Makefile 不可用**：Makefile 硬依赖 Python 3.14 + Unix venv 路径（`.venv/bin/python`），本机 venv 是 Windows Python 3.10（`.venv/Scripts/python.exe`）。日常一律用下列直调命令；Makefile（help/check-python/setup/test/smoke/eval/clean）仅在符合 3.14 的 Unix/CI 环境有效。命令权威另见 `AGENTS.md` §常用命令。

```bash
# 测试（改完代码必须跑；全量 ~5s）
.venv/Scripts/python.exe -m pytest -m fast -q tests/     # 快速门禁（fast 标记）
.venv/Scripts/python.exe -m pytest -q tests/             # 全量（324 个）
.venv/Scripts/python.exe -m pytest tests/test_env.py -v  # 单文件；-k 选单用例

# Lint / 格式化（手动；line-length 120, quote-style preserve）
# ⚠️ 永远不要格式化 ruff.toml exclude 中的 4 个不可变文件
.venv/Scripts/python.exe -m ruff check <file>
.venv/Scripts/python.exe -m ruff format <file>

# 训练
PYTHONPATH=. .venv/Scripts/python.exe scripts/train.py --smoke --exp_name <名>   # 冒烟（PYTHONPATH 必须）
python -m src.training.main --exp_name <名> --total_episodes N --device cpu --csi_mode imperfect
# scripts/train.py 有 ~60 个参数：--obs_history_window/--obs_history_features（时序观测）、
# --solver-expert*（ExIt）、--curriculum/--hard-mining*、--resume/--partial_resume、
# --backpressure_*、--n_r/--fixed_lambda（评估口径对齐）、Lagrangian 旋钮等

# 评估（evaluate_snr_sweep.py 不可变；输出宽表按列位置解析）
python scripts/evaluate_snr_sweep.py --n_episodes 50 --checkpoint logs/<exp>/checkpoints/final.pt
# 时序 (H>0) / 结构变体 checkpoint 必须用 wrapper（否则 obs 维推断错误）：
python scripts/evaluate_snr_sweep_temporal.py --obs_history_window 4 --checkpoint <ckpt> [--n_r --fixed_lambda --no-softmax-power-share ...]
# 剪枝(531 obs)检查点评估须加 --obs-prune-constant-blocks；wrapper 默认关=未剪(603, 冠军兼容)

# 自动优化循环（先读 logs/opt_knowledge.json！）
python scripts/optimize_loop.py --status | --dry-run | --stop | --reset    # 配置: scripts/optimize_loop_config.yaml
```

## 项目结构

```
src/
├── config.py                  # 全部 dataclass 配置（SystemConfig/Lyapunov/Reward/Constraint/PPO/Hierarchical/Ablation）
├── env/
│   ├── channel.py             # 信道模型（AR(1)+路径损耗+SNR依赖CSI误差）⛔不可变
│   ├── traffic.py             # 泊松/自适应λ 到达 + 队列 + QoS 类
│   ├── beamforming.py         # MRT/RZF 预编码
│   ├── rate_calc.py           # MMSE-SIC 速率计算（含 PL 解码序）⛔不可变
│   └── rsma_env.py            # Gymnasium env（物理调用点不可变；obs/reward 接线可改）
├── models/
│   ├── actor_critic.py        # ActorCriticAttn (SA-GAT Worker, 桶专家/backpressure/MF-critic) + 旧 MLP ActorCritic
│   └── meta_controller.py     # MetaActorCritic / MetaRolloutBuffer / MetaPPOTrainer（组级α）
├── optimization/
│   ├── lyapunov.py            # DGLCR + Dinkelbach + CRPO + VQ 奖励控制器
│   ├── meta_reward.py         # Meta 奖励
│   ├── solver_expert.py       # 单槽 surrogate 梯度精修教师（ExIt）
│   ├── diff_engine.py         # 可微 RSMA 速率引擎（torch 影子实现）
│   └── power_control.py       # 功率约束工具（部分占位）
└── training/
    ├── main.py                # 双层 PPO 训练入口（硬例挖掘/CRPO-S/桶路由/ExIt 收集）
    ├── ppo.py                 # PPOTrainer（clip+GAE+KL早停+bc_pretrain+专家头）
    ├── buffer.py              # RolloutBuffer（decoding_order 槽位 + SFL 采样）
    ├── lagrangian.py          # DeliveryConstraint / DeliveryVirtualQueue
    └── curriculum_scheduler.py

scripts/
├── train.py                   # 训练 CLI（~60 参数；需 PYTHONPATH=.）
├── evaluate_snr_sweep.py      # ⛔不可变官方评估
├── evaluate_snr_sweep_temporal.py  # 时序/变体 checkpoint 评估 wrapper
├── optimize_loop.py + optimize_loop_config.yaml  # 自动优化循环 + 不可变约束权威
├── cloud_agents_api.py        # Qoder Cloud Agents HTTP API（optimize_loop 用）
├── _diag_*.py / _scan_*.py    # 一次性诊断脚本（不进主流程）
└── baselines/                 # sdma / noma / equal_power / adaptive_power

tests/                         # 324 个测试（fast 标记子集 322）；新增: test_obs_history, test_diff_engine,
                               # test_solver_expert, test_eval_temporal, test_immutable_guard,
                               # test_backpressure_prior

# 研究与知识层
RESEARCH_BRIEF.md              # 研究方向简报：问题/约束/banlist/当前最优成绩（idea 层数据源）
research-wiki/                 # 持久研究知识库（papers/ideas/experiments/claims/gap_map；index.md 自动生成勿手改）
idea-stage/                    # idea 生成/查新产物（IDEA_REPORT, NOVELTY_*, verified_papers.json）
logs/opt_knowledge.json        # ⭐optimize-loop 跨轮知识库（版本跟踪；提新方向前必读）
logs/                          # 其余训练/评估产物（gitignored）
.trellis/                      # Trellis 任务管理：workflow.md（Plan/Execute/Finish 三阶段）、
                               # spec/backend/eval-compatibility.md（评估脚本对 obs/架构变体的约束，唯一已填 spec）、
                               # tasks/08-24-temporal-worker + 09-02-solver-expert（prd + experiment-results）
.qoder/skills/                 # Qoder CLI 技能（verify/smoke-train/optimize-loop/research-wiki/
                               # idea-creator/novelty-check/comm-lit-review）——非 Claude Code 技能，但编码了规范命令
docs/superpowers/              # 设计文档（MMSE-SIC 集成 spec、augmented Lagrangian plan）
experiments/v2/                # v2 实验代码（需 PYTHONPATH=.）
HANDOFF_*.md / MODEL.md / AGENTS.md  # 历史交接 / 模型说明 / Windows 命令与知识库约定
.aris/                         # 另一 AI 工具的 cache/traces
```

## 编码规范（JSAC/TWC 期刊标准）

- **数学正确性**：β (sic_imperfection) ≠ ε (error_bound)，独立物理机制不可合并；矩阵运算加维度注释 `(N_r, K)`；关键公式 docstring 用 LaTeX
- **命名**：物理量语义命名（`sic_imperfection` 非 `beta`）；矩阵大写 H/V/W，向量小写 h_k；类 CamelCase、函数/变量 snake_case、常量 UPPER_SNAKE_CASE
- **测试纪律**：改动后先跑 fast 门禁再跑全量；新功能必须配测试（形状/数值/边界/可复现）；不可变文件的数值一致性由 `test_immutable_guard.py`、diff engine 由 `test_diff_engine.py` 锁定
- **可复现性**：`np.random.seed` + `torch.manual_seed` + `env.reset(seed=X)` 三重设置；多种子报告 95% CI；benchmark 对比报 均值±CI、Welch's t-test p-value、Cohen's d，多重比较 Holm-Bonferroni
- **实验记录**：任务级结论写入 `.trellis/tasks/<任务>/experiment-results.md`；优化方向级结论由 optimize_loop 写入 `logs/opt_knowledge.json`；通用经验蒸馏进 `.trellis/spec/`（workflow.md Phase 3.3）
