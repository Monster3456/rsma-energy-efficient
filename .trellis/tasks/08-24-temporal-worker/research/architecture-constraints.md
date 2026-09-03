# 时序策略容量 — 架构约束调研（代码事实）

> 来源：2026-08-24 对当前代码库的逐行调研（Explore agent）。所有 file:line 均已核实。

## 核心结论

**真·有状态 GRU（act() 携带 hidden state）在本项目不可行**，两个硬约束：

1. **不可变评估脚本无重置钩子**：`scripts/evaluate_snr_sweep.py:135` 只调 `model.act(obs_t, deterministic=True)`——obs 是唯一张量参数，无 hidden/done/timestep；模型实例跨 50 episodes × 7 SNR 点复用（evaluate_snr_sweep.py:97–103），状态会跨 episode 泄漏。
2. **PPO 更新是无状态重算**：buffer 平铺无序列结构（buffer.py:35–64），minibatch 随机打乱（ppo.py:465–523），`evaluate_actions` 逐样本无状态重算 log-prob（ppo.py:558–566）。act 时用的 hidden 与更新时重算的 hidden 不一致 → PPO ratio 损坏。

**可行路线：历史窗口堆叠进 obs（策略保持无状态）**——时间信息进观测向量，act/evaluate_actions/buffer/ppo 全部零改动，天然兼容评估脚本。

## 历史窗口路线的关键事实（已验证的兼容性）

- **逐用户历史放 obs 头部**（K×d_user 区）：`ActorCriticAttn._split_obs` 切 `obs[:, :K*d_user]`（actor_critic.py:544–546），d_user 变大只影响 user_embed 输入维；**评估脚本从 `user_embed.0.weight` 自动推断 d_user_eff**（evaluate_snr_sweep.py:240–242），`n_inject = d_user_eff - d_user` 吸收差值（246–254）——旧检查点评估新架构自动适配，无需改脚本。
- **构造器约束**：eval 用固定 kwargs 构造 `ActorCriticAttn`（283–298），`strict=False` 加载（301）；新增构造参数不会被 eval 传入 → 历史维必须从 obs 维度自然推断，不能靠新构造开关。
- **默认必须关**：`tests/test_env.py:51` 钉死 `obs_dim == 11K+3`（legacy 配置 K=16）；新增 obs 块需仿 `obs_include_csi_quality` 开关先例（rsma_env.py:133–135/156–157），默认 off。
- **obs 布局**：per-user K 块在前、global 尾在后；`n_global_inject` 取 global 尾部最后 2 维作 (snr, eps) 注入每个 token（actor_critic.py:549–550）——历史若放 global 尾必须插在 snr/eps 之前；放 per-user 区则无此顾虑。
- **零初始化等价性**：main.py 的 padding-aware 加载器 `_load_state_dict_with_padding`（main.py:141–185）对旧→新形状零填充。旧 checkpoint 的 `user_embed.0.weight (hidden, 13)` 零填充进 `(hidden, 13+H·F)` → 历史输入初始权重全零，**resume 后 step-0 行为与旧策略逐 bit 等价**（与 bucket experts 的 step-0 等价断言同款性质，actor_critic.py:633–662 有先例）。
- **物理零容忍边界**：`_get_observation`（rsma_env.py:347–501）属可改非物理段；`self.channel`/`self.rate_calc` 调用段不可碰。

## obs 现有组成（历史特征的原料）

`_get_observation`（rsma_env.py:460–491），obs_dim = 11K+3（默认 +2K QoS = 13K+3 = 315）：

| 块 | 内容 | 时序价值 |
|---|---|---|
| h_gains_norm | 估计信道增益 Σ\|Ĥ\|²/N_rN_t | **AR(1) ρ=0.9**（config.py:17, channel.py:302）——历史增益可预测下一槽 |
| queue_norm | Q/Q_max | 队列轨迹演化 |
| delta_queue_norm | ΔQ/Q_max | 一阶差分已有 |
| power_history | _prev_power/P_max | 功率反馈已有（单步） |
| rate_history | _prev_rates 归一化 | 速率反馈（单步） |
| 其余 | interference/fairness/split/group/distance/qos + global(time_ratio, snr, eps) | state_aug 开启时 7/8/11 块被 λ/DVQ/η 替换（rsma_env.py:434–458） |

`time_ratio = t/T` 在 global 尾第一个（rsma_env.py:394, 480），reset 时 t=0（591）→ 每 episode 首个 obs 的 time_ratio==0.0。

## 训练/检查点事实

- 训练循环：`while total_steps < max`（main.py:1090）→ 每 rollout 2048 步 ≈ 10.24 episodes（T=200, config.py:13），episode 边界处 `done` 存入 buffer（1265–1272），reset 后继续同 rollout。
- main.py 完全可改（mutable_modules full）；构造 Worker 在 main.py:452–474，读 `ppo.worker_d_user/worker_attn_*` 等字段。
- CLI 架构开关先例：`--use_attention`（main.py:1805, 1829–1830）。
- Checkpoint 结构：`worker_state_dict`/`meta_state_dict`/optimizers/counters/config（main.py:1691–1715）；partial_resume（Worker-only）在 main.py:955–984。
- **本仓库无任何 .pt 文件**（*.pt 全部 gitignore）；optimize-loop 的 `logs/opt_best/best.pt` 只存在于原作者机器。

## 测试影响面

- `tests/test_training.py` 只测 MLP `ActorCritic`，**完全不触及 ActorCriticAttn**（grep 零命中）→ 历史窗口路线零测试破坏（默认 off）。
- `tests/test_env.py` legacy 钉死 obs_dim → 默认 off 即绿。
- 全量套件当前 218 用例全绿（2026-08-24 实测）。

## 备选方案存档（不推荐但记录）

**真 GRU + 序列化 PPO**：需重写 buffer（存序列/hidden）、ppo.py update（373 行）按序列 minibatch、main.py rollout 管 hidden 生命周期、评估端用 time_ratio==0 自动重置 hack。改动面巨大（ppo/buffer/main 全动），PPO 数值一致性风险高，且 eval 的 strict=False 下 GRU 权重对旧 checkpoint 无 padding 等价性（RNN 权重非输入维拼接，无法零填充保等价）。仅当历史窗口路线实验失败后再考虑。
