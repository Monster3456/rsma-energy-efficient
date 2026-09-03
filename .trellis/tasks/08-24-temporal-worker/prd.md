# PRD: 时序策略容量 — 历史窗口观测增强 Worker（长期时隙 RSMA 资源优化）

## 需求背景

项目目标本来就是长期平均 EE 最大化（MODEL.md §3.2），但当前 Worker 是逐时隙无状态前馈网络（SA-GAT），只看当前槽的 obs。信道是 AR(1) 过程（ρ=0.9），队列/功率演化跨槽相关——策略没有跨时隙记忆，无法做预判性调度（如信道好时多花功率、队列高位时提前加速）。当前最优成绩 `[5, 0.9163, -11.0186, 0.6914]`（7 个 SNR 点 5 个达标），知识库中约束机制族/奖励重塑族约 30 条方向全部证伪，跨时隙时序结构是唯一未开发的结构轴，且与已验证成功的课程/硬例挖掘族正交可叠加。

## 方案（已调研定案，详见 research/architecture-constraints.md）

**历史窗口堆叠进 obs**：环境维护最近 H 个时隙的关键逐用户特征快照（信道增益、队列、功率），拼进观测向量的 per-user 特征区。策略保持完全无状态——act/evaluate_actions/buffer/ppo 零语义改动，评估脚本零改动。

真·GRU 已被证伪为不可行：不可变评估脚本无 hidden 传递/重置钩子（evaluate_snr_sweep.py:135），且 PPO 打乱 minibatch 无状态重算会破坏 act/更新时 log-prob 一致性。

### 核心设计

| 项 | 决策 |
|---|---|
| 历史特征 F | `h_gains_norm`、`queue_norm`、`power_history`（3 项/槽，均为既有 obs 块的原料） |
| 窗口长度 H | 默认实验值 4（config 可调） |
| 拼接位置 | per-user 特征区末尾（既有 13 个特征块之后、global 尾之前）→ `d_user_eff = worker_d_user + H·F` |
| 开关 | `SystemConfig.obs_history_window: int = 0`（**默认关**，保持 218 测试全绿与旧 checkpoint 兼容） |
| reset 行为 | 历史窗用 reset 时刻的当前值预填充（避免零瞬态） |
| eval 兼容 | d_user_eff 由 `user_embed.0.weight` 自动推断（evaluate_snr_sweep.py:240–254），`strict=False` 加载，零改动 |
| resume 等价性 | 零填充加载器使历史输入初始权重为 0 → **resume 旧 checkpoint 的 step-0 行为与原策略逐 bit 等价**（与 bucket experts step-0 等价先例同性质） |
| 兼容矩阵 | `_split_obs` K·d_user 切片、n_global_inject 尾 2 维 (snr,eps)、bucket experts 头、Meta 通路全部不受影响 |

## 改动范围（全部在 optimize_loop_config 白名单内）

- `src/config.py`（full）：`obs_history_window`、`obs_history_features` 字段
- `src/env/rsma_env.py`（非物理段）：`_get_observation`/`reset` 维护历史 deque 与拼接；**不触碰 `self.channel`/`self.rate_calc` 物理调用段**
- `src/training/main.py`（full）：Worker 构造时 d_user 加历史维；CLI 开关 `--obs_history_window`
- `scripts/train.py`：CLI 透传（注意其 Mac 路径陷阱，运行需 PYTHONPATH=.）
- `scripts/evaluate_snr_sweep_temporal.py`（新增，实现中发现评估不兼容后补录）：评估包装器——注入 temporal 配置（obs_history_window/features + worker_d_user=13+H·F）后经 runpy 原样运行不可变的 `evaluate_snr_sweep.py`，**零口径逻辑复制**。背景：不可变脚本从 checkpoint 推断 d_user_eff 时把每用户历史增长误读为 global 注入维（该机制只为 +2 注入设计），直接评估历史 checkpoint 会 shape 崩溃或静默错位。
- `tests/`：新增历史 obs 单测（默认关 obs_dim 不变；开后 = 11K+3(+2K QoS)+H·F·K；窗口内容正确性；reset 预填充）与包装器单测/集成测试（构造未训练 temporal checkpoint 子进程跑通完整评估链路）

## 验收标准

1. 默认配置下全量 218 测试全绿（零回归）
2. `obs_history_window=4` 下：新增单测全绿 + smoke 训练跑通（/smoke-train）
3. 训练实验：从最佳 checkpoint partial_resume（等价起点）或从头训练，SNR sweep 评估与当前最优 `[5, 0.9163, -11.0186, 0.6914]` 对比
4. 快速门禁 `pytest -m fast` 每次改动后通过

## 已确认决策（2026-08-24 用户确认）

1. **训练起点：从零训练**（不依赖原机器 checkpoint；内置课程+硬例挖掘从零训）
2. **实验预算：全量训练（8000+ eps），使用 GPU**——本机 RTX 5060（8GB, Blackwell sm_120, 驱动 CUDA 13.0）
   - 环境前置：.venv 原 torch 为 CPU-only（2.13.0+cpu），需换装 cu128 构建（Blackwell 支持自 cu128 起）
   - 代码侧：训练入口已有 `--device cuda`；实现时须验证 GPU 路径真实生效（张量/模型全部迁移无 CPU 残留，smoke 用 cuda 跑通）
3. 实验命令形态：`PYTHONPATH=. .venv/Scripts/python.exe scripts/train.py --episodes 8000 --device cuda --exp_name temporal_worker_h4 --obs_history_window 4 ...`（CSI/SNR 域随机化维持默认 imperfect / [-5,35]dB）
4. 评估对照：当前最优分数 `[5, 0.9163, -11.0186, 0.6914]`（evaluate_snr_sweep，CPU 评估即可）
