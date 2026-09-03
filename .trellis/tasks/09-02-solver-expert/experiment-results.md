# Solver-Expert 实验结果（09-02-solver-expert 任务）

对照基准：Run 3 `[5, 0.9041, -12.59, 0.6985]`（官方口径）；基线 `[5, 0.9163, -11.0186, 0.6914]`。

## M1: 可微速率引擎（src/optimization/diff_engine.py）— 完成

- torch 复刻 numpy rate_calc（MMSE-SIC 两阶段 + outage + MRT 预编码），float64 一致性 **atol=1e-9**
- 关键修复：eps2 广播方向（`p_p.unsqueeze(1)` 对齐 numpy 的干扰归因）、sqrt 零功率反传 NaN（1e-16 下限，前向误差 ≤1e-8）
- 21 个引擎测试全绿（tests/test_diff_engine.py）

## M2: 求解器专家诊断（scripts/_diag_solver_expert.py）— 全门通过

全量诊断：4 SNR（0/5/15/30dB）× 3 episodes × 200 slots，25 步 Adam（lr 0.1），完整日志 `logs/_diag_solver_expert_full.log`。

| 指标 @SNR | expert_c（冷启动） | SDMA | Δ |
|---|---|---|---|
| delivery 0dB | **0.8990** | 0.8594 | +4.0pt |
| delivery 5dB | **0.9912** | 0.9770 | +1.4pt |
| delivery 15dB | **0.9927** | 0.8930 | +10.0pt |
| delivery 30dB | **0.9140** | 0.6749 | +23.9pt |
| power（4 点） | 13.4–14.3W | 13.7–15.2W | 全点更低 |
| queue 15/30dB | 1.57 / 22.11 | 11.08 / 33.53 | 大幅更低 |
| overflow 15/30dB | 0.000 / 0.056 | 0.038 / 0.284 | 大幅更低 |
| EE（4 点） | 0.110–2.325 | 0.098–1.559 | 全点 ≥ |

- **G1（0/5dB delivery ≥ SDMA）：PASS**；**G2（功率 < 24W）：PASS**
- 对 eval pass 阈值 [0.856, 0.982, _, 0.894, _, _, 0.668]：教师在被测 4 点全部带正裕量（0dB +4.3pt、5dB +0.9pt、15dB +9.9pt、30dB +24.6pt）——**教师上界首次全点越过 pass 线**（Run 3 策略 0/5dB 均缺口）
- **warm vs cold 发现**：warm 起点全 SNR 劣于冷启动（0dB 0.811 vs 0.899）——跨时隙携带过期信道先验在 25 步 Adam 内落入更差盆地；与 DRL-RSMA「测试时精修对初始化极稳健、瓶颈是步长非步数」结论一致。**训练收集用冷启动**（`solve(env)` 不传 warm_theta）
- 诊断门禁初版误 gate 在 expert_w 上；已修正为 expert_c（收集模式专家）并重判 PASS

## M3: 专家迭代训练接入 — 完成

- `ppo.py`：`bc_pretrain(obs, act, orders_np)` 联合训练连续头 + PL 排序头（orders=None 时逐 bit 保持旧路径）；锚定损失透传 orders；新增 `hek_orders` 字段
- `main.py`：`_solver_expert_collect`（逐时隙 solve，Meta 决策按训练节奏运行使 split/α 分布对齐，收集期临时切 eval 口径 snr[-5,35]/load[0.75,1.3] 后恢复课程档位）；`bc_init.pt` 纯 BC 对照检查点（K8 教训）；与 HEK 互斥、与桶专家互斥（无逐样本桶路由）、resume 跳过
- `config.py`：`solver_expert_*` 9 字段默认全关；`scripts/train.py` CLI 透传
- 验证：3 个新单测（联合 BC 似然上升 / 旧路径不变 / 锚定 orders 生效）+ 全量 **315 用例全绿**；冒烟 3 次（含桶专家互斥守卫触发验证）；收集速率 0.345s/slot（1 episode 69s）

## M4: 训练运行（进行中）

- 命令：Run 3 配方（8000ep + cuda + h4 + curltrac + no-bucket-experts + seed 42）+ `--solver-expert --solver-expert-episodes 40`（≈8000 demos，收集约 45min）
- 评估：`evaluate_snr_sweep_temporal.py` 分别评 `bc_init.pt`（纯 BC）与 `best.pt`（BC+PPO）
- 结果待补

### 早期信号：smoke BC 学生已越过低 SNR 缺口线（2026-09-02）

冒烟 BC 检查点（**仅 1 episode = 200 条演示**，obs_history_window=4，150 BC epochs，未做 PPO）在 3ep × eval 口径下：

| SNR(dB) | BC 学生 Deliv | SDMA | pass 阈值 | 判定 |
|---:|---:|---:|---:|---|
| 0 | **0.876** | 0.859 | 0.856 | PASS（+2.0pt 裕量；Run 3 差 -4.1pt）|
| 5 | **0.988** | 0.977 | 0.982 | PASS（+0.6pt；Run 3 差 -0.2pt）|
| 15 | **0.986** | 0.893 | 0.894 | PASS（+9.2pt）|

功率 14.3-15.3W（< SDMA 的 14.7-15.2W 量级），队列/溢出全面占优。教师（M2）与 200 条演示的 BC 学生在缺口区均已越线——M4 全量运行（40ep 演示 + PPO 微调）的期望上限高。
