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

### M4 重启记录（2026-09-07 08:13，Claude Code 会话）

- **发现**：原 M4 run（`logs/solver_expert_h4/`，9-03 00:41 中断）只完成到 BC——仅存 `bc_init.pt`（8000 demos）+ config.json，PPO 微调未产出任何 checkpoint，"结果待补"由此悬置。
- **不复用 resume 的原因**：`main.py:1183` 的 `se_on` 要求 `not resume_ckpt_path` → resume 会跳过衰减锚定微调（anchor 参数仅在 se_on 分支设置），偏离 M3 预注册协议（BC → 锚定 PPO）。收集仅 ~45min，从头重跑保协议完整性。
- **前置验证**：fast 门禁 322/322 全绿（20.4s）；同旗标冒烟通过（`logs/m4_smoke/`：obs_dim=603、cuda、CuRLTraC+LexProj 启用、桶专家互斥守卫通过、SolverExpert 200 demos/65s、BC joint cont+PL、anchor coef=0.2/decay=0.95/updates=40 就位）。
- **重启命令**（与中断 run 配置逐项核对一致：h4 + curltrac + no-bucket-experts + seed 42 + cuda + 8000ep + 40ep demos）：
  `PYTHONPATH=. python scripts/train.py --exp_name solver_expert_m4 --episodes 8000 --device cuda --seed 42 --obs_history_window 4 --curltrac --no-bucket-experts --solver-expert --solver-expert-episodes 40`
- **运行产物**：`logs/solver_expert_m4/`（config.json + checkpoints），控制台日志 `logs/solver_expert_m4_console.log`。
- **评估计划（M4 预注册）**：`bc_init.pt`（纯 BC 对照，K8 教训）与 `best.pt`（BC+PPO）均走 `evaluate_snr_sweep_temporal.py --obs_history_window 4`，7 点 × 50ep 官方口径；对照 Run 3 `[5, 0.9041, -12.59, 0.6985]` 与基线 `[5, 0.9163, -11.0186, 0.6914]`。

### M4 早期结果：纯 BC 对照官方评估 = **7/7 达标（历史性突破）**（2026-09-07 12:40）

- 对象：`logs/solver_expert_m4/checkpoints/bc_init.pt`（8000 demos，150 BC epochs，**0 步 PPO**）；日志 `logs/_eval_m4_bc_init.txt`
- 分数经 `optimize_loop.parse_eval_output + compute_score` 官方解析：
  **score = [7, 0.9441, −9.3929, 1.0753]**
  - vs 历史最佳基线 `[5, 0.9163, −11.0186, 0.6914]`、Run 3 `[5, 0.9041, −12.59, 0.6985]` —— **字典序四个槽位全面占优**
  - ee_ratio 1.0753 > 1.05 全局门 ✓；avg power 13.69W < SDMA 14.66W ✓ → `check_termination` 全局条件同时满足
- 逐点 delivery（BC / SDMA）：

| SNR(dB) | BC Deliv | SDMA | Δ | BC Q | SDMA Q | 门限(1.1×) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **0.882** | 0.856 | +2.6pt | 2.92 | 3.46 | ✓ |
| 5 | **0.989** | 0.979 | +1.0pt | 0.43 | 0.68 | ✓ |
| 10 | 0.993 | 0.961 | +3.2pt | 0.74 | 2.28 | ✓ |
| 15 | 0.986 | 0.887 | +9.9pt | 2.47 | 11.42 | ✓ |
| 20 | 0.964 | 0.795 | +16.9pt | 8.70 | 22.39 | ✓ |
| 25 | 0.924 | 0.724 | +20.0pt | 18.85 | 28.76 | ✓ |
| 30 | 0.871 | 0.664 | +20.7pt | 31.64 | 34.47 | ✓ (37.9) |

  overflow 高 SNR 大幅更低（30dB 0.082 vs SDMA 0.293）。
- 最薄裕量：5dB delivery +1.0pt、0dB +2.6pt（50ep 口径）——正是 Run 3 的缺口区（−4.1/−0.2pt）
- **解读**：「瓶颈在教师信号而非训练量/机制」假设被直接验证——零 RL 探索、纯蒸馏即越过全部逐点门。36+ 机制族证伪后的首个结构性突破，来自初始化层（ExIt/BC）而非优化期机制
- 观察（训练侧，截至 update 240 / ep 2457）：CuRLTraC 已到终档 S4/4；AvgDeliv≈0.89 平台；每轮 KL 早停（KL 13–280 ≫ target 0.02，LR 已衰减至 6e-5 下限）；best.pt 首版 @ update 200。**风险**：锚定期后 PPO 大幅移动策略（KL 高企），需警惕把 BC 学生拽离可行盆地——best.pt 出分后必须与 bc_init 对比，若劣化则纯 BC 检查点即为交付候选
### M4 终判（2026-09-07 17:50）：纯 BC 胜出，微调净损害

- 训练 17:40 前后完成（8000ep / 780 updates / MetaUpd 781，final.pt 落盘）
- **best.pt（BC+PPO @ u200）官方评估 = `[6, 0.8857, −13.5514, 1.0702]`**（日志 `logs/_eval_m4_best_u200.txt`，快照 `best_u200_snapshot.pt`）：30dB 点 queue 38.83 > 门限 37.92 失守，delivery 仍全点 ≥ SDMA
- best.pt 自 u200 后再未更新（MeanReward 峰值 −0.450 未被超越）；后半场 AvgDeliv 单调下滑 0.888(u200)→0.846(u780)，λ 5.4→20.3 爬升未能拉回 → **final.pt 不再评估**（训练窗口趋势单调，评估口径无翻盘可能，省 2h CPU）

| 检查点 | points_met | avg_deliv | −avg_queue | ee_ratio | 判定 |
|---|---:|---:|---:|---:|---|
| **bc_init.pt（纯 BC, 0 步 PPO）** | **7** | **0.9441** | **−9.39** | **1.0753** | ✅ **M4 交付候选，新纪录** |
| best.pt（BC+PPO @ u200） | 6 | 0.8857 | −13.55 | 1.0702 | 微调净损害 |
| 历史最佳基线 | 5 | 0.9163 | −11.02 | 0.6914 | 被全面超越 |
| Run 3 | 5 | 0.9041 | −12.59 | 0.6985 | 被全面超越 |

- **机制诊断**：锚定项 0.2×0.95^40 ≈ 0.026 在 u40 后名存实亡 → 裸 PPO（每轮 KL 15–280 ≫ 0.02 早停、lr 衰至 6e-5）在已处可行盆地的策略上继续大步移动，把策略拽向训练 reward（DGLCR 复合目标 @ S4/4 全难度混合分布）的最优点——该点与官方评估 score 不重合。K8 教训「纯 BC 与 BC+PPO 都评」本次直接决定了交付物
- **官方成功判据首次全线满足**：points_met 7/7 且 ee_ratio 1.0753 > 1.05 且 power 13.69W < SDMA 14.66W（check_termination 全局门同步通过）
- **后续方向（M5 候选，未启动）**：
  1. 纯 BC 扩容：demos 40→160ep，验证 avg_deliv/ee_ratio 能否继续抬（本次 200 demos 冒烟 → 8000 demos 全量已证明 demo 量正收益）
  2. 修微调配方：锚定加长/加强、KL-to-BC 信任域、actor lr ×0.1（`resume_lr_actor_scale`）、eval 劣化早冻结
  3. 基线切换：`optimize_loop_config.yaml` 的 `baseline.checkpoint` 指向 bc_init（需用户决策，涉及官方口径连续性）
- 知识库注记：`logs/opt_knowledge.json` 仅由 optimize_loop.py 写入（禁手改）；「ExIt/BC 蒸馏 = 成功方向」「锚定 PPO 微调 = 失败方向（本配方）」待下轮 loop 或以本文件为源录入

## M5: 安全微调协议 SFTP + ExIt-2（预注册，2026-09-07）

**目标（用户指令）**：不满足于纯 BC 交付——RL 必须当主角（BC+PPO > 纯 BC），讲好"求解器引导专家迭代 + RL 跨时隙精修"的故事线；时间与算力不设限。

### M4 微调失败的法医结论（代码证据）

1. **锚定在 lex 路径是死代码（头号根因）**：`use_lexicographic_projection=True`（默认）+ RCPO cost 通道 → `update()` 走 `_apply_lexicographic_policy_grad` 分支（ppo.py:584-616），而 HEK/solver 锚定损失只在原 PPO else 分支（ppo.py:668-688）。M4 日志 `[LexProj] enabled` 证实——启动打印的 "anchor: coef=0.2 decay=0.95 updates=40" 从未进过 loss，BC 策略自 update 1 起裸奔
2. **锚定窗口本就太短**：即使生效，0.2×0.95^40≈0.026，u40 后名存实亡（780 updates 的 95% 无保护）
3. **critic 冷启动**：bc_pretrain 只训 actor；value 头从零学 603 维 obs 的复合奖励 → 早期优势估计纯噪声直接砸在 BC 盆地上（ValueLoss 全程 2-5）
4. **log_std 高学习率**（lr_log_std=1e-3）：熵尖峰 −51→−89（探索坍缩），KL 每轮 15-280 早停
5. **Meta 共同学习非平稳**：BC 演示条件于未训练 Meta 的 (α,g) 分布；PPO 期 Meta 同步漂移（MetaLoss 800-13000），Worker 的 BC 映射过时
6. **best.pt 按训练 MeanReward 选** ≠ 官方评估 score（u200 峰值在评估口径已是 6/7）；且无任何回退保护

### 修复设计（全部 run 级旗标，默认关 = 官方口径零变化；全部在 mutable 白名单内）

- **F0 锚定接入 lex 路径**：`_anchor_grad_into(actor_params, hek_coef, n)` 助手——演示批 NLL 对 actor+log_std 参数的梯度加到投影后梯度上（锚定是稳定器，不进零空间投影）；两条路径共用
- **F1 持久锚定**：`--solver-expert-anchor-updates 100000 --solver-expert-anchor-decay 0.999`（780 updates 后系数仍 ≈0.09）
- **F2 critic 预热**：`--actor-warmup-updates 30`——前 30 个 update 只训 critic/cost-critic（actor+log_std 梯度置零；Adam 零梯度+零动量 ⇒ 参数严格不动）
- **F3 小步微调**：`--lr_actor 3e-5 --lr_log_std 1e-4 --clip_ratio 0.1`
- **F4 Meta 冻结**：`--meta-freeze-episodes 2000`——前 2000ep Meta 采样但不更新（Worker 在 BC 时的 (α,g) 条件分布上稳定），之后恢复
- **F5 评估守卫 + 回滚**：`--eval-guard`——每 400ep 用独立 env 实例跑 mini-eval（{0,5,30}dB × 4ep，确定性策略，对齐官方 info 键 delivery_ratio/mean_queue），对照 bc_init 官方参考（avg_deliv@3pts = 0.914，SDMA 门 [0.856,0.979,0.664] / queue 门 1.1×[3.46,0.68,34.47]，出处 logs/_eval_m4_bc_init.txt）：跌破 ref−0.02 → 回滚到 last-good 快照 + actor lr 减半（至多 3 次，之后冻结 actor 只训 critic/Meta）；守卫同时选 best_guard.pt（按 mini-eval avg_deliv）
- **R1 ExIt-2 on-policy relabel**：`--solver-expert-relabel`——BC 后用学生策略（确定性 act）走轨迹、教师从学生状态逐槽打标（DAgger 式协变量偏移校正），与原始演示聚合（16000）二次 BC，存 bc_relabel.pt

### 三臂并行（预算不限，互为消融；同 seed 42 / h4 / curltrac / no-bucket-experts / 8000ep / cuda / 40ep demos）

| 臂 | exp_name | 配方 | 隔离的问题 |
|---|---|---|---|
| C | m5_armC_anchorfix | M4 + F0 + F1（其余与 M4 逐 bit 同） | 死锚定是否唯一主因 |
| A | m5_armA_sftp | C + F2+F3+F4+F5 | 完整安全微调协议 |
| B | m5_armB_exIt2 | A + R1 | ExIt 第二迭代的增量（论文头条臂） |

### 判定门（官方口径 evaluate_snr_sweep_temporal 7点×50ep）

- 对照：bc_init `[7, 0.9441, −9.3929, 1.0753]`
- **成功**：任一臂字典序 > bc_init（points_met=7 前提下 avg_deliv > 0.9441，其后比 queue、ee_ratio）
- **叙事**：B > A ⇒ 完整 ExIt 故事（教师→蒸馏→安全 RL 精修→on-policy 再蒸馏）；C ≥ A ⇒ 单点修复即够，协议做减法；全部 ≤ bc_init ⇒ 微调通道对本案 reward 结构性失配，交付纯 BC + 把结论录入知识库
- 每臂评 best_guard.pt（有守卫时）、best.pt、final.pt 三检查点，防"选点口径"翻车（K8 教训）

### M5 实施与启动记录（2026-09-07 18:37）

- **代码落地**（全部 mutable 白名单内，默认关=官方口径零变化）：
  - `ppo.py`：`_anchor_grad_into()` 锚定梯度注入 lex 投影路径（F0，M4 头号根因修复）；`actor_warmup_updates` critic 预热（F2，warmup 期 actor/log_std grad=None ⇒ Adam 严格跳过）；update() 返回 `in_warmup`
  - `main.py`：`_guard_run_point/_guard_check/_guard_rollback`（F5，参考值常量出处 `logs/_eval_m4_bc_init.txt`）；`meta_freeze_episodes` 门（F4）；`_solver_expert_collect(mode="relabel", student_ac=…)` + 聚合二次 BC + `bc_relabel.pt`（R1）；`best_guard.pt` 按守卫 avg_deliv 选点
  - `config.py`：PPOConfig 7 字段 + HierarchicalConfig 1 字段 + ConstraintConfig 2 字段，全默认关
  - `train.py`：13 个新 CLI 旗标透传
- **验证**：新增 `tests/test_ft_protocol.py` 9 用例全绿（锚定 NLL 下降/无演示零操作/warmup actor 严格不动+critic 学习/守卫结构与回滚-lr减半-超限冻结/relabel 形状与守卫条件）；fast 门禁 **331/331**；全机制冒烟通过（持久锚定 decay=0.999、relabel 聚合 BC n=400、MetaFreeze 生效 MetaUpdates=0、守卫 mini-eval 3/3 + best_guard.pt 落盘）
- **三臂已并行启动**（18:35，cuda，seed 42）：`m5_armC_anchorfix` / `m5_armA_sftp` / `m5_armB_exIt2`，控制台日志 `logs/m5_arm{C,A,B}_console.log`
- 预计：并行收集 ~2h（B 另加 relabel ~1h）→ PPO 8000ep 共享 CPU ~12-18 FPS → **全程 ~30-40h**（9/9 前后出齐）

### M5 终审（2026-09-08 08:10）：**Arm A（SFTP）胜出——RL 微调首次字典序全面超越纯 BC**

三臂 8000ep 于 09-08 04:40 前后完成（实际 PPO 吞吐 ~13.6ep/min，快于预估）。9 检查点官方评估（7点×50ep，`logs/_eval_m5_*.txt`），分数经 `compute_score`：

| 检查点 | score [pts, deliv, −queue, ee] | vs bc_init [7, 0.9441, −9.3929, 1.0753] |
|---|---|---|
| **A/best_guard.pt (ep6901)** | **[7, 0.9519, −8.3971, 1.0804]** | ✅ **三槽位全胜 → M5 交付物** |
| B/bc_relabel.pt | [7, 0.9447, −9.2429, 1.0715] | ≈平（deliv +0.0006 噪声级） |
| B/best_guard.pt (ep1228) | [7, 0.9444, −9.2386, 1.0716] | ≈平 |
| C/final.pt | [7, 0.9436, −9.8157, 1.0840] | deliv 平（−0.0005 噪声级）、ee 全场最高 1.0840、queue 略差 |
| A/final.pt | [7, 0.9337, −9.8543, 1.0744] | 劣 |
| A/best.pt (u200) | [7, 0.9284, −11.1700, 1.0643] | 劣 |
| B/best.pt (u200) | [7, 0.9214, −11.5086, 1.0559] | 劣 |
| C/best.pt (u200) | [7, 0.9171, −12.9486, 1.0662] | 劣 |
| B/final.pt | [6, 0.9074, −12.6686, 1.0354] | ❌ 唯一失点（B 后段熵坍缩 −94.7 的代价） |

**A/best_guard 逐点增益 vs bc_init**（delivery）：0dB +0.4pt、5dB ±0、10dB ±0、15dB +0.5pt、20dB +1.0pt、25dB +1.8pt、30dB +1.7pt；queue 全点更低（30dB 29.6 vs 31.6）。**增益集中在 15-30dB 深过载区**（load_ratio→1.3）——正是跨时隙预判调度（h4 历史窗口）与队列管理能兑现、而近视逐槽教师无法触及的区域。RL 主角叙事成立。

**消融结论（论文表格式）**：
1. **F0 锚定修复单点起效**：C 臂（只修死锚定 bug）训练窗口 AvgDeliv 0.846(M4)→0.923，MeanReward 转正，final 守住 7/7——确认 M4 失败头号根因；但 C 未显著超越 bc_init（deliv 平、queue 略差）⇒ **锚定是"不劣化"的必要条件，不是"超越"的充分条件**
2. **完整 SFTP 才产生超越**：A 在 C 基础上 +warmup/小步/Meta冻结/守卫选点 → 三槽位全面增益；守卫 20 次检查零回滚、全程 3/3，且 **best_guard（按守卫 avg_deliv 选点）> final > best（按 MeanReward 选点）**——守卫选点机制本身贡献了 +0.8~2.4pt deliv 的检查点差异
3. **R1 relabel 判负（本配方下）**：B/bc_relabel ≈ bc_init（聚合 16000 演示无增益），B 的 PPO 段后程熵坍缩（−51→−94.7，尽管 lr_log_std=1e-4）、5dB queue mini-eval 擦线、final 掉到 6/7 ⇒ on-policy relabel 引入的状态分布让微调更难稳；**ExIt 第二迭代不采用**
4. **K8 教训第三次应验**：MeanReward 选点与官方 score 失配（三臂 best.pt 全部劣于同臂 final/best_guard）——best.pt 选点口径应废弃，交付一律走守卫/官方评估选点

**交付物**：`logs/m5_armA_sftp/checkpoints/best_guard.pt`（ep6901）= **[7, 0.9519, −8.3971, 1.0804]**，功率 13.84W < SDMA，ee_ratio 1.0804 > 1.05——官方成功判据（逐点门+全局门）全部满足且为历史最佳。

**后续候选（M5 终审时列出）**：
- A 臂 actor lr 被 KL 早停连击压到 6e-6（近似冻结下仍取得 +0.8pt）——放宽 `kl_stop_lr_decay`/抬 lr 下限，超越幅度可能更大
- 守卫 mini-eval 4ep 噪声 vs 官方 50ep：可在守卫点加 episode 数或引入 queue 判据（B 臂 5dB queue 擦线是 avg_deliv 单判据漏掉的）
- 基线切换决策（`optimize_loop_config.yaml` baseline → A/best_guard.pt）仍待用户拍板

---

## M5.1 预注册（2026-09-08）：解除 actor 学习率饿死

**用户决策**：采纳上列候选 1（"不要求时间和花费，选最好的，但 RL 要当主角"）。候选 2/3 暂不动。

### 假设（H-M5.1）

Arm A 的增益是在 **actor lr 被 KL 早停连击从 3e-5 一路压到 6e-6（≈5× 衰减、策略近似冻结）** 的条件下取得的，仍拿到 delivery +0.78pt / queue −10.6% / ee +0.005。因此 A 臂的 SFTP 配方**处于欠优化侧**：微调预算被早停衰减机制吃掉，而非被守卫回滚吃掉（20 次检查零回滚 ⇒ 安全余量从未被触碰）。放宽学习率饿死后，守卫机制仍提供同一硬下限（最坏 = 回滚到 bc_init 的 7/7），故**风险不对称：下行有保护，上行有空间**。

### 双臂设计（各自对 Arm A 单变量对照）

| 臂 | exp_name | 相对 Arm A 的唯一改动 | 回答的问题 |
|---|---|---|---|
| **A2** | `m51_armA2_lr1e4` | `--lr_actor 1e-4`（基准 lr 抬 3.3×，KL 衰减机制保留） | 抬基准 lr 能否跑赢衰减压制 |
| **A3** | `m51_armA3_nokldecay` | `--kl-stop-lr-decay 1.0`（关闭 KL 早停连击衰减，基准 lr 保持 3e-5） | 衰减机制本身是否就是饿死元凶 |

A2/A3 分离了"基准 lr 太低"与"衰减太狠"两条因果，二者可加性由对照直接读出。其余旗标与 Arm A 逐位一致（`--episodes 8000 --device cuda --seed 42 --obs_history_window 4 --curltrac --no-bucket-experts --solver-expert --solver-expert-episodes 40 --solver-expert-anchor-updates 100000 --solver-expert-anchor-decay 0.999 --lr_log_std 1e-4 --clip_ratio 0.1 --actor-warmup-updates 30 --meta-freeze-episodes 2000 --eval-guard --eval-guard-every 400`）。

### 判定门（预注册，出分后不改）

参照系 = **Arm A/best_guard `[7, 0.9519, −8.3971, 1.0804]`**（非 bc_init）：

1. **胜出**：某臂 `best_guard.pt` 官方 score 字典序压过 A/best_guard ⇒ 成为新交付物，M5.2 沿同方向再推
2. **持平**：points_met=7 且 avg_deliv 差异在 ±0.002 噪声带内 ⇒ 判定"lr 饿死不是当前约束"，转向候选 2（守卫判据加 queue 门 / 提 mini-eval episode 数）
3. **判负**：points_met<7 或守卫出现回滚 ⇒ 记录"放宽 lr 破坏 SFTP 单调安全性"入知识库，**Arm A/best_guard 仍为交付物**，方向关闭
4. 无论胜负，`best.pt`（MeanReward 选点）一律不作判定依据（K8 教训已三次应验）

### 工程前置（本会话已完成）

- `--kl-stop-lr-decay` CLI 透传补齐（`scripts/train.py:220` 注册 + `:418-419` 写入 `config.ppo.kl_stop_lr_decay`；字段本已存在于 `src/config.py:439`，默认 0.9）
- 上一会话遗留的"`--help` exit=1"已定性为**无害**：help 文本含组合字符 `U+0304`，GBK 控制台编码崩溃；加 `PYTHONIOENCODING=utf-8` 后 exit=0、旗标正常回显。非代码缺陷
- fast 门禁 **331/331** 全绿
- 双臂冒烟（1 episode，`--solver-expert-episodes 1`）**通过**：两臂均 200 demos/59s、BC loss 逐位一致 −400.3256（同 seed，且 BC 不依赖 lr_actor/kl_stop_lr_decay，符合预期）、anchor decay=0.999 updates=100000、守卫 ep1 落盘 `best_guard.pt`（avg_deliv 0.9882）、正常收官。日志 `logs/_smoke_m51_A{2,3}.log`
- 旗标链路核实：`lr_actor` 由 `train.py:238` 无条件写入 `config.ppo.lr_actor`；`kl_stop_lr_decay=1.0` 使 `ppo.py:818` 的 `_kl_decay_factor *= 1.0` 恒等 ⇒ 确实关闭饿死通道（非仅改字段）

### M5.1 启动记录（2026-09-08 09:24）

双臂并行启动（cuda，seed 42，8000ep，GPU 起始全空闲 0 MiB / 8151 MiB）：

| 臂 | exp_name | 真实 pid | 控制台日志 |
|---|---|---|---|
| A2 | `m51_armA2_lr1e4` | 8880 | `logs/m51_armA2_console.log` |
| A3 | `m51_armA3_nokldecay` | 50288 | `logs/m51_armA3_console.log` |

启动后 2 分钟核实：两进程各 ~1.5 GB 常驻、config 回显 `max_total_steps=1600000`、处于演示收集阶段（40ep，M5 同阶段 3 臂并行耗时 2371s，本次 2 臂预计更快）。

**ETA**：收集 ~40min（~10:05）→ BC 150 epochs（~10:20）→ PPO 8000ep（M5 实测 13.6 ep/min @3 臂并行，2 臂预计 15–20 ep/min ⇒ 6.7–8.9h）→ **预计今日 17:00–19:30 收官**。

监控：`grep "Guard\] ep=" logs/m51_armA2_console.log`（每 400ep 一行守卫判定）；`grep -E "^\[Step" logs/m51_armA2_console.log | tail -1`（进度/LR/熵）。**关键观测量**：A3 的 `LR=` 是否全程稳定在 3e-5（对照 A 臂被压到 6e-6）、A2 的 LR 衰减轨迹、两臂守卫是否首次出现 ROLLBACK（判定门 3 的触发条件）。

### M5.1 终审（2026-09-08 20:00）：**双臂均未超越 Arm A，冠军不变，H-M5.1 证伪**

双臂 8000ep 于 17:26/17:27 收官（收集各 ~2450s，BC loss 逐位一致 −473.7537 = 与 Arm A 同 seed 同演示）。4 检查点官方评估（7点×50ep，`logs/_eval_m51_*.txt`，`.err` 全 0 字节），分数经 `compute_score`：

| 检查点 | score [pts, deliv, −queue, ee] | 功率 | vs **A/best_guard [7, 0.9519, −8.3971, 1.0804]** |
|---|---|---|---|
| **A3/final.pt** | [7, 0.9471, −8.5471, 1.0777] | 13.66W | deliv −0.0048 ⇒ **字典序负于槽位 2** |
| **A3/best_guard.pt** (ep5683) | [7, 0.9463, −9.4457, 1.0810] | 13.82W | deliv −0.0056 ⇒ 负 |
| **A2/best_guard.pt** (ep1628) | [7, 0.9446, −9.3600, 1.0737] | 13.69W | deliv −0.0073 ⇒ 负 |
| **A2/final.pt** | [7, 0.9014, −13.2114, 1.0356] | 13.68W | deliv −0.0505 ⇒ **惨负，且低于 bc_init 0.9441** |

完整排序（avg_deliv）：**A/best_guard 0.9519** > A3/final 0.9471 > A3/best_guard 0.9463 > A2/best_guard 0.9446 > bc_init 0.9441 > A2/final 0.9014。**交付物仍为 `logs/m5_armA_sftp/checkpoints/best_guard.pt`。**

#### 机制真相：lr 下限由线性退火主导，KL 衰减只是把它提前触底

`src/config.py:438` `lr_decay_final_ratio = 0.2`；`src/training/ppo.py:181-187`：

```
progress_scale = (1-p) + p*0.2            # 线性退火 1.0 → 0.2
scale = max(progress_scale * _kl_decay_factor, 0.2)   # 硬下限 0.2
lr = base * scale
```

三臂控制台 `KLstop=1` 均为 **39/39**——每次 update 都触发 KL 早停，故 `_kl_decay_factor` 被 0.9 连乘至远低于下限：

| run | base lr | u20 | u100 | u380 | u780 | 形态 |
|---|---|---|---|---|---|---|
| Arm A | 3e-5 | 1.56e-05 | **6.00e-06** | 6.00e-06 | 6.00e-06 | u100 起钉死在下限（87% 训练在 1/5 lr） |
| A3（decay=1.0） | 3e-5 | 2.94e-05 | 2.69e-05 | **1.83e-05** | 6.04e-06 | 完整线性退火 |
| A2（base 1e-4） | 1e-4 | 5.21e-05 | **2.00e-05** | 2.00e-05 | 2.00e-05 | 下限随 base 抬到 2e-5，u100 起钉死 |

A3 的 u380 实测 1.83e-05 与公式预测 `3e-5 × [(1−0.487)+0.487×0.2] = 1.83e-05` **逐位吻合** ⇒ 旗标确实生效、`_kl_decay_factor` 恒为 1.0，A3 拿到了约 **3× 的中段 lr 预算**。

#### 结论 1：lr 饿死不是约束项，反而是隐式信任域（H-M5.1 证伪）

A3 用 3× lr 预算换来 deliv **−0.0056**，且逐点看损失**完全集中在 25/30dB**：

| SNR | A3/best_guard | A/best_guard | Δ |
|---|---|---|---|
| 0/5/10/15 dB | 0.886 / 0.989 / 0.993 / 0.991 | 0.886 / 0.989 / 0.993 / 0.991 | **逐点全同** |
| 20 dB | 0.973（Q 7.34） | 0.974（Q 7.20） | ≈平 |
| **25 dB** | 0.934（Q 17.78） | 0.942（Q 16.08） | **−0.8pt，队列更差** |
| **30 dB** | 0.858（Q 35.11） | 0.888（Q 29.63） | **−3.0pt，队列差 18.5%** |

而 25/30dB 正是 Arm A 相对 bc_init 取得 +1.8/+1.7pt 增益的区域。**即：Arm A 的"近似冻结"恰恰保住了深过载区的教师锚定行为，多给的 lr 预算让策略从这里漂走了。** 饿死不是缺陷，是意外起作用的信任域——这与 M5 终审"锚定是不劣化的必要条件"同源，说明真正稀缺的是**对 BC 盆地的约束强度**，不是优化步长。

另注：A3 的 `final.pt`(0.9471) > `best_guard.pt`(0.9463)，**守卫选点优势在 A3 消失**（Arm A 是 best_guard 0.9519 ≫ final 0.9337）⇒ 守卫选点并非普适更优，其收益依赖具体 run 的训练末态是否退化。

#### 结论 2：抬 base lr 明确有害（A2 间歇性失稳）

A2 的失效不是"学太多"而是**周期性大步跳出**：

```
A2:  u60 KL=3092 · u360 KL=11061 · u480 KL=370 · u600 熵=−75.4 · u720 熵=−56.2 · 末态 PolicyLoss=33108 KL=30333 熵=−80.3
A3:  KL 全程 6–33 · 熵全程 −51.3x · 末态 MeanReward −1.589
A2 训练窗口 AvgDeliv: 0.910(u240) → 0.888(u480) → 0.898(u720)
```

官方评估逐点印证：A2/final 在 15/20/25/30dB 分别只有 0.933/0.887/0.840/**0.800**，30dB 队列 36.78 已超 SDMA 的 34.47。`A2/best_guard.pt` 存于 ep1628、早于 u360 的 KL=11061 爆炸，故仍保住 7/7（0.9446）——**守卫选点在这里确实起到了"截取爆炸前状态"的作用**，是唯一为 A2 挽回体面的机制。

#### 结论 3：守卫盲区已被实锤（M5.2 头号目标）

A2 末态 PolicyLoss=33108、KL=30333、熵 −80，**守卫却 19/19 全绿、ROLLBACK=0**，ep7710 仍报 `avg_deliv=0.9791 points_met=3/3`；同期训练窗口 AvgDeliv 只有 0.898。根因：守卫 mini-eval 固定在 {0,5,30}dB × 4ep，三点交付率长期饱和在 0.97–0.99，**对策略空间病态无分辨力**，且单判据 avg_deliv 漏掉队列（M5 终审已列此候选，本次由 A2 提供硬证据）。

附带核查（更正本次分析中的一处误读）：A3/best_guard 30dB 队列 35.11、A2/final 30dB 队列 36.78 虽均已超 SDMA 的 34.47，但**逐点达标判据本就含队列项**——`optimize_loop.py:1762/1777` 判定式为 `delivery ≥ SDMA_delivery × delivery_sdma_ratio(1.0)` **且** `queue ≤ SDMA_queue × queue_sdma_ratio(1.1)`（阈值见 `optimize_loop_config.yaml:23-24`）。30dB 的队列上限 = 34.47 × 1.1 = **37.92**，故两者均在门内，7/7 是官方口径下的合法结果，非计分漏洞。反向印证：M5 的 A/best.pt 正因 30dB 队列 38.8 > 37.92 而失守成 6/7，与该阈值逐位吻合。

但**余量厚度值得记录**：A2/final 的 30dB 队列已达上限的 97.0%（36.78/37.92）、A3/best_guard 达 92.6%（35.11/37.92），而冠军 A/best_guard 仅 78.1%（29.63/37.92）。即两者虽同报 7/7，**距离失守的安全边际远薄于冠军**——这是 avg_queue 槽位（−13.21 / −9.45 vs 冠军 −8.40）已经反映、但 points_met 槽位无法反映的信息，也是字典序把 points_met 放在首位时的固有盲区。

#### 预注册判定门的诚实说明

出分后对照本文件预注册的三分支：**门 1（胜出）未触发**；**门 3（判负：points_met<7 或守卫回滚）字面上也未触发**（四检查点全 7/7、两臂零回滚）；**门 2（持平：Δdeliv 在 ±0.002 内）同样未触发**（Δ = −0.0048 ~ −0.0073，超出噪声带且方向一致向下）。即"守住 7/7、无回滚、但 deliv 系统性低 0.005 量级"这一组合**在预注册里未定义——属本次预注册的疏漏**。按实质判定：**M5.1 无增益，方向关闭**（不以任何事后放宽的门为 A2/A3 追认胜出）。

#### M5.2 候选（按证据强度排序）

1. **修守卫（最高优先，A2 实锤）**：mini-eval 加队列判据 + 提 episode 数（4→16）+ 改在课程全难度分布（含 load_ratio→1.3）上评，而非固定 3 点；否则后续任何微调臂都可能在"全绿"中烂掉
2. **把隐式信任域变显式**：既然 Arm A 的收益来自"被冻在 BC 盆地附近"，直接上 **KL-to-BC 信任域**或**加长/加强锚定**（当前 anchor coef 0.2、decay 0.999），把 lr 恢复到正常退火——分离"步长小"与"离 BC 近"这两个当前被下限机制耦合在一起的因素
3. **深过载区定向**：25/30dB 是所有检查点的失分集中区（A/best_guard 30dB 也只有 0.888），可针对该区间做硬例挖掘/课程加权，而非全局调参
4. ~~复核逐点 pass 门的队列宽容度~~ **已结案**：判据本就含 `queue ≤ SDMA×1.1`（`optimize_loop.py:1777` + yaml:24），7/7 合法。真正遗留的是把**余量厚度**（当前冠军 78.1%、A3 92.6%、A2/final 97.0%）纳入守卫判据——见候选 1
5. **基线切换决策**（`optimize_loop_config.yaml` baseline → A/best_guard.pt）仍待用户拍板——M5.1 之后已无更优候选，此决策的前置条件已满足

---

## M5.2 预注册（2026-09-08）：把隐式信任域变显式 + 守卫判据升级

**用户决策**：M5.1 终审后选定"M5.2 继续"，并同时批准基线切换至 A/best_guard.pt（已改 `optimize_loop_config.yaml:baseline.checkpoint`；该键经查是 `optimize_loop.py:124/2180/2317` 的实际生效值，原值 `logs/opt_best/best.pt` 为悬空路径）。

### 依据（M5.1 终审三条结论 → 设计映射）

| M5.1 结论 | M5.2 设计回应 |
|---|---|
| Arm A 的收益来自被 lr 下限冻在 BC 盆地附近（隐式信任域）；释放 lr（A3）在 25/30dB 丢回增益 | **F6 显式 KL-to-BC 信任域**：lr 走正常退火（`kl_stop_lr_decay=1.0`），盆地约束改由 `relu(log π_BC − log π_θ)` 惩罚承担（D1）；对照臂用**持久强锚定**（coef 0.2→0.6、decay→1.0）承担同一职责（D2） |
| 守卫 ok 判据只看 avg_deliv，A2 策略爆炸（KL=3e4）仍 19/19 全绿 | **F5b 守卫 v2**：ok = avg_deliv 门 AND 逐点门全过 AND avg_queue ≤ bc 参考×1.1；点集 3→6 点（补 15/20/25dB 深过载区）；episodes 4→16 降 mini-eval 噪声 |
| 失分集中在 25/30dB 深过载区 | 守卫点集纳入 25dB；判定门逐点检查 25/30dB |

### 双臂设计（各自对 Arm A 单变量对照；共同项 = 守卫 v2 + 正常 lr 退火）

| 臂 | exp_name | 相对 Arm A 的改动 | 回答的问题 |
|---|---|---|---|
| **D1** | `m52_armD1_klbc` | `--kl-stop-lr-decay 1.0 --kl-bc-coef 1.0` + 守卫 v2（6 点×16ep） | 显式信任域能否在正常 lr 下**保住** 25/30dB 的深过载行为（M5.1 证伪项的正面回答） |
| **D2** | `m52_armD2_anchor` | `--kl-stop-lr-decay 1.0 --solver-expert-anchor-coef 0.6 --solver-expert-anchor-decay 1.0` + 守卫 v2 | 复用既有锚定机制（更便宜）能否达到同一目的；与 D1 对照区分"KL 信任域"与"演示锚定" |

两臂的**共同改动**（守卫 v2 + 正常退火）本身也是一次消融：若 D1/D2 都 ≤ Arm A，则说明 Arm A 的冻结不可被显式约束替代（信任域假设证伪，转向候选 3 深过载定向）；若某臂 ≥ Arm A，则隐式信任域成功显式化，lr 得以解放。

### 判定门（预注册，出分后不改）

参照系 = **Arm A/best_guard `[7, 0.9519, −8.3971, 1.0804]`**：

1. **胜出**：某臂 `best_guard.pt` 字典序压过参照系 ⇒ 新交付物；且要求 25/30dB 逐点 delivery 不低于参照系（防止"均值赢、深过载输"的 M5.1-A3 式假胜）
2. **持平**：points_met=7 且 avg_deliv 在 ±0.002 内 ⇒ 信任域假设部分成立（显式约束复现了冻结的效果但未超越），M5.3 沿"加强约束强度/扩大点集"推进
3. **判负**：points_met<7、或守卫出现回滚、或 25/30dB 逐点低于参照系 1pt 以上 ⇒ 记录"显式信任域不足以替代冻结"，交付物仍为 Arm A/best_guard
4. **守卫 v2 自身的验收**（独立于胜负）：训练期任一臂若出现 KL>1e3 或熵 < −70 的 update，守卫必须在**同一次或下一次**检查中判 ok=False——A2 当年 19/19 全绿即 v2 要消灭的行为；若 v2 仍漏判，守卫修复判失败
5. best.pt（MeanReward 选点）一律不作判定依据（K8 教训）

### 工程前置（本会话已完成）

- `src/config.py`：PPOConfig +4 字段（`kl_bc_coef`、`eval_guard_v2`、`eval_guard_queue_ratio`、`eval_guard_snr_points`），全默认关/空 = 官方口径零变化
- `src/training/ppo.py`：`set_bc_reference()`（deepcopy 冻结 BC 副本）+ `_klbc_term()`（relu 单样本 KL 估计，带 π_θ 侧梯度）+ `_trust_grad_into()`（lex 路径梯度注入，同锚定语义）；标准路径 `loss += coef×term`；update() 返回 `kl_bc_loss/kl_bc_coef`
- `src/training/main.py`：`_GUARD_REF` 七点参考表（单一事实源，出处 `logs/_eval_m4_bc_init.txt`）+ `_guard_ref_avg()`；`_guard_check` v2 判据与可配置点集；BC 快照注入点放在**全部 BC 阶段之后、PPO 之前**（relabel 二次 BC 亦覆盖）；守卫日志补 avg_queue/n_points/v2
- `scripts/train.py`：4 个新旗标透传
- 测试：`tests/test_ft_protocol.py` +6 用例（信任域无操作/冻结/同策略为 0/漂移为正/注入梯度==autograd 梯度/归一化步下降；守卫 v2 抓 A2 型队列盲区、七点集单点失守、全绿放行），`-W error::UserWarning` 下全绿
- fast 门禁 **337/337**（331+6）
- 双臂冒烟（1 episode）验证中：`logs/_smoke_m52_D{1,2}.log`

### M5.2 终审（2026-09-09 14:00）：**双臂均判负，冠军不变；"显式信任域"假设在两种形式下均证伪**

双臂 8000ep 于 09-09 06:28 收官（控制台日志 mtime 核实）。4 检查点官方评估（7点×50ep，`logs/_eval_m52_*.txt`，`.err` 全 0 字节）：

| 检查点 | score [pts, deliv, −queue, ee] | 功率 | vs **A/best_guard [7, 0.9519, −8.3971, 1.0804]** |
|---|---|---|---|
| **D2/best_guard** (ep6092) | [7, 0.9473, −8.8929, 1.0782] | 13.71W | deliv −0.0046 ⇒ 负 |
| **D1/best_guard** (ep1628) | [7, 0.9447, −9.3543, 1.0760] | 13.69W | deliv −0.0072 ⇒ 负 |
| D2/final | [7, 0.9357, −10.6214, 1.0743] | 13.73W | 负 |
| D1/final | [7, 0.9231, −12.0743, 1.0646] | 13.75W | 负 |

排序：**A/best_guard 0.9519** > D2/best_guard 0.9473 > D1/best_guard 0.9447 > bc_init 0.9441 > D2/final 0.9357 > D1/final 0.9231。**交付物仍为 `logs/m5_armA_sftp/checkpoints/best_guard.pt`。**

#### 判定门逐条适用

| 门 | 适用结果 |
|---|---|
| 1 胜出 | 未触发（无人字典序压过参照系） |
| 2 持平（±0.002） | 未触发（D2 Δ=−0.0046、D1 Δ=−0.0072，同向超带） |
| 3 判负 | **D1 触发两条**：守卫回滚 5 次（>0）＋ 25/30dB 逐点 −1.7/−1.6pt（>1pt）；**D2 触发深过载条款**：30dB 0.877 vs 参照 0.888 = −1.1pt（25dB −0.8pt 未超 1pt 线），虽零回滚仍判负 |
| 4 守卫 v2 验收 | **通过**：D1 的 5 次 ok=False 全为 5dB 单点 delivery 0.977 < SDMA 门 0.979（−0.2pt，bc_init 该点 0.989 ⇒ 真回归 −1.2pt）；旧 v1 判据（只看六点均值 0.987≥0.916）**五次全漏**。A2 式"全绿中烂掉"已被 v2 消灭 |
| 5 best.pt 不作依据 | 遵守 |

#### D1 失效机制：单边 relu KL 估计是自我强化的锐化项（设计缺陷，已定位）

`_klbc_term = relu(mean(log π_BC − log π_θ))` 在 **on-policy 批**上最小化 = 把概率质量往"当前采到的动作"上推 ⇒ 不是保方差的信任域，而是锐化项。训练轨迹完全吻合：u80 熵 −75.4 → u~500 熵 **−99.5** → KL 尖峰 347–1515 → 守卫 4 次回滚后 actor 冻结（LR=0）。**M5.3 若再碰信任域必须换估计器**：连续头高斯闭式 KL（分布级、保熵）或双向/对称形式，禁用动作级单边估计。

#### 跨实验综合（M5.1+M5.2 合并读）：约束形式无关，lr 冻结本身才是有效成分

三个"释放 lr（`kl_stop_lr_decay=1.0`）+ 不同约束"的臂落在**同一条窄带**：

| 臂 | 约束形式 | avg_deliv |
|---|---|---|
| A3（M5.1） | 无约束 | 0.9471 / 0.9463 |
| D1 | KL-to-BC 惩罚（coef 1.0） | 0.9447 |
| D2 | 持久强锚定（0.6/decay 1.0） | 0.9473 |
| **Arm A（冠军）** | **lr 下限冻结（u100 起 6e-6）** | **0.9519** |

三种有限强度约束无一复现冻结的收益 ⇒ **伤害来自累积漂移（O(coef·T)），不是单步步长**；有限 coef 的惩罚在 780 个 update 上必然漏。冠军的"冻结"是强度无穷的信任域，这正是它不可被显式项替代的原因。**SFTP 的有效成分应重述为：早期 lr 冻结 + 守卫选点，而非"安全微调"的泛化叙事。**

#### 附带修复

`kl_bc_loss/kl_bc_coef` 原未进 `[Step` 周期日志（D1 训练期无法观察信任域量级，熵坍缩只能事后归因）——已在 `main.py` 补 `KLBC=` 条件字段；fast 门禁 337/337 复绿。

#### 全算法 benchmark 图（用户要求的交付物）

`logs/_eval_bench_all.txt`（冠军 + SDMA/NOMA/EqualPower，7点×50ep）→ `scripts/plot_benchmarks.py all` → **`logs/figs/benchmarks_all_algorithms.png`**（6 面板：delivery/queue/power/EE=R_sum/P/overflow/R_sum）。PPO-RSMA 在 delivery、queue、overflow、EE 四面板全域压过三基线；30dB delivery 0.888 vs SDMA 0.664 / NOMA 0.579 / EqualPower 0.603。消融图 `logs/figs/ablation_delivery.png`（8 检查点 + SDMA）。

#### M5.3 候选（按证据强度排序）

1. **定向冠军自身的两个弱点**（最高优先）：冠军逐点最低在 **0dB 0.886 与 30dB 0.888**，而所有 lr 释放臂恰在这两端丢更多。在**冻结 lr 配方不变**的前提下，对 0dB/30dB 做硬例挖掘加权或课程末端加权——不动已被证伪的 lr/约束轴，只动数据分布
2. **冻结强度消融**：把冻结点从 u100 提前/推后（u50/u200）、或把下限从 0.2×base 改为 0.1×/0.3×，量化"冻结时机×强度"对 0.9519 的贡献——这是当前唯一被证明有效的旋钮，值得画出响应面
3. **分布级 KL 信任域**（若仍要显式约束）：连续头高斯闭式 KL 替换动作级单边估计，coef 扫描 0.1/1/10；预注册门同 M5.2
4. 守卫 v2 已验收，纳入默认配方（后续所有臂默认 `--eval-guard-v2 --eval-guard-snr-points 0,5,15,20,25,30 --eval-guard-episodes 16`）

---

## M5.3 预注册（2026-09-09）：定向冠军弱点——唤醒对冠军失明的硬例挖掘

**用户决策**：M5.2 终审后选定候选 1（定向 0dB/30dB，冻结 lr 配方不动，只动数据分布）。

### 核心洞察：硬例挖掘对冠军**失明**

`src/training/main.py:30` `_HM_TARGETS` = SDMA 逐点 delivery = (0.856, 0.982, 0.960, 0.894, 0.805, 0.730, 0.668)。挖掘权重 `p_b ∝ base + w_hard·shortfall_b·streak_boost`，而 `shortfall = max(target − ema, 0)`。冠军逐点 (0.886, 0.989, 0.993, 0.986, 0.964, 0.942, 0.888) **每点都超 SDMA** ⇒ shortfall 全 0 ⇒ **零重加权、零 streak、零熵激励**。机制从 M4 起就处于空转状态——0/30dB 不是"推不动"，是"没人推"。

### 设计：逐点 target 覆盖（最小改动，复用全部既有 machinery）

`_hm_init_state(point_targets=…)`：7 值覆盖直接替换目标表；采样重加权、streak 倍增、performance-linked 熵激励、达标护栏**全部原样复用**。其余 5 点保持 SDMA 目标（已达标 ⇒ 零压力），压力**只**导向两个弱点桶（bucket 0 = [−2.5,2.5]dB、bucket 6 = [27.5,32.5]dB）。

| 臂 | exp_name | 0dB / 30dB 目标 | 回答的问题 |
|---|---|---|---|
| **E1** | `m53_armE1_hm030` | 0.92 / 0.93 | 温和进取目标能否把 0/30dB 抬过冠军（0.886/0.888）而不伤中段 |
| **E2** | `m53_armE2_hm030w` | 0.95 / 0.95 | 更进取目标的剂量响应；若 E2 劣于 E1 ⇒ 曝光挤占中段已发生 |

两臂其余旗标与冠军（Arm A）**逐位一致**（冻结 lr 配方：`--lr_actor 3e-5`、无 `--kl-stop-lr-decay`、anchor 0.2/0.999、warmup 30、Meta 冻结 2000），另加已验收的守卫 v2（6 点×16ep；冠军轨迹在 v2 下亦应零回滚，故不构成混淆）。

### 判定门（预注册，出分后不改）

参照系 = **Arm A/best_guard `[7, 0.9519, −8.3971, 1.0804]`**：

1. **胜出**：字典序压过参照系 **且** 0dB、30dB 逐点 delivery ≥ 参照系（0.886/0.888）——本轮的目标就是抬这两点，均值赢而弱点不赢不算赢
2. **持平**：points_met=7 且 avg_deliv ±0.002 内 ⇒ 挖掘唤醒但未转化；检查训练期 `sample_count` 是否真的偏向桶 0/6（若偏了仍持平 ⇒ 曝光不是瓶颈，转向冻结响应面）
3. **判负**：points_met<7、或守卫回滚、或**中段 5–25dB 任一点逐点掉 >1pt**（防"偷中段曝光补两端"的零和失败）、或 0/30dB 反而低于参照系 1pt 以上
4. best.pt 一律不作判定依据

### 工程前置（本会话已完成）

- `src/training/main.py`：`_hm_init_state(target_scale, point_targets=None)`（7 值覆盖；长度不符/空回退默认，防配置笔误静默改口径）；启用打印回显实际目标表（`targets=[…](override)`）
- `src/config.py`：`hm_point_targets: tuple = ()`（默认空 = SDMA 目标，官方口径零变化）
- `scripts/train.py`：`--hm-point-targets` 逗号 7 值透传
- 测试：+3 用例（默认=SDMA 表 / 覆盖生效与回退 / 2000 次采样中弱点桶占比 >0.6 且已达标桶保留地板曝光），fast 门禁 **340/340**
- 双臂冒烟验证中：`logs/_smoke_m53_E{1,2}.log`

结果待补。

### M5.3 终审（2026-09-10 08:40）：**双臂判负，挖掘未抬起目标点；0.947 窄带与 lr 无关 ⇒ 种子方差假说升为最高优先**

4 检查点官方评估（7点×50ep，`logs/_eval_m53_*.txt`）：

| 检查点 | score [pts, deliv, −queue, ee] | vs **champ [7, 0.9519, −8.3971, 1.0804]** |
|---|---|---|
| **E1/best_guard** (ep5273) | [7, 0.9474, −8.7857, 1.0791] | deliv −0.0045 ⇒ 负 |
| **E2/best_guard** (ep6092) | [7, 0.9471, −8.9386, 1.0769] | deliv −0.0048 ⇒ 负 |
| E2/final | [7, 0.9460, −9.2257, **1.0860**] | deliv −0.0059（ee 为全场历史最高） |
| E1/final | [7, 0.9369, −10.6586, 1.0789] | 负 |

**逐点（best_guard vs champ）**：

| SNR | E1 | E2 | champ | E1 Δ | E2 Δ |
|---|---|---|---|---|---|
| **0 dB（目标点）** | 0.876 | 0.887 | 0.886 | **−1.0** | +0.1 |
| 5/10/15 dB | 0.987/0.992/0.987 | 0.989/0.993/0.991 | 0.989/0.993/0.991 | −0.2/−0.1/−0.4 | 0/0/0 |
| 20 dB | 0.971 | **0.963** | 0.974 | −0.3 | **−1.1** |
| 25 dB | 0.938 | **0.926** | 0.942 | −0.4 | **−1.6** |
| **30 dB（目标点）** | 0.881 | 0.881 | 0.888 | **−0.7** | −0.7 |

#### 判定门适用

- **门 1 胜出**：未触发
- **门 2 持平**：未触发（Δ=−0.0045/−0.0048 超 ±0.002 带）
- **门 3 判负**：**E2 触发两条**（训练期守卫回滚 1 次 + 中段 20/25dB −1.1/−1.6pt > 1pt）；**E1 按字面未触发**（0 回滚、中段最大 −0.4pt、0dB −1.0pt 未"超过"1pt 线）——但 E1 的**干预目标点不升反降**（0dB −1.0、30dB −0.7），按干预自身目的判**实质判负**
- **门模板第三次缺口坦白**：预注册三分支又漏了"未回滚、中段完好、但目标点反向"这一组合（M5.1 漏"7/7 但低 0.005"、M5.2 靠深过载条款兜住、本轮靠实质判定兜住）。**后续预注册必须加兜底分支："其余一律判负/无增益，不追认"**

#### E 轮非干净性坦白

E 臂相对冠军改了**两处**：采样分布（挖掘目标）**与守卫选点口径**（v2 六点×16ep vs 冠军 v1 三点×4ep）。我预注册时论证"v2 不构成混淆"只考虑了回滚行为，**漏了选点度量本身改变 ⇒ best_guard 是不同度量下选出的不同检查点**。故 E 轮不是挖掘假设的单变量检验；但"E1 目标点反降"这一结论不依赖选点口径，仍然成立。

#### 元结论（本轮最重要产出）：0.947 窄带与 lr 无关

把五轮偏离冠军的臂排在一起：

| 臂 | 偏离维度 | lr 机制 | best_guard deliv |
|---|---|---|---|
| A3 | 释放 lr、无约束 | 正常退火 | 0.9463/0.9471 |
| D1 / D2 | 释放 lr + KL 信任域 / 强锚定 | 正常退火 | 0.9447 / 0.9473 |
| **E1 / E2** | **仅采样分布（lr 未动）** | **涌现冻结（同冠军）** | **0.9474 / 0.9471** |

E 臂保留了冠军的涌现冻结，仍落在 0.947 带 ⇒ **该窄带不是"释放 lr"的后果**。五种互不相同的干预全部收敛到 0.946–0.9474，而冠军 0.9519 孤悬其外且**从未被重复过**（seed 42 单次抽样）。两种解释：

- **(a) 种子方差假说**：冠军配方的真实均值 ≈0.947，0.9519 是 +1σ 幸运抽样 ⇒ 此前所有"X 比冠军差 0.005"的结论都是在和噪声比；"冻结/锚定/守卫是有效成分"的叙事整体降级为未证
- **(b) 脆弱最优假说**：配方真实均值确为 0.952，任何机制扰动（含数据分布）都把轨迹推离它 ⇒ 则冻结响应面仍是正确探针

**(a) 可一次性裁决且代价最低：用冠军配方逐字复跑 seed 43/44。** 若复跑 ≈0.947 ⇒ (a) 成立，论文叙事改为"SFTP 稳定产出 ~0.947±0.003，显著优于 bc_init 0.9441 与历史基线 0.9163"，冠军点降级为单抽样最佳；若复跑 ≈0.952 ⇒ (b) 成立，M5.5 再上冻结响应面。**在裁决前冻结响应面（原 M5.4 草稿）暂缓——在噪声上测响应面无意义。**

---

## M5.4 预注册草稿（2026-09-09，**暂不点火**——用户指示等 E 终审）：冻结响应面

### 零算力前置（已完成）

1. **测量缺陷修复**（M5.3 教训）：重加权采样后批熵跨臂不可比（混入批构成偏移）。`ppo.py` update() 现返回 `entropy_lo/mid/hi`（<7.5 / 7.5–22.5 / ≥22.5 dB 三 band）与 `log_std_mean`（**与采样分布无关，是真坍缩的唯一干净判据**）；`main.py` `[Step` 日志新增 `ENT=[lo,mid,hi] LS=x.xx`，冻结后追加 `FZ` 标记。E 臂当前日志无此字段（改动在 E 启动之后），E 的熵读数仍不可判读——这正是等评估的原因
2. **涌现式冻结显式化**：冠军 Arm A 的"冻结"是 KL 衰减连乘触底 `lr_decay_final_ratio` 下限的涌现结果（u100 起钉死），既不可控也不可消融。新增 `--lr-freeze-update N`（`config.ppo.lr_freeze_update`，`ppo.py:_apply_lr` 冻结分支）：达到 N 个 update 后 scale 恒等于 `lr_decay_final_ratio`，进度退火与 KL 衰减均不再作用。0=关，行为逐位不变。测试 3 用例（钉死/默认无操作/band 拆分正确），fast 门禁 **343/343**

### 设计：onset 单旋钮响应面（3 臂 + 冠军参照）

所有臂配 `--kl-stop-lr-decay 1.0` **关掉涌现通道**，使冻结点成为唯一 lr 机制；其余旗标与冠军逐位一致 + 守卫 v2 + 新诊断日志。

| 臂 | 冻结点 | 回答的问题 |
|---|---|---|
| **F1** `m54_armF1_fz50` | u50 | 更早冻结是否更好（BC 盆地保护窗口前移） |
| **F2** `m54_armF2_fz100` | u100 | 显式冻结能否**复现**冠军的涌现冻结（F2≈冠军 ⇒ 显式化成功，后续实验可摆脱对 KL 爆表的依赖） |
| **F3** `m54_armF3_fz200` | u200 | 冻结太晚是否扼杀有益微调（F3>冠军 ⇒ 前 100–200 update 里有真增益） |
| 参照 | 涌现（KL 衰减触底） | Arm A/best_guard `[7, 0.9519, −8.3971, 1.0804]` |

### 判定门（预注册，出分后不改）

1. **复现门**（F2 vs 冠军）：F2 与冠军 avg_deliv 差在 ±0.002 内 ⇒ 涌现冻结被成功显式化——**此门独立于胜负，是本轮最主要的科学产出**（把"KL 每轮爆表"这个不健康的涌现机制替换为可控旋钮）
2. **响应面读法**：F1>F2 ⇒ 保护窗口应前移，M5.5 扫 u20/u50；F3>F2 ⇒ 前段微调有真增益，M5.5 扫 u100–u400；单调或平坦 ⇒ onset 不是杠杆，转向 floor 旋钮（需新增 `--lr-decay-final-ratio` 旗标）与 0/30dB 定向的交互
3. **胜出**：任一臂字典序压过冠军 **且** 0/30dB 逐点不降（沿用 M5.3 门 1）
4. **判负**：points_met<7、守卫回滚、或中段 5–25dB 掉 >1pt
5. 用 `LS=`（log_std）而非批熵判读探索健康度；best.pt 不作判定依据

### 点火条件

E1/E2 官方评估出分并完成 M5.3 终审之后。若 E 证明挖掘有效 ⇒ M5.4 臂叠加 E 的 target 覆盖（否则响应面在错误的数据分布上测量）。

---

## 种子裁决终审（2026-09-11 00:35）：冠军既非 +1σ 幸运，也非刚性不动点 ⇒ 比较口径需升多 seed 统计

### 执行

冠军配方（m5_armA_sftp）逐字复跑 seed 43/44（`mAdjudS43/44`）：8000ep、**0 回滚、0 次 ok=False**、stderr 0 字节。训练期实证：两臂 `KLstop=1` 全程 → LR 钉死 6.00e-06（涌入冻结逐字重现）；`LS=-2.11/-2.16` 全程稳定（无 log_std 坍缩），守卫 v1 3/3 全绿。4 检查点官方评估（7点×50ep，`logs/_eval_mAdjud_*.txt`，stderr 全 0）。

### 官方分数

| 检查点 | score | vs 冠军 `[7, 0.9519, −8.3971, 1.0804]` |
|---|---|---|
| **S43/best_guard** | [7, **0.9503**, −8.4214, 1.0757] | deliv **−0.0016** ⇒ **复现带 ±0.002 内** |
| S43/final | [7, 0.9496, −8.7271, 1.0735] | −0.0023 |
| S44/final | [7, 0.9491, −8.9114, 1.0729] | −0.0028 |
| **S44/best_guard** | [7, 0.9414, −10.0414, 1.0696] | −0.0105（v1 选点盲区，见下） |

**S43/best_guard 逐点 vs 冠军**：0/5/10/15/20/25/30 dB = 0.881/0.989/0.993/0.990/0.975/0.940/0.884 vs 0.886/0.989/0.993/0.991/0.974/0.942/0.888 —— 全程 ±0.5pt 内、方向随机（0dB −0.5、20dB +0.1、25dB −0.2、30dB −0.4），与噪声特征一致。

### 裁决（对 M5.3 终审 (a)/(b) 二分）

- **假说 (a)（种子方差，"冠军真实均值≈0.947"）强形式被证伪**：逐字复跑未落在 0.947。配方中心 ≈ **0.9494–0.9503**（final 对 {0.9496, 0.9491}、S43/bg 0.9503），冠军 0.9519 属该分布**上缘但不异常**——S43 在 ±0.002 容差内复现了它。
- **假说 (b)（脆弱最优，"任何扰动都会推离它"）不严格成立**：同配方 seed 44 的 best_guard 官方分掉到 0.9414（而 final 0.9491）——**配方自身就有下尾**，冠军不是刚性不动点，扰动不是把它推离的唯一路径。
- **合成结论（本轮核心）**：σ_seed ≈ **0.005**（best_guard 三抽 {0.9519, 0.9503, 0.9414}）。M5.1–M5.3 五个干预臂的"输 0.0045–0.0073"**全部落进冠军自身的 seed 噪声内**——释放 lr / KL 信任域 / 强锚定 / 采样分布，**在 n=1–2 的可分辨度下均无处理效应**。"冻结 lr 是有效成分"叙事不可由臂间差证实，只能重述为"配方对 seed 稳健（42/43 顶复现）、扰动落在其噪声带下尾"。
- **统计门槛升级**：任何"X 比 Y 好/差 0.003–0.005"的声明需要多 seed（n≥3，理想 5）+ 业务级 CI。M5.x 全部单 seed 胜负判定不达此门槛；论文须改为多 seed 比较或降级为区间声称。

### S44/best_guard 的 v1 盲区复现

守卫 mini-eval（3 点 × 4ep）在 ep6092 选中了官方口径下仅 0.9414 的检查点（15/20/25dB 中段 0.984/0.962/0.924 vs 0.991/0.974/0.942），而同 seed final 有 0.9491。守卫 v2 的六点×16ep 本可抓住——**后续任何臂必须带守卫 v2**（留此 note 说明逐字冠军复跑为何用 v1：保持与 0.9519 同口径）。

### M5.4 决策（冻结响应面）

**不建议单 seed 点火**：F1/F2/F3 的 onset 效应若 ≤0.003，在 ±0.005 的 seed 噪声下 n=1 必不可辨。两条出路：
1. **多 seed 响应面**：3 冻结点 × 3 seed = 9 臂（~3 天算力），区分 onset 效应与噪声；
2. **停止新臂，按当前证据写论文**：交付物不变（best_guard.pt 冠军），叙事改为"SFTP 协议产出 ~0.950±0.005（seed 42/43 复现，seed 44 下尾 0.941 为 v1 选点盲区修正后 0.949）、五类干预均无达到 seed 分辨度的处理效应"。

待用户定夺；M5.4 预注册草稿原文与判定门保留（若选 1 则追加"每臂 n≥3"条款）。

---

## M5.4-M 预注册（2026-09-11，用户选定多 seed 冻结响应面）：3 onsets × 3 seeds = 9 臂

### 设计（基于种子裁决升级）

| 臂 | 显式冻结点（`--lr-freeze-update`） | seeds |
|---|---|---|
| F1 `m54_F1_50` | u50 | 43 / 44 / 45 |
| F2 `m54_F2_100` | u100 | 43 / 44 / 45 |
| F3 `m54_F3_200` | u200 | 43 / 44 / 45 |

- **配方** = 冠军逐字基座（8000ep / curltrac / no-bucket-experts / solver-expert 40ep / anchor-decay 0.999 / lr_actor 3e-5 / lr_log_std 1e-4 / clip 0.1 / warmup 30 / meta-freeze 2000）+ **`--kl-stop-lr-decay 1.0`**（关掉涌现通道，使 onset 成为唯一 lr 机制）**+ `--lr-freeze-update N`** + **守卫 v2 强制**（`--eval-guard-v2 --eval-guard-snr-points 0,5,15,20,25,30 --eval-guard-episodes 16`，S44 的 v1 选点盲区教训——best_guard 必须用 v2 选）+ 默认 SDMA 挖掘目标（E 轮已判负，**不加** E 的 target 覆盖）
- **调度**：3 波，每波 1 seed（F1/F2/F3 并起）。内存约束（本机空闲 RAM 常 <3GB，每臂≈1.5GB）不允许 9 并行；逐波也给"seed 43 波后提前读面"的自适应机会
- **评估**：每臂 best_guard + final 双检查点官方 7点×50ep；**主判定用 final**（无选点口径、跨 v1/v2 设置可横比），best_guard 作部署件辅助列——两列全报

### 判定门（预注册，出分后不改）

1. **复现门**（F2 vs 冠军参照 n=3 {0.9519, 0.9503, 0.9414}）：F2 final 单元均值与参照均值 |Δ|<0.003 且 CI 重叠 ⇒ 显式冻结复现涌现冻结，机制显式化成立（后续可摆脱对 KL 爆表的依赖）
2. **响应面门**：onset 效应判据 = max(mean F1,F2,F3) − min(...) > 3×pooled SE_cell（约 0.006–0.009）。显著时读方向：F1 最强 ⇒ 保护窗口前移（M5.5 扫 u20/u50）；F3 最强 ⇒ 前段微调有真增益；平坦 ⇒ onset 不是杠杆（转 floor 旋钮，需新增 `--lr-decay-final-ratio`）
3. **胜出门**：任一单元 final 均值超冠军参照 >2×SE **且** points=7/7 ⇒ 新交付物
4. **判负门**（逐臂）：points<7、守卫回滚、中段 5–25dB 掉 >1pt
5. **自适应复制条款**：seed 43 波出分后若面已平坦/方向明确，用户可停止→省 2/3 算力；不确定才续 44/45 波

### 点火

`logs/_m54_launch.ps1 -seed 43`（之后 44、45）→ 每波 ~10h → 收尾评估脚本 `logs/_m54_eval.ps1`。监视 `LS=` / `FZ`（冻结位点与 onset 值对齐）、守卫 v2 六点判定。

---

## 扩展对照评估终审（2026-09-12）：免训练逐槽优化 = 本网格新上界

官方口径 7点×50ep（`logs/_eval_extended.txt` / `_eval_extended_causal.txt`，脚本
`scripts/evaluate_extended_baselines.py`，仅 import 不可变评估脚本）：

| 对照 | mean deliv | 30dB deliv(Q) | 备注 |
|---|---|---|---|
| **DiffOpt**（oracle 逐槽可微优化） | **0.9574** | 0.905 (24.75) | 7/7 门全过；逐点超冠军 |
| **DiffOptCausal**（仅 Ĥ，因果公平） | **0.9574** | 0.905 (24.74) | 与 oracle 逐点 |Δ|≤0.001 ⇒ 优势非 CSI 泄漏 |
| 冠军 A/best_guard | 0.9519 | 0.888 (29.63) | 被逐点超越 |
| AdaptivePower / WaterFilling | ~0.905 / ~0.905 | 0.777 / 0.778 | 远低于两者 |

**结论（叙事重定位，已入论文 draft/main.tex §V-E+§VI）**：可微引擎+solver expert
本身是本网格最强控制器（模型上界）；BC+SFTP = 将其压缩为单前向快速策略
（0.9574→0.9441→0.9519，收复 ~59% 压缩 gap）；学习策略的剩余价值 = 推理成本与
免模型部署。**"RL 超越教师"措辞撤回**（RL 超的是 BC 学生；活优化器此前从未官方评估）。
因果开关 `SolverExpert(use_true_channel=False)` + 2 测试（门禁 345）。

**在跑**：LDPTRLQ-style 对照臂（单时间尺度 Lyapunov-DPP RL，`--disable-meta`，
8000ep 已收官 `logs/b_ldptrlq_style/`）官方评估 best/final 两检查点
（`logs/_eval_ldptrlq_{best,final}.txt`，~2-3h）→ 出分后并入表 I/III。

### LDPTRLQ-style 对照臂终审（2026-09-12）：0/7 门，全面崩溃

- 配方：冠军基座 **−meta（单时间尺度） −solver-expert（无蒸馏） −guard**，余同（curltrac/mining/DGLCR/Lagrangian/DVQ），8000ep seed 42（`logs/b_ldptrlq_style/`）
- 官方评估（`logs/_eval_ldptrlq_{best,final}.txt`）：**best [0, 0.7824, −18.4300, 0.6838]**、final [0, 0.7839, −17.6143, 0.6892]；功率 23.2/22.9W≈饱和；逐点 0dB 0.753 … 30dB 0.595，**低于 SDMA（mean 0.838）**
- 结论：2025 族单时间尺度 Lyapunov-DPP RL 从零训练**无法解决本过载实例**；与 bc_init（纯蒸馏 7/7）对照 ⇒ **使能成分是专家初始化**，非奖励族/层次结构单独所致。已入论文 §V-E + 主图（LDPTRLQ 列）
