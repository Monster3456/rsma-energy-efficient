# PRD: 求解器专家 + 专家迭代 — 可微速率引擎驱动的长期时隙 RSMA 优化

## 需求背景

当前最优成绩 Run 3 `[5, 0.9041, -12.59, 0.6985]`（追平基线 pass 数），10 轮实验的结论链指向：**瓶颈在策略容量与教师信号，不在训练量/超参/机制改造**（Run 6/7/9 三轮独立证实）。上阶段研究（DRL-RSMA，本机 `F:/PyCharm/project/DRL-rsma`）的可移植核心结论：

1. **BC 蒸馏 >> 直接 DRL**（同预算 DRL 落后 oracle 34-54%，GNN-BC 达 93-98%）
2. **测试时/逐实例精修对初始化极稳健**，瓶颈是步长不是步数（lr 0.1 × 少步最优）
3. 负结果避坑：DAgger 在 iid 状态下退化（本项目状态非 iid，专家迭代成立）、堆容量/数据无效、蒸馏目标必须用强解（K8 自蒸馏弱目标更差）

长期性适配（区别于上阶段瞬时优化）：专家不解"逐实例最优"（不存在——最优依赖值函数），而是**每步在线解单槽 DPP 代理问题**——本环境 reward 本身即 drift-plus-penalty 结构（lyapunov.py DGLCR 分支），逐槽最大化该 surrogate 的策略对长期目标有结构背书。跨槽预测性调度留给 PPO 微调（ExIt 经典分工）。

## 关键洞察（探索期发现）

- **softmax 份额下总功率其实可控**：p 块 logits ∈ [-1,1]，logit 展宽可使总功率从 24W（均匀）压到 ~8.8W（极端展宽）——Run 4"功率钉死"是梯度问题不是动作空间问题，专家求解器在现有动作空间内即可省电
- **α 块在默认配置下是死维**（group_level_alpha=True 时被 split_mask 覆盖）；d_ratio 在 surrogate 上单调最优为 1——专家的真实优化变量是 **p_logits（K 维）+ SIC 顺序**
- **HEK-Lite BC 基础设施已存在且默认关闭**（"已验证无效"——因其规则专家弱：近均匀功率）；本任务用梯度求解器替换规则专家，复用 `_hek_collect_expert` 收集模式 + `bc_pretrain` + 衰减锚定三件套

## 方案（三层递进，逐层设验证门）

### M1: 可微速率引擎 `src/optimization/diff_engine.py`

纯 torch 复刻 `rate_calc.compute_rates_mmse_sic / compute_outage_rates_mmse_sic` + `beamforming.compute_precoding_v2`：
- MRT 方向 = H 右主奇异向量（批量 SVD，**无梯度需求**——H 对专家是常量，no_grad 计算即可精确一致）
- 两阶段 MMSE-SIC 向量化：R_int 用 (K,K) 掩码矩阵 M（由 rank×group_ids 构造）einsum 批量组装，torch.linalg.solve 批量解
- outage 模式 = min(declared[H_hat,ε正则], actual[H_true,无正则])
- dtype 可配（complex128 测试 / complex32... 实际用 float64 vs float32）保证一致性测试容差可控

**门 1**：`tests/test_diff_engine.py` 与 rate_calc 数值一致（float64, atol 1e-8 量级）：随机信道 × {完美/不完美 CSI} × {多种 order/group/eps/β}；端到端与 env.step 的 R_total 一致。

### M2: 专家求解器 `src/optimization/solver_expert.py`

- `SolverExpert.act(env)`：读 env 内部状态（H_hat/H_true/Q/split_mask/group_ids/sigma_est/dinkelbach_eta/Y/X/λ/QoS 权重），对 p_logits（tanh 参数化）跑 Adam（lr≈0.1，~25 步，暖启动）最大化 **surrogate = lyapunov DGLCR 活跃分支的精确镜像**（常数从 env.lyapunov 实例读取，不用 config 副本）
- d=1 固定、α 块置零（死维）；SIC 顺序：gain_desc / queue_desc / best_of（两候选各优化取优）
- overflow 期望近似用 λ（Poisson 均值）

**门 2**：`scripts/_diag_solver_expert.py`（诊断脚本模式，不跟踪）——专家在真实 env 跑 episode，对照 SDMA 与 Run 3 策略：**专家 delivery ≥ SDMA @ 0-5dB**（Run 3 的缺口区）且总功率 < 24W。不达标则先修 surrogate 再进入 M3。

### M3: 专家迭代训练（复用 HEK 通道）

- `main.py`：新增 solver_expert 收集路径（镜像 `_hek_collect_expert` 结构，含 PL 顺序收集）；config 新增 `solver_expert_*` 字段（默认全关 = 官方口径零变化）
- `ppo.py`：`bc_pretrain` 扩展可选 PL 头训练（decoding_orders → pl_log_prob 损失项）
- 收集口径：eval 配置（snr_min=-5, load_max=1.3）下收集 N≈30-60 episodes（不受 CuRLTraC 初始易档影响），BC 预训 → PPO 8000ep（Run 3 配方：h4 + no-bucket-experts + curltrac + seed 42）
- **门 3**：smoke-train 跑通 + 快速门禁全绿 → 全量训练 → `evaluate_snr_sweep_temporal.py` 评估，对照 Run 3 `[5, 0.9041, -12.59, 0.6985]` 与基线 `[5, 0.9163, -11.0186, 0.6914]`；K8 教训——纯 BC 检查点与 BC+PPO 检查点都评估

## 改动范围（全部在 optimize_loop_config 白名单内）

- `src/optimization/diff_engine.py`、`src/optimization/solver_expert.py`（新增，白名单目录）
- `src/config.py`（full）：`solver_expert_*` 字段，默认关
- `src/training/main.py`（full）：收集路径 + 门控
- `src/training/ppo.py`（full）：bc_pretrain PL 扩展
- `scripts/train.py`：CLI 透传（`--solver-expert-*`）
- `tests/test_diff_engine.py`、`tests/test_solver_expert.py`（新增，标 fast）
- `scripts/_diag_solver_expert.py`（诊断模式，不跟踪）

不可触碰：channel.py / rate_calc.py / evaluate_snr_sweep.py / rsma_env.py 物理段（本任务**零 rsma_env.py 改动**——所有接入在 optimization/training 层读取 env 状态）。

## 验收标准

1. 默认配置全量测试全绿（零回归，solver_expert 默认关 = 逐 bit 不变）
2. M1 一致性测试通过（引擎 vs rate_calc）
3. M2 专家诊断达标（低 SNR delivery ≥ SDMA、功率可降）
4. M3 训练评估 ≥ Run 3（pass 数 ≥ 5 且字典序不劣），或给出明确的失败归因写入 experiment-results.md
5. 每次改动后快速门禁 `pytest -m fast` 通过

## 已确认决策（2026-09-02 用户确认）

- 路线：求解器专家 + 专家迭代（用户批准"按照探索优化我现在长期的优化算法和方案"）
- 训练口径：官方口径（adaptive λ + softmax ON + h4 + no-bucket-experts + curltrac + seed 42），与 Run 3 直接可比
