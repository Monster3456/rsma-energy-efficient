# 近期进展总结：Benchmark 对比 + 性能优化（2026-09-20 ~ 2026-10-13）

> 分支：`wip/temporal-worker-solver-expert`（相对 `main`）
> 本文档对应论文 `docs/paper/main.tex`，所有图/表位置均为 `main.tex` 中的**行号**与变量 label。
> 图资产统一在 `docs/paper/figs/`。核心结论亦可追溯 `.trellis/tasks/*/experiment-results.md` 与 `logs/opt_knowledge.json`。

---

## 一句话总览

**算法准确率没有新的增量**（冠军 0.9519 不变）；半个月的实际进展是：**(1)** 把论文评估面补齐到评审标准（TD3/DDPG/SAC 经典三件套 + 模型基/优化深度双上界 + 统计检定），**(2)** 新增事后可解释性分析模块（SVEA + KernelSHAP/EG），**(3)** 在**观测空间**做出唯一一个转正的性能优化（A-prune 常数块剪枝，obs 13→10 块，零性能代价换 −12% 维度）。可解释性驱动的其余 3 条优化线（早停 / 头裁剪 / H 窗口缩容）全部失败或被用户关闭。

---

## 一、算法 Benchmark 对比（评估面补齐）

### 1.1 经典连续控制三件套 (TD3 / DDPG / SAC) — 09-21 ~ 10-01

- 动作空间适配：混合 PL + 连续动作（延迟 actor 给 TD3、单 Q 给 DDPG）；单时间尺度、无蒸馏、无守卫。
- 提交：`dc31543`（09-21 TD3/DDPG 预注册）、`8ac9df0`（09-30 SAC：`--no-perturb plain-SAC` + CUDA wiring）、`0d08b02`（10-01 SAC 定稿 0/7、0.787）。
- 结果：**TD3 0/7（0.818，24W 饱和）、SAC 0/7（0.787，逐点队列失门）、DDPG 5/7（0.886，仍缺两个低 SNR 门）**。

### 1.2 上界对照组 — DiffOpt（oracle 教师） / JointOpt（优化深度）

- **DiffOpt**（我们的可微引擎逐槽在线优化，无学习政策）：0.9574 > 冠军 0.9519（+0.0055），越出 ±0.005 seed 带。causal 变体（仅用估计信道）逐点复现 oracle ≤0.001 → 优势非 CSI 泄漏。
- **JointOpt**（60 步 warm-started）：反而**低于** 25 步冷启动 DiffOpt（0.913 vs 0.957）→ gap 归因于"逐槽在线优化置换为单前向推理"（`main.tex:708-721`）。
- **论文口径（用户拍板，勿改回）**：DiffOpt 只作表 III/IV 的 "oracle 教师上界" 行 + † 脚注，**不再画进主图**。主图故事 = Ours 赢一切。

### 1.3 统计检定

- 配对 t / Cohen's d / Holm-Bonferroni（`35504ed` 09-28）→ RL 臂 p≤0.006、d≥1.6；NOMA 不显著。
- 5-seed CI：baseline 分箱带（champion 同 5 次抽样 std 0.0047）。

### 1.4 相关基线扩充

- 文献扩到 41 篇（09-29）；补 mis-specification 鲁棒性章节 + 信息集/预算解耦 + 复杂度/成本表。

### 📍 图/表位置（Benchmark）

| 内容 | 位置 (`docs/paper/main.tex`) | 资产文件 (`docs/paper/figs/`) |
|---|---|---|
| **Master benchmark 主图**：Ours(5-seed min-max band) vs SDMA/NOMA/equal-power/adaptive-power/LDPTRLQ，6 面板（delivery/queue/power/EE/overflow/sum-rate）| 小节 521 `Full-grid benchmark`；`\includegraphics` **行 555**；figure label `fig:band`（**行 568**）| `benchmarks_master.png` |
| 训练曲线（rollout reward vs PPO update，5-seed band + 干预臂 + LDPTRLQ 崩塌）| `fig:train`（**行 579**），`\includegraphics` **行 573** | `training_curves.png` |
| 逐点 delivery/power/queue 表（50ep/点）| `tab:bench`（**行 537**）| — |
| **Classic-family + 优化深度对比表（TD3/DDPG/SAC/LDPTRLQ/JointOpt/DiffOpt/ours）** | `tab:classic`（caption **行 728**）；叙述在 696-721 | — |
| 训练-free 对比表（DiffOpt / DiffOpt causal / AdaptivePower / WaterFilling / Champion）| `tab:free`（**行 658**）| — |
| 预注册微调变体表（M5.1–M5.3，全 lose）| `tab:campaign`（**行 626**）| — |
| 复杂度/成本表、mismatch 鲁棒性表 | `tab:complex`（**行 792**）、`tab:mismatch`（**行 823**）| — |

---

## 二、性能优化与消融（`10-09-interpretability` 任务线）

### 2.1 ✅ A-prune 观测常数块剪枝 —— 唯一转正项

- 思路：固定 SNR 评估点内，arrivals / λ 桶 / Dinkelbach-η 三个原生块零方差 → 移除（obs 13→10 原生块，603→531 维，worker 输入 27→22）。
- 判定（10-12 H0 接受 → 10-12 转正默认 True `38a32da`）：A-prune s43 0.9526 vs 对照 0.9527，**Δ=−0.0001 无处理效应**；**跨种子成立**（s44 满预算 7/7、0.9527，30dB Q=28.26 ✓）。
- 注意口径：冠军/论文为未剪 603 遗留口径，评估须显式 `--obs-prune-constant-blocks`（temporal wrapper 两侧支持）。

### 2.2 ❌ 其余三条优化线（均已关闭）

| 方向 | 结局 | 证据文件 |
|---|---|---|
| P1 平台早停（guard-plateau）| H0 **拒绝**：3 点 mini-eval 平台致早停过早，30dB 队列门 (Q=39.2>37.9) 失门；s44 满预算对照证明系**早停之过**（s44 本身 30dB Q=28.26 ✓），seed 效应排除 | `.trellis/tasks/10-09-interpretability/ablation_early_stop.md` |
| P2 决策头裁剪（fix_drain/fix_order/--n_groups）| ⛔ **closed-by-decision**（10-13 用户拍板不执行；代码保留默认关）| `ablation_head_pruning.md` |
| P3 历史 H 窗口缩容（H=4→更小）| ❌ **负结果**：H 不可缩，信息跨 lag 散布 | `experiment-results.md`（10-12 §P3）|
| SVEA 分析引擎性能 | 两阶段并行 LP（`--sv_workers`）+ psv_mlp 锁测试（10-13 `ae0915e`）| `scripts/interpretability_analysis.py` |

> 消融证据主档：`.trellis/tasks/10-09-interpretability/ablation_prune_constants.md`（A-prune）、`ablation_early_stop.md`（P1+s44 对照因果归因）。

### 📍 图/表位置（消融）

| 内容 | 位置 | 资产 |
|---|---|---|
| Delivery over lineage（纯 BC / champion SFTP / S43/S44 复现，SDMA 虚线参照）| 小节 582 `From teacher to distilled policy`；`fig:abl`（**行 595**），`\includegraphics` **行 591** | `ablation_delivery_seeds.png` |

---

## 三、可解释性分析模块（SVEA + KernelSHAP/EG）— 10-09 起

- **SVEA 全局特征子集选择**（Tripathi 2020）：13 特征块 hinge-LP 分类博弈，SVEA_neg 支配子集 + t-CI + PSV。
- **局部双估计器**（Fernando 2019 DeepSHAP 族）：Expected Gradients vs KernelSHAP，四种参照敏感性 Jaccard + recall@k 验证。
- 代码：`src/analysis/` + `scripts/interpretability_analysis.py`；测试 `tests/test_analysis_*.py`（fast 16 例）。
- 已写入论文 `\subsection{Interpretability: what the policy attends to}`（`main.tex:841`）+ 2 个 bibitem。

### 📍 图/表位置（可解释性，论文 §7 `main.tex:841` 起）

| 内容 | 位置 | 资产 |
|---|---|---|
| SVEA per-feature 分数（逐 SNR 95% CI，红=SVEA_neg）| `fig:svea`（**行 909**），`\includegraphics` **行 907** | `fig_svea_intervals.png` |
| Top-5 Jaccard（EG 参照变体 vs KernelSHAP，每 target×SNR）| `fig:overlap`（**行 915**），`\includegraphics` **行 913** | `fig_topk_overlap.png` |
| 逐 SNR SVEA_neg 支配子集 + PSV | `tab:svea`（**行 921**）| — |
| EG-vs-KernelSHAP recall@5（power target，参照敏感性）| `tab:recall`（**行 939**）| — |

> ⚠️ 已入库**未接线**的资产：`figs/fig_attributions.png`（SVEA/EG 归因图）与 `figs/algorithm_flowchart.png`（算法流程图，10-13 `9f370e4` 采纳）当前 `main.tex` 未 `\includegraphics`；如需上图需另接线。配套 `docs/paper/one_pager.pptx`。

---

## 四、结论与关键判断

1. **冠军不动**：0.9519 保持交付物；种子判定早已完成（s43 Δ−0.0016 带内复现），σ_seed≈0.005 下任何 ≤0.005 差值单 seed 无法下结论（`main.tex:598` Seed replication 节）。
2. **Benchmark 的贡献是"把话说满"**：三件套全灭、上界明确；论文可稳健宣称 Ours 胜出（表 IV）。
3. **唯一工程优化**：A-prune 跨种子转正——零性能代价、−12% 观测维。
4. **可解释性驱动优化线整体关闭**（负结果/用户决定），可解释性本身留作论文章节。

---

## 五、延伸材料

- 训练/评估命令与口径：`CLAUDE.md` §开发命令、`AGENTS.md`。
- 跨轮知识库（失败/成功方向族）：`logs/opt_knowledge.json`（提新方向前必读）。
- 任务级预注册/判定：`.trellis/tasks/10-09-interpretability/`（含 `task.json`/`prd.md`）。
- 记忆（用户既定决策）：`.claude/.../memory/paper-diffopt-oracle-framing.md`（oracle 只入表）、`interpretability-analysis-module.md`。