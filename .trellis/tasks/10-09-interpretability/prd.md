# PRD: 可解释性分析 — SVEA 全局特征子集选择 + KernelSHAP/EG 局部归因

## 需求背景

当前最优成绩（champion `[7, 0.9519, ...]`）已通过鲁棒性/统计/复杂度多轮验证（论文 Results 各子节），但论文与代码库均无任何「策略为什么这么分配」的可解释性分析。本任务把两篇已入 `docs/paper/cite/` 的文献方法融合，对训练好的分层策略（Worker SA-GAT + Meta）做事后归因：

1. **SVEA**（Tripathi, Hemachandra, Trivedi — IEEE BigData 2020）：*全局*可解释特征子集选择。特征块为玩家，特征函数 v(S) = 截距only误差 − 子集 S 的 hinge-loss 线性 ERM 误差（LP）；SVEA_j = c̃/n − φ_j（MC 排列 Shapley）；**普适阈值 0**：e_j<0 = SVEA_neg 支配子集；t-分布区间估计抗采样偏差；PSV（子集/全集准确率）验证。
2. **DeepSHAP 族**（Fernando, Singh, Anand — SIGIR'19, arXiv:1907.06484）：*局部*模型内省反向传播归因（相对参照输入）；核心方法论教训 = **解释对参照输入高度敏感** → 构造多种参照并做 (a) 参照间 top-k 重叠、(b) 以模型无关代理（原文献 LIME，本任务用 KernelSHAP）recall@k 验证。

定位：**事后分析**，非新训练机制（RESEARCH_BRIEF Non-Goals 已核对：36 个已证伪方向与可解释性无关；`l5-5-csir-attribution` 是另一 pending 的信道诊断 idea，不受影响）。

## 解释对象（用户选定：全部四类决策对象）

| 决策对象 | 可微目标（`src/analysis/targets.py`） | 归因空间 |
|---|---|---|
| Worker 功率分配 | P_total（Σ clip(K·softmax(tanh(p_raw)),0,1)）、p_k | obs 15 块（13 原生 + snr + eps），镜像 env._parse_action |
| Worker 排水 d_ratio | d_mean（mean clip((tanh(d_raw)+1)/2,0,1)） | 同上 |
| Worker 解码序 | z_top1（softmax(softplus(z_raw)) 在 no_grad argmax） | 同上 |
| Meta 分组 + α | alpha_avg / group_logp_0（镜像 _map_alpha/Categorical logits） | meta 12 通道（K,12），get_user_features |

Worker α 块在 group_level_alpha=True 下是死维（被 Meta split_mask 覆盖），不解释并注明。

## 方法（两文献融合，细节见 `src/analysis/` 各模块 docstring）

- **SVEA 博弈（文献 1）**：「状态特征为玩家，最终任务 = 控制器的交付行为」。
  - 样本 = (用户, 时隙)，特征 = 13 原生块，标签 y = 1{served>0}（备选 queue_growth）。
  - tr_er(S) 用 `scipy.optimize.linprog(method="highs")` 解 hinge-LP；**主分支有界变量 + 全局列归一 + 类不平衡缓解**（多数类下采样至 cap=10:1；served 标签在部分 SNR 达 99:1，实测无界/不平衡触发 hiGHS "Status 0: Not Set"——三方加固后消除，均为仿射/加权不变手段，不改变博弈值）。
  - MC 排列 Shapley（B=200, m_sv=1000）；t-CI（G=30 组 × B_ci=15, m_ci=40）；PSV（线性分类器，MLP 可选 `--psv_mlp`）。
- **局部双估计器（文献 2）**：
  - KernelSHAP（模型无关代理，自实现无依赖）：Shapley 核加权最小二乘，15 块博弈，块级掩码=参照填充，互补配对采样（n_pairs=512）+ 全集/空集 ε 行。
  - Expected Gradients（模型内省，DeepSHAP 族）：φ = E_{α~U(0,1), x'~P_ref}[(x−x')∇f(x'+α(x−x'))]，单点参照（zero/means/neutral）或经验背景（boot=64 行）。
  - 验证：EG 参照对 top-5 Jaccard（敏感性）+ EG-vs-KernelSHAP recall@5（代理验证）。
- **关键实现锚点**：env p 块 = K·softmax(tanh(p_raw))（rsma_env.py:372-378）；champion 观测 H=4 → 603 维，13 原生块 = obs[0:13K]；`load_ppo_model`/`load_meta_model` 委托装载；Config patch 镜像 `evaluate_snr_sweep_temporal.py`；不可变文件零触碰。

## 交付物（全部新文件；唯一既有文件改动 = docs/paper/main.tex）

- `src/analysis/{__init__,features,models,games,kernel_shap,expected_gradients,targets,validation}.py`
- `scripts/interpretability_analysis.py`（采集→svea→kernel→eg→validation→figs→stdout 表 + summary.json）
- `tests/test_analysis_{svea,kernel_shap,expected_gradients}.py`（fast 标记，16 用例，闭式真值：线性模型 EG/KernelSHAP == w·(x−ref)、oracle 排列 Shapley 效率性、单特征支配数据的阈值 0 选择）
- `docs/paper/main.tex`：新 Results 子节 `Interpretability: what the policy attends to` + 2 个 \bibitem（tripathi2020svea / fernando2019deepshap）+ 3 图/2 表占位
- `.trellis/tasks/10-09-interpretability/{task.json,prd.md,experiment-results.md}`（本文件）

## 门禁 / 验收标准

1. **fast 测试门禁**：`.venv/Scripts/python.exe -m pytest -m fast -q tests/`（369 通过，含 16 个新用例）保持绿色；ruff check 干净（line 120）。
2. **端到端冒烟**（已由作者跑通）：`--snr_subset 20 --n_episodes 1 --n_pairs 16 --bg_kernel 4 --n_eg 8 --ci_groups 0` 产出 npz/figs/summary.json + 两张 markdown 表；SVEA_neg=[queue, split_mask_alpha]、PSV≈1.018、EG-vs-KernelSHAP recall@5∈[0.2,0.6]、参照对 Jaccard@5∈[0.25,0.67]（与文献 2 的参照敏感性结论形态一致）。
3. **不可变护栏**：channel.py / rate_calc.py / rsma_env.py 物理点 / evaluate_snr_sweep.py 零改动（git diff 核验）。
4. **全量运行（用户执行）**：7 SNR × 12 episodes 出 summary + 3 图入 docs/paper/figs/，数字填入 main.tex 占位。