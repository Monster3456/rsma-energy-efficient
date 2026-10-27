# 可解释性分析协议审查（对照 cite/ 四篇文献）

> 目的：核对 `docs/paper/main.tex` §Interpretability（`\label{sec:interp}`）与 `src/analysis/` +
> `scripts/interpretability_analysis.py` 的协议，与四篇源文献规定的方法学是否一致；列出必须改的措辞。
> 数据核对源 = `logs/interpretability/summary.json`（2026-10-10 主 run，checkpoint `m5_armA_sftp/best_guard.pt`，
> obs_dim=603 未剪枝、H=4、seed=42、12 ep/SNR、runtime 7.7h）。
> 图 `figs/fig_svea_intervals.png` / `figs/fig_topk_overlap.png` 由该 summary 于同日重绘，
> `logs/interpretability_p0/`、`_p3/` 是后续消融 run，**未**进论文图。

---

## 0. 结论摘要

1. **核心公式逐式核对通过**：`v(S)=c̃−tr_er(S)`（Eq.3）、`e_j=c̃/n−φ_j`（Thm 1 Eq.5）、
   阈值 0、PSV=子集/全集测试正确数之比（Def.2 Eq.6）——与 Tripathi 2020 **完全一致**，
   无正则 hinge-LP 也与原文主方案一致（原文证伪了 ℓ1 变体）。KernelSHAP 的移除语义
   = Covert 表 1 的「边际 + 预测 + Shapley」格子，也对。
2. **最重的问题在区间估计**（已按原文口径重算 + 敏感性检验，见 §3.2）：原 t-CI 少了原文的
   "**每组 30 个子样本求组均值**"这一步。补上并把子样本放大到 m=200 后：**queue 的支配在
   0–20 dB 全部被区间支持**（`hi<0`），25 dB 区间均值 −0.0008 贴阈值不成立（m=40 下的"反向"
   是小样本伪影），30 dB 一致判无支配；split-α 在 20 dB 不被区间支持。主博弈点估计与 10-10
   基线逐点 `max|Δ|=0`，未受影响。
3. **引用挂错位置**：Expected Gradients 的定义挂在 `fernando2019deepshap` 名下（Fernando 全文
   未定义 EG）；"bootstrap 作默认参照"也不是 Fernando 的结论（他明确说**无普适最优参照**）。
4. **四篇只引了两篇**。Sundararajan–Najmi（1908.08474）与 Covert et al.（2011.14878）零引用，
   可它们恰恰是"参照必须声明"和"KernelSHAP = 边际移除"这两条的原始出处。
5. **验证指标存在循环性**（Covert §10.2 "指标不中立"）：EG-vs-KernelSHAP 的 recall@5 在
   **同一参照**下比较，两者共享同一背景构造，一致部分是同义反复；有信息量的是跨参照差异。
   现在没披露。
6. **三处数字/叙述自相矛盾**：摘要 "across operating points" vs 30 dB 无支配；
   25 dB 表行；recall 区间 0.8–1.0（实际 30 dB=0.6）。另有 25 dB 漏叙述、PSV>1 未解释、
   KernelSHAP 主驱动在中段 SNR 由 Δqueue 领跑且符号随参照翻转。

---

## 1. 四篇文献各自规定了什么

| 文献 | 文件 | 角色 | 规定的核心协议 |
|---|---|---|---|
| Tripathi, Hemachandra, Trivedi, *Interpretable feature subset selection: A Shapley value based approach*, IEEE BigData 2020 | `Interpretable_feature_subset_selection_...pdf` | **SVEA 全局特征子集选择** | 分类博弈 + Shapley 误差分摊 e_j + 阈值 0 + PSV |
| Fernando, Singh, Anand, *A study on the Interpretability of Neural Retrieval Models using DeepSHAP*, SIGIR'19 | `1907.06484v1.pdf` | **参照输入构造 + 参照敏感性** | 5 种参照构造；Jaccard 重叠矩阵；recall 对照模型无关代理（LIME）；结论=无普适最优参照 |
| Sundararajan & Najmi, *The Many Shapley Values for Model Explanation*, ICML 2020 | `1908.08474v2.pdf` | **操作化的公理基础** | BShap / CES / RBShap / IG 四种操作化互不等价；唯一性只对固定操作化成立；Dummy 公理；必须声明基线 |
| Covert, Lundberg, Lee, *Explaining by Removing*, JMLR 2021 | `2011.14878v2.pdf` | **统一框架 + 移除语义** | 三维度声明（怎么移除 / 解释什么行为 / 怎么汇总）；KernelSHAP=边际移除；特征选择=博弈 excess；指标不中立 |

### 1.1 各篇要点（协议层面）

**Tripathi 2020（SVEA）** — 核心定义（已逐式核对原文）：

- **特征函数（Eq.3）**：`v(S,m) = tr_er(∅,m) − tr_er(S,m)`，`tr_er` = hinge-loss 线性 ERM 的最优平均松弛
  （Eq.1 截距 only / Eq.2 r 维子空间），`c̃(m) := tr_er(∅,m)`。**As v(∅,m)=0, it is a valid
  characteristic function**；Prop.1：`v(·,m)` 对联盟单调。
- **Shapley（Eq.4）**：经典加权边际贡献式；博弈必须用 **H_lin**（非线性会破坏大联盟假设）。
- **Thm 1（Eq.5）**：`e_j(tr_er(N,m)) = c̃(m)/n − φ_j(N, v(·,m))` —— 唯一的误差分摊，是 φ 的**仿射变换**。
  单位 = 训练误差；**阈值 0**：`e_j < 0` = SVEA_neg = dominant（§3.1）。Prop.2 给出两人情形的支配者直觉。
- **Def.1 可解释 FSS 三条件**：(P1) 过程透明（Shapley 计入特征交互）；(P2) 与最终任务相关（用最终分类任务的
  训练误差）；(P3) 重要性值可辩护（e_j 是误差分摊，故可与 0 比）。SVEA 三项全满足（Table 1）。
- **Def.2 / Eq.6 PSV**：`P_SV(K) = (子集分类器测试正确数) / (全集分类器测试正确数)`，
  两个都是**最优线性分类器**，测试集 `m_te` **独立于 D**。比值形式，可 >1。
- **MC 采样**：沿用 [10] 的按需求值近似；n≤10 时对照真值，**100 MC 样本**最大误差 5%（1000 样本 4%），
  且 100 vs 1000 **符号与排序不变** → 原文固定用 100。n≥10 才用近似。
- **§3.2 区间估计**：为抗样本偏差，把数据**切成多个不相交子样本**，选 **G 组、每组 30 个子样本**求
  组均值 `e_j^g`，对 `e_j^p`（总体均值）构造 t 区间 `ē_j ± t*_{α/2,G−1}·(s_j/√G)`。
  判据："**a feature's joint contribution … is substantially high if the interval estimate of SVEA
  for a feature lies on the left of origin**"；原文声明区间估计**与阈值 0 的 FSS 结论一致**。
- 另：ℓ1 正化的 squared-hinge ERM 在 UCI 上**识别不出重要特征**（Heart/Pima/Thyroid）→ 原文主方案
  是**无正则** hinge LP（与我们一致）。

**Fernando 2019** — 解释对象是 NRM 的相关性分数；被解释特征 = 文档词；**query 侧参照固定为 query 本身**。
5 种文档参照：OOV（全零/词表均值嵌入）、IDF lowest、QL lowest、COLLECTION rand doc、TOPK LIST bottom。
指标 = 参照变体间 **term 集合 Jaccard**（Fig.1，多数低于 50%）+ **recall**（DeepSHAP top-50/100 by |shapley|
覆盖 LIME top-10/20/30），**忽略极性**。LIME 作代理（扰动样本 5000），代理自身拟合精度 >90%。
结论：参照敏感性高、**无普适最优参照**（DRMM/PACRR 最优=QL lowest，MatchPyramid 最优=OOV——
因 OOV 嵌入恰好是该模型的 padding 向量），参照选择依赖模型预处理。**论文未建议"报告多参照"**，
也未讨论 SHAP 公理。

**Sundararajan & Najmi 2020** — 核心命题：Shapley 归因**不由数值公式定义**，而由操作化定义
（集合函数/移除语义 × 解释语境/基线 × 端到端还是分层）。四种操作化：
CES（条件期望，Eq.2）、BShap（基线替换，Eq.3）、RBShap（对基线分布取期望，Eq.4）、
IG（路径积分，Eq.5）。唯一性结果：BShap = 唯一满足 (Linearity, Dummy, ASI, **Demand Monotonicity**, Symmetry)；
IG = 唯一满足 (…, **Proportionality**, …)。**经典唯一性定理不选定集合函数**。
CES(D̂) 在一般分布下破坏 Dummy/Linearity/Demand-Monotonicity/Symmetry（反例 Ex.3.3–3.4, C.1–C.3）。
**Dummy 公理**：模型未引用的特征必须得零归因（= missingness）。
DeepSHAP 的分层施加破坏 Symmetry，归因依赖参数排布而非函数 → **必须端到端**。
基线选择承载解释语境，是"额外认知负担"，无普适选择时用 RBShap。
注：**本文未定义 "Expected Gradients"**（Erion et al. 未引），只定义了离散侧的 RBShap；
单基线 IG 的唯一性不自动推广到对基线取期望的 multi-IG。

**Covert et al. 2021** — Definition 1 三维度（互相独立、缺一不可）：
(1) 移除方式 F(x_S)：零值 / 默认值 / **边际** p(X_S̄) / **条件** p(X_S̄|X_S) / 乘积边际 / 均匀 / 代理模型 / 每子集单独模型；
(2) 行为 u(S)：预测 / 预测损失 / 数据集损失 / 对自身输出的损失；
(3) 汇总：Shapley / Banzhaf / 逐个移除 / 逐个包含 / LIME 加性模型 / 子集选择。
**KernelSHAP 的格子 = 「边际移除 + 预测 + Shapley」**（Table 1），是 SHAP 实现的默认行为；
与条件 SHAP **只差第一维**，在特征相关时两者不等价（边际 = 独立性假设下的条件近似，§8.2）。
条件移除是唯一概率一致的移除（Prop.6–7），才能给出互信息/PMI 读数（§8.3）；
但条件移除会让"模型功能上未使用的特征"显得重要，且把信用均分给完全相关特征。
**§10.2：评价指标不中立**——insertion/deletion/sensitivity-n/ROAR 各自内嵌一种移除选择，
会偏向共享该选择的解释方法。特征子集选择 = 在博弈 excess 上解高/低价值子集问题（§7.4），
可与 SVEA_neg 对话。§7.5：Shapley 是最完整的汇总（满足 efficiency，分割大联盟价值）。

---

## 2. 我们的实现/论文声称 ↔ 文献要求 对照

### 2.1 SVEA 全局博弈（对 Tripathi）

| 项目 | 我们的做法（代码/论文） | 文献要求 | 判定 |
|---|---|---|---|
| 玩家 | 13 个 per-user 观测块（`native_blocks`） | 特征=玩家 | ✅ |
| 标签 | `served` 二分类（服务/未服务），±1 | 最终任务标签 | ✅（但注意：这是**代理任务**，非 EE/交付） |
| 特征函数 | `v(S) = c̃ − tr_er(S)`，tr_er = hinge-LP 最优平均松弛 | Eq.3 完全一致 | ✅ **逐式核对通过** |
| 截距项 c̃ | 闭式 `2·min(p,1−p)` | `c̃(m)=tr_er(∅,m)`（Eq.1） | ✅ |
| 误差分摊 | `e_j = c̃/n − φ_j` | Thm 1 Eq.5 逐字一致 | ✅ **逐式核对通过** |
| 阈值 | `e_j < 0` → SVEA_neg | 普适阈值 0（§3.1） | ✅ |
| 分类器族 | 无正则 hinge-LP（`_hinge_lp`） | H_lin + 无正则（ℓ1 变体原文证伪） | ✅ |
| Shapley 估计 | MC 排列采样，**B=200**，去重 mask 两阶段 | 原文 **100 MC**（符号/排序 100↔1000 不变） | ⚠️ 口径略异，须披露（B=200 更稳，无害） |
| 区间估计 | t 95%，G=30 组，**每组 1 个 m_ci=40 子博弈**，ci_B=15 | **不相交子样本 → 每组 30 个子样本求组均值**，再对总体均值做 t 区间 | ⛔ **协议偏离，见 §3.2** |
| PSV | `acc_subset / acc_full`（线性，独立测集 1/5），可选 MLP | Def.2 Eq.6 = 子集/全集测试正确数之比，**最优线性分类器**、m_te 独立于 D | ✅ 主口径一致（`psv_linear`）；`psv_mlp` 是扩展，不应用作表中 PSV |
| Def.1 P1/P2/P3 | **论文未引用** | 三条件是 SVEA 的"可解释"主张依据 | ⚠️ 应补引 |
| 类不平衡 | 分层子采样（class_cap=10） | **文献无此机制** | ⚠️ 我们的扩展，须披露（served 在个别 SNR 达 99:1） |
| 数值稳定 | z-score + 列 max\|·\| 归一 + hiGHS 有界变量 + 抖动兜底 | 文献无 | ⚠️ 我们的实现补丁，须披露（仿射不变性论证在代码 docstring） |
| null player | 零方差块（arrivals/λ桶/η）排除出排列游走，e_j 仍用 n=13 | Shapley Dummy/null player | ✅ 有公理背书（另见 Sundararajan Dummy） |

### 2.2 局部归因（对 Fernando + Sundararajan + Covert）

| 项目 | 我们的做法 | 文献要求 | 判定 |
|---|---|---|---|
| 方法 | Expected Gradients：`φ_i = E_{α∼U(0,1), x'∼P_ref}[(x_i−x'_i)·∇_i f(x'+α(x−x'))]` | Fernando=DeepSHAP 参照口径；Sundararajan=单基线 IG + RBShap 的离散类比 | ⚠️ **引用挂错**，见 §4 |
| 移除语义（KernelSHAP） | 玩家取被解释输入、非玩家取背景行 = **边际移除**（`kernel_shap.mask_fn`） | Covert Eq.5 = KernelSHAP 格子 | ✅ 语义正确，**但论文未声明"边际 vs 条件"** |
| 行为 | 原始输出（P_total / d_mean / z_top1 / alpha / group_logp） | Covert 维度 2 = 预测 | ✅ 未做 log-odds 变换，应声明 |
| 汇总 | Shapley（WLS + π_Sh 核 + 互补配对 + 端点 ε 权行） | Covert Eq.33/36 | ✅ |
| 端到端 | 对复合目标 fn（encoder→head）整体求梯度 | Sundararajan：分层破坏 Symmetry | ✅ |
| 参照变体 | zero / means / boot(64 行) / neutral | Fernando 5 种参照 | ✅ 设计对齐；语义映射：zero≈OOV、boot≈COLLECTION、means≈IDF/QL 泛化词、neutral≈构造语义中性 |
| 参照敏感性验证 | 参照两两 top-5 Jaccard | Fernando Fig.1 Jaccard 矩阵 | ✅ 指标同源（他们用 term 集合，我们用 top-5 块） |
| 代理对照 | KernelSHAP 作模型无关代理，recall@5 | Fernando 用 LIME 作代理、recall@50/100 | ⚠️ 我们的改写，k 与代理都变了，须声明 |
| 极性 | 忽略，按 \|归因\| 排序 | Fernando：忽略极性、按 \|shapley\| 排序 | ✅ |
| 样本量 | 12 ep/SNR 收集状态、64 状态/SNR 归因、n_pairs=512、bg_k=32、n_eg=256 | Fernando 未报背景集大小 | ⚠️ **论文正文未报任何数值**，见 §5 |
| 死维处理 | Worker α 块剔除（被 meta split_mask 覆盖） | Sundararajan Dummy 公理 | ✅ 有公理背书，应引 |

---

## 3. 数据自洽性核查（对 `logs/interpretability/summary.json`）

论文表 `tab:svea` / `tab:recall` 的数字与 summary **逐位一致**（PSV 0.9657/0.9529/0.9897/1.003/1.008/1.0096/0.9971
→ 表中 0.966/0.953/0.990/1.003/1.008/1.010/0.997；recall 四列同理）。问题不在抄写，在**报告口径**。

### 3.1 摘要与正文矛盾（严重）

- 摘要（`main.tex:73-74`）："the queue block as the dominant feature **across operating points**"。
- 正文（`main.tex:855-857`）："no feature dominates at **30 dB**"；表 `tab:svea` 30 dB 行 = (none)。
- 数据：30 dB `sv_neg=[]`，`sv[queue]=+0.022`。

→ 摘要过度声称。应改为 "dominant at six of seven operating points" 或 "the only feature whose dominance recurs"。

### 3.2 25 dB 行：SVEA_neg 与 95% CI 异号（硬伤 + 协议偏离）

| SNR | 点估计 sv[queue] | 表中 95% CI | 表中 SVEA_neg | 按 Tripathi 判据（CI 全在原点左侧） |
|---|---|---|---|---|
| 0 | −0.0812 | (−0.083, 0.008) | queue | 跨 0，不成立 |
| 5 | −0.0847 | (−0.004, 0.084) | queue | 跨 0，不成立 |
| 10 | −0.1204 | (−0.072, 0.017) | queue | 跨 0，不成立 |
| 15 | −0.1895 | (−0.087, 0.000) | queue | 上界≈0，擦边 |
| 20 | −0.0837 / α −0.0333 | (−0.033, 0.005) / (−0.019, 0.037) | queue, α | 跨 0，不成立 |
| **25** | **−0.0477** | **(0.003, 0.039) 全正** | **queue** | **反向，明确不成立** |
| 30 | +0.0224 | (−0.023, 0.026) | (none) | — |

**根因有两层：**

1. **协议偏离**（对照 Tripathi §3.2）：原文的区间估计是"数据切成**不相交子样本** →
   **G 组、每组 30 个子样本求组均值** `e_j^g` → 对**总体均值** `e_j^p` 做 t 区间"。
   组内 30 次平均是它抗噪的关键。我们的 `_ci_estimate` 是 **G=30 组、每组只跑 1 个
   m_ci=40 的小博弈**（ci_B=15 排列），组内无平均，且 40 样本远小子样本应有规模。
   结果是 CI 方差过大、中心系统性偏离主估计（25 dB CI 中心 +0.021 vs 主估计 −0.048）。
2. **两个估计器被并排呈现**：点估计来自主博弈（m_sv=1000, B=200），
   CI 来自上述小博弈。`games.t_ci` 不是主估计的置信区间。

**更严重的是判据本身**：Tripathi 的显著性判据是"**区间估计落在原点左侧**"，
且原文声称区间估计与阈值 0 的 FSS 结论一致。按此判据，**我们没有任何一点满足**
（0/5/10/20 dB 的 CI 都跨 0，25 dB 反向）。也就是说，现有 CI 根本支撑不了
"queue 是支配特征"的任何一行——不是 25 dB 一行的事。

**已落地（2026-10-21）**：`src/analysis/games.py` 新增 `mean_subgame_e`（组内 M 个独立子博弈取均值）
与 `left_of_origin`（Tripathi §3.2 判据）；`scripts/interpretability_analysis.py` 的 `_ci_estimate`
改为「时隙轴不相交分组 → 组内 M=30 子博弈平均 → `t_ci`」，输出新增 `ci95.mean / left_of_origin / G / M`。
顺带修了 `fig_svea_intervals` 的误差棒 bug（旧版 `yerr=hi−lo` 把区间宽度当半宽，误差棒放大一倍）。
测试 `tests/test_analysis_svea.py` +4 例；fast 门禁 398 全过。

**重算结果（`logs/interpretability_ci_v2/summary.json`，G=30×M=30，m_ci=40，B=15）**：

| SNR | sv[queue]（点估计，m=1000） | CI lo | CI hi | CI mean | `left_of_origin` | `sv_neg` | 判定 |
|---|---|---|---|---|---|---|---|
| 0 | −0.0812 | −0.0292 | +0.0085 | −0.0103 | ✗ | queue | 点估计支配，区间跨 0 |
| 5 | −0.0847 | −0.0216 | +0.0108 | −0.0054 | ✗ | queue | 同上 |
| 10 | −0.1204 | −0.0625 | −0.0273 | −0.0449 | ✓ | queue | **一致** |
| 15 | −0.1895 | −0.0700 | −0.0309 | −0.0505 | ✓ | queue | **一致** |
| 20 | −0.0837 | −0.0216 | −0.0045 | −0.0130 | ✓ | queue, split-α | queue 一致；**split-α 不一致**（CI +0.0064~+0.0219） |
| 25 | −0.0477 | **+0.0105** | **+0.0346** | +0.0226 | ✗ | queue | **仍反向** |
| 30 | +0.0224 | +0.0067 | +0.0365 | +0.0216 | ✗ | none | 一致（都无支配） |

要点：

1. **主博弈点估计与 10-10 串行基线逐点 `max|Δ|=0`**（七个 SNR 全部）→ 修复的并行 bug 验证通过，
   论文 `tab:svea` 的 `sv`/`sv_neg` 列不受影响。
2. **组内 30 次平均确实收紧区间**（0 dB 旧 CI 宽 0.091 → 新 0.038），原文的抗噪步骤有效。
3. **按原文判据，queue 的支配只在 10/15/20 dB 成立**（`hi<0`）；0/5 dB 方向一致但不显著；
   **25 dB 在正确口径下仍然反向**；split-α 在 20 dB 的支配不被区间支持。
4. **幅度不可比**：CI 均值系统性小于点估计（15 dB −0.051 vs −0.190），因为子博弈 m_ci=40 ≪ 主博弈
   m_sv=1000，可达成的训练误差下降更小。Tripathi 的用法本来就只比**符号/阈值**（"interval
   estimates also lead to the same threshold of 0"），故 CI 不是点估计的置信区间，
   报告时只能用 `left_of_origin` 判据，不能声称区间覆盖点估计。
5. **敏感性检验已完成（`logs/interpretability_ci_m200/`，m_ci=200，G=30×M=30）**，结论取代 m=40：

   | SNR | 点估计 sv[queue] | 区间均值 | 95% CI | `left_of_origin` | 判定 |
   |---|---|---|---|---|---|
   | 0 | −0.0812 | −0.0586 | (−0.0797, −0.0374) | ✓ | 支配且显著 |
   | 5 | −0.0848 | −0.0393 | (−0.0548, −0.0238) | ✓ | 支配且显著 |
   | 10 | −0.1204 | −0.1023 | (−0.1198, −0.0849) | ✓ | 支配且显著 |
   | 15 | −0.1895 | −0.0446 | (−0.0664, −0.0229) | ✓ | 支配且显著 |
   | 20 | −0.0837 | −0.0149 | (−0.0242, −0.0056) | ✓ | 支配且显著 |
   | 25 | −0.0477 | −0.0008 | (−0.0051, +0.0035) | ✗ | 支配但不显著 |
   | 30 | +0.0224 | +0.0002 | (−0.0030, +0.0035) | ✗ | 非支配 |

   **25 dB 的"反向"是 m=40 小样本伪影**：放大到 m=200 后区间均值从 +0.023 落到 −0.0008，
   不再是正号反转，而是**贴着阈值 0 的临界空值**。0/5 dB 在 m=40 下跨 0，m=200 下转为全负。
   拆分比 α 在 20 dB 的区间均值 +0.014、CI (+0.004, +0.012)，**不支持**其支配。

**对 `tab:svea` 与正文的结论**：不能整表照抄"queue 在 0–25 dB 支配"。诚实口径 =
"点估计上 queue 是唯一反复出现的支配块（6/7 点），区间估计按 Tripathi 判据仅在 10–20 dB 支持；
0/5 dB 方向一致但跨 0；25 dB 点估计与区间估计符号相反，该点支配不稳健；30 dB 两者一致判无支配"。
split-α 在 20 dB 的支配同样只由点估计给出。

处置选项（需写进论文）：
- **(a) 重算 CI 到原文口径**：不相交子样本、每组 30 个子样本平均、与主估计同源（m/B 同级），
  然后按"CI 左于原点"重新判定 SVEA_neg；若多数点不成立，则把结论弱化为
  "点估计上 queue 为唯一反复出现的支配块，区间估计只在 0–15 dB 支持该方向"。
- **(b) 保留现 CI 但改表述**：表下脚注声明 CI 来自分组子博弈、与点估计不同源，
  SVEA_neg 由点估计决定；25 dB 明确标注"区间估计不支持（不稳健）"；
  删除"CI 左于原点"类判据表述。
- 两者都需**披露** m_ci=40 / ci_B=15 / G=30 的实际口径，并说明与原文"每组 30 子样本平均"的差异。

### 3.3 PSV > 1 未解释（但 >1 合法）

15/20/25 dB 的 PSV = 1.003 / 1.008 / 1.010，即 **SVEA_neg 子集分类器测试精度超过全集**。
正文只写 "stays ≥ 0.95"，回避了 >1。

按 Tripathi Def.2 / Eq.6，PSV **本来就是子集/全集测试正确数之比**，>1 在数学上完全合法
（子集泛化优于全集，通常因全集线性分类器在 13 维上过拟合）。原文报告"多数数据集约 0.90"，
没有上限约束。**我们的定义与 Eq.6 逐字一致**（`psv_linear` = acc_sub/acc_full，独立测集），
所以问题只是正文要写明"比值、可 >1、>1 表示子集不损失甚至略优于全集"，而不是把它说成 0–1 的分数。

另注：原文 Def.2 用**最优线性分类器**，非线性分类器只出现在 §4 的最终分类任务里。
因此表 `tab:svea` 的 PSV 必须用 `psv_linear`（目前是对的）；`psv_mlp` 是扩展，不应进表。

### 3.4 正文漏掉 25 dB

`main.tex:855-857` 的叙述是 "0–15 dB / 20 dB / 30 dB"，**25 dB 整点没提**，而表里它有 SVEA_neg。

### 3.5 recall 声称区间有误

正文（`main.tex:881-883`）："bootstrap/marginal-means references agree best with the model-agnostic
proxy on the power target (**recall@5 0.8–1.0 across SNR**)"。
数据（`tab:recall` 与 summary 一致）：

| ref | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 均值 |
|---|---|---|---|---|---|---|---|---|
| zero | 0.6 | 0.6 | 0.4 | 0.4 | 0.8 | 0.8 | 0.6 | 0.629 |
| means | 0.8 | 1.0 | 0.6 | 0.6 | 0.8 | 0.8 | 0.8 | 0.771 |
| **boot** | 0.8 | 0.8 | 0.8 | 0.8 | 1.0 | 0.8 | **0.6** | **0.800** |
| neutral | 0.8 | 0.8 | 0.8 | 0.6 | 0.6 | 0.8 | 1.0 | 0.771 |

- bootstrap 在 30 dB 是 0.6，**不是** 0.8–1.0。应写 "0.6–1.0 (mean 0.80)"。
- "means 与 boot 并列最好"不成立：boot 0.800 vs means 0.771，差距在单点分辨率内。
  更稳的说法：**zero 最差（0.629），其余三者相当（0.77–0.80）**。
- Fernando 的结论恰是"无普适最优参照"，我们的数据也只支持"zero 不可靠"，
  支撑不了"bootstrap 是应选默认"这一强结论（见 §4.3）。

### 3.6 KernelSHAP 主驱动声称的稳定性

正文（`main.tex:884-885`）："KernelSHAP shows **queue-backlog and channel-gain** blocks as the
dominant drivers of the total-power decision"。
按 |归因| 排序的 top-5（P_total）：

| SNR | ref=boot | ref=means |
|---|---|---|
| 0 | queue, group_ids, h_gains, … | queue, qos_lambda, power_history, h_gains, Δqueue |
| 10 | **Δqueue, h_gains**, power_history, … | **h_gains, qos_lambda**, queue, … |
| 20 | **Δqueue, h_gains**, rate_history, power_history, queue | queue, h_gains, qos_priority, qos_lambda, Δqueue |
| 30 | queue, h_gains, rate_history, power_history, dvq | queue, power_history, qos_priority, qos_lambda, rate_history |

→ 方向大体对（queue/Δqueue/h_gains 族主导），但 **10–20 dB + boot 参照下 queue 掉出前三**，
且 **符号随参照翻转**（boot 下 queue 为正 +0.24~+0.95，means 下为负 −4.8~−8.3，量级也差一个数量级）。
现有文字既未提符号不稳定，也未提交互作用块（Δqueue）在中段 SNR 领跑。建议补一句。

---

## 4. 引用精度：哪条声称该挂哪篇

| 现声称（`main.tex`） | 现引 | 问题 | 应改为 |
|---|---|---|---|
| SVEA 分类博弈、e_j、阈值 0 | tripathi2020svea | 无 | 保留（待 Tripathi 终核公式） |
| "Local attribution and reference sensitivity~\cite{fernando2019deepshap}" | fernando | 这句里 **Expected Gradients 的定义挂在 Fernando 名下**，但 Fernando 研究的是 DeepSHAP 参照构造，全文未定义 EG，也未提 Expected Gradients | EG 定义改引 Erion et al.（Expected Gradients）或按 Sundararajan 的 RBShap/IG 表述；Fernando 只保留"参照敏感性 + Jaccard 验证设计" |
| "Mirroring the finding of~\cite{fernando2019deepshap}" | fernando | 可接受但偏强：Fernando 的 Jaccard 是 **term 集合**（多低于 50%），我们是 top-5 块；recall 他们用 LIME 且 k=50/100，我们用 KernelSHAP 且 k=5 | 改为 "consistent with the reference-sensitivity finding of …, measured here as top-5 block Jaccard (their term-set analogue) and recall@5 against a model-agnostic proxy (KernelSHAP; their proxy was LIME)" |
| "motivating the bootstrap reference as the paper's reporting default" | fernando | **Fernando 明确说无普适最优参照**、参照选择依赖模型预处理；他们没有"报告默认"建议 | 把 bootstrap 默认的依据改成**我们自己的**经验判据（与模型无关代理一致度最高），并引 Fernando 说明这本来就是模型依赖的判断；引 Sundararajan Remark 4.7 说明基线承载解释语境 |
| （缺）参照/基线必须声明 | — | 未引 | 加 sundararajan2020manyshap（BShap/RBShap/IG 操作化 + Remark 4.7） |
| （缺）KernelSHAP 的移除语义 = 边际 | — | 未引、未声明 | 加 covert2021explaining（Table 1 格子 + Eq.5）；并在文中写明"边际移除（interventional），特征相关时是条件移除的独立性近似" |
| （缺）死维/null player 公理背书 | — | 未引 | Dummy 公理 → sundararajan（或 covert §7.2） |
| （缺）指标不中立 | — | 未披露 | covert §10.2 → 在 Limitations 补一句"recall 与 KernelSHAP 共享参照构造，一致度部分同义反复" |
| 摘要 "Shapley-value-based interpretability analysis" | — | 泛称可接受 | 若要更准，可写明两条腿：全局误差分摊（SVEA）+ 局部参照敏感归因（EG/KernelSHAP） |

**结论：四篇里只有 2 篇进了参考文献。** Sundararajan–Najmi 与 Covert et al. 应补 bibitem，
否则"多参照报告"与"边际移除"两条方法学决定在论文里无出处。

---

## 5. 论文应补报的协议数值（可复现性缺口）

`main.tex` §Interpretability 目前**没有**报告任何采样/规模参数。按 `scripts/interpretability_analysis.py`
默认口径至少应写：

- 检查点：`m5_armA_sftp/best_guard.pt`（单检查点、单 seed=42，Limitations 已承认）
- 状态采集：12 episode / SNR；本地归因 **64 状态 / SNR**
- SVEA：m_sv=1000（分层子采样，少数类上限 10:1）、B=200 排列（原文 100）；
  CI：G=30 组 × 每组 1 个 m=40 子博弈 × B=15 排列（**原文为不相交子样本、每组 30 子样本平均**）
- SVEA 判据：Def.1 (P1) 交互透明 / (P2) 最终任务训练误差 / (P3) 误差单位可与 0 比；阈值 0（Thm 1）
- PSV 按 Def.2 = 子集/全集测试正确数之比（最优线性分类器，测集独立于训练集）
- KernelSHAP：n_pairs=512 互补联盟对、背景 bg_k=32 行；EG：n_eg=256（α 路径 × 背景抽取）
- 参照：zero（全零/中性构造）、means（块边际均值）、boot（经验 64 行）、neutral（语义中性构造）
- 目标：P_total、d_mean、z_top1、alpha_avg、group_logp_0（α 已按死维剔除）
- 标签：served 二分类；玩家 = 13 per-user 块（+2 global 仅本地归因）
- 数值实现细节：hinge-LP 用有界变量 + hiGHS 三重回退；分层子采样与列归一（实现补丁，仿射不变）

---

## 6. 必改措辞清单（按优先级）

| # | 位置 | 问题 | 动作 |
|---|---|---|---|
| P0 | `tab:svea` 全表 + 判据表述 | CI 已按原文口径重算 + 敏感性检验（§3.2）：queue 在 0–20 dB 支配且显著，25 dB 支配但不显著，30 dB 非支配；split-α 在 20 dB 支配但不显著 | 表中补一列 `CI left of origin`；正文按"0–20 dB 支配且显著、25 dB 支配但不显著、30 dB 非支配"改写 |
| P0 | `main.tex:73-74` 摘要 | "dominant feature across operating points" 与 30 dB 自相矛盾 | 改为 "the only recurrently dominant block"（且需与 CI 结论一致） |
| P0 | `main.tex:493-510` vs 表 IV | Comparator set 声称的 DR-SAC / neural-DPP 无评估结果（见 §7） | 补评估或改措辞 |
| P1 | `main.tex:871` EG 归因挂 Fernando | 引用错位 | EG 定义改引 Erion / 按 RBShap-IG 表述；Fernando 只留参照敏感性 |
| P1 | §Interpretability 全节 | 未声明边际 vs 条件移除 | 补一句 + 引 Covert |
| P1 | 参考文献 | 缺 sundararajan / covert | 加 2 条 bibitem |
| P1 | `main.tex:846-853` SVEA 段 | 未引 Def.1 (P1)(P2)(P3)——SVEA "可解释"主张的依据 | 补一句 + 引 tripathi Def.1 |
| P1 | `main.tex:881-883` | recall 区间 0.8–1.0 不实（30 dB=0.6）；"means 与 boot 并列最好"过强 | 改 0.6–1.0 (mean 0.80)；改述为 "zero 最差，其余三者相当" |
| P2 | `main.tex:855-857` | 漏 25 dB | 补 "0–15 and 25 dB"（并按 CI 结论调整） |
| P2 | `main.tex:860-861` | PSV>1 未解释 | PSV 按 Def.2 是**比值**，>1 合法（子集泛化优于全集）；一句话说明即可 |
| P2 | `main.tex:884-885` | KernelSHAP 主驱动声称未提交互块与符号随参照翻转 | 补一句 |
| P2 | §Interpretability | 无协议数值 | 按 §5 补报 |
| P2 | §Interpretability | MC 样本数与原文（100）不同、类不平衡分层为扩展 | 披露 |
| P3 | Limitations | 未披露验证指标循环性 | 引 Covert §10.2 补一句 |

---

## 7. 附：对比基线缺口备忘（本次一并查实）

| 项 | 状态 |
|---|---|
| LDPTRLQ | 英文：`fig:band` 曲线 + `tab:classic` 行 + 训练曲线图；**逐点表 `tab:bench` 无**；中文 `report-zh` **全无**（正文 519 行提了却没给数） |
| TD3 / DDPG / SAC | 英文：仅 `tab:classic` 数字行，主图**故意不画**（caption 已声明）；中文 **全无** |
| DR-SAC / neural-DPP | `main.tex:499-503` Comparator set **声称已对比**；训练跑过（`logs/comp_drsac/`、`logs/comp_neuraldpp/` 有 final checkpoint），**官方 7×50 从未评估**，无任何结果行。表 IV 的 "SAC" 行是 `train_drsac.py --no-perturb`（= 标准 SAC，非 DR-SAC），TD3/DDPG 亦复用 DR-SAC 的 critic/回放管道 |
| 逐点数据（可直接补表） | LDPTRLQ：0.753/0.945/0.932/0.848/0.742/0.662/0.595，P≈22.8–23.3W；TD3 0.818（24W 饱和）；DDPG 0.886（5/7）；SAC 0.787 |

处置选项：(i) 补跑 DR-SAC / neural-DPP 官方评估并入表；(ii) 把 Comparator set 的 (ii)(iii)
改为"训练但未完成官方评估"或删除，另起一段说明；(iii) 中文报告补一张经典学习族对照表（数据已齐）。
