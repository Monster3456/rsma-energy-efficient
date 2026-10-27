# ML_INNOVATION_SCAN_2026 — 过载上行 RSMA-MU-MIMO 能效在线优化

> 调研窗口：2023–2026 · 生成日期：2026-10-27 · 目标：为**已压缩到 0.6% oracle 缺口**的分层 RL 策略找下一跳创新算法方向
> 本地事实源：`RESEARCH_BRIEF.md`、`logs/opt_knowledge.json` 族、`.trellis/tasks/10-23-hierarchical-opt/experiment-results.md`、`docs/paper/progress-2026-10.md`
> 已证伪机制族（**全文不再提案**）：Lagrangian/PID 对偶调参、CuRLTraC 变体课程、梯度操纵（CRPO / 原始可行域零空间投影）、模型汤/SWA、分位数 critic、SIC 解码序探针、全局/分桶查表 `group_alpha`、`w_power_exceed`/`lagrangian_max` 标量阶梯；以及 idea-stage 已 desk-reject 的朴素 `blend_alpha` 插值、朴素 per-bucket MoE、朴素 obs 增广、分桶 self-imitation。

---

## 1. 摘要

本问题已是**压缩尾声**：可微 DiffOpt 逐槽上界 0.9574，冠军蒸馏策略 0.9519，缺口 **0.6%**（约 1.1σ_seed）；种子噪声 σ≈0.005。近期因果消融又证明：动作 4 头中 α/d 是死重（σ~1e-5），砍到 2K=48 后 7/7 门不掉；用户分组是**近优单组坍缩**（强行分化 −4σ）；慢尺度交付目标 ζ 条件化**机制成立**（功率 4.66→13.61 W 随 ζ 单调）但交付略降。

因此下一轮创新不应再碰奖励/对偶/课程/分组，而应沿三条**已验证未证伪**的轴推进：

1. **把求解器“压进”策略**（amortized / decision-focused / residual distillation），对准那 0.6% 的 amortization gap；
2. **把约束改成目标条件化 / 可行性优先**（ζ 已证明可调操作点），而不是对偶变量；
3. **把测试时算力做成策略的一部分**（anytime 精修 / inference-time residual），用预算换缺口。

结构化生成动作（flow/diffusion/Sinkhorn）与图注意力升级是次优先：动作空间已减半，收益更偏稳健性与论文叙事，而非 points_met。

---

## 2. 候选方向

### A. 求解器 / 优化器蒸馏（线索 a）

#### A1. Decision-Focused residual distillation（决策损失蒸馏，而非动作 MSE）

- **出处**
  - Mandi, Kotary, Berden, Mulamba, Bucarey, Guns, Fioretto. *Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities*. JAIR 2024. https://www.jair.org/index.php/jair/article/view/15320
  - Kotary, Di Vito, Jacob, Van Hentenryck, Fioretto. *Learning Joint Models of Prediction and Optimization*. arXiv 2024. https://arxiv.org/abs/2409.04898
  - Amos & Kolter 系 OptNet/KKT 可微 QP 层综述见 *Differentiable Convex Optimization Layers in Neural Architectures: Foundations and Perspectives*（arXiv 2024 未核实是否已正式发表）. https://arxiv.org/abs/2412.20679
- **核心机制**：训练损失不再是 ‖a_π − a_solver‖²，而是**通过可微求解器把动作映到决策质量**（本仓库已有 `src/optimization/diff_engine.py`，float64 与不可变 `rate_calc` atol=1e-9），例如 L = DGLCR(s, a_solver) − DGLCR(s, a_π) 或对 ee/delivery 的任务损失。
- **适配性**：M4 纯 BC（8000 条演示）→ M5 SFTP 已把策略推到 0.9519，但 **JointOpt（60 步 warm-start）反而低于 25 步冷启动 DiffOpt（0.913 vs 0.957）**——缺口被论文归因为“逐槽在线优化 → 单次前向”的 amortization 损失。DFL 正是针对 predict-then-optimize 的这类损失设计的；本仓库的 diff_engine 使“微分穿过速率引擎”成为可能，而多数无线 RL 工作做不到。
- **与 banlist 区别**：不是对偶/惩罚调参，不是 CRPO/零空间梯度投影；优化器只出现在**损失与教师信号**里，不改 reward 公式、不投影策略梯度。
- **收益 / 成本**：收益指向 **0.6% 缺口**与更稳的收敛；成本 = 每步多一次（或 k 次）engine 反传，CPU 上 8000ep 预算需设计小 batch/低频更新；与 champion 的可比性要求固定评估口径（7×50）。

#### A2. On-policy / residual 求解器蒸馏（教师纠正学生**访问到的状态**）

- **出处**
  - Agarwal et al. *On-policy Distillation of Language Models: Learning from Self-Generated Mistakes*. ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html
  - FA-OPD：*Adversarial Dual On-Policy Distillation from Expressive Teacher*（ICML 2026，flow-matching 教师 + 双通道奖励/动作蒸馏）. https://icml.cc/media/icml-2026/Slides/61158_rJq3wAb.pdf （slides 链接，**正文 arXiv 号未经核实**）
  - SAILOR：*A Smooth Sea Never Made a Skilled SAILOR: Robust Imitation via Learning to Search*（从专家演示学 world/reward model，测试时可恢复）. https://arxiv.org/abs/2506.05294
  - 残差策略谱系：TRANSIC（在线纠正）https://arxiv.org/abs/2405.10315 ；Compliant Residual DAgger https://arxiv.org/abs/2506.16685
- **核心机制**：在**学生 rollout 访问的状态**上调用 SolverExpert/diff_engine 生成纠正目标（或“专家似然”奖励），对 BC 的分布偏移做闭环补救；也可让学生只预测相对教师的 **residual 动作**。
- **适配性**：教师已在 0/5/15/30 dB 全点越过 SDMA（M2 诊断）；当前 SFTP 是“BC + 衰减锚定”，学生一旦偏离演示支撑就无纠正。过载 24 用户的动作残差天然稀疏（α/d 已死、功率 logit 是主自由度），residual 参数化维数低。
- **与 banlist 区别**：不是 self-imitation 分桶，不是模型汤；教师信号来自**求解器前向**，不是策略自身分位数/权重平均。
- **收益 / 成本**：预期把 0.9519 往 DiffOpt 方向推 **0.2–0.5 pt** 量级（若缺口真是 amortization）；成本 = 在线调用教师，每 rollout 约 +25 步 Adam（solver_expert 现行配置）或一次前向残差。

#### A3. Amortized Optimization / L2O 的“学习优化器→策略”映射

- **出处**
  - NeurIPS 2024：*Can Learned Optimization Make Reinforcement Learning Less Difficult?*（OPEN）. https://papers.nips.cc/paper_files/paper/2024/file/09e1944b7f2372f9f81866470c59b663-Paper-Conference.pdf
  - μLO：*Compute-Efficient Meta-Generalization of Learned Optimizers*. https://arxiv.org/abs/2406.00153
  - L2O 收敛性：*Learning Provably Improves the Convergence of Gradient Descent*（NeurIPS 2025 slides，**论文出处未经核实**）. https://nips.cc/media/neurips-2025/Slides/115480.pdf
  - 无线侧邻接：可微预测控制 DPC（IEEE TSMC 2024）https://doi.org/10.1109/tsmc.2024.3368026
- **核心机制**：把 SolverExpert 的 25 步 Adam 精修**摊销**成网络：输入状态，输出精修后的 p_logits / 序分数；或用 L2O 的更新规则学习“比固定步数更快逼近 DiffOpt”的残差算子。
- **适配性**：DiffOpt 就是“每时隙 25 步优化”；策略是“一次前向”。A3 把 25 步**折叠进权重**，保留单时隙 1.16 ms 时延预算（`measure_structural_metrics` 实测）。
- **与 banlist 区别**：与梯度投影/CRPO 无关；不是课程。
- **收益 / 成本**：收益 = 推理时延不变而逼近 oracle；成本 = 需要大量 (s, a*) 对（可用现有 8000 演示 + 在线 DiffOpt 续采），训练比 BC 贵。

---

### B. 约束 RL 新范式（线索 b，**非对偶调参**）

#### B1. 目标条件化 / 可行性优先（ζ 家族的制度化）

- **出处**
  - FCSRL：*Feasibility Consistent Representation Learning for Safe RL*. arXiv 2024. https://arxiv.org/abs/2405.11718
  - *Learning to Generate All Feasible Actions*（可行性模型 → 只采可行动作）. IEEE Access 2024. https://doi.org/10.1109/access.2024.3376739
  - Safety Representations（ICLR 2025）：显式学安全表征做探索，而非重罚约束. https://proceedings.iclr.cc/paper_files/paper/2025/file/99fc8bc48b917c301a80cb74d91c0c06-Paper-Conference.pdf
  - Action Mapping for RL in Continuous Environments with Constraints. https://arxiv.org/abs/2412.04327
  - 边界条件化：*Boundary-to-Region Supervision for Offline Safe RL*（CTG 作硬边界而非对称条件）. https://arxiv.org/abs/2509.25727
- **核心机制**：把“约束”变成**条件变量 / 可行性证书 / 动作映射**，而不是拉格朗日乘子。臂 C 已证明 ζ 条件化可让功率随 ζ 单调缩放——这是本仓库**已付费验证过**的通道。
- **适配性**：评估口径是逐 SNR 点 delivery≥SDMA 且 queue≤1.1×SDMA，本质是**操作点可行性**问题。ζ→(delivery, queue, power) 的响应面一旦学出来，可做“刚好过门”的最省功率操作点，或按 SNR/负载选 ζ*。与“固定工作点冠军”互补：冠军是一个点，ζ-条件化是一条可调曲线（armC 结论原文）。
- **与 banlist 区别**：完全避开对偶/PID/CRPO/投影；也不做全局或查表 `group_alpha`（组已证明应放弃）。
- **收益 / 成本**：收益 = 可控工作点 + 可能提升 ee_ratio（第 4 字典序位）而不掉 points_met；成本 = 需重新定义评估是否允许“条件化选点”（协议问题，须用户裁定），且 armC 单种子交付 −1.8σ，需多种子裁决。

#### B2. 零对偶的安全学习：可行性判别器 + 动作过滤 / 投影到可行集

- **出处**：同 B1（Feasible Actions / Action Mapping）；多智能体图障碍函数 GCBF+（IEEE T-RO 2025）https://doi.org/10.1109/tro.2025.3530348 ；CBF-Learning 综述 https://arxiv.org/abs/2404.16879
- **核心机制**：训练一个轻量可行性/违约预测器（输入 obs+action，输出是否触碰 queue/overflow 红线），推理时把不动作投影或重采样到可行集——**约束满足在动作空间完成，不在奖励权重完成**。
- **适配性**：本问题的硬门是“逐点相对 SDMA”，可用 SDMA 在线指标或 DVQ 积压作为可行性伪标签；比再调 λ 便宜得多。
- **与 banlist 区别**：banlist 是“梯度操纵（投影策略梯度）”与“对偶调参”；此处是**动作层可行性模型**，不改策略梯度几何，不改 Lagrangian 权重。
- **收益 / 成本**：收益 = 保住 points_met 的保险丝；成本 = 误杀可行动作会压交付，需校准；与不可变物理文件无冲突（可行性模型在策略侧）。

---

### C. 结构化多维动作（线索 c）

#### C1. 生成式动作头：flow-matching / 离散扩散（针对剩余 2K 混合动作）

- **出处**
  - RSMA 邻接：*Energy-Efficient RSMA-enabled Low-altitude MEC Optimization Via Generative AI-enhanced DRL*（actor 内嵌 diffusion 生成混合动作）. https://arxiv.org/abs/2507.12910
  - *Reinforcement Learning with Discrete Diffusion Policies for Combinatorial Action Spaces*. https://arxiv.org/abs/2509.22963
  - ReinFlow（flow-matching 策略在线 RL 微调，精确似然）. https://arxiv.org/abs/2505.22094
  - 综述：*Diffusion Models for Reinforcement Learning: Foundations, Taxonomy, and Development*. https://arxiv.org/abs/2510.12253
  - 一步生成：*Revisiting Diffusion Q-Learning*… https://arxiv.org/abs/2508.13904
- **核心机制**：用生成模型表达多峰动作分布（功率份额与序的联合），再用 RL 微调；或把 flow 的确定性路径注入噪声以得到可算 likelihood 的策略（ReinFlow）。
- **适配性**：砍掉 α/d 后只剩 **p（softmax 份额）+ PL 序**；若专家演示在功率分布上多峰（不同用户分簇/水填充样式），Gaussian 对角头会平均掉模式。2507.12910 已在 RSMA+DRL 场景用 diffusion 生成混合动作，是最近的领域内证据。
- **与 banlist 区别**：不是 SIC 序“探针”，不是分桶 MoE；是**动作分布参数化**升级。注意：若只是把序头换成 Gumbel-Sinkhorn/NeuralSort 而不改训练目标，论文增量弱（见 C2）。
- **收益 / 成本**：收益 = 多峰动作拟合 + 对 seed 噪声更稳；成本 = 训练/推理更重（多步去噪 vs 1.16 ms 预算冲突，需一步/少步蒸馏），与“精简接口层”的叙事存在张力，须在论文里论证为何值得。

#### C2. 可微排序 / 排列策略的升级（仅当与目标损失耦合）

- **出处**
  - Gumbel-Sinkhorn（Mena et al., NeurIPS 2018）https://arxiv.org/abs/1805.07010 ；NeuralSort / SoftSort https://arxiv.org/abs/2006.16038 ；OT 可微排序 NeurIPS 2019.
  - REOrdering Patches（PL + Gumbel-top-k 策略）. https://arxiv.org/abs/2505.23751
  - Fast Slate Policy Optimization（超越 PL 的列表策略，TMLR 2023）.
- **核心机制**：把离散解码序的搜索从“z-scores + Gumbel-Top-K”升级为更平滑/更高阶的排列分布，或用 slate 目标训练。
- **适配性**：解码序是 4K 中唯一强结构离散块；但本地可解释性显示 z 头归因弱（首用户频率仅 0.15–0.19 vs 均匀 0.042），**教师本身序候选有限**（gain_desc / queue_last）。
- **与 banlist 区别**：banlist 禁止的是“SIC 解码序**探针**类实验”（双序/纯评估探针）；C2 若做，必须是**训练目标与排序损失耦合的完整方法**，否则就是换皮探针。
- **收益 / 成本**：预期收益**偏低**（序头非当前瓶颈），除非 A1 的决策损失显示 ∂J/∂order 显著。默认降为 A 的附属消融，不单独立项。

---

### D. 测试时计算 / anytime（线索 d）

#### D1. 预算化推理精修（anytime policy + 求解器 residual）

- **出处**
  - *Breaking the Performance Ceiling in Reinforcement Learning requires Inference Strategies*. https://arxiv.org/abs/2505.21236
  - AnytimeReasoner / BRPO（可变预算下的 anytime 目标，LLM 侧但预算机制可移植）. https://arxiv.org/abs/2505.13438
  - Test-Time Graph Search for GCRL（TTGS）. https://arxiv.org/abs/2510.07257
  - GAS: Generative Auto-bidding with Post-training Search（生成策略 + 后训练搜索）. ACM EC 2025. https://dl.acm.org/doi/10.1145/3701716.3715226
- **核心机制**：策略给出初始动作，**测试时**用 0–N 步 diff_engine/solver 在低维残差（p_logits）上精修；N=0 即纯策略（保持 1.16 ms），N>0 换 oracle 接近度。这与训练期 ExIt 互补。
- **适配性**：DiffOpt=0.9574 本身就是“每时隙 25 步测试时计算”；学习策略是摊销端点。D1 把二者变成**同一策略的预算轴**，可直接画 “delivery/EE vs N” 曲线——非常适合论文的 “anytime / compute-accuracy” 叙事，且**零重训**即可先做 gate。
- **与 banlist 区别**：不是梯度操纵；不是混入 `blend_alpha` 插值（那是对 Meta α 的线性混合，已证伪）。精修目标是引擎的真实 surrogate。
- **收益 / 成本**：收益 = 预期把 0.6% 缺口的一部分以可测量方式关掉，并给出部署旋钮；成本 = 推理时延上升，需报复杂度表；CPU 评估可接受，在线部署需用户接受预算。

#### D2. 学习式 anytime 头 / 早停精修门

- **出处**：与 D1 同族；可结合 *Can Learned Optimization…*（OPEN）学“何时停”。
- **机制**：一个小门控网络根据 (SNR, 队列, 残差范数) 决定精修步数；低 SNR 门紧处多修，高 SNR 少修。
- **适配性**：短板点在低 SNR 交付，高 SNR 队列——预算应倾斜到“门紧”维度。
- **与 banlist 区别**：早停本身（guard-plateau）在训练期已证伪；这里是**推理期计算门控**，不是 checkpoint 早停。
- **收益 / 成本**：小；作为 D1 的消融分支。

---

### E. 额外线索（无线 RL / 专家蒸馏 / 图注意力）

#### E1. 自监督 GAT 功率控制（图策略升级）

- **出处**：*Pilot Contamination-Aware Graph Attention Network for Power Control in CFmMIMO*（自监督 GAT，不需要标签求解解）. https://arxiv.org/abs/2506.00967 ；*Joint Power Control and Precoding … Sparse Multi-Dimensional GNN*. https://arxiv.org/abs/2507.01876
- **机制**：以速率/EE 代理目标自监督训练 GAT，免大规模监督标签；对 pilot/导频污染显式建模。
- **适配性**：SA-GAT Worker 已是图注意力；自监督损失可与 diff_engine 任务损失（A1）合并。过载 K=24>N_r=16 的干扰图正是 GAT 强项。
- **与 banlist 区别**：架构侧但非“分桶 MoE/obs 增广”；是图边/损失设计。
- **收益 / 成本**：中等；可能改善泛化与 mismatch 鲁棒性（论文已有 mis-specification 章节）。

#### E2. 无线 EE/RSMA 的 2023–2026 邻接工作（定位用）

- RSMA+UAV MEC+生成式 DRL（上文 C1）. https://arxiv.org/abs/2507.12910
- RIS 全双工 RSMA 多智能体 DRL（TVT 2024）. https://doi.org/10.1109/tvt.2024.3453253
- URLLC UAV-RSMA 双 DDPG（CMC 2025）. https://doi.org/10.32604/cmc.2025.063206
- ReLAx：QoS+能效联合 RAN 资源分配（MASS 2024）. https://doi.org/10.1109/mass62177.2024.00064
- 本地 wiki 已核：Uplink RSMA imperfect CSI（2501.19019，低 SNR CSIR 主导）、Lyapunov DPP for RL（2506.04291）。
- **定位**：领域内多为 **MDP 重写 + DDPG/SAC/PPO**，少有“可微引擎 + 求解器蒸馏 + 字典序硬门 + 摊销缺口量化”的组合——本问题的创新应在 **ML 算法层**（A/B/D），而非再换一个无线 DRL 皮。

#### E3. 从专家到策略的新蒸馏目标（超出普通 BC）

- 除 A2 外：*RLDG: Robotic Generalist Policy Distillation via RL*（RSS 2025，用 RL 蒸馏大策略到小策略）. https://doi.org/10.15607/rss.2025.xxi.028
- CUPID（影响函数筛选演示数据）. https://arxiv.org/abs/2506.19121 —— 与本仓库硬例挖掘/数据整形成功家族同向，可对 8000 条演示做**价值加权**而非均匀 BC。
- **与 banlist 区别**：不是分桶 self-imitation；是演示子集选择 / 蒸馏目标改造。

---

## 3. 优先级排序表

| 优先级 | ID | 方向 | 主线索 | 直接作用对象 | 预期收益 | 成本 | 风险 / 备注 |
|---|---|---|---|---|---|---|---|
| **P0** | **D1** | 预算化推理精修（anytime residual） | (d) | 0.6% amortization gap | 0.2–0.6 pt（可测曲线） | 低（零重训，eval-only gate） | 时延上升；须报预算-性能曲线 |
| **P1** | **A1** | Decision-focused residual 蒸馏 | (a) | 同上 | 0.2–0.5 pt | 中（diff_engine 反传） | CPU 训练预算；勿动不可变文件 |
| **P2** | **A2** | On-policy 求解器蒸馏 / residual | (a) | BC 分布偏移 | 0.1–0.4 pt | 中 | 教师在线调用；与 SFTP 叠加要预注册 |
| **P3** | **B1** | ζ 目标条件化制度化（可行性优先） | (b) | 可控操作点 / ee_ratio | 定性新叙事 + ee 旋钮 | 中（多种子裁决 armC） | 协议是否允许条件化选点待裁定；交付可能 −1.8σ |
| **P4** | **A3** | 摊销 25 步求解器（L2O） | (a) | 时延-精度 | 接近 oracle 且保持 1.16 ms | 高 | 需大量 (s,a*)；训练贵 |
| **P5** | **B2** | 可行性判别器 + 动作过滤 | (b) | points_met 保险丝 | 稳定性 | 低–中 | 误杀风险；需校准 |
| **P6** | **E1** | 自监督 GAT / 图损失 | (e) | 泛化 / mismatch | 中 | 中 | 与 A1 可合并 |
| **P7** | **C1** | Flow/diffusion 动作头 | (c) | 多峰动作 | 不确定 | 高 | 与精简叙事冲突；仅当多峰证据出现 |
| **P8** | **C2** | 可微排序升级 | (c) | 序头 | 低 | 中 | 默认作 A1 消融，不单独立项 |
| — | E3 | 演示价值加权蒸馏 | (e) | 数据整形 | 低–中 | 低 | 可与 A2 捆绑 |

**排序原则**：① 先打已量化缺口（0.6%）；② 复用已付费资产（diff_engine、8000 演示、ζ 通道、2K 动作）；③ 零重训 gate 优先于重训；④ 避开全部 banlist 机制族。

---

## 4. 推荐的首个实验设计

### 名称
**T-ANYTIME：预算化推理精修 gate（P0，零重训）**

### 假设
- **H1**：在**不重训**的前提下，对冠军 `best_guard.pt` 的每时隙动作在 **p_logits 残差子空间**（2K 中的功率份额维；α/d 已死、序可选固定或轻修）做 N∈{0,5,10,25} 步 diff_engine/solver 精修，能把 7 点平均交付从 0.9519 向 DiffOpt 0.9574 推进 **≥0.2 pt**，且 **points_met 保持 7/7**。
- **H0**：Δ≤0.2 pt 或任何点掉门 → amortization 缺口不可由浅层精修关闭，转 P1（训练期 decision-focused）或否定“缺口=摊销”解释。

### 臂（单变量，评估口径完全不变）

| 臂 | 定义 | 备注 |
|---|---|---|
| **T0** | 冠军原样（N=0） | 基线，须复现 7/7、0.9519±seed 带 |
| **T1** | N=5 步，只修 p_logits | 时延增量最小 |
| **T2** | N=10 步，只修 p_logits | |
| **T3** | N=25 步，只修 p_logits（对齐 DiffOpt 步数） | 上界参照，允许时延超预算 |
| **T4**（可选） | N=10 步，p_logits + 序 z 同时修 | 仅当 T1–T3 有信号；**不得**变成 SIC 序探针叙事——只报联合 residual |

- 固定：`scripts/evaluate_snr_sweep*.py` 口径、imperfect CSI、7×50、同一 checkpoint；精修调用 `src/optimization/solver_expert.py` / `diff_engine.py`，**绝不**改 `channel.py` / `rate_calc.py` / 评估脚本。
- 种子：≥3（43/44/45），报告 mean±CI 与配对 t 检验；σ_seed≈0.005 为噪声地板。

### 判定门（预注册，不得事后改）
1. **主门**：points_met = 7 且 mean delivery 提升 **≥0.2 pt**（> 0.4σ_seed）→ 接受 H1，进入 P1 训练期摊销。
2. **次门**：ee_ratio 不劣于 T0 − 0.005，mean power 不高于 SDMA 终止线语义下的冠军值 +5%。
3. **成本门**：报告每时隙精修墙钟；T1/T2 应给出“部署可选预算”定位。

### 失败判据
- 任一点 7→6：立即停止该臂，回退 T0，禁止为保 points_met 调评估口径。
- 3 种子全部 Δ<0.2 pt：记 **负结果**，写入 `logs/opt_knowledge.json` 与 `.trellis/tasks/.../experiment-results.md`，结论改为“缺口非浅层摊销可关”，转向 A1/B1。
- 若 T3 明显优于 T1/T2 但超时延：保留为 oracle 分析行，不进 champion 配方（与论文“DiffOpt 只作上界行”决策一致）。

### 成本估计
- **计算**：零训练；7 点×50 episode×4–5 臂×3 seeds，约与数次官方评估同量级（CPU 上小时级～半天），主要是每步 25 次 engine 前向。
- **工程**：评估侧 wrapper 脚本 + 精修开关；不碰不可变文件；`fast` 门禁（现约 400+ 测试）须全绿。
- **人力**：用户裁决是否接受“评估时精修”的协议合法性（本质是测试时计算，不是作弊——但须在论文中明确标注 N 与墙钟）。

### 后续衔接
- H1 成立 → P1：用同款 residual 目标做训练期 decision-focused 微调，目标是 **N=0 时**逼近 T3。
- H1 失败但 ζ 响应面仍有效 → P3：把 ζ 做成官方“可调操作点”能力（协议新章节），而非平均分竞争。

---

## 5. 参考文献（可核查链接）

> 列表按报告出现顺序。标注 **未核实** 者：未能打开原文核对作者/页码/会议正式录信息，引用前请人工复核。

1. Mandi et al., Decision-Focused Learning: Foundations…, JAIR 2024. https://www.jair.org/index.php/jair/article/view/15320
2. Kotary et al., Learning Joint Models of Prediction and Optimization, arXiv 2024. https://arxiv.org/abs/2409.04898
3. Differentiable Convex Optimization Layers… (OptNet/KKT 综述), arXiv 2024. https://arxiv.org/abs/2412.20679 （发表状态**未核实**）
4. Minimizing Surrogate Losses for DFL using Differentiable Optimization, arXiv 2025. https://arxiv.org/abs/2508.11365
5. Online Decision-Focused Learning, arXiv 2025. https://arxiv.org/abs/2505.13564
6. Agarwal et al., On-policy Distillation of Language Models, ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html
7. FA-OPD, ICML 2026 slides. https://icml.cc/media/icml-2026/Slides/61158_rJq3wAb.pdf （作者/正式版**未核实**）
8. SAILOR: Robust Imitation via Learning to Search, arXiv 2025. https://arxiv.org/abs/2506.05294
9. TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction, arXiv 2024. https://arxiv.org/abs/2405.10315
10. Compliant Residual DAgger, arXiv 2025. https://arxiv.org/abs/2506.16685
11. Can Learned Optimization Make RL Less Difficult? (OPEN), NeurIPS 2024. https://papers.nips.cc/paper_files/paper/2024/file/09e1944b7f2372f9f81866470c59b663-Paper-Conference.pdf
12. μLO: Compute-Efficient Meta-Generalization of Learned Optimizers, arXiv 2024. https://arxiv.org/abs/2406.00153
13. Learning Provably Improves the Convergence of Gradient Descent, NeurIPS 2025 slides. https://nips.cc/media/neurips-2025/Slides/115480.pdf （正式出处**未核实**）
14. Drgoňa et al., Learning Constrained Parametric DPC Policies, IEEE TSMC-S 2024. https://doi.org/10.1109/tsmc.2024.3368026
15. FCSRL: Feasibility Consistent Representation Learning for Safe RL, arXiv 2024. https://arxiv.org/abs/2405.11718
16. Learning to Generate All Feasible Actions, IEEE Access 2024. https://doi.org/10.1109/access.2024.3376739
17. Safety Representations for Safer Policy Learning, ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/99fc8bc48b917c301a80cb74d91c0c06-Paper-Conference.pdf
18. Action Mapping for RL in Continuous Environments with Constraints, arXiv 2024. https://arxiv.org/abs/2412.04327
19. Boundary-to-Region Supervision for Offline Safe RL, arXiv 2025. https://arxiv.org/abs/2509.25727
20. GCBF+: Neural Graph Control Barrier Function, IEEE T-RO 2025. https://doi.org/10.1109/tro.2025.3530348
21. Learning Control Barrier Functions… Survey, arXiv 2024. https://arxiv.org/abs/2404.16879
22. Energy-Efficient RSMA-enabled Low-altitude MEC via Generative AI-enhanced DRL, arXiv 2025. https://arxiv.org/abs/2507.12910
23. RL with Discrete Diffusion Policies for Combinatorial Action Spaces, arXiv 2025. https://arxiv.org/abs/2509.22963
24. ReinFlow: Fine-tuning Flow Matching Policy with Online RL, arXiv 2025. https://arxiv.org/abs/2505.22094
25. Diffusion Models for RL: Foundations, Taxonomy, and Development, arXiv 2025. https://arxiv.org/abs/2510.12253
26. Revisiting Diffusion Q-Learning… One-Step Action Generation, arXiv 2025. https://arxiv.org/abs/2508.13904
27. REOrdering Patches Improves Vision Models (PL policy), arXiv 2025. https://arxiv.org/abs/2505.23751
28. Learning Permutations with Sinkhorn Policy Gradient, arXiv 2018. https://arxiv.org/abs/1805.07010
29. SoftSort / NeuralSort 连续松弛, arXiv 2020. https://arxiv.org/abs/2006.16038
30. Breaking the Performance Ceiling in RL requires Inference Strategies, arXiv 2025. https://arxiv.org/abs/2505.21236
31. AnytimeReasoner / Budget Relative Policy Optimization, arXiv 2025. https://arxiv.org/abs/2505.13438
32. Test-Time Graph Search for Goal-Conditioned RL (TTGS), arXiv 2025. https://arxiv.org/abs/2510.07257
33. GAS: Generative Auto-bidding with Post-training Search, ACM EC 2025. https://dl.acm.org/doi/10.1145/3701716.3715226
34. Pilot Contamination-Aware GAT for Power Control, arXiv 2025. https://arxiv.org/abs/2506.00967
35. Joint Power Control and Precoding with Sparse Multi-Dim GNN, arXiv 2025. https://arxiv.org/abs/2507.01876
36. Pala et al., RIS FD-RSMA Multi-Agent DRL, IEEE TVT 2024. https://doi.org/10.1109/tvt.2024.3453253
37. URLLC Service in UAV RSMA, CMC 2025. https://doi.org/10.32604/cmc.2025.063206
38. ReLAx: DRL Resource Allocation for Next-G RANs, MASS 2024. https://doi.org/10.1109/mass62177.2024.00064
39. RLDG: Robotic Generalist Policy Distillation via RL, RSS 2025. https://doi.org/10.15607/rss.2025.xxi.028
40. CUPID: Curating Data your Robot Loves with Influence Functions, arXiv 2025. https://arxiv.org/abs/2506.19121
41. （本地已核）Uplink RSMA with Imperfect CSI, arXiv 2501.19019；Lyapunov DPP tailored for RL, arXiv 2506.04291；见 `research-wiki/papers/`。

---

## 附：与本地实验事实的对齐备忘

| 本地事实 | 对本调研的约束 |
|---|---|
| DiffOpt 0.9574 vs 冠军 0.9519（+0.6%）；JointOpt 反而更差 | 缺口叙事 = **摊销/单次前向**，不是“优化深度不够” → 主打 A1/A3/D1，不主打“再多几步离线优化” |
| σ_seed≈0.005；M5.1–M5.3 干预全在噪声带 | 任何新臂必须 ≥2–3 seeds，效应量门设 0.2 pt+ |
| α/d 死重，2K=48 保留 7/7 | C 类生成式头只作用于剩余块；勿复活死头 |
| 分组近优单组坍缩，强行分化 −4σ | **禁止**任何分组多样性/组间正则提案 |
| ζ 条件化功率单调、交付略降 | B1 是“可控操作点”能力，不是平均分冠军替代品 |
| 从零 RL 全灭（DDPG 5/7 最好），专家初始化是使能成分 | 蒸馏/专家轴（A/E3）优于“换更强免模型 RL” |
