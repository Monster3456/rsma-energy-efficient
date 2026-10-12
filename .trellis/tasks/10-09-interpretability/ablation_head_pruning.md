# 预注册 Ablation：P2 决策头裁剪（drain / 解码序 / 分组）

> 预注册纪律同前两份方案；交付物与评测口径不变（best_guard + 官方 7×50）。

## 背景与证据（可解释性全量 + rollout 统计 + 18/24 玩家归因）

论文《Interpretability》章节与 `logs/interpretability/summary.json` 证明冠军策略的三个动作头
**近状态不变**（非归因假象，rollout 直测）：

| 头 | 证据 | 幅度 |
|---|---|---|
| **drain d_k** | d ≡ 1.0（std < 1e-4，全部 16,800 时隙、全 SNR）；EG d_mean 归因 ≈ 0 | 完全饱和 |
| **解码序 z** | z_top1 归因 ≈ 0（高 SNR 尤其）；首用户频率 0.15–0.19 vs 均匀 0.042 | 弱结构 |
| **Meta 分组 g** | 每时隙 ~1–2 组（多数 1 组）；group_logp 归因 ≈ 0；α 低 SNR 恒定、30dB 才有方差 | 近坍缩 |

机制：solver-expert 教师本就固定 d=1、解码序候选有限（gain-desc/queue-last）、Meta 分组在
SFTP 蒸馏后收敛到默认——策略忠实复制了教师的"常量"选择。

## 假设

H0：对这些"近常量"头做**结构化裁剪**（同预算重训），官方评估与基线臂**带内无差异**
（points_met 相同、|Δmean deliv| ≤ 0.005）→ 裁剪安全，容量/方差下降。
H1：裁剪后掉点/出带 → 头虽近恒定但训练动力学依赖它们（如熵/探索），拒绝裁剪。

## Arms（单变量，每个独立对照；基座 = seed-43 A-prune 0.9526，已有）

| Arm | 裁剪变量 | 代码改动点 | 其余固定（A-prune 配方） |
|---|---|---|---|
| **P2a drain-fix** | d 块固定 1.0（act 输出 d=1，不学习） | `ActorCriticAttn` d head 输出替换为常数 1（保留 3K 动作接口，仅推理/损失禁用 d 梯度）；或 main.py 侧 mask | 同 |
| **P2b order-fix** | 解码序固定 gain-desc（不学 PL z 头） | act() 采样路径用固定 order；PL log_prob 旁路 | 同 |
| **P2c G1-group** | n_groups=1（分组头退化为单组；保留 α 通路） | `HierarchicalConfig.n_groups=1` + meta 分组分支短路 | 同 |

每个 arm：2 seeds（43/44 复用既有路径；BC-init 复用 `_bc_pruned_init.py` 按 arm 生成），
与 **P1 早停联动**（若 P1 H0 已接受，本 campaign 直接开早停，进一步省预算）。

## 门（accept/reject，预注册不事后改）

1. 每 arm 官方评估 **points_met = 7**，否则拒绝 H0（该 arm）。
2. |mean deliv(arm) − 0.9526| ≤ 0.005（P1 若改判则用对应新基线），超出拒绝。
3. 结果为观察指标：网络参数量、forward 耗时、动作空间维数变化（报告收益）。
4. 三个 arm **独立裁决**（不是一起过才转正）；至少 2 seeds。

## 若 H0 被接受

- 对应头从架构移除（缩小 user_head 输出 / action 空间），`CLAUDE.md`/论文 Proposed Framework
  同步；α 通路与功率头不动（α 在 30dB 有真信号、功率头是唯一强状态驱动）。
- 与 P1（早停）、A-prune（观测剪枝）合并为"新官方配置三件套"，统一预注册一条龙后转正。

## 执行（用户或我，待 P1/seed-44 裁决后启动）

命令前缀统一照 `ablation_early_stop.md` 的 A-prune 模板 + arm 专属改动（P2a/P2b 需先行
小代码实现并过 fast 门禁——若交付则逐 arm 预注册更新本文件）。

## 结果（预注册后补）

| Arm | seeds | points_met | mean deliv | Δ vs 0.9526 | 参数/耗时变化 | 裁决 |
|---|---|---|---|---|---|---|
| P2a drain-fix | 43/44 |  |  |  |  | H0 accept/reject |
| P2b order-fix | 43/44 |  |  |  |  |  |
| P2c G1-group | 43/44 |  |  |  |  |  |