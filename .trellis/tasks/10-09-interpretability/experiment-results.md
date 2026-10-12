# 实验记录：可解释性分析（10-09-interpretability）

## 状态

- 代码 + 单测 + 图表脚本 + 论文章节/引用：**已完成**（2026-10-09）
- 全量 7-SNR 分析运行：**已完成**（用户 2026-10-10 执行，runtime ≈ 7.7h CPU；`logs/interpretability/summary.json`）
- 图入 `docs/paper/figs/`（用户跑在 `docs/paper/fig/`，已复制对齐 `\graphicspath{{figs/}}`）；`main.tex` 占位已用真实数字填充并两遍编译通过（10 页）

## 冒烟结果（SNR 20 dB，1 episode，短 MC —— 仅验证管线，非论文数值）

### SVEA（label=served，m_sv=1000, B=200，类不平衡缓解 cap=10:1）

| SNR | SVEA_neg（e_j<0 支配子集） | c̃ | PSV | acc_sub/acc_full |
|---|---|---|---|---|
| 20 | queue, split_mask_alpha | 0.182 | 1.0181 | 0.9979 / 0.9802 |

- 解读（冒烟级）：队列积压与公共/私有分割 α 是「用户是否被服务」的支配特征；
  MeanFieldCriticBias 手挑通道基线 {h_gains, queue, power_history, qos_priority} 与本博弈
  自动发现的子集在 queue 上重合（α 是 Meta 级输出，critic 基线不含 meta 通道）。

### 局部归因验证（kernel=模型无关代理；eg=Expected Gradients，refs=zero/means/boot/neutral）

| target | recall@zero | @means | @boot | @neutral | 参照对 Jaccard@5 范围 |
|---|---|---|---|---|---|
| P_total | 0.6 | 0.6 | 0.4 | 0.4 | 0.25–0.67 |
| d_mean | 0.2 | 0.4 | 0.6 | 0.4 | 0.25–0.67 |
| z_top1 | 0.6 | 0.6 | 0.2 | 0.0 | 0.25–0.67 |

- 形态与文献 2 结论一致：**归因对参照明显敏感**（Jaccard 低至 0.25）；means/boot
  参照与模型无关代理的 top-k 一致性最好（recall 0.4–0.6），zero 参照在最依赖
  语义的目标（z_top1）上最不可靠（recall 0.0–0.6）——支持「用边际均值/经验背景
  作为论文口径参照」的方法选择。

## 全量结果（用户 2026-10-10 运行，n_episodes=12，7 SNR）

### SVEA（label=served，类不平衡缓解，常数特征按 null player 排除）

| SNR | SVEA_neg | c̃ | PSV | acc_sub/acc_full |
|---|---|---|---|---|
| 0  | queue | 1.000 | 0.966 | 0.852/0.883 |
| 5  | queue | 1.000 | 0.953 | 0.829/0.869 |
| 10 | queue | 1.000 | 0.990 | 0.941/0.951 |
| 15 | queue | 1.000 | 1.003 | 0.984/0.981 |
| 20 | queue, split_mask_alpha | 0.886 | 1.008 | 0.998/0.989 |
| 25 | queue | 0.724 | 1.010 | 0.993/0.984 |
| 30 | (none) | 0.672 | 0.997 | 0.993/0.996 |

- 结论叙述：queue 在 0–25dB 全部是唯一支配特征（20dB 时 split-α 加入）；30dB 全体被服务、
  博弈无可判别信号 → 无支配特征。c̃ 随 SNR 下降（多数类比例降）。PSV ≥ 0.95。
- 常数特征（每固定 SNR 点内确定性）：arrivals(3)、λ_bucket(6)、Dinkelbach_η(10) → 从排列
  游走排除（null-player 公理：边际恒 0 → φ=0、e=c̃/13>0 永不支配；其余特征值不变）；
  同时根治了 99:1 不平衡 + hiGHS "Status 0: Not Set" 退化（列归一 + 有界变量 + 1e-9 抖动兜底）。

### 局部归因验证（EG vs KernelSHAP recall@5，P_total 目标）

| SNR | zero | means | bootstrap | neutral |
|---|---|---|---|---|
| 0  | 0.6 | 0.8 | 0.8 | 0.8 |
| 5  | 0.6 | 1.0 | 0.8 | 0.8 |
| 10 | 0.4 | 0.6 | 0.8 | 0.8 |
| 15 | 0.4 | 0.6 | 0.8 | 0.6 |
| 20 | 0.8 | 0.8 | 1.0 | 0.6 |
| 25 | 0.8 | 0.8 | 0.8 | 0.8 |
| 30 | 0.6 | 0.8 | 0.6 | 1.0 |

- bootstrap/means 参照与模型无关代理一致性最好（多数点 0.8–1.0）；zero 参照最差（0.4–0.8）
  —— 文献 2 参照敏感性结论复现；论文报告口径定为 bootstrap。
- KernelSHAP P_total（SNR15, means）：queue(−3.23)/h_gains(−2.14) 主导（负号=该状态相对参照
  降低总功率，EE 省电行为）；EG z_top1 在 30dB 归因趋零（z 近平坦 → 解码序无信息）。

## 复现/运行命令

```powershell
PYTHONPATH=.; .venv\Scripts\python.exe scripts/interpretability_analysis.py --cache logs/interpretability/states.npz --out_dir logs/interpretability --figs_dir docs/paper/fig
```

## 方法学决策记录（写入知识库，防重复踩坑）

1. **hiGHS "Status 0: Not Set"**：无界变量 + 高度类不平衡（served 标签 99:1）下的数值退化。
   加固 = 有界变量（±1e8）+ 全局列归一（max-|·|）+ 类不平衡缓解（多数类下采样 cap=10:1，可平衡时各半）
   + 求解器回退链（highs → highs presolve=False → highs-ipm）。前三者是仿射/加权不变手段，博弈值精确不变。
2. **served 标签在高 SNR 极度不平衡**（交付率 ~99%）——SVEA 的线性博弈在样本层面几乎退化；
   缓解后该标签仍表示「谁没被服务」的少数类判别，队列/α 成为支配特征，语义自洽。
3. **参照敏感性是特性不是 bug**（文献 2 主旨）：means/boot 最稳，zero 最差；论文口径固定为
   means + boot（如需要，可把 zero 参照的 recall 值作为「敏感性下界」报告）。
4. **meta 特征空间与 worker 观测空间分离**（12 通道 vs 13+2 块）——归因分别进行，不可混比；
   Meta 的 snr/capacity 通道（7/8）是全局广播，在通道级归因里单位语义不同，报告中注明。
## P0 取证（2026-10-11）：历史块玩家归因 + queue_growth 标签

工具升级：`--include-history` 把 12 个历史滞后块按特征类型并成 3 个组玩家（hist_h_gains / hist_queue / hist_power，共 18 玩家），EG/KernelSHAP/掩码/图例全部支持（`features.player_slices`），新增单测 `test_player_slices_history_and_masking`。

运行：SNR {0,15,30} × (queue_growth 标签 SVEA + 含历史块的 KernelSHAP/EG，bootstrap 参照，n_pairs=256/bg=16/n_eg=128/n_attributed=48，状态缓存 `logs/interpretability/states.npz` 复用)。

### KernelSHAP P_total（bootstrap）——历史组玩家归因

| SNR | 前三原生 | 历史组玩家（含 rank） | 3 常数块(arrivals/λ_bucket/η) |
|---|---|---|---|
| 0  | delta_queue −0.101 / power_history +0.061 / queue −0.046 | hist_queue #3 +0.053、hist_power #5 −0.045、hist_h #7 −0.019 | 全部 #8–10，幅度小 ~4–7× |
| 15 | queue +0.119 / rate_history +0.084 / power_history +0.081 | hist_power #4 +0.067、hist_queue #7 +0.027、hist_h #8 +0.010 | 幅度小幅，rank ~10 |
| 30 | queue +0.108 / split_α +0.060 / h_gains −0.035 | **hist_queue #2 −0.063**、hist_power #4 −0.052、hist_h #7 −0.023 | 低 |

**结论：历史窗口不可直接砍。** 三 SNR 点历史块合计归因 0.05–0.14，与头部原生特征同级（30dB 时 hist_queue 排第 2 大功率决策驱动）——"H=4 承载真实时序信息（过载/高 SNR 更明显）"。"观测 603→315（删历史）"的压缩假设被证据否定。可留的次优动作：lag 级归因（当前是 3 组聚合）定位最有效的 lag 数 → H 可缩不可零。

### queue_growth 标签 SVEA

0/15/30dB 全部 **SVEA_neg = ∅**（c̃=1.0 平衡）——"queue 主导"是 **served 标签特有的结论**，跨标签不成立；论文 Limitations 已注明单标签口径，此处增加跨标签证据。

### 与既有结论的合流

- d_mean 归因在含历史块的 18 玩家下依然全部 ≈0（d≡1 恒定）——坍缩结论不受历史块影响。
- 3 常数块（arrivals/λ_bucket/Dinkelbach-η）仍是**最稳的低信息/剪枝候选**（评估点内零方差 + 全点低归因），但收益只占 obs 72/603 维，属小幅化简。
- 优化建议排序见对话结论（P0 之后）：常数块剪枝 > Worker α 死维清理 > H 维持现状（不砍）。

## P3 取证（2026-10-12）：lag 级历史归因 → H 不可收缩（负结果关闭）

工具：`--history-lag-wise` 把 3 个历史组玩家拆成 12 个独立 lag 玩家（H=4 全量 24 玩家），
EG/KernelSHAP/掩码全链支持 + 单测。运行：SNR {0,15,30} × P_total（bootstrap 参照，
n_pairs=128/bg=8/n_eg=64/n_attr=32，缓存状态复用）。

| SNR | 原生 top-3 | 历史 lag 玩家 top-3（|·|） |
|---|---|---|
| 0 | power_hist 0.058 / Δq 0.032 / arrivals 0.029 | hist_power_l3 0.034、hist_queue_l1 0.033、hist_h_l1 0.032（散布）|
| 15 | Δq −0.129 / power_hist 0.105 / queue 0.056 | hist_h_l1 0.059、hist_h_l2 0.042、hist_power_l1 0.038（l1 略强但 l2/l3 同级）|
| 30 | queue 0.415 / λ_bucket −0.129 / dvq 0.091 | **hist_queue_l3 0.302**（最旧 lag 最大！）、hist_h_l2 0.087、hist_queue_l1 −0.064 |

**结论：历史信息跨 4 个 lag 均匀分布（0dB 甚至 old-lag 主导），无"信息集中在前 1–2 lag"
的证据 → H=4 维持，P3（H 收缩）以负结果关闭。** 与 P0 的"历史不可砍"结论互相印证
（P0 看整体、P3 看 lag 分布）。

**附带**：P_total 归因里 λ_bucket 在 30dB 升到第 2（−0.129）——注意 λ_bucket 是训练期
（SNR 域随机化下）的信号，评估点内恒定；不影响 A-prune 决策（其只针对评估点内零方差块）。
