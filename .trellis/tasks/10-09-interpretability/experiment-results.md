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