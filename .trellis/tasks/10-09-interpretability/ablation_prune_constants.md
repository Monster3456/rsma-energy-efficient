# 预注册 Ablation：观测常数块剪枝（可解释性取证驱动）

> 预注册纪律（见 paper `Statistical honesty` / RESEARCH_BRIEF）：先写假设与门，再动实验；
> 门禁决定不接受事后修改；结论按 seed 散布口径报告，不做单种子因果断言。

## 背景与动机（可解释性取证）

`logs/interpretability/summary.json`（7 SNR × 12 episodes，冠军最佳守卫检查点）与 P0 取证
（SNR 0/15/30 × 18 玩家历史块归因）给出三档证据：

1. **常数块低信息**：`arrivals / lambda_bucket / dinkelbach_eta` 三块在每个固定 SNR 评估点内
   **零方差**，且 18 玩家 KernelSHAP 归因在全部测点都位居 10 名开外、幅度比头部
   （queue / h_gains / power_history）小 4–7×。→ 剪枝候选。
2. **历史块不可砍**：hist_* 组归因与头部同级（30dB 时 hist_queue 是功率决策第 2 大驱动）
   → H=4 维持；本 ablation **不**触历史块。
3. **queue_growth 标签下 SVEA_neg=∅**：所述"常数块低信息"结论属 served 标签口径，
   本 ablation 只依赖归因幅值证据（标签无关），不依赖 SVEA_neg 集合。

**注意（诚实边界）**：三个"常数块"只在固定评估点内恒定；训练期它们随 episode（SNR
域随机化、Lagrangian 对偶 λ、Dinkelbach q* 外层更新）变化。因此本 ablation 检验的
不是"删信息"而是"删**点内零信息 + 跨点可由 snr_norm/状态增广替代的低价值输入**后，
训练/交付是否种子带内不变"。

## 假设

H0（零假设）：移除 arr—3 块后重训，7 点门 + 平均交付与冠军**种子带内无差异**
（|Δdelivery| ≤ 0.005，points_met = 7）。
H1：移除后交付/门退化（点数掉 或 delivery 掉出带）→ 训练期信号不可省，保留原 obs。

## Arms（单变量控制）

| Arm | 变量 | 其余固定 |
|---|---|---|
| **A-prune** | system 侧新增 `obs_prune_constant_blocks=True`：从 `_get_observation` 移除块 3/6/10（arrivals、λ_bucket、Dinkelbach-η 三个 K 块 → obs 603→531 维，per-user 25→22） | 训练配方 / 超参 / 守卫 / 评估口径 与 champion 完全一致 |
| **A-ctrl** | 同一配方、原始 603 维 obs 重训 | 同上（同预算对照，控种子与代码版本漂移） |

可选扩展 Arm（不在本次承诺内）：**H-lag 归因**（lag 级历史块归属 → H 可缩性），
以及 **Worker α 头物理删除**（死维出口重排，接口/检查点迁移成本单独评估）。

## 门（accept/reject，预注册，不事后改）

1. A-prune 官方评估 **points_met = 7**（与 champion 同），否则拒绝 H0（=H1 成立）。
2. 平均 delivery 落在 champion seed 带内：|deliv(A-prune) − 0.9519| ≤ 0.005；
   超带则拒绝 H0。
3. 每 arm 至少 **2 个种子**（seed 43/44 复用口径），报告均值±散布；
   Δ 判断用带内比较（paper 的 seed-resolution 标准），不做单种子 t 断言。
4. 队列/溢出/EE 为观察指标（不设门，报告不裁决）。

## 执行（用户主导；我提供改动 + 命令）

1. **代码改动**（本次交付）：给 `src/env/rsma_env.py::_get_observation` 加
   `config.system.obs_prune_constant_blocks`（默认 False = 官方口径零变化）开关，
   命中时跳过块 3/6/10 的拼接；同步更新 `ObsLayout`/分析侧布局解析
   （`src/analysis/features.py` 需知裁剪后块序仍保持 feature-major、13→10 原生块），
   `scripts/evaluate_snr_sweep_temporal.py` 类 wrapper 增加对应 flag 传递。
   ⚠️ 不动 channel/rate_calc，不影响物理调用点；obs 增删属"观测接线性改动"（允许区）。
2. **训练**：A-prune 与 A-ctrl 各 1 次 BC-init 8k + SFTP 全流程（~8 GPU-h/arm，用户机器），
   种子 43/44。
3. **评估**：官方 `scripts/evaluate_snr_sweep(_temporal).py`，guard v2 守卫选点。
4. **分析复查**：砍块后的检查点重新跑可解释性（`--include-history` 工具已就绪），
   确认 queue/h_gains 主导故事未漂移（served 标签）。

## 代码改动清单（交付时逐项核对）

- [x] `src/config.py`: `SystemConfig.obs_prune_constant_blocks: bool = False`（commit 后置）
- [x] `rsma_env._get_observation`: 按开关跳过块 3/6/10（state_aug=False 时只剪 arrivals）；
      `__init__` obs_dim 同步 -3K（或 -K）
- [x] `tests/test_obs_prune.py`（fast，5 用例）：默认关闭、维度降级、块对齐（保留块/历史/全局逐位一致）、state_aug 分支、分析侧布局/玩家切片
- [x] `src/analysis/features.py`：`ObsLayout.prune_constants` / `resolve_layout` / `native_blocks` / `player_slices` 自适应剪枝空间
- [x] `scripts/evaluate_snr_sweep_temporal.py` + `scripts/train.py` + `src/training/main.py`（worker_d_user -3）+ `src/analysis/models.py`：`--obs-prune-constant-blocks` 全链透传
- [ ] 结果回填：本文件"结果"节（用户运行后）

### 运行命令（用户）

```powershell
# A-ctrl：冠军同配方、原始 603 维 obs（种子 43/44）
PYTHONPATH=.; .venv\Scripts\python.exe scripts/train.py --exp_name intp_A_ctrl --total_episodes N --device cpu --csi_mode imperfect --seed 43
# A-prune：同一配方 + 剪枝开关（obs 531 维）
PYTHONPATH=.; .venv\Scripts\python.exe scripts/train.py --exp_name intp_A_prune --total_episodes N --device cpu --csi_mode imperfect --seed 43 --obs_prune_constant_blocks

# 评估（champion 时序 H=4 口径）
# A-ctrl:  python scripts/evaluate_snr_sweep_temporal.py --obs_history_window 4 --checkpoint <A_ctrl_ckpt>
# A-prune: python scripts/evaluate_snr_sweep_temporal.py --obs_history_window 4 --obs-prune-constant-blocks --checkpoint <A_prune_ckpt>
```

## 结果（预注册后补）

| Arm | points_met | mean deliv (seeds) | Δ vs 0.9519 | 裁决 |
|---|---|---|---|---|
| A-ctrl (603 维) |  |  |  | 对照 |
| A-prune (531 维) |  |  |  | H0 accept/reject |

## 若 H0 被接受（剪枝安全）的后续

1. 正式把常数块从官方 obs 移除（默认 False→True），更新论文 System Model/观测描述
   （13→10 原生块）与可解释性章节的玩家口径。
2. 可选：基于 P0 的 H-lag 归因决定 H 是否可缩（H=4→1/2），另行预注册。
3. Worker α 头删除作为独立工程项，评估检查点迁移后执行。