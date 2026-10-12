# 预注册 Ablation：P1 guard-plateau 早停（训练预算削减）

> 预注册纪律：假设与终止规则先写死；门禁不事后修改；不改变评测口径（交付物仍为
> best_guard.pt，早期停止只改训练时长，不改选点逻辑与官方评估）。

## 背景与证据（2026-10-12，A-ctrl/A-prune 两臂数据）

- A-ctrl：guard 峰 **ep=4003 / update 391，avg_deliv 0.9847**；此后 ~2900 集（ep 4003→6900）
  的 guard 评估**从未刷新该 best**——继续训练纯属预算浪费。
- 同规律见于 A-prune（guard 峰 ep=1228）与 champion（best_guard 亦为中途选中）。
- 官方评估显示：best_guard（中途点）与 final.pt（6900 集）的 7×50 结果在 seed 带内等价
  （A-ctrl: best_guard 0.9527；final 同带内）。
- 结论假设：**guard 收敛后训练不再产生更优可交付策略** → 提前终止可省 ~35–40% GPU/时间，
  且不改变交付成绩分布。

## 假设

H0（零假设）：启用 guard-plateau 早停的臂（termination: patience 次 guard 评估未刷新
best 且已训 ≥ min_episodes）与同配方满预算臂，**best_guard 官方评估无差异**
（|Δmean deliv| ≤ 0.005，points_met 相同）。
H1：早停臂 best_guard 显著更差（掉点/出带）→ 守卫选点在收敛后仍可能改善，拒绝早停。

## 终止规则（预注册，先写死，不事后改）

- `eval_guard_early_stop=True`、`patience=2`（连续 2 次 guard 评估未刷新 best_avg_deliv）、
  `min_episodes=3000`（课程/挖掘/α 引导已充分展开后才允许停）。
- 早停触发即 break 主循环并保存 final.pt（带 `early_stopped=True` 标记）；
  交付物仍为 best_guard.pt（按点门选中的那一次评估的状态）。

## Arms

| Arm | 变量 | 其余固定 |
|---|---|---|
| **E-stop-44** | seed 44，A-prune 配方 + 早停开（patience=2, min=3000） | 与 seed-43 A-prune（已跑，0.9526）完全同配方 |
| （对照）seed-43 A-prune 满预算 | 无早停（0.9526，已存在） | — |

单臂新增 + 历史对照即可（早停只影响停止点，不影响配方——对照已存在，无需重跑满预算臂）。

## 门（accept/reject）

1. 早停 arm 官方评估 **points_met = 7**；否则拒绝 H0。
2. **|deliv(E-stop-44) − deliv(A-prune s43)| ≤ 0.005**；超出拒绝 H0。
3. 记录实际停止 episode 与**节约的预算比例**（1 − stop_ep/6900），作为收益报告
   （观察指标，不设门）。
4. 若早停点 < 3000 集（规则失灵）→ 判定规则 bug，回查，不算裁决。

## 执行（用户）

```powershell
# A-prune seed-44 + 早停（复用现有 bc_init? 否——早停验证用独立 seed 44 路径，
# 先跑 _bc_pruned_init.py --seed 44 --out logs\intp_A_prune_s44 生成 BC-init）
$env:PYTHONUNBUFFERED=1; $env:PYTHONIOENCODING='utf-8'; $env:PYTHONPATH='.'
.venv\Scripts\python.exe scripts\train.py --exp_name intp_A_prune_s44_es --device cuda --csi_mode imperfect --seed 44 --obs_history_window 4 --obs_prune_constant_blocks --no-bucket-experts --clip_ratio 0.1 --lr_actor 3e-5 --lr_log_std 1e-4 --actor-warmup-updates 30 --curltrac --hard-mining --eval-guard --eval-guard-early-stop --eval-guard-early-stop-patience 2 --eval-guard-early-stop-min-episodes 3000 --solver-expert --solver-expert-episodes 40 --solver-expert-anchor-updates 100000 --solver-expert-anchor-decay 0.999 --meta-freeze-episodes 2000 --resume logs\intp_A_prune_s44\checkpoints\bc_init.pt --episodes 6900 --save_interval 15

# 评估（同口径）
.venv\Scripts\python.exe scripts\evaluate_snr_sweep_temporal.py --obs_history_window 4 --obs-prune-constant-blocks --checkpoint logs\intp_A_prune_s44_es\checkpoints\best_guard.pt --n_episodes 50
```

## 结果（预注册后补）

| Arm | stop_ep | 预算节约 | points_met | mean deliv | Δ vs s43 (0.9526) | 裁决 |
|---|---|---|---|---|---|---|
| E-stop-44 |  |  |  |  |  | H0 accept/reject |

## 若 H0 被接受

- 早停参数（patience=2, min=3000）写进新官方训练默认（与 A-prune 转正一并），
  论文补"训练预算"一句（guard 收敛后终止，~35–40% 节约）。
- P2（决策头裁剪）循此模板另开 campaign。