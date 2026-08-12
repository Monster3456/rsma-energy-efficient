<!-- TRELLIS:START -->
# Trellis Instructions

These instructions are for AI assistants working in this project.

This project is managed by Trellis. The working knowledge you need lives under `.trellis/`:

- `.trellis/workflow.md` — development phases, when to create tasks, skill routing
- `.trellis/spec/` — package- and layer-scoped coding guidelines (read before writing code in a given layer)
- `.trellis/workspace/` — per-developer journals and session traces
- `.trellis/tasks/` — active and archived tasks (PRDs, research, jsonl context)

If a Trellis command is available on your platform (e.g. `/trellis:finish-work`, `/trellis:continue`), prefer it over manual steps. Not every platform exposes every command.

If you're using Codex or another agent-capable tool, additional project-scoped helpers may live in:
- `.agents/skills/` — reusable Trellis skills
- `.codex/agents/` — optional custom subagents

Managed by Trellis. Edits outside this block are preserved; edits inside may be overwritten by a future `trellis update`.

<!-- TRELLIS:END -->

---

## optimize-loop 跨轮知识库（版本跟踪）

- **路径**：`logs/opt_knowledge.json`
- **用途**：`scripts/optimize_loop.py` 的跨 run 累积知识库，记录优化循环中的「失败方向库 + 成功方向库」。任何会话在提出新的优化方向前，**必须先读取本文件**，避免重复已验证失败的方向。
- **结构**：`{"failed_directions": [...], "successful_directions": [...]}`，每条含 `iter`（轮次）、`channel`、`direction`（方向描述）、`params_tried`、`verdict`（失败/成功结论）、`rounds_tried`。
- **读写约定**：
  - 读：`optimize_loop.py` 启动时经 `load_knowledge()` 加载；代理/人只读参考，不手改条目。
  - 写：仅由 `optimize_loop.py` 在每轮结束时经 `save_knowledge()` 整体写入（全量覆盖式，非追加），人工不要并发编辑。
  - 若需作废某条教训，在下一轮循环内由程序改写，或人工在循环停止状态下编辑后立即 `python -m json.tool` 校验合法性。
- **Git 跟踪策略**：`logs/` 是运行产物目录（默认忽略），但本文件通过 `.gitignore` 中 `logs/*` + `!logs/opt_knowledge.json` 显式豁免，**纳入版本跟踪**，与产物忽略规则区分。知识库变更后应随相关改动一起 commit。
- **与 Trellis 学习通道的分工**：任务级经验教训的沉淀通道是 `.trellis/workflow.md` Phase 3.3（写回 `.trellis/spec/`）；本文件是 optimize-loop 专属的跨轮实验知识。具备普适性的教训（如护栏拦截模式、调参规律）可从本文件提炼后提升到 `.trellis/spec/`；反向引用见 `.trellis/workflow.md` Phase 3.3。

# 项目定位（非托管区，`trellis update` 不会覆盖）

上行 RSMA-MU-MIMO 能效优化研究项目：用分层强化学习（SA-GAT Worker + Meta-Controller 的分层 PPO）在交付率/队列硬约束下最大化能量效率（EE），由无人值守优化循环（`scripts/optimize_loop.py`）驱动迭代。

## 核心目录职责

- `src/env/` — 环境层：信道模型、RSMA 速率计算、gym 环境、波束成形、流量模型
- `src/models/` — 网络层：`actor_critic.py`（Worker）、`meta_controller.py`（Meta）
- `src/training/` — 训练层：PPO、Lagrangian 约束、buffer、课程调度、训练入口
- `src/optimization/` — 优化辅助：Lyapunov、功率控制、Meta 奖励
- `scripts/` — 训练入口、SNR 扫描评估、基线实现（`baselines/`）、优化循环
- `tests/` — 物理模型与训练逻辑的单测；`logs/` — 实验产物（忽略）与跨轮知识库 `opt_knowledge.json`（版本跟踪，见上方专节）

## 不可变模块（改动会使历史实验失去可比性）

权威清单以 [`scripts/optimize_loop_config.yaml`](scripts/optimize_loop_config.yaml) 的 `immutable`、`immutable_modules` 与 `code_guard.immutable_paths` 为准（同源，勿另立副本）：

- `src/env/channel.py` — 信道物理模型，零容忍
- `src/env/rate_calc.py` — RSMA 速率计算，零容忍
- `scripts/evaluate_snr_sweep.py` — 评估脚本，指标口径不可变
- `src/env/rsma_env.py` 中物理调用段（`self.channel` / `self.rate_calc`）零容忍；仅奖励等非物理部分可改

可修改白名单与修改范围见该配置文件的 `mutable_modules` 与 `code_guard.mutable_files`。

## 深链

- [MODEL.md](MODEL.md) — 系统完整数学建模（优化变量、约束、分层 RL 算法）
- [HANDOFF_LYAPUNOV_DRL.md](HANDOFF_LYAPUNOV_DRL.md) — Lyapunov 引导 DRL 实施交接
- [HANDOFF_v13m_fix.md](HANDOFF_v13m_fix.md) — LDPTRLQ 奖励改造交接（含当前基线状态）
- [CLAUDE.md](CLAUDE.md) — 分层架构与常用命令补充说明
