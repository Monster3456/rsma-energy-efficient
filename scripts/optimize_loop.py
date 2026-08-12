#!/usr/bin/env python3
"""
SA-GAT PPO 自动优化循环编排器
=================================
完全无人值守运行，通过 Qoder CLI 驱动 LLM 决策，
自动训练、评估、调参、引入创新方法，直至达到性能目标。

Usage:
    python3 scripts/optimize_loop.py              # 启动优化循环
    python3 scripts/optimize_loop.py --status     # 查看状态
    python3 scripts/optimize_loop.py --stop       # 优雅退出
    python3 scripts/optimize_loop.py --dry-run    # Mock 模式验证流程
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import yaml

# Cloud Agents API
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_agents_api import (
    CloudAgentsClient, CloudAgentsError, CloudAgentsAuthError,
    CloudAgentsTimeoutError, CloudAgentsQuotaError,
)

# ===========================================================================
# 常量
# ===========================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "scripts" / "optimize_loop_config.yaml"
STATE_PATH = PROJECT_ROOT / "scripts" / "optimize_loop_state.json"
STOP_FLAG = PROJECT_ROOT / "scripts" / ".optimize_loop_stop"
KNOWLEDGE_PATH = PROJECT_ROOT / "logs" / "opt_knowledge.json"
BEST_DIR = PROJECT_ROOT / "logs" / "opt_best"
BEST_CKPT = BEST_DIR / "best.pt"
# Cloud Agents Session（长复用，跨轮次保持上下文）
CA_SESSION_ID_FILE = PROJECT_ROOT / "scripts" / ".ca_session_id"

# 模块级：invoke_cloud_agent 重建 session 后回传给主循环的新 session id
_ACTIVE_SESSION_ID = ""

# 超参通道白名单: pending_params key -> train.py 命令行 flag
# LLM 仅允许输出下述 key，其余一律忽略（护栏）
HYPERPARAM_ARG_MAP = {
    "lr_actor": "--lr_actor",
    "lr_critic": "--lr_critic",
    "meta_lr": "--meta_lr",
    "T_s": "--T_s",
    "V": "--V",
    "V_min": "--V_min",
    "V_max": "--V_max",
    "V_exponent": "--V_exponent",
    "blend_alpha": "--blend_alpha",
    "w_overflow": "--w_overflow",
    "w_outage": "--w_outage",
    "w_power_exceed": "--w_power_exceed",
    "delivery_threshold": "--delivery-threshold",
    "lagrangian_lr": "--lagrangian-lr",
    "lagrangian_init": "--lagrangian-init",
    "lagrangian_max": "--lagrangian-max",
    "lagrangian_warmup": "--lagrangian-warmup",
    "alpha_guidance_scale": "--alpha_guidance_scale",
    "alpha_guidance_warmup": "--alpha_guidance_warmup",
    "group_alpha_min": "--group_alpha_min",
    "group_alpha_max": "--group_alpha_max",
}

# ===========================================================================
# 配置加载
# ===========================================================================


def load_config() -> dict:
    """加载 YAML 配置文件"""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ===========================================================================
# 状态管理 (Task 6)
# ===========================================================================


def load_or_init_state(config: dict) -> dict:
    """加载已有状态或初始化新状态"""
    if STATE_PATH.exists():
        with open(STATE_PATH, "r") as f:
            state = json.load(f)
        print(f"[STATE] 恢复迭代状态: iter={state['iteration']}")
        return state

    state = {
        "iteration": 0,
        "start_time": datetime.now().isoformat(),
        "mode": "lightweight",
        "best_score": None,          # 字典序四元组 [达标点数, 平均delivery, -平均queue, ee_ratio]
        "best_metrics": {
            "delivery": 0.0,
            "queue": 999.0,
            "ee_ratio": 0.0,
            "power": 999.0,
        },
        "best_checkpoint": None,     # logs/opt_best/best.pt
        "best_params": {},           # 达到 best 时的累积超参
        "best_git_hash": None,       # 达到 best 时的 git commit
        "history": [],
        "stalled_count": 0,          # 停滞方向数（替代 consecutive_no_improvement）
        "current_direction": None,   # {desc, channel, params, rounds_tried}
        "pending_params": {},        # 下一轮训练注入的超参（累积）
        "failed_directions": [],     # 失败方向库
        "successful_directions": [], # 成功方向库
        "innovations_applied": [],
        "current_checkpoint": config["baseline"]["checkpoint"],
        "last_eval": {},             # 上一轮评估结果，供下轮决策使用（方案 X 先决策后训练）
        "terminated": False,
        "termination_reason": None,
    }
    # 跨 run 加载历史知识库
    kb = load_knowledge()
    state["failed_directions"] = kb.get("failed_directions", [])
    state["successful_directions"] = kb.get("successful_directions", [])
    save_state(state)
    print("[STATE] 初始化新状态")
    return state


def save_state(state: dict) -> None:
    """持久化状态到 JSON"""
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_knowledge() -> dict:
    """加载跨 run 持久化的双知识库（失败库 + 成功库）"""
    if KNOWLEDGE_PATH.exists():
        try:
            with open(KNOWLEDGE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"failed_directions": [], "successful_directions": []}


def save_knowledge(state: dict) -> None:
    """将双知识库持久化到 opt_knowledge.json（跨 run 累积）"""
    KNOWLEDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    kb = {
        "failed_directions": state.get("failed_directions", []),
        "successful_directions": state.get("successful_directions", []),
    }
    with open(KNOWLEDGE_PATH, "w") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)


def print_status(state: dict) -> None:
    """打印当前状态摘要"""
    print("=" * 60)
    print("SA-GAT PPO 自动优化循环 — 当前状态")
    print("=" * 60)
    print(f"  迭代次数: {state['iteration']}")
    print(f"  运行模式: {state['mode']}")
    print(f"  启动时间: {state['start_time']}")
    print(f"  最佳指标:")
    print(f"    delivery: {state['best_metrics']['delivery']:.4f}")
    print(f"    queue:    {state['best_metrics']['queue']:.2f}")
    print(f"    EE ratio: {state['best_metrics']['ee_ratio']:.4f}")
    print(f"  停滞方向数: {state.get('stalled_count', 0)}")
    print(f"  best checkpoint: {state.get('best_checkpoint') or '未建立'}")
    print(f"  best score: {state.get('best_score')}")
    print(f"  当前 checkpoint: {state['current_checkpoint']}")
    print(f"  失败方向库: {len(state.get('failed_directions', []))} 条")
    print(f"  成功方向库: {len(state.get('successful_directions', []))} 条")
    if state["innovations_applied"]:
        print(f"  已引入创新: {len(state['innovations_applied'])} 项")
    print(f"  已终止: {state['terminated']}")
    if state["termination_reason"]:
        print(f"  终止原因: {state['termination_reason']}")
    print("=" * 60)


# ===========================================================================
# 训练模块
# ===========================================================================


def _safe_eval_numeric(val) -> Optional[float]:
    """安全求值纯数字算术表达式（如 '0.25*8' → 2.0）；含变量/非法内容返回 None。
    用于净化 LLM 超参输出（argparse 只认 float，表达式会直接崩溃）。"""
    import ast
    import operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
           ast.UAdd: lambda x: x}

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_ev(node.operand))
        raise ValueError("unsupported expression")

    try:
        return _ev(ast.parse(str(val).strip(), mode="eval"))
    except Exception:
        return None


def run_fast_training(state: dict, config: dict, dry_run: bool = False) -> dict:
    """执行快速验证训练"""
    iter_num = state["iteration"]
    exp_name = f"{config['fast_training']['exp_name_prefix']}_{iter_num}"
    checkpoint = state["current_checkpoint"]

    cmd = [
        "python3", "scripts/train.py",
        "--exp_name", exp_name,
        "--episodes", str(config["fast_training"]["episodes"]),
        "--rollout_steps", str(config["fast_training"]["rollout_steps"]),
        "--batch_size", str(config["fast_training"]["batch_size"]),
        "--device", config["fast_training"]["device"],
        "--csi_mode", config["fast_training"]["csi_mode"],
    ]

    # 如果有 checkpoint，使用 partial_resume
    if checkpoint and Path(PROJECT_ROOT / checkpoint).exists():
        cmd.extend(["--partial_resume", str(PROJECT_ROOT / checkpoint)])

    # 超参通道注入：pending_params 映射为 train.py 命令行参数（数值净化后注入）
    for key, val in state.get("pending_params", {}).items():
        flag = HYPERPARAM_ARG_MAP.get(key)
        if not flag:
            continue
        sval = val
        try:
            float(str(val).strip())
        except (ValueError, TypeError):
            sval = _safe_eval_numeric(val)
        if sval is None:
            print(f"[TRAIN] !!! 超参 {key}={val!r} 非数值且无法折算，丢弃该参数")
            continue
        cmd.extend([flag, str(sval)])

    print(f"[TRAIN] 启动训练 iter-{iter_num}: {exp_name}")
    print(f"[TRAIN] 命令: {' '.join(cmd)}")

    if dry_run:
        print("[TRAIN] DRY-RUN: 跳过实际训练")
        # 返回 mock checkpoint 路径
        mock_ckpt = f"logs/{exp_name}/checkpoints/best.pt"
        return {"checkpoint": mock_ckpt, "log_file": f"logs/{exp_name}_train.log"}

    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=7200  # 2h 超时
    )

    if result.returncode != 0:
        print(f"[TRAIN] 训练失败:\n{result.stderr[-500:]}")
        return {"checkpoint": None, "error": result.stderr[-500:]}

    # 查找生成的 best checkpoint
    ckpt_dir = PROJECT_ROOT / "logs" / exp_name / "checkpoints"
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        ckpt_path = f"logs/{exp_name}/checkpoints/best.pt"
    else:
        # fallback to final
        ckpt_path = f"logs/{exp_name}/checkpoints/final.pt"

    log_file = f"logs/{exp_name}_train.log"
    print(f"[TRAIN] 训练完成: {ckpt_path}")
    return {"checkpoint": ckpt_path, "log_file": log_file}


# ===========================================================================
# 评估模块
# ===========================================================================


def run_snr_sweep_eval(
    checkpoint: str, config: dict, dry_run: bool = False
) -> dict:
    """执行 SNR sweep 评估，解析输出为结构化指标"""
    cmd = [
        "python3", "scripts/evaluate_snr_sweep.py",
        "--checkpoint", str(PROJECT_ROOT / checkpoint),
        "--n_episodes", str(config["evaluation"]["n_episodes"]),
        "--csi_mode", "imperfect",
        "--baselines", ",".join(config.get("evaluation", {}).get("baselines", ["sdma"])),
    ]

    print(f"[EVAL] 启动 SNR sweep 评估: {checkpoint}")

    if dry_run:
        print("[EVAL] DRY-RUN: 返回 mock 指标")
        return _mock_eval_metrics()

    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=3600
    )

    if result.returncode != 0:
        print(f"[EVAL] 评估失败:\n{result.stderr[-500:]}")
        return {"error": result.stderr[-500:]}

    # 解析评估输出
    metrics = parse_eval_output(result.stdout)
    print(f"[EVAL] 评估完成: delivery={metrics.get('avg_delivery', 0):.4f}, "
          f"queue={metrics.get('avg_queue', 999):.2f}, "
          f"EE_ratio={metrics.get('ee_ratio', 0):.4f}")
    return metrics


def parse_eval_output(output: str) -> dict:
    """解析 evaluate_snr_sweep.py 的输出为结构化指标"""
    metrics = {
        "per_snr": {},
        "avg_delivery": 0.0,
        "avg_queue": 999.0,
        "avg_power": 999.0,
        "avg_ee": 0.0,
        "sdma_avg_ee": 0.0,
        "sdma_avg_power": 999.0,
        "ee_ratio": 0.0,
        "raw_output": output[-2000:],  # 保留原始输出尾部
    }

    # 解析表1宽表: 每行一个 SNR 点，列顺序固定
    # | SNR | PPO-RSMA P | PPO-RSMA Q | PPO-RSMA Deliv | PPO-RSMA Rsum | PPO-RSMA Ovf
    #       | SDMA P | SDMA Q | SDMA Deliv | SDMA Rsum | SDMA Ovf | NOMA... | EqualPower... |
    lines = output.split("\n")

    for line in lines:
        if "|" not in line:
            continue
        cols = [c.strip() for c in line.split("|") if c.strip()]
        # 表1宽表数据行至少 11 列 (SNR + PPO 5列 + SDMA 5列)
        if len(cols) < 11:
            continue
        # 第一列必须是数字(SNR)，否则为表头/分隔行
        try:
            snr = int(float(cols[0]))
            ppo_power = float(cols[1])
            ppo_queue = float(cols[2])
            ppo_delivery = float(cols[3])
            ppo_rsum = float(cols[4])
            sdma_power = float(cols[6])
            sdma_queue = float(cols[7])
            sdma_delivery = float(cols[8])
            sdma_rsum = float(cols[9])
        except (ValueError, IndexError):
            continue
        metrics["per_snr"][snr] = {
            "ppo": {
                "power": ppo_power,
                "queue": ppo_queue,
                "delivery": ppo_delivery,
                "r_sum": ppo_rsum,
            },
            "sdma": {
                "power": sdma_power,
                "queue": sdma_queue,
                "delivery": sdma_delivery,
                "r_sum": sdma_rsum,
            },
        }

    # 计算全域平均
    ppo_deliveries = []
    ppo_queues = []
    ppo_powers = []
    ppo_ees = []
    sdma_ees = []
    sdma_powers = []

    for snr, data in metrics["per_snr"].items():
        if "ppo" in data:
            ppo_deliveries.append(data["ppo"].get("delivery", 0))
            ppo_queues.append(data["ppo"].get("queue", 999))
            ppo_powers.append(data["ppo"].get("power", 999))
            r_sum = data["ppo"].get("r_sum", 0)
            power = data["ppo"].get("power", 1)
            if power > 0:
                ppo_ees.append(r_sum / power)
        if "sdma" in data:
            sdma_powers.append(data["sdma"].get("power", 999))
            r_sum_s = data["sdma"].get("r_sum", 0) if "r_sum" in data["sdma"] else 0
            power_s = data["sdma"].get("power", 1)
            if power_s > 0 and r_sum_s > 0:
                sdma_ees.append(r_sum_s / power_s)

    if ppo_deliveries:
        metrics["avg_delivery"] = sum(ppo_deliveries) / len(ppo_deliveries)
    if ppo_queues:
        metrics["avg_queue"] = sum(ppo_queues) / len(ppo_queues)
    if ppo_powers:
        metrics["avg_power"] = sum(ppo_powers) / len(ppo_powers)
    if ppo_ees:
        metrics["avg_ee"] = sum(ppo_ees) / len(ppo_ees)
    if sdma_ees:
        metrics["sdma_avg_ee"] = sum(sdma_ees) / len(sdma_ees)
    if sdma_powers:
        metrics["sdma_avg_power"] = sum(sdma_powers) / len(sdma_powers)

    # EE ratio vs SDMA
    if metrics["sdma_avg_ee"] > 0:
        metrics["ee_ratio"] = metrics["avg_ee"] / metrics["sdma_avg_ee"]

    return metrics


_MOCK_CALL_COUNT = 0


def _mock_eval_metrics() -> dict:
    """Dry-run mock 指标：前2轮 score 递增(accept 建立 best)，之后停滞(reject→回滚→升级)"""
    global _MOCK_CALL_COUNT
    _MOCK_CALL_COUNT += 1
    c = _MOCK_CALL_COUNT

    if c == 1:
        pass_snrs, avg_deliv, avg_q, ee = 2, 0.90, 18.0, 1.02
    elif c == 2:
        pass_snrs, avg_deliv, avg_q, ee = 3, 0.93, 14.0, 1.05
    else:
        # 停滞：始终不如第2轮（触发 reject + 回滚）
        pass_snrs, avg_deliv, avg_q, ee = 2, 0.91, 16.0, 1.03

    snr_points = [0, 5, 10, 15, 20, 25, 30]
    per_snr = {}
    for i, snr in enumerate(snr_points):
        if i < pass_snrs:
            ppo = {"power": 0.5, "queue": 5.0, "delivery": 0.99, "r_sum": 40.0}
        else:
            ppo = {"power": 0.6, "queue": 20.0, "delivery": 0.90, "r_sum": 35.0}
        per_snr[snr] = {
            "ppo": ppo,
            "sdma": {"power": 0.7, "queue": 12.0, "delivery": 0.95, "r_sum": 30.0},
        }

    return {
        "per_snr": per_snr,
        "avg_delivery": avg_deliv,
        "avg_queue": avg_q,
        "avg_power": 0.5,
        "avg_ee": 30.0 * ee,
        "sdma_avg_ee": 30.0,
        "sdma_avg_power": 0.7,
        "ee_ratio": ee,
        "raw_output": f"[MOCK iter={c}] pass_snrs={pass_snrs}",
    }


# ===========================================================================
# 终止条件检查
# ===========================================================================


def check_termination(eval_result: dict, config: dict) -> bool:
    """检查是否满足所有终止条件"""
    term = config["termination"]
    d_ratio = term.get("delivery_sdma_ratio", 1.0)
    q_ratio = term.get("queue_sdma_ratio", 1.1)

    # 检查每个 SNR 点的 delivery 和 queue（相对 SDMA 基线）
    for snr in term["snr_points"]:
        data = eval_result.get("per_snr", {}).get(snr, {})
        ppo = data.get("ppo", {})
        sdma = data.get("sdma", {})
        if not ppo or not sdma:
            return False
        if ppo.get("delivery", 0) < sdma.get("delivery", 0) * d_ratio:
            return False
        if ppo.get("queue", 999) > sdma.get("queue", 999) * q_ratio:
            return False

    # 检查 EE 优势
    if eval_result.get("ee_ratio", 0) < term["ee_advantage_min"]:
        return False

    # 检查功率约束
    if eval_result.get("avg_power", 999) >= eval_result.get("sdma_avg_power", 0):
        return False

    return True


def check_safety_bounds(state: dict, config: dict) -> Optional[str]:
    """检查安全边界，返回终止原因或 None"""
    safety = config["safety"]

    if state["iteration"] >= safety["max_iterations"]:
        return f"达到最大迭代次数 ({safety['max_iterations']})"

    max_stalled = safety.get("max_stalled_directions",
                             safety.get("max_consecutive_no_improvement", 10))
    if state.get("stalled_count", 0) >= max_stalled:
        return f"连续 {max_stalled} 个方向停滞无突破"

    start = datetime.fromisoformat(state["start_time"])
    elapsed = (datetime.now() - start).total_seconds() / 3600
    if elapsed >= safety["max_runtime_hours"]:
        return f"超过最大运行时间 ({safety['max_runtime_hours']}h)"

    return None


# ===========================================================================
# 模式升级判断
# ===========================================================================


def check_escalation(state: dict, eval_result: dict, config: dict) -> str:
    """基于停滞方向数的三级升级判断。

    返回运行模式:
      - lightweight: 超参精调
      - innovation_search: 代码创新 + IEEE/GitHub 双源检索
      - aggressive_innovation: 加大创新力度（更激进的结构改动）
    """
    esc = config["escalation"]
    stalled = state.get("stalled_count", 0)

    aggressive_th = esc.get("aggressive_directions", 6)
    innovation_th = esc.get("innovation_directions",
                            esc.get("no_improvement_rounds", 3))

    # 停滞方向数达到激进阈值
    if stalled >= aggressive_th:
        return "aggressive_innovation"

    # 停滞方向数达到创新阈值 → 代码创新 + 双源检索
    if stalled >= innovation_th:
        return "innovation_search"

    # delivery 差距过大 → 直接创新
    if eval_result.get("avg_delivery", 0) < esc.get("delivery_gap_threshold", 0.90):
        return "innovation_search"

    # EE 全域低于 SDMA → 直接创新
    if esc.get("ee_below_sdma", True) and eval_result.get("ee_ratio", 0) < 1.0:
        return "innovation_search"

    return "lightweight"


# ===========================================================================
# Qoder CLI 调用 (Task 5)
# ===========================================================================


def build_analysis_prompt(
    state: dict, eval_result: dict, config: dict, mode: str
) -> str:
    """构建传给 Qoder CLI 的分析 prompt"""
    # Gap 分析
    term = config["termination"]
    gap_lines = [
        "## Gap 分析（当前 vs 终止条件）",
        f"- delivery: {eval_result.get('avg_delivery', 0):.4f} (目标: 逐点 ≥ SDMA×{term.get('delivery_sdma_ratio', 1.0)})",
        f"- queue: {eval_result.get('avg_queue', 999):.2f} (目标: 逐点 ≤ SDMA×{term.get('queue_sdma_ratio', 1.1)})",
        f"- EE ratio vs SDMA: {eval_result.get('ee_ratio', 0):.4f} (目标 >= {term['ee_advantage_min']})",
        f"- power vs SDMA: {eval_result.get('avg_power', 999):.4f} vs {eval_result.get('sdma_avg_power', 0):.4f}",
    ]

    # 趋势（最近3轮）
    trend_lines = ["## 最近 3 轮指标趋势"]
    for h in state["history"][-3:]:
        trend_lines.append(
            f"- iter-{h['iter']}: delivery={h['metrics'].get('delivery', 0):.4f} "
            f"queue={h['metrics'].get('queue', 999):.2f} "
            f"EE_ratio={h['metrics'].get('ee_ratio', 0):.4f} "
            f"[{'improved' if h.get('improvement') else 'no improvement'}]"
        )

    # 指标优先级硬约束（最高优先，所有模式共享）
    priority_lines = [
        "## 指标优先级硬约束（最高优先，违反=方向判负）",
        "- delivery 是第一优先硬约束：任何方向必须保持或提升 delivery，绝对不允许牺牲 delivery 换 EE/queue",
        "- 若某方向导致 delivery 下降，即使 EE_ratio 提升也判负",
        "- 评分字典序：先比达标点数（delivery 逐点 >= SDMA），再比平均 delivery，再比 -queue，最后才比 EE_ratio",
        "- 禁止：功率投影/功率封顶/BC 克隆 SDMA 等以牺牲 delivery 为代价的方向",
        "- 允许：提升 delivery 的前提下顺带优化 EE/queue 的方向",
    ]

    # 历史决策（简化——长 Session 自动累积，只注入最近2轮防压缩丢失）
    history_lines = ["## 最近2轮决策（完整历史由 Session 上下文保持）"]
    for h in state["history"][-2:]:
        history_lines.append(
            f"- iter-{h['iter']}: score={h.get('score', [0,0,0,0])[0]}点 "
            f"{h.get('decision', 'N/A')[:60]}"
        )

    # 读取当前关键代码片段
    code_lines = ["## 当前关键代码片段"]
    code_lines.extend(_read_code_snippets())

    # 训练日志尾部
    log_lines = ["## 训练日志（最后50行）"]
    log_lines.extend(_read_train_log_tail(state, config))

    # 不可变约束
    constraint_lines = [
        "## 不可变约束（绝对不可修改）",
        f"- 系统模型: {config['immutable']['system_model']}",
        f"- 动作空间: {config['immutable']['action_space']}",
        "- 不可修改代码: 信道物理模型 src/env/channel.py、速率计算 src/env/rate_calc.py（改了实验不可比）",
        "- rsma_env.py 中禁止改动 self.rate_calc / self.channel 物理模型调用",
        "",
        "## 可修改代码白名单（已放宽，支持目录前缀与新增文件）",
        "- src/env/rsma_env.py (奖励函数 + 观测增广/动作解析，保持维度不变)",
        "- src/training/ (整目录，含新增文件)",
        "- src/models/ (整目录，含新增文件，保持输入输出维度不变)",
        "- src/config.py",
        "- src/optimization/ (整目录)",
        "- scripts/train.py (训练入口)",
        "- scripts/baselines/ (基线脚本目录)",
        "- scripts/evaluate_snr_sweep.py (仅限兼容性改动，禁止改评估逻辑/指标口径)",
    ]

    # 知识库注入（失败库 + 成功库）
    knowledge_section = _build_knowledge_section(state)

    # 结构化输出格式要求（方案 X：code_innovation 只出战略，不出具体代码）
    output_format = (
        "## 输出格式要求（必须严格遵守）\n"
        "你只负责出【战略】，具体代码由本地代码 agent（qodercli）读真实文件后自主实现。\n"
        "你必须在回复末尾输出一个 JSON 对象（用 ```json 代码块包裹）：\n"
        "```json\n"
        "{\n"
        '  "channel": "hyperparameter" 或 "code_innovation",\n'
        '  "direction": "本轮创新方向的简短描述（一句话）",\n'
        '  "hyperparams": {"参数名": 数值},              // 仅 hyperparameter 通道\n'
        '  "rationale": "为何这个方向能破瓶颈的理论依据（仅 code_innovation）",\n'
        '  "references": "IEEE/arXiv/GitHub 参考方法或链接（仅 code_innovation，来自 InnovationScout）"\n'
        "}\n"
        "```\n"
        "说明：code_innovation 通道不要输出具体代码或 code_edit_instruction，"
        "只需给出创新方向(direction) + 理论依据(rationale) + 参考方法(references)；"
        "本地 qodercli 会读真实代码自主设计实现。\n"
        "超参通道仅允许以下参数：lr_actor, lr_critic, meta_lr, T_s, V, V_min, "
        "V_max, V_exponent, blend_alpha, w_overflow, w_outage, w_power_exceed, "
        "delivery_threshold, lagrangian_lr, lagrangian_init, lagrangian_max, "
        "lagrangian_warmup, alpha_guidance_scale, group_alpha_min, group_alpha_max。\n"
    )

    # 组装 task 指令（三级模式）
    if mode == "aggressive_innovation":
        task_instruction = (
            "## 任务指令\n"
            "当前处于【激进创新模式】。已有 6+ 个方向停滞无突破，常规创新不足。\n"
            "**硬性要求：必须先调用 WebSearch 工具联网检索 IEEE Xplore 和 GitHub，不能只用知识库**\n"
            "1. 【必须执行】调用 WebSearch 检索关键词：\n"
            "   - \"constrained RL delivery guarantee 2024 2025\"\n"
            "   - \"safe RL primal-dual Lyapunov IEEE 2024\"\n"
            "   - \"GitHub constrained RL implementation PyTorch\"\n"
            "2. 从检索结果中筛选 1-2 个结构上全新的方法（不能只是知识库里的老方法）\n"
            "3. 结合下方失败库，彻底排除已知无效方向\n"
            "4. 提出一个基于检索到的新方法的方案（可涉及奖励结构重设计、约束机制替换）\n"
            "5. 在 rationale 里必须引用检索到的具体论文标题/GitHub 仓库名+链接\n"
            "6. 通过 code_innovation 通道给出具体代码修改指令，保持维度不变\n"
        )
    elif mode == "innovation_search":
        task_instruction = (
            "## 任务指令\n"
            "当前处于【创新搜索模式】。纯调参已停滞多个方向。\n"
            "1. 分析当前瓶颈根因（delivery/queue/EE 哪个是主要短板）\n"
            "2. 联网检索 IEEE Xplore 和 GitHub 上相关的 constrained RL / safe RL / "
            "Lyapunov-based 优化方法\n"
            "3. 结合下方失败库排除已知无效方向，选出最可能突破瓶颈的 1 个新方法\n"
            "4. 通过 code_innovation 通道给出具体代码修改指令（文件名+函数+内容）\n"
            "5. 修改必须与 SA-GAT PPO 框架兼容，保持维度不变\n"
        )
    else:
        task_instruction = (
            "## 任务指令\n"
            "当前处于【轻量调参模式】。\n"
            "1. 分析当前指标与终止条件的 gap\n"
            "2. 诊断主要瓶颈（delivery 不足？queue 过大？EE 不够？）\n"
            "3. 结合下方失败库，避开已知无效参数区域\n"
            "4. 通过 hyperparameter 通道给出具体参数调整（精确数值）\n"
        )

    prompt = "\n".join([
        "\n".join(priority_lines),
        "",
        task_instruction,
        "",
        "\n".join(gap_lines),
        "",
        "\n".join(trend_lines),
        "",
        knowledge_section,
        "",
        "\n".join(history_lines),
        "",
        "\n".join(code_lines),
        "",
        "\n".join(log_lines),
        "",
        "\n".join(constraint_lines),
        "",
        output_format,
    ])

    return prompt


def _build_knowledge_section(state: dict) -> str:
    """构建失败库/成功库注入段落，引导 LLM 避开已知无效方向"""
    lines = ["## 已验证无效的方向（禁止重复尝试）"]
    failed = state.get("failed_directions", [])
    successful = state.get("successful_directions", [])

    if failed:
        for f in failed[-10:]:
            params = f.get("params_tried", {})
            lines.append(
                f"- [{f.get('channel', '?')}] {f.get('direction', '')} "
                f"| 参数={params} | 结论: {f.get('verdict', '')}"
            )
        lines.append(
            "\n**硬性要求：以上方向已被验证为负向，禁止重复尝试或换个说法重试，"
            "必须探索结构上不同的新方向。**"
        )
    else:
        lines.append("- （暂无失败记录）")

    lines.append("")
    lines.append("## 已验证有效的方向（可在此基础上深化）")
    if successful:
        for s in successful[-5:]:
            lines.append(
                f"- [{s.get('channel', '?')}] {s.get('direction', '')} "
                f"| 参数={s.get('params_tried', {})} | 增益: {s.get('gain', '')}"
            )
    else:
        lines.append("- （暂无成功记录）")

    return "\n".join(lines)


def _read_code_snippets() -> list:
    """读取当前奖励函数和 Lagrangian 参数的关键代码"""
    snippets = []

    # 读取 config.py 中的关键参数
    config_path = PROJECT_ROOT / "src" / "config.py"
    if config_path.exists():
        content = config_path.read_text()
        snippets.append("### src/config.py (关键参数)")
        # 提取前100行或全部
        lines = content.split("\n")[:100]
        snippets.extend(lines)

    # 读取 lagrangian.py
    lag_path = PROJECT_ROOT / "src" / "training" / "lagrangian.py"
    if lag_path.exists():
        content = lag_path.read_text()
        snippets.append("\n### src/training/lagrangian.py (前80行)")
        lines = content.split("\n")[:80]
        snippets.extend(lines)

    return snippets


def _read_train_log_tail(state: dict, config: dict) -> list:
    """读取最近训练日志的尾部"""
    n_lines = config["cloud_agents"].get("train_log_tail_lines", 50)

    # 尝试找到最近的训练日志
    iter_num = state["iteration"]
    prefix = config["fast_training"]["exp_name_prefix"]
    log_path = PROJECT_ROOT / "logs" / f"{prefix}_{iter_num}_train.log"

    if not log_path.exists():
        # 尝试目录内的日志
        log_dir = PROJECT_ROOT / "logs" / f"{prefix}_{iter_num}"
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                log_path = log_files[0]

    if log_path.exists():
        lines = log_path.read_text().split("\n")
        return lines[-n_lines:]

    return ["[无可用训练日志]"]


def _rebuild_session(ca_client, config: dict) -> str:
    """重建一个全新 Cloud Agents Session（旧 session 损坏/All models failed 时）。

    归档旧 session，创建新 session 并写回 .ca_session_id。返回新 session_id，失败返回 ""。
    """
    ca = config.get("cloud_agents", {})
    coord_id = ca.get("coordinator_agent_id", "")
    env_id = ca.get("environment_id", "")
    if not (coord_id and env_id):
        return ""
    # 归档旧 session（尽力，不阻塞）
    try:
        if CA_SESSION_ID_FILE.exists():
            old = CA_SESSION_ID_FILE.read_text().strip()
            if old:
                try:
                    ca_client.archive_session(old)
                except Exception:
                    pass
    except Exception:
        pass
    try:
        new_sid = ca_client.create_session(
            coord_id, env_id, title="opt-loop-rebuilt",
            metadata={"purpose": "session-rebuild-after-failure"})
        CA_SESSION_ID_FILE.write_text(new_sid)
        print(f"[CA] 已重建新 Session: {new_sid}")
        return new_sid
    except Exception as e:
        print(f"[CA] 重建 Session 失败: {e}")
        return ""


def _wait_for_quota(ca_client, session_id: str, config: dict, max_wait_min: int = 120) -> bool:
    """额度耗尽时轮询等待恢复（指数退避，上限 max_wait_min 分钟）。

    返回 True 表示额度已恢复可继续；False 表示超时仍未恢复。
    额度是暂时性外部限制（共享信用额会随时间恢复），不应消耗优化方向名额。
    """
    import time as _t
    deadline = _t.time() + max_wait_min * 60
    wait_s = 60  # 首次等 1 分钟，翻倍至上限 15 分钟
    probe_prompt = "quota-probe: 回复OK"
    while _t.time() < deadline:
        print(f"[CA] 额度耗尽，等待 {wait_s}s 后重试（共享信用额会随时间恢复）...")
        _t.sleep(wait_s)
        try:
            ca_client.send_and_wait(session_id, probe_prompt, timeout=60)
            print("[CA] 额度已恢复，继续优化")
            return True
        except CloudAgentsQuotaError:
            wait_s = min(wait_s * 2, 900)
        except Exception as e:
            # 其他错误（如网络抖动）：也等一会儿再试
            print(f"[CA] 探测异常（非额度）: {e}")
            wait_s = min(wait_s * 2, 900)
    return False


def invoke_cloud_agent(
    state: dict, eval_result: dict, config: dict,
    ca_client: Optional[CloudAgentsClient] = None,
    session_id: str = "",
    dry_run: bool = False,
) -> dict:
    """通过 Cloud Agents API 调用 Coordinator Agent 进行多智能体分析决策。

    Coordinator 内部编排 Analyst/InnovationScout/CodeArchitect 子 Agent，
    返回结构化 JSON 决策。
    """
    mode = check_escalation(state, eval_result, config)
    state["mode"] = mode

    prompt = build_analysis_prompt(state, eval_result, config, mode)

    # 停滞时指示 Coordinator 委派 Innovation Scout
    stalled = state.get("stalled_count", 0)
    if stalled >= 1:
        prompt += (
            "\n\n## 特别指令\n"
            f"当前已连续 {stalled} 个方向停滞无突破。"
            "请务必委派 InnovationScout 子 Agent 使用 WebSearch 搜索 "
            "IEEE/arXiv/GitHub 上的 constrained RL / safe RL / multi-objective RL 最新方法，"
            "结合失败库排除已知无效方向，给出结构上不同的创新方案。"
        )

    print(f"[CA] 调用 Cloud Agents Coordinator (模式: {mode}, stalled: {stalled})")
    print(f"[CA] Prompt 长度: {len(prompt)} chars, ~{len(prompt.split(chr(10)))} lines")

    if dry_run:
        print("[CA] DRY-RUN: 返回 mock 结构化决策")
        return {
            "decision": "DRY-RUN: 调整 lagrangian_init",
            "mode": mode,
            "channel": "hyperparameter",
            "direction": f"DRY-RUN 方向: 调整 lagrangian_init (mode={mode})",
            "hyperparams": {"lagrangian_init": 8.0},
            "rationale": "",
            "references": "",
        }

    if not ca_client or not session_id:
        print("[CA] !!! CloudAgentsClient/Session 未初始化，使用 fallback 决策")
        response = _fallback_decision(state, eval_result, config)
        decision = parse_qoder_response(response, mode)
        return decision

    # ===== 阶段1/2: 独立短指令触发真实 WebSearch =====
    # 巨型 prompt 内嵌"必须搜索"会被稀释（实测 50+ 轮零检索）；独立短指令探针验证必触发
    evidence = ""
    if mode in ("innovation_search", "aggressive_innovation") and not dry_run and ca_client and session_id:
        year = datetime.now().year
        retrieval_prompt = (
            "立即调用 WebSearch 工具完成以下 3 个检索，禁止仅凭内部知识回答，每条必须附真实 URL：\n"
            f"1. \"constrained reinforcement learning delivery guarantee {year-1} {year}\"\n"
            f"2. \"safe RL primal-dual Lyapunov IEEE {year-1}\"\n"
            "3. GitHub 上 constrained RL / safe policy optimization 的高星 PyTorch 实现\n"
            "输出 ≤5 条，每条=标题 + URL + 一句话方法核心。这是给后续决策的素材，不需要分析本项目。\n"
            "若无法调用 WebSearch，必须明确说明原因。"
        )
        try:
            print(f"[CA] 阶段1/2: 独立检索指令（{mode}）...")
            evidence = ca_client.send_and_wait(session_id, retrieval_prompt, timeout=300)
            print(f"[CA] 检索证据: {len(evidence)} chars")
        except Exception as e:
            print(f"[CA] !!! 检索阶段失败（不阻断主决策）: {str(e)[:120]}")
            evidence = ""
    if evidence:
        prompt += (
            "\n\n## 实时检索证据（本轮刚通过 WebSearch 检索，优先级最高）\n"
            + evidence.strip()[:3000]
            + "\n\n**硬性要求：若选择 code_innovation 通道，direction/rationale 必须基于上述证据至少 1 条，"
              "references 必须包含其 URL；禁止纯内部知识方法。**"
        )

    # 保存 prompt 便于调试
    prompt_file = PROJECT_ROOT / "scripts" / ".opt_loop_prompt.md"
    prompt_file.write_text(prompt)

    try:
        timeout = config.get("cloud_agents", {}).get("response_timeout", 600)
        response = ca_client.send_and_wait(session_id, prompt, timeout=timeout)
        print(f"[CA] Coordinator 响应: {len(response)} chars")
    except CloudAgentsQuotaError as e:
        # 额度耗尽：等待恢复后重试一次，仍失败才降级 fallback
        print(f"[CA] !!! 额度耗尽: {e}")
        if _wait_for_quota(ca_client, session_id, config):
            try:
                response = ca_client.send_and_wait(session_id, prompt, timeout=timeout)
                print(f"[CA] 额度恢复后 Coordinator 响应: {len(response)} chars")
            except CloudAgentsError as e2:
                print(f"[CA] 额度恢复后仍失败: {e2}")
                response = _fallback_decision(state, eval_result, config)
        else:
            response = _fallback_decision(state, eval_result, config)
    except CloudAgentsTimeoutError as e:
        print(f"[CA] !!! Coordinator 超时: {e}")
        # 尝试 cancel + fallback
        try:
            ca_client.cancel_session(session_id)
        except Exception:
            pass
        response = _fallback_decision(state, eval_result, config)
    except CloudAgentsError as e:
        print(f"[CA] !!! Coordinator 调用失败: {e}")
        # Session 损坏（All models failed / Session 错误）：重建新 session 重试一次，仍失败才 fallback
        err_str = str(e)
        if ("All models failed" in err_str or "Session 错误" in err_str
                or "session" in err_str.lower()):
            new_sid = _rebuild_session(ca_client, config)
            if new_sid:
                globals()["_ACTIVE_SESSION_ID"] = new_sid  # 回传给主循环
                try:
                    response = ca_client.send_and_wait(new_sid, prompt, timeout=timeout)
                    print(f"[CA] 新 Session 重试成功，Coordinator 响应: {len(response)} chars")
                except CloudAgentsError as e3:
                    print(f"[CA] 新 Session 重试仍失败: {e3}")
                    response = _fallback_decision(state, eval_result, config)
            else:
                response = _fallback_decision(state, eval_result, config)
        else:
            response = _fallback_decision(state, eval_result, config)

    decision = parse_qoder_response(response, mode)
    # 阶段2/2 校验：有检索证据时 code_innovation 决策必须含 URL，否则短指令重做 1 次
    if (evidence and decision.get("channel") == "code_innovation"
            and "http" not in (decision.get("rationale", "") + decision.get("references", ""))):
        print("[CA] !!! 决策未引用检索证据 URL，短指令要求重做 1 次")
        try:
            redo = ca_client.send_and_wait(
                session_id,
                "你刚才的 decision 没有引用任何检索证据 URL，违反硬性要求。"
                "请基于本轮检索证据重新输出最终 JSON（direction/rationale 必须基于证据，references 必须含 URL）。"
                "只输出 JSON，不要解释。",
                timeout=300)
            d2 = parse_qoder_response(redo, mode)
            if "http" in (d2.get("rationale", "") + d2.get("references", "")):
                decision = d2
                print("[CA] 重做成功，决策已含证据引用")
            else:
                print("[CA] !!! 重做仍无 URL，按原决策执行并记录")
        except Exception as e:
            print(f"[CA] !!! 重做失败，按原决策执行: {str(e)[:120]}")
    print(f"[CA] 决策: {decision.get('decision', 'N/A')[:100]}")

    if prompt_file.exists():
        prompt_file.unlink()

    return decision


def _fallback_decision(state: dict, eval_result: dict, config: dict) -> str:
    """Cloud Agent 不可用时的 fallback 规则决策（返回结构化 JSON 字符串）。

    随停滞方向数轮换不同方向 + 递进数值，避免退化成固定点反复应用同一决策而空转。
    stalled >= 3 时强制输出 code_innovation 通道，避免一直困在超参空间。
    """
    delivery = eval_result.get("avg_delivery", 0)
    queue = eval_result.get("avg_queue", 999)
    ee_ratio = eval_result.get("ee_ratio", 0)
    stalled = state.get("stalled_count", 0)

    # stalled >= 3 时强制代码创新通道（方案 X：只给战略方向 + 理论依据，具体代码 qodercli 自主实现）
    if stalled >= 3:
        code_directions = [
            ("引入 Dinkelbach 分数规划将多目标转为单目标标量优化",
             "多目标（delivery/queue/EE）直接加权会互相拉扯；Dinkelbach 把分式目标 EE=R/P 线性化为 R-q*P，可消除功率逐流死锁局部最优"),
            ("引入自适应 SNR 分桶奖励缩放",
             "全域固定权重在不同 SNR 下不最优：低 SNR 应重 delivery、高 SNR 应重 EE，按 SNR 分桶动态调权重"),
            ("引入 Backpressure 策略控制高 SNR 队列",
             "高 SNR 过载时队列爆炸；Lyapunov backpressure 理论支持在过载时主动降服务速率上限、丢弃低优先级用户以稳定队列"),
            ("将 Actor-Critic 注意力机制改为多头注意力",
             "单头 Self-Attention 对跨用户干扰感知不足；多头注意力+LayerNorm+残差可提升表征能力（保持输入输出维度不变）"),
            ("引入 CVaR 风险感知约束替代期望约束",
             "期望约束对尾部低 SNR 不鲁棒；CVaR 优化 delivery 分布的 5% 分位数而非均值，提升最差情况表现"),
        ]
        direction, rationale = code_directions[stalled % len(code_directions)]
        return json.dumps({
            "channel": "code_innovation",
            "direction": direction,
            "hyperparams": {},
            "rationale": rationale,
            "references": "（fallback 无联网检索，由 qodercli 自主设计实现）",
        }, ensure_ascii=False)

    # stalled < 3: 超参通道（候选方向池，用 stalled 轮换）
    candidates = []
    if delivery < 0.95:
        candidates.append(("提升 Lagrangian 约束强度拉高 delivery",
                           {"lagrangian_init": 8.0 + 2.0 * stalled,
                            "delivery_threshold": 0.98}))
        candidates.append(("加大 Lagrangian 学习率加速约束收敛",
                           {"lagrangian_lr": 0.05 + 0.02 * stalled,
                            "lagrangian_max": 30.0}))
    if queue > 12:
        candidates.append(("增强队列漂移与溢出惩罚压低队列",
                           {"blend_alpha": 2.0 + 0.5 * stalled,
                            "w_overflow": 25.0 + 5.0 * stalled}))
    if ee_ratio < 1.05:
        candidates.append(("放宽自适应 V 上限提升能效",
                           {"V_max": 5.0 + stalled, "w_power_exceed": 8.0}))
    # 兜底通用方向
    candidates.append(("微调学习率与 warmup 稳定训练",
                       {"lr_actor": 3e-4, "lagrangian_warmup": 100 + 50 * stalled}))

    direction, hp = candidates[stalled % len(candidates)]
    return json.dumps({"channel": "hyperparameter",
                       "direction": direction, "hyperparams": hp},
                      ensure_ascii=False)


def _extract_json(text: str) -> Optional[dict]:
    """从文本中提取第一个 JSON 对象（支持 ```json 包裹或裸 JSON）"""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def parse_qoder_response(response: str, mode: str) -> dict:
    """解析 Qoder CLI 响应为结构化决策。

    期望 LLM 输出 JSON（可裹在 ```json 块中）：
      {"channel": "hyperparameter"|"code_innovation",
       "direction": "简短方向描述",
       "hyperparams": {...},               # 超参通道
       "rationale": "...", "references": "..."}  # 代码通道（战略）
    """
    decision = {
        "decision": (response or "无响应")[:500],
        "mode": mode,
        "channel": "hyperparameter",
        "direction": "",
        "hyperparams": {},
        "rationale": "",
        "references": "",
        "raw_response": response,
    }
    parsed = _extract_json(response)
    if parsed:
        decision["channel"] = parsed.get("channel", "hyperparameter")
        decision["direction"] = str(parsed.get("direction", ""))[:200]
        decision["hyperparams"] = parsed.get("hyperparams") or {}
        decision["rationale"] = str(parsed.get("rationale", ""))[:600]
        decision["references"] = str(parsed.get("references", ""))[:600]
        if decision["direction"]:
            decision["decision"] = decision["direction"]
    return decision


# ===========================================================================
# 决策执行
# ===========================================================================


def apply_decision(decision: dict, state: dict) -> None:
    """执行决策：超参通道存入 pending_params；代码通道记录待 edit 指令。

    代码通道的实际 edit + 冒烟测试由主循环调用 apply_code_innovation 执行。
    """
    channel = decision.get("channel", "hyperparameter")
    direction = decision.get("direction") or decision.get("decision", "")

    # 初始化当前方向（主循环在接受/回滚时重置）
    if state.get("current_direction") is None:
        state["current_direction"] = {
            "desc": direction,
            "channel": channel,
            "params": {},
            "rationale": decision.get("rationale", ""),
            "references": decision.get("references", ""),
            "rounds_tried": 0,
        }

    if channel == "hyperparameter":
        hp = {k: v for k, v in (decision.get("hyperparams") or {}).items()
              if k in HYPERPARAM_ARG_MAP}
        if hp:
            state["pending_params"].update(hp)
            state["current_direction"]["params"] = dict(state["pending_params"])
            print(f"[APPLY] 超参通道注入: {hp}")
        else:
            print("[APPLY] 超参通道：无有效超参（白名单外或空）")
    else:
        # 代码创新通道：方向已记录，具体代码由 qodercli 本地施工
        print("[APPLY] 代码通道方向已记录，主循环将调 qodercli 施工 + 护栏校验")


def run_smoke_test(checkpoint: Optional[str] = None) -> tuple:
    """代码改动后的冒烟测试：import 检查 + 1ep 试跑。

    若提供 checkpoint，一并验证 partial_resume 加载（维度兼容性）。
    Returns: (ok: bool, reason: str)
    """
    # 1) import 检查：核心模块能否导入
    import_check = subprocess.run(
        ["python3", "-c",
         "import sys; sys.path.insert(0,'.'); import importlib; "
         "importlib.import_module('src.config'); "
         "importlib.import_module('src.env.rsma_env'); "
         "importlib.import_module('src.training.lagrangian'); "
         "importlib.import_module('src.training.ppo'); "
         "importlib.import_module('src.training.main')"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120
    )
    if import_check.returncode != 0:
        return False, f"import 失败: {import_check.stderr[-300:]}"

    # 2) 1ep 试跑：train.py --smoke
    smoke_cmd = [
        "python3", "scripts/train.py", "--exp_name", "smoke_test",
        "--smoke", "--device", "cpu", "--csi_mode", "imperfect",
    ]
    if checkpoint and (PROJECT_ROOT / checkpoint).exists():
        smoke_cmd.extend(["--partial_resume", str(PROJECT_ROOT / checkpoint)])
    smoke = subprocess.run(
        smoke_cmd, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=600
    )
    if smoke.returncode != 0:
        return False, f"1ep 试跑失败: {smoke.stderr[-300:]}"

    return True, "冒烟测试通过"


def _git_env() -> dict:
    """git 提交身份（环境变量传递，不修改用户 git config）"""
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": "optimize-loop",
        "GIT_AUTHOR_EMAIL": "optimize-loop@localhost",
        "GIT_COMMITTER_NAME": "optimize-loop",
        "GIT_COMMITTER_EMAIL": "optimize-loop@localhost",
    })
    return env


def _git_has_commits() -> bool:
    r = subprocess.run(["git", "rev-parse", "HEAD"],
                       cwd=str(PROJECT_ROOT), capture_output=True, timeout=15)
    return r.returncode == 0


def ensure_git_baseline() -> None:
    """确保 git 仓库有初始 commit（回滚点）。无 commit 时 git checkout 回滚无效。"""
    if _git_has_commits():
        return
    print("[GIT] 仓库无 commit 历史，创建初始基线 commit（回滚点）...")
    subprocess.run(["git", "add", "-A"], cwd=str(PROJECT_ROOT),
                   capture_output=True, timeout=60, env=_git_env())
    r = subprocess.run(
        ["git", "commit", "-m", "opt-loop baseline: initial commit (rollback point)", "--allow-empty"],
        cwd=str(PROJECT_ROOT), capture_output=True, timeout=60, env=_git_env(), text=True)
    if r.returncode == 0:
        print("[GIT] 初始基线 commit 已创建")
    else:
        print(f"[GIT] !!! 初始 commit 失败: {r.stderr.strip()[:200]}")


def _snapshot_worktree() -> dict:
    """工作区未提交改动快照：{path: porcelain状态码}（含未跟踪文件）。"""
    r = subprocess.run(["git", "status", "--porcelain"],
                       cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=30)
    snap = {}
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ")[-1]
        snap[path] = line[:2]
    return snap


def _revert_to_pre_hash(pre_hash) -> None:
    """方向被拒绝后精确还原代码到施工前基线 commit：删除方向期间新增的文件 +
    恢复 src//scripts/ 下被改内容。不改写历史，还原结果由 Phase G 自动 commit。"""
    if not pre_hash or not _git_has_commits():
        _git_rollback_worktree()
        return
    # 1) 删除 pre_hash..HEAD 间新增的文件（方向施工产物）
    r = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=A", pre_hash, "HEAD", "--", "src/", "scripts/"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=30)
    added = [f.strip() for f in r.stdout.splitlines() if f.strip()]
    if added:
        subprocess.run(["git", "rm", "-q", "-f", "--"] + added,
                       cwd=str(PROJECT_ROOT), capture_output=True, timeout=30)
    # 2) 恢复两个代码树到 pre_hash 内容（index+worktree）
    r2 = subprocess.run(["git", "checkout", pre_hash, "--", "src/", "scripts/"],
                        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=30)
    if r2.returncode != 0:
        print(f"[GIT] !!! 还原到施工前基线失败: {r2.stderr.strip()[:200]}，退回宽回滚")
        _git_rollback_worktree()
        return
    print(f"[GIT] 已还原代码到施工前基线 {pre_hash[:8]}（含 scripts/ 与新增文件）")


def _git_rollback_worktree() -> None:
    """回滚工作区 src/ 下未提交的代码改动"""
    if not _git_has_commits():
        print("[GIT] 无 commit 历史，git checkout 回滚不可用（依赖内存备份还原）")
        return
    r = subprocess.run(["git", "checkout", "--", "src/"],
                       cwd=str(PROJECT_ROOT), capture_output=True, timeout=30, text=True)
    if r.returncode != 0:
        print(f"[GIT] 代码回滚失败: {r.stderr.strip()[:200]}")


def _resolve_qodercli_model(config: dict) -> tuple:
    """返回 (model, effort)：主力不可用时回退。effort 为 None 表示不传思考档。"""
    qc = config.get("qodercli", {})
    return qc.get("model", "Kimi-K2.7-Code"), qc.get("reasoning_effort", "high")


def apply_code_innovation(state: dict, config: dict, eval_result: dict,
                         ca_client: Optional[CloudAgentsClient] = None,
                         session_id: str = "",
                         dry_run: bool = False) -> tuple:
    """代码通道（方案 X）：qodercli 本地自主施工 + git diff 护栏 + 冒烟测试。

    Cloud Agent 只出战略（current_direction 的 desc/rationale/references），
    qodercli 拿工单在真实 repo 上自主读文件、设计、实现、自测自修。
    Returns: (ok: bool, reason: str)。
    """
    cd = state.get("current_direction") or {}
    direction = cd.get("desc", "")
    if not direction:
        return False, "无创新方向"

    if dry_run:
        print("[CODE] DRY-RUN: 跳过实际 qodercli 施工")
        return True, "dry-run"

    # 修改前内存备份（护栏/冒烟失败时还原的兜底）：
    # 白名单目录条目展开为其下全部 .py 文件；同时备份施工前已脏的跟踪文件，
    # 保证回滚能精确恢复到施工前状态（不波及用户既有未提交改动）
    guard = config.get("code_guard", {})
    mutable_files = guard.get("mutable_files", [])
    backup = {}
    for rel in mutable_files:
        p = PROJECT_ROOT / rel
        if p.is_dir():
            for f in sorted(p.rglob("*.py")):
                if f.is_file() and "__pycache__" not in f.parts:
                    backup[str(f.relative_to(PROJECT_ROOT))] = f.read_text()
        elif p.is_file():
            backup[rel] = p.read_text()
    before_snap = _snapshot_worktree()
    for path, st in before_snap.items():
        p = PROJECT_ROOT / path
        if st.strip() != "??" and p.is_file() and path not in backup:
            try:
                backup[path] = p.read_text()
            except (UnicodeDecodeError, OSError):
                pass  # 二进制等无法文本备份的文件跳过（极端情况由 git checkout 兜底）

    def _restore_from_backup():
        # 1) 还原所有备份文件内容（白名单内被改的 + 施工前已脏的）
        for rel, content in backup.items():
            (PROJECT_ROOT / rel).write_text(content)
        # 2) 施工期间新出现的改动：跟踪文件 git checkout 精确还原（含禁区文件），
        #    新增 .py 文件删除，白名单目录下的新增目录整体移除
        for path, st in _snapshot_worktree().items():
            if path in backup or path in before_snap:
                continue
            p = PROJECT_ROOT / path
            if st.strip() == "??":
                if p.is_file() and p.suffix == ".py":
                    p.unlink()
                elif p.is_dir() and any(path.startswith(m.rstrip("/")) for m in mutable_files):
                    shutil.rmtree(p, ignore_errors=True)
            else:
                subprocess.run(["git", "checkout", "--", path],
                               cwd=str(PROJECT_ROOT), capture_output=True, timeout=30, text=True)
        print("[CODE] 已从内存备份+git 精确还原所有施工改动")

    # 生成结构化工单，交给本地 qodercli 自主施工
    work_order = build_work_order(state, eval_result, config)
    model, effort = _resolve_qodercli_model(config)
    qc = config.get("qodercli", {})
    qoder_cmd = qc.get("command", "qodercli")
    timeout = qc.get("timeout", 900)

    cmd = [qoder_cmd, "-p", "-w", str(PROJECT_ROOT), "-m", model]
    if effort:
        cmd += ["--reasoning-effort", effort]
    cmd += ["--dangerously-skip-permissions", work_order]

    print(f"[CODE] qodercli 本地施工（model={model}, effort={effort}, timeout={timeout}s）...")
    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=timeout,
        )
        tail = (result.stdout or "")[-600:]
        print(f"[CODE] qodercli 施工回报（尾部）:\n{tail}")
    except subprocess.TimeoutExpired:
        print(f"[CODE] qodercli 施工超时（{timeout}s），还原改动")
        _restore_from_backup()
        return False, f"qodercli 施工超时（{timeout}s）"
    except Exception as e:
        print(f"[CODE] qodercli 施工异常: {e}")
        _restore_from_backup()
        return False, f"qodercli 施工异常: {e}"

    # 护栏校验：施工前后快照差集硬检查（白名单 + forbidden + rsma_env 物理段）
    guard_ok, guard_reason = _verify_code_guard(config, before_snap)
    if not guard_ok:
        print(f"[CODE] !!! 安全护栏拦截: {guard_reason}\uff0c回滚所有改动")
        _restore_from_backup()
        return False, f"护栏拦截: {guard_reason}"

    # 不可变路径校验：本轮实际改动文件命中不可变清单（环境物理模型、评估脚本等）
    # 即中止本轮。独立于白名单，在接受与提交之前执行（零容忍）
    after_snap_imm = _snapshot_worktree()
    changed_imm = [p for p in after_snap_imm if not (before_snap and p in before_snap)]
    violations = find_immutable_violations(changed_imm, guard.get("immutable_paths", []))
    if violations:
        rej_reason = f"改动命中不可变路径: {', '.join(violations)}"
        print(f"[CODE] !!! {rej_reason}，中止本轮并回滚所有改动")
        _restore_from_backup()
        # 拒绝原因写入状态文件（随主循环 save_state 持久化）
        state.setdefault("immutable_rejections", []).append({
            "iter": state.get("iteration"),
            "direction": direction[:200],
            "violated_files": violations,
            "reason": rej_reason,
            "time": datetime.now().isoformat(),
        })
        return False, f"IMMUTABLE: {rej_reason}"

    # 最后一道保险：主循环冒烟测试（import + 1ep + partial_resume）
    ok, reason = run_smoke_test(state.get("best_checkpoint") or state.get("current_checkpoint"))
    if not ok:
        print(f"[CODE] 冒烟测试失败，回滚代码改动: {reason}")
        _restore_from_backup()
        return False, f"冒烟测试失败: {reason}"

    return True, f"qodercli 施工成功，{guard_reason}，冒烟测试通过"


def _py_compile_ok(path: Path) -> bool:
    """写入后语法验证"""
    r = subprocess.run(["python3", "-m", "py_compile", str(path)],
                       capture_output=True, timeout=30)
    return r.returncode == 0


def build_work_order(state: dict, eval_result: dict, config: dict) -> str:
    """生成给 qodercli 的结构化创新工单（方案 X 决策1）。

    工单是完整的施工 prompt，让 qodercli 自主读真实文件、设计并实现代码、
    自测自修。含 6 部分：创新方向 / 瓶颈证据 / 失败库 / 可改边界 / 验收标准 / 自主施工要求。
    """
    cd = state.get("current_direction") or {}
    term = config["termination"]
    guard = config.get("code_guard", {})
    qc = config.get("qodercli", {})
    repair_rounds = qc.get("self_repair_rounds", 3)

    # 1) 创新方向（战略层，来自 Cloud Agent，不含代码）
    direction = cd.get("desc", "")
    rationale = cd.get("rationale", "")
    references = cd.get("references", "")

    # 2) 瓶颈证据（per-SNR + gap）
    per_snr = eval_result.get("per_snr", {})
    snr_rows = []
    for snr in term.get("snr_points", []):
        d = per_snr.get(snr, {})
        ppo = d.get("ppo", {})
        sdma = d.get("sdma", {})
        if ppo and sdma:
            snr_rows.append(
                f"  SNR={snr}dB: PPO(deliv={ppo.get('delivery',0):.3f} q={ppo.get('queue',999):.1f}) "
                f"vs SDMA(deliv={sdma.get('delivery',0):.3f} q={sdma.get('queue',999):.1f})"
            )
    bottleneck = "\n".join(snr_rows) if snr_rows else "  (无 per-SNR 数据)"

    # 3) 失败库（避免重复）
    failed = state.get("failed_directions", [])
    fail_lines = []
    seen = set()
    for f in failed[-12:]:
        d = str(f.get("direction", ""))[:70]
        if d and d not in seen:
            seen.add(d)
            fail_lines.append(f"  - {d} | 结论: {str(f.get('verdict',''))[:50]}")
    fail_section = "\n".join(fail_lines) if fail_lines else "  (暂无)"

    # 4) 可改边界
    mutable = guard.get("mutable_files", [])
    forbidden = guard.get("forbidden_files", [])

    work_order = f"""你是资深强化学习工程师，负责在真实代码库上实现一个通信+DRL 的算法创新。

# 自主执行铁律（无人值守循环，没有人类在线，违反=本轮直接判负）
- 禁止提问或等待确认（如"你同意 X 吗？""选 A 还是 B？"）——不会有人回答你，提问=失败
- 遇到方案分支时，由你自己权衡并直接采用你的推荐方案（优先改动最小的 MVP 路径）实施到底
- 必须产生实际代码改动（git status 中有白名单内文件变更）才算完成；只分析/只出方案不动手 = 失败

# 项目
MIMO-RSMA 上行能效优化，分层 PPO（SA-GAT Worker + Meta-Controller）。
工作目录就是当前 repo，你可以直接用 Read/Edit/Bash 工具读改真实文件。

# 本轮创新方向（战略，由分析团队给出）
方向: {direction}
理论依据: {rationale}
参考方法/文献: {references}

# 当前瓶颈证据（逐 SNR 点，PPO-RSMA vs SDMA 基线）
{bottleneck}
全域: delivery={eval_result.get('avg_delivery',0):.4f} queue={eval_result.get('avg_queue',999):.2f} EE_ratio={eval_result.get('ee_ratio',0):.4f}
目标: 逐点 delivery≥SDMA×{term.get('delivery_sdma_ratio',1.0)} 且 queue≤SDMA×{term.get('queue_sdma_ratio',1.1)}，mean(EE)>SDMA×{term.get('ee_advantage_min',1.05)}

# 已验证无效的方向（禁止重复，换思路）
{fail_section}

# 你的任务（分步跑通原则，重要）
1. 先用 Read/Grep 读懂相关真实文件的当前实现（不要猜，一定要读真实代码）
2. 【分步实现】先实现该创新方向的**最小可用版本**（MVP）：用最少的代码改动把核心机制接入并跑通 smoke，
   不要一次性写大量复杂代码。先保证能跑通，再在此基础上适度扩展。
3. 每改完一步立即自测：运行 `python3 -m py_compile <改动的文件>` 验证语法，
   再运行 `python3 scripts/train.py --smoke --device cpu --csi_mode imperfect` 验证 1ep 能跑通
4. 若自测失败，自己诊断并修复，最多重试 {repair_rounds} 轮；若方向过于复杂时间不够，
   宁可只交付能跑通的 MVP 版本，也不要交付跑不通的半成品

# 硬约束（越界会被自动回滚并判失败）
- 只能修改这些文件/目录: {', '.join(mutable)}（目录前缀下允许新增文件）
- 绝对禁止修改: {', '.join(forbidden)}（信道/速率物理模型，改了实验不可比）
- rsma_env.py 允许改奖励/观测增广/动作解析（保持维度不变），禁止改动 self.rate_calc / self.channel 物理模型调用
- 绝对禁止修改 scripts/evaluate_snr_sweep.py（评估脚本属不可变清单，触碰=本轮直接拒绝并回滚）
- 必须保持模型输入/输出维度不变（否则 checkpoint 无法加载）
- 动作空间 4K=96 维不可改

# 完成后报告
最后清楚说明：改了哪些文件、每个文件改了什么、自测（py_compile + smoke）结果。
"""
    return work_order


def find_immutable_violations(changed_files: list, immutable_paths: list) -> list:
    """不可变路径校验（纯函数，独立可单测）。

    返回 changed_files 中命中不可变清单的文件子集。
    匹配规则：精确匹配，或位于不可变目录条目（可带尾随 '/'）之下。
    """
    violations = []
    for f in changed_files:
        for ip in immutable_paths:
            ip_norm = str(ip).rstrip("/")
            if f == ip_norm or f.startswith(ip_norm + "/"):
                violations.append(f)
                break
    return violations


def _verify_code_guard(config: dict, before_snap: dict = None) -> tuple:
    """qodercli 施工后的安全护栏硬校验（方案 X 决策5，快照差集版）。

    三重护栏：文件级白名单 + forbidden 零容忍 + rsma_env.py 物理段语义检查。
    Returns: (ok: bool, reason: str)。ok=False 时调用方负责回滚。
    """
    guard = config.get("code_guard", {})
    mutable = guard.get("mutable_files", [])
    forbidden = guard.get("forbidden_files", [])
    markers = guard.get("rsma_env_physical_markers", [])

    # 只校验【施工期间新出现】的改动：施工前已存在的未提交改动（如既有未跟踪目录）
    # 与 qodercli 无关，不能误判拦截；新增文件也会出现在快照差集中，绕过漏洞仍堵死
    after_snap = _snapshot_worktree()
    changed = [p for p in after_snap if not (before_snap and p in before_snap)]
    if not changed:
        return False, "qodercli 未产生任何文件改动"

    print(f"[GUARD] 改动文件: {changed}")

    # 护栏1: forbidden 零容忍
    for f in changed:
        if any(f == fb or f.startswith(fb) for fb in forbidden):
            return False, f"越界修改禁区文件: {f}"

    # 护栏2: 文件级白名单（改动必须全在白名单内）
    for f in changed:
        if not any(f == m or f.startswith(m) for m in mutable):
            return False, f"修改了白名单外文件: {f}"

    # 护栏3: rsma_env.py 物理段语义检查
    if "src/env/rsma_env.py" in changed:
        dr = subprocess.run(
            ["git", "diff", "HEAD", "--", "src/env/rsma_env.py"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=30,
        )
        # 只看新增/删除行（+/-，排除 diff 头 +++/---）
        changed_lines = [
            ln for ln in dr.stdout.splitlines()
            if (ln.startswith("+") or ln.startswith("-"))
            and not ln.startswith("+++") and not ln.startswith("---")
        ]
        for ln in changed_lines:
            for mk in markers:
                if mk in ln:
                    return False, f"改动触及 rsma_env.py 物理段（含 '{mk}'）: {ln.strip()[:80]}"

    return True, f"护栏通过，改动文件: {changed}"


def _apply_instruction_locally(instruction: str) -> None:
    """[已废弃] 隔空转录时代的本地文本解析，方案 X 下 qodercli 直接改真实文件，不再需要。"""
    raise NotImplementedError("方案 X 下不再使用本地文本解析，qodercli 直接施工真实文件")


# ===========================================================================
# 快速测试门禁（训练前置信号层）
# ===========================================================================


def run_fast_test_gate(config: dict, dry_run: bool = False) -> tuple:
    """快速测试子集门禁：训练前运行标记为 fast 的轻量用例。

    失败则本轮直接判失败、跳过训练，避免在坏代码上浪费训练算力。
    失败输出可定位到文件与用例（pytest 的 FAILED 短摘要行）。
    Returns: (ok: bool, detail: dict)，detail 含 duration_sec / failed_tests / output_tail。
    """
    gate_cfg = config.get("fast_gate", {})
    if not gate_cfg.get("enabled", True):
        print("[FAST-GATE] 已禁用 (fast_gate.enabled=false)，跳过")
        return True, {"skipped": True}

    marker = gate_cfg.get("marker", "fast")
    timeout_sec = int(gate_cfg.get("timeout_seconds", 120))
    test_paths = [str(p) for p in gate_cfg.get("test_paths", ["tests/"])]
    cmd = ["python3", "-m", "pytest", "-m", marker, "-q",
           "--tb=short", "--no-header"] + test_paths
    print(f"[FAST-GATE] 运行快速测试子集（超时 {timeout_sec}s）...")

    start = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=timeout_sec,
        )
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        rc = result.returncode
    except subprocess.TimeoutExpired as e:
        dur = time.time() - start
        tail = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        print(f"[FAST-GATE] !!! 超时（>{timeout_sec}s），门禁判失败")
        return False, {
            "duration_sec": dur,
            "failed_tests": [],
            "output_tail": f"TIMEOUT after {timeout_sec}s\n{tail[-800:]}",
        }
    dur = time.time() - start

    # 失败定位：pytest 短摘要行 "FAILED tests/xxx.py::Class::test_x"
    failed_tests = re.findall(r"^FAILED\s+(\S+)", output, flags=re.MULTILINE)

    if rc == 0:
        print(f"[FAST-GATE] PASS ({dur:.1f}s)")
        return True, {"duration_sec": dur, "failed_tests": [],
                      "output_tail": output[-300:].strip()}

    if rc == 5:
        # 未收集到任何 fast 用例：标记配置错误，门禁不可信，判失败
        print(f"[FAST-GATE] !!! 未收集到任何 '{marker}' 标记用例，门禁判失败")
        return False, {"duration_sec": dur, "failed_tests": [],
                       "output_tail": f"NO TESTS COLLECTED for marker '{marker}'"}

    print(f"[FAST-GATE] !!! FAIL ({dur:.1f}s): {len(failed_tests)} 个用例未通过")
    for t in failed_tests[:10]:
        print(f"[FAST-GATE]   - {t}")
    return False, {
        "duration_sec": dur,
        "failed_tests": failed_tests,
        "output_tail": output[-800:].strip(),
    }


# ===========================================================================
# Git 版本管理
# ===========================================================================


def commit_gate_check(config: dict) -> tuple:
    """提交前置门：语法检查 + 快速测试子集，任一失败即拦截本轮提交。

    Returns: (ok: bool, reason: str)
    """
    gate = config.get("git", {}).get("commit_gate", {})
    if not gate.get("enabled", False):
        return True, "commit_gate 未启用"

    timeout = gate.get("timeout", 300)

    # 1) 语法检查：对指定目录整体编译（不产生任何副作用文件以外的变更）
    syntax_dirs = gate.get("syntax_dirs", ["src", "scripts"])
    try:
        r = subprocess.run(
            ["python3", "-m", "compileall", "-q"] + syntax_dirs,
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return False, f"语法检查超时（>{timeout}s）"
    if r.returncode != 0:
        return False, f"语法检查失败: {((r.stderr or '') + (r.stdout or '')).strip()[-300:]}"

    # 2) 快速测试子集（-x 首失败即停，控制拦截耗时）
    tests = gate.get("tests", [])
    if tests:
        try:
            r = subprocess.run(
                ["python3", "-m", "pytest", "-q", "-x"] + tests,
                cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return False, f"快速测试子集超时（>{timeout}s）"
        if r.returncode != 0:
            tail = (r.stdout or "").strip()[-300:] or (r.stderr or "").strip()[-300:]
            return False, f"快速测试子集失败: {tail}"

    return True, "提交前置门通过"


def _record_commit_gate_rejection(state: dict, reason: str) -> None:
    """前置门拦截原因写入状态文件（随主循环 save_state 持久化）"""
    state.setdefault("commit_gate_rejections", []).append({
        "iter": state.get("iteration"),
        "reason": reason,
        "time": datetime.now().isoformat(timespec="seconds"),
    })


def git_commit_iteration(state: dict, eval_result: dict, config: dict) -> None:
    """每轮结束后自动 Git commit"""
    if not config["git"]["auto_commit"]:
        return

    # 提交前置门：语法检查 + 快速测试子集，任一失败则跳过本轮提交并记录原因
    gate_ok, gate_reason = commit_gate_check(config)
    if not gate_ok:
        _record_commit_gate_rejection(state, gate_reason)
        print(f"[GIT] !!! 提交前置门未通过，跳过本轮 commit: {gate_reason[:200]}")
        return

    prefix = config["git"]["commit_prefix"]
    iter_num = state["iteration"]
    delivery = eval_result.get("avg_delivery", 0)
    queue = eval_result.get("avg_queue", 999)
    ee_ratio = eval_result.get("ee_ratio", 0)
    mode = state["mode"]

    msg = (
        f"{prefix} iter-{iter_num}: "
        f"delivery={delivery:.4f} queue={queue:.1f} "
        f"EE_ratio={ee_ratio:.4f} [{mode}]"
    )

    try:
        r1 = subprocess.run(
            ["git", "add", "-A"],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=60, env=_git_env(), text=True
        )
        r2 = subprocess.run(
            ["git", "commit", "-m", msg, "--allow-empty"],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=60, env=_git_env(), text=True
        )
        if r2.returncode == 0:
            print(f"[GIT] Committed: {msg}")
        else:
            print(f"[GIT] !!! Commit 失败: {r2.stderr.strip()[:200]}")
    except Exception as e:
        print(f"[GIT] Commit 异常: {e}")


# ===========================================================================
# 改善判断
# ===========================================================================


def compute_score(eval_result: dict, config: dict) -> list:
    """计算字典序评分 [达标点数, 平均delivery, -平均queue, ee_ratio]

    达标 = delivery ≥ SDMA_delivery×delivery_sdma_ratio 且 queue ≤ SDMA_queue×queue_sdma_ratio
    （逐 SNR 点，相对基线；高 SNR 下绝对值目标物理不可达）。
    四元组按字典序比较，达标点数绝对优先，能效 ee_ratio 垫底。
    """
    term = config["termination"]
    d_ratio = term.get("delivery_sdma_ratio", 1.0)
    q_ratio = term.get("queue_sdma_ratio", 1.1)

    per_snr = eval_result.get("per_snr", {})
    pass_count = 0
    for _snr, data in per_snr.items():
        ppo = data.get("ppo", {})
        sdma = data.get("sdma", {})
        sdma_deliv = sdma.get("delivery", 0)
        sdma_queue = sdma.get("queue", 999)
        if (ppo.get("delivery", 0) >= sdma_deliv * d_ratio
                and ppo.get("queue", 999) <= sdma_queue * q_ratio):
            pass_count += 1

    avg_delivery = eval_result.get("avg_delivery", 0)
    avg_queue = eval_result.get("avg_queue", 999)
    ee_ratio = eval_result.get("ee_ratio", 0)

    return [pass_count, round(avg_delivery, 4), round(-avg_queue, 4), round(ee_ratio, 4)]


def is_better(new_score: list, best_score: Optional[list]) -> bool:
    """字典序比较：new_score 是否严格优于 best_score（None 视为最差）"""
    if best_score is None:
        return True
    return list(new_score) > list(best_score)


def save_best_checkpoint(current_ckpt: str) -> Optional[str]:
    """将当前 checkpoint 复制到 best 目录，返回 best 相对路径"""
    src = PROJECT_ROOT / current_ckpt
    if not src.exists():
        return None
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, BEST_CKPT)
    return str(BEST_CKPT.relative_to(PROJECT_ROOT))


def get_git_hash() -> Optional[str]:
    """获取当前 git HEAD commit hash"""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


# ===========================================================================
# 报告生成 (Task 7)
# ===========================================================================


def generate_final_report(state: dict, config: dict) -> None:
    """生成最终优化报告"""
    report_path = PROJECT_ROOT / config["state"]["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# SA-GAT PPO 自动优化循环 — 最终报告",
        "",
        f"**生成时间**: {datetime.now().isoformat()}",
        f"**总迭代次数**: {state['iteration']}",
        f"**终止原因**: {state.get('termination_reason', '成功达到终止条件')}",
        f"**运行模式**: {state['mode']}",
        "",
        "## 最终指标",
        "",
        f"| 指标 | 最佳值 | 目标 |",
        f"|------|--------|------|",
        f"| Delivery | {state['best_metrics']['delivery']:.4f} | ≥ SDMA×{config['termination'].get('delivery_sdma_ratio', 1.0)} |",
        f"| Queue | {state['best_metrics']['queue']:.2f} | ≤ SDMA×{config['termination'].get('queue_sdma_ratio', 1.1)} |",
        f"| EE Ratio | {state['best_metrics']['ee_ratio']:.4f} | >= 1.05 |",
        "",
        "## 迭代历史",
        "",
        "| Iter | Delivery | Queue | EE Ratio | Mode | Decision |",
        "|------|----------|-------|----------|------|----------|",
    ]

    for h in state["history"]:
        m = h.get("metrics", {})
        lines.append(
            f"| {h['iter']} | {m.get('delivery', 0):.4f} | "
            f"{m.get('queue', 999):.2f} | {m.get('ee_ratio', 0):.4f} | "
            f"{h.get('mode', 'N/A')} | {h.get('decision', 'N/A')[:50]} |"
        )

    if state["innovations_applied"]:
        lines.extend([
            "",
            "## 引入的创新方法",
            "",
        ])
        for innov in state["innovations_applied"]:
            lines.append(f"- **iter-{innov.get('iter', '?')}**: {innov.get('description', 'N/A')}")

    lines.extend([
        "",
        "## 知识库统计",
        "",
        f"- best score: {state.get('best_score')}",
        f"- 停滞方向数: {state.get('stalled_count', 0)}",
        f"- 失败方向库: {len(state.get('failed_directions', []))} 条",
        f"- 成功方向库: {len(state.get('successful_directions', []))} 条",
        "",
        "## 最佳 Checkpoint",
        "",
        f"`{state.get('best_checkpoint') or state.get('current_checkpoint', 'N/A')}`",
        "",
        "---",
        f"*报告由 optimize_loop.py 自动生成*",
    ])

    report_path.write_text("\n".join(lines))
    print(f"[REPORT] 报告已生成: {report_path}")


# ===========================================================================
# 主循环
# ===========================================================================


def _record_failure(state: dict, verdict: str) -> None:
    """将当前方向记入失败库"""
    cd = state.get("current_direction") or {}
    state["failed_directions"].append({
        "iter": state["iteration"],
        "channel": cd.get("channel", "?"),
        "direction": cd.get("desc", ""),
        "params_tried": cd.get("params", {}),
        "verdict": verdict,
        "rounds_tried": cd.get("rounds_tried", 0),
    })


def _record_success(state: dict, gain: str) -> None:
    """将当前方向记入成功库"""
    cd = state.get("current_direction") or {}
    state["successful_directions"].append({
        "iter": state["iteration"],
        "channel": cd.get("channel", "?"),
        "direction": cd.get("desc", ""),
        "params_tried": cd.get("params", {}),
        "gain": gain,
    })


def main_loop(dry_run: bool = False) -> None:
    """主优化循环"""
    config = load_config()
    state = load_or_init_state(config)

    if state["terminated"]:
        print("[LOOP] 上次运行已终止，如需重启请删除 state 文件")
        print_status(state)
        return

    # ===== Git 基线：确保有 commit 作为回滚点（否则代码通道回滚无效） =====
    if not dry_run:
        ensure_git_baseline()

    # ===== Cloud Agents 初始化 =====
    ca_client = None
    session_id = ""
    if not dry_run:
        try:
            ca_config = config.get("cloud_agents", {})
            ca_client = CloudAgentsClient()
            # 复用已有 Session 或创建新 Session
            session_id = ""
            if CA_SESSION_ID_FILE.exists():
                old_sid = CA_SESSION_ID_FILE.read_text().strip()
                try:
                    status = ca_client.get_session_status(old_sid)
                    if status == "idle":
                        session_id = old_sid
                        print(f"[CA] 复用 Session: {session_id}")
                except Exception:
                    print(f"[CA] 旧 Session {old_sid} 已失效，创建新 Session...")
            if not session_id:
                coord_id = ca_config.get("coordinator_agent_id", "")
                env_id = ca_config.get("environment_id", "")
                if coord_id and env_id:
                    session_id = ca_client.create_session(
                        coord_id, env_id,
                        title=f"opt-loop-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        metadata={"purpose": "rsma-ee-optimize-loop"}
                    )
                    CA_SESSION_ID_FILE.write_text(session_id)
                    print(f"[CA] 新 Session 创建: {session_id}")
                else:
                    print("[CA] !!! config 中缺少 coordinator_agent_id 或 environment_id")
                    ca_client = None
        except CloudAgentsAuthError as e:
            print(f"[CA] !!! 认证失败: {e}")
            ca_client = None
        except Exception as e:
            print(f"[CA] !!! 初始化失败: {e}")
            ca_client = None

    print("=" * 60)
    print("SA-GAT PPO 自动优化循环 — 启动 (Cloud Agents 多智能体)")
    print(f"  起始 checkpoint: {state['current_checkpoint']}")
    print(f"  Session: {session_id or 'dry-run/fallback'}")
    print(f"  终止条件: 逐点 delivery≥SDMA×{config['termination'].get('delivery_sdma_ratio', 1.0)}, "
          f"queue≤SDMA×{config['termination'].get('queue_sdma_ratio', 1.1)}, EE>SDMA*1.05")
    print(f"  安全边界: {config['safety']['max_iterations']}轮 / "
          f"{config['safety']['max_runtime_hours']}h")
    print("=" * 60)

    # 注册信号处理（优雅退出）
    def signal_handler(sig, frame):
        print("\n[LOOP] 收到中断信号，保存状态并退出...")
        state["termination_reason"] = "用户手动中断"
        state["terminated"] = True
        save_state(state)
        generate_final_report(state, config)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while True:
        # 每轮迭代重读 config：护栏/安全边界/qodercli 等配置修改即时生效，无需重启循环
        config = load_config()

        # 检查 stop flag
        if STOP_FLAG.exists():
            print("[LOOP] 检测到 stop flag，优雅退出")
            STOP_FLAG.unlink()
            state["termination_reason"] = "stop flag 触发"
            state["terminated"] = True
            break

        # 检查安全边界
        safety_reason = check_safety_bounds(state, config)
        if safety_reason:
            print(f"[LOOP] 安全边界触发: {safety_reason}")
            state["termination_reason"] = safety_reason
            state["terminated"] = True
            break

        # 迭代计数
        state["iteration"] += 1
        iter_num = state["iteration"]
        print(f"\n{'='*60}")
        print(f"[LOOP] === 迭代 {iter_num} 开始 ===")
        print(f"{'='*60}")

        # ============================================================
        # Phase A: 决策 —— Cloud Agent 出战略（方向/超参/是否代码创新）
        # ============================================================
        # 基准轮：尚未建立真实 best（best_score is None）时，跳过 LLM 决策，
        # 直接用当前基线训练+评估建立真实基准。否则 LLM 拿到哨兵值 999/0
        # 会误判为“评估管线坏了”，指挥 qodercli 去改框架代码→护栏拦截→死循环。
        is_baseline_round = state.get("best_score") is None
        eval_for_decision = state.get("last_eval") or {}
        if is_baseline_round:
            print("[LOOP] 基准轮：尚无真实 best，跳过 LLM 决策，直接训练+评估建立基准")
            state["current_direction"] = None
            state["pending_params"] = {}
        elif state.get("current_direction") is None:
            # 网络瞬断容错：DNS/连接等外部异常不杀循环，退避重试后跳过本轮（不消耗 stalled）
            decision = None
            for attempt in range(5):
                try:
                    decision = invoke_cloud_agent(state, eval_for_decision, config,
                                                  ca_client=ca_client, session_id=session_id,
                                                  dry_run=dry_run)
                    break
                except Exception as e:
                    wait = 60 * (attempt + 1)
                    print(f"[LOOP] !!! Coordinator 调用异常（第 {attempt+1}/5 次）: "
                          f"{str(e)[:150]}，{wait}s 后重试")
                    time.sleep(wait)
            if decision is None:
                print("[LOOP] Coordinator 连续 5 次调用失败，保存状态跳过本轮（不计停滞）")
                save_state(state)
                continue
            # invoke_cloud_agent 重建了 session 时同步新 session_id
            global _ACTIVE_SESSION_ID
            if _ACTIVE_SESSION_ID and _ACTIVE_SESSION_ID != session_id:
                session_id = _ACTIVE_SESSION_ID
                print(f"[LOOP] Session 已切换为重建的: {session_id}")
                _ACTIVE_SESSION_ID = ""
            apply_decision(decision, state)
        else:
            print(f"[LOOP] 沿用当前方向再验证一轮（不重新决策）")

        cd = state.get("current_direction") or {}
        channel = cd.get("channel", "hyperparameter")

        # ============================================================
        # Phase B: 施工 —— 代码通道走 qodercli；超参通道注入 pending_params
        # ============================================================
        code_applied = False
        if channel == "code_innovation" and cd.get("desc") and cd.get("applied"):
            # 方向代码已落地并 commit（上轮施工完成、训练/评估未完成），直接训练验证，不重复施工
            code_applied = True
            print("[LOOP] 方向代码已落地（已 commit），跳过重复施工，直接训练验证")
        elif channel == "code_innovation" and cd.get("desc"):
            cd["pre_hash"] = get_git_hash()  # 施工前基线 hash，方向被拒绝时精确还原用
            ok, reason = apply_code_innovation(
                state, config, eval_for_decision,
                ca_client=ca_client, session_id=session_id, dry_run=dry_run)
            if ok:
                code_applied = True
                state["innovations_applied"].append({
                    "iter": iter_num,
                    "description": cd.get("desc", "")[:100],
                })
                print(f"[LOOP] 代码创新已落地: {reason}")
                # 落地即 commit：施工成果立即进入 git，防止进程中断后改动悬空
                # （悬空改动会污染后续迭代的快照差集 → 后续施工被误判零改动回滚）
                if not dry_run and config["git"].get("auto_commit", False):
                    # 提交前置门：语法检查 + 快速测试子集，任一失败则跳过本次提交并记录原因
                    gate_ok, gate_reason = commit_gate_check(config)
                    if not gate_ok:
                        _record_commit_gate_rejection(state, gate_reason)
                        print(f"[GIT] !!! 提交前置门未通过，跳过落地 commit: {gate_reason[:200]}")
                    else:
                        try:
                            subprocess.run(["git", "add", "-A"], cwd=str(PROJECT_ROOT),
                                           capture_output=True, timeout=60, env=_git_env(), text=True)
                            cmsg = (f"{config['git']['commit_prefix']} iter-{iter_num}: "
                                    f"施工落地待验证 {cd.get('desc', '')[:60]}")
                            rc = subprocess.run(["git", "commit", "-m", cmsg],
                                                cwd=str(PROJECT_ROOT), capture_output=True,
                                                timeout=60, env=_git_env(), text=True)
                            if rc.returncode == 0:
                                print(f"[GIT] 落地代码已 commit（待训练验证）")
                            else:
                                print(f"[GIT] !!! 落地 commit 失败: {rc.stderr.strip()[:200]}")
                        except Exception as e:
                            print(f"[GIT] 落地 commit 异常: {e}")
                cd["applied"] = True
                save_state(state)
            else:
                print(f"[LOOP] 代码创新未通过: {reason}")
                # 额度问题是暂时性外部错误，保留方向下轮重试，不消耗 stalled
                if str(reason).startswith("QUOTA:"):
                    print("[LOOP] 额度问题导致，保留当前方向下轮重试（不计入停滞）")
                    save_state(state)
                    continue
                else:
                    _record_failure(state, reason)
                    state["stalled_count"] += 1
                    state["current_direction"] = None
                    save_state(state)
                    save_knowledge(state)
                    continue  # 无真变化，不训练，直接换方向

        # ============================================================
        # Phase C: 变化门禁 —— 无真变化则跳过训练，直接换方向（根治原地打转）
        # 基准轮豁免：它就是要“无变化地训练一次建立基准”
        # ============================================================
        has_param_change = bool(state.get("pending_params"))
        has_real_change = code_applied or has_param_change
        if not has_real_change and not is_baseline_round:
            print("[LOOP] 本轮无真变化（代码未落地且无超参变化），跳过训练，直接换方向")
            _record_failure(state, "决策未产生可落地的代码或超参变化")
            state["stalled_count"] += 1
            state["current_direction"] = None
            save_state(state)
            save_knowledge(state)
            continue

        # ============================================================
        # Phase C.5: 快速测试门禁 —— 训练前置信号层
        # 运行标记为 fast 的轻量用例（约 2 分钟内）；失败则本轮直接判失败，
        # 不进入训练（避免在坏代码上浪费训练算力）
        # ============================================================
        gate_ok, gate_detail = run_fast_test_gate(config, dry_run=dry_run)
        if not gate_ok:
            failed_ids = gate_detail.get("failed_tests", [])
            if failed_ids:
                verdict = f"快速测试门禁失败 ({len(failed_ids)} 用例): " + "; ".join(failed_ids[:5])
            else:
                verdict = f"快速测试门禁失败: {gate_detail.get('output_tail', '')[:200]}"
            print(f"[LOOP] !!! 本轮判失败: {verdict}，跳过训练")
            # 代码通道涉案：精确还原到施工前基线，不留坏代码；
            # 无 pre_hash（本轮无施工）则仅记失败，不动工作区
            if (code_applied or cd.get("applied")) and cd.get("pre_hash"):
                _revert_to_pre_hash(cd["pre_hash"])
                cd["applied"] = False
                print("[LOOP] 代码通道: 已回滚到施工前基线")
            elif not dry_run:
                print("[LOOP] !!! 本轮无施工记录但快速测试失败: "
                      "工作区可能存在遗留坏代码，请人工检查")
            _record_failure(state, verdict)
            state["failed_directions"].append({
                "iter": iter_num,
                "channel": channel,
                "direction": cd.get("desc", ""),
                "params_tried": cd.get("params", {}),
                "verdict": verdict[:200],
                "rounds_tried": 1,
            })
            # 代码事故（非方向失败）：不消耗 stalled，直接换方向
            state["current_direction"] = None
            state["pending_params"] = {}
            save_state(state)
            save_knowledge(state)
            continue

        # ============================================================
        # Phase D: 训练 —— 始终从 best 出发探索（消除链式退化）
        # ============================================================
        train_start = state.get("best_checkpoint") or config["baseline"]["checkpoint"]
        if train_start and Path(PROJECT_ROOT / train_start).exists():
            state["current_checkpoint"] = train_start
            print(f"[LOOP] 训练起点: {train_start}")
        train_result = run_fast_training(state, config, dry_run=dry_run)
        if train_result.get("error"):
            err_text = train_result["error"]
            print(f"[LOOP] 训练失败，跳过本轮: {err_text[:100]}")
            # 语法/导入错误 = 代码事故（非方向失败）：立即回滚代码 + 记失败库，不消耗 stalled
            if any(k in err_text for k in ("SyntaxError", "IndentationError", "ImportError", "ModuleNotFoundError")):
                print("[LOOP] !!! 检测到代码语法/导入错误，立即回滚 src/ 代码改动 !!!")
                _git_rollback_worktree()
                state["failed_directions"].append({
                    "iter": iter_num,
                    "channel": channel,
                    "direction": cd.get("desc", ""),
                    "params_tried": cd.get("params", {}),
                    "verdict": f"代码事故：{err_text.strip().splitlines()[-1][:120] if err_text.strip() else '语法错误'}，已自动回滚",
                    "rounds_tried": 1,
                })
                state["current_direction"] = None
                save_knowledge(state)
            else:
                state["stalled_count"] += 1
            save_state(state)
            continue

        if train_result.get("checkpoint"):
            state["current_checkpoint"] = train_result["checkpoint"]

        # ============================================================
        # Phase E: 评估
        # ============================================================
        eval_result = run_snr_sweep_eval(
            state["current_checkpoint"], config, dry_run=dry_run
        )
        if eval_result.get("error"):
            print(f"[LOOP] 评估失败，跳过本轮")
            state["stalled_count"] += 1
            save_state(state)
            continue
        state["last_eval"] = eval_result  # 供下一轮决策使用

        # 检查终止条件
        if check_termination(eval_result, config):
            print("[LOOP] *** 终止条件满足！优化成功！***")
            state["termination_reason"] = "成功达到终止条件"
            state["terminated"] = True
            save_best_checkpoint(state["current_checkpoint"])
            state["history"].append({
                "iter": iter_num,
                "metrics": {
                    "delivery": eval_result.get("avg_delivery", 0),
                    "queue": eval_result.get("avg_queue", 999),
                    "ee_ratio": eval_result.get("ee_ratio", 0),
                },
                "decision": "TERMINATED: 成功",
                "accepted": True,
                "mode": state["mode"],
            })
            break

        # ============================================================
        # Phase F: 字典序判定接受/拒绝/回滚
        # ============================================================
        new_score = compute_score(eval_result, config)
        accepted = is_better(new_score, state.get("best_score"))
        tolerance = config.get("rollback", {}).get("tolerance_rounds", 2)

        if accepted:
            old_best = state.get("best_score")
            state["best_score"] = new_score
            saved = save_best_checkpoint(state["current_checkpoint"])
            if saved:
                state["best_checkpoint"] = saved
            state["best_params"] = dict(state.get("pending_params", {}))
            state["stalled_count"] = 0
            if state.get("current_direction"):
                _record_success(state, f"score {old_best} -> {new_score}")
            # 接受：立即 commit 代码改动（确保 qodercli 的 Sauté/HAPPO 等创新真正进入 git，
            # 防止后续回滚/重启时代码丢失导致 checkpoint 无法加载）
            if not dry_run and config["git"].get("auto_commit", False):
                # 提交前置门：语法检查 + 快速测试子集，任一失败则跳过本次提交并记录原因
                gate_ok, gate_reason = commit_gate_check(config)
                if not gate_ok:
                    _record_commit_gate_rejection(state, gate_reason)
                    print(f"[GIT] !!! 提交前置门未通过，跳过接受方向立即 commit: {gate_reason[:200]}")
                else:
                    try:
                        r1 = subprocess.run(
                            ["git", "add", "-A"],
                            cwd=str(PROJECT_ROOT), capture_output=True, timeout=60, env=_git_env(), text=True
                        )
                        prefix = config["git"]["commit_prefix"]
                        msg = (
                            f"{prefix} iter-{iter_num}: "
                            f"delivery={eval_result.get('avg_delivery', 0):.4f} "
                            f"queue={eval_result.get('avg_queue', 999):.1f} "
                            f"EE_ratio={eval_result.get('ee_ratio', 0):.4f} [{state['mode']}]"
                        )
                        r2 = subprocess.run(
                            ["git", "commit", "-m", msg],
                            cwd=str(PROJECT_ROOT), capture_output=True, timeout=60, env=_git_env(), text=True
                        )
                        if r2.returncode == 0:
                            state["best_git_hash"] = get_git_hash()
                            print(f"[GIT] 接受方向已立即 commit: {msg}")
                        elif "nothing to commit" in ((r2.stdout or "") + (r2.stderr or "")):
                            # 代码已在落地时 commit，无新改动属正常
                            state["best_git_hash"] = get_git_hash()
                            print(f"[GIT] 接受方向：代码已于落地时 commit，hash={state['best_git_hash'][:8]}")
                        else:
                            print(f"[GIT] !!! 接受方向 commit 失败: {r2.stderr.strip()[:200]}")
                    except Exception as e:
                        print(f"[GIT] 接受方向 commit 异常: {e}")
            state["best_metrics"] = {
                "delivery": eval_result.get("avg_delivery", 0),
                "queue": eval_result.get("avg_queue", 999),
                "ee_ratio": eval_result.get("ee_ratio", 0),
                "power": eval_result.get("avg_power", 999),
            }
            # 接受：代码改动已随 best checkpoint 固化，清空方向和超参
            state["current_direction"] = None
            state["pending_params"] = {}
            print(f"[LOOP] 接受: score={new_score}，更新 best checkpoint")
        else:
            cd2 = state.get("current_direction")
            if cd2 is not None:
                cd2["rounds_tried"] = cd2.get("rounds_tried", 0) + 1
                if cd2["rounds_tried"] >= tolerance:
                    verdict = (f"score {new_score} 未超过 best "
                               f"{state.get('best_score')}，{tolerance} 轮验证失败")
                    _record_failure(state, verdict)
                    # 代码通道：精确还原到施工前基线（含 scripts/ 与新增文件，不留方向残骸）
                    if cd2.get("channel") == "code_innovation":
                        _revert_to_pre_hash(cd2.get("pre_hash"))
                    state["current_checkpoint"] = (
                        state.get("best_checkpoint") or config["baseline"]["checkpoint"])
                    state["pending_params"] = dict(state.get("best_params", {}))
                    state["stalled_count"] += 1
                    state["current_direction"] = None
                    print(f"[LOOP] 方向失败，回滚到 best + 记入失败库 "
                          f"(停滞方向数={state['stalled_count']})")
                else:
                    print(f"[LOOP] 方向未突破，第 {cd2['rounds_tried']}/{tolerance} 轮验证，继续")
            else:
                state["stalled_count"] += 1

        # 记录历史
        state["history"].append({
            "iter": iter_num,
            "metrics": {
                "delivery": eval_result.get("avg_delivery", 0),
                "queue": eval_result.get("avg_queue", 999),
                "ee_ratio": eval_result.get("ee_ratio", 0),
            },
            "score": new_score,
            "accepted": accepted,
            "decision": (cd or {}).get("desc", "")[:200],
            "mode": state["mode"],
        })

        # Phase G: Git commit + 保存状态 + 知识库
        if not dry_run:
            git_commit_iteration(state, eval_result, config)
        save_state(state)
        save_knowledge(state)

        # Phase G.5: 上下文压缩检测（长 Session 超阈值时压缩）
        if ca_client and session_id and not dry_run:
            try:
                threshold = config.get("cloud_agents", {}).get("context_compress_threshold", 800000)
                ctx_tokens = ca_client.estimate_context_tokens(session_id)
                if ctx_tokens > threshold:
                    print(f"[CA] 上下文 {ctx_tokens}K tokens > {threshold}K 阈值，执行压缩...")
                    ca_client.send_compress_instruction(session_id)
                    ca_client.wait_for_response(session_id, timeout=300)
                    print("[CA] 上下文压缩完成")
            except Exception as e:
                print(f"[CA] 上下文压缩失败（非致命）: {e}")

        print(f"[LOOP] === 迭代 {iter_num} 完成 === "
              f"接受: {'YES' if accepted else 'NO'} | "
              f"停滞方向数: {state['stalled_count']}")

    # 循环结束
    save_state(state)
    generate_final_report(state, config)
    print(f"\n[LOOP] 优化循环结束: {state['termination_reason']}")


# ===========================================================================
# CLI 入口
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SA-GAT PPO 自动优化循环编排器"
    )
    parser.add_argument("--status", action="store_true", help="查看当前状态")
    parser.add_argument("--stop", action="store_true", help="优雅退出")
    parser.add_argument("--dry-run", action="store_true", help="Mock 模式验证流程")
    parser.add_argument("--reset", action="store_true", help="重置状态（谨慎使用）")
    args = parser.parse_args()

    if args.status:
        if STATE_PATH.exists():
            with open(STATE_PATH, "r") as f:
                state = json.load(f)
            print_status(state)
        else:
            print("尚未启动优化循环（无状态文件）")
        return

    if args.stop:
        STOP_FLAG.touch()
        print("[STOP] 已设置 stop flag，循环将在当前迭代完成后退出")
        return

    if args.reset:
        if STATE_PATH.exists():
            STATE_PATH.unlink()
            print("[RESET] 状态文件已删除")
        return

    # 启动主循环
    main_loop(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
