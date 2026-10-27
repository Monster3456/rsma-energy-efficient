"""Shapley 可解释性分析的 PPT 用中文图表（Worker 层 + Meta 层）。

设计约束（PPT 口径）：一张图只讲一件事、大字号、每图不超过 2 个面板、
关键条带直接标数值。详细 7 面板版仍在 ``plot_interp_zh.py``。

三张主图：
  fig5_10_svea_ppt      Worker 全局：队列是唯一支配特征
  fig5_11_worker_ppt    Worker 局部：功率决策的驱动块
  fig5_12_meta_ppt      Meta 层：拆分比与分组的驱动通道

数据来源：logs/interpretability/summary.json（SVEA + KernelSHAP/EG，
Worker 13 块 / Meta 12 通道，4 种参照）。

用法（仓库根目录，venv python）：
  .venv/Scripts/python.exe scripts/plot_shapley_zh.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "RSMA-papers" / "thesis" / "figures"
SUMMARY = ROOT / "logs" / "interpretability" / "summary.json"
FONT = ["Microsoft YaHei", "SimHei", "SimSun"]

# Okabe-Ito 核心，validate_palette.js 全项 PASS
C_BLUE, C_ORANGE, C_GREEN = "#0072B2", "#D55E00", "#009E73"
C_GRAY = "#7A7A7A"
SNR_LIST = [0, 5, 10, 15, 20, 25, 30]

ZH_WORKER = {
    "h_gains": "信道增益", "queue": "队列", "delta_queue": "队列增量",
    "arrivals": "到达量", "power_history": "功率历史", "rate_history": "速率历史",
    "lambda_bucket": "λ乘子", "dvq_backlog": "交付欠账", "split_mask_alpha": "拆分比α",
    "group_ids": "分组", "dinkelbach_eta": "η比值", "qos_lambda": "QoS负载",
    "qos_priority": "QoS权",
    # 局部归因的玩家 = 13 原生块 + 2 全局（src/analysis/features.SHAP_PLAYER_NAMES）
    "global_snr": "全局SNR", "global_eps": "全局CSI误差",
}
# Meta 输入 12 通道（src/env/rsma_env.get_user_features 顺序）
ZH_META = [
    "信道增益", "队列比率", "上次拆分", "平均速率", "溢出频率", "功率利用率",
    "组ID", "SNR", "系统容量", "QoS类别", "QoS负载", "QoS优先级",
]


def _rc(ppt: bool = True):
    plt.rcParams["font.sans-serif"] = FONT
    plt.rcParams["axes.unicode_minus"] = False
    base = {"pdf.fonttype": 42, "ps.fonttype": 42}
    if ppt:
        base.update({"font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11,
                     "axes.linewidth": 1.0, "grid.linewidth": 0.6, "grid.alpha": 0.3})
    plt.rcParams.update(base)


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png -> {OUT_DIR}")


def _load():
    with open(SUMMARY, encoding="utf-8") as fh:
        return json.load(fh)


# ==================================================================== #
# 图 1：Worker 全局 —— 队列是唯一支配特征
# ==================================================================== #
def fig_svea(S) -> None:
    svea = S["svea"]
    names = [ZH_WORKER[n] for n in svea["feature_names"]]
    n = len(names)
    # 跨 SNR 平均 e_j + 跨 SNR 极差
    E = np.array([svea[str(s)]["sv"] for s in SNR_LIST])   # (7, n)
    mean = E.mean(axis=0)
    # 支配频次：几个 SNR 点上 e_j < 0
    neg_cnt = (E < 0).sum(axis=0)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.5, 5.6), width_ratios=[1.45, 1.0], gridspec_kw={"wspace": 0.30})

    # --- 左：13 块的跨 SNR 平均 e_j ---
    order = np.argsort(mean)              # 升序，负的在左
    ys = np.arange(n)
    colors = [C_ORANGE if mean[j] < 0 else C_BLUE for j in order]
    ax1.barh(ys, mean[order], color=colors, height=0.62, zorder=3)
    for y, j in zip(ys, order):
        v = mean[j]
        ax1.text(v + (0.012 if v >= 0 else -0.012), y, f"{v:+.3f}",
                 va="center", ha="left" if v >= 0 else "right",
                 fontsize=10, color=C_GRAY, zorder=4)
    ax1.axvline(0.0, color="k", lw=1.2, zorder=2)
    ax1.set_yticks(ys, [names[j] for j in order], fontsize=11.5)
    ax1.set_xlabel("SVEA 值 $e_j$（7 点平均）")
    ax1.set_title("(a) 只有队列的 $e_j$ 为负", loc="left", fontsize=15)
    ax1.set_xlim(min(mean) - 0.09, max(mean) + 0.09)
    ax1.grid(axis="x")
    ax1.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=C_ORANGE, edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_BLUE, edgecolor="none"),
    ], labels=["支配（$e_j<0$）", "非支配"], frameon=False,
        loc="lower right", fontsize=11)

    # --- 右：队列的 e_j 随 SNR ---
    q = names.index("队列")
    ax2.plot(SNR_LIST, E[:, q], color=C_ORANGE, marker="o", ms=9, lw=2.6, zorder=4)
    ax2.axhline(0.0, color="k", lw=1.2, ls="--", zorder=2)
    for s, v in zip(SNR_LIST, E[:, q]):
        ax2.annotate(f"{v:+.3f}", (s, v), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=10, color=C_GRAY)
    ax2.fill_between(SNR_LIST, E[:, q], 0, where=E[:, q] < 0, color=C_ORANGE, alpha=0.18)
    ax2.set_xticks(SNR_LIST)
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("队列块的 $e_j$")
    ax2.set_title("(b) 0--25 dB 均为支配，30 dB 转正", loc="left", fontsize=15)
    ax2.set_ylim(min(E[:, q]) - 0.05, max(E[:, q]) + 0.06)
    ax2.grid()

    _save(fig, "fig5_10_svea_ppt")
    print("    支配频次(7点中e_j<0的点数):",
          {names[j]: int(neg_cnt[j]) for j in range(n) if neg_cnt[j] > 0})


# ==================================================================== #
# 图 2：Worker 局部 —— 功率决策的驱动块
# ==================================================================== #
def fig_worker(S) -> None:
    """KernelSHAP 对总功率 P_total 的块归因，跨 SNR × 参照聚合。"""
    ker = S["kernel_shap"]
    # 局部归因玩家 = 13 原生块 + global_snr + global_eps（共 15）
    player_names = list(S["svea"]["feature_names"]) + ["global_snr", "global_eps"]
    names = [ZH_WORKER.get(n, n) for n in player_names]
    n = len(names)
    refs = ["boot", "means", "neutral"]   # 排除全零（已知最不可靠）
    A = np.zeros((len(SNR_LIST) * len(refs), n))
    k = 0
    for s in SNR_LIST:
        for r in refs:
            A[k] = np.array(ker[str(s)]["P_total"][r])
            k += 1
    # 只报强度，不报符号：归因符号随参照构造翻转（bootstrap 下队列为正、
    # 边际均值下为负），均值符号无意义。PPT 也只需讲"哪个块在驱动"。
    mag = np.abs(A).mean(axis=0)

    order = np.argsort(-mag)[:8]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ys = np.arange(len(order))
    ax.barh(ys, mag[order], color=C_BLUE, height=0.6, zorder=3)
    for y, j in zip(ys, order):
        ax.text(mag[j] + mag[order].max() * 0.02, y, f"{mag[j]:.3f}",
                va="center", fontsize=10.5, color=C_GRAY)
    ax.set_yticks(ys, [names[j] for j in order], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("KernelSHAP 归因强度 $|\\phi|$（跨 SNR 与参照平均）")
    ax.set_title("总功率决策的主导块：队列与信道增益", loc="left", fontsize=15)
    ax.text(0.99, 0.03, "只报强度；符号随参照构造翻转，不作结论",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color=C_GRAY)
    ax.set_xlim(0, mag[order].max() * 1.18)
    ax.grid(axis="x")
    _save(fig, "fig5_11_worker_ppt")


# ==================================================================== #
# 图 3：Meta 层 —— 拆分比与分组的驱动通道
# ==================================================================== #
def fig_meta(S) -> None:
    """EG + KernelSHAP 对 Meta 两个输出的通道归因（12 通道）。"""
    targets = [("alpha_avg", "组级拆分比 $\\alpha$"),
               ("group_logp_0", "用户分组 logits")]
    refs = ["boot", "means", "neutral"]
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4), gridspec_kw={"wspace": 0.34})

    for ax, (tgt, title) in zip(axes, targets):
        # 合并 EG 与 KernelSHAP，跨 SNR × 参照取 |归因| 平均
        A = []
        for s in SNR_LIST:
            for r in refs:
                A.append(np.abs(np.array(S["kernel_shap"][str(s)][tgt][r])))
                A.append(np.abs(np.array(S["eg"][str(s)][tgt][r])))
        mag = np.mean(A, axis=0)
        order = np.argsort(-mag)[:8]
        ys = np.arange(len(order))
        ax.barh(ys, mag[order], color=C_GREEN, height=0.6, zorder=3)
        for y, j in zip(ys, order):
            ax.text(mag[j] + mag[order].max() * 0.02, y, f"{mag[j]:.3f}",
                    va="center", fontsize=10.5, color=C_GRAY)
        ax.set_yticks(ys, [ZH_META[j] for j in order], fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel("归因强度 $|\\phi|$（EG+KernelSHAP 平均）")
        ax.set_title(f"{title} 的主导通道", loc="left", fontsize=15)
        ax.set_xlim(0, mag[order].max() * 1.18)
        ax.grid(axis="x")
    _save(fig, "fig5_12_meta_ppt")


def main() -> int:
    S = _load()
    _rc(ppt=True)
    fig_svea(S)
    fig_worker(S)
    fig_meta(S)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
