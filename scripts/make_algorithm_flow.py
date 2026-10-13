"""Concise flowchart of the proposed algorithm (one PNG, no embedded title).

Offline: solver expert -> BC -> SFTP PPO fine-tune. Online: meta (slow) ->
worker SA-GAT (fast) -> environment, with the DGLCR reward feedback arrow and
the Lagrangian / lexicographic-projection constraint box.

Usage:
  python scripts/make_algorithm_flow.py [--out docs/paper/figs/algorithm_flowchart.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OFF_FC, OFF_EC = "#DEEBF7", "#2E75B6"
ON_FC, ON_EC = "#E2EFDA", "#548235"
ENV_FC, ENV_EC = "#FCE4D6", "#C55A11"
CON_FC, CON_EC = "#FFF2CC", "#BF8F00"


def add_box(ax, cx, cy, w, h, text, fc, ec, fs=9.5):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.4,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color="#1a1a1a", linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, ls="-", ec="#404040"):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=ec, lw=1.4, linestyle=ls, shrinkA=0, shrinkB=0),
    )


def build(out: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    ax.set_xlim(0.8, 11.7)
    ax.set_ylim(3.9, 13.5)
    ax.axis("off")

    add_box(
        ax, 4.6, 12.4, 6.2, 1.5,
        "求解器专家（训练自由、模型基上界）\n可微 MMSE-SIC 影子引擎\n逐槽 Adam 精炼功率 logits + 解码序候选",
        OFF_FC, OFF_EC,
    )
    add_box(
        ax, 4.6, 10.4, 6.2, 1.5,
        "蒸馏训练\n行为克隆 BC（8k 演示）→ PPO 微调\nSFTP 步冻 + guard 逐点选点",
        OFF_FC, OFF_EC,
    )
    add_box(ax, 4.6, 8.6, 6.2, 1.1, "Meta 控制器 · 慢时间尺度（每 20 槽）\n用户分组 g + 组级流分裂 α", ON_FC, ON_EC)
    add_box(
        ax, 4.6, 6.9, 6.2, 1.1,
        "Worker SA-GAT · 快时间尺度（每槽）\n功率份额 p · 排泄比 d · Plackett-Luce SIC 序 π",
        ON_FC, ON_EC,
    )
    add_box(
        ax, 4.6, 5.0, 6.2, 1.5,
        "环境\nRSMA MMSE-SIC 速率 · 队列更新\n逐 SNR 点交付率 / 队列硬门",
        ENV_FC, ENV_EC,
    )
    add_box(ax, 10.0, 10.4, 2.5, 1.1, "Lagrangian 对偶 λ\n+ 词典序梯度投影", CON_FC, CON_EC, fs=9)

    arrow(ax, 4.6, 11.65, 4.6, 11.15)
    ax.text(4.75, 11.4, "演示 (状态, 动作, 序)", fontsize=8, va="center", color="#404040")
    arrow(ax, 4.6, 9.65, 4.6, 9.15)
    ax.text(4.75, 9.4, "蒸馏出的分层策略", fontsize=8, va="center", color="#404040")
    arrow(ax, 4.6, 8.05, 4.6, 7.45)
    ax.text(4.75, 7.75, "g, α（保持 20 槽）", fontsize=8, va="center", color="#404040")
    arrow(ax, 4.6, 6.35, 4.6, 5.75)
    ax.text(4.75, 6.05, "逐槽动作", fontsize=8, va="center", color="#404040")

    # reward / state feedback (environment -> worker)
    arrow(ax, 8.0, 5.0, 8.0, 6.9, ec=ENV_EC)
    ax.text(
        8.15, 5.95,
        "状态 + DGLCR 奖励\n(Dinkelbach EE + max-weight\n+ CVaR 交付虚拟队列)",
        fontsize=8, va="center", color=ENV_EC,
    )
    # constraint stack into the training update
    arrow(ax, 8.75, 10.4, 7.7, 10.4, ls="--", ec=CON_EC)
    ax.text(8.25, 10.62, "硬约束保障", fontsize=8, va="bottom", ha="center", color=CON_EC)

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/paper/figs/algorithm_flowchart.png")
    args = ap.parse_args(argv)
    build(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
