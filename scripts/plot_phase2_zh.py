# -*- coding: utf-8 -*-
"""第 5 章仿真结果图的中文重绘（无图题，标题由 LaTeX caption 承担）。

复刻英文图的同一数据与版式，仅做三点处理:
  1) 图内不调用 set_title / suptitle（"不要有图题"）；多面板靠左上角带括号
     小节标注 (a)(b)... 与 LaTeX caption 一一对应;
  2) 坐标轴、图例、面板全部中文化;
  3) 输出到 RSMA-papers/thesis/figures/fig5_*.png, 覆盖英文源图。

数据来源(官方口径):
  - fig5_1 训练稳定: logs/<run>/checkpoint_meta.jsonl (update/mean_reward)
  - fig5_2 主基准:   logs/_eval_bench_all.txt + 种子复核报告 (seedband 6 面板)
  - fig5_3 谱系:     _eval_m4_bc_init / _eval_m5_A_bestguard / _eval_mAdjud_S43..44
  - fig5_4 SVEA:     logs/interpretability/summary.json (svea)
  - fig5_5 归因重叠: logs/interpretability/summary.json (validation.jaccard_refs)

用法(仓库根目录, venv python):
  .venv/Scripts/python.exe scripts/plot_phase2_zh.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from plot_benchmarks import (  # noqa: E402
    METRIC_LABEL,
    ALGO_ORDER,
    STYLE,
    parse_overview,
)

OUT_DIR = ROOT / "RSMA-papers" / "thesis" / "figures"
FONT = ["Microsoft YaHei", "SimHei", "SimSun"]

# ------------------------------------------------------------------ #
# 中文渲染统一设置（无图题）
# ------------------------------------------------------------------ #
def _setup_rc() -> None:
    plt.rcParams["font.sans-serif"] = FONT
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ------------------------------------------------------------------ #
# fig5_1 训练稳定性
# ------------------------------------------------------------------ #
SEEDS = ["m5_armA_sftp", "mAdjudS43", "mAdjudS44", "mAdjudS45", "mAdjudS46"]
ARMS = {
    "m51_armA2_lr1e4": "释放学习率(×3.3)",
    "m51_armA3_nokldecay": "移除KL衰减",
    "m52_armD1_klbc": "KL→BC信任域",
    "m52_armD2_anchor": "强锚定",
    "m53_armE1_hm030": "硬例目标0.92/0.93",
    "m53_armE2_hm030w": "硬例目标0.95/0.95",
}
LDP = "b_ldptrlq_style"


def _load_curve(run: str) -> tuple[list[float], list[float]]:
    path = ROOT / "logs" / run / "checkpoint_meta.jsonl"
    xs, ys = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        rec = json.loads(line)
        xs.append(rec["update"])
        ys.append(rec["mean_reward"])
    return xs, ys


def fig_training(out: Path) -> None:
    curves = [_load_curve(r) for r in SEEDS]
    xs = curves[0][0]
    grid = np.array([np.interp(xs, cx, cy) for cx, cy in curves])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.fill_between(xs, grid.min(0), grid.max(0), color="#d62728", alpha=0.15,
                    label="PPO-RSMA(种子42–46,min–max)")
    ax.plot(xs, grid.mean(0), color="#d62728", lw=2.2, label="PPO-RSMA(5种子均值)")
    for run, label in ARMS.items():
        try:
            cx, cy = _load_curve(run)
        except FileNotFoundError:
            continue
        ax.plot(cx, cy, lw=1.0, alpha=0.75, label=label)
    try:
        cx, cy = _load_curve(LDP)
        ax.plot(cx, cy, lw=1.4, ls="--", color="#8c564b", label="LDPTRLQ")
    except FileNotFoundError:
        pass
    ax.set_xlabel("PPO 更新步")
    ax.set_ylabel("rollout 平均奖励")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, out, "fig5_1_training")


# ------------------------------------------------------------------ #
# fig5_2 主基准（seedband 六面板）+ fig5_3 谱系
# ------------------------------------------------------------------ #
# 面板: (metric_key, ylabel, 是否越低越好, 括号标号)
PANELS = [
    ("delivery_ratio", "交付率", False, "(a)"),
    ("mean_queue", "平均队列积压", True, "(b)"),
    ("total_power", "平均发射功率 (W)", True, "(c)"),
    ("ee", "能效  R_sum/P", False, "(d)"),
    ("overflow_rate", "溢出率", True, "(e)"),
    ("R_sum", "和速率 (bps/Hz)", False, "(f)"),
]
ZH_LABEL = {
    "PPO-RSMA": "PPO-RSMA(本文)",
    "SDMA": "SDMA",
    "NOMA": "NOMA",
    "EqualPower": "等功率分配",
    "LDPTRLQ": "LDPTRLQ",
}
SEED_BAND_REPORTS = [
    ("seed42", "_eval_m5_A_bestguard.txt"),
    ("seed43", "_eval_mAdjud_S43_bestguard.txt"),
    ("seed44", "_eval_mAdjud_S44_bestguard.txt"),
    ("seed45", "_eval_mAdjud_S45_bestguard.txt"),
    ("seed46", "_eval_mAdjud_S46_bestguard.txt"),
]


def _member(tag: str, snrs: list[float], data: dict) -> tuple[str, list[float], dict]:
    algo = "PPO-RSMA" if "PPO-RSMA" in data else next(iter(data))
    return tag, snrs, data[algo]


def fig_benchmarks(out: Path) -> None:
    # 基线 + 冠军逐点
    snrs_b, data_b = parse_overview(ROOT / "logs" / "_eval_bench_all.txt")
    # 合并各 PPO 种子报告为冠军种子带成员
    members: list[tuple[str, list[float], dict]] = []
    for tag, fn in SEED_BAND_REPORTS:
        p = ROOT / "logs" / fn
        if not p.exists():
            continue
        s, d = parse_overview(p)
        members.append(_member(tag, s, d))
    baselines = [a for a in ("SDMA", "NOMA", "EqualPower") if a in data_b]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    for ax, (key, ylbl, _low, tag) in zip(axes.ravel(), PANELS):
        for algo in baselines:
            vals = data_b[algo].get(key, [])
            xs = [s for s, v in zip(snrs_b, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            if xs:
                ax.plot(xs, ys, label=ZH_LABEL.get(algo, algo), **STYLE[algo])
        # 冠军种子带
        columns = [
            [m[2].get(key, [])[i] if i < len(m[2].get(key, [])) else None
             for m in members] for i in range(len(snrs_b))
        ]
        mean = [float(np.nanmean([v for v in c if v is not None])) if any(v is not None for v in c) else None
                for c in columns]
        lo = [min(v for v in c if v is not None) if any(v is not None for v in c) else None for c in columns]
        hi = [max(v for v in c if v is not None) if any(v is not None for v in c) else None for c in columns]
        xs = [s for s, m in zip(snrs_b, mean) if m is not None]
        ax.plot(xs, [m for m in mean if m is not None],
                label="PPO-RSMA(5种子均值)", **STYLE["PPO-RSMA"])
        ax.fill_between(
            snrs_b,
            [v if v is not None else float("nan") for v in lo],
            [v if v is not None else float("nan") for v in hi],
            color="#d62728", alpha=0.15, zorder=1, label="5种子min–max带",
        )
        ax.text(0.02, 0.96, tag, transform=ax.transAxes, fontsize=11,
                va="top", ha="left", fontweight="bold")
        ax.set_ylabel(ylbl)
        ax.set_xlabel("SNR (dB)")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(snrs_b)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, out, "fig5_2_benchmarks")


def fig_lineage(out: Path) -> None:
    specs = [
        ("纯BC", "_eval_m4_bc_init.txt"),
        ("SFTP微调(冠军)", "_eval_m5_A_bestguard.txt"),
        ("种子43复现", "_eval_mAdjud_S43_bestguard.txt"),
        ("种子44复现", "_eval_mAdjud_S44_bestguard.txt"),
    ]
    snrs_b, data_b = parse_overview(ROOT / "logs" / "_eval_bench_all.txt")
    sdma = (snrs_b, data_b["SDMA"].get("delivery_ratio", []))

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    cmap = plt.get_cmap("tab10")
    for i, (tag, fn) in enumerate(specs):
        p = ROOT / "logs" / fn
        if not p.exists():
            continue
        snrs, data = parse_overview(p)
        algo = "PPO-RSMA" if "PPO-RSMA" in data else next(iter(data))
        vals = data[algo].get("delivery_ratio", [])
        xs = [s for s, v in zip(snrs, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, label=tag, marker="o", lw=1.9, ms=6,
                color=cmap((i + 1) % 10))
    xs = [s for s, v in zip(sdma[0], sdma[1]) if v is not None]
    ys = [v for v in sdma[1] if v is not None]
    ax.plot(xs, ys, label="SDMA(参考)", marker="s", ls="--", lw=2.6, ms=8,
            color="#1f77b4", zorder=6)
    ax.set_ylabel("交付率")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylim(0.7, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, out, "fig5_3_lineage")


# ------------------------------------------------------------------ #
# fig5_4 SVEA 特征区间 + fig5_5 归因参考重叠
# ------------------------------------------------------------------ #
ZH_FEATURE = {
    "h_gains": "信道增益", "queue": "队列", "delta_queue": "队列增量",
    "arrivals": "到达量", "power_history": "功率历史", "rate_history": "速率历史",
    "lambda_bucket": "λ乘子", "dvq_backlog": "交付欠账", "split_mask_alpha": "拆分比α",
    "group_ids": "分组", "dinkelbach_eta": "Dinkelbach比值", "qos_lambda": "QoS负载",
    "qos_priority": "QoS优先级",
}
ZH_TARGET = {"P_total": "总功率", "d_mean": "排空比例均值", "z_top1": "解码序置信"}
ZH_REF = {"zero": "全零", "means": "边际均值", "boot": "bootstrap", "neutral": "中性"}
SNR_LIST = [0, 5, 10, 15, 20, 25, 30]


def fig_svea(out: Path) -> None:
    S = json.load(open(ROOT / "logs" / "interpretability" / "summary.json", encoding="utf-8"))
    svea = S["svea"]
    names = [ZH_FEATURE.get(n, n) for n in svea["feature_names"]]
    n = len(SNR_LIST)
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.2))
    flat = [ax for row in axes for ax in row]
    for ax in flat:
        ax.axis("off")
    for ax, s in zip(flat, SNR_LIST):
        d = svea[str(s)]
        e = np.array(d["sv"])
        lo = np.array(d["ci95"]["lo"]) if d.get("ci95") else e
        hi = np.array(d["ci95"]["hi"]) if d.get("ci95") else e
        # 误差棒须画到点估计的上下半宽（旧版误用全区间宽度 hi−lo，误差棒放大一倍）
        yerr = np.vstack([np.clip(e - lo, 0.0, None), np.clip(hi - e, 0.0, None)])
        colors = ["#d62728" if j in d["sv_neg"] else "#1f77b4" for j in range(len(names))]
        ax.axis("on")
        ax.bar(range(len(names)), e, yerr=yerr, color=colors, capsize=2,
               error_kw={"lw": 1, "ecolor": "#7f7f7f"})
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, fontsize=6)
        ax.text(0.02, 0.96, f"{s} dB", transform=ax.transAxes, fontsize=9,
                va="top", ha="left")
    for ax in flat[len(SNR_LIST):]:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, out, "fig5_4_svea")


def fig_overlap(out: Path) -> None:
    S = json.load(open(ROOT / "logs" / "interpretability" / "summary.json", encoding="utf-8"))
    val = S["validation"]
    refs = ["zero", "means", "boot", "neutral"]
    pairs = [f"{r1}-{r2}" for i, r1 in enumerate(refs) for r2 in refs[i + 1:]]
    targets = ["P_total", "d_mean", "z_top1"]
    fig, axes = plt.subplots(1, len(targets), figsize=(8.6, 3.5), squeeze=False)
    for ax, t in zip(axes[0], targets):
        M = np.full((len(SNR_LIST), len(pairs)), np.nan)
        for i, s in enumerate(SNR_LIST):
            jm = val.get(str(s), {}).get(t, {}).get("jaccard_refs", {})
            for j, pr in enumerate(pairs):
                a, b = pr.split("-")
                M[i, j] = jm.get(f"{a}_vs_{b}", np.nan)
        im = ax.imshow(M, cmap="RdYlBu_r", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_yticks(range(len(SNR_LIST)), labels=[f"{s} dB" for s in SNR_LIST], fontsize=7)
        ax.set_xticks(range(len(pairs)),
                      labels=[f"{ZH_REF[a]}-{ZH_REF[b]}" for a, b in [p.split("-") for p in pairs]],
                      rotation=30, ha="right", fontsize=6)
        ax.set_xlabel(ZH_TARGET.get(t, t))
    fig.subplots_adjust(right=0.84, bottom=0.36, wspace=0.5)
    cbar_ax = fig.add_axes([0.865, 0.16, 0.02, 0.6])
    fig.colorbar(im, cax=cbar_ax, label="top-5 Jaccard 重叠")
    _save(fig, out, "fig5_5_overlap")


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> int:
    _setup_rc()
    fig_training(OUT_DIR)
    fig_benchmarks(OUT_DIR)
    fig_lineage(OUT_DIR)
    fig_svea(OUT_DIR)
    fig_overlap(OUT_DIR)
    print(f"全部输出到 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())