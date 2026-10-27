"""可解释性分析的中文图表（学位论文第 5 章用）。

与 ``plot_phase2_zh.py`` 同一版式约定：图内无图题，标题由 LaTeX caption 承担；
面板左上角标 SNR；坐标轴/图例全中文；输出 ``RSMA-papers/thesis/figures/``。

出三样产物：
  1. fig5_4_svea.png   —— SVEA 点估计 e_j（主博弈 m=1000），红色 = 支配子集 SVEA_neg
  2. fig5_6_ci.png     —— 区间估计（Tripathi §3.2 判据）：均值 ± 95% CI 随 SNR 变化
  3. 表：支配集与显著性对照（stdout + ``interp_zh_table.md``）

为什么分两张图：点估计与区间估计是**两个不同估计器**（主博弈 m_sv=1000 vs 子样本
m_ci=200），幅度不可比。Tripathi 的用法本来就只比符号与阈值 0，不比幅度，
所以点估计与区间估计必须分开呈现，不能把区间套在点估计柱子上。

数据来源：
  - 点估计：logs/interpretability_ci_m200/summary*.json 的 svea[*].sv / sv_neg
  - 区间估计：同文件 svea[*].ci95（G=30 组 × M=30 子博弈平均，m_ci=200）

用法（仓库根目录，venv python）：
  .venv/Scripts/python.exe scripts/plot_interp_zh.py
  .venv/Scripts/python.exe scripts/plot_interp_zh.py --src logs/interpretability_ci_v2/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "RSMA-papers" / "thesis" / "figures"
FONT = ["Microsoft YaHei", "SimHei", "SimSun"]

SNR_LIST = [0, 5, 10, 15, 20, 25, 30]
ZH_FEATURE = {
    "h_gains": "信道增益", "queue": "队列", "delta_queue": "队列增量",
    "arrivals": "到达量", "power_history": "功率历史", "rate_history": "速率历史",
    "lambda_bucket": "λ乘子", "dvq_backlog": "交付欠账", "split_mask_alpha": "拆分比α",
    "group_ids": "分组", "dinkelbach_eta": "η比值", "qos_lambda": "QoS负载",
    "qos_priority": "QoS权",
}
# 关注的两条：队列块是唯一反复出现的支配块；拆分比 α 在 20 dB 被点估计判支配
FOCUS = [("queue", "#d62728", "o"), ("split_mask_alpha", "#1f77b4", "s")]


def _setup_rc() -> None:
    plt.rcParams["font.sans-serif"] = FONT
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _load(paths):
    """合并多个 summary.json 的 svea 段（按 SNR 覆盖合并）。"""
    merged = {"feature_names": None}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            s = json.load(fh)["svea"]
        if merged["feature_names"] is None:
            merged["feature_names"] = s["feature_names"]
        for k, v in s.items():
            if k.isdigit():
                merged[k] = v
    if merged["feature_names"] is None:
        raise SystemExit("找不到任何 summary.json，请先跑 interpretability_analysis.py")
    return merged


# ------------------------------------------------------------------ #
# fig5_4：SVEA 点估计
# ------------------------------------------------------------------ #
def fig_svea(svea, out: Path) -> None:
    names = [ZH_FEATURE.get(n, n) for n in svea["feature_names"]]
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.0))
    flat = [ax for row in axes for ax in row]
    for ax in flat:
        ax.axis("off")
    for ax, s in zip(flat, SNR_LIST):
        d = svea.get(str(s))
        if d is None:
            continue
        e = np.array(d["sv"])
        ci = d.get("ci95")
        left = np.array(ci["left_of_origin"], dtype=bool) if ci else np.zeros(len(e), bool)
        # 实心红 = 支配且显著（e_j<0 且 95% CI 上界 <0）
        # 斜纹红 = 支配但不显著（e_j<0 但 CI 含 0）
        # 蓝色 = 非支配（e_j>=0）
        ax.axis("on")
        for j in range(len(names)):
            if j in d["sv_neg"]:
                robust = bool(left[j])
                ax.bar(j, e[j],
                       color="#d62728" if robust else "#f0a0a0",
                       edgecolor="#7f0000" if robust else "#d62728",
                       hatch="" if robust else "///", linewidth=0.8)
            else:
                ax.bar(j, e[j], color="#4c72b0")
        ax.axhline(0.0, color="k", lw=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=55, fontsize=6.5, ha="right")
        ax.text(0.02, 0.96, f"{s} dB", transform=ax.transAxes, fontsize=9,
                va="top", ha="left")
        ax.set_ylabel("SVEA 值 $e_j$")
    for ax in flat[len(SNR_LIST):]:
        ax.axis("off")
    # 空白面板放图例，说明三种柱型
    lg = flat[len(SNR_LIST)] if len(flat) > len(SNR_LIST) else None
    if lg is not None:
        lg.axis("on")
        lg.set_xticks([]); lg.set_yticks([])
        for sp in lg.spines.values():
            sp.set_visible(False)
        lg.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, facecolor="#d62728", edgecolor="#7f0000"),
                plt.Rectangle((0, 0), 1, 1, facecolor="#f0a0a0", edgecolor="#d62728", hatch="///"),
                plt.Rectangle((0, 0), 1, 1, facecolor="#4c72b0", edgecolor="none"),
            ],
            labels=[
                "支配且显著 ($e_j<0$, CI上界$<0$)",
                "支配但不显著 ($e_j<0$, CI含$0$)",
                "非支配 ($e_j\\geq 0$)",
            ],
            loc="center", fontsize=8.5, frameon=False,
        )
    fig.tight_layout()
    _save(fig, out, "fig5_4_svea")


# ------------------------------------------------------------------ #
# fig5_6：区间估计判据
# ------------------------------------------------------------------ #
def fig_ci(svea, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    have = [s for s in SNR_LIST if svea.get(str(s), {}).get("ci95")]
    if not have:
        print("（无 ci95 数据，跳过 fig5_6）")
        return
    xs = np.array(have, dtype=float)
    for key, color, marker in FOCUS:
        j = svea["feature_names"].index(key)
        mean = np.array([svea[str(s)]["ci95"]["mean"][j] for s in have])
        lo = np.array([svea[str(s)]["ci95"]["lo"][j] for s in have])
        hi = np.array([svea[str(s)]["ci95"]["hi"][j] for s in have])
        left = np.array([svea[str(s)]["ci95"]["left_of_origin"][j] for s in have])
        label = ZH_FEATURE.get(key, key)
        ax.plot(xs, mean, color=color, marker=marker, lw=1.5, ms=5, label=label)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)
        # 实心点 = 显著（95% CI 上界 <0）；空心点 = 不显著
        if not left.all():
            ax.plot(xs[~left], mean[~left], marker=marker, ls="none",
                    mfc="white", mec=color, ms=5.5, mew=1.3)
    ax.axhline(0.0, color="k", lw=0.9, ls="--")
    ax.annotate("阈值 0", xy=(0.015, 0.0), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=8, color="k",
                xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(SNR_LIST)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("区间估计均值 $e_j$")
    ax.legend(loc="center right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    _save(fig, out, "fig5_6_ci")


# ------------------------------------------------------------------ #
# 表：支配集与显著性对照
# ------------------------------------------------------------------ #
def table(svea, out: Path) -> str:
    rows = [
        "| SNR (dB) | $e_j$ (队列) | 支配集 $\\mathrm{SVEA}_{\\mathrm{neg}}$ | 区间估计均值 | 95\\% CI | CI上界$<0$ | 判定 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in SNR_LIST:
        d = svea.get(str(s))
        if d is None:
            continue
        j = svea["feature_names"].index("queue")
        e = d["sv"][j]
        neg = "，".join(ZH_FEATURE.get(svea["feature_names"][k], k) for k in d["sv_neg"]) or "无"
        ci = d.get("ci95")
        if ci is None:
            rows.append(f"| {s} | {e:+.4f} | {neg} | — | — | — | — |")
            continue
        mean, lo, hi = ci["mean"][j], ci["lo"][j], ci["hi"][j]
        left = bool(ci["left_of_origin"][j])
        if j in d["sv_neg"] and left:
            verdict = "支配且显著"
        elif j in d["sv_neg"] and not left:
            verdict = "支配但不显著"
        elif not (j in d["sv_neg"]) and left:
            verdict = "显著但未达支配阈值"
        else:
            verdict = "非支配"
        rows.append(
            f"| {s} | {e:+.4f} | {neg} | {mean:+.4f} | ({lo:+.4f}, {hi:+.4f}) | "
            f"{'是' if left else '否'} | {verdict} |"
        )
    text = "\n".join(rows)
    (out / "interp_zh_table.md").write_text(text + "\n", encoding="utf-8")
    return text


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png",):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200)
    print(f"  {name}.png -> {out_dir}")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="可解释性分析中文图表")
    ap.add_argument("--src", nargs="*", default=None,
                    help="summary.json 路径（可多个，按 SNR 覆盖合并）")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    src = args.src or [
        str(ROOT / "logs" / "interpretability_ci_m200" / "summary.json"),
        str(ROOT / "logs" / "interpretability_ci_m200" / "summary_s00_05_25.json"),
    ]
    svea = _load(src)
    have = [s for s in SNR_LIST if str(s) in svea]
    print(f"数据 SNR 点: {have}")

    _setup_rc()
    out = Path(args.out)
    fig_svea(svea, out)
    fig_ci(svea, out)
    print("\n### 可解释性：两判据对照\n")
    print(table(svea, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
