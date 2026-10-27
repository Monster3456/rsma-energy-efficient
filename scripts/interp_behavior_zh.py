"""策略行为因果分析（中文图表，学位论文第 5 章用）。

取代 SVEA 的相关性证据，改用三类**机制原生 / 因果**证据回答两个问题：
  策略到底依赖哪些输入？输入信息够不够？

  1. ``stats``     行为统计（纯 rollout 事实，零归因假设）
  2. ``ablation``  推理期因果消融：逐块 shuffle 观测，看交付掉多少
  3. ``attention`` 自注意力权重：策略跨用户比较的机制

产物（``RSMA-papers/thesis/figures/``）：
  fig5_7_behavior.png   四个动作头的状态驱动程度
  fig5_8_ablation.png   逐块 shuffle 的交付损失（因果）
  fig5_9_attention.png  注意力权重 vs 队列秩

用法（仓库根目录，venv python）：
  PYTHONPATH=. .venv/Scripts/python.exe scripts/interp_behavior_zh.py --tasks stats
  PYTHONPATH=. .venv/Scripts/python.exe scripts/interp_behavior_zh.py --tasks ablation,attention
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "RSMA-papers" / "thesis" / "figures"
CACHE = ROOT / "logs" / "interpretability" / "states.npz"
FONT = ["Microsoft YaHei", "SimHei", "SimSun"]

# 已通过 validate_palette.js 的分类色板（Okabe-Ito 核心，全项 PASS）
C_BLUE, C_ORANGE, C_GREEN = "#0072B2", "#D55E00", "#009E73"
C_PINK, C_GRAY = "#CC79A7", "#8A8A8A"

SNR_LIST = [0, 5, 10, 15, 20, 25, 30]
K = 24
# acont 布局：3K = Worker α 块(K) + 功率 logits(K) + 排空 d(K)
SL_ALPHA, SL_PLOGIT, SL_D = slice(0, K), slice(K, 2 * K), slice(2 * K, 3 * K)

# 观测原生块（未剪枝 13 块，与 src/analysis/features.NATIVE_FEATURE_NAMES 同序）
BLOCKS = [
    "h_gains", "queue", "delta_queue", "arrivals", "power_history", "rate_history",
    "lambda_bucket", "dvq_backlog", "split_mask_alpha", "group_ids",
    "dinkelbach_eta", "qos_lambda", "qos_priority",
]
ZH_BLOCK = {
    "h_gains": "信道增益", "queue": "队列", "delta_queue": "队列增量",
    "arrivals": "到达量", "power_history": "功率历史", "rate_history": "速率历史",
    "lambda_bucket": "λ乘子", "dvq_backlog": "交付欠账", "split_mask_alpha": "拆分比α",
    "group_ids": "分组", "dinkelbach_eta": "η比值", "qos_lambda": "QoS负载",
    "qos_priority": "QoS权",
}


def _rc():
    plt.rcParams["font.sans-serif"] = FONT
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.8,
        "grid.linewidth": 0.5, "grid.alpha": 0.28,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {name}.png -> {OUT_DIR}")


def _load_states():
    z = np.load(CACHE, allow_pickle=False)
    out = {}
    for s in SNR_LIST:
        k = f"_s{s:02d}"
        if f"obs{k}" in z:
            out[s] = {name.replace(k, ""): z[name] for name in z.files if name.endswith(k)}
    return out


# ==================================================================== #
# 1) 行为统计
# ==================================================================== #
def task_stats() -> None:
    st = _load_states()
    fig = plt.figure(figsize=(11.2, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26)

    # (a) 各动作输出的状态方差：直接回答"哪个头在随状态变化"
    ax = fig.add_subplot(gs[0, 0])
    heads = ["Worker $\\alpha$", "排空 $d_k$", "Meta 组级 $\\alpha$", "功率份额 $p_k$"]
    stds = [
        float(np.mean([st[s]["acont"][:, SL_ALPHA].std() for s in SNR_LIST])),
        float(np.mean([st[s]["acont"][:, SL_D].std() for s in SNR_LIST])),
        float(np.mean([st[s]["alpha"].std() for s in SNR_LIST])),
        float(np.mean([st[s]["p_total"].std() for s in SNR_LIST])),
    ]
    cols = [C_GRAY, C_GRAY, C_ORANGE, C_GREEN]
    bars = ax.bar(range(4), stds, color=cols, width=0.6, edgecolor="none")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 3.0)
    ax.set_xticks(range(4), heads)
    ax.set_ylabel("输出的跨状态标准差（对数轴）")
    ax.set_title("(a) Worker $\\alpha$ 与排空为常数，功率头方差最大", loc="left", fontsize=9.5)
    for b, v in zip(bars, stds):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.35, f"{v:.2e}",
                ha="center", va="bottom", fontsize=7.2, color=C_GRAY)
    ax.axhline(1e-3, color=C_GRAY, lw=0.9, ls=":")
    ax.text(3.45, 1.15e-3, "判为常数", fontsize=7, color=C_GRAY, ha="right")
    ax.grid(axis="y")

    # (b) Worker α 死维 vs Meta 组级 α
    ax = fig.add_subplot(gs[0, 1])
    w_alpha = np.concatenate([st[s]["acont"][:, SL_ALPHA].ravel() for s in SNR_LIST])
    m_alpha = np.concatenate([st[s]["alpha"].ravel() for s in SNR_LIST])
    ax.hist(w_alpha, bins=41, range=(0.0, 1.05), density=True, color=C_BLUE, alpha=0.30,
            edgecolor=C_BLUE, lw=0.8, label="Worker $\\alpha$ 输出")
    ax.hist(m_alpha, bins=41, range=(0.0, 1.05), density=True, color=C_ORANGE, alpha=0.50,
            edgecolor=C_ORANGE, lw=0.8, label="Meta 组级 $\\alpha$")
    ax.set_xlabel("$\\alpha$ 取值")
    ax.set_ylabel("密度")
    ax.set_title("(b) Worker $\\alpha$ 为死维，实际拆分由 Meta 给出", loc="left", fontsize=9.5)
    ax.legend(frameon=False, loc="upper center")
    ax.text(0.98, 0.55,
            f"Worker $\\sigma={w_alpha.std():.1e}$\n"
            f"Meta $\\mu={m_alpha.mean():.3f}$, $\\sigma={m_alpha.std():.3f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=C_GRAY)
    ax.grid(axis="y")

    # (c) 解码序 top-1 用户频次
    ax = fig.add_subplot(gs[1, 0])
    counts = np.zeros(K, dtype=float)
    for s in SNR_LIST:
        counts += np.bincount(st[s]["order"][:, 0], minlength=K)
    frac = counts / counts.sum()
    order = np.argsort(-frac)
    ax.bar(range(K), frac[order], color=[C_ORANGE if i == 0 else C_BLUE for i in range(K)],
           width=0.72)
    ax.axhline(1.0 / K, color=C_GRAY, lw=1.1, ls="--")
    ax.text(K - 0.4, 1.0 / K, "均匀 $1/K$", color=C_GRAY, fontsize=8, ha="right", va="bottom")
    ax.set_xticks(range(K))
    ax.set_xticklabels([str(i + 1) for i in order], fontsize=6)
    ax.set_xlabel("用户（按 top-1 频次降序）")
    ax.set_ylabel("被排首位的比例")
    ax.set_title(f"(c) 解码序 top-1：最高频用户 {100 * frac[order][0]:.1f}%"
                 f"（均匀 {100 / K:.1f}%）", loc="left", fontsize=9.5)
    ax.grid(axis="y")

    # (d) 功率份额 vs 队列（分箱均值 + 标准误）
    ax = fig.add_subplot(gs[1, 1])
    for s, color in zip([0, 15, 30], [C_GREEN, C_BLUE, C_ORANGE]):
        qb = st[s]["qb"].ravel()
        p = st[s]["p_total"].ravel()
        bins = np.quantile(qb, np.linspace(0, 1, 13))
        bins[0], bins[-1] = qb.min() - 1e-9, qb.max() + 1e-9
        idx = np.digitize(qb, bins) - 1
        xs, ys, es = [], [], []
        for b in range(len(bins) - 1):
            m = idx == b
            if m.sum() < 20:
                continue
            xs.append(qb[m].mean())
            ys.append(p[m].mean())
            es.append(p[m].std() / np.sqrt(m.sum()))
        ax.errorbar(xs, ys, yerr=es, color=color, marker="o", ms=3.6, lw=1.6,
                    capsize=2.2, label=f"{s} dB")
    ax.set_xlabel("队列长度 $Q_k$")
    ax.set_ylabel("分配功率 $p_k$ (W)")
    ax.set_title("(d) 功率头随队列上升", loc="left", fontsize=9.5)
    ax.legend(frameon=False, title="SNR", loc="lower right", fontsize=7.5)
    ax.grid()

    _save(fig, "fig5_7_behavior")

    # 汇总数据（供 LaTeX 表引用）
    summary = {
        "d_ratio": {
            "mean": float(np.mean([st[s]["acont"][:, SL_D].mean() for s in SNR_LIST])),
            "std": float(np.mean([st[s]["acont"][:, SL_D].std() for s in SNR_LIST])),
            "frac_at_one": float(np.mean([
                (st[s]["acont"][:, SL_D] > 0.999).mean() for s in SNR_LIST])),
        },
        "worker_alpha_std": float(w_alpha.std()),
        "meta_alpha_mean": float(m_alpha.mean()),
        "meta_alpha_std": float(m_alpha.std()),
        "order_top1_max_frac": float(frac[order][0]),
        "order_top1_uniform": float(1.0 / K),
        "groups_per_slot": float(np.mean([
            np.mean([len(np.unique(r)) for r in st[s]["gid"][::10]]) for s in SNR_LIST])),
        "p_share_std": float(np.mean([st[s]["acont"][:, SL_PLOGIT].std() for s in SNR_LIST])),
    }
    (OUT_DIR / "interp_behavior_stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  行为统计 ->", json.dumps(summary, ensure_ascii=False))


# ==================================================================== #
# 2) 因果消融：逐块 shuffle
# ==================================================================== #
ABL_ARMS = [
    ("queue", "队列"),
    ("h_gains", "信道增益"),
    ("delta_queue", "队列增量"),
    ("power_history", "功率历史"),
    ("rate_history", "速率历史"),
    ("dvq_backlog", "交付欠账"),
    ("split_mask_alpha", "拆分比α"),
    ("qos_lambda", "QoS负载"),
    ("qos_priority", "QoS权"),
]


def task_ablation(args) -> None:
    """把指定观测块在**同一时隙内跨用户 shuffle**（保边际、破对应），测交付损失。

    shuffle 而非置零：置零会把输入推到分布外，shuffle 保边际分布，只打断
    "这块信息归属于哪个用户"，更干净地测因果依赖。
    """
    from interpretability_analysis import SNR_SIGMA, build_env, load_meta, load_worker

    from src.analysis.models import make_temporal_config_cls

    # champion 口径：H=4 时序窗口 + 未剪枝 603 维（与 best_guard 检查点一致）
    cfg_cls = make_temporal_config_cls(window=4, prune_constants=False)
    out = {}

    def run_arm(snr_db, shuffle_block: str | None, n_ep: int, seed0: int):
        sigma2 = SNR_SIGMA[snr_db]
        env, _ = build_env(cfg_cls, snr_db, sigma2, args.seed)
        worker = load_worker(args.checkpoint, env)
        meta_ac, t_s = load_meta(args.checkpoint, env)
        rng = np.random.default_rng(seed0)
        deliv, queues, pw = [], [], []
        for ep in range(n_ep):
            obs, _ = env.reset(seed=int(seed0) + ep)
            uf = None
            # meta 定
            import torch
            uf = torch.from_numpy(env.get_user_features().copy()).float().unsqueeze(0)
            with torch.no_grad():
                primary, gids, _, _ = meta_ac.act(uf, deterministic=True)
            env.set_split_mask(primary.detach().cpu().numpy().astype(np.float32))
            env.set_group_ids(gids.detach().cpu().numpy().astype(np.int64))
            last_deliv = 0.0
            for t in range(env.T):
                o = np.asarray(obs, dtype=np.float32).copy()
                if shuffle_block is not None:
                    j = BLOCKS.index(shuffle_block)
                    o[j * K:(j + 1) * K] = o[rng.permutation(K) + j * K]
                with torch.no_grad():
                    res = worker.act(
                        __import__("torch").from_numpy(o).unsqueeze(0), deterministic=True)
                    acont = res[0].squeeze(0).cpu().numpy()
                    order = res[1].squeeze(0).cpu().numpy()
                action = np.concatenate([acont, order.astype(np.float64)])
                obs, _, term, trunc, info = env.step(action)
                last_deliv = float(info.get("delivery_ratio", last_deliv))
                if (t + 1) % t_s == 0 and not (term or trunc):
                    with torch.no_grad():
                        primary, gids, _, _ = meta_ac.act(uf, deterministic=True)
                    env.set_split_mask(primary.detach().cpu().numpy().astype(np.float32))
                    env.set_group_ids(gids.detach().cpu().numpy().astype(np.int64))
                if term or trunc:
                    break
            deliv.append(last_deliv)
            queues.append(float(env.traffic.queue_lengths.mean()))
            pw.append(float(env._prev_power.sum()))
        return float(np.mean(deliv)), float(np.mean(queues)), float(np.mean(pw))

    t0 = time.time()
    for snr_db in SNR_LIST:
        print(f"[ablation] SNR {snr_db} dB ...", flush=True)
        base = run_arm(snr_db, None, args.n_episodes, args.seed)
        out[str(snr_db)] = {"baseline": {"deliv": base[0], "queue": base[1], "power": base[2]}}
        for key, _zh in ABL_ARMS:
            r = run_arm(snr_db, key, args.n_episodes, args.seed + 7)
            out[str(snr_db)][key] = {"deliv": r[0], "queue": r[1], "power": r[2]}
            print(f"    shuffle {key:18s} deliv={r[0]:.4f}  Δ={r[0]-base[0]:+.4f}", flush=True)
    print(f"[ablation] done in {time.time()-t0:.0f}s")
    (OUT_DIR / "interp_ablation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_ablation(out)


def plot_ablation(out: dict) -> None:
    """交付损失（baseline − shuffle）按块排序；正值 = 该块被拿掉后交付下降。

    形式：横向条形（7 点均值）+ 跨 SNR 极差须 + 逐点散点。横排便于读中文标签，
    极差须给出效应随 SNR 的跨度，避免 7 条折线糊在一起。
    """
    arms = [k for k, _ in ABL_ARMS]
    loss = {k: [] for k in arms}
    for s in SNR_LIST:
        base = out[str(s)]["baseline"]["deliv"]
        for k in arms:
            loss[k].append(base - out[str(s)][k]["deliv"])
    mean_loss = {k: float(np.mean(v)) for k, v in loss.items()}
    order = sorted(arms, key=lambda k: mean_loss[k])   # 升序：损失大在下方→翻转后在上

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ys = np.arange(len(order))
    means = [mean_loss[k] for k in order]
    lo = [min(loss[k]) for k in order]
    hi = [max(loss[k]) for k in order]
    colors = [C_ORANGE if m > 0.005 else C_GRAY for m in means]
    ax.barh(ys, means, color=colors, height=0.58, zorder=3)
    for y, m, l, h in zip(ys, means, lo, hi):
        ax.plot([l, h], [y, y], color="k", lw=1.0, zorder=4, alpha=0.75)
        ax.plot([l, l], [y - 0.12, y + 0.12], color="k", lw=1.0, zorder=4, alpha=0.75)
        ax.plot([h, h], [y - 0.12, y + 0.12], color="k", lw=1.0, zorder=4, alpha=0.75)
    # 逐 SNR 散点
    for i, s in enumerate(SNR_LIST):
        ax.scatter([loss[k][i] for k in order], ys, s=12, color=C_BLUE,
                   alpha=0.55, zorder=5,
                   label="逐 SNR 点" if i == 0 else None)
    xmax = max(hi) if hi else 1.0
    for y, m in zip(ys, means):
        off = 0.006 * xmax / 0.08 if xmax > 0 else 0.004
        ax.text(m + off, y, f"{m:+.3f}", va="center", fontsize=7.6,
                color=C_GRAY, zorder=6)
    ax.axvline(0.0, color="k", lw=0.8, zorder=2)
    ax.set_yticks(ys, [ZH_BLOCK[k] for k in order], fontsize=9)
    ax.set_xlabel("shuffle 该块的交付率损失（baseline − shuffle）")
    ax.set_title("因果消融：逐块跨用户 shuffle 的交付损失", loc="left", fontsize=10)
    ax.set_ylim(-0.6, len(order) - 0.35)
    ax.grid(axis="x")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _save(fig, "fig5_8_ablation")


# ==================================================================== #
# 3) 注意力分析
# ==================================================================== #
def task_attention(args) -> None:
    import torch
    from interpretability_analysis import SNR_SIGMA, build_env, load_meta, load_worker

    from src.analysis.models import make_temporal_config_cls

    cfg_cls = make_temporal_config_cls(window=4, prune_constants=False)
    attns = {}
    for snr_db in SNR_LIST:
        sigma2 = SNR_SIGMA[snr_db]
        env, _ = build_env(cfg_cls, snr_db, sigma2, args.seed)
        worker = load_worker(args.checkpoint, env)
        meta_ac, _t_s = load_meta(args.checkpoint, env)
        captured: list = []
        hooks = []
        for layer in worker.attn_layers:
            def _hook(_m, _i, o, bag=captured):
                # nn.MultiheadAttention 返回 (out, weights)
                w = o[1] if isinstance(o, tuple) else None
                if w is not None:
                    bag.append(w.detach().cpu().numpy())
            hooks.append(layer["attn"].register_forward_hook(_hook))
        qb_rank_all, attn_all = [], []
        for ep in range(args.n_episodes):
            obs, _ = env.reset(seed=int(args.seed) + ep)
            uf = torch.from_numpy(env.get_user_features().copy()).float().unsqueeze(0)
            with torch.no_grad():
                primary, gids, _, _ = meta_ac.act(uf, deterministic=True)
            env.set_split_mask(primary.detach().cpu().numpy().astype(np.float32))
            env.set_group_ids(gids.detach().cpu().numpy().astype(np.int64))
            for t in range(env.T):
                captured.clear()
                with torch.no_grad():
                    res = worker.act(
                        torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0),
                        deterministic=True)
                    acont = res[0].squeeze(0).cpu().numpy()
                    order = res[1].squeeze(0).cpu().numpy()
                if captured:
                    # 取最后一层、多头平均 → (K, K)
                    w = captured[-1]
                    w = w.mean(axis=1) if w.ndim == 4 else w
                    attn_all.append(w[0])
                    qb = env.traffic.queue_lengths
                    qb_rank_all.append(np.argsort(np.argsort(-qb)))  # 0=队列最高
                action = np.concatenate([acont, order.astype(np.float64)])
                obs, _, term, trunc, _info = env.step(action)
                if term or trunc:
                    break
        for h in hooks:
            h.remove()
        attns[snr_db] = (np.mean(attn_all, axis=0), np.array(qb_rank_all))
        print(f"[attention] SNR {snr_db}: {len(attn_all)} slots", flush=True)
    np.savez(OUT_DIR / "interp_attention.npz",
             **{f"attn_{s}": attns[s][0] for s in SNR_LIST},
             **{f"rank_{s}": attns[s][1] for s in SNR_LIST})
    plot_attention(attns)


def plot_attention(attns: dict) -> None:
    """按队列秩重排的平均注意力矩阵 + 注意力与队列秩的关系。

    三个热图共用一个色标（否则无法横向比较）；(d) 用逐用户散点而非分箱，
    24 个点全保留，不丢两端秩。
    """
    fig = plt.figure(figsize=(11.8, 4.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.05, 1.25], wspace=0.62)

    # 统一色标
    vmax = max(float(attns[s][0].max()) for s in [0, 15, 30])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, s in zip(axes, [0, 15, 30]):
        A, ranks = attns[s]
        mean_rank = ranks.mean(axis=0)
        perm = np.argsort(mean_rank)
        M = A[np.ix_(perm, perm)]
        im = ax.imshow(M, cmap="viridis", aspect="auto", interpolation="nearest",
                       vmin=0.0, vmax=vmax)
        ax.set_title(f"{s} dB", fontsize=9)
        ax.set_xlabel("被关注者（队列秩→）")
        ax.set_ylabel("关注者（队列秩→）")
        ax.set_xticks([]); ax.set_yticks([])
    cax = fig.add_subplot(gs[0, 3])
    fig.colorbar(im, cax=cax, label="注意力权重")

    # (d) 逐用户：队列秩 vs 收到的平均注意力
    ax = fig.add_subplot(gs[0, 4])
    for s, color in zip([0, 15, 30], [C_GREEN, C_BLUE, C_ORANGE]):
        A, ranks = attns[s]
        rb = ranks.mean(axis=0)          # (K,) 每用户平均队列秩
        att_to = A.mean(axis=0)          # (K,) 每用户收到的平均注意力
        r = float(np.corrcoef(rb, att_to)[0, 1])
        ax.scatter(rb, att_to, s=26, color=color, alpha=0.82, zorder=4,
                   label=f"{s} dB  ($r$={r:+.2f})")
    ax.axhline(1.0 / K, color=C_GRAY, lw=1.0, ls="--")
    ax.text(0.98, 1.0 / K, " 均匀 $1/K$", transform=ax.get_yaxis_transform(),
            color=C_GRAY, fontsize=7.5, ha="right", va="bottom")
    ax.set_xlabel("被关注者的平均队列秩（0=队列最高）")
    ax.set_ylabel("收到的平均注意力")
    ax.set_title("(d) 注意力 vs 队列秩", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, "fig5_9_attention")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="策略行为因果分析（中文图表）")
    ap.add_argument("--tasks", default="stats")
    ap.add_argument("--checkpoint", default="logs/m5_armA_sftp/checkpoints/best_guard.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_episodes", type=int, default=12)
    args = ap.parse_args(argv)
    _rc()
    for t in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        print(f"== task {t} ==", flush=True)
        if t == "stats":
            task_stats()
        elif t == "ablation":
            task_ablation(args)
        elif t == "attention":
            task_attention(args)
        else:
            raise SystemExit(f"未知 task {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
