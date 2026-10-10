"""事后可解释性分析 CLI：SVEA 全局特征子集选择 + KernelSHAP/EG 局部归因。

两篇文献融合（方法详见 src/analysis/ 各模块 docstring）：
1. SVEA（Tripathi et al., IEEE BigData 2020）：特征块为玩家的分类博弈，
   Shapley 值误差分摊 → SVEA_neg 支配子集（普适阈值 0）+ t-CI + PSV。
2. DeepSHAP 族（Fernando et al., SIGIR'19）：Expected Gradients（模型内省
   反向传播）vs KernelSHAP（模型无关代理），四种参照变体 → 敏感性 + recall@k。

流程: 采集 champion 策略确定性 rollout 状态（npz 缓存）→ 按 --tasks 跑分析 →
写 summary.json + 图 → stdout 两张 markdown 表。

用法（Windows venv，PYTHONPATH=.）：
    # 快速冒烟（1 个 SNR、2 个 episode、短 MC）
    python scripts/interpretability_analysis.py --snr_subset 20 --n_episodes 2 \
        --n_pairs 64 --bg_kernel 8 --n_eg 16 --ci_groups 0
    # 全量 7 SNR 论文口径
    python scripts/interpretability_analysis.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):  # Windows GBK 控制台：stdout 统一 UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import matplotlib

matplotlib.use("Agg")

from src.analysis import validation as val_mod
from src.analysis.expected_gradients import eg_blocks, eg_channels
from src.analysis.features import (
    NATIVE_FEATURE_NAMES,
    SHAP_PLAYER_NAMES,
    ObsLayout,
    build_neutral_obs,
    global_tail,
    native_blocks,
    ref_bootstrap,
    ref_means,
    ref_zero,
    resolve_layout,
)
from src.analysis.games import psv_linear, svea_game, t_ci
from src.analysis.kernel_shap import kernel_shap_blocks, kernel_shap_channels
from src.analysis.models import (
    assert_softmax_power_share,
    build_env,
    load_meta,
    load_worker,
    make_temporal_config_cls,
)
from src.analysis.targets import meta_targets, worker_targets

#: 评估口径 SNR 点（镜像 scripts/evaluate_snr_sweep.py 的 SNR_POINTS 常量）
SNR_POINTS = [
    (0, 1.0),
    (5, 0.3162),
    (10, 0.1),
    (15, 0.03162),
    (20, 0.01),
    (25, 0.003162),
    (30, 0.001),
]
#: snr_db -> sigma2 查表
SNR_SIGMA = {s: sig for s, sig in SNR_POINTS}

#: MeanFieldCriticBias 既有手挑通道（论文对照基线）：h_gains/queue/power_history/qos_priority
MEAN_FIELD_BASELINE_CHANNELS = (0, 1, 4, 12)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="logs/m5_armA_sftp/checkpoints/best_guard.pt")
    p.add_argument("--snr_subset", default="", help="逗号分隔 SNR 点（如 0,10,20,30；空=全部 7 点）")
    p.add_argument("--n_episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--obs_history_window", type=int, default=4, help="训练期时序窗口（champion=4）")
    p.add_argument("--obs_history_features", default="h_gains,queue,power")
    p.add_argument("--out_dir", default="logs/interpretability", help="npz/JSON 输出目录")
    p.add_argument("--cache", default="", help="状态缓存 npz 路径；存在则复用，否则采集后写入")
    p.add_argument("--tasks", default="svea,kernel,eg,validation,figs", help="逗号分隔任务集合")
    # SVEA
    p.add_argument("--label", default="served", choices=["served", "queue_growth"])
    p.add_argument("--m_sv", type=int, default=1000)
    p.add_argument("--B_sv", type=int, default=200)
    p.add_argument("--ci_groups", type=int, default=30, help="t-CI 分组数（0=跳过 CI）")
    p.add_argument("--ci_B", type=int, default=15, help="每组 CI 的排列样本数")
    p.add_argument("--m_ci", type=int, default=40, help="每组 CI 的子采样样本数")
    p.add_argument("--psv_mlp", action="store_true", help="同时跑非线性 PSV（torch 小 MLP，默认关）")
    # KernelSHAP / EG
    p.add_argument("--n_pairs", type=int, default=512, help="KernelSHAP 互补联盟对数")
    p.add_argument("--bg_kernel", type=int, default=32)
    p.add_argument("--n_eg", type=int, default=256)
    p.add_argument("--n_attributed", type=int, default=64, help="每 SNR 归因的状态数")
    p.add_argument("--refs", default="zero,means,boot,neutral")
    p.add_argument("--targets", default="P_total,d_mean,z_top1,alpha_avg,group_logp_0")
    p.add_argument("--figs_dir", default="docs/paper/figs")
    p.add_argument("--plot-summary", default="", help="只重绘模式：既有 summary.json 路径（跳过采集/分析）")
    return p.parse_args(argv)


def _parse_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv=None):
    args = parse_args(argv)
    if args.plot_summary:
        _plot_from_summary(args)
        return
    t0 = time.time()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figs_dir).mkdir(parents=True, exist_ok=True)

    cfg_cls = make_temporal_config_cls(
        window=args.obs_history_window,
        features=tuple(_parse_list(args.obs_history_features)),
    )
    layout = resolve_layout(cfg_cls())

    snr_list = list(SNR_SIGMA.keys())
    if args.snr_subset:
        snr_list = [int(s) for s in _parse_list(args.snr_subset)]
        unknown = [s for s in snr_list if s not in SNR_SIGMA]
        if unknown:
            raise SystemExit(f"--snr_subset 含未知点 {unknown}，可用 {list(SNR_SIGMA)}")
    refs = _parse_list(args.refs)
    targets = _parse_list(args.targets)
    tasks = set(_parse_list(args.tasks))
    meta_target_names = [t for t in targets if t.startswith(("alpha", "group"))]

    print(
        f"# 可解释性分析  checkpoint={args.checkpoint}  obs_dim={layout.obs_dim()}  H={layout.H}  SNR={snr_list}",
        flush=True,
    )

    # ---------------- M1 采集（重放 champion 策略状态） ----------------
    dataset = _load_or_collect(args, cfg_cls, layout, snr_list)

    summary: dict = {
        "meta": {
            "checkpoint": str(args.checkpoint),
            "snr_points": snr_list,
            "n_episodes": args.n_episodes,
            "obs_dim": layout.obs_dim(),
            "temporal": {"H": layout.H, "features": list(layout.history_features)},
            "seed": args.seed,
        },
        "svea": {},
        "kernel_shap": {},
        "eg": {},
        "validation": {},
    }

    if "svea" in tasks:
        print("\n## SVEA 全局特征子集选择（文献 1）", flush=True)
        summary["svea"] = _run_svea(args, dataset, layout, snr_list)

    # ---------------- 局部归因（需模型权重） ----------------
    need_worker = ("kernel" in tasks or "eg" in tasks) and bool(
        [t for t in targets if not t.startswith(("alpha", "group"))]
    )
    need_meta = ("kernel" in tasks or "eg" in tasks) and bool(meta_target_names)
    if need_worker or need_meta:
        workers, metas = _load_models(args, cfg_cls, snr_list, need_worker, need_meta)
    else:
        workers, metas = {}, {}

    if "kernel" in tasks:
        print("\n## KernelSHAP 局部归因（模型无关代理，文献 2 验证代理）", flush=True)
        summary["kernel_shap"] = _run_local_attribution(
            args, dataset, layout, snr_list, refs, targets, "kernel", workers, metas
        )

    if "eg" in tasks:
        print("\n## Expected Gradients 局部归因（DeepSHAP 族内省）", flush=True)
        summary["eg"] = _run_local_attribution(args, dataset, layout, snr_list, refs, targets, "eg", workers, metas)

    if "validation" in tasks:
        print("\n## 验证协议（文献 2：参照敏感性 + 模型无关代理对照）", flush=True)
        summary["validation"] = val_mod.build_validation(summary["kernel_shap"], summary["eg"], refs, k=5)

    if "figs" in tasks:
        _make_figures(args, summary, snr_list, refs, targets)

    if "svea" in tasks:
        _print_svea_table(args, summary, snr_list)
    if "validation" in tasks and summary["validation"]:
        _print_validation_table(summary, snr_list, refs, targets)

    summary["meta"]["runtime_s"] = round(time.time() - t0, 1)
    out_json = Path(args.out_dir) / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsummary -> {out_json}", flush=True)


def _plot_from_summary(args):
    """只读既有 summary.json 重出图（无需重跑 7.7h 分析）。"""
    Path(args.figs_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.plot_summary)
    summary = json.loads(path.read_text(encoding="utf-8"))
    snr_list = [int(s) for s in summary["meta"]["snr_points"]]
    first_snr = str(snr_list[0])
    targets = list(summary["kernel_shap"].get(first_snr, {}).keys())
    refs = list(summary["kernel_shap"].get(first_snr, {}).get(targets[0], {}).keys()) if targets else []
    print(f"# 依据 {path} 重绘（SNR={snr_list} refs={refs}）", flush=True)
    _make_figures(args, summary, snr_list, refs, targets)
    print(f"figures -> {args.figs_dir}", flush=True)


# ---------------------------------------------------------------- 采集 -------


def _load_or_collect(args, cfg_cls, layout, snr_list):
    cache_path = Path(args.cache) if args.cache else None
    if cache_path is not None and cache_path.exists():
        print(f"复用状态缓存 {cache_path}", flush=True)
        raw = np.load(cache_path, allow_pickle=False)
        return {snr: _npz_to_block(raw, snr) for snr in snr_list}

    print("采集 champion 确定性 rollout 状态 …", flush=True)
    dataset: dict = {}
    for snr_db in snr_list:
        sigma2 = SNR_SIGMA[snr_db]
        env, layout_i = build_env(cfg_cls, snr_db, sigma2, args.seed)
        if layout_i != layout:
            raise SystemExit(f"layout 不匹配: {layout_i} vs {layout}")
        worker = load_worker(args.checkpoint, env)
        assert_softmax_power_share(env)
        meta_ac, t_s = load_meta(args.checkpoint, env)
        ds = _collect_snr(env, worker, meta_ac, t_s, args.n_episodes, args.seed)
        dataset[snr_db] = ds
        print(f"  SNR {snr_db:>2} dB: {ds['obs'].shape[0]} slots", flush=True)

    if cache_path is not None:
        flat = {}
        for snr in snr_list:
            for k in _NPZ_KEYS:
                flat[f"{k}_s{snr:02d}"] = dataset[snr][k]
        flat["snr_db"] = np.asarray(snr_list, dtype=np.int64)
        np.savez_compressed(cache_path, **flat)
        print(f"状态缓存 -> {cache_path}", flush=True)
    return dataset


_NPZ_KEYS = ("obs", "user_feat", "alpha", "gid", "served", "arrivals", "qb", "qa", "p_total", "acont", "order")


def _collect_snr(env, worker, meta_ac, t_s, n_episodes, seed0):
    rec: dict[str, list] = {k: [] for k in _NPZ_KEYS}

    def meta_decide():
        uf = torch.from_numpy(env.get_user_features().copy()).float().unsqueeze(0)
        with torch.no_grad():
            primary, group_ids, _, _ = meta_ac.act(uf, deterministic=True)
        env.set_split_mask(primary.detach().cpu().numpy().astype(np.float32))
        env.set_group_ids(group_ids.detach().cpu().numpy().astype(np.int64))

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(seed0) + ep)
        meta_decide()
        for t in range(env.T):
            rec["obs"].append(np.asarray(obs, dtype=np.float32))
            rec["user_feat"].append(env.get_user_features().copy().astype(np.float32))
            rec["alpha"].append(env.split_mask.copy())
            rec["gid"].append(np.asarray(env.group_ids, dtype=np.int64).copy())
            rec["qb"].append(env.traffic.queue_lengths.copy())
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                res = worker.act(obs_t, deterministic=True)
                acont = res[0].squeeze(0).cpu().numpy()
                order = res[1].squeeze(0).cpu().numpy()
            action = np.concatenate([acont, order.astype(np.float64)])
            obs, _, term, trunc, info = env.step(action)
            rec["served"].append(np.asarray(info["served"], dtype=np.float32))
            rec["arrivals"].append(np.asarray(info["arrivals"], dtype=np.float32))
            rec["qa"].append(env.traffic.queue_lengths.copy())
            rec["p_total"].append(env._prev_power.copy())
            rec["acont"].append(acont.astype(np.float32))
            rec["order"].append(order.astype(np.int64))
            if (t + 1) % t_s == 0 and not (term or trunc):
                meta_decide()
            if term or trunc:
                break
    return {k: np.stack(v) for k, v in rec.items()}


# ---------------------------------------------------------------- SVEA -------


def _svea_dataset(ds, layout: ObsLayout, label: str):
    """(用户, 时隙) 样本：13 原生特征 + ±1 标签。"""
    feats = native_blocks(ds["obs"], layout)  # (N, K, 13)
    X = feats.reshape(-1, layout.n_native)  # (N*K, 13)
    if label == "served":
        y = (ds["served"].reshape(-1) > 0).astype(np.float64) * 2.0 - 1.0
    else:
        y = (ds["qa"].reshape(-1) > ds["qb"].reshape(-1)).astype(np.float64) * 2.0 - 1.0
    return X, y


def _run_svea(args, dataset, layout, snr_list):
    out = {"label": args.label}
    from src.analysis.games import constant_feature_cols, psv_mlp, standardize

    for snr in snr_list:
        X, y = _svea_dataset(dataset[snr], layout, args.label)
        Xz, _, _ = standardize(X)  # 全集一次标准化，CI 分组沿用同一变换
        # 固定 SNR 评估点内确定性/零方差特征（null player，如 arrivals/λ_bucket/η）
        const_feats = constant_feature_cols(Xz)
        rng = np.random.default_rng(args.seed * 1000 + snr)
        game = svea_game(Xz, y, rng, m_sv=args.m_sv, B=args.B_sv,
                         zscore=False, exclude=const_feats)
        ci95 = _ci_estimate(Xz, y, layout, args, snr, rng, exclude=const_feats) if args.ci_groups > 0 else None
        psv, acc_sub, acc_full = psv_linear(X, y, game["sv_neg"], rng, exclude=const_feats)
        entry = {
            "c_tilde": game["c_tilde"],
            "tr_er_N": game["tr_er_N"],
            "phi": [round(float(v), 6) for v in game["phi"]],
            "sv": [round(float(v), 6) for v in game["sv"]],
            "sv_neg": [int(j) for j in game["sv_neg"]],
            "ci95": None
            if ci95 is None
            else {
                "lo": [round(float(v), 6) for v in ci95[1]],
                "hi": [round(float(v), 6) for v in ci95[2]],
            },
            "psv_linear": _r4(psv),
            "acc_subset": _r4(acc_sub),
            "acc_full": _r4(acc_full),
            "m_sv": args.m_sv,
            "class_frac_pos": _r4(float((y > 0).mean())),
        }
        if args.psv_mlp:
            psv_mlp_v, acc_mlp_sub, acc_mlp_full = psv_mlp(X, y, game["sv_neg"], rng)
            entry["psv_mlp"] = _r4(psv_mlp_v)
            entry["acc_mlp_subset"] = _r4(acc_mlp_sub)
            entry["acc_mlp_full"] = _r4(acc_mlp_full)
        out[str(snr)] = entry
        print(f"  SNR {snr:>2}: SVEA_neg={out[str(snr)]['sv_neg']}  "
              f"PSV={out[str(snr)]['psv_linear']}  excluded={list(const_feats)}", flush=True)
    out["feature_names"] = list(NATIVE_FEATURE_NAMES)
    out["mean_field_channels"] = list(MEAN_FIELD_BASELINE_CHANNELS)
    return out


def _r4(v):
    return None if v is None else round(float(v), 4)


def _ci_estimate(X, y, layout, args, snr, rng, exclude=None):
    """分组 t-CI：时隙分 G 组，每组小样本重跑分类博弈（文献 1 §3.2）。

    exclude 与主博弈一致（同一 null-player 集合），保证各组 e 向量同一特征空间。
    """
    n_slots = X.shape[0] // layout.K
    g = max(1, n_slots // args.ci_groups)
    G = max(1, n_slots // g)
    e_rows = []
    for gi in range(G):
        slot0, slot1 = gi * g, min(n_slots, (gi + 1) * g)
        idx = np.arange(slot0 * layout.K, slot1 * layout.K)
        rng2 = np.random.default_rng(args.seed * 1000 + snr * 100 + gi)
        game = svea_game(X[idx], y[idx], rng2, m_sv=args.m_ci, B=args.ci_B,
                         zscore=False, exclude=exclude)
        e_rows.append(game["sv"])
    return t_ci(np.stack(e_rows))


# ---------------------------------------------------------------- 局部归因 ----


def _load_models(args, cfg_cls, snr_list, need_worker, need_meta):
    workers, metas = {}, {}
    for snr in snr_list:
        sigma2 = SNR_SIGMA[snr]
        env, _ = build_env(cfg_cls, snr, sigma2, args.seed)
        if need_worker:
            workers[snr] = load_worker(args.checkpoint, env)
        if need_meta:
            metas[snr], _ = load_meta(args.checkpoint, env)
    return workers, metas


def _select_states(ds, n_attributed: int):
    n = ds["obs"].shape[0]
    sel = np.linspace(0, n - 1, min(n, n_attributed)).astype(int)
    return ds["obs"][sel], ds["user_feat"][sel]


def _build_refs(ds_obs, ds_user_feat, layout, snr, refs, seed):
    """obs 参照（15 玩家口径）与 meta 参照（12 通道口径）。"""
    rng = np.random.default_rng(seed * 31 + snr)
    g = global_tail(ds_obs[0], layout) if len(ds_obs) else np.zeros(3)
    obs_refs = {}
    for name in refs:
        if name == "zero":
            obs_refs[name] = ref_zero(layout)
        elif name == "means":
            obs_refs[name] = ref_means(ds_obs, layout)
        elif name == "boot":
            obs_refs[name] = ref_bootstrap(ds_obs, layout, rng, n=64)
        elif name == "neutral":
            obs_refs[name] = build_neutral_obs(layout, global_obs=g)
        else:
            raise SystemExit(f"未知参照 {name}")
    meta_refs = {name: _meta_ref_from_uf(ds_user_feat, name) for name in refs}
    return obs_refs, meta_refs


def _meta_ref_from_uf(ds_user_feat, name):
    """meta 参照（12 通道语义）：zero / means / neutral / boot。"""
    u = np.asarray(ds_user_feat, dtype=np.float64)
    if name == "zero":
        return np.zeros((u.shape[1], u.shape[2]))
    if name == "means":
        return u.reshape(-1, u.shape[2]).mean(axis=0)
    if name == "neutral":
        v = u.reshape(-1, u.shape[2]).mean(axis=0)
        return np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, v[7], v[8], 0.0, 1.0, 1.0])
    rng = np.random.default_rng(7)
    idx = rng.integers(0, u.shape[0], size=64)
    return u[idx]


def _is_meta_target(target: str) -> bool:
    return target.startswith(("alpha", "group"))


def _run_local_attribution(args, dataset, layout, snr_list, refs, targets, kind, workers, metas):
    out: dict = {}
    for snr in snr_list:
        obs_sel, uf_sel = _select_states(dataset[snr], args.n_attributed)
        obs_refs, meta_refs = _build_refs(dataset[snr]["obs"], uf_sel, layout, snr, refs, args.seed)
        worker, meta = workers.get(snr), metas.get(snr)
        out[str(snr)] = {}
        for target in targets:
            out[str(snr)][target] = {}
            for ref in refs:
                if _is_meta_target(target):
                    attr = _attr_meta(args, meta, target, uf_sel, meta_refs[ref], kind, snr)
                else:
                    attr = _attr_worker(args, worker, target, obs_sel, obs_refs[ref], layout, kind, snr)
                out[str(snr)][target][ref] = np.asarray(attr, dtype=np.float64).tolist()
            print(f"  SNR {snr:>2} {kind:7s} {target:14s} done", flush=True)
    return out


def _attr_worker(args, worker, target, obs_sel, ref, layout, kind, snr):
    fn = worker_targets(worker, target)
    acc = np.zeros(layout.n_players)
    for i in range(obs_sel.shape[0]):
        seed = args.seed + snr + i
        if kind == "kernel":
            phi, _, _ = kernel_shap_blocks(
                fn, obs_sel[i], layout, ref, n_pairs=args.n_pairs, bg_k=args.bg_kernel, seed=seed
            )
        else:
            phi = eg_blocks(fn, obs_sel[i], layout, ref, n_eg=args.n_eg, seed=seed)
        acc += phi
    return acc / obs_sel.shape[0]


def _attr_meta(args, meta, target, uf_sel, ref, kind, snr):
    fn = meta_targets(meta, target)
    C = uf_sel.shape[2]
    acc = np.zeros(C)
    for i in range(uf_sel.shape[0]):
        seed = args.seed + snr + i
        if kind == "kernel":
            phi, _, _ = kernel_shap_channels(
                fn, uf_sel[i], ref, C, n_pairs=args.n_pairs, bg_k=args.bg_kernel, seed=seed
            )
        else:
            phi = eg_channels(fn, uf_sel[i], ref, C, n_eg=args.n_eg, seed=seed)
        acc += phi
    return acc / uf_sel.shape[0]


# ---------------------------------------------------------------- 图表/表格 ----


def _make_figures(args, summary, snr_list, refs, targets):
    if summary.get("svea"):
        _fig_svea_intervals(args, summary["svea"], snr_list)
    if summary.get("kernel_shap") and summary.get("eg") and summary.get("validation"):
        _fig_overlap(args, summary, snr_list, refs, targets)
        _fig_attributions(args, summary, snr_list)
    print(f"figures -> {args.figs_dir}", flush=True)


def _fig_svea_intervals(args, svea, snr_list):
    import matplotlib.pyplot as plt

    names = svea["feature_names"]
    n = len(snr_list)
    ncols, nrows = (min(n, 4), max(1, int(np.ceil(n / 4))))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 6.5), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]
    for ax in flat_axes:
        ax.axis("off")
    for ax, (snr, d) in zip(flat_axes, [(s, svea[str(s)]) for s in snr_list]):
        ax.axis("on")
        e = np.array(d["sv"])
        lo = np.array(d["ci95"]["lo"]) if d.get("ci95") else e
        hi = np.array(d["ci95"]["hi"]) if d.get("ci95") else e
        colors = ["#d62728" if j in d["sv_neg"] else "#1f77b4" for j in range(len(names))]
        ax.bar(
            range(len(names)), e, yerr=np.abs(hi - lo), color=colors, capsize=2, error_kw={"lw": 1, "ecolor": "#7f7f7f"}
        )
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, fontsize=6)
        ax.set_title(f"SNR {snr} dB", fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(args.figs_dir) / "fig_svea_intervals.png", dpi=150)
    plt.close(fig)


def _fig_overlap(args, summary, snr_list, refs, targets):
    """参照敏感性热图：每目标一子面板（x = 参照对 y = SNR），同色标。"""
    import matplotlib.pyplot as plt

    worker_targets_used = [t for t in targets if not _is_meta_target(t)]
    pairs = [f"{r1}-{r2}" for i, r1 in enumerate(refs) for r2 in refs[i + 1 :]]
    n_t = len(worker_targets_used)
    ncols = min(n_t, 3)
    nrows = int(np.ceil(n_t / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 1.6 * nrows + 1.4), squeeze=False)
    vmin, vmax = 0.0, 1.0
    for ax, t in zip([a for row in axes for a in row], worker_targets_used):
        M = np.full((len(snr_list), len(pairs)), np.nan)
        for i, s in enumerate(snr_list):
            for j, pr in enumerate(pairs):
                v = summary["validation"].get(str(s), {}).get(t, {}).get("jaccard_refs", {})
                M[i, j] = v.get(f"{pr.split('-')[0]}_vs_{pr.split('-')[1]}", np.nan)
        im = ax.imshow(M, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(snr_list)), labels=[f"{s} dB" for s in snr_list], fontsize=7)
        ax.set_xticks(range(len(pairs)), labels=pairs, rotation=90, fontsize=6)
        ax.set_title(t, fontsize=9)
        ax.grid(False)
    for ax in [a for row in axes for a in row][len(worker_targets_used) :]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, label="Jaccard@5", fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(Path(args.figs_dir) / "fig_topk_overlap.png", dpi=200)
    plt.close(fig)


def _fig_attributions(args, summary, snr_list):
    """EG(KernelSHAP 既定口径参照) 归因条形图：target × {0,30}dB，top-8 块。"""
    import matplotlib.pyplot as plt

    ref = "boot" if "boot" in list(
        summary["kernel_shap"].get(str(snr_list[0]), {}).get("P_total", {}).keys()
    ) else "means"
    snrs = [s for s in snr_list if s in (0, 30)] or snr_list[:2]
    targets_used = [
        t
        for t in ("P_total", "d_mean", "z_top1")
        if not _is_meta_target(t) and t in summary["kernel_shap"].get(str(snrs[0]), {})
    ]
    if not targets_used:
        return
    nrows, ncols = len(targets_used), len(snrs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 4.6), squeeze=False)
    names = SHAP_PLAYER_NAMES
    for r, target in enumerate(targets_used):
        for c, snr in enumerate(snrs):
            ker = np.array(summary["kernel_shap"][str(snr)][target][ref])
            eg = np.array(summary["eg"][str(snr)][target][ref])
            idx = np.argsort(np.abs(ker))[-8:]
            ax = axes[r][c]
            x = np.arange(8)
            ax.bar(x - 0.2, ker[idx], 0.4, color="#1f77b4", label="KernelSHAP")
            ax.bar(x + 0.2, eg[idx], 0.4, color="#d62728", label="EG")
            ax.axhline(0.0, color="k", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([names[i].replace("_", " ") for i in idx], rotation=60, fontsize=5)
            ax.set_title(f"{target} @ {snr}dB", fontsize=9)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(args.figs_dir) / "fig_attributions.png", dpi=200)
    plt.close(fig)


def _print_svea_table(args, summary, snr_list):
    svea = summary["svea"]
    names = svea["feature_names"]
    print(f"\n### 表 1：per-SNR SVEA 特征评价（label={svea['label']}，阈值 0 → SVEA_neg）")
    print("| SNR | 支配子集 SVEA_neg | c̃ | PSV | acc_sub/acc_full |")
    print("|---|---|---|---|---|")
    for snr in snr_list:
        d = svea[str(snr)]
        sub = ", ".join(names[j] for j in d["sv_neg"]) or "∅"
        print(f"| {snr:>2} dB | {sub} | {d['c_tilde']:.3f} | {d['psv_linear']} | {d['acc_subset']}/{d['acc_full']} |")
    mf = svea["mean_field_channels"]
    print(f"\nMeanFieldCriticBias 既有手挑通道基线 = {mf} (" + ", ".join(names[j] for j in mf) + ")")


def _print_validation_table(summary, snr_list, refs, targets):
    val = summary["validation"]
    tgt = [t for t in targets if not _is_meta_target(t)] or list(targets)[:1]
    pair_names = [f"{r1}-{r2}" for i, r1 in enumerate(refs) for r2 in refs[i + 1 :]]
    hdr = "| SNR | target | " + " | ".join(f"recall@{r}" for r in refs) + " | " + " | ".join(pair_names) + " |"
    print("\n### 表 2：局部归因验证（EG vs KernelSHAP recall@5；参照对 Jaccard@5，EG 用对应参照）")
    print(hdr)
    print("|" + "---|" * (2 + len(refs) + len(pair_names)))
    for snr in snr_list:
        for t in tgt:
            d = val.get(str(snr), {}).get(t, {})
            rec = " | ".join(f"{d.get('recall_k', {}).get(r, '-'):>5}" for r in refs)
            jac = " | ".join(
                f"{d.get('jaccard_refs', {}).get(f'{r1}_vs_{r2}', '-'):>5}"
                for i, r1 in enumerate(refs)
                for r2 in refs[i + 1 :]
            )
            print(f"| {snr:>2} | {t:10s} | {rec} | {jac} |")


# ---------------------------------------------------------------- npz 工具 ----


def _npz_to_block(raw, snr):
    pre = lambda k: raw[f"{k}_s{snr:02d}"]
    return {k: pre(k) for k in _NPZ_KEYS}


if __name__ == "__main__":
    main()
