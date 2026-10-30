# -*- coding: utf-8 -*-
"""D2: tau-集中度投影 M4 评估桩（spec-stage1 §2，零重训）。

在冠军 checkpoint 上推理侧插桩 tau-集中度投影，输出：
  (a) 官方 7x50 对比：投影 OFF vs ON（交付/队列/功率/EE 全点）；
  (b) 逐状态证书：样本状态集上 zeta_off/on vs gate 的违约率、no-cert（logit 地板）占比；
  (c) 最紧点身份（cert 范围内违约率最高 / 冠军余量最小的点）。

投影语义（方向封锁，spec §2.1）：
  p_logits_raw -> l0；只能向集中方向 tau in [1.0, 1.2] 推进，
  若 f(tau=1) >= gate 投影为恒等（noop）；若 f(1.2) < gate 记 no_cert 保持原动作。

用法：
  PYTHONPATH=. python scripts/m4_projection_eval.py [--checkpoint ...] \\
      [--n_episodes 50] [--points 0,15] [--cert_states 8] [--cert_slots 10] [--warmup 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_snr_sweep import (  # noqa: E402
    SNR_POINTS,
    _run_episode,
    build_config_for_snr,
    load_meta_model,
)
from scripts.evaluate_snr_sweep_temporal import make_temporal_config_cls  # noqa: E402
from scripts.interpretability_analysis import load_worker  # noqa: E402
from src.env.rsma_env import RSMAEnergyEnv  # noqa: E402
from src.optimization.solver_expert import SolverExpert  # noqa: E402

K = 24
ACT_BOUND = 3.0
TAU_MAX = 1.2
TAU_CHAMP = 1.0
GATE_EPS = 1e-6

# SDMA 交付参照（50ep 官方口径，tab:bench）——即逐点 gate
ZETA_SDMA = {0: 0.856, 5: 0.979, 10: 0.961, 15: 0.887, 20: 0.795, 25: 0.724, 30: 0.664}

# 官方全量 7 点；--points 为逗号分隔的子集
ALL_POINTS = [int(s) for s, _ in SNR_POINTS]


def _zeta_snapshot(expert, env, l: np.ndarray, order: np.ndarray) -> float:
    """单时隙交付代理 zeta = sum(D)/A_total（镜像探针口径）。"""
    r = expert.evaluate(env, l, order)
    D = float(np.sum(np.maximum(r["D"], 0.0)))
    st = expert._read_state(env)  # noqa: SLF001
    lam = st["lam"]
    lam_np = lam.detach().cpu().numpy() if hasattr(lam, "detach") else np.asarray(lam)
    A = float(max(np.sum(lam_np), 1e-8))
    return D / A


def project_tau(expert, env, p_logits_raw: np.ndarray, order: np.ndarray,
                gate: float, iters: int = 5):
    """tau-集中度投影：只向集中方向（tau in [1.0, 1.2]）推进。

    Returns:
        (l_star, tau_star, status, zeta_star)
        status: "noop"（冠军点已可行）/ "corrected"（推进至最小可行 tau）/
                "no_cert"（1.2 仍不可行，保持原动作，tau_star=None）
    """
    l0 = np.clip(np.asarray(p_logits_raw, dtype=np.float64), -ACT_BOUND, ACT_BOUND)
    l_mean = float(l0.mean())

    def zeta_of(tau: float) -> float:
        l = np.clip(l_mean + tau * (l0 - l_mean), -ACT_BOUND, ACT_BOUND)
        return _zeta_snapshot(expert, env, l, order)

    z_champ = zeta_of(TAU_CHAMP)
    if z_champ >= gate + GATE_EPS:
        return l0, TAU_CHAMP, "noop", z_champ
    z_max = zeta_of(TAU_MAX)
    if z_max < gate - GATE_EPS:
        return None, None, "no_cert", z_max
    lo, hi = TAU_CHAMP, TAU_MAX
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if zeta_of(mid) >= gate:
            hi = mid
        else:
            lo = mid
    tau_star = hi
    l_star = np.clip(l_mean + tau_star * (l0 - l_mean), -ACT_BOUND, ACT_BOUND)
    return l_star, tau_star, "corrected", _zeta_snapshot(expert, env, l_star, order)


def _p_slice(cont_size: int) -> slice:
    if cont_size == 3 * K:      # 非精简：cont = [alpha, p, d]（order 单独返回）
        return slice(K, 2 * K)
    if cont_size == 1 * K:      # armA 精简：cont = [p]
        return slice(0, K)
    raise SystemExit(f"未知 cont 长度 {cont_size}")


def make_action_fn(policy, expert, env, gate, p_sl, mode: str):
    def action_fn(obs, env_):
        o = torch.as_tensor(obs[None, :], dtype=torch.float32)
        with torch.no_grad():
            cont, order, *_ = policy.act(o, deterministic=True)
        cont = cont.numpy()[0].astype(np.float64)
        order = order.numpy()[0].astype(np.float64)
        if mode == "on":
            l_star, _tau, status, _z = project_tau(expert, env_, cont[p_sl], order, gate)
            if status in ("noop", "corrected"):
                cont[p_sl] = l_star
        return np.concatenate([cont, order])

    return action_fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="logs/m5_armA_sftp/checkpoints/best_guard.pt")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--points", type=str, default="all",
                    help="逗号分隔 SNR 子集，如 '0,15'；'all' = 全部 7 点")
    ap.add_argument("--cert_states", type=int, default=8)
    ap.add_argument("--cert_slots", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    points = [int(x) for x in args.points.split(",")] if args.points != "all" else ALL_POINTS
    mode = {}
    report = {"checkpoint": args.checkpoint, "points": points,
              "n_episodes": args.n_episodes, "cert": {"states": args.cert_states,
                                                      "slots": args.cert_slots}}
    for snr_db in points:
        cfg = build_config_for_snr(dict(SNR_POINTS)[snr_db], "imperfect")
        cfg.system.obs_prune_constant_blocks = False
        cfg.system.obs_history_window = 4
        cfg.system.obs_history_features = ("h_gains", "queue", "power")
        cfg.ppo.worker_d_user = 13 + 4 * 3
        env = RSMAEnergyEnv(cfg)
        assert env.obs_dim == 25 * K + 3, f"env.obs_dim={env.obs_dim}"
        expert = SolverExpert(env, n_steps=1, simplified_action=False)
        worker = load_worker(args.checkpoint, env)
        meta_ac, t_s, err = load_meta_model(args.checkpoint, env)
        if err is not None:
            print(f"[warn] meta 加载失败({err})，meta off")
        cont_probe = None
        obs_v, _ = env.reset(seed=args.seed)
        with torch.no_grad():
            c_p, *_ = worker.act(torch.from_numpy(obs_v).unsqueeze(0), deterministic=True)
        p_sl = _p_slice(int(c_p.shape[-1]))
        gate = ZETA_SDMA[snr_db]

        def run_eps(fn_mode: str):
            dls, pws, qs = [], [], []
            for ep in range(args.n_episodes):
                m = _run_episode(make_action_fn(worker, expert, env, gate, p_sl, fn_mode),
                                 env, seed=ep, meta_ac=meta_ac, T_s=t_s)
                dls.append(m["delivery_ratio"])
                pws.append(m["total_power"])
                qs.append(m["mean_queue"])
            return np.asarray(dls), np.asarray(pws), np.asarray(qs)

        d_off, p_off, q_off = run_eps("off")
        d_on, p_on, q_on = run_eps("on")

        # ---- 证书（官方口径）：逐 episode 交付违约率 = E1 门 ----
        ep_viol_off = float(np.mean(d_off < gate))
        ep_viol_on = float(np.mean(d_on < gate))

        # ---- slot 代理诊断（不入门，供最紧点/失败模式画像）----
        stats = {"no_cert": 0, "slots": 0, "z_off": [], "z_on": []}
        for si in range(min(args.cert_states, args.n_episodes)):
            obs, _ = env.reset(seed=args.seed + si)
            env.set_split_mask(np.ones(K) * 0.5)
            for _ in range(args.warmup):
                with torch.no_grad():
                    cb, co, *_ = worker.act(torch.from_numpy(obs).unsqueeze(0), deterministic=True)
                act = np.concatenate([cb.numpy()[0], co.numpy()[0].astype(np.float64)])
                obs, _, te, tr, _ = env.step(act)
                if te or tr:
                    obs, _ = env.reset(seed=args.seed + si)
                env.set_split_mask(np.ones(K) * 0.5)
            with torch.no_grad():
                cb, co, *_ = worker.act(torch.from_numpy(obs).unsqueeze(0), deterministic=True)
            l0_off = cb.numpy()[0][p_sl]      # project_tau 内部自己 clip
            order0 = co.numpy()[0].astype(np.int64)
            for _s in range(args.cert_slots):
                z_off = _zeta_snapshot(expert, env, l0_off, order0)
                _ls, _tau, status, z_on = project_tau(expert, env, l0_off, order0, gate)
                if status == "no_cert":
                    stats["no_cert"] += 1
                stats["z_off"].append(z_off)
                stats["z_on"].append(z_on)
                obs, _, te, tr, _ = env.step(np.concatenate(
                    [cb.numpy()[0], co.numpy()[0].astype(np.float64)]))
                if te or tr:
                    obs, _ = env.reset(seed=args.seed + si)
            stats["slots"] += args.cert_slots

        zoff = np.asarray(stats["z_off"]); zon = np.asarray(stats["z_on"])
        row = {
            "gate": gate,
            "off": {"delivery_ratio": float(d_off.mean()), "delivery_std": float(d_off.std()),
                    "mean_queue": float(q_off.mean()), "total_power": float(p_off.mean())},
            "on": {"delivery_ratio": float(d_on.mean()), "delivery_std": float(d_on.std()),
                   "mean_queue": float(q_on.mean()), "total_power": float(p_on.mean())},
            "cert": {
                "episodes": args.n_episodes,
                "ep_viol_off": ep_viol_off, "ep_viol_on": ep_viol_on,
                "slot_diag": {
                    "slots": stats["slots"],
                    "no_cert_fraction": stats["no_cert"] / stats["slots"],
                    "zeta_off_med": float(np.median(zoff)), "zeta_on_med": float(np.median(zon)),
                    "zeta_off_min": float(zoff.min()), "zeta_on_min": float(zon.min()),
                },
            },
        }
        report[str(snr_db)] = row
        print(f"SNR {snr_db:>2} dB | gate={gate:.3f} | off={d_off.mean():.4f} "
              f"on={d_on.mean():.4f} (P {p_off.mean():.2f}->{p_on.mean():.2f} W) | "
              f"ep 违约 {ep_viol_off:.3f} -> {ep_viol_on:.3f} | "
              f"no_cert_frac={row['cert']['slot_diag']['no_cert_fraction']:.2f}", flush=True)

    certs = [report[str(p)]["cert"] for p in points]
    viol_on_agg = sum(c["ep_viol_on"] * c["episodes"] for c in certs) / sum(c["episodes"] for c in certs)
    viol_off_agg = sum(c["ep_viol_off"] * c["episodes"] for c in certs) / sum(c["episodes"] for c in certs)
    dels_on = np.mean([report[str(p)]["on"]["delivery_ratio"] for p in points])
    dels_off = np.mean([report[str(p)]["off"]["delivery_ratio"] for p in points])
    print(f"\n=== D2 汇总 ===", flush=True)
    print(f"官方交付: off={dels_off:.4f} on={dels_on:.4f}  d={dels_on - dels_off:+.4f}", flush=True)
    print(f"cert ep-违约率: off={viol_off_agg:.4f} on={viol_on_agg:.4f} (C1 门 <= 1%)", flush=True)
    Path("logs/m4_projection_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("结果 -> logs/m4_projection_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())