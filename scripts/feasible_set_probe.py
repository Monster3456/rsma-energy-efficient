# -*- coding: utf-8 -*-
"""可行域几何刻画（M3 前置，服务命题 T 与可行域映射设计）。

问题：联合可行域 F = ∩_s F_s 在动作空间里有多大、什么形状？
  F_s = {a : ζ(s,a) ≥ ζ_SDMA(s),  Q̄(s,a) ≤ 1.1·Q̄_SDMA(s)}

做法：在每个 SNR 点取定状态，用可微引擎（经 SolverExpert.evaluate 镜像）扫描动作空间，
统计满足逐点门的动作占比与几何特征（连通性、维度、与策略动作的距离）。

**这是零训练的数值刻画**，不改任何模型权重。

用法：
  PYTHONPATH=. python scripts/feasible_set_probe.py --n_states 8 --n_samples 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_snr_sweep import SNR_POINTS  # noqa: E402
from scripts.interpretability_analysis import build_env, load_meta, load_worker  # noqa: E402
from scripts.evaluate_snr_sweep_temporal import make_temporal_config_cls  # noqa: E402
from src.optimization.solver_expert import SolverExpert  # noqa: E402

K = 24


def sdmaref(snr_db: float) -> tuple[float, float]:
    """SDMA 参照（表 tab:bench，50ep 官方口径）：交付率与平均队列。"""
    tab = {
        0: (0.856, 3.46), 5: (0.979, 0.68), 10: (0.961, 2.28),
        15: (0.887, 11.42), 20: (0.795, 22.39), 25: (0.724, 28.76), 30: (0.664, 34.47),
    }
    return tab[int(snr_db)]


def probe(args) -> dict:
    cfg_cls, _ = make_temporal_config_cls(4, ("h_gains", "queue", "power"), prune_constants=False)
    out = {}
    for snr_db, sigma2 in SNR_POINTS:
        env, _ = build_env(cfg_cls, snr_db, sigma2, args.seed)
        expert = SolverExpert(env, n_steps=1, simplified_action=False)
        worker = load_worker(args.checkpoint, env)
        meta, t_s = load_meta(args.checkpoint, env)
        zeta_sdma, q_sdma = sdmaref(snr_db)

        feas, margins, dists = [], [], []
        for si in range(args.n_states):
            obs, _ = env.reset(seed=args.seed + si)
            env.set_split_mask(np.ones(K) * 0.5)
            # warmup：队列需先建立，否则 D=min(d·R·τ, Q)=0，探针全判不可行
            import torch as _t
            for _ in range(args.warmup):
                with _t.no_grad():
                    _r = worker.act(_t.from_numpy(obs).unsqueeze(0), deterministic=True)
                obs, _, _te, _tr, _ = env.step(np.concatenate([
                    _r[0].squeeze(0).numpy(),
                    _r[1].squeeze(0).numpy().astype(np.float64)]))
                if _te or _tr:
                    obs, _ = env.reset(seed=args.seed + si)
            env.set_split_mask(np.ones(K) * 0.5)
            import torch
            with torch.no_grad():
                res = worker.act(torch.from_numpy(obs).unsqueeze(0), deterministic=True)
                a0 = res[0].squeeze(0).numpy()
            # 策略动作的 p_logits 块（3K: alpha, p, d）
            p0 = a0[K:2 * K]
            A_tot = float(max(env.traffic.lambda_per_user.sum(), 1e-8))
            for _ in range(args.n_samples):
                # 在策略动作附近扰动 + 均匀探索
                if np.random.rand() < 0.5:
                    p = p0 + np.random.normal(0, args.sigma, size=K)
                else:
                    p = np.random.uniform(-2.0, 2.0, size=K)
                p = np.clip(p, -3.0, 3.0)
                order = np.argsort(np.random.randn(K))
                r = expert.evaluate(env, p, order)
                D = float(np.sum(np.maximum(r.get("D", np.zeros(K)), 0.0)))
                zeta = D / A_tot
                margin = zeta - zeta_sdma          # >0 则该动作在该点可行（交付侧）
                ok = margin >= 0.0
                feas.append(ok)
                margins.append(margin)
                dists.append(float(np.linalg.norm(p - p0)))
        feas = np.array(feas)
        margins = np.array(margins)
        dists = np.array(dists)
        out[str(int(snr_db))] = {
            "frac_feasible": float(feas.mean()),
            "margin_mean": float(margins.mean()),
            "margin_std": float(margins.std()),
            "q95_margin": float(np.percentile(margins, 95)),
            "dist_feasible_mean": float(dists[feas].mean()) if feas.any() else None,
            "dist_infeasible_mean": float(dists[~feas].mean()) if (~feas).any() else None,
            "n": int(feas.size),
            "zeta_sdma": zeta_sdma,
        }
        print(f"SNR {snr_db:>2} dB: 可行动作占比={feas.mean():.4f}  "
              f"margin mean={margins.mean():+.4f} std={margins.std():.4f}  "
              f"95%分位={np.percentile(margins,95):+.4f}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="logs/m5_armA_sftp/checkpoints/best_guard.pt")
    ap.add_argument("--n_states", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=400)
    ap.add_argument("--sigma", type=float, default=0.5, help="策略动作扰动尺度")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=50, help="队列建立所需时隙数")
    args = ap.parse_args(argv)
    import time
    t0 = time.time()
    out = probe(args)
    print(f"\n[done] {time.time()-t0:.0f}s")
    print("\n=== 可行域几何小结 ===")
    fr = [out[k]["frac_feasible"] for k in out]
    print(f"各点可行动作占比: {[round(x,4) for x in fr]}")
    print(f"跨点平均: {np.mean(fr):.4f}   最低点: {np.min(fr):.4f}   最高点: {np.max(fr):.4f}")
    import json
    Path("logs/feasible_set_probe.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("结果 -> logs/feasible_set_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
