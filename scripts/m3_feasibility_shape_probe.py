# -*- coding: utf-8 -*-
"""M3 前置：可行域形状探针（零重训）——份额形状版。

问题（THEORY_RESEARCH_PLAN.md §3.1 解析/学习分叉）：
  交付可行域 F_s = {a : zeta(s,a) >= zeta_SDMA(s)} 沿哪个方向可达？
  v1（总功率阈值）已被 2026-10-29 探针否定：0 dB 功率越高交付越低
  （形状收益 > 总量收益），15/30 dB 交付对功率平坦或反向。故本版刻画
  **形状坐标**（logit 展宽 tau）下的三名量：
    (i)  冠军操作点余量 margin(tau=1)        ——哪个点最紧、冠军距门多远
    (ii) zeta 的单调方向（Spearman，含反向/平坦判据）——可行域沿哪个方向开
    (iii) zeta 展幅 std ——坡度/饱和区
  逐状态原始 (tau, P, zeta, margin) 曲线写 JSON，供 M3 设计文档作证。

用法：
  PYTHONPATH=. python scripts/m3_feasibility_shape_probe.py \
      [--checkpoint ...] [--n_states 4] [--tau_grid ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_snr_sweep import SNR_POINTS  # noqa: E402
from scripts.evaluate_snr_sweep_temporal import make_temporal_config_cls  # noqa: E402
from scripts.interpretability_analysis import build_env, load_worker  # noqa: E402
from src.optimization.solver_expert import SolverExpert  # noqa: E402

K = 24
P_K_MAX = 1.0
ACT_BOUND = 3.0

# SDMA 交付参照（50ep 官方口径，tab:bench）
ZETA_SDMA = {0: 0.856, 5: 0.979, 10: 0.961, 15: 0.887, 20: 0.795, 25: 0.724, 30: 0.664}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """手工 Spearman（不依赖 scipy）：秩 x 与秩 y 的 Pearson。"""
    def _rank(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(a) + 1)
        return ranks

    rx, ry = _rank(x), _rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    return float(np.sum(rx * ry) / denom) if denom > 0 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="logs/m5_armA_sftp/checkpoints/best_guard.pt")
    ap.add_argument("--n_states", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=50,
                    help="队列建立所需时隙数（重置后 Q=0 => D=0 全判不可行）")
    ap.add_argument("--tau_grid", type=str,
                    default="0,0.2,0.4,0.6,0.8,1.0,1.3,1.6,2.0,2.5,3.0,4.0,6.0",
                    help="logit 展宽扫描（tau=0 均匀/总功率≈24W，tau=1 冠军展宽）")
    args = ap.parse_args(argv)

    cfg_cls, _ = make_temporal_config_cls(4, ("h_gains", "queue", "power"), prune_constants=False)
    tau_grid = [float(x) for x in args.tau_grid.split(",")]
    tau_champ = 1.0
    champ_idx = tau_grid.index(tau_champ) if tau_champ in tau_grid else None

    import torch  # noqa: PLC0415
    out = {}
    for snr_db, sigma2 in SNR_POINTS:
        env, _ = build_env(cfg_cls, snr_db, sigma2, args.seed)
        expert = SolverExpert(env, n_steps=1, simplified_action=False)
        worker = load_worker(args.checkpoint, env)
        zeta_gate = ZETA_SDMA[int(snr_db)]
        key = str(int(snr_db))

        per_state = []
        for si in range(args.n_states):
            obs, _ = env.reset(seed=args.seed + si)
            env.set_split_mask(np.ones(K) * 0.5)
            for _ in range(args.warmup):
                with torch.no_grad():
                    r_w = worker.act(torch.from_numpy(obs).unsqueeze(0), deterministic=True)
                action = np.concatenate([
                    r_w[0].squeeze(0).numpy(),
                    r_w[1].squeeze(0).numpy().astype(np.float64),
                ])
                obs, _, te, tr, _ = env.step(action)
                if te or tr:
                    obs, _ = env.reset(seed=args.seed + si)
                env.set_split_mask(np.ones(K) * 0.5)
            with torch.no_grad():
                r_p = worker.act(torch.from_numpy(obs).unsqueeze(0), deterministic=True)
            cont = r_p[0].squeeze(0).numpy()          # 冠军 4K: [alpha, p, d]
            p0_logits = cont[K:2 * K]
            order_pol = r_p[1].squeeze(0).numpy().astype(np.int64)
            st = expert._read_state(env)              # noqa: SLF001
            # 快照含 torch 张量；统一转 numpy（evaluate 内部自会处理张量）
            st = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v))
                  for k, v in st.items()}
            H_hat = st["H_hat"] if "H_hat" in st else st["H_hat_est"]
            gains = np.sum(np.abs(H_hat) ** 2, axis=(1, 2))   # 天线维求和 → (K,)
            order_gain = np.argsort(-gains).astype(np.int64)
            A_tot = float(max(np.sum(st["lam"]), 1e-8))
            l0 = np.clip(p0_logits, -ACT_BOUND, ACT_BOUND)
            l_mean = float(np.mean(l0))

            state_rec = {"Q_mean": float(st["Q"].mean()), "A_mean": float(st["lam"].mean())}
            for order_name, order in (("policy", order_pol), ("gain_desc", order_gain)):
                taus, powers, zetas, margins = [], [], [], []
                for tau in tau_grid:
                    l = np.clip(l_mean + tau * (l0 - l_mean), -ACT_BOUND, ACT_BOUND)
                    r = expert.evaluate(env, l, order)
                    D = float(np.sum(np.maximum(r["D"], 0.0)))
                    zeta = D / A_tot
                    taus.append(tau)
                    powers.append(float(r["total_power"]))
                    zetas.append(zeta)
                    margins.append(zeta - zeta_gate)
                taus = np.asarray(taus)
                powers = np.asarray(powers)
                zetas = np.asarray(zetas)
                margins = np.asarray(margins)
                rec = {
                    "taus": [round(float(x), 3) for x in taus],
                    "power_w": [round(float(x), 3) for x in powers],
                    "zeta": [round(float(x, ) if isinstance(x, (int, float, np.floating)) else 0.0, 5) for x in zetas],
                    "champion_margin": (round(float(margins[champ_idx]), 5) if champ_idx is not None else None),
                    "spearman_pow_zeta": round(_spearman(powers, zetas), 4),
                    "zeta_std": round(float(zetas.std()), 5),
                    "never_feasible": bool(np.all(margins < 0.0)),
                }
                state_rec[f"order_{order_name}"] = rec
            per_state.append(state_rec)

        # 跨状态汇总（按 order）
        summary = {"n_states": args.n_states, "zeta_gate": zeta_gate}
        for order_name in ("policy", "gain_desc"):
            key_o = f"order_{order_name}"
            cm = [s[key_o]["champion_margin"] for s in per_state if s[key_o]["champion_margin"] is not None]
            rho = [s[key_o]["spearman_pow_zeta"] for s in per_state]
            zstd = max(s[key_o]["zeta_std"] for s in per_state)
            nf = sum(1 for s in per_state if s[key_o]["never_feasible"])
            n_inv = sum(1 for s in per_state if s[key_o]["spearman_pow_zeta"] < -0.3)
            n_flat = sum(1 for s in per_state if abs(s[key_o]["spearman_pow_zeta"]) <= 0.3)
            summary[f"champion_margin_med_{order_name}"] = round(float(np.median(cm)), 5) if cm else None
            summary[f"champion_margin_min_{order_name}"] = round(float(np.min(cm)), 5) if cm else None
            summary[f"spearman_min_{order_name}"] = round(float(min(rho)), 4)
            summary[f"n_inverted_{order_name}"] = n_inv
            summary[f"n_flat_{order_name}"] = n_flat
            summary[f"zeta_std_max_{order_name}"] = round(float(zstd), 5)
            summary[f"never_feasible_states_{order_name}"] = nf
        summary["states"] = per_state
        out[key] = summary
        print(f"SNR {snr_db:>2} dB | gate={zeta_gate:.3f} | ", flush=True)
        for order_name in ("policy", "gain_desc"):
            s = summary
            print(f"   {order_name:8s}: champ_margin_med={s.get('champion_margin_med_' + order_name)} "
                  f"min={s.get('champion_margin_min_' + order_name)} "
                  f"rho_min={s.get('spearman_min_' + order_name)} "
                  f"inv#={s.get('n_inverted_' + order_name)} flat#={s.get('n_flat_' + order_name)} "
                  f"none#={s.get('never_feasible_states_' + order_name)}", flush=True)

    # 绑定点与方向小结（解释留给 M3 设计文档）
    print("\n=== 冠军操作点余量（med, min）按点 ===", flush=True)
    for k in out:
        v = out[k]
        print(f"   {int(k):>2} dB: {v.get('champion_margin_med_policy')} / "
              f"{v.get('champion_margin_min_policy')}", flush=True)
    Path("logs/m3_feasibility_shape.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("结果 -> logs/m3_feasibility_shape.json（含逐状态曲线）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())