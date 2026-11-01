# -*- coding: utf-8 -*-
"""CSI 误差界鲁棒性扫描（零重训，全 7 点 × 逐 episode 双口径）。

按固定 CSI 误差界 ε 分开全仿真：snr_dependent_csi=False + epsilon_csi=ε
（隔离"误差界"维；"perfect" = csi_mode=perfect 完全 CSI 上界参照），
每个 (ε, SNR) 逐 episode 跑官方口径，同时输出均值交付与逐回合违约率。

方法集：PPO checkpoint（champion/armA/D3，按 checkpoint 形状自动重建，
       D3 的 hm_pressure 自动注入）+ 基线（sdma/noma/equal）。

用法:
  PYTHONPATH=. python scripts/eval_csi_bound_sweep.py [--checkpoints ...] \\
      [--epsilons perfect,0,0.01,0.03,0.05,0.1,0.15,0.2] [--n_episodes 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.baselines.equal_power import EqualPowerBaseline  # noqa: E402
from scripts.baselines.noma_baseline import NOMABaseline  # noqa: E402
from scripts.baselines.sdma_baseline import SDMABaseline  # noqa: E402
from scripts.eval_simplified_arm import (  # noqa: E402
    _continuous_dim,
    _head_dim,
    _load_states,
)
from scripts.evaluate_snr_sweep import (  # noqa: E402
    SNR_POINTS,
    _run_episode,
    build_config_for_snr,
    load_meta_model,
    make_baseline_action_fn,
)
from src.env.rsma_env import RSMAEnergyEnv  # noqa: E402
from src.models.actor_critic import ActorCriticAttn  # noqa: E402

K = 24
ZETA_SDMA = {0: 0.856, 5: 0.979, 10: 0.961, 15: 0.887, 20: 0.795, 25: 0.724, 30: 0.664}
BASELINE_FACTORY = {"sdma": SDMABaseline, "noma": NOMABaseline, "equal": EqualPowerBaseline}


def _make_cfg(sigma2: float, eps: str):
    csi = "perfect" if eps == "perfect" else "imperfect"
    cfg = build_config_for_snr(sigma2, csi)
    # 冠军/armA 共同评估口径：未剪 + H=4（与 eval_simplified/m4 一致）
    cfg.system.obs_prune_constant_blocks = False
    cfg.system.obs_history_window = 4
    cfg.system.obs_history_features = ("h_gains", "queue", "power")
    cfg.ppo.worker_d_user = 13 + 4 * 3
    if eps != "perfect":
        cfg.system.snr_dependent_csi = False
        cfg.system.epsilon_csi = float(eps)
    return cfg, (0.0 if eps == "perfect" else float(eps))


def _load_policy(ck_path: str, obs_dim: int, d_user: int):
    ck, w_state, _m = _load_states(ck_path)
    action_dim = _continuous_dim(w_state) + K
    simplified = _head_dim(w_state) == 2
    d_global = 3
    has_pressure = "hm_pressure" in ck and ck.get("hm_pressure") is not None
    if has_pressure:
        d_global += 7
        obs_dim += 7
    policy = ActorCriticAttn(
        obs_dim=obs_dim, action_dim=action_dim, K=K, d_user=d_user,
        d_global=d_global, hidden_dim=128, n_heads=4, n_attn_layers=2,
        use_plackett_luce=True, simplified_action=simplified,
    )
    mm, uu = policy.load_state_dict(w_state, strict=False)
    if mm or uu:
        print(f"  [warn] {Path(ck_path).name} missing={len(mm)} unexpected={len(uu)}")
    policy.eval()
    return policy, ck, has_pressure


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--baselines", type=str, default="sdma,noma,equal")
    ap.add_argument("--epsilons", type=str, default="perfect,0,0.01,0.03,0.05,0.1,0.15,0.2")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="logs/csi_bound_sweep.json")
    args = ap.parse_args(argv)

    eps_list = args.epsilons.split(",")
    bnames = [b for b in args.baselines.split(",") if b in BASELINE_FACTORY]
    d_user = 13 + 4 * 3
    obs_dim_base = d_user * K + 3

    methods = {}
    for ck in args.checkpoints:
        policy, ck_, has_p = _load_policy(ck, obs_dim_base, d_user)
        p = Path(ck)
        name = f"{p.parent.parent.name}/{p.stem}"
        methods[name] = {"policy": policy, "pressure": has_p, "ck": ck_, "ckpt": ck}
    for b in bnames:
        methods[b] = {"baseline": b}

    report = {}
    print("| eps | " + " | ".join(f"{m}" for m in methods) +
          " | (每格=mean deliv / ep-viol)", flush=True)
    for eps in eps_list:
        row = {}
        for snr_db, sigma2 in SNR_POINTS:
            cfg, _eps_v = _make_cfg(sigma2, eps)
            env = RSMAEnergyEnv(cfg)
            for name, m in methods.items():
                row.setdefault(name, {"deliver": [], "epv": [], "power": []})
                gate = ZETA_SDMA[int(snr_db)]
                if "baseline" in m:
                    base = BASELINE_FACTORY[m["baseline"]](cfg)
                    fn = make_baseline_action_fn(base)
                    meta_ac, t_s = None, 20
                else:
                    policy = m["policy"]
                    if m["pressure"]:
                        _pv = m["ck"].get("hm_pressure") or np.zeros(7)
                        env.set_pressure_vector(np.asarray(_pv))
                    obs_dim_check = obs_dim_base + (7 if m["pressure"] else 0)
                    if env.obs_dim != obs_dim_check:
                        raise SystemExit(
                            f"{name}: env obs {env.obs_dim} != {obs_dim_check}")
                    meta_ac = None
                    t_s = 20
                    if m["ck"].get("meta_state_dict") is not None:
                        meta_ac, t_s, err = load_meta_model(m["ckpt"], env)
                        if err is not None:
                            meta_ac, t_s = None, 20

                    def fn(obs_, env_, _p=policy):
                        o_ = torch.as_tensor(obs_[None, :], dtype=torch.float32)
                        with torch.no_grad():
                            c_, od_, *_ = _p.act(o_, deterministic=True)
                        return np.concatenate([c_.numpy()[0], od_.numpy()[0].astype(np.float64)])
                for ep in range(args.n_episodes):
                    mm = _run_episode(fn, env, seed=ep, meta_ac=meta_ac, T_s=t_s)
                    row[name]["deliver"].append(float(mm["delivery_ratio"]))
                    row[name]["epv"].append(int(mm["delivery_ratio"] < gate))
                    row[name]["power"].append(float(mm["total_power"]))
        cells = []
        for name in methods:
            dv = np.asarray(row[name]["deliver"])
            ev = np.asarray(row[name]["epv"])
            cells.append(f"{dv.mean():.4f}/{ev.mean():.3f}")
        report[eps] = {name: {
            "mean_deliv_7pt": float(np.mean(row[name]["deliver"])),
            "ep_viol": float(np.mean(row[name]["epv"])),
            "mean_power": float(np.mean(row[name]["power"])),
        } for name in methods}
        print(f"| {eps:>7} | " + " | ".join(f"{c}" for c in cells) + " |", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"结果 -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())