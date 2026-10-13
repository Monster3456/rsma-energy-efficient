"""Cross-scale zero-shot transfer evaluation (Scale-Gen #1, supplement).

Tests whether the K=24 champion policy (SA-GAT worker + meta, both constructed
per-token with shared heads) transfers to other user counts K in {16,24,32}
WITHOUT retraining: rebuild the modules at the target K and load the champion
state_dict with strict=False (the only K-sized tensor is worker log_std, which
is re-initialized from the champion's per-user statistics). Baselines
(SDMA/NOMA/EqualPower) and, optionally, our engine (DiffOpt) are evaluated at
each K for reference. Official-ish protocol: 7 SNR points x n_episodes,
imperfect CSI, obs_history_window=4 (as trained).

Usage:
  PYTHONPATH=. python scripts/scale_transfer_eval.py \
      --ckpt logs/m5_armA_sftp/checkpoints/best_guard.pt --ks 16,24,32 \
      --n_episodes 50 --methods ppo,sdma
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_snr_sweep import SNR_POINTS, build_config_for_snr, evaluate_method, make_baseline_action_fn
from scripts.baselines.sdma_baseline import SDMABaseline
from scripts.baselines.noma_baseline import NOMABaseline
from scripts.baselines.equal_power import EqualPowerBaseline
from src.env.rsma_env import RSMAEnergyEnv
from src.models.actor_critic import ActorCriticAttn
from src.models.meta_controller import MetaActorCritic


def build_modules(cfg, K: int, d_user_eff: int):
    obs_dim = K * d_user_eff + 3
    worker = ActorCriticAttn(
        obs_dim=obs_dim, action_dim=4 * K, K=K, d_user=d_user_eff,
        d_global=3, hidden_dim=128, n_heads=4, n_attn_layers=2,
        use_plackett_luce=True,
    )
    meta = MetaActorCritic(K=K, d_user=cfg.hierarchical.meta_d_user,
                           hidden_dim=64, n_heads=4, n_groups=4,
                           group_level_alpha=True)
    return worker, meta, obs_dim


def load_champion(worker, meta, ckpt: dict, K: int) -> dict:
    ws = dict(ckpt.get("worker_state_dict") or ckpt.get("model_state"))
    # log_std 是唯一随 K 变尺寸的参数（3K）：pop 后按冠军跨用户分量均值重初始化，
    # 保留探索量级（strict=False 对 size 不匹配也会抛错，必须显式处理）。
    champ_ls = ws.pop("log_std", None)
    skip_shared = set(ws) - set(dict(worker.named_parameters()))
    worker.load_state_dict(ws, strict=False)
    if champ_ls is not None:
        t = torch.as_tensor(champ_ls).float()
        comp = t.view(-1, 3).mean(0) if t.numel() % 3 == 0 else t.mean().expand(3)
        worker.log_std.data.copy_(comp.repeat(K))
    if "meta_state_dict" in ckpt:
        skip_meta = set(ckpt["meta_state_dict"]) - set(dict(meta.named_parameters()))
        meta.load_state_dict(ckpt["meta_state_dict"], strict=False)
    else:
        skip_meta = set()
    return {"worker_skipped": sorted(skip_shared)[:8], "meta_skipped": sorted(skip_meta)[:8]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ks", default="16,24,32")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--methods", default="ppo,sdma,noma,equal_power")
    ap.add_argument("--csi_mode", default="imperfect")
    args = ap.parse_args(argv)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    for K in [int(x) for x in args.ks.split(",")]:
        print(f"\n########## K={K} ##########", flush=True)
        hdr = ["SNR(dB)"]
        m_ = [m.strip() for m in args.methods.split(",") if m.strip()]
        LABEL = {"ppo": "Ours(zt)", "sdma": "SDMA", "noma": "NOMA", "equal_power": "EqualPower"}
        rows = {LABEL[m]: [] for m in m_}
        for snr_db, sigma2 in SNR_POINTS:
            cfg = build_config_for_snr(sigma2, args.csi_mode)
            cfg.system.K = K
            cfg.system.obs_history_window = 4
            cfg.ppo.worker_d_user = 13 + 4 * 3  # H=4 历史观测
            env = RSMAEnergyEnv(cfg)
            worker, meta, obs_dim = build_modules(cfg, K, d_user_eff=13 + 4 * 3)
            transfer = load_champion(worker, meta, ckpt, K)

            def ppo_fn(obs, env, _w=worker, _m=meta):
                o = torch.as_tensor(obs[None, :], dtype=torch.float32)
                with torch.no_grad():
                    cont, order, *_ = _w.act(o, deterministic=True)
                return np.concatenate([cont.numpy()[0], order.numpy()[0].astype(np.float64)])

            m_ = [m.strip() for m in args.methods.split(",") if m.strip()]
            res = {}
            if "ppo" in m_:
                res[LABEL["ppo"]] = evaluate_method(ppo_fn, env, args.n_episodes)
            if "sdma" in m_:
                res[LABEL["sdma"]] = evaluate_method(make_baseline_action_fn(SDMABaseline(cfg)), env, args.n_episodes)
            if "noma" in m_:
                res[LABEL["noma"]] = evaluate_method(make_baseline_action_fn(NOMABaseline(cfg)), env, args.n_episodes)
            if "equal_power" in m_:
                res[LABEL["equal_power"]] = evaluate_method(make_baseline_action_fn(EqualPowerBaseline(cfg)), env, args.n_episodes)
            for name, r in res.items():
                rows[name].append(r["delivery_ratio"])
            if snr_db == 0:
                print("transfer-keys:", transfer, flush=True)
        print("| " + " | ".join(hdr + [m for m in rows]) + " |")
        print("|---" + "|---" * (1 + len(rows)) + "|")
        for i, (snr_db, _) in enumerate(SNR_POINTS):
            print("| " + " | ".join([str(snr_db)] + [f"{rows[m][i]:.3f}" for m in rows]) + " |")
        print("| mean | " + " | ".join([f"{np.mean(rows[m]):.3f}" for m in rows]) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())