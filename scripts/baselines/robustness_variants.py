"""Robustness of the distilled policy under channel/traffic mis-specification (supplement K).

Runtime-injected variants; the immutable files under src/env/ are never modified:
  rician : per-episode rank-1 LOS component with Rician factor kappa_dB plus a
           Doppler-rate AR(1) coherence rho_d (faster fading than the rho=0.9 prior)
  mmpp   : two-state Markov-modulated Poisson arrivals per user (bursty traffic),
           rate states {0.5x, 2.0x} with stationary mean 1.0x

Usage:
  PYTHONPATH=. python scripts/baselines/robustness_variants.py \
      --checkpoint logs/m5_armA_sftp/checkpoints/best_guard.pt \
      --channel rician --traffic mmpp --points 0,15,25,30 --n_episodes 30
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from scripts.baselines.sdma_baseline import SDMABaseline
from scripts.evaluate_snr_sweep import (
    SNR_POINTS,
    build_config_for_snr,
    evaluate_method,
    load_meta_model,
    load_ppo_model,
    make_baseline_action_fn,
    make_ppo_action_fn,
)
from src.env.channel import ChannelGenerator, _cn_randn
from src.env.rsma_env import RSMAEnergyEnv
from src.env.traffic import TrafficManager


class RicianDopplerChannel(ChannelGenerator):
    """Rician fading + Doppler-rate AR(1) coherence, injected at runtime."""

    def _mix(self) -> None:
        w = np.sqrt(1.0 / (1.0 + self.kappa))
        l = np.sqrt(self.kappa / (1.0 + self.kappa))
        self._H_true = w * self._H_ray + l * self._H_los

    def reset(self) -> dict:
        K, N_r, N_t = self.K, self.N_r, self.N_t
        los = np.empty((K, N_r, N_t), dtype=complex)
        for k in range(K):
            a = _cn_randn(self._los_rng, (N_r, 1))
            b = _cn_randn(self._los_rng, (N_t, 1))
            m = a @ b.conj().T
            los[k] = m / (np.linalg.norm(m) + 1e-12)
        self._H_los = los
        self._H_ray = _cn_randn(self._rng, (K, N_r, N_t))
        self._mix()
        self._history.clear()
        self._history.append(self._H_true.copy())
        return self._build_observation()

    def step(self) -> dict:
        W = self._sample_innovation()
        self._H_ray = self.rho_d * self._H_ray + np.sqrt(1.0 - self.rho_d**2) * W
        self._mix()
        self._history.append(self._H_true.copy())
        return self._build_observation()


class MMPPTraffic(TrafficManager):
    """Two-state MMPP arrivals: sojourn 40/20 slots, rates {0.5x, 2.0x}, mean 1.0x."""

    def reset(self):
        self._state = self._mmpp_rng.integers(0, 2, size=self.K)
        return super().reset()

    def step(self, served_bits: np.ndarray):
        if getattr(self, "_state", None) is None:
            self._state = np.zeros(self.K, dtype=int)
        stay = np.where(self._state == 0, 1.0 - 1.0 / 40.0, 1.0 - 1.0 / 20.0)
        jump = self._mmpp_rng.random(self.K) > stay
        self._state = np.where(jump, 1 - self._state, self._state)
        mult = np.where(self._state == 0, 0.5, 2.0)
        base = self.lambda_per_user
        self.lambda_per_user = np.asarray(base, dtype=float) * mult
        out = super().step(served_bits)
        self.lambda_per_user = base
        return out


def _bind_instance_methods(obj, cls, names) -> None:
    """把子类方法绑定到基类实例上（`obj.__class__ = cls` 在本环境静默无效）。"""
    for n in names:
        if n in cls.__dict__:
            bound = cls.__dict__[n].__get__(obj, type(obj))
            obj.__dict__[n] = bound


def inject(env, args) -> None:
    if args.channel == "rician":
        _bind_instance_methods(env.channel, RicianDopplerChannel,
                               ("reset", "step", "_mix"))
        env.channel.kappa = 10.0 ** (args.kappa_db / 10.0)
        env.channel.rho_d = args.rho_d
        env.channel._los_rng = np.random.default_rng(7777)
        env.channel._H_ray = None
        env.channel._H_los = None
    if args.traffic == "mmpp":
        tr = env.traffic
        tr._mmpp_rng = np.random.default_rng(31337)
        tr._state = None
        base_reset, base_step = tr.reset, tr.step

        def _tr_reset():
            base_reset()
            tr._state = tr._mmpp_rng.integers(0, 2, size=tr.K)

        def _tr_step(served_bits):
            if getattr(tr, "_state", None) is None:
                tr._state = np.zeros(tr.K, dtype=int)
            stay = np.where(tr._state == 0, 1.0 - 1.0 / 40.0, 1.0 - 1.0 / 20.0)
            jump = tr._mmpp_rng.random(tr.K) > stay
            tr._state = np.where(jump, 1 - tr._state, tr._state)
            mult = np.where(tr._state == 0, 0.5, 2.0)
            base = np.asarray(tr.lambda_per_user, dtype=float)
            tr.lambda_per_user = base * mult
            out = base_step(served_bits)
            tr.lambda_per_user = base  # 恢复到相对基线的 λ（burst 只作用于本 episode）
            return out

        tr.reset, tr.step = _tr_reset, _tr_step


def set_history_window(cfg, window: int) -> None:
    if hasattr(cfg, "system") and hasattr(cfg.system, "obs_history_window"):
        cfg.system.obs_history_window = window
        feats = getattr(cfg.system, "obs_history_features", ("h_gains", "queue", "power"))
        cfg.ppo.worker_d_user = 13 + window * len(feats)


def run_point(snr_db, sigma2, args):
    print(
        f"\n=== SNR={snr_db:>2} dB (sigma2={sigma2:.4g}) channel={args.channel} traffic={args.traffic} ===", flush=True
    )
    cfg = build_config_for_snr(sigma2, args.csi_mode)
    set_history_window(cfg, args.obs_history_window)
    env = RSMAEnergyEnv(cfg)
    # env.reset(seed=...) 内部 _build_modules() 会把 channel/traffic 重建回基线类，
    # 把注入实例冲掉 → 包一层：reset 完成后重新注入并重渲首帧 H_true。
    _base_reset = env.reset

    def _reset_patched(seed=None, options=None):
        _base_reset(seed=seed, options=options)
        inject(env, args)
        env._channel_obs = env.channel.reset()
        return env._get_observation(), {}

    env.reset = _reset_patched
    inject(env, args)
    res = {}
    if "sdma" in args.methods:
        res["SDMA"] = evaluate_method(make_baseline_action_fn(SDMABaseline(cfg)), env, args.n_episodes)
    if "ppo" in args.methods:
        model, ckpt_dim, err = load_ppo_model(args.checkpoint, env)
        if model is None:
            raise SystemExit(f"checkpoint load failed: {err}")
        meta_ac, T_s, _merr = load_meta_model(args.checkpoint, env)
        res["PPO-RSMA"] = evaluate_method(
            make_ppo_action_fn(model, ckpt_dim, env.obs_dim, K=env.K),
            env,
            args.n_episodes,
            meta_ac=meta_ac,
            T_s=T_s,
        )
    for name, r in res.items():
        print(
            f"    {name:<11s} P={r['total_power']:.3f} Q={r['mean_queue']:.2f} "
            f"Deliv={r['delivery_ratio']:.3f} R_sum={r['R_sum']:.2f} Ovf={r['overflow_rate']:.4f}",
            flush=True,
        )
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--channel", choices=["rayleigh", "rician"], default="rician")
    ap.add_argument("--traffic", choices=["poisson", "mmpp"], default="mmpp")
    ap.add_argument("--kappa_db", type=float, default=5.0)
    ap.add_argument("--rho_d", type=float, default=0.8)
    ap.add_argument("--points", default="0,15,25,30")
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--methods", default="ppo,sdma")
    ap.add_argument("--obs_history_window", type=int, default=4)
    ap.add_argument("--csi_mode", default="imperfect")
    args = ap.parse_args(argv)

    want = {int(x) for x in args.points.split(",")}
    print(
        f"### robustness table: channel={args.channel}(kappa={args.kappa_db}dB,rho_d={args.rho_d}) traffic={args.traffic}"
    )
    print("| SNR(dB) | PPO Deliv | PPO Q | PPO P(W) | SDMA Deliv | SDMA Q | SDMA P(W) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for snr_db, sigma2 in SNR_POINTS:
        if snr_db not in want:
            continue
        res = run_point(snr_db, sigma2, args)
        p = res.get("PPO-RSMA", {})
        s = res.get("SDMA", {})
        print(
            f"| {snr_db} | {p.get('delivery_ratio', float('nan')):.3f} | {p.get('mean_queue', float('nan')):.2f} "
            f"| {p.get('total_power', float('nan')):.2f} | {s.get('delivery_ratio', float('nan')):.3f} "
            f"| {s.get('mean_queue', float('nan')):.2f} | {s.get('total_power', float('nan')):.2f} |",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
