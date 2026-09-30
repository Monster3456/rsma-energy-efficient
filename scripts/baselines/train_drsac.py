"""DR-SAC comparator: distributionally-robust SAC for the RSMA-EE instance (supplement A).

Single-timescale, no hierarchy, no distillation, no guard (same comparator family
protocol as LDPTRLQ). The distributional robustness follows Cui et al. (DR-SAC,
ICLR 2026): the critics are trained against the worst case over an uncertainty
set of next-state perturbations (multiplicative CSI-gain errors and queue
perturbations), so the value function lower-bounds performance under CSI/traffic
mismatch within that set.

The worker policy reuses src/models/actor_critic.ActorCriticAttn (same obs/action
interface as the champion); critics are plain MLPs over (obs, action).

Usage:
  PYTHONPATH=. python scripts/baselines/train_drsac.py --exp_name comp_drsac \
      --episodes 8000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts.evaluate_snr_sweep import build_config_for_snr
from src.env.rsma_env import RSMAEnergyEnv
from src.models.actor_critic import ActorCriticAttn


class Critic(nn.Module):
    def __init__(self, dim_in: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


def perturb_obs(obs: torch.Tensor, K: int, rng: torch.Generator, mag: float = 0.2,
                device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Worst-case set: multiplicative CSI-gain error and +-queue perturbation per user."""
    out = obs.clone()
    u = (torch.rand(obs.shape[0], K, device=device, generator=rng) * 2 - 1) * mag
    out[:, :K] = out[:, :K] * (1.0 + u)
    v = (torch.rand(obs.shape[0], K, device=device, generator=rng) * 2 - 1) * 0.1
    out[:, K : 2 * K] = torch.clamp(out[:, K : 2 * K] * (1.0 + v), min=0.0)
    return out


def make_env(seed: int, snr_lo: float, snr_hi: float, rng: np.random.Generator):
    snr = rng.uniform(snr_lo, snr_hi)
    sigma2 = 10.0 ** (-snr / 10.0)
    cfg = build_config_for_snr(sigma2, "imperfect")
    env = RSMAEnergyEnv(cfg)
    obs, _info = env.reset(seed=int(seed))
    return env, np.asarray(obs, dtype=np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_name", default="comp_drsac")
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snr_lo", type=float, default=-5.0)
    ap.add_argument("--snr_hi", type=float, default=35.0)
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--update_every", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--alpha", type=float, default=0.05, help="initial entropy temperature")
    ap.add_argument("--no-perturb", action="store_true", help="plain SAC (skip the distributionally-robust worst-case over state perturbations)")
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    pert_rng = torch.Generator().manual_seed(args.seed + 1)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    log_dir = Path("logs") / args.exp_name / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path("logs") / args.exp_name / "checkpoint_meta.jsonl"

    env, obs = make_env(args.seed, args.snr_lo, args.snr_hi, np_rng)
    K = env.K
    obs_dim = int(env.obs_dim)
    act_dim = 4 * K
    policy = ActorCriticAttn(
        obs_dim=obs_dim,
        action_dim=act_dim,
        K=K,
        d_user=13,
        d_global=obs_dim - 13 * K,
        hidden_dim=128,
        n_heads=4,
        n_attn_layers=2,
        use_plackett_luce=True,
    ).to(dev)
    q1, q2 = Critic(obs_dim + act_dim).to(dev), Critic(obs_dim + act_dim).to(dev)
    q1t, q2t = Critic(obs_dim + act_dim).to(dev), Critic(obs_dim + act_dim).to(dev)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    log_alpha = torch.tensor(np.log(args.alpha), requires_grad=True, device=dev)
    opt_pol = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)
    opt_a = torch.optim.Adam([log_alpha], lr=args.lr)
    target_entropy = -0.3 * act_dim

    buf_s = np.zeros((args.buffer, obs_dim), np.float32)
    buf_a = np.zeros((args.buffer, act_dim), np.float32)
    buf_r = np.zeros((args.buffer,), np.float32)
    buf_s2 = np.zeros((args.buffer, obs_dim), np.float32)
    buf_d = np.zeros((args.buffer,), np.float32)
    ptr, n_buf = 0, 0

    def sample_action(o: np.ndarray):
        with torch.no_grad():
            cont, order, logp = policy.act(torch.as_tensor(o[None, :], device=dev))[:3]
        full = np.concatenate([cont.cpu().numpy()[0], order.cpu().numpy()[0].astype(np.float32)])
        return full, float(logp.cpu().numpy()[0])

    updates = 0
    for ep in range(args.episodes):
        env, obs = make_env(args.seed + ep, args.snr_lo, args.snr_hi, np_rng)
        ep_ret = 0.0
        for t in range(args.T):
            action, _lp = sample_action(obs)
            obs2, reward, term, trunc, info = env.step(action)
            obs2 = np.asarray(obs2, dtype=np.float32)
            done = bool(term or trunc) or (t == args.T - 1)
            buf_s[ptr] = obs
            buf_a[ptr] = action
            buf_r[ptr] = reward
            buf_s2[ptr] = obs2
            buf_d[ptr] = float(done)
            ptr = (ptr + 1) % args.buffer
            n_buf = min(n_buf + 1, args.buffer)
            ep_ret += reward
            obs = obs2

            if n_buf >= 2000 and t % args.update_every == 0:
                idx = torch.randint(0, n_buf, (args.batch,))
                s = torch.as_tensor(buf_s[idx], device=dev)
                a = torch.as_tensor(buf_a[idx], device=dev)
                r = torch.as_tensor(buf_r[idx], device=dev)
                s2 = torch.as_tensor(buf_s2[idx], device=dev)
                d = torch.as_tensor(buf_d[idx], device=dev)
                with torch.no_grad():
                    cont2, order2, logp2 = policy.act(s2)[:3]
                    a2 = torch.cat([cont2, order2.float()], dim=-1)
                    if args.no_perturb:
                        worst = torch.min(q1t(s2, a2), q2t(s2, a2))  # plain SAC twin-Q target
                    else:
                        worst = None
                        for m in range(3):
                            s2m = perturb_obs(s2, K, pert_rng, device=dev) if m else s2
                            qt = torch.min(q1t(s2m, a2), q2t(s2m, a2))
                            worst = qt if worst is None else torch.min(worst, qt)
                    alpha = log_alpha.exp()
                    target = r + args.gamma * (1 - d) * (worst - alpha * logp2)
                lq = ((q1(s, a) - target) ** 2).mean() + ((q2(s, a) - target) ** 2).mean()
                opt_q.zero_grad()
                lq.backward()
                opt_q.step()

                cont_s, order_s, logp_s = policy.act(s)[:3]
                a_s = torch.cat([cont_s, order_s.float()], dim=-1)
                if args.no_perturb:
                    worst_s = torch.min(q1(s, a_s), q2(s, a_s))  # plain SAC actor
                else:
                    worst_s = None
                    for m in range(3):
                        sm = perturb_obs(s, K, pert_rng, device=dev) if m else s
                        qs = q1(sm, a_s)
                        worst_s = qs if worst_s is None else torch.min(worst_s, qs)
                alpha = log_alpha.exp()
                lp = torch.cat([logp_s], dim=0) if logp_s.dim() else logp_s
                lpolicy = (alpha * lp - worst_s).mean()
                opt_pol.zero_grad()
                lpolicy.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt_pol.step()

                la = -(log_alpha.exp() * (lp.detach() + target_entropy)).mean()
                opt_a.zero_grad()
                la.backward()
                opt_a.step()

                with torch.no_grad():
                    for p, pt in zip(q1.parameters(), q1t.parameters()):
                        pt.mul_(0.995)
                        pt.add_(p.detach(), alpha=0.005)
                    for p, pt in zip(q2.parameters(), q2t.parameters()):
                        pt.mul_(0.995)
                        pt.add_(p.detach(), alpha=0.005)
                updates += 1

        if ep % 500 == 0 or ep == args.episodes - 1:
            deliv = env.traffic.get_delivery_ratio()
            with meta_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"update": updates, "episode": ep, "mean_reward": ep_ret / args.T, "delivery": float(deliv)}
                    )
                    + "\n"
                )
            print(f"[DRSAC] ep={ep} ret/slot={ep_ret / args.T:.3f} deliv={deliv:.3f} updates={updates}", flush=True)
            torch.save(
                {"model_state": policy.state_dict(), "obs_dim": obs_dim, "episode": ep, "delivery": float(deliv)},
                log_dir / "final.pt",
            )
    print(f"[DRSAC] done. ckpt={log_dir / 'final.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
