"""TD3 / DDPG-family continuous-control comparators for the RSMA-EE instance (supplement).

Same comparator protocol as LDPTRLQ / DR-SAC: single-timescale, no hierarchy, no
distillation, no guard. Because the policy is a stochastic-Plackett-Luce + Gaussian
continuous head, a textbook deterministic actor is not available verbatim; we adapt
the classic family honestly:

  - critics: twin-Q TD3-style (min of two) with soft target nets; --algo ddpg uses
    a single critic (no target-policy smoothing);
  - target actions: policy.deterministic act (tanh(mean) continuous + greedy PL
    order, order detached), plus clipped Gaussian smoothing noise on the
    continuous block for TD3 (target-policy smoothing).
  - actor: pathwise gradient -Q(s, pi_det(s)) through the differentiable
    continuous mean; PL order contributes to the action but is detached (greedy,
    non-differentiable), i.e. the actor learns the power/drain allocation while
    the order is served greedily — the same division of labour the partitioner+PL
    head enforces anyway.
  - TD3 delayed policy update (every 2 critic updates); DDPG updates actor every
    step.

Usage:
  PYTHONPATH=. python scripts/baselines/train_td3ddpg.py --algo td3 --exp_name comp_td3 \
      --episodes 8000 --seed 42 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.baselines.train_drsac import Critic, make_env
from src.models.actor_critic import ActorCriticAttn

POLICY_UPDATE_EVERY = 2  # TD3 delayed actor updates


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", choices=["td3", "ddpg"], default="td3")
    ap.add_argument("--exp_name", default="comp_td3")
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snr_lo", type=float, default=-5.0)
    ap.add_argument("--snr_hi", type=float, default=35.0)
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--update_every", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--noise_std", type=float, default=0.15, help="exploration action noise")
    ap.add_argument("--smooth_noise", type=float, default=0.1, help="TD3 target-policy smoothing std")
    ap.add_argument("--smooth_clip", type=float, default=0.2)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    twin = args.algo == "td3"

    log_dir = Path("logs") / args.exp_name / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path("logs") / args.exp_name / "checkpoint_meta.jsonl"

    env, obs = make_env(args.seed, args.snr_lo, args.snr_hi, np_rng)
    K = env.K
    obs_dim = int(env.obs_dim)
    act_dim = 4 * K
    policy = ActorCriticAttn(
        obs_dim=obs_dim, action_dim=act_dim, K=K, d_user=13,
        d_global=obs_dim - 13 * K, hidden_dim=128, n_heads=4, n_attn_layers=2,
        use_plackett_luce=True,
    ).to(dev)
    policy_target = ActorCriticAttn(
        obs_dim=obs_dim, action_dim=act_dim, K=K, d_user=13,
        d_global=obs_dim - 13 * K, hidden_dim=128, n_heads=4, n_attn_layers=2,
        use_plackett_luce=True,
    ).to(dev)
    policy_target.load_state_dict(policy.state_dict())

    q1 = Critic(obs_dim + act_dim).to(dev)
    q1t = Critic(obs_dim + act_dim).to(dev)
    q1t.load_state_dict(q1.state_dict())
    q_targets = [(q1, q1t)]
    if twin:
        q2 = Critic(obs_dim + act_dim).to(dev)
        q2t = Critic(obs_dim + act_dim).to(dev)
        q2t.load_state_dict(q2.state_dict())
        q_targets.append((q2, q2t))

    opt_q = torch.optim.Adam(
        [p for pair in q_targets for p in pair[0].parameters()], lr=args.lr)
    opt_pol = torch.optim.Adam(policy.parameters(), lr=args.lr)

    buf_s = np.zeros((args.buffer, obs_dim), np.float32)
    buf_a = np.zeros((args.buffer, act_dim), np.float32)
    buf_r = np.zeros((args.buffer,), np.float32)
    buf_s2 = np.zeros((args.buffer, obs_dim), np.float32)
    buf_d = np.zeros((args.buffer,), np.float32)
    ptr, n_buf = 0, 0

    def det_action_tensor(o):
        """可微的确定性动作：连续 = tanh(mean)（有梯度），PL order = greedy（detach）。"""
        cont, order, *_ = policy.act(torch.as_tensor(o, device=dev), deterministic=True)
        return torch.cat([cont, order.detach().float()], dim=-1)

    @torch.no_grad()
    def target_action(s2):
        cont2, order2, *_ = policy_target.act(s2, deterministic=True)
        noise = torch.randn_like(cont2) * args.smooth_noise
        if twin:
            noise = torch.clamp(noise, -args.smooth_clip, args.smooth_clip)
        return torch.cat([cont2 + noise, order2.float()], dim=-1)

    def sample_action(o: np.ndarray):
        # 探索：随机采样 + 连续块加高斯噪声（行为策略；训练目标用确定性目标动作）
        with torch.no_grad():
            cont, order, _ = policy.act(torch.as_tensor(o[None, :], device=dev))[:3]
        cont = cont.cpu().numpy()[0] + np_rng.normal(0, args.noise_std)
        return np.concatenate([cont, order.cpu().numpy()[0].astype(np.float32)])

    updates = 0
    for ep in range(args.episodes):
        env, obs = make_env(args.seed + ep, args.snr_lo, args.snr_hi, np_rng)
        ep_ret = 0.0
        for t in range(args.T):
            action = sample_action(obs)
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

            if n_buf >= 2000 and t % args.update_every == 0 and n_buf >= args.batch:
                idx = torch.randint(0, n_buf, (args.batch,))
                s = torch.as_tensor(buf_s[idx], device=dev)
                a = torch.as_tensor(buf_a[idx], device=dev)
                r = torch.as_tensor(buf_r[idx], device=dev)
                s2 = torch.as_tensor(buf_s2[idx], device=dev)
                d = torch.as_tensor(buf_d[idx], device=dev)
                a2 = target_action(s2)
                qs = [q(s2, a2) for q, _ in q_targets]
                q_min = torch.min(torch.stack(qs), dim=0).values
                target = r + args.gamma * (1 - d) * q_min
                lq = sum(((q(s, a) - target) ** 2).mean() for q, _ in q_targets)
                opt_q.zero_grad()
                lq.backward()
                opt_q.step()

                if twin and updates % POLICY_UPDATE_EVERY != 0:
                    # TD3 delayed actor update
                    updates += 1
                    with torch.no_grad():
                        for (q_, qt_) in q_targets:
                            for p, pt in zip(q_.parameters(), qt_.parameters()):
                                pt.mul_(1 - args.tau).add_(p.detach(), alpha=args.tau)
                    continue

                a_pi = det_action_tensor(s)
                lpol = -q1(s, a_pi).mean()
                opt_pol.zero_grad()
                lpol.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt_pol.step()

                with torch.no_grad():
                    for p, pt in zip(policy.parameters(), policy_target.parameters()):
                        pt.mul_(1 - args.tau).add_(p.detach(), alpha=args.tau)
                    for (q_, qt_) in q_targets:
                        for p, pt in zip(q_.parameters(), qt_.parameters()):
                            pt.mul_(1 - args.tau).add_(p.detach(), alpha=args.tau)
                updates += 1

        if ep % 500 == 0 or ep == args.episodes - 1:
            deliv = env.traffic.get_delivery_ratio()
            with meta_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(
                    {"algo": args.algo, "update": updates, "episode": ep,
                     "mean_reward": ep_ret / args.T, "delivery": float(deliv)}) + "\n")
            print(f"[{args.algo.upper()}] ep={ep} ret/slot={ep_ret / args.T:.3f} "
                  f"deliv={deliv:.3f} updates={updates}", flush=True)
            torch.save({"model_state": policy.state_dict(),
                        "worker_state_dict": policy.state_dict(),  # 标准评估管道键
                        "obs_dim": obs_dim, "action_dim": act_dim,
                        "episode": ep, "delivery": float(deliv)},
                       log_dir / "final.pt")
    print(f"[{args.algo.upper()}] done. ckpt={log_dir / 'final.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())