"""A-prune 专用：从冠军 rollout 自蒸馏出剪枝观测(531)版的 BC-init。

背景（ablation_prune_constants.md）：A-prune 必须与 A-ctrl 共享同一强度的
BC 起点，但剪枝观测变体无法直接加载冠军的 27 维 user_embed。本脚本：
1) 用冠军确定性策略（+Meta）在未剪 env（603 观测）rollout N 个 episode，
   记录 (obs, 连续动作, PL 解码序)；
2) 把 obs 按 A-prune 语义去掉块 3/6/10（= 剪枝 env 的观测，有测试对齐保证）；
3) 在剪枝观测上 BC 预训练一个全新的 Worker（d_user=22，架构对齐冠军配方）；
   Meta 复用冠军 meta_state_dict（meta 特征空间 12 维独立于 worker 观测剪枝）。

输出 logs/intp_A_prune/checkpoints/bc_init.pt（结构对齐 main.py 保存格式，
可直接 --resume 续跑 SFTP 微调）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.models import (  # noqa: E402
    build_env,
    load_meta,
    load_worker,
    make_temporal_config_cls,
)
from src.config import Config  # noqa: E402
from src.env.rsma_env import RSMAEnergyEnv  # noqa: E402
from src.models.actor_critic import ActorCriticAttn  # noqa: E402
from src.training.ppo import PPOTrainer  # noqa: E402

CHAMPION_CKPT = "logs/m5_armA_sftp/checkpoints/best_guard.pt"
PRUNED_NATIVE_IDX = (3, 6, 10)


def _prune_obs(obs: np.ndarray, K: int, d_user_full: int) -> np.ndarray:
    """603 观测 → 531：丢原生块 3/6/10（与剪枝 env 逐位一致，见 test_obs_prune）。"""
    o = np.asarray(obs, dtype=np.float64)
    per_user = o[: K * d_user_full].reshape(d_user_full, K)
    keep = [j for j in range(d_user_full) if j not in PRUNED_NATIVE_IDX]
    return np.concatenate([per_user[keep].reshape(-1), o[K * d_user_full :]])


def collect_rollouts(worker, meta_ac, t_s, n_episodes, seed, env_full):
    obs_list, act_list, order_list = [], [], []
    for ep in range(n_episodes):
        obs, _ = env_full.reset(seed=int(seed) + ep)
        uf = torch.from_numpy(env_full.get_user_features().copy()).float().unsqueeze(0)
        uf = uf.to(next(meta_ac.parameters()).device)
        with torch.no_grad():
            primary, group_ids, _, _ = meta_ac.act(uf, deterministic=True)
        env_full.set_split_mask(primary.detach().cpu().numpy().astype(np.float32))
        env_full.set_group_ids(group_ids.detach().cpu().numpy().astype(np.int64))
        for _ in range(env_full.T):
            obs_list.append(np.asarray(obs, dtype=np.float32))
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(next(worker.parameters()).device)
            with torch.no_grad():
                res = worker.act(obs_t, deterministic=True)
                acont = res[0].squeeze(0).cpu().numpy()
                order = res[1].squeeze(0).cpu().numpy()
            action = np.concatenate([acont, order.astype(np.float64)])
            obs, _, term, trunc, _ = env_full.step(action)
            act_list.append(acont.astype(np.float32))
            order_list.append(order.astype(np.int64))
            if term or trunc:
                break
    return (np.stack(obs_list), np.stack(act_list), np.stack(order_list))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200, help="冠军 rollout 的 episode 数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_groups", type=int, default=4,
                   help="P2c：meta 分组数（默认 4=冠军；G≠4 时用随机初始化 meta，不继承冠军 G=4 权重）")
    p.add_argument("--checkpoint", default=CHAMPION_CKPT)
    p.add_argument("--bc-epochs", type=int, default=60)
    p.add_argument("--out", default="logs/intp_A_prune")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    t0 = time.time()
    Path(args.out, "checkpoints").mkdir(parents=True, exist_ok=True)

    # 1) 收集 env（未剪，603 观测）加载冠军 worker+meta；多 SNR 点采集保证覆盖。
    #    无论是否复用演示缓存，worker/meta 都必须加载（meta_state_dict 要进 bc_init.pt）。
    cfg_cls_full = make_temporal_config_cls(4, ("h_gains", "queue", "power"), prune_constants=False)
    from src.analysis.features import resolve_layout

    layout_full = resolve_layout(cfg_cls_full())
    env_full, _ = build_env(cfg_cls_full, 20.0, 0.01, args.seed)
    worker = load_worker(args.checkpoint, env_full)
    if int(args.n_groups) == 4:
        meta_ac, t_s = load_meta(args.checkpoint, env_full)
    else:
        # P2c（G≠4）：冠军 meta 权重形状不匹配，随机初始化 G 分组 meta
        from src.models.meta_controller import MetaActorCritic

        cfg_m = cfg_cls_full().hierarchical
        meta_ac = MetaActorCritic(
            K=env_full.K, d_user=cfg_m.meta_d_user, hidden_dim=cfg_m.meta_hidden_dim,
            n_heads=cfg_m.meta_n_heads, n_groups=int(args.n_groups),
            group_level_alpha=cfg_m.group_level_alpha,
            group_alpha_min=cfg_m.group_alpha_min, group_alpha_max=cfg_m.group_alpha_max,
        )
        t_s = cfg_m.T_s
        print(f"[BC-init] P2c: meta 随机初始化 n_groups={args.n_groups}（不继承冠军 G=4）", flush=True)
    worker = worker.to(args.device)
    meta_ac = meta_ac.to(args.device)

    snr_grid = [(0, 1.0), (10, 0.1), (20, 0.01), (30, 0.001)]
    per_snr = max(1, args.episodes // len(snr_grid))
    demo_cache = Path(args.out, "demos.npz")
    if demo_cache.exists():
        print(f"[BC-init] 复用演示缓存 {demo_cache}", flush=True)
        z = np.load(demo_cache)
        obs_np, act_np, order_np = z["obs"], z["act"], z["order"]
    else:
        obs_list, act_list, order_list = [], [], []
        for snr_db, sigma2 in snr_grid:
            env_full, _ = build_env(cfg_cls_full, snr_db, sigma2, args.seed)
            o, a, od = collect_rollouts(worker, meta_ac, t_s, per_snr, args.seed + snr_db, env_full)
            obs_list.append(o)
            act_list.append(a)
            order_list.append(od)
            print(f"[BC-init] SNR {snr_db:>2} dB: {o.shape[0]} states", flush=True)
        obs_np = np.concatenate(obs_list)
        act_np = np.concatenate(act_list)
        order_np = np.concatenate(order_list)
        np.savez_compressed(demo_cache, obs=obs_np, act=act_np, order=order_np)
        print(f"[BC-init] collected {obs_np.shape[0]} states ({per_snr}x{len(snr_grid)} ep) "
              f"in {time.time()-t0:.0f}s -> {demo_cache}", flush=True)

    # 2) obs 剪枝 → 531
    K, d_user_full = layout_full.K, layout_full.d_user_eff
    obs_pruned = np.stack([_prune_obs(o, K, d_user_full) for o in obs_np])
    print(f"[BC-init] obs {obs_np.shape[1]} -> {obs_pruned.shape[1]} (prune {PRUNED_NATIVE_IDX})", flush=True)

    # 3) 剪枝版 worker 构造（架构对齐冠军配方，d_user=22）+ BC
    cls_prune = make_temporal_config_cls(4, ("h_gains", "queue", "power"), prune_constants=True)
    cfg_prune = cls_prune()
    env_prune, _ = build_env(cls_prune, 20.0, 0.01, args.seed)
    prune_ac = ActorCriticAttn(
        obs_dim=int(env_prune.obs_dim),
        action_dim=4 * K,
        K=K,
        d_user=int(getattr(cfg_prune.ppo, "worker_d_user")),
        d_global=3,
        hidden_dim=cfg_prune.ppo.worker_attn_hidden,
        n_heads=cfg_prune.ppo.worker_attn_heads,
        n_attn_layers=cfg_prune.ppo.worker_attn_layers,
        use_plackett_luce=True,
        use_mean_field_critic=cfg_prune.ppo.use_mean_field_critic,
        mean_field_hidden_dim=cfg_prune.ppo.mean_field_hidden_dim,
        qos_drift_weights=tuple(cfg_prune.system.qos_drift_weights)
        if cfg_prune.ppo.use_mean_field_critic else None,
        use_bucket_experts=False,
        backpressure_beta=0.0,
        backpressure_mode="queue",
    ).to(args.device)
    cfg_prune.constraint.hek_bc_epochs = args.bc_epochs
    cfg_prune.constraint.hek_bc_lr = 3e-4
    trainer = PPOTrainer(prune_ac, cfg_prune.ppo)
    bc_info = trainer.bc_pretrain(obs_pruned, act_np, orders_np=order_np)
    print(f"[BC-init] worker BC done: loss={bc_info['bc_loss']:.4f} epochs={bc_info['bc_epochs']}", flush=True)

    # 4) 保存 bc_init.pt（worker=剪枝 BC 版；meta=冠军原版；结构对齐 main.py 保存）
    from src.models.meta_controller import MetaPPOTrainer

    meta_t = MetaPPOTrainer(meta_ac, cfg_prune.hierarchical)
    ckpt = {
        "episode": 0,
        "total_steps": 0,
        "update_count": 0,
        "meta_update_count": 0,
        "worker_state_dict": prune_ac.state_dict(),
        "meta_state_dict": meta_ac.state_dict(),
        "worker_optimizer": trainer.optimizer.state_dict(),
        "meta_optimizer": meta_t.optimizer.state_dict(),
        "config": vars(cfg_prune.ppo),
        "bc_stage": True,
    }
    out_path = Path(args.out, "checkpoints", "bc_init.pt")
    torch.save(ckpt, out_path)
    print(f"[BC-init] -> {out_path} (obs_dim={int(env_prune.obs_dim)}, worker_d_user={cfg_prune.ppo.worker_d_user}) "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()