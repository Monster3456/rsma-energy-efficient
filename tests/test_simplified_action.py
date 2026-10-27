"""Arm A 精简动作模式（10-23-hierarchical-opt）：Worker 只输出 (p, 解码序)，动作 4K→2K。

覆盖：
- env 接受 2K 动作，α 取 Meta split_mask、d_k≡1；
- ActorCriticAttn 精简模式输出 1K 连续块，参数量下降；
- act / evaluate_actions 形状自洽（PPO 训练可用）；
- **回滚安全**：默认关闭时与常规模式逐位一致。
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.fast

from src.config import Config
from src.env.rsma_env import RSMAEnergyEnv
from src.models.actor_critic import ActorCriticAttn


def _cfg(simplified: bool, **system_overrides) -> Config:
    cfg = Config()
    cfg.system.obs_history_window = 0
    cfg.system.obs_prune_constant_blocks = False
    cfg.ppo.worker_d_user = 13
    cfg.ppo.worker_simplified_action = bool(simplified)
    cfg.system.group_level_alpha = True
    for k, v in system_overrides.items():
        setattr(cfg.system, k, v)
    return cfg


def _env(simplified: bool) -> RSMAEnergyEnv:
    return RSMAEnergyEnv(_cfg(simplified))


def _actor(simplified: bool, K: int = 24, obs_dim: int = 315) -> ActorCriticAttn:
    return ActorCriticAttn(
        obs_dim=obs_dim, action_dim=(1 if simplified else 3) * K + K, K=K,
        d_user=13, d_global=3, hidden_dim=32, n_heads=2, n_attn_layers=1,
        use_plackett_luce=True, simplified_action=simplified,
    )


# ---------------------------------------------------------------- env --


def test_env_accepts_2k_action_and_pins_drain():
    """2K 动作 = [p(K), 解码序(K)]：d_k≡1、α 由 split_mask 给出。"""
    env = _env(simplified=True)
    K = env.K
    env.reset(seed=0)
    split = np.full(K, 0.35)
    env.set_split_mask(split.copy())
    p = np.full(K, 0.5)
    order = np.arange(K)
    obs, _r, _t, _tr, _info = env.step(np.concatenate([p, order.astype(np.float64)]))
    assert obs.shape == (env._obs_dim,)
    np.testing.assert_allclose(env.split_mask, split, atol=1e-12)  # α 不被 Worker 覆盖


def test_env_3k_action_still_accepted():
    """回滚安全：常规 3K/4K 路径不受影响。"""
    env = _env(simplified=False)
    K = env.K
    env.reset(seed=0)
    env.set_split_mask(np.full(K, 0.4))
    a3 = np.zeros(3 * K)
    a3[2 * K:] = 1.0  # d=1.0
    env.step(a3)
    a4 = np.concatenate([a3, np.arange(K).astype(np.float64)])
    env.step(a4)


def test_env_rejects_wrong_action_dim():
    env = _env(simplified=True)
    with pytest.raises(ValueError):
        env.step(np.zeros(5))


# ---------------------------------------------------------------- actor --


def test_simplified_actor_shapes():
    """精简模式：连续块 1K，act/evaluate_actions 自洽。"""
    K = 24
    ac = _actor(True, K=K)
    obs = torch.randn(2, 315)
    cont, order, _lp_c, _lp_pl, _val = ac.act(obs, deterministic=True)
    assert cont.shape == (2, K), f"连续块应为 1K，实际 {tuple(cont.shape)}"
    assert order.shape == (2, K)
    assert ac.continuous_dim == K
    assert ac.user_head.out_features == 2

    out = ac.evaluate_actions(obs, cont, decoding_orders=order)
    log_prob, value, entropy, _lp = out
    assert log_prob.shape == (2,)
    assert value.shape == (2,)
    assert entropy.shape == (2,)
    assert torch.isfinite(log_prob).all()


def test_regular_actor_shapes_unchanged():
    """常规模式仍是 3K 连续块（回滚安全）。"""
    K = 24
    ac = _actor(False, K=K)
    obs = torch.randn(2, 315)
    cont, _order, *_rest = ac.act(obs, deterministic=True)
    assert cont.shape == (2, 3 * K)
    assert ac.user_head.out_features == 4


def test_simplified_actor_fewer_parameters():
    """结构判据：精简模式可学习参数量下降。"""
    K = 24
    full = _actor(False, K=K)
    small = _actor(True, K=K)
    n_full = sum(p.numel() for p in full.parameters() if p.requires_grad)
    n_small = sum(p.numel() for p in small.parameters() if p.requires_grad)
    assert n_small < n_full, f"{n_small} 应小于 {n_full}"
    # 至少少掉 user_head 的 2 行 + alpha bias + log_std 的 2K
    assert n_full - n_small >= 2 * (32 + 1) + 2 * K


def test_default_flag_is_off():
    """默认关闭：Config 与模型都不启用精简。"""
    cfg = Config()
    assert cfg.ppo.worker_simplified_action is False
    ac = ActorCriticAttn(obs_dim=315, action_dim=96, K=24, d_user=13, d_global=3,
                         hidden_dim=32, n_heads=2, n_attn_layers=1)
    assert ac.simplified_action is False
    assert ac.continuous_dim == 72


def test_simplified_action_dim_matches_env():
    """训练侧动作维与 env 接受的维一致：2K = 1K 连续 + K 解码序。"""
    K = 24
    env = _env(simplified=True)
    ac = _actor(True, K=K)
    env.reset(seed=1)
    env.set_split_mask(np.full(K, 0.5))
    obs_t = torch.randn(1, env._obs_dim)
    cont, order, *_rest = ac.act(obs_t, deterministic=True)
    act = np.concatenate([
        cont.squeeze(0).detach().numpy(),
        order.squeeze(0).detach().numpy().astype(np.float64),
    ])
    assert act.shape[0] == 2 * K
    env.step(act)  # 不抛异常即通过


def test_simplified_with_bucket_experts():
    """回归锁定：桶专家 gather 的 head_dim 不能硬编码 4。

    曾经的 bug：_head_forward 里 ``.expand(1, batch, K, 4)`` 在精简模式（head_dim=2）
    下与专家输出的最后一维不匹配，训练第一步即崩。
    """
    K = 24
    ac = ActorCriticAttn(
        obs_dim=315, action_dim=(1 * K) + K, K=K, d_user=13, d_global=3,
        hidden_dim=32, n_heads=2, n_attn_layers=1, use_plackett_luce=True,
        simplified_action=True, use_bucket_experts=True, n_experts=4,
    )
    obs = torch.randn(3, 315)
    cont, order, *_ = ac.act(obs, deterministic=True)
    assert cont.shape == (3, K)
    assert ac.head_dim == 2
    out = ac.evaluate_actions(obs, cont, decoding_orders=order, bucket_indices=[0, 1, 2])
    assert out[0].shape == (3,)
    assert torch.isfinite(out[0]).all()


def test_solver_expert_simplified_cont_action():
    """回归锁定：SolverExpert 的 cont_action 必须与动作模式同维。

    曾经的 bug：cont_action 固定拼 (α, p, d)=3K=72，精简模式（1K=24）下与
    policy 输出不匹配，BC 收集时报
    "Value is not broadcastable with batch_shape+event_shape: [64,72] vs [64,24]"
    并把整个 SolverExpert 禁用（等于丢失蒸馏）。
    """
    from src.optimization.solver_expert import SolverExpert

    K = 24
    env = _env(simplified=True)
    env.reset(seed=2)
    env.set_split_mask(np.full(K, 0.5))

    exp_full = SolverExpert(env, n_steps=1, simplified_action=False)
    exp_small = SolverExpert(env, n_steps=1, simplified_action=True)

    r_full = exp_full.solve(env)
    r_small = exp_small.solve(env)

    assert r_full["cont_action"].shape == (3 * K,), f"{r_full['cont_action'].shape}"
    assert r_small["cont_action"].shape == (K,), f"{r_small['cont_action'].shape}"
    assert r_full["action"].shape == (4 * K,)
    assert r_small["action"].shape == (2 * K,)
    # env 接受两者
    env.step(r_full["action"])
    env.step(r_small["action"])
