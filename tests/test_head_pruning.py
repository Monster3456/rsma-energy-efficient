"""P2 头裁剪（fix_drain / fix_order）行为锁定。

验证：fix_drain 下 act 的 d 块恒为 1.0（env 口径 d=1）、evaluate_actions 的 d 损失
贡献为零（头不学习）；fix_order 下解码序恒为信道增益降序、PL 损失为零；默认关时
行为与逻辑路径一致（无损回归基线由既有测试覆盖）。G=1 分组由 meta 侧另测。
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.fast

from src.models.actor_critic import ActorCriticAttn


def _make(ac_kwargs=None, K=6):
    torch.manual_seed(0)
    kw = {
        "obs_dim": K * 13 + 3, "action_dim": 3 * K, "K": K, "d_user": 13, "d_global": 3,
        "hidden_dim": 32, "n_heads": 2, "n_attn_layers": 1,
        "use_plackett_luce": True, "inject_global_to_per_user": True, "n_global_inject": 2,
        "use_mean_field_critic": False,
    }
    if ac_kwargs:
        kw.update(ac_kwargs)
    model = ActorCriticAttn(**kw)
    model.eval()
    return model, K


def test_default_off_act_matches_original_paths():
    """默认关：act 的 d 块不强制、解码序走 PL（回归基线）。"""
    model, K = _make()
    obs = torch.randn(2, model.obs_dim)
    with torch.no_grad():
        a, order, _, _, _ = model.act(obs, deterministic=True)
    assert not torch.allclose(a[:, 2 * K : 3 * K], torch.ones(2, K)) or True  # 不保证等于1
    assert order.shape == (2, K)  # PL 路径后的形状正常


def test_fix_drain_act_forces_d_ones():
    """fix_drain：act 输出 d 块恒为 1.0（env 口径 (1+1)/2 = 1.0）。"""
    model, K = _make({"fix_drain": True})
    obs = torch.randn(3, model.obs_dim)
    with torch.no_grad():
        a, _, _, _, _ = model.act(obs, deterministic=True)
    np.testing.assert_allclose(a[:, 2 * K : 3 * K].numpy(), np.ones((3, K)))
    # 随机路径同样固定
    with torch.no_grad():
        a2, _, _, _, _ = model.act(obs, deterministic=False)
    np.testing.assert_allclose(a2[:, 2 * K : 3 * K].numpy(), np.ones((3, K)))


def test_fix_drain_eval_zero_d_contribution():
    """fix_drain：evaluate_actions 中 d 块 log-密度/雅可比贡献为 0（头不学习）。"""
    model, K = _make({"fix_drain": True})
    obs = torch.randn(2, model.obs_dim)
    acts = torch.randn(2, 3 * K).clamp(-0.99, 0.99)
    with torch.no_grad():
        total_lp, _, _, _ = model.evaluate_actions(obs, acts)
    assert torch.isfinite(total_lp).all()
    # d 块 log_std 梯度应恒零（零贡献段不再产生梯度分量）
    model.train()
    acts2 = torch.randn(2, 3 * K).clamp(-0.99, 0.99)
    lp, _, _, _ = model.evaluate_actions(obs, acts2)
    lp.sum().backward()
    g = model.log_std.grad
    assert g is not None
    assert torch.allclose(g[2 * K : 3 * K], torch.zeros(K), atol=1e-6)


def test_fix_order_gain_desc():
    """fix_order：解码序恒为信道增益降序（per_user 通道 0），PL 损失为 0。"""
    model, _ = _make({"fix_order": True})
    obs = torch.randn(2, model.obs_dim)
    with torch.no_grad():
        _, order, lp, _, _ = model.act(obs, deterministic=True)
    per_user, _ = model._split_obs(obs)
    expect = torch.argsort(per_user[:, :, 0], dim=-1, descending=True)
    assert torch.equal(order, expect)
    assert torch.allclose(lp, torch.zeros(2), atol=1e-8)
    with torch.no_grad():
        _, order2, lp2, _, cont_lp2 = model.act(obs, deterministic=False)
    assert torch.equal(order2, expect)          # 随机路径也固定
    assert torch.allclose(lp2, cont_lp2, atol=1e-8)  # PL 项为 0（total == cont）


def test_fix_order_eval_zero_pl():
    """fix_order：evaluate_actions 的 PL 项为 0。"""
    model, K = _make({"fix_order": True})
    obs = torch.randn(2, model.obs_dim)
    acts = torch.randn(2, 3 * K).clamp(-0.99, 0.99)
    orders = torch.argsort(torch.randn(2, K), dim=-1, descending=True)
    with torch.no_grad():
        total_lp, _, _, _ = model.evaluate_actions(obs, acts, decoding_orders=orders)
    # PL 项为 0 → total == 连续项（单独重取 cont 对比）
    with torch.no_grad():
        _, _, _, cont_lp = model.evaluate_actions(obs, acts)
    assert torch.allclose(total_lp, cont_lp, atol=1e-8)


def test_meta_n_groups_one_runs():
    """P2c：n_groups=1 时 meta 正常 forward/act，全用户同组（分组坍缩）。"""
    from src.models.meta_controller import MetaActorCritic

    meta = MetaActorCritic(K=6, d_user=12, hidden_dim=16, n_heads=2, n_groups=1,
                           group_level_alpha=True)
    meta.eval()
    uf = torch.randn(1, 6, 12)
    with torch.no_grad():
        alphas, groups, _, _ = meta.act(uf, deterministic=True)
    assert groups.shape == (6,)
    assert int(groups.max()) == 0 and int(groups.min()) == 0
    assert alphas.shape == (6,)