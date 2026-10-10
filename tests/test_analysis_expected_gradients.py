"""Expected-Gradients（DeepSHAP 族）与参照构造的快速单元测试。

闭式真值：线性模型上 EG = w·(x − ref)（block 聚合后 = 块分量和）；Worker
目标闭包在随机权重网络上求梯度有限非零，且 P_total 与 act() 的功率份额数学一致。
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.fast

from src.analysis.expected_gradients import eg_blocks
from src.analysis.features import (
    ObsLayout,
    build_neutral_obs,
    ref_bootstrap,
    ref_means,
    ref_zero,
)
from src.analysis.targets import smoke_grad_flow, worker_targets


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_eg_linear_recovery():
    """线性模型闭式真值：EG 块归因 = 该块分量的 w·(x − ref) 之和。"""
    layout = ObsLayout(K=3, H=0)  # obs_dim = 3*13 + 3 = 42
    d = layout.obs_dim()
    rng = _rng(1)
    w = rng.normal(size=d)
    lin = torch.nn.Linear(d, 1, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(w))

    def fn(t):
        return lin(t).squeeze(-1)

    x = rng.normal(size=d)
    ref = rng.normal(size=d)
    phi = eg_blocks(fn, x, layout, ref, n_eg=400, seed=2)
    expected = np.concatenate(
        [(w * (x - ref))[j * 3 : (j + 1) * 3].sum(keepdims=True) for j in range(13)]
        + [(w * (x - ref))[-2:-1], (w * (x - ref))[-1:]]
    )
    np.testing.assert_allclose(phi, expected, atol=1e-2)


def test_reference_variants_shapes_ranges():
    """四参照的形状与值域（DP 文献双估计器的参照协议）。"""
    layout = ObsLayout(K=4, H=1, history_features=("h_gains", "queue", "power"))
    rng = _rng(3)
    stack = rng.uniform(size=(20, layout.obs_dim()))
    zero = ref_zero(layout)
    means = ref_means(stack, layout)
    boot = ref_bootstrap(stack, layout, rng, n=8)
    neutral = build_neutral_obs(layout)
    assert zero.shape == (layout.obs_dim(),) and (zero == 0).all()
    assert means.shape == (layout.obs_dim(),)
    np.testing.assert_allclose(means[: layout.K], stack[:, : layout.K].mean(), atol=1e-9)
    assert boot.shape == (8, layout.obs_dim())
    assert np.isin(boot, stack).all()
    assert neutral.shape == (layout.obs_dim(),)
    assert neutral.min() >= 0.0 and neutral.max() <= 1.0
    assert neutral[1 * layout.K] == pytest.approx(0.5)  # queue 半满
    assert neutral[4 * layout.K] == pytest.approx(0.0)  # power_history 无历史


def _make_worker_toy():
    from src.models.actor_critic import ActorCriticAttn

    torch.manual_seed(0)
    K, d_user, d_global = 4, 13, 3
    obs_dim = K * d_user + d_global
    model = ActorCriticAttn(
        obs_dim=obs_dim,
        action_dim=3 * K,
        K=K,
        d_user=d_user,
        d_global=d_global,
        hidden_dim=32,
        n_heads=2,
        n_attn_layers=1,
        use_plackett_luce=True,
        inject_global_to_per_user=True,
        n_global_inject=2,
        use_mean_field_critic=False,
    )
    model.eval()
    return model, ObsLayout(K=K, H=0)


def test_worker_targets_grad_flow():
    """P_total/d_mean/z_top1/p_k 梯度有限非零（随机权重网络冒烟）。"""
    model, layout = _make_worker_toy()
    grads = smoke_grad_flow(model, layout)
    for name, g in grads.items():
        assert g["finite"], f"{name} 梯度非有限"
        assert g["max_abs"] > 0.0, f"{name} 梯度为零"


def test_worker_p_total_matches_act():
    """P_total 目标与 act() 输出（tanh→softmax 份额）精确一致。"""

    model, layout = _make_worker_toy()
    obs = torch.randn(2, layout.obs_dim())
    fn = worker_targets(model, "P_total")
    got = fn(obs).detach().numpy()
    with torch.no_grad():
        acts = model.act(obs, deterministic=True)[0].numpy()  # (2, 3K)
    K = layout.K
    manual = []
    for b in range(acts.shape[0]):
        p_logits = acts[b, K : 2 * K]
        p_exp = np.exp(p_logits - p_logits.max())
        share = np.clip(K * p_exp / p_exp.sum(), 0.0, 1.0)
        manual.append(share.sum())
    np.testing.assert_allclose(got, manual, atol=1e-4)


def test_jaccard_recall_properties():
    """top-k 度量的对称性/自洽性（validation 模块）。"""
    from src.analysis.validation import jaccard_topk, recall_at_k

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    assert jaccard_topk(a, b, 3) == pytest.approx(jaccard_topk(b, a, 3))
    assert jaccard_topk(a, a, 5) == pytest.approx(1.0)
    assert recall_at_k(a, a, 4) == pytest.approx(1.0)
    assert 0.0 <= recall_at_k(a, b, 4) <= 1.0
