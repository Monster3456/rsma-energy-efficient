"""可微目标闭包：把策略「决策输出」镜像为可归因标量。

Worker 目标精确镜像 ``ActorCriticAttn.act`` 的采样前段（_actor_forward →
_head_forward → P3 biases → _backpressure_shift → tanh/softmax），在 PL
采样/argmax 之前截断；Meta 目标镜像 ``MetaActorCritic.act`` 的确定性路径。

选择口径（与 env._parse_action 一致）：
- p 块 = tanh 后当 softmax 分数，p_k = clip(K·softmax, 0, 1)（rsma_env.py:372-378）；
- d_ratio = clip((tanh(d_raw)+1)/2, 0, 1)；
- z (解码序) = softmax(softplus(z_raw)) 在 no_grad argmax 处的质量（决策的「首用户」置信）；
- Worker 的 α 输出在 group_level_alpha=True 下为死维（被 Meta split_mask 覆盖），不解释；
- Meta：α = α_min + (α_max−α_min)·sigmoid(raw)（_map_alpha），
  group = mean_K log-softmax(group_logits)[:, g]（贪心决策的原始 logits，无温度）。
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from src.analysis.features import ObsLayout

Tensor = torch.Tensor


def worker_targets(model, name: str, k: int | None = None) -> Callable[[Tensor], Tensor]:
    """构造 Worker 决策目标的求值闭包。

    Args:
        model: ActorCriticAttn（eval 模式；无 expert_heads 时走单 user_head）。
        name: 目标名 —— "P_total" | "p_k"(需 k) | "d_mean" | "z_top1"。
        k: p_k 目标的用户下标。

    Returns:
        fn(obs (B, obs_dim) tensor) -> (B,) 标量（可带 requires_grad 求梯度）。
    """
    if name == "P_total":

        def fn(obs):
            return _worker_power_share(model, obs).sum(dim=-1)

    elif name == "p_k":
        if k is None:
            raise ValueError("p_k 目标需要用户下标 k")
        kk = int(k)

        def fn(obs):
            return _worker_power_share(model, obs)[:, kk]

    elif name == "d_mean":

        def fn(obs):
            d_raw = _worker_raw_heads(model, obs)[2]
            d = (torch.tanh(d_raw) + 1.0) / 2.0
            return d.clamp(0.0, 1.0).mean(dim=-1)

    elif name == "z_top1":

        def fn(obs):
            z_raw = _worker_raw_heads(model, obs)[3]
            zs = torch.softmax(F.softplus(z_raw) + 1e-6, dim=-1)
            with torch.no_grad():
                idx = zs.argmax(dim=-1, keepdim=True)  # (B,1)
            return zs.gather(-1, idx).squeeze(-1)

    else:
        raise ValueError(f"未知 worker 目标名: {name}")
    return fn


def _worker_raw_heads(model, obs: Tensor):
    """镜像 act() 采样前段 → (alpha_raw, p_raw, d_raw, z_raw) 各 (B,K)。"""
    per_user, g = model._split_obs(obs)
    h = model._actor_forward(obs)
    head = model._head_forward(h)  # (B,K,4)
    a_bias = model.global_to_alpha_bias(g).unsqueeze(1)  # (B,1,1)
    p_bias = model.global_to_p_bias(g).unsqueeze(1)
    alpha_raw = head[..., 0] + a_bias.squeeze(-1)
    p_raw = head[..., 1] + p_bias.squeeze(-1)
    bp = model._backpressure_shift(per_user)
    if int(model.backpressure_mode_buf) in (2, 3):
        p_raw = bp + head[..., 1]
    else:
        p_raw = p_raw + bp
    d_raw = head[..., 2]
    z_raw = head[..., 3]
    return alpha_raw, p_raw, d_raw, z_raw


def _worker_power_share(model, obs: Tensor) -> Tensor:
    """env 口径的 per-user 功率份额 K·softmax(tanh(p_raw)) clip [0,1]（(B,K)）。"""
    _, p_raw, _, _ = _worker_raw_heads(model, obs)
    p_logits = torch.tanh(p_raw)
    share = model.K * torch.softmax(p_logits, dim=-1)
    return share.clamp(0.0, 1.0)


def meta_targets(meta, name: str, group_idx: int = 0) -> Callable[[Tensor], Tensor]:
    """构造 Meta 决策目标的求值闭包。

    Args:
        meta: MetaActorCritic（group_level_alpha=True）。
        name: "alpha_group0"（组 0 映射 α）| "alpha_avg"（G 组映射 α 均值）
              | "group_logp_G"（mean_K log-softmax(group_logits)[:, g], g=group_idx）。
        group_idx: group_logp_G 的组下标。

    Returns:
        fn(user_features (B,K,12) tensor) -> (B,) 标量。
    """
    if name == "alpha_group0":
        gi = 0

        def fn(u):
            return _meta_alphas(meta, u)[:, gi]

    elif name == "alpha_avg":

        def fn(u):
            return _meta_alphas(meta, u).mean(dim=-1)

    elif name.startswith("group_logp"):
        gi = int(group_idx)

        def fn(u):
            _, group_logits, _ = meta.forward(u)  # (B,K,G)
            return F.log_softmax(group_logits, dim=-1)[:, :, gi].mean(dim=-1)

    else:
        raise ValueError(f"未知 meta 目标名: {name}")
    return fn


def _meta_alphas(meta, u: Tensor) -> Tensor:
    """映射到 [α_min, α_max] 的组级 α（(B,G)），镜像 act() 确定性路径。"""
    split_or_alpha, _, _ = meta.forward(u)  # (B,G) raw alpha means
    raw = split_or_alpha
    mapped = meta.group_alpha_min + meta._alpha_span * torch.sigmoid(raw)
    return mapped


def smoke_grad_flow(model, layout: ObsLayout) -> dict:
    """冒烟检查：随机噪声 obs 上各 Worker 目标的梯度有限且非零。"""
    torch.manual_seed(0)
    obs = torch.randn(4, layout.obs_dim(), requires_grad=True)
    out = {}
    for name, k in (("P_total", None), ("d_mean", None), ("z_top1", None), ("p_k", 3)):
        fn = worker_targets(model, name, k=k)
        val = fn(obs).sum()
        val.backward(retain_graph=True)
        g = obs.grad.clone()
        out[name] = {
            "finite": bool(torch.isfinite(g).all()),
            "max_abs": float(g.abs().max()),
        }
        obs.grad = None
    return out
