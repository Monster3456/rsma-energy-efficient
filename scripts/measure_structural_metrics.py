"""Arm A 结构指标测量（预注册 §6 三口径）。

对照常规模式（动作 4K）与精简模式（动作 2K）：
  1) 动作空间维度
  2) 可学习参数量
  3) 单时隙推理时延（≥1000 次前向，报均值±std）

用法（仓库根目录）：
  .venv\\Scripts\\python.exe scripts\\measure_structural_metrics.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.actor_critic import ActorCriticAttn

K = 24
# champion 口径：H=4 → d_user = 13 + 4*3 = 25；未剪 obs_dim = 25*24+3 = 603
D_USER = 25
OBS_DIM = 603


def build(simplified: bool) -> ActorCriticAttn:
    return ActorCriticAttn(
        obs_dim=OBS_DIM, action_dim=(1 if simplified else 3) * K + K, K=K,
        d_user=D_USER, d_global=OBS_DIM - K * D_USER,
        hidden_dim=128, n_heads=4, n_attn_layers=2,
        use_plackett_luce=True, simplified_action=simplified,
    )


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def latency(m: ActorCriticAttn, n_warm: int = 200, n_run: int = 1000) -> tuple[float, float]:
    m.eval()
    obs = torch.randn(1, OBS_DIM)
    with torch.no_grad():
        for _ in range(n_warm):
            m.act(obs, deterministic=True)
        ts = []
        for _ in range(n_run):
            t0 = time.perf_counter()
            m.act(obs, deterministic=True)
            ts.append((time.perf_counter() - t0) * 1e3)
    import statistics
    return statistics.mean(ts), statistics.pstdev(ts)


def main() -> int:
    torch.manual_seed(42)
    full, small = build(False), build(True)

    act_full = 3 * K + K
    act_small = 1 * K + K
    pf, ps = n_params(full), n_params(small)

    print("=== Arm A 结构指标（预注册 §6）===")
    print(f"{'口径':<12}{'动作维度':>10}{'可学习参数':>14}{'连续块':>10}")
    print(f"{'常规(4K)':<12}{act_full:>10}{pf:>14}{3 * K:>10}")
    print(f"{'精简(2K)':<12}{act_small:>10}{ps:>14}{1 * K:>10}")
    print(f"\n动作空间: {act_full} -> {act_small}  减少 {act_full - act_small}"
          f" ({(act_full - act_small) / act_full * 100:.1f}%)")
    print(f"参数量:   {pf} -> {ps}  减少 {pf - ps} ({(pf - ps) / pf * 100:.2f}%)")

    mf, sf = latency(full)
    ms, ss = latency(small)
    print(f"\n单时隙推理时延（{1000} 次前向均值±std，ms）：")
    print(f"  常规: {mf:.4f} ± {sf:.4f}")
    print(f"  精简: {ms:.4f} ± {ss:.4f}")
    print(f"  变化: {(ms - mf) / mf * 100:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
