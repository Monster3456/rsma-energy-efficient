"""v2 评估入口：挂 ASOPA solver + v2 Meta-Controller 后复用 evaluate_snr_sweep 全流程。

evaluate_snr_sweep.py 从 src.models.meta_controller 导入 MetaActorCritic；
v2 checkpoint 的 meta_state_dict 含 bucket_alpha_head.* 权重，必须替换为
v2 MetaActorCritic（use_bucket_meta_alpha=True 默认）才能加载。

同时挂载 _patch_env_step（ASOPA solver），确保训练/评估行为一致；
--gate-scale 0 时跳过 solver（用于评估无 solver 基线如 v2_primary）。

用法（与 evaluate_snr_sweep.py 相同参数）：
    python3 experiments/v2/evaluate.py --checkpoint logs/v2_powertax/checkpoints/final.pt
    python3 experiments/v2/evaluate.py --checkpoint logs/v2_primary/checkpoints/final.pt --gate-scale 0
"""
import sys
import os

sys.path.insert(0, '/Users/libo/PycharmProjects/rsma-energy-efficient')

import scripts.evaluate_snr_sweep as ev
from experiments.v2.models.meta_controller import MetaActorCritic as V2MetaAC
from experiments.v2.train import _patch_env_step

# 1. 替换 MetaActorCritic 为 v2 版本（per-bucket alpha 缩放）
ev.MetaActorCritic = V2MetaAC
print("[v2-eval] MetaActorCritic 已替换为 v2 (per-bucket alpha 缩放)", flush=True)

# 2. 解析 --gate-scale（默认 1.0；0 时跳过 solver，用于无 solver 基线）
gate_scale = 1.0
if '--gate-scale' in sys.argv:
    idx = sys.argv.index('--gate-scale')
    if idx + 1 < len(sys.argv):
        try:
            gate_scale = float(sys.argv[idx + 1])
        except ValueError:
            pass

# 3. 挂载 ASOPA solver env.step patch（与训练一致：steps=24 全覆盖, interval=20, outer=2）
_patch_env_step(gate_scale=gate_scale, n_solver_steps=24, solver_step_size=0.08,
                solver_interval=20, power_budget=14.0, n_solver_outer=2)
if gate_scale > 0:
    print("[v2-eval] ASOPA solver 已挂载 (gate_scale=1.0, steps=24全覆盖, interval=20, outer=2, power_budget=14W)", flush=True)
else:
    print("[v2-eval] solver 已跳过 (gate_scale=0, 纯 PPO 评估)", flush=True)


if __name__ == "__main__":
    sys.argv[0] = 'evaluate_v2'
    ev.main()