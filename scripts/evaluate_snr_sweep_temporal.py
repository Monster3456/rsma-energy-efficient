"""时序历史窗口检查点评估包装器（薄封装，零口径复制）

问题背景: 不可变的 scripts/evaluate_snr_sweep.py 在重建 ActorCriticAttn 时，
以 `config.ppo.worker_d_user`（默认 13）为基准推断检查点结构：
    d_user_eff  = user_embed.0.weight.shape[1]          # 从 checkpoint 推断
    d_user      = config.ppo.worker_d_user              # 未打补丁时恒为 13
    n_inject    = d_user_eff - d_user                   # 差值被误判为全局注入维
    ckpt_obs_dim = K * d_user + d_global
对 obs_history_window=H 训练出的检查点（真实 d_user = 13 + H·F，user_embed
输入维 = 13 + H·F + 2），未打补丁的 config 会把 H·F 全部误判进 n_inject，
得到 ckpt_obs_dim = K·13+3（如 315）而非真实 obs_dim = K·(13+H·F)+3（如 603），
model.act() 因维度错配崩溃。

解决方案: 本包装器在执行不可变脚本之前，把 src.config.Config 替换为一个子类
（__init__ 调 super() 后仅做实例级覆盖: system.obs_history_window /
system.obs_history_features / ppo.worker_d_user），再用 runpy 以 __main__
原样执行评估脚本。评估的全部指标、表格与口径 100% 来自不可变脚本本身，
本文件不含任何指标逻辑；非时序参数（--checkpoint/--n_episodes/--snr_subset/
--baselines/--csi_mode/--force_alpha_zero 等）原样转发。
"""

import argparse
import runpy
import sys
from pathlib import Path

# 仓库根目录入 sys.path（本脚本以 `python scripts/evaluate_snr_sweep_temporal.py`
# 运行时 sys.path[0] 是 scripts/，需显式补上仓库根；镜像 evaluate_snr_sweep.py
# 顶部的同款路径修复）
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 必须在打补丁之前导入 src.config（runpy 执行脚本时其
# `from src.config import Config` 会命中 sys.modules 中已补丁的模块属性）
import src.config

# 模块导入期捕获原始 Config 类：工厂可重复调用且不受后续补丁影响
_ORIGINAL_CONFIG = src.config.Config

# 不可变评估脚本路径（相对本文件解析，不从 cwd 出发）
EVAL_SCRIPT = Path(__file__).resolve().parent / "evaluate_snr_sweep.py"

DEFAULT_WINDOW = 4
DEFAULT_FEATURES = "h_gains,queue,power"


def make_temporal_config_cls(window: int, features: tuple, power_free: bool = False,
                             n_r: int | None = None, fixed_lambda: float | None = None,
                             prune_constants: bool = False):
    """构造打补丁的 Config 子类。

    子配置（SystemConfig/PPOConfig 等）是 dataclass，实例化后在实例上
    mutate 属性即可；不要改 dataclass 类级默认值（那会污染所有 Config 用户）。

    Args:
        power_free: True 时同时关闭 use_softmax_power_share（训练用了
            --no-softmax-power-share 的检查点必须配套，否则评估环境会把
            p 块重新解释为 softmax 份额、总功率被钉回 ~24W）。
        n_r: 非空时覆盖 system.N_r（训练用了 --n_r 的检查点必须配套，
            否则评估环境按 N_r=16 重建信道维度不匹配）。注意 N_r 变化
            会同步改变 SDMA 基线，得分与默认口径不可横比。
        fixed_lambda: 非空时关闭 adaptive_lambda 并把 system.lambda_rate
            钉为该值 — 固定外生流量口径（文献惯例：到达为外生固定过程，
            不随信道容量自缩放）。物理层/参数改进在此口径下才会在
            delivery/queue 上可见；注意检查点是在自适应 λ 下训练的，
            结果属分布外泛化诊断，须如实披露。
        prune_constants: 训练用了 --obs-prune-constant-blocks 的检查点须加此
            flag（A-prune：评估环境同步移除观测中的 3 个常数原生块，
            worker_d_user 相应 -3）。

    Returns:
        (TemporalConfig, worker_d_user_eff): 子类与扩展后的 per-user 特征维
        （= 原始 worker_d_user + window * len(features) [-3 if prune]，不硬编码基值 13）。
    """
    base_d_user = int(_ORIGINAL_CONFIG().ppo.worker_d_user)
    d_user_eff = base_d_user + int(window) * len(features)
    if prune_constants:
        d_user_eff -= 3

    class TemporalConfig(_ORIGINAL_CONFIG):
        def __init__(self):
            super().__init__()
            self.system.obs_history_window = int(window)
            self.system.obs_history_features = tuple(features)
            self.system.obs_prune_constant_blocks = bool(prune_constants)
            self.ppo.worker_d_user = int(d_user_eff)
            if power_free:
                self.system.use_softmax_power_share = False
            if n_r is not None:
                self.system.N_r = int(n_r)
            if fixed_lambda is not None:
                self.system.adaptive_lambda = False
                self.system.lambda_rate = float(fixed_lambda)

    return TemporalConfig, d_user_eff


def parse_features(spec: str) -> tuple:
    """解析逗号分隔的历史特征名并校验（与 src.config 允许集一致）。"""
    features = tuple(f.strip() for f in spec.split(",") if f.strip())
    allowed = src.config.OBS_HISTORY_ALLOWED_FEATURES
    unknown = [f for f in features if f not in allowed]
    if unknown:
        raise ValueError(f"obs_history_features 含未知特征 {unknown}，允许值: {list(allowed)}")
    if not features:
        raise ValueError("obs_history_features 不能为空")
    if len(set(features)) != len(features):
        raise ValueError(f"obs_history_features 含重复项: {list(features)}")
    return features


def _extract_ckpt_arg(forwarded: list) -> str | None:
    """从转发参数中提取 --checkpoint 的值（两种写法），未提供返回 None。"""
    for i, arg in enumerate(forwarded):
        if arg == "--checkpoint" and i + 1 < len(forwarded):
            return forwarded[i + 1]
        if arg.startswith("--checkpoint="):
            return arg.split("=", 1)[1]
    return None


def preflight_check(ckpt_arg, d_user_eff: int) -> None:
    """检查点兼容性预检（仅诊断，不含任何评估逻辑）。

    本仓库训练端（src/training/main.py）构造 ActorCriticAttn 时
    inject_global_to_per_user 默认 True、n_global_inject=2（snr+eps 注入），
    故时序检查点 user_embed 输入维 = worker_d_user_eff + 2。这正是不可变脚本
    精确重建（无截断、无错位）的充要条件：
        d_user_eff(ckpt) == worker_d_user(补丁) + 2
    不满足时在此给出明确错误并退出，而不是让脚本深处爆 size-mismatch 追溯。
    非注意力检查点（无 user_embed 键）或文件不存在/不可读时跳过预检，
    交由不可变脚本的自然路径（CKPT-DIAG 告警 / MLP 分支）处理。
    """
    if ckpt_arg is None:
        return
    path = Path(ckpt_arg)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        return
    import torch

    try:
        try:
            # 快路径：纯张量 checkpoint 用安全加载；训练端产物含 optimizer
            # state/numpy 等非张量对象时退回 weights_only=False（与不可变脚本
            # 的加载约定一致，仅信任本仓库自产检查点）
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return  # 预检自身失败不阻断评估，交给脚本自然报错
    if not isinstance(ckpt, dict):
        return
    state = ckpt.get("worker_state_dict") or ckpt.get("model_state_dict") or ckpt
    if not isinstance(state, dict) or "user_embed.0.weight" not in state:
        return
    ckpt_in = int(state["user_embed.0.weight"].shape[1])
    expected_in = d_user_eff + 2  # n_global_inject=2 (snr, eps)
    if ckpt_in != expected_in:
        print(
            f"[TEMPORAL-EVAL] ERROR: 检查点与时序配置不匹配 — "
            f"user_embed 输入维 {ckpt_in} != worker_d_user({d_user_eff}) + "
            f"n_global_inject(2) = {expected_in}。\n"
            f"  请核对该检查点训练时使用的 --obs_history_window / "
            f"--obs_history_features；非时序检查点请直接运行 "
            f"scripts/evaluate_snr_sweep.py。",
            flush=True,
        )
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "时序历史窗口(obs_history_window)检查点评估包装器：先给 "
            "src.config.Config 打时序补丁，再原样运行不可变的 "
            "scripts/evaluate_snr_sweep.py（指标口径与后者完全一致）。"
            "以下未列出的参数均原样转发给评估脚本。"
        )
    )
    parser.add_argument(
        "--obs_history_window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"训练时使用的时序历史窗口长度 H（默认 {DEFAULT_WINDOW}，须与检查点一致）",
    )
    parser.add_argument(
        "--obs_history_features",
        type=str,
        default=DEFAULT_FEATURES,
        help=(f"逗号分隔的历史特征名，须与训练一致（默认 '{DEFAULT_FEATURES}'，允许值 h_gains/queue/power）"),
    )
    parser.add_argument(
        "--no-softmax-power-share",
        action="store_true",
        help="训练用了 --no-softmax-power-share 的检查点须加此 flag（评估环境同步关闭 softmax 功率份额，p 块按每用户绝对功率解释）",
    )
    parser.add_argument(
        "--n_r",
        type=int,
        default=None,
        help="训练用了 --n_r 的检查点须加此参数（评估环境同步覆盖 system.N_r）；注意 N_r 变化会同步改变 SDMA 基线，得分与默认口径不可横比",
    )
    parser.add_argument(
        "--fixed_lambda",
        type=float,
        default=None,
        help=(
            "固定外生流量口径：关闭 adaptive_lambda 并把 lambda_rate 钉为该值 "
            "（文献惯例，到达不随容量自缩放）。与默认口径（λ∝容量）不可横比；"
            "自适应 λ 下训练的检查点在此口径属分布外泛化诊断"
        ),
    )
    parser.add_argument(
        "--obs-prune-constant-blocks",
        action="store_true",
        help="训练用了 --obs_prune_constant_blocks 的检查点须加此 flag（A-prune：评估环境同步移除观测常数块，worker_d_user 相应 -3）",
    )
    wrapper_args, forwarded = parser.parse_known_args()

    if wrapper_args.obs_history_window < 1:
        parser.error("--obs_history_window 必须 >= 1（非时序检查点请直接运行 scripts/evaluate_snr_sweep.py）")
    if wrapper_args.n_r is not None and wrapper_args.n_r < 1:
        parser.error(f"--n_r 必须 >= 1，got {wrapper_args.n_r}")
    if wrapper_args.fixed_lambda is not None and wrapper_args.fixed_lambda <= 0:
        parser.error(f"--fixed_lambda 必须 > 0，got {wrapper_args.fixed_lambda}")
    try:
        features = parse_features(wrapper_args.obs_history_features)
    except ValueError as e:
        parser.error(str(e))

    temporal_cls, d_user_eff = make_temporal_config_cls(
        wrapper_args.obs_history_window, features,
        power_free=wrapper_args.no_softmax_power_share,
        n_r=wrapper_args.n_r,
        fixed_lambda=wrapper_args.fixed_lambda,
        prune_constants=wrapper_args.obs_prune_constant_blocks,
    )
    base_d_user = d_user_eff - wrapper_args.obs_history_window * len(features)
    print(
        f"[TEMPORAL-EVAL] obs_history_window={wrapper_args.obs_history_window}, "
        f"features={list(features)} -> "
        f"worker_d_user: {base_d_user} -> {d_user_eff} "
        f"(user_embed 输入维 {d_user_eff + 2})"
        + (" | power_free: use_softmax_power_share=False" if wrapper_args.no_softmax_power_share else "")
        + (f" | N_r={wrapper_args.n_r}" if wrapper_args.n_r is not None else "")
        + (f" | fixed_lambda={wrapper_args.fixed_lambda:g} (adaptive_lambda=False, 固定外生流量口径)"
           if wrapper_args.fixed_lambda is not None else ""),
        flush=True,
    )

    preflight_check(_extract_ckpt_arg(forwarded), d_user_eff)

    # 关键补丁：替换模块属性后，runpy 执行脚本时其
    # `from src.config import Config`（脚本第 22 行，模块级导入）取到本子类，
    # build_config_for_snr() 的每个 Config() 实例都带时序配置。
    src.config.Config = temporal_cls

    # 以 __main__ 身份原样执行不可变脚本；argv 重写为去掉包装器专有参数后的
    # 转发参数，脚本内 argparse（位于 main()，由 __main__ 守卫触发）照常解析。
    sys.argv = [str(EVAL_SCRIPT)] + forwarded
    runpy.run_path(str(EVAL_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
