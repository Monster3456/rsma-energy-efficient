"""模型/环境装载（wrapper 模式，零改动不可变文件）。

评估口径与 ``scripts/evaluate_snr_sweep_temporal.py`` 一致：先以 Config 子类
patch 训练期参数（obs_history_window/H、worker_d_user），再构造 env 与模型。
模型装载委托给不可变评估脚本的 ``load_ppo_model`` / ``load_meta_model``
（strict=False、维度自推断），本模块只做断言与形态校验。
"""

from __future__ import annotations

from src.analysis.features import ObsLayout, resolve_layout

#: 评估口径的合同常量（champion 训练期默认）
DEFAULT_HISTORY_WINDOW = 4
DEFAULT_HISTORY_FEATURES = ("h_gains", "queue", "power")


def make_temporal_config_cls(
    window: int = DEFAULT_HISTORY_WINDOW,
    features: tuple[str, ...] = DEFAULT_HISTORY_FEATURES,
):
    """构造打补丁的 Config 子类（实例级 mutate，不改 dataclass 类默认值）。

    镜像 ``scripts/evaluate_snr_sweep_temporal.py:48``，供本分析复用。
    """
    from src.config import Config

    d_user_eff = int(Config().ppo.worker_d_user) + window * len(features)

    class AnalysisConfig(Config):
        def __init__(self):
            super().__init__()
            self.system.obs_history_window = int(window)
            self.system.obs_history_features = tuple(features)
            self.ppo.worker_d_user = int(d_user_eff)

    return AnalysisConfig


def build_env(cfg_cls, snr_db: float, sigma2: float, seed: int):
    """按评估口径构造固定 sigma2 的 env（镜像 build_config_for_snr）。

    Returns:
        (env, layout): RSMAEnergyEnv 与其 ObsLayout。
    """
    from src.env.rsma_env import RSMAEnergyEnv

    cfg = cfg_cls()
    cfg.system.sigma2 = float(sigma2)
    cfg.system.snr_domain_randomization = False
    env = RSMAEnergyEnv(cfg)
    layout = resolve_layout(cfg)
    env.reset(seed=int(seed))
    return env, layout


def load_worker(ckpt_path, env):
    """加载 Worker ActorCriticAttn；校验 obs 维与 env 一致。"""
    from scripts.evaluate_snr_sweep import load_ppo_model

    model, ckpt_obs_dim, err = load_ppo_model(str(ckpt_path), env)
    if err is not None:
        raise RuntimeError(f"Worker 加载失败: {err}")
    if int(ckpt_obs_dim) != int(env.obs_dim):
        raise RuntimeError(
            f"Worker obs_dim {ckpt_obs_dim} != env.obs_dim {env.obs_dim}；"
            "检查 --obs_history_window 是否与检查点训练期一致"
        )
    model.eval()
    return model


def load_meta(ckpt_path, env):
    """加载 Meta ActorCritic（group_level_alpha 由 checkpoint 自推断）。"""
    from scripts.evaluate_snr_sweep import load_meta_model

    meta_ac, t_s, err = load_meta_model(str(ckpt_path), env)
    if err is not None:
        raise RuntimeError(f"Meta 加载失败: {err}")
    return meta_ac, t_s


def assert_softmax_power_share(env) -> None:
    """p 目标公式依赖 softmax 功率份额口径；不满足时 loud 拒绝。"""
    if not bool(getattr(env.config.system, "use_softmax_power_share", False)):
        raise RuntimeError("分析要求 use_softmax_power_share=True（p 块为 softmax 份额 logits）；当前环境配置不满足。")


def validate_obs_dim(env, layout: ObsLayout) -> None:
    """env 实际 obs_dim 与布局推导一致（除幂宫/结构漂移的快速护栏）。"""
    if int(env.obs_dim) != layout.obs_dim():
        raise RuntimeError(
            f"env.obs_dim({env.obs_dim}) != layout.obs_dim({layout.obs_dim()})；"
            "布局解析与 env 不一致（可能 unknown 观测维度被开启）"
        )
