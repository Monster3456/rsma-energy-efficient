"""时序评估包装器 scripts/evaluate_snr_sweep_temporal.py 测试

覆盖:
1. Config 子类工厂: 实例级覆盖 system.obs_history_window/features 与 ppo.worker_d_user
   （基值取自原始 Config().ppo.worker_d_user，不硬编码 13；工厂幂等、默认 Config 零污染）
2. 补丁 Config 与训练端自洽: env.obs_dim == K * worker_d_user_eff + 3（精确重建前提）
3. 特征名解析: 未知/空/重复特征名报错
4. 预检 preflight_check: 维度不匹配明确报错退出；文件不存在/非注意力检查点放行
5. 集成（子进程）: 伪造时序 checkpoint（main.py 同款构造，零训练）→ 包装器 →
   不可变脚本完整评估成功（d_user=25/n_inject=2 精确重建、无截断告警、完整表格输出）
6. 拒绝（子进程）: 非时序 checkpoint → 包装器预检明确报错、非零退出
"""

import os
import subprocess
import sys
import time

import pytest
import torch

from scripts.evaluate_snr_sweep_temporal import (
    REPO_ROOT,
    make_temporal_config_cls,
    parse_features,
    preflight_check,
)
from src.config import Config
from src.env.rsma_env import RSMAEnergyEnv
from src.models.actor_critic import ActorCriticAttn
from src.models.meta_controller import MetaActorCritic

WRAPPER_PATH = REPO_ROOT / "scripts" / "evaluate_snr_sweep_temporal.py"


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _run_wrapper(*args, timeout=300):
    """以子进程运行包装器（UTF-8 输入输出、仓库根为 cwd，与本机用法一致）。"""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.run(
        [sys.executable, str(WRAPPER_PATH), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
        env=env,
        timeout=timeout,
        check=False,
    )


def _build_attention_checkpoint(path, window, features=("h_gains", "queue", "power")):
    """按 main.py 的构造方式生成（无需训练的）注意力 Worker+Meta 检查点。

    window=0 时即普通非时序检查点（d_user=13，user_embed 输入 15）。
    Meta 构造 kwargs 与不可变脚本 load_meta_model 的推断逻辑一一对应。
    Returns: (path, d_user)
    """
    cfg = Config()
    cfg.system.obs_history_window = window
    cfg.system.obs_history_features = tuple(features)
    env = RSMAEnergyEnv(cfg)
    K = env.K
    d_user = cfg.ppo.worker_d_user + window * len(features)

    worker = ActorCriticAttn(
        obs_dim=env.obs_dim,
        action_dim=4 * K,
        K=K,
        d_user=d_user,
        d_global=env.obs_dim - K * d_user,
        hidden_dim=cfg.ppo.worker_attn_hidden,
        n_heads=cfg.ppo.worker_attn_heads,
        n_attn_layers=cfg.ppo.worker_attn_layers,
        use_plackett_luce=True,
        use_mean_field_critic=cfg.ppo.use_mean_field_critic,
        mean_field_hidden_dim=cfg.ppo.mean_field_hidden_dim,
        qos_drift_weights=(tuple(cfg.system.qos_drift_weights) if cfg.ppo.use_mean_field_critic else None),
        use_bucket_experts=cfg.ppo.use_bucket_experts,
        n_experts=cfg.ppo.n_bucket_experts,
        bucket_snr_min=cfg.ppo.bucket_snr_min,
        bucket_snr_max=cfg.ppo.bucket_snr_max,
    )
    meta = MetaActorCritic(
        K=K,
        d_user=cfg.hierarchical.meta_d_user,
        hidden_dim=cfg.hierarchical.meta_hidden_dim,
        n_heads=cfg.hierarchical.meta_n_heads,
        n_groups=cfg.hierarchical.n_groups,
        group_level_alpha=cfg.hierarchical.group_level_alpha,
        group_alpha_min=cfg.hierarchical.group_alpha_min,
        group_alpha_max=cfg.hierarchical.group_alpha_max,
    )
    torch.save(
        {
            "worker_state_dict": worker.state_dict(),
            "meta_state_dict": meta.state_dict(),
        },
        path,
    )
    return path, d_user


# ---------------------------------------------------------------------- #
# 1-2. Config 子类工厂（单元）
# ---------------------------------------------------------------------- #
class TestTemporalConfigFactory:
    pytestmark = pytest.mark.fast

    @pytest.mark.parametrize(
        "window,features",
        [
            (4, ("h_gains", "queue", "power")),
            (2, ("queue",)),
            (1, ("h_gains", "power")),
            (3, ("power", "queue", "h_gains")),
        ],
    )
    def test_factory_overrides_instance_fields(self, window, features):
        base = Config().ppo.worker_d_user
        expected = base + window * len(features)
        cls, d_user_eff = make_temporal_config_cls(window, features)
        assert d_user_eff == expected
        cfg = cls()
        assert cfg.system.obs_history_window == window
        assert cfg.system.obs_history_features == features
        assert cfg.ppo.worker_d_user == expected
        # 每次实例化一致
        assert cls().ppo.worker_d_user == expected

    def test_factory_env_obs_dim_consistent_with_training(self):
        """补丁 Config 构建的 env obs_dim 与 main.py 训练端完全一致：
        obs = K * worker_d_user_eff + d_global(3) —— 这正是不可变脚本
        load_ppo_model 精确重建（无截断、无错位）的前提。"""
        cls, d_user_eff = make_temporal_config_cls(4, ("h_gains", "queue", "power"))
        env = RSMAEnergyEnv(cls())
        assert env.obs_dim == env.K * d_user_eff + 3

    def test_factory_idempotent_and_default_config_untouched(self):
        """工厂可重复调用（始终以原始 Config 为基类）；默认 Config 不被污染。"""
        cls1, _ = make_temporal_config_cls(4, ("h_gains", "queue", "power"))
        cls2, d2 = make_temporal_config_cls(2, ("queue",))
        assert d2 == Config().ppo.worker_d_user + 2
        assert cls2().ppo.worker_d_user == d2
        assert not issubclass(cls2, cls1)
        # 原始默认 Config 的 worker_d_user / 历史窗口不受影响
        fresh = Config()
        assert fresh.ppo.worker_d_user == Config().ppo.worker_d_user
        assert fresh.system.obs_history_window == 0


# ---------------------------------------------------------------------- #
# 3. 特征名解析（单元）
# ---------------------------------------------------------------------- #
class TestParseFeatures:
    pytestmark = pytest.mark.fast

    def test_default_spec(self):
        assert parse_features("h_gains,queue,power") == ("h_gains", "queue", "power")

    def test_strips_whitespace_and_empty_entries(self):
        assert parse_features(" h_gains , queue ,, ") == ("h_gains", "queue")

    def test_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            parse_features("h_gains,bogus")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="不能为空"):
            parse_features(" , ,")

    def test_duplicate_raises(self):
        with pytest.raises(ValueError, match="重复"):
            parse_features("queue,queue")


# ---------------------------------------------------------------------- #
# 4. 预检（单元）
# ---------------------------------------------------------------------- #
class TestPreflightCheck:
    pytestmark = pytest.mark.fast

    def test_dimension_mismatch_exits_with_clear_error(self, tmp_path, capsys):
        ckpt, _ = _build_attention_checkpoint(tmp_path / "plain.pt", window=0)
        # window=4 补丁下期望 user_embed 输入 = 13+12+2 = 27，实际 15
        with pytest.raises(SystemExit) as exc_info:
            preflight_check(str(ckpt), d_user_eff=25)
        assert exc_info.value.code == 2
        out = capsys.readouterr().out
        assert "[TEMPORAL-EVAL] ERROR" in out
        assert "15" in out and "27" in out

    def test_matching_checkpoint_passes(self, tmp_path):
        ckpt, d_user = _build_attention_checkpoint(tmp_path / "temporal.pt", window=4)
        # 不抛 SystemExit 即通过
        preflight_check(str(ckpt), d_user_eff=d_user)

    def test_missing_file_passes_through(self, tmp_path):
        preflight_check(str(tmp_path / "nonexistent.pt"), d_user_eff=25)

    def test_non_attention_checkpoint_passes_through(self, tmp_path):
        """MLP/无 user_embed 键的检查点交由不可变脚本自然处理（不预检拦截）。"""
        p = tmp_path / "mlp.pt"
        torch.save({"worker_state_dict": {"actor_net.0.weight": torch.zeros(4, 5)}}, p)
        preflight_check(str(p), d_user_eff=25)


# ---------------------------------------------------------------------- #
# 5-6. 子进程集成：包装器 → 不可变评估脚本
# ---------------------------------------------------------------------- #
class TestWrapperSubprocess:
    """端到端：伪造时序 checkpoint → 包装器子进程 → 不可变脚本完整评估。

    实测单 SNR 点 × 1 episode（SDMA + PPO，含解释器与 torch 导入）约 6s，
    低于 15s 阈值，纳入 fast 门禁。"""

    pytestmark = pytest.mark.fast

    def test_wrapper_evaluates_temporal_checkpoint(self, tmp_path):
        ckpt, d_user = _build_attention_checkpoint(tmp_path / "fake_temporal.pt", window=4)
        t0 = time.time()
        proc = _run_wrapper(
            "--checkpoint",
            str(ckpt),
            "--n_episodes",
            "1",
            "--snr_subset",
            "10",
            "--obs_history_window",
            "4",
        )
        elapsed = time.time() - t0
        assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        out = proc.stdout
        # 包装器横幅
        assert "[TEMPORAL-EVAL] obs_history_window=4" in out
        # 不可变脚本的精确重建：d_user=补丁值、n_inject=2、无截断兼容告警
        assert f"d_user={d_user}" in out
        assert "n_inject=2" in out
        assert "!= env.obs_dim" not in out
        # Meta-Controller 正常加载
        assert "[META] Loading MetaActorCritic" in out
        # 完整评估输出：逐点指标行 + 汇总表
        assert "SNR SWEEP REPORT" in out
        assert "PPO-RSMA" in out and "SDMA" in out
        assert "Deliv=" in out
        assert "表1" in out
        # 快速门禁预算自检（CI 慢机上的劣化信号）
        assert elapsed < 60, f"wrapper 子进程耗时 {elapsed:.1f}s，超出预算"

    def test_wrapper_rejects_non_temporal_checkpoint(self, tmp_path):
        """非时序检查点（window=0 训练）在 window=4 补丁下被预检明确拒绝。"""
        ckpt, _ = _build_attention_checkpoint(tmp_path / "fake_plain.pt", window=0)
        proc = _run_wrapper(
            "--checkpoint",
            str(ckpt),
            "--n_episodes",
            "1",
            "--snr_subset",
            "10",
            "--obs_history_window",
            "4",
        )
        assert proc.returncode != 0
        combined = proc.stdout + proc.stderr
        assert "[TEMPORAL-EVAL] ERROR" in combined
        assert "scripts/evaluate_snr_sweep.py" in combined  # 指引直连原脚本
