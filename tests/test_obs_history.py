"""时序历史窗口观测（obs_history_window）单元测试

覆盖:
1. 默认关（window=0）时 obs_dim 与历史版本完全一致（legacy 钉死配置 + 默认 Config）
2. 开启后 obs_dim = base + window * len(features) * K
3. 历史区内容正确性：reset 预填充 t=0 快照；逐步后 lag 块 = 槽 t-H..t-1 的快照（最旧在前）
4. global 尾位置不变：最后 3 维仍为 [time_ratio, snr, eps]，per-user 区首块仍为 h_gains
5. 未知特征名报错（config 构造期 + env 构造期）
6. eval 兼容回路：load_ppo_model 可从 enlarged d_user 检查点构建模型并加载 user_embed 权重
7. ActorCriticAttn（enlarged d_user）act()/evaluate_actions 冒烟
"""

import numpy as np
import pytest
import torch

from src.config import Config, SystemConfig
from src.env.rsma_env import RSMAEnergyEnv
from src.models.actor_critic import ActorCriticAttn

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _legacy_cfg(T=16, seed=0, window=0, features=("h_gains", "queue", "power")) -> Config:
    """legacy 钉死配置（与 tests/test_env.py::_make_env 同款，K=16 / 无异构 QoS），
    obs 布局为 11K + [history] + 3，便于按块索引定位。"""
    cfg = Config()
    cfg.system.T = T
    cfg.system.csi_mode = "imperfect"
    cfg.seed = seed
    cfg.system.K = 16
    cfg.system.N_r = 16
    cfg.system.enable_hetero_qos = False
    cfg.system.use_softmax_power_share = False
    cfg.hierarchical.group_level_alpha = False
    cfg.system.obs_prune_constant_blocks = False  # 遗留口径：钦住未剪 obs
    cfg.system.obs_history_window = window
    cfg.system.obs_history_features = tuple(features)
    return cfg


def _current_feature_blocks(obs: np.ndarray, K: int, features: tuple) -> list:
    """从当前槽 obs 中提取各历史特征块（块位置：h_gains=0, queue=1, power=4）。"""
    blocks = {
        "h_gains": obs[0:K],
        "queue": obs[K : 2 * K],
        "power": obs[4 * K : 5 * K],
    }
    return [blocks[f] for f in features]


def _history_lags(obs: np.ndarray, K: int, per_user_dim: int, features: tuple, H: int) -> list:
    """解析 obs 历史区：返回 H 个 lag（最旧在前），每个 lag 为 len(features) 个 K 维块。"""
    F = len(features)
    region = obs[per_user_dim : per_user_dim + H * F * K]
    lags = []
    for lag_idx in range(H):
        lag = region[lag_idx * F * K : (lag_idx + 1) * F * K]
        lags.append([lag[j * K : (j + 1) * K] for j in range(F)])
    return lags


# ---------------------------------------------------------------------- #
# 1. 默认关（零回归）
# ---------------------------------------------------------------------- #
class TestDefaultOff:
    def test_legacy_pinned_obs_dim_unchanged(self):
        env = RSMAEnergyEnv(_legacy_cfg(window=0))
        assert env.obs_dim == 11 * env.K + 3
        obs, _ = env.reset(seed=0)
        assert obs.shape == (env.obs_dim,)

    def test_default_config_obs_dim_unchanged(self):
        cfg = Config()
        assert cfg.system.obs_history_window == 0
        env = RSMAEnergyEnv(cfg)
        # 默认配置（2026-10-12 A-prune 转正）: K=24, 异构 QoS 开, 剪 3 常数块
        # → (11K+3+2K) − 3K = 10K+3
        assert env.obs_dim == 10 * env.K + 3
        obs, _ = env.reset(seed=0)
        assert obs.shape == (env.obs_dim,)


# ---------------------------------------------------------------------- #
# 2. 开启后 obs_dim 公式
# ---------------------------------------------------------------------- #
class TestObsDim:
    @pytest.mark.parametrize(
        "window,features,extra_blocks",
        [
            (4, ("h_gains", "queue", "power"), 12),
            (2, ("queue",), 2),
            (1, ("h_gains",), 1),
            (3, ("power", "queue"), 6),
        ],
    )
    def test_obs_dim_formula_legacy(self, window, features, extra_blocks):
        env = RSMAEnergyEnv(_legacy_cfg(window=window, features=features))
        assert env.obs_dim == 11 * env.K + 3 + extra_blocks * env.K
        obs, _ = env.reset(seed=0)
        assert obs.shape == (env.obs_dim,)

    def test_obs_dim_formula_default_config(self):
        cfg = Config()
        cfg.system.obs_history_window = 4
        cfg.system.obs_prune_constant_blocks = False  # 公式口径（未剪 13 块布局）
        env = RSMAEnergyEnv(cfg)
        F = len(cfg.system.obs_history_features)
        base = 11 * env.K + 3 + 2 * env.K  # 默认异构 QoS 开
        assert env.obs_dim == base + 4 * F * env.K


# ---------------------------------------------------------------------- #
# 3. 历史区内容正确性
# ---------------------------------------------------------------------- #
class TestHistoryContents:
    def test_reset_prefill_with_t0_snapshot(self):
        K, H = 16, 4
        cfg = _legacy_cfg(window=H)
        env = RSMAEnergyEnv(cfg)
        obs, _ = env.reset(seed=0)
        features = cfg.system.obs_history_features
        snap0 = np.concatenate(_current_feature_blocks(obs, K, features))
        lags = _history_lags(obs, K, 11 * K, features, H)
        assert len(lags) == H
        for lag in lags:
            assert np.allclose(np.concatenate(lag), snap0, atol=1e-6)

    def test_lag_blocks_track_previous_slots(self):
        """逐步校验：slot t 的历史区 = 槽 max(0,t-H)..t-1 的快照（最旧在前）。"""
        K, H, n_steps = 16, 4, 8
        cfg = _legacy_cfg(window=H)
        env = RSMAEnergyEnv(cfg)
        features = cfg.system.obs_history_features
        obs, _ = env.reset(seed=0)
        snapshots = [np.concatenate(_current_feature_blocks(obs, K, features))]
        rng = np.random.default_rng(7)
        for t in range(1, n_steps + 1):
            action = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
            obs, _, _, _, _ = env.step(action)
            snapshots.append(np.concatenate(_current_feature_blocks(obs, K, features)))
            lags = _history_lags(obs, K, 11 * K, features, H)
            for i, lag in enumerate(lags):
                expected = snapshots[max(0, t - H + i)]
                assert np.allclose(np.concatenate(lag), expected, atol=1e-6), f"t={t}, lag={i}"
            # 状态级交叉验证：queue 块与 env 当前队列一致
            assert np.allclose(obs[K : 2 * K], env.traffic.queue_lengths / env.Q_max, atol=1e-5)
        assert len(snapshots) == n_steps + 1

    def test_history_feature_order_respected(self):
        """features=("power", "queue") 反序时，lag 内块顺序随配置。"""
        K, H, n_steps = 16, 2, 3
        features = ("power", "queue")
        cfg = _legacy_cfg(window=H, features=features)
        env = RSMAEnergyEnv(cfg)
        obs, _ = env.reset(seed=0)
        snapshots = [np.concatenate(_current_feature_blocks(obs, K, features))]
        rng = np.random.default_rng(5)
        for t in range(1, n_steps + 1):
            action = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
            obs, _, _, _, _ = env.step(action)
            snapshots.append(np.concatenate(_current_feature_blocks(obs, K, features)))
            lags = _history_lags(obs, K, 11 * K, features, H)
            for i, lag in enumerate(lags):
                expected = snapshots[max(0, t - H + i)]
                # lag[0] = power 块, lag[1] = queue 块（按配置顺序）
                assert np.allclose(np.concatenate(lag), expected, atol=1e-6), f"t={t}, lag={i}"

    def test_history_reset_between_episodes(self):
        """episode 结束 reset 后，历史区重新预填充为新 episode 的 t=0 快照。"""
        K, H = 16, 2
        cfg = _legacy_cfg(T=3, window=H)
        env = RSMAEnergyEnv(cfg)
        features = cfg.system.obs_history_features
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(2)
        for _ in range(3):  # 跑满一个 episode（T=3）
            obs, _, _, trunc, _ = env.step(rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32))
        assert trunc
        obs2, _ = env.reset(seed=1)
        snap = np.concatenate(_current_feature_blocks(obs2, K, features))
        for lag in _history_lags(obs2, K, 11 * K, features, H):
            assert np.allclose(np.concatenate(lag), snap, atol=1e-6)
        # 与旧 episode 的最后 obs 不同（预填充来自新 episode）
        assert not np.allclose(obs2[11 * K :], obs[11 * K :], atol=1e-6)

    def test_double_reset_reprefills_window(self):
        """episode 中途连续两次 reset：窗口不残留旧内容，两次 obs 逐元素一致。

        若 reset 未清空历史窗，第二次 reset 会走「非空追加」分支，
        历史区将残留第一次 reset 前的最后快照，两次 obs 即出现差异。"""
        K, H = 16, 2
        cfg = _legacy_cfg(T=8, window=H)
        env = RSMAEnergyEnv(cfg)
        env.reset(seed=0)
        rng = np.random.default_rng(3)
        for _ in range(2):  # episode 中途
            env.step(rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32))
        obs2, _ = env.reset(seed=0)
        obs3, _ = env.reset(seed=0)
        features = cfg.system.obs_history_features
        snap = np.concatenate(_current_feature_blocks(obs3, K, features))
        for lag in _history_lags(obs3, K, 11 * K, features, H):
            assert np.allclose(np.concatenate(lag), snap, atol=1e-6)
        assert np.array_equal(obs2, obs3)


# ---------------------------------------------------------------------- #
# 4. global 尾位置 + 当前区零影响
# ---------------------------------------------------------------------- #
class TestTailPosition:
    def test_window_on_off_identical_current_and_tail(self):
        """同 seed 同动作下，开/关窗口的 env 演化一致：
        当前 per-user 区与 global 尾逐元素相同（历史窗口无副作用）。"""
        K, H, T = 16, 4, 12
        env_off = RSMAEnergyEnv(_legacy_cfg(T=T, window=0))
        env_on = RSMAEnergyEnv(_legacy_cfg(T=T, window=H))
        obs_off, _ = env_off.reset(seed=3)
        obs_on, info_on = env_on.reset(seed=3)
        F = 3
        assert obs_on.shape == (11 * K + H * F * K + 3,)
        assert obs_off.shape == (11 * K + 3,)
        # per-user 区首块仍为 h_gains（≈ 信道增益 / (N_r·N_t)）
        assert np.allclose(obs_on[:K], info_on["channel_gains"] / (16 * 2), atol=1e-6)
        # time_ratio 位于 global 尾第一位
        assert obs_on[-3] == pytest.approx(0.0, abs=1e-7)
        rng = np.random.default_rng(11)
        for _ in range(T - 1):
            action = rng.uniform(-1.0, 1.0, size=env_off.action_dim).astype(np.float32)
            obs_off, _, _, _, _ = env_off.step(action)
            obs_on, _, _, _, _ = env_on.step(action)
            # 当前 per-user 区（含 h_gains 首块）与关闭时逐元素一致
            assert np.array_equal(obs_on[: 11 * K], obs_off[: 11 * K])
            # global 尾 [time_ratio, snr_norm, eps_norm] 位置与数值不变
            assert np.array_equal(obs_on[-3:], obs_off[-3:])
            assert obs_on[-3] == pytest.approx(env_on.t / T, abs=1e-6)


# ---------------------------------------------------------------------- #
# 5. 配置校验
# ---------------------------------------------------------------------- #
class TestValidation:
    def test_unknown_feature_env_raises(self):
        cfg = _legacy_cfg(window=2)
        cfg.system.obs_history_features = ("h_gains", "bogus")
        with pytest.raises(ValueError, match="bogus"):
            RSMAEnergyEnv(cfg)

    def test_unknown_feature_config_init_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            SystemConfig(obs_history_features=("h_gains", "bogus"))

    def test_negative_window_raises(self):
        with pytest.raises(ValueError, match="obs_history_window"):
            SystemConfig(obs_history_window=-1)
        cfg = _legacy_cfg()
        cfg.system.obs_history_window = -1
        with pytest.raises(ValueError, match="obs_history_window"):
            RSMAEnergyEnv(cfg)


# ---------------------------------------------------------------------- #
# 6. eval 兼容回路（scripts/evaluate_snr_sweep.load_ppo_model，脚本本身零改动）
# ---------------------------------------------------------------------- #
class TestEvalCompat:
    def test_load_ppo_model_roundtrip(self, tmp_path):
        """main.py 方式构造 enlarged d_user 的 Worker → 存检查点 →
        load_ppo_model 按 user_embed.0.weight 推断 d_user_eff，以 d_user=13 +
        n_inject 吸收差值重建模型并加载权重。"""
        H, F = 4, 3
        cfg = Config()
        cfg.system.obs_history_window = H
        cfg.system.obs_prune_constant_blocks = False  # 未剪时序检查点往返
        env = RSMAEnergyEnv(cfg)
        K = env.K
        d_user = cfg.ppo.worker_d_user + H * F  # main.py 的扩展 d_user

        model = ActorCriticAttn(
            obs_dim=env.obs_dim,
            action_dim=4 * K,
            K=K,
            d_user=d_user,
            d_global=env.obs_dim - K * d_user,
            hidden_dim=128,
            n_heads=4,
            n_attn_layers=2,
            use_plackett_luce=True,
        )
        sd = model.state_dict()
        # user_embed 输入维 = d_user + n_global_inject(2)
        assert sd["user_embed.0.weight"].shape == (128, d_user + 2)

        ckpt_path = tmp_path / "fake_hist_ckpt.pt"
        torch.save(
            {"worker_state_dict": sd, "meta_state_dict": {}},
            ckpt_path,
        )

        from scripts.evaluate_snr_sweep import load_ppo_model

        loaded, ckpt_obs_dim, err = load_ppo_model(str(ckpt_path), env)
        assert err is None and loaded is not None
        # eval 端推断：d_user_eff = d_user+2，回退 d_user=13、n_inject 吸收差值
        assert loaded.d_user == cfg.ppo.worker_d_user
        assert loaded.n_global_inject == d_user + 2 - cfg.ppo.worker_d_user
        assert loaded.obs_dim == ckpt_obs_dim == K * cfg.ppo.worker_d_user + 3
        # user_embed 权重被完整加载（形状匹配）
        assert torch.equal(loaded.user_embed[0].weight, sd["user_embed.0.weight"])
        assert torch.equal(loaded.global_proj.weight, sd["global_proj.weight"])


# ---------------------------------------------------------------------- #
# 7. ActorCriticAttn（enlarged d_user）冒烟
# ---------------------------------------------------------------------- #
class TestAttnWorkerSmoke:
    def test_act_returns_five_tuple_correct_shapes(self):
        H = 4
        cfg = Config()
        cfg.system.obs_history_window = H
        cfg.system.obs_prune_constant_blocks = False  # 未剪时序冒烟：d_user 与 obs 对齐
        env = RSMAEnergyEnv(cfg)
        K = env.K
        d_user = cfg.ppo.worker_d_user + H * len(cfg.system.obs_history_features)
        # 镜像 main.py 的完整构造路径（均值场 critic + 桶专家头）
        model = ActorCriticAttn(
            obs_dim=env.obs_dim,
            action_dim=4 * K,
            K=K,
            d_user=d_user,
            d_global=env.obs_dim - K * d_user,
            hidden_dim=128,
            n_heads=4,
            n_attn_layers=2,
            use_plackett_luce=True,
            use_mean_field_critic=True,
            mean_field_hidden_dim=64,
            qos_drift_weights=tuple(cfg.system.qos_drift_weights),
            use_bucket_experts=True,
            n_experts=4,
        )
        obs, _ = env.reset(seed=0)
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model.act(obs_t)
        assert len(out) == 5
        action, decoding_order, log_prob, value, cont_log_prob = out
        assert action.shape == (1, 3 * K)
        assert decoding_order.shape == (1, K)
        assert log_prob.shape == (1,)
        assert value.shape == (1,)
        assert cont_log_prob.shape == (1,)

        # deterministic 模式
        with torch.no_grad():
            out_det = model.act(obs_t, deterministic=True)
        assert out_det[0].shape == (1, 3 * K)
        assert out_det[1].shape == (1, K)

        # 批量 evaluate_actions 回路
        obs_b = torch.as_tensor(np.tile(obs, (3, 1)), dtype=torch.float32)
        act_b = out[0].repeat(3, 1)
        order_b = out[1].repeat(3, 1)
        with torch.no_grad():
            lp, v, ent, clp = model.evaluate_actions(obs_b, act_b, order_b)
        assert lp.shape == (3,)
        assert v.shape == (3,)
        assert ent.shape == (3,)
        assert clp.shape == (3,)
