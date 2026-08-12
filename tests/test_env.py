"""RSMAEnergyEnv 集成测试

覆盖:
1. reset 返回 (obs, info), shape 正确
2. observation/action space 定义一致
3. 随机动作连跑 100 步无 crash, 无 NaN/Inf reward
4. truncated 在 t==T 时触发
5. 完美CSI模式下 outage_count == 0
"""
import numpy as np
import pytest

from src.config import Config
from src.env.rsma_env import RSMAEnergyEnv


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _make_env(T=20, csi_mode="perfect", seed=0) -> RSMAEnergyEnv:
    cfg = Config()
    cfg.system.T = T
    cfg.system.csi_mode = csi_mode
    cfg.seed = seed
    # 本测试文件面向遗留语义（split_mask 二值 gate 乘 Worker α），
    # 显式钉住旧版配置，避免后续默认值演进（K=24 / 异构 QoS / softmax
    # 功率份额 / group-level α）导致维度与映射断言漂移。
    cfg.system.K = 16
    cfg.system.N_r = 16
    cfg.system.enable_hetero_qos = False
    cfg.system.use_softmax_power_share = False
    cfg.hierarchical.group_level_alpha = False
    return RSMAEnergyEnv(cfg)


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #
@pytest.mark.fast
class TestEnvSpaces:
    def test_action_space_shape(self):
        env = _make_env()
        assert env.action_space.shape == (3 * env.K,)
        assert env.action_space.low.min() == -1.0
        assert env.action_space.high.max() == 1.0

    def test_observation_space_shape(self):
        env = _make_env()
        assert env.observation_space.shape == (env.obs_dim,)
        # Task#13: 11K 状态块 + 1 time_ratio + 2 (snr_norm, eps_norm)
        assert env.obs_dim == 11 * env.K + 3

    def test_default_dims_match_config(self):
        env = _make_env()
        assert env.K == env.config.system.K
        assert env.action_space.shape[0] == 3 * env.K
        assert env.observation_space.shape[0] == 11 * env.K + 3


@pytest.mark.fast
class TestEnvReset:
    def test_reset_returns_tuple(self):
        env = _make_env()
        out = env.reset()
        assert isinstance(out, tuple) and len(out) == 2
        obs, info = out
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_obs_shape_dtype(self):
        env = _make_env()
        obs, _ = env.reset()
        assert obs.shape == (env.obs_dim,)
        assert obs.dtype == np.float32

    def test_reset_initial_queue_zero(self):
        env = _make_env()
        env.reset()
        # Q_k = 0 在 reset 后
        assert np.allclose(env.traffic.queue_lengths, 0.0)

    def test_reset_seed_reproducible(self):
        env1 = _make_env(seed=123)
        env2 = _make_env(seed=123)
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        assert np.allclose(obs1, obs2)


@pytest.mark.fast
class TestEnvStep:
    def test_step_signature(self):
        env = _make_env()
        env.reset(seed=0)
        action = env.action_space.sample()
        out = env.step(action)
        assert len(out) == 5
        obs, reward, terminated, truncated, info = out
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (env.obs_dim,)
        assert isinstance(reward, float)
        assert terminated is False
        assert isinstance(info, dict)

    def test_step_finite_reward(self):
        """连跑 100 步, 奖励始终为有限实数."""
        env = _make_env(T=200)
        env.reset(seed=42)
        rng = np.random.default_rng(0)
        for _ in range(100):
            action = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward), f"reward not finite: {reward}"
            assert np.all(np.isfinite(obs)), "observation contains NaN/Inf"
            if truncated:
                env.reset(seed=42)

    def test_random_actions_no_crash(self):
        env = _make_env(T=200)
        env.reset(seed=1)
        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)

    def test_truncated_at_T(self):
        T = 5
        env = _make_env(T=T)
        env.reset(seed=0)
        for t in range(T - 1):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert trunc is False, f"truncated triggered too early at t={t}"
        # 第 T 步触发 truncated
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert trunc is True
        assert term is False

    def test_zero_action_zero_power(self):
        """动作 = -1 -> 映射到 0 -> 总功率 = 0."""
        env = _make_env()
        env.reset(seed=0)
        action = -np.ones(env.action_dim, dtype=np.float32)
        obs, r, term, trunc, info = env.step(action)
        assert info["total_power"] == pytest.approx(0.0, abs=1e-9)
        assert info["mean_alpha"] == pytest.approx(0.0, abs=1e-9)

    def test_max_action_max_power(self):
        """动作 = +1 -> 映射到 1 -> 每用户功率 = P_k_max."""
        env = _make_env()
        env.reset(seed=0)
        action = np.ones(env.action_dim, dtype=np.float32)
        obs, r, term, trunc, info = env.step(action)
        expected_total = env.K * env.P_k_max
        assert info["total_power"] == pytest.approx(expected_total, rel=1e-6)


@pytest.mark.fast
class TestEnvCSIModes:
    def test_perfect_csi_no_outage(self):
        env = _make_env(csi_mode="perfect")
        env.reset(seed=0)
        for _ in range(20):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert info["outage_count"] == 0

    def test_imperfect_csi_runs(self):
        env = _make_env(csi_mode="imperfect")
        env.reset(seed=0)
        for _ in range(20):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert np.isfinite(r)
            assert info["outage_count"] >= 0


@pytest.mark.fast
class TestActionParsing:
    def test_parse_action_shape(self):
        env = _make_env()
        a = np.zeros(env.action_dim, dtype=np.float32)
        alpha, p, d = env._parse_action(a)
        assert alpha.shape == (env.K,)
        assert p.shape == (env.K,)
        assert d.shape == (env.K,)

    def test_parse_action_mapping(self):
        env = _make_env()
        # 全零 raw -> (0+1)/2 = 0.5
        a = np.zeros(env.action_dim, dtype=np.float32)
        alpha, p, d = env._parse_action(a)
        assert np.allclose(alpha, 0.5)
        assert np.allclose(p, 0.5)
        assert np.allclose(d, 0.5)

    def test_parse_action_clipped(self):
        env = _make_env()
        a = np.full(env.action_dim, 5.0, dtype=np.float32)  # 越界
        alpha, p, d = env._parse_action(a)
        assert np.all(alpha <= 1.0) and np.all(alpha >= 0.0)
        assert np.all(p <= 1.0) and np.all(p >= 0.0)
        assert np.all(d <= 1.0) and np.all(d >= 0.0)

    def test_parse_action_wrong_shape_raises(self):
        env = _make_env()
        with pytest.raises(ValueError):
            env._parse_action(np.zeros(env.action_dim - 1))


@pytest.mark.fast
class TestObservation:
    def test_observation_finite(self):
        env = _make_env()
        obs, _ = env.reset(seed=0)
        assert np.all(np.isfinite(obs))

    def test_observation_in_space(self):
        env = _make_env()
        obs, _ = env.reset(seed=0)
        # observation_space 边界宽松, obs 应在其中
        assert env.observation_space.contains(obs) or np.all(
            (obs >= env.observation_space.low) & (obs <= env.observation_space.high)
        )


# ---------------------------------------------------------------------- #
# Split Mask Interface Tests
# ---------------------------------------------------------------------- #
@pytest.mark.fast
class TestSplitMask:
    def test_set_split_mask_basic(self):
        """set_split_mask 正确设置 split_mask"""
        env = _make_env()
        env.reset(seed=0)
        mask = np.zeros(env.K, dtype=np.float32)
        mask[:8] = 1.0  # 前 8 用户启用分裂
        env.set_split_mask(mask)
        assert np.array_equal(env.split_mask, mask)

    def test_set_split_mask_all_ones(self):
        """全 1 mask: 所有用户启用分裂 (默认)"""
        env = _make_env()
        env.reset(seed=0)
        mask = np.ones(env.K, dtype=np.float32)
        env.set_split_mask(mask)
        assert np.all(env.split_mask == 1.0)

    def test_set_split_mask_all_zeros(self):
        """全 0 mask: 所有用户不分裂"""
        env = _make_env()
        env.reset(seed=0)
        mask = np.zeros(env.K, dtype=np.float32)
        env.set_split_mask(mask)
        assert np.all(env.split_mask == 0.0)

    def test_split_mask_constrains_alpha(self):
        """split_mask=0 时, alpha 应被约束为 0"""
        env = _make_env()
        env.reset(seed=0)
        # 设置全 0 mask
        env.set_split_mask(np.zeros(env.K, dtype=np.float32))
        # 动作全 1 -> alpha 映射到 1.0, 但被 mask 约束为 0
        action = np.ones(env.action_dim, dtype=np.float32)
        alpha, p, d = env._parse_action(action)
        assert np.allclose(alpha, 0.0), f"alpha should be 0 when mask=0, got {alpha}"

    def test_split_mask_partial(self):
        """部分用户分裂, 只有 mask=1 的用户 alpha > 0"""
        env = _make_env()
        env.reset(seed=0)
        mask = np.zeros(env.K, dtype=np.float32)
        mask[0] = 1.0
        mask[3] = 1.0
        env.set_split_mask(mask)
        action = np.ones(env.action_dim, dtype=np.float32)
        alpha, p, d = env._parse_action(action)
        # mask=0 的用户 alpha 应为 0
        for k in range(env.K):
            if mask[k] == 0:
                assert alpha[k] == 0.0
            else:
                assert alpha[k] > 0.0

    def test_split_mask_in_observation(self):
        """split_mask 应出现在观测向量的第 8K:9K 段"""
        env = _make_env()
        env.reset(seed=0)
        mask = np.zeros(env.K, dtype=np.float32)
        mask[0] = 1.0
        mask[5] = 1.0
        env.set_split_mask(mask)
        # 执行一步以更新观测
        obs, _, _, _, _ = env.step(env.action_space.sample())
        # split_mask 在观测中的位置: 第 8K 到 9K
        obs_mask = obs[8 * env.K: 9 * env.K]
        assert np.array_equal(obs_mask, mask)

    def test_group_ids_in_observation(self):
        """group_ids 归一化后应出现在观测向量的第 9K:10K 段"""
        env = _make_env()
        env.reset(seed=0)
        group_ids = np.array([0, 1, 2, 3] * 4, dtype=np.int64)
        env.set_group_ids(group_ids)
        obs, _, _, _, _ = env.step(env.action_space.sample())
        obs_groups = obs[9 * env.K: 10 * env.K]
        expected = group_ids / 3.0  # n_groups=4, 除以 G-1=3
        assert np.allclose(obs_groups, expected, atol=1e-5)

    def test_split_mask_reset(self):
        """reset 后 split_mask 恢复为初始值 0.5（bug fix：避免首次决策前 α 全 1 纯公共流）"""
        env = _make_env()
        env.reset(seed=0)
        env.set_split_mask(np.zeros(env.K, dtype=np.float32))
        env.reset(seed=1)
        assert np.all(env.split_mask == 0.5)


# ---------------------------------------------------------------------- #
# Group IDs Interface Tests
# ---------------------------------------------------------------------- #
@pytest.mark.fast
class TestSetGroupIds:
    def test_set_group_ids_basic(self):
        """set_group_ids 正确设置 group_ids"""
        env = _make_env()
        env.reset(seed=0)
        gids = np.array([0, 1, 2, 3] * 4, dtype=np.int64)
        env.set_group_ids(gids)
        assert np.array_equal(env.group_ids, gids)

    def test_set_group_ids_all_same(self):
        """所有用户同组"""
        env = _make_env()
        env.reset(seed=0)
        gids = np.zeros(env.K, dtype=np.int64)
        env.set_group_ids(gids)
        assert np.all(env.group_ids == 0)

    def test_group_ids_reset(self):
        """reset 后 group_ids 恢复为全 0"""
        env = _make_env()
        env.reset(seed=0)
        env.set_group_ids(np.array([1, 2, 3, 0] * 4, dtype=np.int64))
        env.reset(seed=1)
        assert np.all(env.group_ids == 0)


# ---------------------------------------------------------------------- #
# 4K Action (含解码顺序) Tests
# ---------------------------------------------------------------------- #
@pytest.mark.fast
class TestAction4K:
    def test_step_with_4k_action(self):
        """4K=64 维 action 能正常执行"""
        env = _make_env()
        env.reset(seed=0)
        K = env.K
        # 前 3K=48 维连续 + 后 K=16 维解码顺序
        action = np.zeros(4 * K, dtype=np.float32)
        action[:3*K] = np.random.uniform(-1, 1, 3*K)
        action[3*K:] = np.arange(K)  # 顺序解码
        obs, reward, term, trunc, info = env.step(action)
        assert np.isfinite(reward)
        assert obs.shape == (env.obs_dim,)

    def test_step_4k_sets_decoding_order(self):
        """4K action 中的后 K 维应设置 env.decoding_order"""
        env = _make_env()
        env.reset(seed=0)
        K = env.K
        action = np.zeros(4 * K, dtype=np.float32)
        order = np.arange(K)[::-1].astype(np.float64)  # 反序
        action[3*K:] = order
        env.step(action)
        assert np.array_equal(env.decoding_order, order.astype(np.int64))

    def test_step_3k_backward_compat(self):
        """3K 维 action 仍向后兼容"""
        env = _make_env()
        env.reset(seed=0)
        action = np.zeros(3 * env.K, dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        assert np.isfinite(reward)

    def test_invalid_action_shape_raises(self):
        """非 3K 或 4K 维 action 应报错"""
        env = _make_env()
        env.reset(seed=0)
        with pytest.raises(ValueError):
            env.step(np.zeros(5 * env.K, dtype=np.float32))

    def test_4k_random_actions_no_crash(self):
        """使用 4K 维随机动作连跑 50 步无 crash"""
        env = _make_env(T=100)
        env.reset(seed=0)
        K = env.K
        rng = np.random.default_rng(42)
        for _ in range(50):
            action = np.zeros(4 * K, dtype=np.float32)
            action[:3*K] = rng.uniform(-1, 1, 3*K)
            action[3*K:] = rng.permutation(K)
            obs, reward, term, trunc, info = env.step(action)
            assert np.isfinite(reward)
            assert np.all(np.isfinite(obs))
            if trunc:
                env.reset(seed=0)


# ---------------------------------------------------------------------- #
# User Features Interface Tests
# ---------------------------------------------------------------------- #
@pytest.mark.fast
class TestGetUserFeatures:
    def test_shape(self):
        """get_user_features 返回 (K, 9)"""
        env = _make_env()
        env.reset(seed=0)
        features = env.get_user_features()
        assert features.shape == (env.K, 9)

    def test_dtype(self):
        """get_user_features 返回 float32"""
        env = _make_env()
        env.reset(seed=0)
        features = env.get_user_features()
        assert features.dtype == np.float32

    def test_finite_values(self):
        """所有特征有限"""
        env = _make_env()
        env.reset(seed=0)
        for _ in range(5):
            env.step(env.action_space.sample())
        features = env.get_user_features()
        assert np.all(np.isfinite(features))

    def test_queue_ratio_bounded(self):
        """queue_ratio 特征应在 [0, 1] 范围"""
        env = _make_env(T=50)
        env.reset(seed=0)
        for _ in range(20):
            env.step(env.action_space.sample())
        features = env.get_user_features()
        queue_ratio = features[:, 1]
        assert np.all(queue_ratio >= 0.0)
        assert np.all(queue_ratio <= 1.0 + 1e-6)

    def test_prev_split_reflects_mask(self):
        """特征中的 prev_split 应反映当前 split_mask"""
        env = _make_env()
        env.reset(seed=0)
        mask = np.zeros(env.K, dtype=np.float32)
        mask[2] = 1.0
        mask[7] = 1.0
        env.set_split_mask(mask)
        features = env.get_user_features()
        assert np.array_equal(features[:, 2], mask)

    def test_features_update_after_steps(self):
        """运行几步后特征应有变化（队列/速率/功率等）"""
        env = _make_env(T=50)
        env.reset(seed=0)
        f0 = env.get_user_features().copy()
        for _ in range(10):
            env.step(env.action_space.sample())
        f1 = env.get_user_features()
        # 至少某些特征应该有变化
        assert not np.allclose(f0, f1)

    def test_group_id_norm_in_features(self):
        """user_features 中 group_id_norm 反映 set_group_ids"""
        env = _make_env()
        env.reset(seed=0)
        gids = np.array([0, 1, 2, 3] * 4, dtype=np.int64)
        env.set_group_ids(gids)
        features = env.get_user_features()
        # 第 6 列 = group_id / (G-1) = group_id / 3
        expected = gids / 3.0
        np.testing.assert_allclose(features[:, 6], expected, atol=1e-5)


# ---------------------------------------------------------------------- #
# Beamforming New Functions Tests
# ---------------------------------------------------------------------- #
@pytest.mark.fast
class TestBeamformingNew:
    def test_compute_rzf_precoding_normalized(self):
        """compute_rzf_precoding 输出每列单位范数"""
        from src.env.beamforming import BeamformingDesigner
        designer = BeamformingDesigner(K=16, N_r=16, N_t=2)
        np.random.seed(0)
        # H_hat: (N_r, K)
        H_hat = (np.random.randn(16, 16) + 1j * np.random.randn(16, 16)) / np.sqrt(2)
        W = designer.compute_rzf_precoding(H_hat, sigma2=0.01, P_avg=1.0)
        assert W.shape == (16, 16)
        for k in range(16):
            col_norm = np.linalg.norm(W[:, k])
            np.testing.assert_allclose(col_norm, 1.0, rtol=1e-5)

    def test_compute_common_precoding_groups(self):
        """compute_common_precoding 对不同分组产生不同方向"""
        from src.env.beamforming import BeamformingDesigner
        designer = BeamformingDesigner(K=16, N_r=16, N_t=2)
        np.random.seed(1)
        H_hat = (np.random.randn(16, 16) + 1j * np.random.randn(16, 16)) / np.sqrt(2)
        group_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        split_mask = np.ones(16)
        W_c = designer.compute_common_precoding(H_hat, group_ids, split_mask)
        assert W_c.shape == (16, 4)  # (N_r, G)
        # 每列归一化
        for g in range(4):
            col_norm = np.linalg.norm(W_c[:, g])
            np.testing.assert_allclose(col_norm, 1.0, rtol=1e-5)

    def test_compute_common_precoding_no_split(self):
        """当某组所有用户 split_mask=0 时应使用该组所有用户信道平均"""
        from src.env.beamforming import BeamformingDesigner
        designer = BeamformingDesigner(K=4, N_r=4, N_t=2)
        np.random.seed(2)
        H_hat = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2)
        group_ids = np.array([0, 0, 1, 1])
        split_mask = np.array([0, 0, 1, 1])  # 组 0 无分裂用户
        W_c = designer.compute_common_precoding(H_hat, group_ids, split_mask)
        # 应正常运行不报错，输出 (4, 2) 归一化
        assert W_c.shape == (4, 2)
        for g in range(2):
            col_norm = np.linalg.norm(W_c[:, g])
            np.testing.assert_allclose(col_norm, 1.0, rtol=1e-5)

    def test_get_effective_channel(self):
        """get_effective_channel 从 (K, N_r, N_t) 转为 (N_r, K)"""
        from src.env.beamforming import BeamformingDesigner
        K, N_r, N_t = 8, 8, 2
        designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
        np.random.seed(3)
        H = (np.random.randn(K, N_r, N_t) + 1j * np.random.randn(K, N_r, N_t)) / np.sqrt(2)
        H_eff = designer.get_effective_channel(H)
        assert H_eff.shape == (N_r, K)
        # 有效信道不应为零
        for k in range(K):
            assert np.linalg.norm(H_eff[:, k]) > 1e-8
