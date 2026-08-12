"""测试 Meta-Controller (Self-Attention PPO)

覆盖:
1. MetaActorCritic forward/act/evaluate_actions 的新接口 (含 n_groups 参数)
2. act() 返回 4-tuple (split_mask, group_ids, log_prob, value)
3. group_ids shape (K,) 且值在 [0, G-1]
4. evaluate_actions 含 group_actions (5 返回值)
5. MetaRolloutBuffer 存储 group_action
6. MetaPPOTrainer 含分组熵的更新
"""
import pytest
import torch
import numpy as np
from src.models.meta_controller import MetaActorCritic, MetaRolloutBuffer, MetaPPOTrainer
from src.config import Config

# fast 子集：Meta-Controller 接口/形状/梯度校验，轻量且确定性
pytestmark = pytest.mark.fast


class TestMetaActorCritic:
    def setup_method(self):
        self.K = 16
        self.d_user = 7
        self.n_groups = 4
        self.model = MetaActorCritic(
            K=self.K, d_user=self.d_user, n_groups=self.n_groups
        )

    def test_forward_shape(self):
        """前向传播输出 (split_logits, group_logits, value)"""
        x = torch.randn(4, self.K, self.d_user)
        split_logits, group_logits, value = self.model(x)
        assert split_logits.shape == (4, self.K)
        assert group_logits.shape == (4, self.K, self.n_groups)
        assert value.shape == (4, 1)

    def test_forward_batch_1(self):
        """单 batch 输入"""
        x = torch.randn(1, self.K, self.d_user)
        split_logits, group_logits, value = self.model(x)
        assert split_logits.shape == (1, self.K)
        assert group_logits.shape == (1, self.K, self.n_groups)
        assert value.shape == (1, 1)

    def test_act_returns_4_tuple(self):
        """act 返回 4-tuple (split_mask, group_ids, log_prob, value)"""
        x = torch.randn(self.K, self.d_user)
        result = self.model.act(x)
        assert len(result) == 4
        split_mask, group_ids, log_prob, value = result
        assert split_mask.shape == (self.K,)
        assert group_ids.shape == (self.K,)
        assert log_prob.dim() == 0  # scalar
        assert value.dim() == 0  # scalar

    def test_act_split_mask_binary(self):
        """split_mask 为二值 {0, 1}"""
        x = torch.randn(self.K, self.d_user)
        split_mask, _, _, _ = self.model.act(x)
        assert split_mask.dtype == torch.float32
        assert ((split_mask == 0) | (split_mask == 1)).all()

    def test_act_group_ids_range(self):
        """group_ids 应在 [0, G-1] 范围"""
        x = torch.randn(self.K, self.d_user)
        _, group_ids, _, _ = self.model.act(x)
        assert (group_ids >= 0).all()
        assert (group_ids < self.n_groups).all()
        assert group_ids.dtype == torch.int64

    def test_act_deterministic(self):
        """确定性模式下两次调用应一致"""
        x = torch.randn(self.K, self.d_user)
        mask1, gids1, _, _ = self.model.act(x, deterministic=True)
        mask2, gids2, _, _ = self.model.act(x, deterministic=True)
        assert torch.equal(mask1, mask2)
        assert torch.equal(gids1, gids2)

    def test_act_stochastic(self):
        """随机模式下多次调用应有不同结果 (概率上)"""
        x = torch.randn(self.K, self.d_user)
        torch.manual_seed(1)
        _, gids1, _, _ = self.model.act(x)
        torch.manual_seed(2)
        _, gids2, _, _ = self.model.act(x)
        # 不一定每次不同，但大概率不同
        # 使用多次尝试保证
        different = False
        for seed in range(10):
            torch.manual_seed(seed)
            _, g, _, _ = self.model.act(x)
            if not torch.equal(g, gids1):
                different = True
                break
        assert different

    def test_act_batched(self):
        """act 接受 (1, K, d_user) 输入"""
        x = torch.randn(1, self.K, self.d_user)
        split_mask, group_ids, log_prob, value = self.model.act(x)
        assert split_mask.shape == (self.K,)
        assert group_ids.shape == (self.K,)

    def test_log_prob_finite(self):
        """log_prob 应为有限值"""
        x = torch.randn(self.K, self.d_user)
        _, _, log_prob, _ = self.model.act(x)
        assert torch.isfinite(log_prob)

    def test_evaluate_actions_returns_5(self):
        """evaluate_actions 返回 5 值: (total_log_prob, values, split_entropy, group_entropy, total_entropy)"""
        x = torch.randn(8, self.K, self.d_user)
        split_actions = torch.randint(0, 2, (8, self.K)).float()
        group_actions = torch.randint(0, self.n_groups, (8, self.K))
        result = self.model.evaluate_actions(x, split_actions, group_actions)
        assert len(result) == 5
        total_log_prob, values, split_entropy, group_entropy, total_entropy = result
        assert total_log_prob.shape == (8,)
        assert values.shape == (8,)
        assert split_entropy.shape == (8,)
        assert group_entropy.shape == (8,)
        assert total_entropy.shape == (8,)
        assert (split_entropy >= 0).all()
        assert (group_entropy >= 0).all()
        assert (total_entropy >= 0).all()

    def test_evaluate_actions_backward_compat(self):
        """evaluate_actions 不传 group_actions 时仍能运行 (向后兼容)"""
        x = torch.randn(4, self.K, self.d_user)
        split_actions = torch.randint(0, 2, (4, self.K)).float()
        result = self.model.evaluate_actions(x, split_actions, group_actions=None)
        assert len(result) == 5
        total_log_prob, _, _, _, _ = result
        assert torch.isfinite(total_log_prob).all()

    def test_n_groups_parameter(self):
        """n_groups 参数正确影响模型输出"""
        model_2g = MetaActorCritic(K=self.K, d_user=self.d_user, n_groups=2)
        model_8g = MetaActorCritic(K=self.K, d_user=self.d_user, n_groups=8)
        x = torch.randn(2, self.K, self.d_user)
        _, gl_2, _ = model_2g(x)
        _, gl_8, _ = model_8g(x)
        assert gl_2.shape == (2, self.K, 2)
        assert gl_8.shape == (2, self.K, 8)

    def test_gradient_flows(self):
        """梯度能回传到所有参数"""
        x = torch.randn(2, self.K, self.d_user, requires_grad=False)
        split_logits, group_logits, value = self.model(x)
        loss = split_logits.sum() + group_logits.sum() + value.sum()
        loss.backward()
        for name, p in self.model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


class TestMetaRolloutBuffer:
    def test_store_and_size(self):
        buffer = MetaRolloutBuffer(max_size=10)
        for i in range(5):
            buffer.store(
                user_features=np.random.randn(16, 7),
                action=np.random.randint(0, 2, 16).astype(float),
                log_prob=-0.5,
                reward=1.0,
                value=0.5,
                done=False,
                group_action=np.random.randint(0, 4, 16),
            )
        assert buffer.size == 5

    def test_store_group_action(self):
        """存储 group_action 后可正确读取"""
        buffer = MetaRolloutBuffer(max_size=10)
        ga = np.array([0, 1, 2, 3] * 4, dtype=np.int64)
        buffer.store(
            user_features=np.random.randn(16, 7),
            action=np.ones(16),
            log_prob=-0.5,
            reward=1.0,
            value=0.5,
            done=False,
            group_action=ga,
        )
        assert np.array_equal(buffer.group_actions[0], ga)

    def test_store_without_group_action(self):
        """不传 group_action 时使用零填充"""
        buffer = MetaRolloutBuffer(max_size=10)
        buffer.store(
            user_features=np.random.randn(16, 7),
            action=np.ones(16),
            log_prob=-0.5,
            reward=1.0,
            value=0.5,
            done=False,
        )
        assert np.all(buffer.group_actions[0] == 0)

    def test_max_size_limit(self):
        buffer = MetaRolloutBuffer(max_size=5)
        for i in range(10):
            buffer.store(
                np.random.randn(16, 7), np.ones(16), -0.5, 1.0, 0.5,
                group_action=np.random.randint(0, 4, 16),
            )
        assert buffer.size == 5

    def test_compute_returns(self):
        buffer = MetaRolloutBuffer()
        for i in range(10):
            buffer.store(
                np.random.randn(16, 7), np.ones(16), -0.5, 1.0, 0.5,
                group_action=np.random.randint(0, 4, 16),
            )
        returns, advantages = buffer.compute_returns_and_advantages(0.95, 0.95, 0.0)
        assert len(returns) == 10
        assert len(advantages) == 10

    def test_reset_clears(self):
        buffer = MetaRolloutBuffer()
        for i in range(5):
            buffer.store(
                np.random.randn(16, 7), np.ones(16), -0.5, 1.0, 0.5,
                group_action=np.random.randint(0, 4, 16),
            )
        buffer.reset()
        assert buffer.size == 0

    def test_get_batches_dict(self):
        """get_batches 返回 dict 含 group_actions 键"""
        buffer = MetaRolloutBuffer()
        for i in range(20):
            buffer.store(
                np.random.randn(16, 7), np.ones(16), -0.5, 1.0, 0.5, False,
                group_action=np.random.randint(0, 4, 16),
            )
        batch_count = 0
        for batch in buffer.get_batches(batch_size=8, gamma=0.95, gae_lambda=0.95):
            assert 'user_features' in batch
            assert 'split_actions' in batch
            assert 'group_actions' in batch
            assert 'old_log_probs' in batch
            assert 'returns' in batch
            assert 'advantages' in batch
            assert batch['user_features'].shape[1] == 16  # K
            assert batch['user_features'].shape[2] == 7   # d_user
            assert batch['split_actions'].shape[1] == 16
            assert batch['group_actions'].shape[1] == 16
            batch_count += 1
        assert batch_count > 0


class TestMetaPPOTrainer:
    def test_update(self):
        """测试一次完整 PPO 更新 (含 group 分支)"""
        config = Config()
        meta_ac = MetaActorCritic(
            K=config.system.K,
            d_user=config.hierarchical.meta_d_user,
            hidden_dim=config.hierarchical.meta_hidden_dim,
            n_heads=config.hierarchical.meta_n_heads,
            n_groups=config.hierarchical.n_groups,
        )
        trainer = MetaPPOTrainer(meta_ac, config.hierarchical)
        buffer = MetaRolloutBuffer()

        K = config.system.K
        for i in range(60):
            buffer.store(
                np.random.randn(K, config.hierarchical.meta_d_user),
                np.random.randint(0, 2, K).astype(float),
                -0.5, np.random.randn(), 0.5, False,
                group_action=np.random.randint(0, config.hierarchical.n_groups, K),
            )

        stats = trainer.update(buffer)
        assert 'meta_loss' in stats
        assert 'meta_updates' in stats
        assert stats['meta_updates'] > 0
        assert buffer.size == 0  # buffer should be cleared

    def test_update_loss_finite(self):
        """Meta PPO loss 应为有限值"""
        config = Config()
        meta_ac = MetaActorCritic(
            K=config.system.K,
            d_user=config.hierarchical.meta_d_user,
            hidden_dim=config.hierarchical.meta_hidden_dim,
            n_heads=config.hierarchical.meta_n_heads,
            n_groups=config.hierarchical.n_groups,
        )
        trainer = MetaPPOTrainer(meta_ac, config.hierarchical)
        buffer = MetaRolloutBuffer()

        K = config.system.K
        for i in range(60):
            buffer.store(
                np.random.randn(K, config.hierarchical.meta_d_user),
                np.random.randint(0, 2, K).astype(float),
                -0.5, 1.0, 0.5, False,
                group_action=np.random.randint(0, config.hierarchical.n_groups, K),
            )

        stats = trainer.update(buffer)
        assert np.isfinite(stats['meta_loss'])

    def test_group_entropy_coef_used(self):
        """group_entropy_coef 参数被正确使用"""
        config = Config()
        config.hierarchical.meta_group_entropy_coef = 0.1
        meta_ac = MetaActorCritic(
            K=config.system.K,
            d_user=config.hierarchical.meta_d_user,
            hidden_dim=config.hierarchical.meta_hidden_dim,
            n_heads=config.hierarchical.meta_n_heads,
            n_groups=config.hierarchical.n_groups,
        )
        trainer = MetaPPOTrainer(meta_ac, config.hierarchical)
        assert trainer.group_entropy_coef == 0.1

    def test_full_act_evaluate_consistency(self):
        """act 的 log_prob 与 evaluate_actions 的 log_prob 一致"""
        config = Config()
        K = config.system.K
        meta_ac = MetaActorCritic(
            K=K,
            d_user=config.hierarchical.meta_d_user,
            hidden_dim=config.hierarchical.meta_hidden_dim,
            n_heads=config.hierarchical.meta_n_heads,
            n_groups=config.hierarchical.n_groups,
        )
        meta_ac.eval()
        x = torch.randn(K, config.hierarchical.meta_d_user)
        with torch.no_grad():
            split_mask, group_ids, log_prob_act, _ = meta_ac.act(x)

        # evaluate_actions (batch dim)
        x_b = x.unsqueeze(0)  # (1, K, d)
        split_b = split_mask.unsqueeze(0)
        group_b = group_ids.unsqueeze(0)
        with torch.no_grad():
            log_prob_eval, _, _, _, _ = meta_ac.evaluate_actions(x_b, split_b, group_b)

        assert torch.allclose(log_prob_act, log_prob_eval.squeeze(), atol=1e-5)
