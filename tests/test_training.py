"""训练流程测试"""
import sys
sys.path.insert(0, '/Users/libo/PycharmProjects/rsma-energy-efficient')

import numpy as np
import pytest
import torch

from src.config import Config, PPOConfig
from src.models.actor_critic import ActorCritic
from src.training.buffer import RolloutBuffer
from src.training.ppo import PPOTrainer


# ============================================================
# ActorCritic Tests
# ============================================================

@pytest.mark.fast
class TestActorCritic:
    """ActorCritic网络测试"""

    @pytest.fixture
    def ac(self):
        return ActorCritic(obs_dim=145, action_dim=48, hidden_dim=256, n_layers=3,
                           use_plackett_luce=False)

    def test_forward_shape(self, ac):
        """前向传播输出正确shape"""
        obs = torch.randn(4, 145)
        action_mean, value = ac(obs)
        assert action_mean.shape == (4, 48)
        assert value.shape == (4, 1)

    def test_act_batch_shape(self, ac):
        """act方法batch输入输出shape"""
        obs = torch.randn(8, 145)
        action, log_prob, value = ac.act(obs)
        assert action.shape == (8, 48)
        assert log_prob.shape == (8,)
        assert value.shape == (8,)

    def test_act_single_shape(self, ac):
        """act方法单个输入输出shape"""
        obs = torch.randn(145)
        action, log_prob, value = ac.act(obs)
        assert action.shape == (48,)
        assert log_prob.shape == ()
        assert value.shape == ()

    def test_act_action_range(self, ac):
        """动作在[-1, 1]范围内（tanh）"""
        obs = torch.randn(32, 145)
        action, _, _ = ac.act(obs)
        assert (action >= -1.0).all()
        assert (action <= 1.0).all()

    def test_deterministic_no_randomness(self, ac):
        """确定性模式无随机性"""
        obs = torch.randn(4, 145)
        action1, _, _ = ac.act(obs, deterministic=True)
        action2, _, _ = ac.act(obs, deterministic=True)
        assert torch.allclose(action1, action2)

    def test_stochastic_has_randomness(self, ac):
        """随机模式有随机性"""
        obs = torch.randn(4, 145)
        torch.manual_seed(1)
        action1, _, _ = ac.act(obs, deterministic=False)
        torch.manual_seed(2)
        action2, _, _ = ac.act(obs, deterministic=False)
        # 极小概率相等
        assert not torch.allclose(action1, action2)

    def test_evaluate_actions_shape(self, ac):
        """evaluate_actions输出正确shape"""
        obs = torch.randn(16, 145)
        actions = torch.randn(16, 48).clamp(-0.99, 0.99)
        log_prob, value, entropy, cont_log_prob = ac.evaluate_actions(obs, actions)
        assert log_prob.shape == (16,)
        assert value.shape == (16,)
        assert entropy.shape == (16,)
        assert cont_log_prob.shape == (16,)

    def test_log_prob_finite(self, ac):
        """log_prob应该是有限值"""
        obs = torch.randn(8, 145)
        action, log_prob, _ = ac.act(obs)
        assert torch.isfinite(log_prob).all()

        log_prob2, _, _, _ = ac.evaluate_actions(obs, action)
        assert torch.isfinite(log_prob2).all()

    def test_actor_critic_params_separation(self, ac):
        """actor和critic参数正确分离"""
        actor_params = set(id(p) for p in ac.actor_params())
        critic_params = set(id(p) for p in ac.critic_params())
        # critic_head的参数不应出现在actor_params中
        for p in ac.critic_head.parameters():
            assert id(p) not in actor_params


# ============================================================
# RolloutBuffer Tests
# ============================================================

@pytest.mark.fast
class TestRolloutBuffer:
    """RolloutBuffer测试"""

    @pytest.fixture
    def buffer(self):
        return RolloutBuffer(
            rollout_steps=64, obs_dim=145, action_dim=48,
            gamma=0.99, gae_lambda=0.95
        )

    def test_add_and_pos(self, buffer):
        """添加数据后pos正确"""
        obs = np.zeros(145)
        action = np.zeros(48)
        buffer.add(obs, action, reward=1.0, value=0.5, log_prob=-1.0, done=False)
        assert buffer.pos == 1

    def test_full_buffer(self, buffer):
        """填满后full标记"""
        for i in range(64):
            buffer.add(np.zeros(145), np.zeros(48), 1.0, 0.5, -1.0, False)
        assert buffer.full

    def test_overflow_raises(self, buffer):
        """buffer满后再添加应报错"""
        for i in range(64):
            buffer.add(np.zeros(145), np.zeros(48), 1.0, 0.5, -1.0, False)
        with pytest.raises(RuntimeError):
            buffer.add(np.zeros(145), np.zeros(48), 1.0, 0.5, -1.0, False)

    def test_compute_returns(self, buffer):
        """GAE计算后returns和advantages形状正确"""
        for i in range(64):
            buffer.add(np.random.randn(145), np.random.randn(48),
                       reward=np.random.randn(), value=np.random.randn(),
                       log_prob=np.random.randn(), done=False)
        buffer.compute_returns_and_advantages(last_value=0.0)

        # 检查advantages已归一化（均值约0，标准差约1）
        adv = buffer.advantages[:64]
        assert abs(adv.mean()) < 0.1
        assert abs(adv.std() - 1.0) < 0.1

    def test_get_batches(self, buffer):
        """get_batches生成正确的batch"""
        for i in range(64):
            buffer.add(np.random.randn(145), np.random.randn(48),
                       reward=1.0, value=0.5, log_prob=-1.0, done=False)
        buffer.compute_returns_and_advantages(last_value=0.0)

        batch_count = 0
        total_samples = 0
        for batch in buffer.get_batches(batch_size=16):
            assert "obs" in batch
            assert "actions" in batch
            assert "old_log_probs" in batch
            assert "returns" in batch
            assert "advantages" in batch
            assert batch["obs"].shape[1] == 145
            assert batch["actions"].shape[1] == 48
            batch_count += 1
            total_samples += batch["obs"].shape[0]
        assert total_samples == 64
        assert batch_count == 4

    def test_clear(self, buffer):
        """clear后pos归零"""
        buffer.add(np.zeros(145), np.zeros(48), 1.0, 0.5, -1.0, False)
        buffer.clear()
        assert buffer.pos == 0
        assert not buffer.full


# ============================================================
# PPOTrainer Tests
# ============================================================

@pytest.mark.fast
class TestPPOTrainer:
    """PPOTrainer测试"""

    @pytest.fixture
    def trainer_and_buffer(self):
        config = PPOConfig()
        config.n_epochs = 2
        config.batch_size = 32
        config.rollout_steps = 64

        ac = ActorCritic(obs_dim=145, action_dim=48, hidden_dim=64, n_layers=2,
                           use_plackett_luce=False)
        trainer = PPOTrainer(ac, config)

        buffer = RolloutBuffer(
            rollout_steps=64, obs_dim=145, action_dim=48,
            gamma=config.gamma, gae_lambda=config.gae_lambda
        )
        return trainer, buffer, ac

    def test_single_update(self, trainer_and_buffer):
        """PPO单步更新不crash，loss为有限数"""
        trainer, buffer, ac = trainer_and_buffer

        # 填充buffer
        for i in range(64):
            obs = torch.randn(145)
            with torch.no_grad():
                action, log_prob, value = ac.act(obs)
            buffer.add(
                obs.numpy(), action.numpy(),
                reward=np.random.randn(), value=value.item(),
                log_prob=log_prob.item(), done=False
            )
        buffer.compute_returns_and_advantages(last_value=0.0)

        # 执行更新
        info = trainer.update(buffer)

        assert np.isfinite(info['policy_loss'])
        assert np.isfinite(info['value_loss'])
        assert np.isfinite(info['entropy'])
        assert np.isfinite(info['clip_fraction'])
        assert np.isfinite(info['approx_kl'])
        assert info['entropy'] > 0  # 初始熵应该为正

    def test_update_returns_dict(self, trainer_and_buffer):
        """更新返回正确的字典键"""
        trainer, buffer, ac = trainer_and_buffer

        for i in range(64):
            obs = torch.randn(145)
            with torch.no_grad():
                action, log_prob, value = ac.act(obs)
            buffer.add(
                obs.numpy(), action.numpy(),
                reward=1.0, value=value.item(),
                log_prob=log_prob.item(), done=False
            )
        buffer.compute_returns_and_advantages(last_value=0.0)

        info = trainer.update(buffer)
        expected_keys = {'policy_loss', 'value_loss', 'entropy', 'clip_fraction',
                         'approx_kl', 'early_stopped', 'lr_actor', 'kl_decay_factor'}
        # 基础键必须存在；trainer 可能额外返回 HEK/Lagrangian 等扩展键
        # （hek_coef / hek_anchor_loss / power_incentive_loss / cost_value_loss 等）
        assert expected_keys <= set(info.keys())


# ============================================================
# Plackett-Luce Mode Tests
# ============================================================

@pytest.mark.fast
class TestActorCriticPL:
    """Plackett-Luce 模式的 ActorCritic 测试"""

    @pytest.fixture
    def ac_pl(self):
        return ActorCritic(obs_dim=161, action_dim=64, hidden_dim=64, n_layers=2,
                           use_plackett_luce=True)

    def test_forward_shape_pl(self, ac_pl):
        """PL模式前向传播输出正确 shape"""
        obs = torch.randn(4, 161)
        action_mean, value = ac_pl(obs)
        assert action_mean.shape == (4, 64)
        assert value.shape == (4, 1)

    def test_act_returns_4_tuple_pl(self, ac_pl):
        """PL模式 act 返回 5-tuple (含 cont_log_prob)"""
        obs = torch.randn(4, 161)
        result = ac_pl.act(obs)
        assert len(result) == 5
        action_cont, decoding_order, log_prob, value, cont_log_prob = result
        assert action_cont.shape == (4, 48)  # 3K=48
        assert decoding_order.shape == (4, 16)  # K=16
        assert log_prob.shape == (4,)
        assert value.shape == (4,)
        assert cont_log_prob.shape == (4,)

    def test_act_single_pl(self, ac_pl):
        """PL模式单样本 act"""
        obs = torch.randn(161)
        action_cont, decoding_order, log_prob, value, cont_log_prob = ac_pl.act(obs)
        assert action_cont.shape == (48,)
        assert decoding_order.shape == (16,)
        assert log_prob.shape == ()
        assert value.shape == ()
        assert cont_log_prob.shape == ()

    def test_decoding_order_is_permutation(self, ac_pl):
        """解码顺序应为 0..K-1 的排列"""
        obs = torch.randn(8, 161)
        _, decoding_order, _, _, _ = ac_pl.act(obs)
        for i in range(8):
            perm = decoding_order[i].numpy()
            assert set(perm.tolist()) == set(range(16))

    def test_action_continuous_range(self, ac_pl):
        """PL模式连续动作在 [-1, 1]"""
        obs = torch.randn(16, 161)
        action_cont, _, _, _, _ = ac_pl.act(obs)
        assert (action_cont >= -1.0).all()
        assert (action_cont <= 1.0).all()

    def test_evaluate_actions_pl(self, ac_pl):
        """PL模式 evaluate_actions 传 decoding_orders"""
        obs = torch.randn(8, 161)
        actions = torch.randn(8, 48).clamp(-0.99, 0.99)
        # 生成合法排列
        decoding_orders = torch.stack([torch.randperm(16) for _ in range(8)])
        log_prob, value, entropy, cont_log_prob = ac_pl.evaluate_actions(obs, actions, decoding_orders)
        assert log_prob.shape == (8,)
        assert value.shape == (8,)
        assert entropy.shape == (8,)
        assert cont_log_prob.shape == (8,)
        assert torch.isfinite(log_prob).all()
        # cont_log_prob 应是连续部分，与 total log_prob 不同
        assert not torch.allclose(log_prob, cont_log_prob)

    def test_evaluate_actions_without_decoding_orders(self, ac_pl):
        """PL模式不传 decoding_orders 时向后兼容"""
        obs = torch.randn(8, 161)
        actions = torch.randn(8, 48).clamp(-0.99, 0.99)
        log_prob, value, entropy, cont_log_prob = ac_pl.evaluate_actions(obs, actions, None)
        assert log_prob.shape == (8,)
        assert torch.isfinite(log_prob).all()
        # 不传 decoding_orders 时 cont_log_prob == log_prob
        assert torch.allclose(log_prob, cont_log_prob)

    def test_deterministic_pl(self, ac_pl):
        """确定性模式 PL"""
        obs = torch.randn(4, 161)
        a1, o1, _, _, _ = ac_pl.act(obs, deterministic=True)
        a2, o2, _, _, _ = ac_pl.act(obs, deterministic=True)
        assert torch.allclose(a1, a2)
        assert torch.equal(o1, o2)


@pytest.mark.fast
class TestRolloutBufferPL:
    """PL模式的 RolloutBuffer 测试"""

    @pytest.fixture
    def buffer_pl(self):
        return RolloutBuffer(
            rollout_steps=32, obs_dim=161, action_dim=48,
            gamma=0.99, gae_lambda=0.95, decoding_order_dim=16
        )

    def test_decoding_orders_storage(self, buffer_pl):
        """buffer 储存 decoding_orders"""
        assert buffer_pl.decoding_orders is not None
        assert buffer_pl.decoding_orders.shape == (32, 16)

    def test_add_with_decoding_order(self, buffer_pl):
        """添加包含 decoding_order 的 transition"""
        obs = np.zeros(161)
        action = np.zeros(48)
        order = np.arange(16)
        buffer_pl.add(obs, action, reward=1.0, value=0.5, log_prob=-1.0,
                      done=False, decoding_order=order)
        assert buffer_pl.pos == 1
        assert np.array_equal(buffer_pl.decoding_orders[0], order)

    def test_get_batches_includes_decoding_orders(self, buffer_pl):
        """get_batches 产生的 batch 包含 decoding_orders"""
        for i in range(32):
            buffer_pl.add(
                np.random.randn(161), np.random.randn(48),
                reward=1.0, value=0.5, log_prob=-1.0, done=False,
                decoding_order=np.random.permutation(16)
            )
        buffer_pl.compute_returns_and_advantages(last_value=0.0)
        for batch in buffer_pl.get_batches(batch_size=16):
            assert "decoding_orders" in batch
            assert batch["decoding_orders"].shape[1] == 16
            assert batch["decoding_orders"].dtype == torch.long
            break

    def test_buffer_without_pl(self):
        """decoding_order_dim=0 时不存储 decoding_orders"""
        buf = RolloutBuffer(rollout_steps=16, obs_dim=161, action_dim=48,
                            gamma=0.99, gae_lambda=0.95, decoding_order_dim=0)
        assert buf.decoding_orders is None
        buf.add(np.zeros(161), np.zeros(48), 1.0, 0.5, -1.0, False)
        assert buf.pos == 1


@pytest.mark.fast
class TestPPOTrainerPL:
    """PPO Trainer 与 PL 模式兑容性测试"""

    def test_ppo_update_with_pl(self):
        """PPO 更新在 PL 模式下正常工作"""
        config = PPOConfig()
        config.n_epochs = 2
        config.batch_size = 16
        config.rollout_steps = 32

        ac = ActorCritic(obs_dim=161, action_dim=64, hidden_dim=64, n_layers=2,
                         use_plackett_luce=True)
        trainer = PPOTrainer(ac, config)
        buffer = RolloutBuffer(
            rollout_steps=32, obs_dim=161, action_dim=48,
            gamma=config.gamma, gae_lambda=config.gae_lambda,
            decoding_order_dim=16
        )

        for i in range(32):
            obs = torch.randn(161)
            with torch.no_grad():
                action_cont, dec_order, log_prob, value, cont_log_prob = ac.act(obs)
            buffer.add(
                obs.numpy(), action_cont.numpy(),
                reward=np.random.randn(), value=value.item(),
                log_prob=log_prob.item(), done=False,
                decoding_order=dec_order.numpy(),
                cont_log_prob=cont_log_prob.item(),
            )
        buffer.compute_returns_and_advantages(last_value=0.0)

        info = trainer.update(buffer)
        assert np.isfinite(info['policy_loss'])
        assert np.isfinite(info['value_loss'])
        assert np.isfinite(info['entropy'])


@pytest.mark.fast
class TestBCPretrainPL:
    """Solver-Expert BC 联合训练通道（连续头 + PL 排序头 + 锚定）"""

    K = 16

    def _make_trainer(self):
        config = PPOConfig()
        ac = ActorCritic(obs_dim=161, action_dim=64, hidden_dim=64, n_layers=2,
                         use_plackett_luce=True)
        trainer = PPOTrainer(ac, config)
        trainer.hek_bc_epochs = 5
        trainer.hek_bc_lr = 1e-3
        return trainer

    def _make_demos(self, n):
        rng = np.random.RandomState(0)
        obs = rng.randn(n, 161).astype(np.float32)
        act = np.clip(rng.randn(n, 3 * self.K) * 0.3, -0.9, 0.9).astype(np.float32)
        orders = np.stack(
            [rng.permutation(self.K) for _ in range(n)]).astype(np.int64)
        return obs, act, orders

    def test_bc_pretrain_pl_joint(self):
        """传入 orders 时联合训练：缓存 hek_orders 且 demo 似然上升"""
        trainer = self._make_trainer()
        obs, act, orders = self._make_demos(32)

        obs_t = torch.from_numpy(obs)
        act_t = torch.from_numpy(act)
        orders_t = torch.from_numpy(orders)
        with torch.no_grad():
            lp_before, _, _, _ = trainer.ac.evaluate_actions(obs_t, act_t, orders_t)
        info = trainer.bc_pretrain(obs, act, orders_np=orders)
        with torch.no_grad():
            lp_after, _, _, _ = trainer.ac.evaluate_actions(obs_t, act_t, orders_t)

        assert np.isfinite(info['bc_loss'])
        assert trainer.hek_orders is not None
        assert trainer.hek_orders.shape == (32, self.K)
        assert trainer.hek_obs.shape == (32, 161)
        assert float(lp_after.mean()) > float(lp_before.mean())

    def test_bc_pretrain_without_orders_legacy(self):
        """orders_np=None 保持旧路径：不缓存 orders，似然同样上升"""
        trainer = self._make_trainer()
        obs, act, _ = self._make_demos(16)

        obs_t = torch.from_numpy(obs)
        act_t = torch.from_numpy(act)
        with torch.no_grad():
            _, _, _, lp_before = trainer.ac.evaluate_actions(obs_t, act_t, None)
        info = trainer.bc_pretrain(obs, act)
        with torch.no_grad():
            _, _, _, lp_after = trainer.ac.evaluate_actions(obs_t, act_t, None)

        assert np.isfinite(info['bc_loss'])
        assert trainer.hek_orders is None
        assert float(lp_after.mean()) > float(lp_before.mean())

    def test_anchor_loss_with_orders(self):
        """update() 锚定路径：hek_orders 生效时 hek_anchor_loss 有限且系数>0"""
        trainer = self._make_trainer()
        obs, act, orders = self._make_demos(16)
        trainer.bc_pretrain(obs, act, orders_np=orders)
        trainer.hek_enabled = True
        trainer.hek_anchor_updates = 3
        trainer.hek_anchor_coef = 0.2

        config = PPOConfig()
        config.n_epochs = 1
        config.batch_size = 16
        ac = trainer.ac
        buffer = RolloutBuffer(
            rollout_steps=16, obs_dim=161, action_dim=48,
            gamma=config.gamma, gae_lambda=config.gae_lambda,
            decoding_order_dim=self.K
        )
        for _ in range(16):
            o = torch.randn(161)
            with torch.no_grad():
                action_cont, dec_order, log_prob, value, cont_log_prob = ac.act(o)
            buffer.add(
                o.numpy(), action_cont.numpy(),
                reward=float(np.random.randn()), value=value.item(),
                log_prob=log_prob.item(), done=False,
                decoding_order=dec_order.numpy(),
                cont_log_prob=cont_log_prob.item(),
            )
        buffer.compute_returns_and_advantages(last_value=0.0)

        info = trainer.update(buffer)
        assert np.isfinite(info['hek_anchor_loss'])
        assert info['hek_coef'] > 0.0
        assert np.isfinite(info['policy_loss'])


# ============================================================
# Integration Test: Mini Training Loop
# ============================================================

@pytest.mark.fast
class TestMiniTrainingLoop:
    """3步mini训练循环集成测试"""

    def test_env_collect_update_loop(self):
        """环境reset→collect→update能跑通 (非PL模式)"""
        from src.env.rsma_env import RSMAEnergyEnv

        config = Config()
        config.system.T = 20  # 短episode
        env = RSMAEnergyEnv(config)

        ac = ActorCritic(
            obs_dim=env.obs_dim, action_dim=env.action_dim,
            hidden_dim=64, n_layers=2, use_plackett_luce=False
        )

        ppo_config = PPOConfig()
        ppo_config.n_epochs = 2
        ppo_config.batch_size = 16
        ppo_config.rollout_steps = 32

        trainer = PPOTrainer(ac, ppo_config)
        buffer = RolloutBuffer(
            rollout_steps=32, obs_dim=env.obs_dim, action_dim=env.action_dim,
            gamma=ppo_config.gamma, gae_lambda=ppo_config.gae_lambda
        )

        obs, info = env.reset(seed=42)

        # 收集数据
        for step in range(32):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, log_prob, value = ac.act(obs_tensor)

            action_np = action.numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            buffer.add(obs, action_np, reward, value.item(), log_prob.item(), done)

            if done:
                obs, info = env.reset()
            else:
                obs = next_obs

        # Bootstrap
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            _, _, last_value = ac.act(obs_tensor, deterministic=True)

        buffer.compute_returns_and_advantages(last_value.item())

        # PPO更新
        update_info = trainer.update(buffer)

        assert np.isfinite(update_info['policy_loss'])
        assert np.isfinite(update_info['value_loss'])

    def test_env_collect_update_loop_pl(self):
        """环境reset→collect→update能跑通 (PL模式, 4K action)"""
        from src.env.rsma_env import RSMAEnergyEnv

        config = Config()
        config.system.T = 20
        env = RSMAEnergyEnv(config)
        K = env.K

        ac = ActorCritic(
            obs_dim=env.obs_dim, action_dim=4*K,
            hidden_dim=64, n_layers=2, use_plackett_luce=True
        )

        ppo_config = PPOConfig()
        ppo_config.n_epochs = 2
        ppo_config.batch_size = 16
        ppo_config.rollout_steps = 32

        trainer = PPOTrainer(ac, ppo_config)
        buffer = RolloutBuffer(
            rollout_steps=32, obs_dim=env.obs_dim, action_dim=3*K,
            gamma=ppo_config.gamma, gae_lambda=ppo_config.gae_lambda,
            decoding_order_dim=K
        )

        obs, info = env.reset(seed=42)

        for step in range(32):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action_cont, dec_order, log_prob, value, cont_log_prob = ac.act(obs_tensor)

            # 构造 4K 维 action 传给 env
            action_4k = np.concatenate([action_cont.numpy(), dec_order.numpy().astype(np.float64)])
            next_obs, reward, terminated, truncated, info = env.step(action_4k)
            done = terminated or truncated

            buffer.add(
                obs, action_cont.numpy(), reward, value.item(),
                log_prob.item(), done, decoding_order=dec_order.numpy(),
                cont_log_prob=cont_log_prob.item(),
            )

            if done:
                obs, info = env.reset()
            else:
                obs = next_obs

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            _, _, _, last_value, _ = ac.act(obs_tensor, deterministic=True)

        buffer.compute_returns_and_advantages(last_value.item())

        update_info = trainer.update(buffer)
        assert np.isfinite(update_info['policy_loss'])
        assert np.isfinite(update_info['value_loss'])
