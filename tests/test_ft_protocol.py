"""M5 SFTP（安全微调协议）单元测试。

覆盖：
- F0: 锚定梯度注入助手 _anchor_grad_into（lex 路径缺口的修复核心）
- F2: critic 预热（actor_warmup_updates 期间 actor+log_std 严格不动）
- F5: 评估守卫 mini-eval（_guard_run_point/_guard_check）与回滚助手（_guard_rollback）
- R1: on-policy relabel 收集（mode="relabel" 形状与守卫条件）

Meta 冻结（F4）为 train() 内 4 行门控，由冒烟日志 [MetaFreeze]/MetaUpd=0 验证。
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.env.rsma_env import RSMAEnergyEnv  # noqa: E402
from src.models.actor_critic import ActorCriticAttn  # noqa: E402
from src.models.meta_controller import MetaActorCritic  # noqa: E402
from src.training.buffer import RolloutBuffer  # noqa: E402
from src.training.main import (  # noqa: E402
    _HM_TARGETS,
    _guard_check,
    _guard_rollback,
    _guard_run_point,
    _hm_bucket,
    _hm_init_state,
    _hm_sample_snr,
    _solver_expert_collect,
)
from src.training.ppo import PPOTrainer  # noqa: E402

pytestmark = pytest.mark.fast

K_MINI = 4
T_MINI = 8


def _mini_config():
    cfg = Config()
    cfg.system.K = K_MINI
    cfg.system.N_r = 4
    cfg.system.T = T_MINI
    cfg.system.snr_domain_randomization = False
    cfg.system.adaptive_lambda = False
    cfg.system.sigma2 = 1.0
    # QoS 保持默认开启：obs=13K+3、worker d_user=13、meta d_user=12 的口径一致性
    cfg.ppo.use_bucket_experts = False
    cfg.ppo.use_mean_field_critic = False
    cfg.ppo.batch_size = 8
    cfg.ppo.n_epochs = 1
    cfg.ppo.rollout_steps = 16
    cfg.device = "cpu"
    return cfg


def _mini_env(cfg, seed=0):
    env = RSMAEnergyEnv(cfg)
    env.reset(seed=seed)
    return env


def _mini_worker(cfg, env):
    return ActorCriticAttn(
        obs_dim=env.obs_dim,
        action_dim=4 * cfg.system.K,
        K=cfg.system.K,
        d_user=cfg.ppo.worker_d_user,          # 13（QoS 开）
        d_global=env.obs_dim - cfg.system.K * cfg.ppo.worker_d_user,
        hidden_dim=16, n_heads=2, n_attn_layers=1,
        use_plackett_luce=True,
        use_mean_field_critic=False,
        use_bucket_experts=False,
    )


def _mini_meta(cfg):
    return MetaActorCritic(
        K=cfg.system.K,
        d_user=cfg.hierarchical.meta_d_user,   # 12
        hidden_dim=16, n_heads=2, n_groups=4,
        group_level_alpha=True,
    )


def _fill_buffer(cfg, env, buffer, n=None):
    n = n or buffer.rollout_steps
    K = cfg.system.K
    for i in range(n):
        buffer.add(
            np.random.randn(env.obs_dim).astype(np.float32),
            np.random.uniform(-0.9, 0.9, 3 * K).astype(np.float32),
            reward=0.1, value=0.5, log_prob=-1.0,
            done=(i == n - 1),
            decoding_order=np.arange(K, dtype=np.int64),
            cont_log_prob=-0.5, snr_db=10.0,
        )
    buffer.compute_returns_and_advantages(last_value=0.5)


def _demo_tensors(env, K, n=8):
    obs = np.random.randn(n, env.obs_dim).astype(np.float32)
    act = np.random.uniform(-0.9, 0.9, (n, 3 * K)).astype(np.float32)
    orders = np.tile(np.arange(K, dtype=np.int64), (n, 1))
    return torch.as_tensor(obs), torch.as_tensor(act), torch.as_tensor(orders)


class TestAnchorGradInjection:
    """F0: 锚定梯度助手——lex 路径此前缺失该通道（M4 失败头号根因）。"""

    def test_anchor_grad_descent_lowers_demo_nll(self):
        cfg = _mini_config()
        env = _mini_env(cfg)
        ac = _mini_worker(cfg, env)
        trainer = PPOTrainer(ac, cfg.ppo)
        obs, act, orders = _demo_tensors(env, cfg.system.K)
        trainer.hek_obs, trainer.hek_act, trainer.hek_orders = obs, act, orders

        params = list(ac.actor_params()) + list(ac.log_std_params())
        opt = torch.optim.SGD(params, lr=1e-2)

        def demo_nll():
            lp, _, _, _ = ac.evaluate_actions(obs, act.clamp(-0.999, 0.999), orders)
            return float((-lp).mean().detach())

        before = demo_nll()
        last_val = 0.0
        for _ in range(5):
            opt.zero_grad()
            for p in params:
                p.grad = torch.zeros_like(p)
            last_val = trainer._anchor_grad_into(params, 1.0, 8)
            opt.step()
        after = demo_nll()
        assert last_val > 0.0                      # 返回演示 NLL
        assert after < before                      # 沿锚定梯度下降 → NLL 降低
        assert all(torch.isfinite(p).all() for p in params)

    def test_anchor_grad_noop_without_demos(self):
        cfg = _mini_config()
        env = _mini_env(cfg)
        ac = _mini_worker(cfg, env)
        trainer = PPOTrainer(ac, cfg.ppo)
        params = list(ac.actor_params())
        for p in params:
            p.grad = torch.zeros_like(p)
        val = trainer._anchor_grad_into(params, 0.2, 4)
        assert val == 0.0
        assert all(float(p.grad.abs().sum()) == 0.0 for p in params)


class TestCriticWarmup:
    """F2: 预热期 actor+log_std 严格不动，critic 正常学习；预热结束后 actor 恢复更新。"""

    def test_warmup_freezes_actor_trains_critic(self):
        torch.manual_seed(0)
        np.random.seed(0)
        cfg = _mini_config()
        cfg.ppo.actor_warmup_updates = 1
        env = _mini_env(cfg)
        ac = _mini_worker(cfg, env)
        trainer = PPOTrainer(ac, cfg.ppo)

        actor_before = [p.detach().clone() for p in ac.actor_params()]
        logstd_before = ac.log_std.detach().clone()
        critic_before = [p.detach().clone() for p in ac.critic_params()]

        buffer = RolloutBuffer(
            rollout_steps=cfg.ppo.rollout_steps, obs_dim=env.obs_dim,
            action_dim=3 * cfg.system.K, gamma=0.99, gae_lambda=0.95,
            device="cpu", decoding_order_dim=cfg.system.K)
        _fill_buffer(cfg, env, buffer)
        info = trainer.update(buffer)
        assert info["in_warmup"] == 1.0
        for pb, pa in zip(actor_before, ac.actor_params()):
            assert torch.equal(pb, pa.detach()), "warmup 期 actor 参数必须严格不动"
        assert torch.equal(logstd_before, ac.log_std.detach())
        assert any(not torch.equal(pb, pa.detach())
                   for pb, pa in zip(critic_before, ac.critic_params())), \
            "warmup 期 critic 必须学习"

        # 预热结束：第二次 update actor 恢复移动
        buffer.clear()
        _fill_buffer(cfg, env, buffer)
        info2 = trainer.update(buffer)
        assert info2["in_warmup"] == 0.0
        assert any(not torch.equal(pb, pa.detach())
                   for pb, pa in zip(actor_before, ac.actor_params())), \
            "预热结束后 actor 应恢复更新"

    def test_warmup_disabled_by_default(self):
        cfg = _mini_config()
        env = _mini_env(cfg)
        ac = _mini_worker(cfg, env)
        trainer = PPOTrainer(ac, cfg.ppo)
        assert trainer.actor_warmup_updates == 0
        buffer = RolloutBuffer(
            rollout_steps=cfg.ppo.rollout_steps, obs_dim=env.obs_dim,
            action_dim=3 * cfg.system.K, gamma=0.99, gae_lambda=0.95,
            device="cpu", decoding_order_dim=cfg.system.K)
        _fill_buffer(cfg, env, buffer)
        info = trainer.update(buffer)
        assert info["in_warmup"] == 0.0


class TestEvalGuard:
    """F5: 守卫 mini-eval 与回滚助手。"""

    def test_guard_run_point_metrics(self):
        cfg = _mini_config()
        cfg.ppo.eval_guard_episodes = 1
        env_g = _mini_env(cfg)
        env_g.snr_domain_randomization = False
        ac = _mini_worker(cfg, env_g)
        meta = _mini_meta(cfg)
        dv, q = _guard_run_point(
            env_g, ac, meta, torch.device("cpu"), cfg, 0.0, 1, 2000)
        assert 0.0 <= dv <= 1.0
        assert q >= 0.0

    def test_guard_check_structure(self):
        cfg = _mini_config()
        cfg.ppo.eval_guard_episodes = 1
        env_g = _mini_env(cfg)
        env_g.snr_domain_randomization = False
        ac = _mini_worker(cfg, env_g)
        meta = _mini_meta(cfg)
        res = _guard_check(env_g, ac, meta, torch.device("cpu"), cfg)
        assert set(res) >= {"avg_deliv", "points_met", "ok", "per_point"}
        assert len(res["per_point"]) == 3
        assert res["points_met"] in (0, 1, 2, 3)
        assert isinstance(res["ok"], bool)

    def test_guard_rollback_restores_and_halves_lr(self):
        cfg = _mini_config()
        env = _mini_env(cfg)
        ac = _mini_worker(cfg, env)
        meta = _mini_meta(cfg)
        trainer = PPOTrainer(ac, cfg.ppo)

        actor_ids = ({id(p) for p in ac.actor_params()}
                     | {id(p) for p in ac.log_std_params()})
        lr_groups = [i for i, g in enumerate(trainer.optimizer.param_groups)
                     if any(id(p) in actor_ids for p in g["params"])]
        assert lr_groups == [0, 2]  # 非桶专家: [actor, critic, log_std]

        good = ({k: v.detach().cpu().clone() for k, v in ac.state_dict().items()},
                {k: v.detach().cpu().clone() for k, v in meta.state_dict().items()})
        # 扰动 actor
        with torch.no_grad():
            for p in ac.actor_params():
                p.add_(torch.randn_like(p) * 0.5)
        lr0_before = float(trainer.optimizer.param_groups[0]["lr"])
        base0_before = float(trainer._base_lrs[0])

        h = _guard_rollback(ac, meta, trainer, good, 0, lr_groups, cfg)
        assert h == 1
        for k, v in good[0].items():
            assert torch.equal(ac.state_dict()[k].cpu(), v), "回滚必须恢复 last-good 权重"
        assert float(trainer.optimizer.param_groups[0]["lr"]) == pytest.approx(lr0_before * 0.5)
        assert float(trainer._base_lrs[0]) == pytest.approx(base0_before * 0.5)

        # 超过 max_halvings(3) → actor lr 冻结为 0
        h = _guard_rollback(ac, meta, trainer, good, h, lr_groups, cfg)   # 2
        h = _guard_rollback(ac, meta, trainer, good, h, lr_groups, cfg)   # 3
        h = _guard_rollback(ac, meta, trainer, good, h, lr_groups, cfg)   # 4 > 3
        assert h == 4
        assert float(trainer.optimizer.param_groups[0]["lr"]) == 0.0
        assert float(trainer.optimizer.param_groups[2]["lr"]) == 0.0
        assert float(trainer._base_lrs[0]) == 0.0
        # critic 组不受影响
        assert float(trainer.optimizer.param_groups[1]["lr"]) > 0.0


class TestRelabelCollect:
    """R1: on-policy relabel 收集（ExIt 第二迭代）。"""

    def test_relabel_requires_student(self):
        cfg = _mini_config()
        env = _mini_env(cfg)
        meta = _mini_meta(cfg)
        with pytest.raises(ValueError):
            _solver_expert_collect(env, cfg, meta, torch.device("cpu"),
                                   n_episodes=1, n_steps=2, mode="relabel")

    def test_relabel_shapes_and_progress(self):
        cfg = _mini_config()
        cfg.system.T = 6
        env = _mini_env(cfg)
        ac = _mini_worker(cfg, env)
        meta = _mini_meta(cfg)
        obs, act, orders = _solver_expert_collect(
            env, cfg, meta, torch.device("cpu"),
            n_episodes=1, n_steps=2, lr=0.1,
            mode="relabel", student_ac=ac)
        K = cfg.system.K
        assert obs.shape == (6, env.obs_dim)
        assert act.shape == (6, 3 * K)
        assert orders.shape == (6, K)
        assert obs.dtype == np.float32 and orders.dtype == np.int64
        # teacher 模式（M4 口径）不受影响
        obs2, act2, orders2 = _solver_expert_collect(
            env, cfg, meta, torch.device("cpu"),
            n_episodes=1, n_steps=2, lr=0.1)
        assert obs2.shape == obs.shape


class TestKLtoBCTrustRegion:
    """M5.2-F6: KL(π_θ‖π_BC) 显式信任域。"""

    def _batch(self, cfg, env):
        obs = torch.randn(8, env.obs_dim)
        K = cfg.system.K
        actions = (torch.rand(8, 3 * K) * 1.8 - 0.9)
        orders = torch.arange(K).unsqueeze(0).repeat(8, 1)
        return obs, actions, orders

    def test_disabled_without_reference(self):
        cfg = _mini_config()
        env = _mini_env(cfg)
        ac = _mini_worker(cfg, env)
        trainer = PPOTrainer(ac, cfg.ppo)  # kl_bc_coef 默认 0
        assert trainer.set_bc_reference(ac.state_dict()) is False
        assert trainer._bc_ac is None
        obs, actions, orders = self._batch(cfg, env)
        lp, _, _, _ = ac.evaluate_actions(obs, actions, orders)
        assert trainer._klbc_term(obs, actions, orders, None, lp) is None

    def test_reference_frozen_and_term_semantics(self):
        cfg = _mini_config()
        cfg.ppo.kl_bc_coef = 1.0
        env = _mini_env(cfg)
        ac = _mini_worker(cfg, env)
        trainer = PPOTrainer(ac, cfg.ppo)
        assert trainer.set_bc_reference(ac.state_dict()) is True
        assert all(not p.requires_grad for p in trainer._bc_ac.parameters())

        obs, actions, orders = self._batch(cfg, env)
        lp, _, _, _ = ac.evaluate_actions(obs, actions, orders)
        term0 = trainer._klbc_term(obs, actions, orders, None, lp)
        assert term0 is not None
        assert float(term0.detach()) == pytest.approx(0.0, abs=1e-6), "同策略时信任域项必须为 0"

        # 朝"降低所采动作 log-prob"的方向走一步 → 漂离 BC，项严格为正
        (-lp.mean()).backward()
        with torch.no_grad():
            for p in ac.actor_params():
                if p.grad is not None:
                    p.add_(p.grad * 0.5)
            ac.zero_grad(set_to_none=True)
        lp2, _, _, _ = ac.evaluate_actions(obs, actions, orders)
        term2 = trainer._klbc_term(obs, actions, orders, None, lp2)
        assert float(term2.detach()) > 0.0
        assert term2.requires_grad

        # _trust_grad_into 注入的梯度必须 == coef × d(term)/d(params)
        params = list(ac.actor_params()) + list(ac.log_std_params())
        ref_grads = torch.autograd.grad(
            term2, params, allow_unused=True, retain_graph=True)
        for p in params:
            p.grad = torch.zeros_like(p)
        trainer._trust_grad_into(params, term2, 1.0)
        for p, gr in zip(params, ref_grads):
            if gr is not None:
                assert torch.allclose(p.grad, gr, atol=1e-5)
            else:
                assert float(p.grad.abs().sum()) == 0.0

        # 参数空间长度 1e-3 的归一化负梯度步 → 项必须严格减小
        # （固定步长在梯度范数大时会过冲，故按范数归一化）
        gnorm = float(torch.sqrt(torch.stack(
            [p.grad.pow(2).sum() for p in params if p.grad is not None]).sum()))
        step = 1e-3 / max(gnorm, 1e-12)
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.add_(p.grad * -step)
        lp3, _, _, _ = ac.evaluate_actions(obs, actions, orders)
        term3 = trainer._klbc_term(obs, actions, orders, None, lp3)
        assert float(term3.detach()) < float(term2.detach()), "信任域梯度应把策略拉回 BC 盆地"

    def test_update_logs_kl_bc(self):
        cfg = _mini_config()
        cfg.ppo.kl_bc_coef = 0.5
        env = _mini_env(cfg)
        ac = _mini_worker(cfg, env)
        trainer = PPOTrainer(ac, cfg.ppo)
        assert trainer.set_bc_reference(ac.state_dict()) is True
        buffer = RolloutBuffer(
            rollout_steps=cfg.ppo.rollout_steps, obs_dim=env.obs_dim,
            action_dim=3 * cfg.system.K, gamma=0.99, gae_lambda=0.95,
            device="cpu", decoding_order_dim=cfg.system.K)
        _fill_buffer(cfg, env, buffer)
        info = trainer.update(buffer)
        assert info["kl_bc_coef"] == pytest.approx(0.5)
        assert info["kl_bc_loss"] >= 0.0

        # 未注入参考时 coef 报 0（日志不谎报生效）
        cfg2 = _mini_config()
        cfg2.ppo.kl_bc_coef = 0.5
        env2 = _mini_env(cfg2)
        ac2 = _mini_worker(cfg2, env2)
        tr2 = PPOTrainer(ac2, cfg2.ppo)
        buf2 = RolloutBuffer(
            rollout_steps=cfg2.ppo.rollout_steps, obs_dim=env2.obs_dim,
            action_dim=3 * cfg2.system.K, gamma=0.99, gae_lambda=0.95,
            device="cpu", decoding_order_dim=cfg2.system.K)
        _fill_buffer(cfg2, env2, buf2)
        info2 = tr2.update(buf2)
        assert info2["kl_bc_coef"] == 0.0
        assert info2["kl_bc_loss"] == 0.0


class TestGuardV2:
    """M5.2-F5b: 守卫判据升级（A2 盲区：策略爆炸仍全绿）。"""

    @staticmethod
    def _patch(monkeypatch, table):
        import src.training.main as main_mod

        def _fake(env_g, worker_ac, meta_ac, device, config,
                  snr_db, n_eps, seed0):
            return table[float(snr_db)]

        monkeypatch.setattr(main_mod, "_guard_run_point", _fake)

    def _cfg(self, **kw):
        cfg = _mini_config()
        for k, v in kw.items():
            setattr(cfg.ppo, k, v)
        return cfg

    def test_v2_catches_queue_drift_legacy_misses(self, monkeypatch):
        # A2 型盲区：逐点门全过（30dB 队列 36.0 < SDMA×1.1=37.92）、avg_deliv 达标，
        # 但 30dB 队列已超 bc 参考 31.64×1.1=34.8 → v2 必须判负、旧判据漏判
        table = {0.0: (0.882, 2.92), 5.0: (0.989, 0.43), 30.0: (0.871, 36.0)}
        self._patch(monkeypatch, table)

        legacy = _guard_check(None, None, None, None, self._cfg())
        assert legacy["ok"] is True
        assert legacy["points_met"] == 3

        v2 = _guard_check(None, None, None, None,
                          self._cfg(eval_guard_v2=True))
        assert v2["ok"] is False, "v2 必须抓住队列漂离 bc 参考"
        assert v2["points_met"] == 3

    def test_v2_requires_all_points_with_rich_set(self, monkeypatch):
        # 七点集：单点 delivery 失守但均值仍高于门槛 → 旧判据漏判、v2 判负
        table = {
            0.0: (0.99, 2.0), 5.0: (0.99, 0.4), 10.0: (0.99, 0.7),
            15.0: (0.99, 2.0), 20.0: (0.79, 8.0),   # < SDMA 0.795 → 失守
            25.0: (0.99, 18.0), 30.0: (0.99, 30.0),
        }
        self._patch(monkeypatch, table)
        pts = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)

        legacy = _guard_check(None, None, None, None,
                              self._cfg(eval_guard_snr_points=pts))
        assert legacy["n_points"] == 7
        assert legacy["points_met"] == 6
        assert legacy["ok"] is True, "旧判据只看均值，单点失守被平均掉"

        v2 = _guard_check(None, None, None, None,
                          self._cfg(eval_guard_v2=True,
                                    eval_guard_snr_points=pts))
        assert v2["ok"] is False

    def test_v2_passes_when_all_gates_green(self, monkeypatch):
        # 全绿时 v2 与旧判据一致放行（不误伤）
        table = {0.0: (0.90, 2.5), 5.0: (0.99, 0.4), 30.0: (0.88, 29.0)}
        self._patch(monkeypatch, table)
        v2 = _guard_check(None, None, None, None,
                          self._cfg(eval_guard_v2=True))
        assert v2["ok"] is True
        assert v2["points_met"] == v2["n_points"] == 3


class TestHardMiningTargets:
    """M5.3: 硬例挖掘逐点目标覆盖（默认 SDMA 目标对冠军失明）。"""

    CHAMP_PER_POINT = (0.886, 0.989, 0.993, 0.986, 0.964, 0.942, 0.888)

    def test_default_targets_are_sdma(self):
        st = _hm_init_state(1.0)
        assert np.allclose(st["targets"], _HM_TARGETS)
        st2 = _hm_init_state(1.1)
        assert np.allclose(st2["targets"], _HM_TARGETS * 1.1)

    def test_point_targets_override_and_fallback(self):
        override = (0.92, 0.982, 0.96, 0.894, 0.805, 0.73, 0.93)
        st = _hm_init_state(1.0, override)
        assert np.allclose(st["targets"], override)
        assert np.allclose(st["ema"], override), "EMA 初值必须跟随覆盖目标"
        # 长度不符 → 回退默认（防配置笔误静默改口径）
        st_bad = _hm_init_state(1.0, (0.9, 0.9, 0.9))
        assert np.allclose(st_bad["targets"], _HM_TARGETS)
        # 空 → 默认
        assert np.allclose(_hm_init_state(1.0, ())["targets"], _HM_TARGETS)

    def test_sampling_mass_moves_to_failing_points(self):
        override = (0.92, 0.982, 0.96, 0.894, 0.805, 0.73, 0.93)
        st = _hm_init_state(1.0, override)
        st["ema"] = np.array(self.CHAMP_PER_POINT)
        st["achieved"] = st["ema"] >= st["targets"]
        st["streak"] = np.array([3, 0, 0, 0, 0, 0, 3], dtype=np.int32)
        assert st["achieved"].tolist() == [False, True, True, True, True, True, False]

        rng = np.random.default_rng(0)
        edges_lo, edges_hi = -2.5, 32.5
        counts = np.zeros(7, dtype=int)
        n_fail = 0
        ent_seen = set()
        for _ in range(2000):
            snr, b, is_failing, ent = _hm_sample_snr(
                st, w_hard=2.0, floor_frac=0.5, streak_mult=1.5, streak_cap=4.0,
                rng=rng, fail_entropy_scale=1.5, performance_linked=True)
            counts[b] += 1
            n_fail += int(is_failing)
            ent_seen.add(round(float(ent), 3))
            assert edges_lo <= snr <= edges_hi
            assert _hm_bucket(snr) == b, "采样 SNR 必须落在被选桶的边内"

        weak_frac = (counts[0] + counts[6]) / 2000.0
        assert weak_frac > 0.6, f"弱点桶曝光应远高于均匀 2/7，实际 {weak_frac:.3f}"
        assert n_fail > 1000, "未达标点应占采样多数"
        assert max(ent_seen) > 1.5, "performance_linked 应对缺口点放大熵缩放"
        # 已达标桶仍有地板曝光（防完全饿死）
        assert counts[1:6].min() > 0


class TestLRFreezeAndDiagnostics:
    """M5.4: 显式 lr 冻结旋钮 + band 熵/log_std 诊断。"""

    def _trainer(self, freeze_at):
        cfg = _mini_config()
        cfg.ppo.kl_stop_lr_decay = 1.0   # 隔离 KL 衰减，防触底混淆冻结判定
        cfg.ppo.lr_freeze_update = freeze_at
        env = _mini_env(cfg)
        ac = _mini_worker(cfg, env)
        return cfg, env, ac, PPOTrainer(ac, cfg.ppo)

    def _buffer(self, cfg, env):
        buf = RolloutBuffer(
            rollout_steps=cfg.ppo.rollout_steps, obs_dim=env.obs_dim,
            action_dim=3 * cfg.system.K, gamma=0.99, gae_lambda=0.95,
            device="cpu", decoding_order_dim=cfg.system.K)
        _fill_buffer(cfg, env, buf)
        return buf

    def test_freeze_pins_lr_at_floor(self):
        cfg, env, ac, tr = self._trainer(freeze_at=2)
        base0 = float(tr._base_lrs[0])
        tr.set_progress(0.5)
        info1 = tr.update(self._buffer(cfg, env))
        assert info1["lr_frozen"] == 0.0
        # 未冻结：跟随进度退火 (1-0.5)+0.5*0.2 = 0.6
        assert float(tr.optimizer.param_groups[0]["lr"]) == pytest.approx(base0 * 0.6)

        info2 = tr.update(self._buffer(cfg, env))
        assert info2["lr_frozen"] == 1.0
        assert float(tr.optimizer.param_groups[0]["lr"]) == pytest.approx(
            base0 * cfg.ppo.lr_decay_final_ratio)

        # 冻结后进度推进不再改变 lr
        tr.set_progress(0.95)
        assert float(tr.optimizer.param_groups[0]["lr"]) == pytest.approx(
            base0 * cfg.ppo.lr_decay_final_ratio)

    def test_off_by_default(self):
        cfg, env, ac, tr = self._trainer(freeze_at=0)
        base0 = float(tr._base_lrs[0])
        tr.set_progress(0.5)
        for _ in range(3):
            info = tr.update(self._buffer(cfg, env))
        assert info["lr_frozen"] == 0.0
        assert float(tr.optimizer.param_groups[0]["lr"]) == pytest.approx(base0 * 0.6)

    def test_diagnostics_keys_and_band_split(self):
        cfg, env, ac, tr = self._trainer(freeze_at=0)
        info = tr.update(self._buffer(cfg, env))
        for k in ("entropy_lo", "entropy_mid", "entropy_hi", "log_std_mean"):
            assert k in info and np.isfinite(info[k]), k
        # _fill_buffer 全 snr_db=10 ⇒ 只有 mid band 有样本
        assert info["entropy_lo"] == 0.0 and info["entropy_hi"] == 0.0
        assert info["entropy_mid"] == pytest.approx(info["entropy"], abs=1e-6)
        with torch.no_grad():
            expect = float(torch.cat(
                [p.flatten() for p in ac.log_std_params()]).mean())
        assert info["log_std_mean"] == pytest.approx(expect, abs=1e-6)


class TestResumeOptimizerCompatibility:
    """Fix#2: param 组结构变化时 resume 的 optimizer.load_state_dict 安全降级。"""

    def test_same_layout_compatible(self):
        from src.training.main import _optimizer_state_compatible
        model = torch.nn.Linear(3, 2)
        opt1 = torch.optim.Adam(model.parameters(), lr=0.01)
        opt2 = torch.optim.Adam(model.parameters(), lr=0.01)
        assert _optimizer_state_compatible(opt2, opt1.state_dict()) is True

    def test_group_count_mismatch_incompatible(self):
        from src.training.main import _optimizer_state_compatible
        extra = torch.nn.Parameter(torch.zeros(1))
        model = torch.nn.Linear(3, 2)
        opt1 = torch.optim.Adam([
            {"params": model.parameters(), "lr": 1e-3},
            {"params": [extra], "lr": 1e-4},
        ])
        opt2 = torch.optim.Adam(model.parameters(), lr=0.01)  # 1 group vs 2
        assert _optimizer_state_compatible(opt2, opt1.state_dict()) is False

    def test_non_dict_returns_false(self):
        from src.training.main import _optimizer_state_compatible
        model = torch.nn.Linear(3, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        assert _optimizer_state_compatible(opt, None) is False


class TestMWGateTrainable:
    """Fix#3: mw_gate 不得随桶专家共享主干冻结（mode=mw_gated 需学习负载开关）。"""

    def _bucket_worker(self, cfg, env):
        return ActorCriticAttn(
            obs_dim=env.obs_dim,
            action_dim=4 * cfg.system.K,
            K=cfg.system.K,
            d_user=cfg.ppo.worker_d_user,
            d_global=env.obs_dim - cfg.system.K * cfg.ppo.worker_d_user,
            hidden_dim=16, n_heads=2, n_attn_layers=1,
            use_plackett_luce=True,
            use_mean_field_critic=False,
            use_bucket_experts=True,
            backpressure_mode="mw_gated",
        )

    def test_mw_gate_not_in_frozen_shared_trunk(self):
        cfg = _mini_config()
        env = _mini_env(cfg)
        ac = self._bucket_worker(cfg, env)
        shared = ac.shared_trunk_params()
        # 用同一性比较（torch 张量 == 对不同形状会抛 RuntimeError）
        assert all(p is not ac.mw_gate.weight for p in shared)
        assert all(p is not ac.mw_gate.bias for p in shared)

    def test_mw_gate_in_trainable_group(self):
        cfg = _mini_config()
        env = _mini_env(cfg)
        cfg.ppo.use_bucket_experts = True
        ac = self._bucket_worker(cfg, env)
        trainer = PPOTrainer(ac, cfg.ppo)
        trainable = [
            g for g in trainer.optimizer.param_groups
            if any(p is ac.mw_gate.weight for p in g["params"]) and g["lr"] > 0.0
        ]
        assert len(trainable) == 1


class TestGuardBucketAndCliValidation:
    """Fix#1 机制 + Fix#4/#5 CLI 校验助手。"""

    def test_set_worker_bucket_follows_reset_snr(self):
        from src.training.main import _set_worker_bucket
        cfg = _mini_config()
        cfg.ppo.use_bucket_experts = True
        env = _mini_env(cfg)
        ac = ActorCriticAttn(
            obs_dim=env.obs_dim, action_dim=4 * cfg.system.K, K=cfg.system.K,
            d_user=cfg.ppo.worker_d_user,
            d_global=env.obs_dim - cfg.system.K * cfg.ppo.worker_d_user,
            hidden_dim=16, n_heads=2, n_attn_layers=1,
            use_plackett_luce=True, use_mean_field_critic=False,
            use_bucket_experts=True,
        )
        env.reset(seed=0, options={"snr_db": 30.0})
        _set_worker_bucket(ac, env)
        assert ac.active_bucket == ac.set_active_bucket(30.0)
        assert ac.active_bucket != ac.set_active_bucket(0.0) or cfg.system.K <= 4  # 30dB 是最高桶

    def test_hm_targets_must_be_7(self):
        from scripts.train import _validate_hm_point_targets
        with pytest.raises(ValueError):
            _validate_hm_point_targets([0.9, 0.95])
        ok = _validate_hm_point_targets(list(range(0, 35, 5)))
        assert len(ok) == 7

    def test_guard_points_must_be_on_grid(self):
        from scripts.train import _validate_eval_guard_snr_points
        with pytest.raises(ValueError):
            _validate_eval_guard_snr_points([0.0, 3.0, 30.0])
        assert _validate_eval_guard_snr_points([0.0, 5.0, 30.0]) == (0.0, 5.0, 30.0)
