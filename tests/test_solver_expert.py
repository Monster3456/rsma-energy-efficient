"""求解器专家单元测试：surrogate 镜像精确性 + 动作语义一致性 + 只读性。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.env.rsma_env import RSMAEnergyEnv  # noqa: E402
from src.optimization.solver_expert import (  # noqa: E402
    ACT_BOUND,
    D_FRAC,
    SolverExpert,
)

pytestmark = pytest.mark.fast


def _make_env(snr_db=0.0, seed=0):
    cfg = Config()
    cfg.system.csi_mode = "imperfect"
    cfg.system.sigma2 = 10 ** (-snr_db / 10.0)
    cfg.system.snr_domain_randomization = False
    cfg.system.adaptive_lambda = False
    env = RSMAEnergyEnv(cfg)
    env.reset(seed=seed)
    return env


def _warmup(env, n_steps=25):
    """以均匀满功率动作填充队列，制造非平凡状态（0dB 下 R << λ）。"""
    K = env.K
    action = np.concatenate([np.zeros(3 * K), np.arange(K, dtype=np.float64)])
    for _ in range(n_steps):
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break


class TestSurrogateMirror:
    def test_reward_mirror_matches_compute_reward(self):
        """专家 reward_pred 与 lyapunov.compute_reward 在相同 (D, P, OV_exp) 下逐位一致。"""
        env = _make_env(snr_db=0.0)
        _warmup(env)
        expert = SolverExpert(env, n_steps=3)
        result = expert.solve(env)

        Q = np.asarray(env.traffic.queue_lengths, dtype=np.float64)
        lam = np.asarray(env.traffic.lambda_per_user, dtype=np.float64) \
            * float(env.traffic.packet_size)
        D = result["D"]
        p = result["p"]

        # 期望到达的溢出代理
        OV_exp = np.maximum(Q - D + lam - env.Q_max, 0.0)
        rew_info = env.lyapunov.compute_reward(
            queue_lengths=Q,
            served_bits=D,
            power_consumed=p,
            overflow=OV_exp,
            prev_queue=env._prev_queue,
            sigma2=env.sigma2,
            P_target=None,
            user_drift_weights=env.qos_user_drift_weights if env.enable_hetero_qos else None,
            user_overflow_weights=env.qos_user_overflow_weights if env.enable_hetero_qos else None,
            delivery_running=env.traffic.get_delivery_ratio(),
            arrivals=lam,
        )
        assert rew_info["reward"] == pytest.approx(result["J"], abs=1e-9)

    def test_predicted_physics_matches_env_chain(self):
        """预测的 p/R/D 与 env numpy 物理链（parse→precoding→outage→deliver）一致。"""
        env = _make_env(snr_db=5.0)
        _warmup(env)
        expert = SolverExpert(env, n_steps=3)
        result = expert.solve(env)

        alpha, p_total, d_ratio = env._parse_action(result["cont_action"])
        assert alpha == pytest.approx(np.clip(env.split_mask, 0, 1), abs=1e-12)
        assert d_ratio == pytest.approx(np.full(env.K, D_FRAC), abs=1e-12)
        p_np = p_total * env.P_k_max
        assert p_np == pytest.approx(result["p"], abs=1e-9)

        W_c, W_p = env.beamformer.compute_precoding_v2(
            env._channel_obs["H_hat"], alpha, p_np, env.split_mask
        )
        rr = env.rate_calc.compute_outage_rates_mmse_sic(
            env._channel_obs["H_hat"], env._channel_obs["H_true"], W_c, W_p,
            error_bound=float(env._channel_obs["sigma_est"]),
            sic_imperfection=float(env.config.system.sic_imperfection),
            decoding_order=result["order"],
            group_ids=env.group_ids,
        )
        Q = np.asarray(env.traffic.queue_lengths, dtype=np.float64)
        R_np = rr["R_total"]
        D_np = d_ratio * np.maximum(np.minimum(R_np * env.tau, Q), 0.0)
        assert R_np == pytest.approx(result["R"], abs=1e-9)
        assert D_np == pytest.approx(result["D"], abs=1e-9)


class TestActionSemantics:
    def test_env_step_reproduces_prediction(self):
        """env.step(专家动作) 的 info 与预测逐位一致（功率/速率/服务量确定性）。"""
        env = _make_env(snr_db=0.0)
        _warmup(env)
        expert = SolverExpert(env, n_steps=5)
        result = expert.solve(env)

        action = result["action"]
        assert np.all(np.isfinite(action))
        assert action.shape == (4 * env.K,)
        _, _, _, _, info = env.step(action)
        assert info["total_power"] == pytest.approx(result["total_power"], abs=1e-8)
        assert info["R_sum"] == pytest.approx(float(np.sum(result["R"])), abs=1e-8)
        assert np.allclose(info["served"], result["D"], atol=1e-8)

    def test_solve_is_readonly(self):
        """solve 不触碰 env / lyapunov / traffic 状态机（专家只读）。"""
        env = _make_env(snr_db=5.0)
        _warmup(env)
        lya = env.lyapunov
        snap = {
            "Q": env.traffic.queue_lengths.copy(),
            "Y": lya._Y,
            "X": lya._X,
            "ema_s": lya._dkl_served_ema,
            "ema_p": lya._dkl_power_ema,
            "q": lya.dinkelbach_q,
            "avg_served": lya._avg_served.copy(),
            "t": env.t,
            "H": env._channel_obs["H_hat"].copy(),
        }
        expert = SolverExpert(env, n_steps=3)
        expert.solve(env)
        expert.solve(env, warm_theta=np.full(env.K, 0.3))

        assert np.array_equal(env.traffic.queue_lengths, snap["Q"])
        assert lya._Y == snap["Y"]
        assert lya._X == snap["X"]
        assert lya._dkl_served_ema == snap["ema_s"]
        assert lya._dkl_power_ema == snap["ema_p"]
        assert lya.dinkelbach_q == snap["q"]
        assert np.array_equal(lya._avg_served, snap["avg_served"])
        assert env.t == snap["t"]
        assert np.array_equal(env._channel_obs["H_hat"], snap["H"])


class TestOptimization:
    def test_solve_beats_uniform_logits(self):
        """Adam 精修后 J 不低于均匀 logits 起点（优化有效性）。"""
        env = _make_env(snr_db=0.0)
        _warmup(env)
        expert = SolverExpert(env, n_steps=25)
        result = expert.solve(env)
        base = expert.evaluate(env, np.zeros(env.K), result["order"])
        assert result["J"] >= base["reward"] - 1e-9

    @pytest.mark.parametrize("d_frac", [0.5, 0.8, 0.95])
    def test_d1_is_optimal(self, d_frac):
        """d_ratio=1 单调性：J(d=1) >= J(d<1)（队列非空状态）。"""
        env = _make_env(snr_db=0.0)
        _warmup(env)
        assert np.sum(env.traffic.queue_lengths) > 0.0  # 前置：非平凡队列
        expert = SolverExpert(env, n_steps=1)
        order = np.arange(env.K, dtype=np.int64)
        j_full = expert.evaluate(env, np.zeros(env.K), order, d_frac=1.0)["reward"]
        j_less = expert.evaluate(env, np.zeros(env.K), order, d_frac=d_frac)["reward"]
        assert j_full >= j_less - 1e-9

    def test_action_bounds(self):
        env = _make_env(snr_db=15.0)
        expert = SolverExpert(env, n_steps=3)
        result = expert.solve(env)
        cont = result["cont_action"]
        assert np.all(cont <= ACT_BOUND + 1e-12)
        assert np.all(cont >= -ACT_BOUND - 1e-12)
        assert result["order"].shape == (env.K,)
