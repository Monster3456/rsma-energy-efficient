"""TrafficManager 与 LyapunovController 单元测试。

覆盖：
1. 泊松到达统计正确（均值≈lambda_rate）
2. 队列更新逻辑正确（服务后减少，到达后增加）
3. 溢出正确触发（Q > Q_max）
4. 服务量不超过队列长度
5. Lyapunov reward 手算验证
6. V 越大，功率惩罚权重越大
"""
import numpy as np
import pytest

from src.env.traffic import TrafficManager
from src.optimization.lyapunov import LyapunovController

# fast 子集：流量/队列/Lyapunov 逻辑校验，轻量且确定性
pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------- #
# TrafficManager
# ---------------------------------------------------------------------- #
class TestTrafficManager:
    def test_reset_state(self):
        tm = TrafficManager(K=4, lambda_rate=4.0, Q_max=100.0, packet_size=1.0, seed=0)
        out = tm.reset()
        assert out["queue_lengths"].shape == (4,)
        assert np.allclose(out["queue_lengths"], 0.0)
        assert np.allclose(out["arrivals"], 0.0)
        assert tm.mean_queue_length == 0.0

    def test_poisson_arrivals_mean(self):
        """泊松到达的样本均值应接近 lambda_rate * packet_size。"""
        K = 8
        lam = 4.0
        pkt = 1.0
        tm = TrafficManager(K=K, lambda_rate=lam, Q_max=1e9, packet_size=pkt, seed=123)
        tm.reset()

        T = 5000
        arrivals_sum = np.zeros(K)
        for _ in range(T):
            res = tm.step(np.zeros(K))  # 不服务，仅积累到达统计
            arrivals_sum += res["arrivals"]

        mean_per_slot = arrivals_sum / T
        # 大数定律：均值应接近 lambda_rate*pkt（容差 5%）
        assert np.allclose(mean_per_slot, lam * pkt, rtol=0.05, atol=0.1)

    def test_queue_grows_without_service(self):
        """无服务时队列单调不减（除非到达为 0）。"""
        tm = TrafficManager(K=2, lambda_rate=2.0, Q_max=1e9, packet_size=1.0, seed=7)
        tm.reset()
        prev = np.zeros(2)
        for _ in range(50):
            res = tm.step(np.zeros(2))
            assert np.all(res["queue_lengths"] >= prev - 1e-9)
            prev = res["queue_lengths"]

    def test_service_reduces_queue(self):
        """服务后队列减少 (Q_after = Q_before - served + arrival)。"""
        tm = TrafficManager(K=2, lambda_rate=0.0, Q_max=1e9, packet_size=1.0, seed=1)
        tm.reset()
        # lambda=0 -> 无到达，便于验证服务逻辑
        # 先注入一些初始队列：手动设置
        tm.queue_lengths = np.array([10.0, 5.0])
        res = tm.step(np.array([3.0, 2.0]))
        assert np.allclose(res["served"], [3.0, 2.0])
        assert np.allclose(res["queue_before"], [10.0, 5.0])
        assert np.allclose(res["arrivals"], 0.0)
        assert np.allclose(res["queue_lengths"], [7.0, 3.0])

    def test_service_capped_by_queue(self):
        """服务量 D_k 不能超过 Q_k[t-1]。"""
        tm = TrafficManager(K=2, lambda_rate=0.0, Q_max=1e9, packet_size=1.0, seed=2)
        tm.reset()
        tm.queue_lengths = np.array([2.0, 0.0])
        res = tm.step(np.array([100.0, 50.0]))  # 远超存量
        assert np.allclose(res["served"], [2.0, 0.0])
        assert np.all(res["queue_lengths"] <= res["queue_before"] + 1e-9)
        assert np.all(res["served"] <= res["queue_before"] + 1e-9)

    def test_overflow_triggered(self):
        """到达使 Q > Q_max 时应记录 overflow 并将队列截断为 Q_max。"""
        tm = TrafficManager(K=1, lambda_rate=20.0, Q_max=5.0, packet_size=1.0, seed=42)
        tm.reset()
        any_overflow = False
        for _ in range(100):
            res = tm.step(np.zeros(1))
            assert res["queue_lengths"][0] <= 5.0 + 1e-9
            if res["overflow"][0] > 0:
                any_overflow = True
        assert any_overflow, "Q_max=5 with lambda=20 should overflow within 100 slots"

    def test_negative_service_clipped(self):
        tm = TrafficManager(K=1, lambda_rate=0.0, Q_max=10.0, packet_size=1.0, seed=0)
        tm.reset()
        tm.queue_lengths = np.array([5.0])
        res = tm.step(np.array([-3.0]))
        assert res["served"][0] == 0.0
        assert res["queue_lengths"][0] == 5.0

    def test_delivery_ratio(self):
        tm = TrafficManager(K=1, lambda_rate=0.0, Q_max=1e9, packet_size=1.0, seed=0)
        tm.reset()
        tm.queue_lengths = np.array([10.0])
        tm._total_arrived[:] = 10.0  # 模拟之前到达
        tm.step(np.array([4.0]))
        # 服务 4，累计到达 10 → ratio = 4/10
        assert tm.get_delivery_ratio() == pytest.approx(0.4)

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            TrafficManager(K=0, lambda_rate=1.0, Q_max=10.0, packet_size=1.0)
        with pytest.raises(ValueError):
            TrafficManager(K=1, lambda_rate=-1.0, Q_max=10.0, packet_size=1.0)
        with pytest.raises(ValueError):
            TrafficManager(K=1, lambda_rate=1.0, Q_max=0.0, packet_size=1.0)


# ---------------------------------------------------------------------- #
# LyapunovController
# ---------------------------------------------------------------------- #
class TestLyapunovController:
    def test_manual_reward_calc(self):
        """验证精简3项奖励: ee_bonus, drift_signal, overflow_penalty"""
        K = 3
        V = 10.0
        ctrl = LyapunovController(
            K=K, V=V, V_min=1.0, V_max=100.0,
            reward_scale=1.0, w_overflow=5.0,
            ee_weight=8.0, ee_power_floor=0.5,
            blend_alpha=2.0,
        )
        Q = np.array([2.0, 4.0, 6.0])
        D = np.array([1.0, 2.0, 1.0])
        P = np.array([0.1, 0.2, 0.3])    # total_power = 0.6
        OV = np.array([0.0, 1.0, 0.0])   # w_overflow*sum(OV) = 5*1 = 5
        out = ctrl.compute_reward(Q, D, P, OV, outage_flags=None, R_sum=4.0)

        # 验证输出键
        assert "reward" in out
        assert "ee_bonus" in out
        assert "drift_signal" in out
        assert "overflow_penalty" in out
        assert "queue_weight" in out
        assert "avg_queue_ratio" in out
        assert "total_power" in out
        # overflow_penalty = 5.0 * 1.0 = 5.0
        assert out["overflow_penalty"] == pytest.approx(5.0)
        assert out["total_power"] == pytest.approx(0.6)
        # reward 应该是有限值
        assert np.isfinite(out["reward"])

    def test_no_outage_flags_means_no_crash(self):
        ctrl = LyapunovController(K=2, V=1.0, reward_scale=1.0)
        out = ctrl.compute_reward(
            queue_lengths=np.array([1.0, 1.0]),
            served_bits=np.array([1.0, 1.0]),
            power_consumed=np.array([0.0, 0.0]),
            overflow=np.array([0.0, 0.0]),
            outage_flags=None,
        )
        # overflow=0 -> overflow_penalty = 0
        assert out["overflow_penalty"] == 0.0
        assert np.isfinite(out["reward"])

    def test_larger_overflow_gives_lower_reward(self):
        """溢出越大，奖励越低。"""
        Q = np.array([5.0, 5.0])
        D = np.array([1.0, 1.0])
        P = np.array([0.5, 0.5])

        ctrl = LyapunovController(K=2, V=5.0, V_min=1.0, V_max=100.0,
                                  reward_scale=1.0, w_overflow=5.0)
        out_no_ov = ctrl.compute_reward(Q, D, P, np.zeros(2))
        out_with_ov = ctrl.compute_reward(Q, D, P, np.array([1.0, 1.0]))
        assert out_with_ov["reward"] < out_no_ov["reward"]

    def test_reward_scale_applied(self):
        """reward_scale 正确缩放最终奖励"""
        ctrl_1 = LyapunovController(K=2, V=1.0, reward_scale=1.0,
                                    ee_weight=0.0, blend_alpha=2.0)
        ctrl_01 = LyapunovController(K=2, V=1.0, reward_scale=0.1,
                                     ee_weight=0.0, blend_alpha=2.0)
        Q = np.array([5.0, 5.0])
        D = np.array([1.0, 1.0])
        P = np.array([0.5, 0.5])
        OV = np.zeros(2)

        r1 = ctrl_1.compute_reward(Q, D, P, OV)["reward"]
        r01 = ctrl_01.compute_reward(Q, D, P, OV)["reward"]
        # reward_scale=0.1 的奖励应约为 reward_scale=1.0 的 1/10
        assert r01 == pytest.approx(r1 * 0.1, rel=1e-6)

    def test_virtual_queue_update(self):
        ctrl = LyapunovController(K=2, V=1.0, use_virtual_queue=True,
                                  vq_step_size=0.5)
        ctrl.reset()
        assert np.allclose(ctrl.virtual_queue, 0.0)

        # 违反量 (positive) -> 虚拟队列累积
        ctrl.update_virtual_queue(np.zeros(2), np.array([2.0, -1.0]))
        # Z = max(0 + 0.5*2, 0) = 1.0, max(0 + 0.5*(-1), 0) = 0
        assert np.allclose(ctrl.virtual_queue, [1.0, 0.0])

        # 第二步：再加一次正违反 / 一次负反馈
        ctrl.update_virtual_queue(np.zeros(2), np.array([-3.0, 4.0]))
        # Z1: max(1 + 0.5*(-3), 0) = max(-0.5,0) = 0
        # Z2: max(0 + 0.5*4, 0) = 2.0
        assert np.allclose(ctrl.virtual_queue, [0.0, 2.0])

    def test_virtual_queue_disabled(self):
        ctrl = LyapunovController(K=2, V=1.0, use_virtual_queue=False)
        ctrl.reset()
        ctrl.update_virtual_queue(np.zeros(2), np.array([5.0, 5.0]))
        assert np.allclose(ctrl.virtual_queue, 0.0)

    def test_reset_clears_virtual_queue(self):
        ctrl = LyapunovController(K=2, V=1.0, vq_step_size=1.0)
        ctrl.update_virtual_queue(np.zeros(2), np.array([3.0, 3.0]))
        assert np.any(ctrl.virtual_queue > 0)
        ctrl.reset()
        assert np.allclose(ctrl.virtual_queue, 0.0)

    def test_v_tradeoff_info(self):
        ctrl = LyapunovController(K=2, V=10.0, V_min=1.0, V_max=100.0)
        info = ctrl.get_v_tradeoff_info()
        assert info["V"] == pytest.approx(10.0, rel=1e-2)
        assert "blend_alpha" in info
        assert "ee_weight" in info
        assert "V_min" in info
        assert "V_max" in info

    def test_v_clipped_to_range(self):
        # V=1000, V_min=1.0, V_max=100.0 -> clip(1000, 1.0, 101.0) = 101.0
        ctrl = LyapunovController(K=1, V=1000.0, V_min=1.0, V_max=100.0)
        assert ctrl.V == pytest.approx(101.0)
        ctrl2 = LyapunovController(K=1, V=0.01, V_min=1.0, V_max=100.0)
        assert ctrl2.V == pytest.approx(1.0)

    def test_shape_validation(self):
        ctrl = LyapunovController(K=3, V=1.0)
        with pytest.raises(ValueError):
            ctrl.compute_reward(
                queue_lengths=np.zeros(2),
                served_bits=np.zeros(3),
                power_consumed=np.zeros(3),
                overflow=np.zeros(3),
            )

    # ---------------- reward_v2: 逐用户加权 + q* 冻结 ---------------- #
    @staticmethod
    def _v2_ctrl(**overrides):
        base = dict(
            K=4, V=1.0, reward_scale=1.0, ee_weight=8.0,
            use_dinkelbach=True, dinkelbach_outer_update=True,
            dinkelbach_q_init=0.5,
        )
        base.update(overrides)
        return LyapunovController(**base)

    def test_reward_v2_default_off_matches_baseline(self):
        """默认参数关闭时 reward 与 v2 机制引入前逐位一致。"""
        Q = np.array([10.0, 60.0, 10.0, 60.0])
        D = np.array([0.5, 1.0, 0.5, 1.0])
        P = np.array([1.0, 1.0, 1.0, 1.0])
        OV = np.zeros(4)
        ctrl = self._v2_ctrl()
        out = ctrl.compute_reward(Q, D, P, OV, R_sum=3.0)
        # dkl = served_total - q_init*P_eff，q 未被改动
        served_total, P_eff = 3.0, max(4.0 / 0.35 + 5.0, 3.0)
        assert out["ee_bonus"] == pytest.approx(8.0 * (served_total - 0.5 * P_eff))
        assert ctrl.dinkelbach_q == pytest.approx(0.5)  # 外层模式不内部更新

    def test_reward_v2_per_user_weight_shifts_served(self):
        """积压用户 served 计 2×、饥饿用户计 0.5×（clip 生效）。"""
        Q = np.array([1.0, 1.0, 1.0, 99.0])   # Q̄ = 25.5
        D = np.array([1.0, 1.0, 1.0, 1.0])
        P = np.array([1.0, 1.0, 1.0, 1.0])
        OV = np.zeros(4)
        ctrl = self._v2_ctrl(reward_v2_per_user_weight=True, reward_v2_fixed_q=1.9)
        out = ctrl.compute_reward(Q, D, P, OV, R_sum=4.0)
        # w = clip(Q/Q̄, 0.5, 2): [0.5, 0.5, 0.5, 2] -> weighted served = 3.5
        P_eff = max(4.0 / 0.35 + 5.0, 3.0)
        assert out["ee_bonus"] == pytest.approx(8.0 * (3.5 - 1.9 * P_eff))

    def test_reward_v2_fixed_q_freezes_and_overrides_external_writes(self):
        """冻结 q* 后外部写入（rsma_env eta 赋值路径）在下一步被重申覆盖。"""
        ctrl = self._v2_ctrl(reward_v2_fixed_q=1.9)
        args = (np.array([1.0] * 4), np.array([1.0] * 4),
                np.array([1.0] * 4), np.zeros(4))
        ctrl.compute_reward(*args, R_sum=4.0)
        assert ctrl.dinkelbach_q == pytest.approx(1.9)
        ctrl.dinkelbach_q = 0.01  # 模拟外层 eta 覆写
        ctrl.compute_reward(*args, R_sum=4.0)
        assert ctrl.dinkelbach_q == pytest.approx(1.9)  # 冻结重申生效

    def test_reward_v2_equal_queues_degenerate_to_total(self):
        """队列全等时 w≡1，加权 served 精确等于 served_total。"""
        Q = np.array([30.0] * 4)
        D = np.array([0.3, 0.6, 0.9, 1.2])
        P = np.array([1.0] * 4)
        ctrl = self._v2_ctrl(reward_v2_per_user_weight=True, reward_v2_fixed_q=1.9)
        out_w = ctrl.compute_reward(Q, D, P, np.zeros(4), R_sum=3.0)
        ctrl2 = self._v2_ctrl(reward_v2_fixed_q=1.9)
        out_b = ctrl2.compute_reward(Q, D, P, np.zeros(4), R_sum=3.0)
        assert out_w["ee_bonus"] == pytest.approx(out_b["ee_bonus"])
