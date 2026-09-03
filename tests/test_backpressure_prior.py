"""Backpressure 先验注入测试（ActorCriticAttn 功率头）

覆盖:
1. 先验计算单元: Q/Q̄ 比值与 β 缩放的手算对拍；β=0 时 shift 精确为 0（官方口径逐 bit 不变）
2. 队列反压单调性: 高队列用户的 deterministic p-logit 随 β 单调升、低队列用户相对受抑；
   alpha/d 块不受注入影响（局部性）
3. act ↔ evaluate_actions log_prob 一致性（PPO 重要性比不变式；含桶专家路径）
4. β 经 state_dict 传输: 构造 β=0 的模型 strict=False 加载 β 检查点 → buffer 生效且行为一致
   （复现 evaluate_snr_sweep.load_ppo_model 的重建流程）；缺 key 的老检查点保持默认 0
5. Config 默认关闭（ppo.backpressure_beta == 0.0）
"""

import pytest
import torch

from src.config import Config
from src.models.actor_critic import ActorCriticAttn

pytestmark = pytest.mark.fast

K = 4
D_USER = 10
OBS_DIM = K * D_USER + 3


def _make_model(beta=0.0, seed=0, use_bucket_experts=False, n_experts=2, mode="queue"):
    torch.manual_seed(seed)
    return ActorCriticAttn(
        obs_dim=OBS_DIM,
        action_dim=4 * K,
        K=K,
        d_user=D_USER,
        d_global=3,
        hidden_dim=32,
        n_heads=4,
        n_attn_layers=1,
        use_bucket_experts=use_bucket_experts,
        n_experts=n_experts,
        backpressure_beta=beta,
        backpressure_mode=mode,
    )


def _make_obs(queues):
    """构造 (1, K*10+3) obs，特征块主序：[f0(K), queue(K), f2..f9(K), global(3)]。"""
    obs = torch.zeros(1, OBS_DIM)
    obs[0, K:2 * K] = torch.as_tensor(queues, dtype=torch.float32)
    obs[0, K * D_USER + 1] = 0.5   # snr_norm
    obs[0, K * D_USER + 2] = 0.5   # eps_norm
    return obs


class TestBackpressureShift:
    def test_ratio_math(self):
        model = _make_model(beta=2.0, seed=0)
        per_user = torch.zeros(1, K, D_USER + 2)  # inject_global 后 d_user+2，queue 仍在索引 1
        per_user[0, :, 1] = torch.tensor([1.0, 1.0, 1.0, 3.0])
        shift = model._backpressure_shift(per_user)
        # Q̄ = 1.5, 中心化比值 = [-1/3, -1/3, -1/3, 1], shift = β*ratio
        expected = torch.tensor([[-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0, 2.0]])
        assert torch.allclose(shift, expected, atol=1e-6)

    def test_beta_zero_is_exact_noop(self):
        model = _make_model(beta=0.0, seed=0)
        assert float(model.backpressure_beta_buf) == 0.0
        per_user = torch.rand(2, K, D_USER + 2)
        shift = model._backpressure_shift(per_user)
        assert torch.equal(shift, torch.zeros_like(shift))

    def test_equal_queues_degenerate_to_zero(self):
        # 队列相等 → 中心化比值全 0 → 偏移归零（退化为原策略；eps 守卫留 ~1e-6 残差）
        model = _make_model(beta=1.5, seed=0)
        per_user = torch.zeros(1, K, D_USER + 2)
        per_user[0, :, 1] = 0.7
        shift = model._backpressure_shift(per_user)
        assert torch.allclose(shift, torch.zeros_like(shift), atol=1e-4)

    def test_all_empty_queues_gives_zero_shift(self):
        model = _make_model(beta=1.5, seed=0)
        per_user = torch.zeros(1, K, D_USER + 2)
        shift = model._backpressure_shift(per_user)
        assert torch.equal(shift, torch.zeros_like(shift))


class TestMwResidualMode:
    def test_base_math(self):
        # Q=[1,1,1,3], K=4: p_mw = K*Q/ΣQ = [2/3,2/3,2/3,2]→clip[0,1]→[2/3,2/3,2/3,1]
        # 再 clamp[0.01,0.95] → [2/3,2/3,2/3,0.95]; shift = atanh(...)
        import math
        model = _make_model(beta=0.0, seed=0, mode="mw_residual")
        per_user = torch.zeros(1, K, D_USER + 2)
        per_user[0, :, 1] = torch.tensor([1.0, 1.0, 1.0, 3.0])
        shift = model._backpressure_shift(per_user)
        expected = torch.tensor([[math.atanh(2.0 / 3.0)] * 3 + [math.atanh(0.95)]])
        assert torch.allclose(shift, expected, atol=1e-6)

    def test_equal_queues_full_power_base(self):
        # 队列相等 → p_mw = K·(1/K) = 1.0（份额语义：每用户满 P_k_max，深过载
        # SDMA 全开行为）→ clamp 上限 0.95
        model = _make_model(beta=0.0, seed=0, mode="mw_residual")
        per_user = torch.zeros(1, K, D_USER + 2)
        per_user[0, :, 1] = 0.7
        shift = model._backpressure_shift(per_user)
        assert torch.allclose(shift, torch.full((1, K), float(torch.atanh(torch.tensor(0.95)))), atol=1e-6)

    def test_all_empty_queues_uniform(self):
        model = _make_model(beta=0.0, seed=0, mode="mw_residual")
        per_user = torch.zeros(1, K, D_USER + 2)
        shift = model._backpressure_shift(per_user)
        assert torch.allclose(shift, torch.full((1, K), float(torch.atanh(torch.tensor(1.0 / K)))), atol=1e-6)

    def test_init_parity_with_sdma_rule(self):
        # SDMA 平价: head 正交初始化 gain≈0 → deterministic p 块 ≈ tanh(atanh(p_mw)) = p_mw
        # （忠实空间口径下 env 把 p 块映射为 (a+1)/2 → 2*p_mw-1 的 tanh 输出对齐 sdma action）
        model = _make_model(beta=0.0, seed=0, mode="mw_residual")
        model.eval()
        obs = _make_obs([1.0, 1.0, 1.0, 3.0])
        with torch.no_grad():
            action, _, _, _, _ = model.act(obs, deterministic=True)
        p_block = action[0, K:2 * K]
        p_mw = torch.tensor([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.95])
        assert torch.allclose(p_block, p_mw, atol=0.05), f"{p_block} vs {p_mw}"

    def test_act_evaluate_consistency_mw_residual(self):
        model = _make_model(beta=0.0, seed=19, mode="mw_residual")
        obs = _make_obs([0.4, 0.1, 0.9, 0.25])
        torch.manual_seed(55)
        action, order, total_lp, _, cont_lp = model.act(obs)
        ev_total_lp, _, _, ev_cont_lp = model.evaluate_actions(obs, action, order)
        assert torch.allclose(cont_lp, ev_cont_lp, atol=1e-4)
        assert torch.allclose(total_lp, ev_total_lp, atol=1e-4)

    def test_mode_rides_checkpoint(self):
        donor = _make_model(beta=0.0, seed=29, mode="mw_residual")
        receiver = _make_model(beta=0.0, seed=99)   # 默认 queue
        receiver.load_state_dict(donor.state_dict(), strict=False)
        assert int(receiver.backpressure_mode_buf) == 2
        obs = _make_obs([0.4, 0.1, 0.9, 0.25])
        donor.eval(); receiver.eval()
        with torch.no_grad():
            a_d, _, _, _, _ = donor.act(obs, deterministic=True)
            a_r, _, _, _, _ = receiver.act(obs, deterministic=True)
        assert torch.equal(a_d, a_r)

    def test_alpha_d_untouched_mw_residual(self):
        # mw_residual 只重写 p 块基座，alpha/d/z 不受影响
        base = _make_model(beta=0.0, seed=3)
        mw = _make_model(beta=0.0, seed=3, mode="mw_residual")
        base.eval(); mw.eval()
        obs = _make_obs([0.4, 0.1, 0.9, 0.25])
        with torch.no_grad():
            a0, _, _, _, _ = base.act(obs, deterministic=True)
            a2, _, _, _, _ = mw.act(obs, deterministic=True)
        assert torch.equal(a0[0, :K], a2[0, :K])
        assert torch.equal(a0[0, 2 * K:3 * K], a2[0, 2 * K:3 * K])
        assert not torch.equal(a0[0, K:2 * K], a2[0, K:2 * K])


class TestMwGatedMode:
    def test_gate_init_scales_base_by_sigmoid4(self):
        # 初始 g = sigmoid(4) ≈ 0.982：shift = g·atanh(p_mw)（≈Run 12 平价起步）
        import math
        model = _make_model(beta=0.0, seed=0, mode="mw_gated")
        per_user = torch.zeros(1, K, D_USER + 2)
        per_user[0, :, 1] = torch.tensor([1.0, 1.0, 1.0, 3.0])
        shift = model._backpressure_shift(per_user)
        g0 = float(torch.sigmoid(torch.tensor(4.0)))
        expected = g0 * torch.tensor([[math.atanh(2.0 / 3.0)] * 3 + [math.atanh(0.95)]])
        assert torch.allclose(shift, expected, atol=1e-6)

    def test_gate_open_recovers_mw_residual(self):
        # bias→+大 → g→1：精确退化为 mw_residual（mode 2）——基座语义不变
        gated = _make_model(beta=0.0, seed=0, mode="mw_gated")
        mw = _make_model(beta=0.0, seed=1, mode="mw_residual")
        with torch.no_grad():
            gated.mw_gate.bias.fill_(20.0)
        per_user = torch.rand(2, K, D_USER + 2)
        assert torch.allclose(gated._backpressure_shift(per_user),
                              mw._backpressure_shift(per_user), atol=1e-4)

    def test_gate_closed_zeroes_base(self):
        # bias→−大 → g→0：基座关闭（欠载省电端点），head 残差完全接管 p_raw
        model = _make_model(beta=0.0, seed=0, mode="mw_gated")
        with torch.no_grad():
            model.mw_gate.bias.fill_(-20.0)
        per_user = torch.rand(2, K, D_USER + 2)
        shift = model._backpressure_shift(per_user)
        assert torch.allclose(shift, torch.zeros_like(shift), atol=1e-4)

    def test_gate_load_sensitivity(self):
        # 门输入含队列负载：mean_q 正权重 → 高负载样本基座缩放更大（深过载保基座）
        model = _make_model(beta=0.0, seed=0, mode="mw_gated")
        with torch.no_grad():
            model.mw_gate.weight.zero_()
            model.mw_gate.weight[0, 0] = 5.0   # mean_q
            model.mw_gate.bias.zero_()
        base_val = float(torch.atanh(torch.tensor(0.95)))  # 队列相等 → 基座均匀
        overload = torch.zeros(1, K, D_USER + 2)
        overload[0, :, 1] = 0.6     # 深过载 queue_norm
        underload = torch.zeros(1, K, D_USER + 2)
        underload[0, :, 1] = 0.03   # 欠载
        g_over = model._backpressure_shift(overload)[0, 0].item() / base_val
        g_under = model._backpressure_shift(underload)[0, 0].item() / base_val
        assert g_over > 0.9 and g_under < 0.6 and g_over > g_under

    def test_gate_gradient_flows(self):
        # 门是可学习参数：shift 反传到达 mw_gate 权重（PPO 经 evaluate_actions 训练的前提）
        model = _make_model(beta=0.0, seed=0, mode="mw_gated")
        with torch.no_grad():
            model.mw_gate.weight.normal_(0.0, 1.0)
        per_user = torch.rand(2, K, D_USER + 2)
        model._backpressure_shift(per_user).sum().backward()
        assert model.mw_gate.weight.grad is not None
        assert model.mw_gate.weight.grad.abs().sum() > 0

    def test_gate_in_actor_params(self):
        # 优化器分组：mw_gate 必须落在 actor 参数组（非桶专家路径，lr=lr_actor）
        model = _make_model(beta=0.0, seed=0, mode="mw_gated")
        ids = {id(p) for p in model.actor_params()}
        assert id(model.mw_gate.weight) in ids
        assert id(model.mw_gate.bias) in ids

    def test_act_evaluate_consistency_mw_gated(self):
        model = _make_model(beta=0.0, seed=21, mode="mw_gated")
        with torch.no_grad():
            model.mw_gate.weight.normal_(0.0, 1.0)   # 让门依赖输入
        obs = _make_obs([0.4, 0.1, 0.9, 0.25])
        torch.manual_seed(55)
        action, order, total_lp, _, cont_lp = model.act(obs)
        ev_total_lp, _, _, ev_cont_lp = model.evaluate_actions(obs, action, order)
        assert torch.allclose(cont_lp, ev_cont_lp, atol=1e-4)
        assert torch.allclose(total_lp, ev_total_lp, atol=1e-4)

    def test_mode_and_gate_ride_checkpoint(self):
        # mode=3 buffer + mw_gate 权重均随 checkpoint 传输：默认 mode 重建（模拟
        # evaluate_snr_sweep.load_ppo_model 不传行为开关）后行为逐 bit 一致
        donor = _make_model(beta=0.0, seed=31, mode="mw_gated")
        with torch.no_grad():
            donor.mw_gate.weight.normal_(0.0, 1.0)
            donor.mw_gate.bias.fill_(1.5)
        receiver = _make_model(beta=0.0, seed=99)   # 默认 queue
        receiver.load_state_dict(donor.state_dict(), strict=False)
        assert int(receiver.backpressure_mode_buf) == 3
        assert torch.allclose(receiver.mw_gate.weight, donor.mw_gate.weight)
        assert torch.allclose(receiver.mw_gate.bias, donor.mw_gate.bias)
        obs = _make_obs([0.4, 0.1, 0.9, 0.25])
        donor.eval(); receiver.eval()
        with torch.no_grad():
            a_d, _, _, _, _ = donor.act(obs, deterministic=True)
            a_r, _, _, _, _ = receiver.act(obs, deterministic=True)
        assert torch.equal(a_d, a_r)

    def test_gate_module_exists_in_default_model(self):
        # 无条件构造：mode≠3 的模型也持有 mw_gate（旧检查点 strict=False 加载 + 评估侧
        # 重建的结构前提）；且默认 mode 下不参与前向 → shift 恒零、无行为变化
        model = _make_model(beta=0.0, seed=0)
        assert "mw_gate.weight" in dict(model.named_parameters())
        per_user = torch.rand(1, K, D_USER + 2)
        shift = model._backpressure_shift(per_user)
        assert torch.equal(shift, torch.zeros_like(shift))


class TestQueueChannelMode:
    def test_product_math(self):
        # w = Q·ĥ = [1,2,1,3], mean=1.75, 中心化比值 = [-3/7, 1/7, -3/7, 5/7], β=2
        model = _make_model(beta=2.0, seed=0, mode="queue_channel")
        per_user = torch.zeros(1, K, D_USER + 2)
        per_user[0, :, 0] = torch.tensor([1.0, 2.0, 1.0, 1.0])   # ĥ
        per_user[0, :, 1] = torch.tensor([1.0, 1.0, 1.0, 3.0])   # Q
        shift = model._backpressure_shift(per_user)
        expected = torch.tensor([[-6.0 / 7.0, 2.0 / 7.0, -6.0 / 7.0, 10.0 / 7.0]])
        assert torch.allclose(shift, expected, atol=1e-6)

    def test_equal_queues_channel_decides(self):
        # Q 相等时乘积形式退化为信道加权（与 Run 10 纯 Q 形式的关键差异）
        model = _make_model(beta=1.0, seed=0, mode="queue_channel")
        per_user = torch.zeros(1, K, D_USER + 2)
        per_user[0, :, 0] = torch.tensor([1.0, 1.0, 1.0, 3.0])
        per_user[0, :, 1] = 0.5
        shift = model._backpressure_shift(per_user)
        # w = [0.5,0.5,0.5,1.5], mean=0.75, 比值 = [-1/3,-1/3,-1/3,1]
        assert float(shift[0, 3]) > 0.0
        assert float(shift[0, 0]) < 0.0

    def test_all_equal_product_zero_shift(self):
        model = _make_model(beta=1.5, seed=0, mode="queue_channel")
        per_user = torch.zeros(1, K, D_USER + 2)
        per_user[0, :, 0] = 0.4
        per_user[0, :, 1] = 0.6
        shift = model._backpressure_shift(per_user)
        assert torch.allclose(shift, torch.zeros_like(shift), atol=1e-4)

    def test_higher_channel_higher_p_logit_same_queue(self):
        # 队列相同、信道不同：queue_channel 下强信道用户 p-logit 更高（恢复转换效率）
        obs = torch.zeros(1, OBS_DIM)
        obs[0, 0:K] = torch.tensor([0.2, 0.2, 0.2, 1.0])   # ĥ
        obs[0, K:2 * K] = 0.5                              # Q 相等
        obs[0, K * D_USER + 1] = 0.5
        obs[0, K * D_USER + 2] = 0.5
        m_queue = _make_model(beta=1.0, seed=5)
        m_qh = _make_model(beta=1.0, seed=5, mode="queue_channel")
        m_queue.eval(); m_qh.eval()
        with torch.no_grad():
            a_q, _, _, _, _ = m_queue.act(obs, deterministic=True)
            a_qh, _, _, _, _ = m_qh.act(obs, deterministic=True)
        # 纯 Q 形式：队列相等 → 与 β=0 相同；乘积形式：强信道用户被抬升
        assert float(a_qh[0, K + 3]) > float(a_q[0, K + 3])
        assert float(a_qh[0, K + 0]) < float(a_q[0, K + 0])

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="backpressure_mode"):
            _make_model(beta=1.0, mode="bogus")

    def test_act_evaluate_consistency_queue_channel(self):
        model = _make_model(beta=1.5, seed=17, mode="queue_channel")
        obs = torch.zeros(1, OBS_DIM)
        obs[0, 0:K] = torch.tensor([0.3, 1.0, 0.1, 0.7])
        obs[0, K:2 * K] = torch.tensor([0.4, 0.1, 0.9, 0.25])
        torch.manual_seed(77)
        action, order, total_lp, _, cont_lp = model.act(obs)
        ev_total_lp, _, _, ev_cont_lp = model.evaluate_actions(obs, action, order)
        assert torch.allclose(cont_lp, ev_cont_lp, atol=1e-4)
        assert torch.allclose(total_lp, ev_total_lp, atol=1e-4)


class TestMonotonicity:
    def test_higher_queue_higher_p_logit(self):
        # 用户 3 队列 1.0，其余 0.2；同一权重（同 seed）下 β 单调抬升高队列用户 p-logit
        obs = _make_obs([0.2, 0.2, 0.2, 1.0])
        p_user3, p_user0 = {}, {}
        for beta in (0.0, 1.0, 2.0):
            model = _make_model(beta=beta, seed=7)
            model.eval()
            with torch.no_grad():
                action, _, _, _, _ = model.act(obs, deterministic=True)
            p_block = action[0, K:2 * K]
            p_user3[beta] = float(p_block[3])
            p_user0[beta] = float(p_block[0])
        assert p_user3[2.0] > p_user3[1.0] > p_user3[0.0]
        assert p_user0[2.0] < p_user0[0.0]

    def test_alpha_and_d_blocks_untouched(self):
        obs = _make_obs([0.2, 0.2, 0.2, 1.0])
        base, biased = _make_model(beta=0.0, seed=3), _make_model(beta=2.0, seed=3)
        base.eval(); biased.eval()
        with torch.no_grad():
            a0, _, _, _, _ = base.act(obs, deterministic=True)
            a2, _, _, _, _ = biased.act(obs, deterministic=True)
        assert torch.equal(a0[0, :K], a2[0, :K])      # alpha 块
        assert torch.equal(a0[0, 2 * K:3 * K], a2[0, 2 * K:3 * K])  # d 块
        assert not torch.equal(a0[0, K:2 * K], a2[0, K:2 * K])      # p 块被注入


class TestLogProbConsistency:
    def test_act_matches_evaluate_actions(self):
        model = _make_model(beta=1.5, seed=11)
        obs = _make_obs([0.4, 0.1, 0.9, 0.25])
        torch.manual_seed(123)
        action, order, total_lp, _, cont_lp = model.act(obs)
        ev_total_lp, _, _, ev_cont_lp = model.evaluate_actions(obs, action, order)
        # tanh/atanh + clamp 往返误差容差
        assert torch.allclose(cont_lp, ev_cont_lp, atol=1e-4)
        assert torch.allclose(total_lp, ev_total_lp, atol=1e-4)

    def test_consistency_with_bucket_experts(self):
        model = _make_model(beta=1.0, seed=13, use_bucket_experts=True, n_experts=2)
        obs = _make_obs([0.4, 0.1, 0.9, 0.25])
        torch.manual_seed(321)
        action, order, total_lp, _, cont_lp = model.act(obs, bucket_indices=torch.tensor([1]))
        ev_total_lp, _, _, ev_cont_lp = model.evaluate_actions(
            obs, action, order, bucket_indices=torch.tensor([1])
        )
        assert torch.allclose(cont_lp, ev_cont_lp, atol=1e-4)
        assert torch.allclose(total_lp, ev_total_lp, atol=1e-4)


class TestStateDictTransport:
    def test_beta_rides_checkpoint(self):
        # 复现 load_ppo_model 流程：构造时不传 β（默认 0），strict=False 加载后 β 生效
        donor = _make_model(beta=1.5, seed=21)
        state = donor.state_dict()
        assert "backpressure_beta_buf" in state
        receiver = _make_model(beta=0.0, seed=99)
        receiver.load_state_dict(state, strict=False)
        assert float(receiver.backpressure_beta_buf) == pytest.approx(1.5)
        obs = _make_obs([0.2, 0.2, 0.2, 1.0])
        donor.eval(); receiver.eval()
        with torch.no_grad():
            a_d, _, _, _, _ = donor.act(obs, deterministic=True)
            a_r, _, _, _, _ = receiver.act(obs, deterministic=True)
        assert torch.equal(a_d, a_r)

    def test_mode_rides_checkpoint(self):
        # mode buffer 同样随 checkpoint 传输：评估侧构造默认 queue，加载后变 queue_channel
        donor = _make_model(beta=1.2, seed=23, mode="queue_channel")
        receiver = _make_model(beta=0.0, seed=99)   # 默认 queue / β=0
        receiver.load_state_dict(donor.state_dict(), strict=False)
        assert int(receiver.backpressure_mode_buf) == 1
        assert float(receiver.backpressure_beta_buf) == pytest.approx(1.2)
        obs = _make_obs([0.4, 0.1, 0.9, 0.25])
        donor.eval(); receiver.eval()
        with torch.no_grad():
            a_d, _, _, _, _ = donor.act(obs, deterministic=True)
            a_r, _, _, _, _ = receiver.act(obs, deterministic=True)
        assert torch.equal(a_d, a_r)

    def test_legacy_checkpoint_without_key_keeps_default(self):
        donor = _make_model(beta=1.5, seed=21)
        state = donor.state_dict()
        state = {k: v for k, v in state.items() if k not in
                 ("backpressure_beta_buf", "backpressure_mode_buf")}
        receiver = _make_model(beta=0.0, seed=99)
        receiver.load_state_dict(state, strict=False)
        assert float(receiver.backpressure_beta_buf) == 0.0
        assert int(receiver.backpressure_mode_buf) == 0


class TestConfigDefault:
    def test_default_off(self):
        assert Config().ppo.backpressure_beta == 0.0
