"""Plackett-Luce 采样与 log_prob 测试

覆盖:
1. plackett_luce_sample() 输出是合法排列
2. plackett_luce_log_prob() 返回有限值
3. log_prob 与手动计算一致 (K=3 简单 case)
4. 分数越大的用户越可能排在前面
5. 梯度可以回传
"""
import pytest
import torch
import numpy as np

from src.models.actor_critic import plackett_luce_sample, plackett_luce_log_prob


@pytest.mark.fast
class TestPlackettLuceSample:
    def test_output_is_valid_permutation(self):
        """每行应为 0..K-1 的排列"""
        scores = torch.rand(8, 16) + 0.1  # 正值
        perm = plackett_luce_sample(scores)
        assert perm.shape == (8, 16)
        for i in range(8):
            row = perm[i].numpy()
            assert set(row.tolist()) == set(range(16)), f"Row {i} not a valid permutation"

    def test_output_is_valid_permutation_small(self):
        """K=3 小规模排列"""
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        perm = plackett_luce_sample(scores)
        assert perm.shape == (1, 3)
        assert set(perm[0].numpy().tolist()) == {0, 1, 2}

    def test_batch_independence(self):
        """不同 batch 使用不同噪声，排列不应完全相同"""
        scores = torch.ones(100, 8)  # 等分数
        perm = plackett_luce_sample(scores)
        # 100 个排列不应全部相同
        unique_count = len(set(tuple(perm[i].numpy().tolist()) for i in range(100)))
        assert unique_count > 1

    def test_deterministic_with_extreme_scores(self):
        """极端分数 (一个远大于其他) 应几乎确定性地将其排在首位"""
        scores = torch.tensor([[100.0, 0.01, 0.01, 0.01]])
        n_first = 0
        for _ in range(100):
            perm = plackett_luce_sample(scores)
            if perm[0, 0].item() == 0:
                n_first += 1
        assert n_first >= 95  # 几乎确定排在第一


@pytest.mark.fast
class TestPlackettLuceLogProb:
    def test_finite_values(self):
        """log_prob 应为有限值"""
        scores = torch.rand(8, 16) + 0.1
        perm = plackett_luce_sample(scores)
        log_prob = plackett_luce_log_prob(scores, perm)
        assert log_prob.shape == (8,)
        assert torch.isfinite(log_prob).all()

    def test_negative_values(self):
        """log_prob 应为非正值 (概率 <= 1)"""
        scores = torch.rand(8, 16) + 0.1
        perm = plackett_luce_sample(scores)
        log_prob = plackett_luce_log_prob(scores, perm)
        assert (log_prob <= 0).all()

    def test_manual_computation_k3(self):
        """K=3 手动计算验证

        scores = [3, 2, 1], perm = [0, 1, 2]
        log P = log(3/(3+2+1)) + log(2/(2+1)) + log(1/1)
              = log(3/6) + log(2/3) + log(1)
              = log(0.5) + log(0.667) + 0
        """
        scores = torch.tensor([[3.0, 2.0, 1.0]])
        perm = torch.tensor([[0, 1, 2]])  # 按分数从大到小
        log_prob = plackett_luce_log_prob(scores, perm)

        # 手动计算
        manual = (np.log(3.0 / 6.0) + np.log(2.0 / 3.0) + np.log(1.0 / 1.0))
        assert abs(log_prob.item() - manual) < 1e-5, \
            f"Expected {manual}, got {log_prob.item()}"

    def test_manual_computation_k3_reverse(self):
        """K=3 反序: scores = [3, 2, 1], perm = [2, 1, 0]"""
        scores = torch.tensor([[3.0, 2.0, 1.0]])
        perm = torch.tensor([[2, 1, 0]])  # 反序：先选分数最小的
        log_prob = plackett_luce_log_prob(scores, perm)

        # 手动: log(1/6) + log(2/5) + log(3/3)
        manual = np.log(1.0 / 6.0) + np.log(2.0 / 5.0) + np.log(3.0 / 3.0)
        assert abs(log_prob.item() - manual) < 1e-5

    def test_higher_score_more_probable_first(self):
        """分数越大越可能排在前面：正序 log_prob > 反序 log_prob"""
        scores = torch.tensor([[10.0, 5.0, 1.0]])
        perm_forward = torch.tensor([[0, 1, 2]])  # 高分在前
        perm_reverse = torch.tensor([[2, 1, 0]])  # 低分在前
        lp_forward = plackett_luce_log_prob(scores, perm_forward)
        lp_reverse = plackett_luce_log_prob(scores, perm_reverse)
        assert lp_forward.item() > lp_reverse.item()

    def test_uniform_scores_equal_prob(self):
        """等分数时所有排列概率相等"""
        scores = torch.ones(1, 4) * 5.0
        perm1 = torch.tensor([[0, 1, 2, 3]])
        perm2 = torch.tensor([[3, 2, 1, 0]])
        lp1 = plackett_luce_log_prob(scores, perm1)
        lp2 = plackett_luce_log_prob(scores, perm2)
        assert abs(lp1.item() - lp2.item()) < 1e-5

    def test_consistency_with_sample(self):
        """sample 出的排列的 log_prob 应有限且合理"""
        scores = torch.rand(32, 8) + 0.5
        perm = plackett_luce_sample(scores)
        log_prob = plackett_luce_log_prob(scores, perm)
        assert torch.isfinite(log_prob).all()
        assert (log_prob <= 0).all()


@pytest.mark.fast
class TestPlackettLuceGradient:
    def test_gradient_flows_through_log_prob(self):
        """梯度可以通过 log_prob 回传到 scores"""
        scores_raw = torch.rand(4, 8)
        scores = (scores_raw + 0.1).requires_grad_(True)
        scores.retain_grad()
        perm = plackett_luce_sample(scores.detach())
        log_prob = plackett_luce_log_prob(scores, perm)
        loss = -log_prob.mean()
        loss.backward()
        assert scores.grad is not None
        assert torch.isfinite(scores.grad).all()

    def test_gradient_magnitude_reasonable(self):
        """梯度幅度不应爆炸"""
        scores_raw = torch.rand(8, 16)
        scores = (scores_raw + 0.1).requires_grad_(True)
        scores.retain_grad()
        perm = plackett_luce_sample(scores.detach())
        log_prob = plackett_luce_log_prob(scores, perm)
        loss = -log_prob.mean()
        loss.backward()
        assert scores.grad.abs().max() < 100.0

    def test_higher_score_gradient_direction(self):
        """对于排在前面的位置，梯度应鼓励增大分数"""
        # 排列 [0, 1, 2]，最大化 log_prob 应鼓励 score[0] 增大
        scores = torch.tensor([[2.0, 2.0, 2.0]], requires_grad=True)
        perm = torch.tensor([[0, 1, 2]])
        log_prob = plackett_luce_log_prob(scores, perm)
        log_prob.backward()
        # 对于排在第一位的 score[0], 梯度应为正 (增大它增加概率)
        # 注意：这取决于相对大小，在等分数时梯度为正
        assert scores.grad[0, 0] > 0


# 注意：本类为随机统计性用例（概率断言），不标记 fast，
# 避免假阳性拦截优化循环训练前置门禁
class TestPlackettLuceStatistical:
    def test_sampling_frequency_matches_scores(self):
        """高分数用户更频繁地出现在排列前面"""
        # 用户 0 分数远大于其他
        scores = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
        scores = scores.expand(1000, -1)
        first_position_counts = torch.zeros(4)
        for _ in range(10):  # 多次采样
            perm = plackett_luce_sample(scores)
            for k in range(4):
                first_position_counts[k] += (perm[:, 0] == k).sum().item()

        # 用户 0 应该在第一位出现最频繁
        assert first_position_counts[0] > first_position_counts[1]
        assert first_position_counts[0] > first_position_counts[2]
        assert first_position_counts[0] > first_position_counts[3]
        # 用户 0 应占绝大多数 (理论概率 = 10/13 ≈ 77%)
        total = first_position_counts.sum()
        assert first_position_counts[0] / total > 0.6

    def test_temperature_scaling(self):
        """分数对比度越大，首位排列越确定。

        注：PL 首位概率 P(first=k)=z_k/Σz 对 z 的整体缩放不变（尺度不变性），
        因此直接比较 scores/T 的两种温度无区分度；改为比较
        高对比度分数 vs 均匀分数的首位确定性。
        """
        # 高对比度：首位理论概率 = 5/9 ≈ 0.556
        scores_sharp = torch.tensor([[5.0, 3.0, 1.0]])
        # 均匀分数：首位理论概率 = 1/3 ≈ 0.333
        scores_flat = torch.tensor([[1.0, 1.0, 1.0]])

        n_trials = 1000
        sharp_first_0 = 0
        flat_first_0 = 0
        for _ in range(n_trials):
            perm_sharp = plackett_luce_sample(scores_sharp)
            perm_flat = plackett_luce_sample(scores_flat)
            if perm_sharp[0, 0] == 0:
                sharp_first_0 += 1
            if perm_flat[0, 0] == 0:
                flat_first_0 += 1

        # 高对比度应更确定地将高分放首位（两理论概率差 ~0.22，n=1000 下区分充分）
        assert sharp_first_0 > flat_first_0
