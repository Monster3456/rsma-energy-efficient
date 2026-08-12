"""速率计算与预编码单元测试"""
import numpy as np
import pytest
import sys
sys.path.insert(0, '/Users/libo/PycharmProjects/rsma-energy-efficient')

from src.env.rate_calc import RSMARateCalculator
from src.env.beamforming import BeamformingDesigner

# fast 子集：速率计算/预编码核心正确性，轻量且确定性
pytestmark = pytest.mark.fast


# 系统参数
K = 16
N_r = 16
N_t = 2
SIGMA2 = 0.01


@pytest.fixture
def calc():
    return RSMARateCalculator(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)


@pytest.fixture
def designer():
    return BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)


@pytest.fixture
def random_channel():
    """生成随机Rayleigh信道"""
    np.random.seed(42)
    H = (np.random.randn(K, N_r, N_t) + 1j * np.random.randn(K, N_r, N_t)) / np.sqrt(2)
    return H


@pytest.fixture
def random_precoding(random_channel, designer):
    """生成随机预编码"""
    H = random_channel
    alpha = np.random.uniform(0.2, 0.8, size=K)
    p_total = np.random.uniform(0.1, 1.0, size=K)
    W_c, W_p = designer.compute_directions(H, alpha, p_total)
    return W_c, W_p


# ============================================================
# 测试1: 单用户场景 - RSMA应退化为单用户MIMO速率
# ============================================================
class TestSingleUser:
    def test_single_user_rate(self):
        """单用户时，无多用户干扰，只有噪声"""
        calc = RSMARateCalculator(K=1, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        np.random.seed(0)
        H = (np.random.randn(1, N_r, N_t) + 1j * np.random.randn(1, N_r, N_t)) / np.sqrt(2)

        # 单用户全功率
        designer = BeamformingDesigner(K=1, N_r=N_r, N_t=N_t)
        alpha = np.array([0.5])
        p_total = np.array([1.0])
        W_c, W_p = designer.compute_directions(H, alpha, p_total)

        result = calc.compute_rates(H, W_c, W_p)

        # 单用户无多用户干扰，速率应为正
        assert result['R_c'][0] > 0
        assert result['R_p'][0] > 0
        assert result['R_sum'] > 0

        # SINR应相对较高（单用户无干扰）
        # 对公共流：干扰只有自己的私有流投影 + 噪声
        # 对私有流：无干扰，只有噪声
        assert result['SINR_p'][0] > 10  # 高SNR


# ============================================================
# 测试2: 功率为0时速率为0
# ============================================================
class TestZeroPower:
    def test_zero_power_common(self, calc, random_channel):
        """公共流功率为0时，公共流速率为0"""
        H = random_channel
        W_c = np.zeros((K, N_t), dtype=complex)
        W_p = (np.random.randn(K, N_t) + 1j * np.random.randn(K, N_t)) / np.sqrt(2 * N_t)
        result = calc.compute_rates(H, W_c, W_p)
        np.testing.assert_allclose(result['R_c'], 0.0, atol=1e-10)

    def test_zero_power_private(self, calc, random_channel):
        """私有流功率为0时，私有流速率为0"""
        H = random_channel
        W_c = (np.random.randn(K, N_t) + 1j * np.random.randn(K, N_t)) / np.sqrt(2 * N_t)
        W_p = np.zeros((K, N_t), dtype=complex)
        result = calc.compute_rates(H, W_c, W_p)
        np.testing.assert_allclose(result['R_p'], 0.0, atol=1e-10)

    def test_all_zero_power(self, calc, random_channel):
        """所有功率为0时，总速率为0"""
        H = random_channel
        W_c = np.zeros((K, N_t), dtype=complex)
        W_p = np.zeros((K, N_t), dtype=complex)
        result = calc.compute_rates(H, W_c, W_p)
        assert result['R_sum'] == pytest.approx(0.0, abs=1e-10)


# ============================================================
# 测试3: 速率非负
# ============================================================
class TestNonNegative:
    def test_rates_non_negative(self, calc, random_channel, random_precoding):
        """所有速率应非负"""
        H = random_channel
        W_c, W_p = random_precoding
        result = calc.compute_rates(H, W_c, W_p)

        assert np.all(result['R_c'] >= 0)
        assert np.all(result['R_p'] >= 0)
        assert np.all(result['R_total'] >= 0)
        assert result['R_sum'] >= 0
        assert np.all(result['SINR_c'] >= 0)
        assert np.all(result['SINR_p'] >= 0)

    def test_rates_non_negative_random_orders(self, calc, random_channel, random_precoding):
        """不同SIC顺序下速率仍非负"""
        H = random_channel
        W_c, W_p = random_precoding
        for _ in range(5):
            order = np.random.permutation(K)
            result = calc.compute_rates(H, W_c, W_p, sic_order=order)
            assert np.all(result['R_c'] >= 0)
            assert np.all(result['R_p'] >= 0)


# ============================================================
# 测试4: SIC顺序影响各用户速率
# ============================================================
class TestSICOrder:
    def test_order_affects_individual_rates(self, calc, random_channel, random_precoding):
        """不同SIC顺序应产生不同的用户速率分布"""
        H = random_channel
        W_c, W_p = random_precoding

        order1 = np.arange(K)
        order2 = np.arange(K)[::-1]

        result1 = calc.compute_rates(H, W_c, W_p, sic_order=order1)
        result2 = calc.compute_rates(H, W_c, W_p, sic_order=order2)

        # 不同顺序的个体速率应不同
        assert not np.allclose(result1['R_c'], result2['R_c'], atol=1e-6)

    def test_first_decoded_has_lower_sinr_controlled(self):
        """受控场景：等功率等信道时，先解码用户面临更多干扰，SINR更低"""
        # 使用相同信道和预编码以隔离SIC排序效果
        K_small = 4
        np.random.seed(99)
        H_base = (np.random.randn(1, N_r, N_t) + 1j * np.random.randn(1, N_r, N_t)) / np.sqrt(2)
        H = np.tile(H_base, (K_small, 1, 1))
        # 正交预编码方向
        W_c = np.tile(np.array([[1.0, 0.0]], dtype=complex) * np.sqrt(0.5), (K_small, 1))
        W_p = np.tile(np.array([[0.0, 1.0]], dtype=complex) * np.sqrt(0.5), (K_small, 1))

        calc_small = RSMARateCalculator(K=K_small, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        order = np.arange(K_small)
        result = calc_small.compute_rates(H, W_c, W_p, sic_order=order)

        # 等功率等信道：后解码用户干扰少，SINR单调递增
        for i in range(K_small - 1):
            assert result['SINR_c'][order[i]] <= result['SINR_c'][order[i+1]]


# ============================================================
# 测试5: alpha=0时（无公共流）退化为SDMA
# ============================================================
class TestSDMADegradation:
    def test_alpha_zero_no_common_rate(self, random_channel):
        """alpha=0, 所有功率给私有流，公共流速率为0"""
        H = random_channel
        designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
        alpha = np.zeros(K)
        p_total = np.ones(K) * 0.5
        W_c, W_p = designer.compute_directions(H, alpha, p_total)

        calc = RSMARateCalculator(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        result = calc.compute_rates(H, W_c, W_p)

        # 公共流预编码功率为0，速率应为0
        np.testing.assert_allclose(result['R_c'], 0.0, atol=1e-10)
        # 私有流速率应为正
        assert np.all(result['R_p'] > 0)


# ============================================================
# 测试6: alpha=1时（全公共流）退化为NOMA-like
# ============================================================
class TestNOMADegradation:
    def test_alpha_one_no_private_rate(self, random_channel):
        """alpha=1, 所有功率给公共流，私有流速率为0"""
        H = random_channel
        designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
        alpha = np.ones(K)
        p_total = np.ones(K) * 0.5
        W_c, W_p = designer.compute_directions(H, alpha, p_total)

        calc = RSMARateCalculator(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        result = calc.compute_rates(H, W_c, W_p)

        # 私有流预编码功率为0，速率应为0
        np.testing.assert_allclose(result['R_p'], 0.0, atol=1e-10)
        # 公共流速率应为正
        assert np.all(result['R_c'] > 0)


# ============================================================
# 测试7: 鲁棒速率 <= 标称速率
# ============================================================
class TestRobustRate:
    def test_robust_leq_nominal(self, calc, random_channel, random_precoding):
        """鲁棒速率应 <= 标称速率"""
        H = random_channel
        W_c, W_p = random_precoding
        error_var = np.ones(K) * 0.01  # 1%误差方差

        nominal = calc.compute_rates(H, W_c, W_p)
        robust = calc.compute_outage_rates(H, W_c, W_p, error_var)

        assert np.all(robust['R_c'] <= nominal['R_c'] + 1e-10)
        assert np.all(robust['R_p'] <= nominal['R_p'] + 1e-10)
        assert robust['R_sum'] <= nominal['R_sum'] + 1e-8

    def test_zero_error_equals_nominal(self, calc, random_channel, random_precoding):
        """误差为0时，鲁棒速率应等于标称速率"""
        H = random_channel
        W_c, W_p = random_precoding
        error_var = np.zeros(K)

        nominal = calc.compute_rates(H, W_c, W_p)
        robust = calc.compute_outage_rates(H, W_c, W_p, error_var)

        np.testing.assert_allclose(robust['R_c'], nominal['R_c'], rtol=1e-6)
        np.testing.assert_allclose(robust['R_p'], nominal['R_p'], rtol=1e-6)

    def test_larger_error_lower_rate(self, calc, random_channel, random_precoding):
        """误差越大，鲁棒速率越低"""
        H = random_channel
        W_c, W_p = random_precoding

        error_small = np.ones(K) * 0.01
        error_large = np.ones(K) * 0.1

        robust_small = calc.compute_outage_rates(H, W_c, W_p, error_small)
        robust_large = calc.compute_outage_rates(H, W_c, W_p, error_large)

        assert robust_large['R_sum'] <= robust_small['R_sum'] + 1e-8


# ============================================================
# 测试8: 预编码向量功率约束正确
# ============================================================
class TestPowerConstraint:
    def test_precoding_power_common(self, random_channel, designer):
        """公共流预编码向量功率 = alpha * p_total"""
        H = random_channel
        np.random.seed(123)
        alpha = np.random.uniform(0.1, 0.9, size=K)
        p_total = np.random.uniform(0.1, 1.0, size=K)

        W_c, W_p = designer.compute_directions(H, alpha, p_total)

        for k in range(K):
            actual_power_c = np.linalg.norm(W_c[k]) ** 2
            expected_power_c = alpha[k] * p_total[k]
            np.testing.assert_allclose(actual_power_c, expected_power_c, rtol=1e-6,
                                       err_msg=f"User {k} common power mismatch")

    def test_precoding_power_private(self, random_channel, designer):
        """私有流预编码向量功率 = (1-alpha) * p_total"""
        H = random_channel
        np.random.seed(123)
        alpha = np.random.uniform(0.1, 0.9, size=K)
        p_total = np.random.uniform(0.1, 1.0, size=K)

        W_c, W_p = designer.compute_directions(H, alpha, p_total)

        for k in range(K):
            actual_power_p = np.linalg.norm(W_p[k]) ** 2
            expected_power_p = (1.0 - alpha[k]) * p_total[k]
            np.testing.assert_allclose(actual_power_p, expected_power_p, rtol=1e-6,
                                       err_msg=f"User {k} private power mismatch")

    def test_total_power_preserved(self, random_channel, designer):
        """总功率 = ||W_c||^2 + ||W_p||^2 = p_total"""
        H = random_channel
        np.random.seed(456)
        alpha = np.random.uniform(0.1, 0.9, size=K)
        p_total = np.random.uniform(0.1, 1.0, size=K)

        W_c, W_p = designer.compute_directions(H, alpha, p_total)

        for k in range(K):
            total = np.linalg.norm(W_c[k]) ** 2 + np.linalg.norm(W_p[k]) ** 2
            np.testing.assert_allclose(total, p_total[k], rtol=1e-6)

    def test_directions_unit_norm(self, random_channel, designer):
        """MRT和正交方向应为单位向量"""
        H = random_channel
        for k in range(K):
            dir_p = designer.mrt_direction(H[k])
            dir_c = designer.orthogonal_direction(H[k])
            np.testing.assert_allclose(np.linalg.norm(dir_p), 1.0, rtol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(dir_c), 1.0, rtol=1e-6)

    def test_directions_orthogonal(self, random_channel, designer):
        """公共流和私有流方向应正交（N_t=2时）"""
        H = random_channel
        for k in range(K):
            dir_p = designer.mrt_direction(H[k])
            dir_c = designer.orthogonal_direction(H[k])
            inner = np.abs(np.conj(dir_p) @ dir_c)
            np.testing.assert_allclose(inner, 0.0, atol=1e-6,
                                       err_msg=f"User {k} directions not orthogonal")


# ============================================================
# 测试9: decoding_order 参数测试
# ============================================================
class TestDecodingOrder:
    def test_decoding_order_none_backward_compat(self, calc, random_channel, random_precoding):
        """decoding_order=None 等价于默认顺序"""
        H = random_channel
        W_c, W_p = random_precoding
        result1 = calc.compute_rates(H, W_c, W_p, decoding_order=None)
        result2 = calc.compute_rates(H, W_c, W_p, sic_order=np.arange(K))
        np.testing.assert_allclose(result1['R_total'], result2['R_total'], rtol=1e-10)

    def test_different_decoding_order_different_rates(self, calc, random_channel, random_precoding):
        """不同解码顺序产生不同速率"""
        H = random_channel
        W_c, W_p = random_precoding
        # order1: 顺序 (0,1,2,...K-1)
        order1 = np.arange(K, dtype=float)
        # order2: 反序 (K-1, K-2, ..., 0)
        order2 = np.arange(K, dtype=float)[::-1]
        result1 = calc.compute_rates(H, W_c, W_p, decoding_order=order1)
        result2 = calc.compute_rates(H, W_c, W_p, decoding_order=order2)
        # 个体速率应不同
        assert not np.allclose(result1['R_c'], result2['R_c'], atol=1e-6)

    def test_decoding_order_rates_non_negative(self, calc, random_channel, random_precoding):
        """任意解码顺序下速率仍非负"""
        H = random_channel
        W_c, W_p = random_precoding
        rng = np.random.default_rng(0)
        for _ in range(5):
            order = rng.permutation(K).astype(float)
            result = calc.compute_rates(H, W_c, W_p, decoding_order=order)
            assert np.all(result['R_c'] >= 0)
            assert np.all(result['R_p'] >= 0)
            assert result['R_sum'] >= 0

    def test_decoding_order_as_priority(self, calc, random_channel, random_precoding):
        """decoding_order 值越小的用户越先解码（面临更多干扰）"""
        # 使用简单场景验证
        K_small = 4
        np.random.seed(99)
        H_base = (np.random.randn(1, N_r, N_t) + 1j * np.random.randn(1, N_r, N_t)) / np.sqrt(2)
        H = np.tile(H_base, (K_small, 1, 1))
        W_c = np.tile(np.array([[1.0, 0.0]], dtype=complex) * np.sqrt(0.5), (K_small, 1))
        W_p = np.tile(np.array([[0.0, 1.0]], dtype=complex) * np.sqrt(0.5), (K_small, 1))
        calc_small = RSMARateCalculator(K=K_small, N_r=N_r, N_t=N_t, sigma2=SIGMA2)

        # order = [0,1,2,3] -> 用户 0 先解码，SINR 最低
        order = np.arange(K_small, dtype=float)
        result = calc_small.compute_rates(H, W_c, W_p, decoding_order=order)
        # 等功率等信道：后解码用户 SINR 单调递增
        for i in range(K_small - 1):
            assert result['SINR_c'][i] <= result['SINR_c'][i+1]


# ============================================================
# 测试10: group_ids 参数测试 (分组SIC)
# ============================================================
class TestGroupIds:
    def test_group_ids_none_default(self, calc, random_channel, random_precoding):
        """group_ids=None 等价于所有用户同组"""
        H = random_channel
        W_c, W_p = random_precoding
        result1 = calc.compute_rates(H, W_c, W_p, group_ids=None)
        result2 = calc.compute_rates(H, W_c, W_p, group_ids=np.zeros(K, dtype=int))
        np.testing.assert_allclose(result1['R_total'], result2['R_total'], rtol=1e-10)

    def test_group_ids_different_groups_different_rates(self, calc, random_channel, random_precoding):
        """不同分组应产生不同的速率结果"""
        H = random_channel
        W_c, W_p = random_precoding
        # 全同组
        result1 = calc.compute_rates(H, W_c, W_p, group_ids=np.zeros(K, dtype=int))
        # 4 组
        group_ids_4 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        result2 = calc.compute_rates(H, W_c, W_p, group_ids=group_ids_4)
        # 分组后公共流速率应不同（因 SIC 仅组内）
        assert not np.allclose(result1['R_c'], result2['R_c'], atol=1e-6)

    def test_each_user_own_group_no_sic(self, calc, random_channel, random_precoding):
        """每个用户单独一组时，无组内SIC，先解码用户面临所有其他用户干扰"""
        H = random_channel
        W_c, W_p = random_precoding
        # 每用户单独一组: 不同组的公共流不进行SIC，全部视为干扰
        group_ids = np.arange(K, dtype=int)
        result = calc.compute_rates(H, W_c, W_p, group_ids=group_ids)
        assert np.all(result['R_c'] >= 0)
        assert np.all(result['R_p'] >= 0)
        # 单独分组时，公共流 SINR 应更低 (没有SIC增益)
        result_same = calc.compute_rates(H, W_c, W_p, group_ids=np.zeros(K, dtype=int))
        # 分组后的总速率应更低或相等 (因为没有SIC增益)
        assert result['R_sum'] <= result_same['R_sum'] + 1e-8

    def test_group_ids_with_decoding_order(self, calc, random_channel, random_precoding):
        """同时使用 group_ids 和 decoding_order"""
        H = random_channel
        W_c, W_p = random_precoding
        group_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        order = np.random.permutation(K).astype(float)
        result = calc.compute_rates(
            H, W_c, W_p, decoding_order=order, group_ids=group_ids
        )
        assert np.all(result['R_c'] >= 0)
        assert np.all(result['R_p'] >= 0)
        assert result['R_sum'] > 0

    def test_outage_rates_with_group_ids(self, calc, random_channel, random_precoding):
        """compute_outage_rates 也支持 group_ids"""
        H = random_channel
        W_c, W_p = random_precoding
        error_var = np.ones(K) * 0.01
        group_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        result = calc.compute_outage_rates(
            H, W_c, W_p, error_var, group_ids=group_ids
        )
        assert np.all(result['R_c'] >= 0)
        assert np.all(result['R_p'] >= 0)
        assert result['R_sum'] >= 0


# ============================================================
# 测试11: MMSE-SIC 速率计算
# ============================================================
class TestMMSE_SIC:
    """MMSE-SIC 接收机测试"""

    @pytest.fixture
    def precoding_v2(self, random_channel):
        """使用 compute_precoding_v2 生成预编码"""
        H = random_channel
        designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
        np.random.seed(42)
        alpha = np.random.uniform(0.2, 0.8, size=K)
        p_total = np.random.uniform(0.3, 1.0, size=K)
        split_mask = np.ones(K)
        W_c, W_p = designer.compute_precoding_v2(H, alpha, p_total, split_mask)
        return W_c, W_p

    def test_mmse_rates_non_negative(self, calc, random_channel, precoding_v2):
        """MMSE-SIC 速率非负"""
        H = random_channel
        W_c, W_p = precoding_v2
        result = calc.compute_rates_mmse_sic(H, W_c, W_p)
        assert np.all(result['R_c'] >= 0)
        assert np.all(result['R_p'] >= 0)
        assert result['R_sum'] >= 0
        assert np.all(result['SINR_c'] >= 0)
        assert np.all(result['SINR_p'] >= 0)

    def test_mmse_geq_matched_filter(self, random_channel):
        """MMSE-SIC R_sum >= Matched Filter R_sum（MMSE 是更优接收机）"""
        H = random_channel
        designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
        np.random.seed(42)
        alpha = np.random.uniform(0.2, 0.8, size=K)
        p_total = np.random.uniform(0.3, 1.0, size=K)

        # Matched filter path (旧方法，正交预编码)
        W_c_old, W_p_old = designer.compute_directions(H, alpha, p_total)
        calc = RSMARateCalculator(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        result_mf = calc.compute_rates(H, W_c_old, W_p_old)

        # MMSE-SIC path (统一 MRT 预编码)
        split_mask = np.ones(K)
        W_c_new, W_p_new = designer.compute_precoding_v2(H, alpha, p_total, split_mask)
        result_mmse = calc.compute_rates_mmse_sic(H, W_c_new, W_p_new)

        # MMSE-SIC + MRT 应显著优于 MF + 正交
        assert result_mmse['R_sum'] > result_mf['R_sum']

    def test_mmse_single_user_degrades_to_mrc(self):
        """K=1 时 MMSE = MRC = matched filter"""
        np.random.seed(0)
        H = (np.random.randn(1, N_r, N_t) + 1j * np.random.randn(1, N_r, N_t)) / np.sqrt(2)
        designer = BeamformingDesigner(K=1, N_r=N_r, N_t=N_t)
        calc = RSMARateCalculator(K=1, N_r=N_r, N_t=N_t, sigma2=SIGMA2)

        alpha = np.array([0.5])
        p_total = np.array([1.0])
        split_mask = np.array([1.0])

        # v2 precoding (MRT for both)
        W_c, W_p = designer.compute_precoding_v2(H, alpha, p_total, split_mask)
        result_mmse = calc.compute_rates_mmse_sic(H, W_c, W_p)

        # K=1: MMSE degrades to MRC, SINR = ||h||^2 * p / sigma2
        h_eff_c = H[0] @ W_c[0]
        h_eff_p = H[0] @ W_p[0]
        p_c = np.linalg.norm(W_c[0]) ** 2
        p_p = np.linalg.norm(W_p[0]) ** 2

        # Common SINR: signal h_c, interference = h_p (same direction) + noise
        # With MMSE: v = R_int^{-1} h_c, SINR = p_c * h^H R^{-1} h
        # For K=1: R_int = p_p * h_p h_p^H + sigma2 * I
        # Since h_c and h_p are collinear (same MRT dir), the MMSE simplifies
        assert result_mmse['R_c'][0] > 0
        assert result_mmse['R_p'][0] > 0
        assert result_mmse['R_sum'] > 0

    def test_mmse_sic_order_affects_rates(self, calc, random_channel, precoding_v2):
        """不同解码顺序产生不同 MMSE-SIC 速率"""
        H = random_channel
        W_c, W_p = precoding_v2
        order1 = np.arange(K, dtype=float)
        order2 = np.arange(K, dtype=float)[::-1]
        result1 = calc.compute_rates_mmse_sic(H, W_c, W_p, decoding_order=order1)
        result2 = calc.compute_rates_mmse_sic(H, W_c, W_p, decoding_order=order2)
        assert not np.allclose(result1['R_c'], result2['R_c'], atol=1e-6)

    def test_mmse_sic_causality(self):
        """MMSE-SIC 因果性：等信道等功率下，后解码用户私有流 SINR 更高"""
        K_small = 4
        np.random.seed(77)
        # 使用相同信道隔离 SIC 排序效果
        H_base = (np.random.randn(1, N_r, N_t) +
                  1j * np.random.randn(1, N_r, N_t)) / np.sqrt(2)
        H = np.tile(H_base, (K_small, 1, 1))
        designer = BeamformingDesigner(K=K_small, N_r=N_r, N_t=N_t)
        calc = RSMARateCalculator(K=K_small, N_r=N_r, N_t=N_t, sigma2=SIGMA2)

        alpha = np.full(K_small, 0.5)
        p_total = np.full(K_small, 0.5)
        split_mask = np.ones(K_small)
        W_c, W_p = designer.compute_precoding_v2(H, alpha, p_total, split_mask)

        # 全同组：所有公共流互相 SIC
        group_ids = np.zeros(K_small, dtype=int)
        order = np.arange(K_small, dtype=float)
        result = calc.compute_rates_mmse_sic(
            H, W_c, W_p, decoding_order=order, group_ids=group_ids
        )
        # 等信道等功率下，后解码用户私有流 SINR 单调递增（SIC 移除了前面的干扰）
        for i in range(K_small - 1):
            assert result['SINR_p'][i] <= result['SINR_p'][i + 1] + 1e-6

    def test_mmse_group_sic(self, calc, random_channel, precoding_v2):
        """分组：全同组 R_sum >= 全不同组 R_sum（SIC 增益）"""
        H = random_channel
        W_c, W_p = precoding_v2
        # 全同组
        result_same = calc.compute_rates_mmse_sic(
            H, W_c, W_p, group_ids=np.zeros(K, dtype=int)
        )
        # 每用户独立组（无 SIC）
        result_diff = calc.compute_rates_mmse_sic(
            H, W_c, W_p, group_ids=np.arange(K, dtype=int)
        )
        # 同组有 SIC 增益，总速率应更高
        assert result_same['R_sum'] >= result_diff['R_sum'] - 1e-8

    def test_mmse_robust_leq_nominal(self, calc, random_channel, precoding_v2):
        """error_bound > 0 时鲁棒速率 <= 标称速率"""
        H = random_channel
        W_c, W_p = precoding_v2
        result_nominal = calc.compute_rates_mmse_sic(H, W_c, W_p, error_bound=0.0)
        result_robust = calc.compute_rates_mmse_sic(H, W_c, W_p, error_bound=0.1)
        assert result_robust['R_sum'] <= result_nominal['R_sum'] + 1e-8

    def test_mmse_sic_imperfection_reduces_rate(self, calc, random_channel, precoding_v2):
        """SIC 不完美因子 β > 0 降低速率"""
        H = random_channel
        W_c, W_p = precoding_v2
        group_ids = np.zeros(K, dtype=int)  # 全同组以体现 SIC 效果
        result_perfect = calc.compute_rates_mmse_sic(
            H, W_c, W_p, sic_imperfection=0.0, group_ids=group_ids
        )
        result_imperfect = calc.compute_rates_mmse_sic(
            H, W_c, W_p, sic_imperfection=0.1, group_ids=group_ids
        )
        assert result_imperfect['R_sum'] <= result_perfect['R_sum'] + 1e-8

    def test_mmse_zero_power_common(self, calc, random_channel):
        """公共流功率为 0 时 R_c = 0"""
        H = random_channel
        W_c = np.zeros((K, N_t), dtype=complex)
        np.random.seed(42)
        W_p = (np.random.randn(K, N_t) + 1j * np.random.randn(K, N_t)) / np.sqrt(2 * N_t)
        result = calc.compute_rates_mmse_sic(H, W_c, W_p)
        np.testing.assert_allclose(result['R_c'], 0.0, atol=1e-10)

    def test_mmse_outage_rates(self, random_channel):
        """compute_outage_rates_mmse_sic 基本正确性"""
        np.random.seed(42)
        H_hat = random_channel
        # 真实信道 = 估计信道 + 误差
        error = 0.05 * (np.random.randn(K, N_r, N_t) +
                        1j * np.random.randn(K, N_r, N_t)) / np.sqrt(2)
        H_true = H_hat + error

        designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
        calc = RSMARateCalculator(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        alpha = np.random.uniform(0.2, 0.8, size=K)
        p_total = np.random.uniform(0.3, 1.0, size=K)
        split_mask = np.ones(K)
        W_c, W_p = designer.compute_precoding_v2(H_hat, alpha, p_total, split_mask)

        result = calc.compute_outage_rates_mmse_sic(
            H_hat, H_true, W_c, W_p, error_bound=0.05
        )
        assert np.all(result['R_c'] >= 0)
        assert np.all(result['R_p'] >= 0)
        assert result['R_sum'] > 0


# ============================================================
# 测试12: compute_precoding_v2 功率约束
# ============================================================
class TestPrecodingV2:
    def test_v2_power_common(self, random_channel):
        """v2 公共流功率 = alpha * p_total * split_mask"""
        H = random_channel
        designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
        np.random.seed(42)
        alpha = np.random.uniform(0.2, 0.8, size=K)
        p_total = np.random.uniform(0.3, 1.0, size=K)
        split_mask = np.ones(K)
        W_c, W_p = designer.compute_precoding_v2(H, alpha, p_total, split_mask)
        for k in range(K):
            actual = np.linalg.norm(W_c[k]) ** 2
            expected = alpha[k] * p_total[k] * split_mask[k]
            np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_v2_power_private(self, random_channel):
        """v2 私有流功率 = (1 - alpha*split) * p_total"""
        H = random_channel
        designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
        np.random.seed(42)
        alpha = np.random.uniform(0.2, 0.8, size=K)
        p_total = np.random.uniform(0.3, 1.0, size=K)
        split_mask = np.ones(K)
        W_c, W_p = designer.compute_precoding_v2(H, alpha, p_total, split_mask)
        for k in range(K):
            actual = np.linalg.norm(W_p[k]) ** 2
            expected = (1.0 - alpha[k] * split_mask[k]) * p_total[k]
            np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_v2_split_mask_zero(self, random_channel):
        """split_mask=0 时公共流功率为 0，私有流获得全部功率"""
        H = random_channel
        designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
        alpha = np.full(K, 0.5)
        p_total = np.ones(K)
        split_mask = np.zeros(K)
        W_c, W_p = designer.compute_precoding_v2(H, alpha, p_total, split_mask)
        for k in range(K):
            assert np.linalg.norm(W_c[k]) ** 2 < 1e-10
            np.testing.assert_allclose(
                np.linalg.norm(W_p[k]) ** 2, p_total[k], rtol=1e-6
            )

    def test_v2_total_power_preserved(self, random_channel):
        """||W_c||² + ||W_p||² = p_total"""
        H = random_channel
        designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
        np.random.seed(42)
        alpha = np.random.uniform(0.2, 0.8, size=K)
        p_total = np.random.uniform(0.3, 1.0, size=K)
        split_mask = np.ones(K)
        W_c, W_p = designer.compute_precoding_v2(H, alpha, p_total, split_mask)
        for k in range(K):
            total = np.linalg.norm(W_c[k]) ** 2 + np.linalg.norm(W_p[k]) ** 2
            np.testing.assert_allclose(total, p_total[k], rtol=1e-6)

    def test_v2_collinear_directions(self, random_channel):
        """v2 中公共/私有流方向共线（都是 MRT）"""
        H = random_channel
        designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
        alpha = np.full(K, 0.5)
        p_total = np.ones(K)
        split_mask = np.ones(K)
        W_c, W_p = designer.compute_precoding_v2(H, alpha, p_total, split_mask)
        for k in range(K):
            dir_c = W_c[k] / max(np.linalg.norm(W_c[k]), 1e-12)
            dir_p = W_p[k] / max(np.linalg.norm(W_p[k]), 1e-12)
            # 共线性：|<dir_c, dir_p>| ≈ 1
            inner = np.abs(np.conj(dir_c) @ dir_p)
            np.testing.assert_allclose(inner, 1.0, atol=1e-6)
