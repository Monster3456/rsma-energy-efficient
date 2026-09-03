"""可微速率引擎一致性测试：与不可变 rate_calc/beamforming 数值对齐。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.env.beamforming import BeamformingDesigner  # noqa: E402
from src.env.rate_calc import RSMARateCalculator  # noqa: E402
from src.optimization.diff_engine import DiffRSMAEngine  # noqa: E402

pytestmark = pytest.mark.fast

SIGMA2 = 0.01


def _random_channel(K, N_r, N_t, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((K, N_r, N_t))
            + 1j * rng.standard_normal((K, N_r, N_t))) / np.sqrt(2.0)


def _make_case(K, N_r, N_t, seed, n_groups=1):
    rng = np.random.default_rng(seed)
    H = _random_channel(K, N_r, N_t, seed)
    alpha = rng.uniform(0.0, 1.0, size=K)
    p = rng.uniform(0.05, 1.0, size=K)
    split = rng.uniform(0.0, 1.0, size=K) if n_groups > 1 else np.full(K, 0.5)
    designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
    W_c, W_p = designer.compute_precoding_v2(H, alpha, p, split)
    order = rng.permutation(K).astype(np.float64)
    groups = (rng.integers(0, n_groups, size=K).astype(np.int64)
              if n_groups > 1 else np.zeros(K, dtype=np.int64))
    return H, W_c, W_p, alpha, p, split, order, groups


SIZES = [(5, 6, 2), (24, 16, 2)]


class TestRatesConsistency:
    @pytest.mark.parametrize("K,N_r,N_t", SIZES)
    @pytest.mark.parametrize("eps,beta", [(0.0, 0.0), (0.05, 0.0), (0.0, 0.1), (0.05, 0.1)])
    def test_rates_match_numpy(self, K, N_r, N_t, eps, beta):
        H, W_c, W_p, *_ , order, groups = _make_case(K, N_r, N_t, seed=7)
        calc = RSMARateCalculator(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        engine = DiffRSMAEngine(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)

        ref = calc.compute_rates_mmse_sic(
            H, W_c, W_p, decoding_order=order, group_ids=groups,
            error_bound=eps, sic_imperfection=beta,
        )
        out = engine.rates(
            H, W_c, W_p, decoding_order=order, group_ids=groups,
            error_bound=eps, sic_imperfection=beta,
        )
        for key in ("R_c", "R_p", "SINR_c", "SINR_p"):
            np.testing.assert_allclose(
                out[key].detach().cpu().numpy(), ref[key], atol=1e-9, rtol=1e-9,
                err_msg=f"{key} mismatch (eps={eps}, beta={beta})")
        assert abs(float(out["R_sum"].detach()) - ref["R_sum"]) < 1e-8

    @pytest.mark.parametrize("K,N_r,N_t", SIZES)
    def test_rates_match_numpy_multigroup(self, K, N_r, N_t):
        H, W_c, W_p, *_ , order, groups = _make_case(K, N_r, N_t, seed=11, n_groups=3)
        calc = RSMARateCalculator(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        engine = DiffRSMAEngine(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        ref = calc.compute_rates_mmse_sic(
            H, W_c, W_p, decoding_order=order, group_ids=groups, error_bound=0.05)
        out = engine.rates(
            H, W_c, W_p, decoding_order=order, group_ids=groups, error_bound=0.05)
        np.testing.assert_allclose(
            out["R_total"].detach().cpu().numpy(), ref["R_total"], atol=1e-9, rtol=1e-9)

    @pytest.mark.parametrize("K,N_r,N_t", SIZES)
    def test_rates_match_numpy_no_order(self, K, N_r, N_t):
        """decoding_order=None 时的默认顺序（0..K-1）一致性。"""
        H, W_c, W_p, *_ , _, groups = _make_case(K, N_r, N_t, seed=13)
        calc = RSMARateCalculator(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        engine = DiffRSMAEngine(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        ref = calc.compute_rates_mmse_sic(H, W_c, W_p, group_ids=groups, error_bound=0.05)
        out = engine.rates(H, W_c, W_p, group_ids=groups, error_bound=0.05)
        np.testing.assert_allclose(
            out["R_total"].detach().cpu().numpy(), ref["R_total"], atol=1e-9, rtol=1e-9)


class TestOutageConsistency:
    @pytest.mark.parametrize("K,N_r,N_t", SIZES)
    @pytest.mark.parametrize("eps", [0.05, 0.2])
    def test_outage_rates_match_numpy(self, K, N_r, N_t, eps):
        H_hat, W_c, W_p, *_ , order, groups = _make_case(K, N_r, N_t, seed=17)
        H_true = _random_channel(K, N_r, N_t, seed=18)
        calc = RSMARateCalculator(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        engine = DiffRSMAEngine(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        ref = calc.compute_outage_rates_mmse_sic(
            H_hat, H_true, W_c, W_p, error_bound=eps,
            decoding_order=order, group_ids=groups)
        out = engine.outage_rates(
            H_hat, H_true, W_c, W_p, error_bound=eps,
            decoding_order=order, group_ids=groups)
        for key in ("R_c", "R_p"):
            np.testing.assert_allclose(
                out[key].detach().cpu().numpy(), ref[key], atol=1e-9, rtol=1e-9)


class TestPrecodingChain:
    @pytest.mark.parametrize("K,N_r,N_t", SIZES)
    def test_end_to_end_action_chain(self, K, N_r, N_t):
        """engine.precoding+rates 与 beamforming+rate_calc 端到端一致
        （SVD 相位不唯一但速率相位不变）。"""
        H, _, _, alpha, p, split, order, groups = _make_case(K, N_r, N_t, seed=23)
        calc = RSMARateCalculator(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
        W_c_np, W_p_np = designer.compute_precoding_v2(H, alpha, p, split)
        ref = calc.compute_outage_rates_mmse_sic(
            H, H, W_c_np, W_p_np, error_bound=0.05,
            decoding_order=order, group_ids=groups)

        engine = DiffRSMAEngine(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        W_c_t, W_p_t = engine.precoding(H, alpha, p, split)
        out = engine.outage_rates(
            H, H, W_c_t, W_p_t, error_bound=0.05,
            decoding_order=order, group_ids=groups)
        np.testing.assert_allclose(
            out["R_total"].detach().cpu().numpy(), ref["R_total"], atol=1e-9, rtol=1e-9)

    def test_zero_power_user_zero_rate(self):
        """零功率用户速率恒为 0，且不干扰其他用户（与 numpy 行为一致）。"""
        K, N_r, N_t = 6, 8, 2
        H, _, _, alpha, p, split, order, groups = _make_case(K, N_r, N_t, seed=29)
        p = p.copy()
        p[0] = 0.0
        calc = RSMARateCalculator(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        designer = BeamformingDesigner(K=K, N_r=N_r, N_t=N_t)
        W_c_np, W_p_np = designer.compute_precoding_v2(H, alpha, p, split)
        ref = calc.compute_rates_mmse_sic(
            H, W_c_np, W_p_np, decoding_order=order, group_ids=groups, error_bound=0.05)

        engine = DiffRSMAEngine(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        W_c_t, W_p_t = engine.precoding(H, alpha, p, split)
        out = engine.rates(
            H, W_c_t, W_p_t, decoding_order=order, group_ids=groups, error_bound=0.05)
        # sqrt 下限 1e-16 使零功率用户速率 ~1e-16（数值零），不再精确为 0
        assert float(out["R_total"][0].detach()) < 1e-12
        np.testing.assert_allclose(
            out["R_total"].detach().cpu().numpy(), ref["R_total"], atol=1e-9, rtol=1e-9)


class TestGradientFlow:
    def test_power_gradient(self):
        K, N_r, N_t = 24, 16, 2
        H, _, _, alpha, p, split, order, groups = _make_case(K, N_r, N_t, seed=31)
        H_true = _random_channel(K, N_r, N_t, seed=32)
        engine = DiffRSMAEngine(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2)
        p_t = torch.tensor(p, dtype=torch.float64, requires_grad=True)
        W_c, W_p = engine.precoding(H, alpha, p_t, split)
        out = engine.outage_rates(
            H, H_true, W_c, W_p, error_bound=0.05,
            decoding_order=order, group_ids=groups)
        out["R_sum"].backward()
        assert p_t.grad is not None
        assert torch.isfinite(p_t.grad).all()
        assert float(p_t.grad.abs().sum()) > 0.0

    def test_float32_mode(self):
        K, N_r, N_t = 24, 16, 2
        H, _, _, alpha, p, split, order, groups = _make_case(K, N_r, N_t, seed=33)
        engine32 = DiffRSMAEngine(K=K, N_r=N_r, N_t=N_t, sigma2=SIGMA2, dtype=torch.float32)
        p_t = torch.tensor(p, dtype=torch.float32, requires_grad=True)
        W_c, W_p = engine32.precoding(H, alpha, p_t, split)
        out = engine32.rates(H, W_c, W_p, decoding_order=order, group_ids=groups,
                             error_bound=0.05)
        out["R_sum"].backward()
        assert torch.isfinite(p_t.grad).all()
