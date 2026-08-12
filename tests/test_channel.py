"""信道模型单元测试

覆盖:
1. reset/step 接口返回正确的 dict 结构与维度
2. AR(1) 统计特性: E[||H||_F^2] ≈ N_r * N_t (稳态)
3. 时间相关性: corr(H[t], H[t+1]) ≈ rho
4. CSI 误差范数被椭球 epsilon_k 严格约束
5. perfect CSI 下 H_hat == H_true, 误差为 0
6. get_eigenvalues 输出形状与降序排列
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import SystemConfig
from src.env.channel import ChannelGenerator

# fast 子集：纯物理/统计特性校验，轻量且确定性
pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def cfg() -> SystemConfig:
    return SystemConfig()


@pytest.fixture
def gen_perfect(cfg: SystemConfig) -> ChannelGenerator:
    return ChannelGenerator(
        K=cfg.K, N_r=cfg.N_r, N_t=cfg.N_t,
        rho=cfg.rho, csi_mode="perfect",
        epsilon_csi=cfg.epsilon_csi, csi_delay=cfg.csi_delay,
        seed=42,
    )


@pytest.fixture
def gen_imperfect(cfg: SystemConfig) -> ChannelGenerator:
    # Task#20: 默认使用固定 epsilon_csi (snr_dependent_csi=False) 以保证旧有静态不变性测试
    return ChannelGenerator(
        K=cfg.K, N_r=cfg.N_r, N_t=cfg.N_t,
        rho=cfg.rho, csi_mode="imperfect",
        epsilon_csi=cfg.epsilon_csi, csi_delay=cfg.csi_delay,
        seed=42,
        snr_dependent_csi=False,
    )


# ---------------------------------------------------------------------------
# 1. 接口与维度
# ---------------------------------------------------------------------------
def test_reset_returns_dict_with_required_keys(gen_perfect: ChannelGenerator, cfg: SystemConfig):
    obs = gen_perfect.reset()
    assert isinstance(obs, dict)
    for key in ("H_true", "H_hat", "error_variance", "channel_gains"):
        assert key in obs, f"missing key {key}"

    K, N_r, N_t = cfg.K, cfg.N_r, cfg.N_t
    assert obs["H_true"].shape == (K, N_r, N_t)
    assert obs["H_hat"].shape == (K, N_r, N_t)
    assert obs["error_variance"].shape == (K,)
    assert obs["channel_gains"].shape == (K,)
    # 复数与实数类型检查
    assert np.iscomplexobj(obs["H_true"])
    assert np.iscomplexobj(obs["H_hat"])
    assert np.isrealobj(obs["error_variance"])
    assert np.isrealobj(obs["channel_gains"])


def test_step_returns_same_structure(gen_perfect: ChannelGenerator, cfg: SystemConfig):
    gen_perfect.reset()
    obs = gen_perfect.step()
    K, N_r, N_t = cfg.K, cfg.N_r, cfg.N_t
    assert obs["H_true"].shape == (K, N_r, N_t)
    assert obs["H_hat"].shape == (K, N_r, N_t)
    assert obs["channel_gains"].shape == (K,)


def test_step_before_reset_raises(gen_perfect: ChannelGenerator):
    with pytest.raises(RuntimeError):
        gen_perfect.step()


# ---------------------------------------------------------------------------
# 2. AR(1) 统计特性
# ---------------------------------------------------------------------------
def test_average_frobenius_norm_matches_theory(cfg: SystemConfig):
    """E[||H_k||_F^2] = N_r * N_t (稳态, 标准 CN(0, I) 元素)."""
    rng_seed = 7
    gen = ChannelGenerator(
        K=cfg.K, N_r=cfg.N_r, N_t=cfg.N_t,
        rho=cfg.rho, csi_mode="perfect",
        epsilon_csi=cfg.epsilon_csi, csi_delay=cfg.csi_delay,
        seed=rng_seed,
    )
    gen.reset()
    n_steps = 500
    accum = 0.0
    count = 0
    for _ in range(n_steps):
        obs = gen.step()
        accum += float(np.mean(obs["channel_gains"]))
        count += 1
    avg = accum / count
    expected = cfg.N_r * cfg.N_t
    # 5% 相对误差
    assert abs(avg - expected) / expected < 0.05, (
        f"avg gain {avg:.3f} deviates from {expected} too much"
    )


def test_temporal_correlation_matches_rho(cfg: SystemConfig):
    """单步样本相关性 corr(H[t], H[t+1]) ≈ rho."""
    gen = ChannelGenerator(
        K=cfg.K, N_r=cfg.N_r, N_t=cfg.N_t,
        rho=cfg.rho, csi_mode="perfect",
        epsilon_csi=cfg.epsilon_csi, csi_delay=cfg.csi_delay,
        seed=2024,
    )
    obs_prev = gen.reset()
    n_steps = 800

    # 累积 sum(H_t* . H_{t+1}) / (N_r*N_t*K), 实部应≈rho
    inner_sum = 0.0
    norm_prev_sum = 0.0
    norm_cur_sum = 0.0
    for _ in range(n_steps):
        obs_cur = gen.step()
        h_prev = obs_prev["H_true"]
        h_cur = obs_cur["H_true"]
        inner_sum += float(np.real(np.sum(np.conj(h_prev) * h_cur)))
        norm_prev_sum += float(np.sum(np.abs(h_prev) ** 2))
        norm_cur_sum += float(np.sum(np.abs(h_cur) ** 2))
        obs_prev = obs_cur

    corr = inner_sum / np.sqrt(norm_prev_sum * norm_cur_sum)
    assert abs(corr - cfg.rho) < 0.05, (
        f"corr={corr:.3f} deviates from rho={cfg.rho}"
    )


def test_initial_channel_is_complex_gaussian():
    """H[0] ~ CN(0, I): 实部、虚部均值≈0, 方差≈0.5."""
    K, N_r, N_t = 8, 8, 2
    gen = ChannelGenerator(K=K, N_r=N_r, N_t=N_t, rho=0.9, csi_mode="perfect", seed=11)

    samples_real = []
    samples_imag = []
    for _ in range(200):
        obs = gen.reset()
        h = obs["H_true"]
        samples_real.append(h.real.flatten())
        samples_imag.append(h.imag.flatten())

    real = np.concatenate(samples_real)
    imag = np.concatenate(samples_imag)

    assert abs(real.mean()) < 0.05
    assert abs(imag.mean()) < 0.05
    assert abs(real.var() - 0.5) < 0.05
    assert abs(imag.var() - 0.5) < 0.05


# ---------------------------------------------------------------------------
# 3. CSI 模式
# ---------------------------------------------------------------------------
def test_perfect_csi_zero_error(gen_perfect: ChannelGenerator):
    obs = gen_perfect.reset()
    assert np.allclose(obs["H_true"], obs["H_hat"])
    assert np.allclose(obs["error_variance"], 0.0)

    for _ in range(20):
        obs = gen_perfect.step()
        assert np.allclose(obs["H_true"], obs["H_hat"])
        assert np.allclose(obs["error_variance"], 0.0)


def test_imperfect_csi_error_within_ellipsoid(gen_imperfect: ChannelGenerator, cfg: SystemConfig):
    """每用户 ||H_k - H_hat_k||_F <= epsilon_k."""
    chi_factor = 3.0
    sigma_e2 = (1.0 - cfg.rho ** (2 * cfg.csi_delay)) + cfg.epsilon_csi ** 2
    eps_k = np.sqrt(cfg.N_r * cfg.N_t * sigma_e2) * chi_factor
    tol = 1e-9

    gen_imperfect.reset()
    for _ in range(50):
        obs = gen_imperfect.step()
        diff = obs["H_true"] - obs["H_hat"]
        norms = np.linalg.norm(diff.reshape(cfg.K, -1), axis=1)
        assert np.all(norms <= eps_k + tol), (
            f"max ||error||_F={norms.max():.4f} exceeds eps_k={eps_k:.4f}"
        )


def test_imperfect_csi_error_variance_formula(gen_imperfect: ChannelGenerator, cfg: SystemConfig):
    obs = gen_imperfect.reset()
    # error_variance 现在仅包含估计噪声 (epsilon_csi^2), 信道老化通过 H_hat 隐式体现
    expected = cfg.epsilon_csi ** 2
    assert np.allclose(obs["error_variance"], expected)


def test_imperfect_csi_introduces_real_error(gen_imperfect: ChannelGenerator):
    """imperfect 模式下经过若干步后, H_hat 应明显偏离 H_true."""
    gen_imperfect.reset()
    # 经过 csi_delay+1 步以后历史缓存才填满
    for _ in range(5):
        obs = gen_imperfect.step()
    diff = obs["H_true"] - obs["H_hat"]
    assert float(np.linalg.norm(diff)) > 1e-3


# ---------------------------------------------------------------------------
# 4. get_eigenvalues
# ---------------------------------------------------------------------------
def test_get_eigenvalues_shape_and_order(gen_perfect: ChannelGenerator, cfg: SystemConfig):
    obs = gen_perfect.reset()
    sv = gen_perfect.get_eigenvalues(obs["H_hat"])
    assert sv.shape == (cfg.K, 2)
    # 非负且降序
    assert np.all(sv >= 0)
    assert np.all(sv[:, 0] >= sv[:, 1] - 1e-9)


def test_get_eigenvalues_matches_svd_manual(gen_perfect: ChannelGenerator, cfg: SystemConfig):
    obs = gen_perfect.reset()
    sv = gen_perfect.get_eigenvalues(obs["H_hat"])
    for k in range(cfg.K):
        s_ref = np.linalg.svd(obs["H_hat"][k], compute_uv=False)
        n = min(2, s_ref.shape[0])
        assert np.allclose(sv[k, :n], s_ref[:n], atol=1e-8)


def test_get_eigenvalues_wrong_shape_raises(gen_perfect: ChannelGenerator, cfg: SystemConfig):
    bad = np.zeros((cfg.K + 1, cfg.N_r, cfg.N_t), dtype=np.complex128)
    with pytest.raises(ValueError):
        gen_perfect.get_eigenvalues(bad)


# ---------------------------------------------------------------------------
# 5. 可复现性
# ---------------------------------------------------------------------------
def test_reproducibility_with_same_seed(cfg: SystemConfig):
    g1 = ChannelGenerator(
        K=cfg.K, N_r=cfg.N_r, N_t=cfg.N_t, rho=cfg.rho,
        csi_mode="imperfect", epsilon_csi=cfg.epsilon_csi,
        csi_delay=cfg.csi_delay, seed=123,
        snr_dependent_csi=False,
    )
    g2 = ChannelGenerator(
        K=cfg.K, N_r=cfg.N_r, N_t=cfg.N_t, rho=cfg.rho,
        csi_mode="imperfect", epsilon_csi=cfg.epsilon_csi,
        csi_delay=cfg.csi_delay, seed=123,
        snr_dependent_csi=False,
    )
    o1 = g1.reset()
    o2 = g2.reset()
    assert np.allclose(o1["H_true"], o2["H_true"])
    assert np.allclose(o1["H_hat"], o2["H_hat"])
    for _ in range(5):
        s1 = g1.step()
        s2 = g2.step()
        assert np.allclose(s1["H_true"], s2["H_true"])
        assert np.allclose(s1["H_hat"], s2["H_hat"])


# ---------------------------------------------------------------------------
# 6. 参数校验
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_mode", ["partial", "", "PERFECT"])
def test_invalid_csi_mode_raises(bad_mode: str):
    with pytest.raises(ValueError):
        ChannelGenerator(K=4, N_r=4, N_t=2, csi_mode=bad_mode)


@pytest.mark.parametrize("bad_rho", [-0.1, 1.0, 1.5])
def test_invalid_rho_raises(bad_rho: float):
    with pytest.raises(ValueError):
        ChannelGenerator(K=4, N_r=4, N_t=2, rho=bad_rho)


def test_invalid_delay_raises():
    with pytest.raises(ValueError):
        ChannelGenerator(K=4, N_r=4, N_t=2, csi_delay=-1)


# ---------------------------------------------------------------------------
# Task#20: SNR 依赖估计噪声 + obs 使用 H_hat
# ---------------------------------------------------------------------------
def test_set_sigma2_updates_sigma_est_when_snr_dependent():
    """启用 snr_dependent_csi 后，sigma_est 应随 sigma2 变化。"""
    gen = ChannelGenerator(
        K=8, N_r=8, N_t=2, rho=0.9, csi_mode="imperfect",
        epsilon_csi=0.05, csi_delay=1, seed=0,
        snr_dependent_csi=True, pilot_power=1.0, epsilon_floor=0.01,
    )
    gen.set_sigma2(1.0)        # 低 SNR
    s_low = gen._current_sigma_est()
    gen.set_sigma2(0.001)      # 高 SNR
    s_high = gen._current_sigma_est()
    assert s_low > s_high, f"sigma_est should be larger at low SNR: low={s_low}, high={s_high}"
    # 高 SNR 时应接近 epsilon_floor=0.01
    assert s_high >= 0.01 - 1e-9
    assert s_high < 0.05  # 明显低于默认 epsilon_csi=0.05


def test_snr_dependent_disabled_falls_back_to_epsilon_csi():
    """snr_dependent_csi=False 时 sigma_est == epsilon_csi (与 sigma2 无关)。"""
    gen = ChannelGenerator(
        K=8, N_r=8, N_t=2, rho=0.9, csi_mode="imperfect",
        epsilon_csi=0.05, csi_delay=1, seed=0,
        snr_dependent_csi=False,
    )
    gen.set_sigma2(1.0)
    assert abs(gen._current_sigma_est() - 0.05) < 1e-12
    gen.set_sigma2(0.001)
    assert abs(gen._current_sigma_est() - 0.05) < 1e-12


def test_obs_channel_gains_from_H_hat_in_imperfect():
    """imperfect 模式下 channel_gains 应与 |H_hat|² 一致，而非 |H_true|²。"""
    gen = ChannelGenerator(
        K=4, N_r=8, N_t=2, rho=0.9, csi_mode="imperfect",
        epsilon_csi=0.05, csi_delay=1, seed=7,
        snr_dependent_csi=False,
    )
    obs = gen.reset()
    for _ in range(5):
        obs = gen.step()
    expected = np.sum(np.abs(obs["H_hat"]) ** 2, axis=(1, 2))
    assert np.allclose(obs["channel_gains"], expected)
    assert "channel_gains_true" in obs
    expected_true = np.sum(np.abs(obs["H_true"]) ** 2, axis=(1, 2))
    assert np.allclose(obs["channel_gains_true"], expected_true)


def test_obs_channel_gains_equal_in_perfect_mode():
    """perfect CSI: channel_gains 与 channel_gains_true 完全一致。"""
    gen = ChannelGenerator(
        K=4, N_r=8, N_t=2, rho=0.9, csi_mode="perfect", seed=0,
    )
    obs = gen.reset()
    assert np.allclose(obs["channel_gains"], obs["channel_gains_true"])


def test_obs_includes_sigma_est_field():
    """channel_obs 返回字典包含 sigma_est，perfect 模式下为 0。"""
    gen_p = ChannelGenerator(K=4, N_r=8, N_t=2, csi_mode="perfect", seed=0)
    obs_p = gen_p.reset()
    assert obs_p["sigma_est"] == 0.0

    gen_i = ChannelGenerator(
        K=4, N_r=8, N_t=2, csi_mode="imperfect",
        epsilon_csi=0.05, snr_dependent_csi=False, seed=0,
    )
    obs_i = gen_i.reset()
    assert abs(obs_i["sigma_est"] - 0.05) < 1e-12


def test_imperfect_csi_error_scales_with_snr():
    """snr_dependent_csi=True 时，低 SNR 下平均估计误差明显大于高 SNR。"""
    rng_seed = 11
    K, N_r, N_t = 8, 8, 2

    def mean_err_norm(sigma2: float) -> float:
        gen = ChannelGenerator(
            K=K, N_r=N_r, N_t=N_t, rho=0.9, csi_mode="imperfect",
            epsilon_csi=0.05, csi_delay=1, seed=rng_seed,
            snr_dependent_csi=True, pilot_power=1.0, epsilon_floor=0.01,
            sigma2=sigma2,
        )
        gen.reset()
        accum = 0.0
        n = 30
        for _ in range(n):
            obs = gen.step()
            diff = obs["H_true"] - obs["H_hat"]
            accum += float(np.linalg.norm(diff.reshape(K, -1), axis=1).mean())
        return accum / n

    err_low = mean_err_norm(1.0)     # 低 SNR
    err_high = mean_err_norm(1e-4)   # 高 SNR
    assert err_low > err_high * 1.05, (
        f"err(low SNR)={err_low:.4f} should exceed err(high SNR)={err_high:.4f}"
    )
