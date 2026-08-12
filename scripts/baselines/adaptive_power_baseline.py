"""功率自适应基线（Target-Rate Power Control）

目的：与 PPO-RSMA 进行更公平的对比 —— 排除"会调功率"这一变量，
凸显 RSMA 分裂/分组/SIC 顺序优化的独立增益。

策略：
- alpha_k = 0  (纯 SDMA, 不做 RSMA 分裂)
- d_ratio_k = 1.0 (尽量发)
- p_k：每用户用最小功率达到目标速率 = current_lambda_rate / tau (bps/Hz)
  使用 MMSE-SIC 速率计算，按信道增益降序的标准 SIC 顺序解码。
  采用 Foschini-Miljanic 风格的迭代功率控制（坐标下降，5 次迭代）。

与现有基线（SDMA/NOMA/EqualPower）相同接口：select_action(obs, env) -> (3K,) action.

Note: 通过设置 env.decoding_order 控制 SIC 顺序（env.step 在 3K 动作下保留此设置）。
"""
import numpy as np

from src.config import Config

EPS = 1e-12


class AdaptivePowerBaseline:
    """Target-Rate 功率自适应基线（纯 SDMA + MMSE-SIC + 最小功率达目标速率）

    每个用户独立通过迭代功率控制找到最小 p_k，使 MMSE-SIC 解码后的
    SINR 达到 gamma_target = 2^(r_target) - 1，其中 r_target = lambda / tau.

    关键差异 vs 其他基线：
    - SDMA/NOMA/EqualPower: 固定功率策略，无视实际负载
    - AdaptivePower:        根据 lambda(SNR) 自适应调整功率，"刚好够用"
    """

    def __init__(self, config: Config, n_iter: int = 5, tol: float = 1e-4):
        self.config = config
        self.K = config.system.K
        self.N_r = config.system.N_r
        self.N_t = config.system.N_t
        self.P_k_max = config.system.P_k_max
        self.tau = float(config.system.tau)
        self.n_iter = int(n_iter)
        self.tol = float(tol)

    # ------------------------------------------------------------------ #
    # 内部工具：MRT 方向 + 有效信道
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mrt_direction(H_k: np.ndarray) -> np.ndarray:
        """主右奇异向量（与 BeamformingDesigner.mrt_direction 一致）。"""
        _, _, Vh = np.linalg.svd(H_k, full_matrices=False)
        d = Vh[0].conj()
        n = np.linalg.norm(d)
        if n < EPS:
            d = np.ones(H_k.shape[1], dtype=complex) / np.sqrt(H_k.shape[1])
        else:
            d = d / n
        return d

    def _effective_channels(self, H: np.ndarray) -> np.ndarray:
        """计算单位功率有效信道 g_k = H_k @ mrt_dir(H_k), shape (K, N_r)."""
        K = self.K
        g = np.zeros((K, self.N_r), dtype=complex)
        for k in range(K):
            d = self._mrt_direction(H[k])
            g[k] = H[k] @ d
        return g

    # ------------------------------------------------------------------ #
    # 核心：迭代功率控制
    # ------------------------------------------------------------------ #
    def _solve_min_power(
        self,
        g: np.ndarray,
        sic_order: np.ndarray,
        gamma_target: float,
        sigma2: float,
    ) -> np.ndarray:
        """坐标下降 Foschini-Miljanic：

        对每个用户 k（按 sic_order 解码顺序）：
            R_int_k = sigma2 * I + Σ_{j 未解码, j≠k} p_j g_j g_j^H
            unit_sinr_k = g_k^H R_int_k^{-1} g_k
            p_k = clip(gamma_target / unit_sinr_k, 0, P_k_max)

        外层迭代直到 p 收敛或达到 n_iter。
        """
        K = self.K
        N_r = self.N_r
        I = np.eye(N_r, dtype=complex)

        p = np.full(K, 0.5 * self.P_k_max, dtype=np.float64)

        for _ in range(self.n_iter):
            new_p = p.copy()
            decoded = set()
            for k in sic_order:
                R = sigma2 * I.copy()
                for j in range(K):
                    if j == k or j in decoded:
                        continue
                    R = R + new_p[j] * np.outer(g[j], g[j].conj())
                try:
                    v = np.linalg.solve(R, g[k])
                    unit_sinr = float(np.real(np.vdot(g[k], v)))
                except np.linalg.LinAlgError:
                    unit_sinr = 0.0

                if unit_sinr <= EPS:
                    new_p[k] = self.P_k_max
                else:
                    p_req = gamma_target / unit_sinr
                    new_p[k] = float(np.clip(p_req, 0.0, self.P_k_max))
                decoded.add(int(k))

            if np.max(np.abs(new_p - p)) < self.tol * self.P_k_max:
                p = new_p
                break
            p = new_p

        return p

    # ------------------------------------------------------------------ #
    # 接口
    # ------------------------------------------------------------------ #
    def select_action(self, obs: np.ndarray, env) -> np.ndarray:
        """返回 3K 维动作 (alpha=0, p=自适应, d_ratio=1)，并设置 env.decoding_order.

        Returns: action (3K,) in [-1, 1] range.
        """
        K = self.K

        # 目标速率：刚好消化平均到达 (bps/Hz)
        lam = float(getattr(env, 'current_lambda_rate', env.config.system.lambda_rate))
        target_rate = lam / max(self.tau, EPS)
        gamma_target = max(2.0 ** target_rate - 1.0, 1e-6)

        H_hat = env._channel_obs['H_hat']                # (K, N_r, N_t)
        sigma2 = float(env.sigma2)

        # 单位功率有效信道
        g = self._effective_channels(H_hat)

        # 标准 SIC 顺序：信道增益降序（强信号先解码）
        gains = env._channel_obs.get('channel_gains',
                                     np.sum(np.abs(H_hat) ** 2, axis=(1, 2)))
        sic_order = np.argsort(-gains).astype(int)

        # decoding_order 数组（rate_calc 内部 argsort 取小值优先）
        priorities = np.zeros(K, dtype=np.int64)
        for rank, k in enumerate(sic_order):
            priorities[int(k)] = rank
        env.decoding_order = priorities

        # 迭代求解最小功率
        p = self._solve_min_power(g, sic_order, gamma_target, sigma2)

        # --- 构造动作 ---
        # alpha = 0: a = -1
        alpha_action = np.full(K, -1.0, dtype=np.float64)
        # p_total: ratio = p / P_max, action = 2*ratio - 1
        p_ratio = np.clip(p / max(self.P_k_max, EPS), 0.0, 1.0)
        p_action = 2.0 * p_ratio - 1.0
        # d_ratio = 1: a = 1
        d_action = np.full(K, 1.0, dtype=np.float64)

        action = np.concatenate([alpha_action, p_action, d_action])
        return action.astype(np.float32)

    # ------------------------------------------------------------------ #
    # 简单 evaluate（与其他基线接口一致，便于独立测试）
    # ------------------------------------------------------------------ #
    def evaluate(self, env, n_episodes: int = 100) -> dict:
        powers, queues, deliveries = [], [], []
        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            ep_power, ep_queue = 0.0, 0.0
            steps = 0
            for _ in range(env.T):
                action = self.select_action(obs, env)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_power += info['total_power']
                ep_queue += info['mean_queue']
                steps += 1
                if terminated or truncated:
                    break
            steps = max(steps, 1)
            powers.append(ep_power / steps)
            queues.append(ep_queue / steps)
            deliveries.append(info['delivery_ratio'])

        mean_queue = float(np.mean(queues))
        return {
            'mean_power': float(np.mean(powers)),
            'mean_queue': mean_queue,
            'delivery_ratio': float(np.mean(deliveries)),
            'queue_stable': mean_queue < env.Q_max * 0.8,
        }


# ============================================================
# 可选：Water-Filling 基线（基于 MMSE-SIC 等效信道增益）
# ============================================================

class WaterFillingBaseline(AdaptivePowerBaseline):
    """注水功率分配基线（在总功率约束下最大化加权和速率）

    对每个用户计算 MMSE-SIC 等效单位信道增益 lambda_k = g_k^H R_int_{-k}^{-1} g_k,
    然后注水：p_k = max(0, mu - 1/lambda_k), mu 由 sum(p_k) = P_budget 二分确定.

    总功率预算 P_budget = K * P_k_max（与其他基线一致）。
    """

    def __init__(self, config: Config, n_iter: int = 5):
        super().__init__(config, n_iter=n_iter)
        self.P_budget = self.K * self.P_k_max

    def _solve_water_filling(
        self,
        g: np.ndarray,
        sic_order: np.ndarray,
        sigma2: float,
    ) -> np.ndarray:
        """迭代注水：交替 (1) 给定 p 计算 unit_sinr_k；(2) 注水重新分配 p."""
        K = self.K
        N_r = self.N_r
        I = np.eye(N_r, dtype=complex)

        p = np.full(K, self.P_budget / max(K, 1), dtype=np.float64)
        for _ in range(self.n_iter):
            # Step 1: 计算每用户单位 SINR
            unit_sinr = np.zeros(K, dtype=np.float64)
            decoded = set()
            for k in sic_order:
                R = sigma2 * I.copy()
                for j in range(K):
                    if j == k or j in decoded:
                        continue
                    R = R + p[j] * np.outer(g[j], g[j].conj())
                try:
                    v = np.linalg.solve(R, g[k])
                    unit_sinr[int(k)] = max(float(np.real(np.vdot(g[k], v))), EPS)
                except np.linalg.LinAlgError:
                    unit_sinr[int(k)] = EPS
                decoded.add(int(k))

            # Step 2: 注水: p_k = max(0, mu - 1/lambda_k), 受每用户上限 P_k_max
            inv_lambda = 1.0 / np.maximum(unit_sinr, EPS)

            # 二分搜索 mu 使得 sum(clip(mu - inv_lambda, 0, P_k_max)) == P_budget
            mu_lo = 0.0
            mu_hi = float(np.max(inv_lambda) + self.P_k_max + self.P_budget)
            for _ in range(40):
                mu = 0.5 * (mu_lo + mu_hi)
                p_try = np.clip(mu - inv_lambda, 0.0, self.P_k_max)
                if p_try.sum() > self.P_budget:
                    mu_hi = mu
                else:
                    mu_lo = mu
                if mu_hi - mu_lo < 1e-6:
                    break
            new_p = np.clip(0.5 * (mu_lo + mu_hi) - inv_lambda, 0.0, self.P_k_max)

            if np.max(np.abs(new_p - p)) < self.tol * self.P_k_max:
                p = new_p
                break
            p = new_p

        return p

    def select_action(self, obs: np.ndarray, env) -> np.ndarray:
        K = self.K
        H_hat = env._channel_obs['H_hat']
        sigma2 = float(env.sigma2)

        g = self._effective_channels(H_hat)
        gains = env._channel_obs.get('channel_gains',
                                     np.sum(np.abs(H_hat) ** 2, axis=(1, 2)))
        sic_order = np.argsort(-gains).astype(int)

        priorities = np.zeros(K, dtype=np.int64)
        for rank, k in enumerate(sic_order):
            priorities[int(k)] = rank
        env.decoding_order = priorities

        p = self._solve_water_filling(g, sic_order, sigma2)

        alpha_action = np.full(K, -1.0, dtype=np.float64)
        p_ratio = np.clip(p / max(self.P_k_max, EPS), 0.0, 1.0)
        p_action = 2.0 * p_ratio - 1.0
        d_action = np.full(K, 1.0, dtype=np.float64)

        action = np.concatenate([alpha_action, p_action, d_action])
        return action.astype(np.float32)
