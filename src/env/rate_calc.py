"""速率计算模块

实现上行RSMA的SIC解码速率计算。
- 公共流速率：R_c,k = log2(1 + SINR_c,k)
- 私有流速率：R_p,k = log2(1 + SINR_p,k)
- 总可达速率：R_k = R_c,k + R_p,k

支持可变解码顺序 (decoding_order) 和分组SIC (group_ids)。
"""
import numpy as np

EPS = 1e-12  # 数值稳定性常量


class RSMARateCalculator:
    """上行RSMA速率计算器

    基站通过SIC依次解码各用户的公共流和私有流。
    解码顺序：先解码所有公共流（按decoding_order），再解码所有私有流（按decoding_order）。

    支持:
    - 可变解码顺序 (decoding_order): 值越小越先解码
    - 分组SIC (group_ids): 只有同组内的公共流互相进行SIC
    - SIC不完美性 (sic_imperfection): 已解码信号的残余干扰
    """

    def __init__(self, K: int, N_r: int, N_t: int, sigma2: float):
        """
        Args:
            K: 用户数
            N_r: 基站接收天线数
            N_t: 每用户发射天线数
            sigma2: 噪声功率
        """
        self.K = K
        self.N_r = N_r
        self.N_t = N_t
        self.sigma2 = sigma2

    def _effective_channel(self, H, W):
        """计算有效信道向量 h_eff_k = H_k @ w_k

        Args:
            H: (K, N_r, N_t) complex
            W: (K, N_t) complex

        Returns:
            h_eff: (K, N_r) complex - 各用户有效信道
        """
        K = H.shape[0]
        h_eff = np.zeros((K, self.N_r), dtype=complex)
        for k in range(K):
            h_eff[k] = H[k] @ W[k]  # (N_r,)
        return h_eff

    def _signal_power(self, h_eff):
        """计算信号功率 ||h_eff_k||^2

        Args:
            h_eff: (K, N_r) complex

        Returns:
            power: (K,) float
        """
        return np.sum(np.abs(h_eff) ** 2, axis=1)  # (K,)

    def compute_rates(self, H, W_c, W_p, sic_order=None,
                      sic_imperfection=0.0,
                      decoding_order=None,
                      group_ids=None):
        """计算所有用户的公共流和私有流速率

        使用matched filter接收。对用户k的某个流，接收合并向量为该流有效信道方向。

        SIC解码过程：
        1. 公共流优先级高于私有流（RSMA标准假设）
        2. 在公共流内部：按 decoding_order 从小到大解码
        3. 私有流：公共流全部解码后，按 decoding_order 从小到大解码
        4. 先解码的用户面临更多干扰，但解码后其信号从后续用户干扰中消除

        分组SIC:
        - 只有同组内的公共流互相进行SIC
        - 不同组的公共流视为干扰（不消除）

        Args:
            H: (K, N_r, N_t) complex - 信道矩阵
            W_c: (K, N_t) complex - 公共流预编码向量
            W_p: (K, N_t) complex - 私有流预编码向量
            sic_order: (K,) int - [Deprecated] SIC解码顺序(0-indexed用户索引),
                       None则按0..K-1。当 decoding_order 不为 None 时忽略此参数。
            sic_imperfection: float - SIC不完美因子 ∈[0,1]，0=完美SIC
            decoding_order: (K,) float/int - 解码优先级，值越小=越先被解码。
                           None=使用sic_order参数（向后兼容）
            group_ids: (K,) int - 用户分组 {0,...,G-1}。None=所有用户同组

        Returns: dict {
            'R_c': (K,) - 每用户公共流速率 (bps/Hz)
            'R_p': (K,) - 每用户私有流速率 (bps/Hz)
            'R_total': (K,) - 每用户总速率 R_k = R_{k,c} + R_{k,p}
            'R_sum': float - 系统和速率
            'SINR_c': (K,) - 公共流SINR
            'SINR_p': (K,) - 私有流SINR
        }
        """
        K = self.K

        # --- 确定解码顺序 ---
        if decoding_order is not None:
            # 新接口: decoding_order[k] 值越小越先解码
            decoding_order = np.asarray(decoding_order, dtype=float)
            # 转为排序后的用户索引序列 (从小到大)
            sic_order_resolved = np.argsort(decoding_order).astype(int)
        elif sic_order is not None:
            sic_order_resolved = np.asarray(sic_order, dtype=int)
        else:
            sic_order_resolved = np.arange(K)

        # --- 分组设置 ---
        if group_ids is not None:
            group_ids = np.asarray(group_ids, dtype=int)
        else:
            # 默认所有用户同组
            group_ids = np.zeros(K, dtype=int)

        # 有效信道: h_c[k] = H_k @ W_c[k], h_p[k] = H_k @ W_p[k]
        h_c = self._effective_channel(H, W_c)  # (K, N_r)
        h_p = self._effective_channel(H, W_p)  # (K, N_r)

        # 各流信号功率 ||h_eff||^2
        power_c = self._signal_power(h_c)  # (K,)
        power_p = self._signal_power(h_p)  # (K,)

        # --- 计算公共流SINR ---
        # 对于sic_order_resolved中第i个被解码的用户k:
        # 干扰 = 尚未解码的同组公共流 + 不同组公共流(不消除) + 所有私有流 + 噪声
        # 使用matched filter: r_k = h_c[k] / ||h_c[k]||
        SINR_c = np.zeros(K)
        # 记录已解码用户(用于SIC)
        decoded_common = set()

        for idx, k in enumerate(sic_order_resolved):
            # 信号功率 = ||h_c[k]||^2 (matched filter自身增益)
            sig = power_c[k]

            # Matched filter方向
            norm_hc = np.linalg.norm(h_c[k])
            if norm_hc < EPS:
                SINR_c[k] = 0.0
                decoded_common.add(k)
                continue
            r_k = h_c[k] / norm_hc  # (N_r,) unit vector

            # 公共流干扰计算
            interf_c = 0.0
            k_group = group_ids[k]

            for j in range(K):
                if j == k:
                    continue
                j_group = group_ids[j]

                if j_group == k_group:
                    # 同组: 已SIC解码的流被消除(考虑不完美性)，未解码的是干扰
                    if j in decoded_common:
                        # 已解码: 残余干扰 = sic_imperfection * 原始干扰
                        proj = np.abs(np.conj(r_k) @ h_c[j]) ** 2
                        interf_c += sic_imperfection * proj
                    else:
                        # 未解码: 完整干扰
                        proj = np.abs(np.conj(r_k) @ h_c[j]) ** 2
                        interf_c += proj
                else:
                    # 不同组: 视为干扰（不消除）
                    proj = np.abs(np.conj(r_k) @ h_c[j]) ** 2
                    interf_c += proj

            # 私有流干扰：所有私有流（公共流解码阶段，私有流尚未解码）
            interf_p = 0.0
            for j in range(K):
                proj = np.abs(np.conj(r_k) @ h_p[j]) ** 2
                interf_p += proj

            # 噪声项: sigma^2 * ||r_k||^2 = sigma^2 (r_k是单位向量)
            noise = self.sigma2

            SINR_c[k] = sig / (interf_c + interf_p + noise + EPS)
            decoded_common.add(k)

        # --- 计算私有流SINR ---
        # 公共流已全部消除(考虑SIC不完美性)，按sic_order解码私有流
        SINR_p = np.zeros(K)
        decoded_private = set()

        for idx, k in enumerate(sic_order_resolved):
            sig = power_p[k]

            # Matched filter方向
            norm_hp = np.linalg.norm(h_p[k])
            if norm_hp < EPS:
                SINR_p[k] = 0.0
                decoded_private.add(k)
                continue
            r_k = h_p[k] / norm_hp  # (N_r,)

            # 公共流残余干扰 (SIC不完美)
            interf_common_residual = 0.0
            if sic_imperfection > 0:
                for j in range(K):
                    proj = np.abs(np.conj(r_k) @ h_c[j]) ** 2
                    interf_common_residual += sic_imperfection * proj

            # 私有流干扰
            interf_p = 0.0
            for j in range(K):
                if j == k:
                    continue
                if j in decoded_private:
                    # 已解码: 残余干扰
                    proj = np.abs(np.conj(r_k) @ h_p[j]) ** 2
                    interf_p += sic_imperfection * proj
                else:
                    # 未解码: 完整干扰
                    proj = np.abs(np.conj(r_k) @ h_p[j]) ** 2
                    interf_p += proj

            noise = self.sigma2
            SINR_p[k] = sig / (interf_common_residual + interf_p + noise + EPS)
            decoded_private.add(k)

        # 速率计算
        R_c = np.log2(1.0 + SINR_c)
        R_p = np.log2(1.0 + SINR_p)
        R_total = R_c + R_p
        R_sum = float(np.sum(R_total))

        return {
            'R_c': R_c,
            'R_p': R_p,
            'R_total': R_total,
            'R_sum': R_sum,
            'SINR_c': SINR_c,
            'SINR_p': SINR_p,
        }

    # ================================================================
    # MMSE-SIC 速率计算（替代 matched filter）
    # ================================================================

    def compute_rates_mmse_sic(self, H, W_c, W_p, sic_order=None,
                                sic_imperfection=0.0,
                                decoding_order=None,
                                group_ids=None,
                                error_bound=0.0):
        """MMSE-SIC 速率计算（每步 SIC 重算干扰协方差）

        Phase 1: 按 decoding_order 解码公共流（组内 SIC）
        Phase 2: 按 decoding_order 解码私有流（公共流已全部 SIC）

        信号模型: y = Σ_k (h_c_k s_k^c + h_p_k s_k^p) + n
        其中 h_c_k = H_k @ W_c[k]（功率已嵌入预编码向量）

        SINR_k = h_k^H R_int^{-1} h_k
        R_int = Σ_{干扰流 j} h_j h_j^H + SIC残余 + σ²I

        Args:
            H: (K, N_r, N_t) complex - 信道矩阵
            W_c: (K, N_t) complex - 公共流预编码向量（含功率）
            W_p: (K, N_t) complex - 私有流预编码向量（含功率）
            sic_order: (K,) int - [Deprecated] 兼容旧接口
            sic_imperfection: float - SIC 硬件不完美因子 β ∈[0,1]
            decoding_order: (K,) float/int - 解码优先级（值越小越先解码）
            group_ids: (K,) int - 用户分组 {0,...,G-1}
            error_bound: float - CSI 误差上界 ε（robust MMSE 正则化）

        Returns: dict {
            'R_c': (K,), 'R_p': (K,), 'R_total': (K,), 'R_sum': float,
            'SINR_c': (K,), 'SINR_p': (K,)
        }
        """
        K = self.K
        N_r = self.N_r

        # --- 确定解码顺序 ---
        if decoding_order is not None:
            decoding_order = np.asarray(decoding_order, dtype=float)
            sic_order_resolved = np.argsort(decoding_order).astype(int)
        elif sic_order is not None:
            sic_order_resolved = np.asarray(sic_order, dtype=int)
        else:
            sic_order_resolved = np.arange(K)

        # --- 分组设置 ---
        if group_ids is not None:
            group_ids = np.asarray(group_ids, dtype=int)
        else:
            group_ids = np.zeros(K, dtype=int)

        # --- 计算有效信道 h_k = H_k @ W_k（功率已嵌入）---
        h_c = np.zeros((K, N_r), dtype=complex)
        h_p = np.zeros((K, N_r), dtype=complex)
        p_c = np.zeros(K)  # 发射功率 ||W_c[k]||²（仅用于 CSI 误差正则化）
        p_p = np.zeros(K)

        for k in range(K):
            h_c[k] = H[k] @ W_c[k]
            h_p[k] = H[k] @ W_p[k]
            p_c[k] = np.real(np.vdot(W_c[k], W_c[k]))
            p_p[k] = np.real(np.vdot(W_p[k], W_p[k]))

        h_c_norm = np.array([np.linalg.norm(h_c[k]) for k in range(K)])
        h_p_norm = np.array([np.linalg.norm(h_p[k]) for k in range(K)])

        # Robust MMSE 正则化项（CSI 误差引起的等效各向同性干扰）
        # Σ_k ε² ||W_k||² I = ε² (Σ p_c + Σ p_p) I
        robust_reg = np.zeros((N_r, N_r), dtype=complex)
        if error_bound > 0:
            total_tx_power = float(np.sum(p_c) + np.sum(p_p))
            robust_reg = error_bound ** 2 * total_tx_power * np.eye(N_r, dtype=complex)

        # --- Phase 1: 公共流 MMSE-SIC ---
        SINR_c = np.zeros(K)
        decoded_common = set()

        for k in sic_order_resolved:
            if h_c_norm[k] < EPS:
                SINR_c[k] = 0.0
                decoded_common.add(k)
                continue

            k_group = group_ids[k]
            # 基础：噪声 + robust 正则化
            R_int = self.sigma2 * np.eye(N_r, dtype=complex) + robust_reg

            # 自身私有流是干扰
            R_int += np.outer(h_p[k], h_p[k].conj())

            for j in range(K):
                if j == k:
                    continue
                j_group = group_ids[j]

                # 公共流干扰
                if j_group == k_group:
                    if j in decoded_common:
                        # 同组已 SIC：硬件残余 β² h h^H + CSI 误差残余 ε² p I
                        if sic_imperfection > 0:
                            R_int += sic_imperfection ** 2 * np.outer(h_c[j], h_c[j].conj())
                        if error_bound > 0:
                            R_int += error_bound ** 2 * p_c[j] * np.eye(N_r, dtype=complex)
                    else:
                        # 同组未解码：完整干扰
                        R_int += np.outer(h_c[j], h_c[j].conj())
                else:
                    # 不同组：始终为完整干扰（不做 SIC）
                    R_int += np.outer(h_c[j], h_c[j].conj())

                # 所有其他用户私有流都是干扰
                R_int += np.outer(h_p[j], h_p[j].conj())

            # SINR = h_k^H R_int^{-1} h_k
            try:
                v = np.linalg.solve(R_int, h_c[k])
                SINR_c[k] = float(np.real(np.vdot(h_c[k], v)))
            except np.linalg.LinAlgError:
                SINR_c[k] = 0.0

            SINR_c[k] = max(SINR_c[k], 0.0)
            decoded_common.add(k)

        # --- Phase 2: 私有流 MMSE-SIC ---
        SINR_p = np.zeros(K)
        decoded_private = set()

        for k in sic_order_resolved:
            if h_p_norm[k] < EPS:
                SINR_p[k] = 0.0
                decoded_private.add(k)
                continue

            # 基础：噪声 + robust 正则化
            R_int = self.sigma2 * np.eye(N_r, dtype=complex) + robust_reg

            # 所有公共流已 SIC（残余）
            for j in range(K):
                if h_c_norm[j] < EPS:
                    continue
                if sic_imperfection > 0:
                    R_int += sic_imperfection ** 2 * np.outer(h_c[j], h_c[j].conj())
                if error_bound > 0:
                    R_int += error_bound ** 2 * p_c[j] * np.eye(N_r, dtype=complex)

            # 私有流干扰
            for j in range(K):
                if j == k:
                    continue
                if j in decoded_private:
                    # 已 SIC：残余
                    if sic_imperfection > 0:
                        R_int += sic_imperfection ** 2 * np.outer(h_p[j], h_p[j].conj())
                    if error_bound > 0:
                        R_int += error_bound ** 2 * p_p[j] * np.eye(N_r, dtype=complex)
                else:
                    # 未解码：完整干扰
                    R_int += np.outer(h_p[j], h_p[j].conj())

            # SINR = h_k^H R_int^{-1} h_k
            try:
                v = np.linalg.solve(R_int, h_p[k])
                SINR_p[k] = float(np.real(np.vdot(h_p[k], v)))
            except np.linalg.LinAlgError:
                SINR_p[k] = 0.0

            SINR_p[k] = max(SINR_p[k], 0.0)
            decoded_private.add(k)

        # 速率计算
        R_c = np.log2(1.0 + SINR_c)
        R_p = np.log2(1.0 + SINR_p)
        R_total = R_c + R_p
        R_sum = float(np.sum(R_total))

        return {
            'R_c': R_c,
            'R_p': R_p,
            'R_total': R_total,
            'R_sum': R_sum,
            'SINR_c': SINR_c,
            'SINR_p': SINR_p,
        }

    def compute_outage_rates_mmse_sic(self, H_hat, H_true, W_c, W_p,
                                       error_bound,
                                       sic_order=None,
                                       sic_imperfection=0.0,
                                       decoding_order=None,
                                       group_ids=None):
        """鲁棒 MMSE-SIC（imperfect CSI 模式）

        1. 用 H_hat + robust 正则化计算 declared rate（保守速率）
        2. 用 H_true 计算 actual rate（真实可达）
        3. outage: actual < declared 的用户标记为中断，使用 actual rate

        Args:
            H_hat: (K, N_r, N_t) complex - 估计信道
            H_true: (K, N_r, N_t) complex - 真实信道
            W_c: (K, N_t) complex - 公共流预编码
            W_p: (K, N_t) complex - 私有流预编码
            error_bound: float - CSI 误差上界 ε
            sic_order: (K,) int - [Deprecated]
            sic_imperfection: float - SIC 不完美因子 β
            decoding_order: (K,) - 解码优先级
            group_ids: (K,) - 用户分组

        Returns: dict (同 compute_rates_mmse_sic，速率为鲁棒值)
        """
        # Declared rate: 基于估计信道 + robust 正则化
        declared = self.compute_rates_mmse_sic(
            H_hat, W_c, W_p,
            sic_order=sic_order,
            sic_imperfection=sic_imperfection,
            decoding_order=decoding_order,
            group_ids=group_ids,
            error_bound=error_bound,
        )

        # Actual rate: 基于真实信道，无正则化
        actual = self.compute_rates_mmse_sic(
            H_true, W_c, W_p,
            sic_order=sic_order,
            sic_imperfection=sic_imperfection,
            decoding_order=decoding_order,
            group_ids=group_ids,
            error_bound=0.0,
        )

        # 使用 min(declared, actual) 作为可用速率
        R_c = np.minimum(declared['R_c'], actual['R_c'])
        R_p = np.minimum(declared['R_p'], actual['R_p'])
        R_total = R_c + R_p
        R_sum = float(np.sum(R_total))

        return {
            'R_c': R_c,
            'R_p': R_p,
            'R_total': R_total,
            'R_sum': R_sum,
            'SINR_c': np.minimum(declared['SINR_c'], actual['SINR_c']),
            'SINR_p': np.minimum(declared['SINR_p'], actual['SINR_p']),
        }

    # ================================================================
    # 旧方法 (matched filter, 保留向后兼容)
    # ================================================================

    def compute_outage_rates(self, H_hat, W_c, W_p, error_variance,
                             outage_prob=0.05, sic_order=None,
                             sic_imperfection=0.0,
                             decoding_order=None,
                             group_ids=None):
        """鲁棒模式：计算考虑CSI误差的中断速率

        使用worst-case保守估计：
        SINR_robust = SINR_nominal / (1 + SINR_nominal * error_factor)
        error_factor = error_variance[k] * N_r

        Args:
            H_hat: (K, N_r, N_t) complex - 估计信道矩阵
            W_c: (K, N_t) complex - 公共流预编码向量
            W_p: (K, N_t) complex - 私有流预编码向量
            error_variance: (K,) float - 每用户CSI误差方差
            outage_prob: float - 中断概率（保留参数）
            sic_order: (K,) int - SIC解码顺序
            sic_imperfection: float - SIC不完美因子
            decoding_order: (K,) - 解码优先级 (可选)
            group_ids: (K,) - 用户分组 (可选)

        Returns: dict (同compute_rates，但速率为鲁棒中断速率)
        """
        # 先计算标称速率
        nominal = self.compute_rates(
            H_hat, W_c, W_p, sic_order=sic_order,
            sic_imperfection=sic_imperfection,
            decoding_order=decoding_order,
            group_ids=group_ids,
        )

        SINR_c_nom = nominal['SINR_c']
        SINR_p_nom = nominal['SINR_p']

        # 误差因子
        error_variance = np.asarray(error_variance, dtype=float)
        error_factor = error_variance * self.N_r  # (K,)

        # 鲁棒SINR：保守估计
        SINR_c_robust = SINR_c_nom / (1.0 + SINR_c_nom * error_factor + EPS)
        SINR_p_robust = SINR_p_nom / (1.0 + SINR_p_nom * error_factor + EPS)

        # 确保非负
        SINR_c_robust = np.maximum(SINR_c_robust, 0.0)
        SINR_p_robust = np.maximum(SINR_p_robust, 0.0)

        R_c = np.log2(1.0 + SINR_c_robust)
        R_p = np.log2(1.0 + SINR_p_robust)
        R_total = R_c + R_p
        R_sum = float(np.sum(R_total))

        return {
            'R_c': R_c,
            'R_p': R_p,
            'R_total': R_total,
            'R_sum': R_sum,
            'SINR_c': SINR_c_robust,
            'SINR_p': SINR_p_robust,
        }
