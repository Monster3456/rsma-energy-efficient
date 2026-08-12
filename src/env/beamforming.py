"""波束成形模块

实现预编码方向计算：
- RZF（正则化迫零）私有流预编码
- 组内MRT公共流预编码
- [Deprecated] MRT+正交补预编码（向后兼容）
"""
import warnings
import numpy as np

EPS = 1e-12


class BeamformingDesigner:
    """预编码方向计算（规则-based，不由PPO学习）

    新方案：
    - 私有流方向：RZF（正则化迫零）
    - 公共流方向：组内用户信道的MRT方向

    旧方案（deprecated，保留向后兼容）：
    - 私有流方向：MRT（最大比发射），取H_k的主右奇异向量
    - 公共流方向：正交方向，取H_k的第二右奇异向量
    """

    def __init__(self, K: int, N_r: int, N_t: int):
        """
        Args:
            K: 用户数
            N_r: 基站接收天线数
            N_t: 每用户发射天线数
        """
        self.K = K
        self.N_r = N_r
        self.N_t = N_t

    # ================================================================
    # 新方法: RZF 预编码
    # ================================================================

    def compute_rzf_precoding(self, H_hat, sigma2, P_avg):
        """RZF 预编码矩阵

        W = H_hat^H (H_hat @ H_hat^H + alpha_reg * I)^{-1}
        alpha_reg = K * sigma2 / P_avg

        Args:
            H_hat: (N_r, K) 估计信道矩阵 (complex)
            sigma2: 噪声功率
            P_avg: 平均发射功率
        Returns:
            W: (N_r, K) 归一化预编码矩阵 (每列单位范数)
        """
        H_hat = np.asarray(H_hat, dtype=complex)
        N_r, K = H_hat.shape

        # 正则化参数
        alpha_reg = K * sigma2 / max(P_avg, EPS)

        # Gram 矩阵 + 正则化: (N_r, N_r)
        gram = H_hat @ H_hat.conj().T + alpha_reg * np.eye(N_r)

        # 求逆
        gram_inv = np.linalg.inv(gram)

        # RZF 矩阵: (N_r, K)
        W = gram_inv @ H_hat

        # 每列归一化为单位范数
        for k in range(K):
            col_norm = np.linalg.norm(W[:, k])
            if col_norm < EPS:
                # 退化：使用信道方向
                W[:, k] = H_hat[:, k] / max(np.linalg.norm(H_hat[:, k]), EPS)
            else:
                W[:, k] = W[:, k] / col_norm

        return W

    def compute_common_precoding(self, H_hat, group_ids, split_mask):
        """公共流预编码方向 (组内用户信道的MRT方向)

        对每个组 g, 启用分裂的用户的信道求和后归一化：
        w_c^g = sum(h_k for k in group_g and split_k=1) / norm(...)

        Args:
            H_hat: (N_r, K) 估计信道 (effective channel matrix)
            group_ids: (K,) 用户分组 {0,...,G-1}
            split_mask: (K,) 分裂掩码 (1=启用公共流分裂)
        Returns:
            W_c: (N_r, G) 每组的公共流预编码方向 (每列单位范数)
        """
        H_hat = np.asarray(H_hat, dtype=complex)
        group_ids = np.asarray(group_ids, dtype=int)
        split_mask = np.asarray(split_mask, dtype=float)
        N_r, K = H_hat.shape

        G = int(np.max(group_ids)) + 1 if len(group_ids) > 0 else 1
        W_c = np.zeros((N_r, G), dtype=complex)

        for g in range(G):
            # 找到属于该组且启用分裂的用户
            mask = (group_ids == g) & (split_mask > 0.5)
            if not np.any(mask):
                # 无启用用户：使用该组所有用户的信道平均
                mask_all = (group_ids == g)
                if np.any(mask_all):
                    w_g = np.sum(H_hat[:, mask_all], axis=1)
                else:
                    w_g = np.ones(N_r, dtype=complex) / np.sqrt(N_r)
            else:
                # 组内MRT：启用用户信道求和
                w_g = np.sum(H_hat[:, mask], axis=1)

            # 归一化
            norm_wg = np.linalg.norm(w_g)
            if norm_wg < EPS:
                W_c[:, g] = np.ones(N_r, dtype=complex) / np.sqrt(N_r)
            else:
                W_c[:, g] = w_g / norm_wg

        return W_c

    def get_effective_channel(self, H):
        """将 (K, N_r, N_t) 信道矩阵转换为 (N_r, K) 有效信道

        对每个用户取MRT方向后得到有效信道向量。

        Args:
            H: (K, N_r, N_t) complex - 信道矩阵
        Returns:
            H_eff: (N_r, K) complex - 有效信道矩阵 (每列为用户有效信道)
        """
        K = H.shape[0]
        H_eff = np.zeros((self.N_r, K), dtype=complex)
        for k in range(K):
            # MRT方向
            dir_k = self.mrt_direction(H[k])
            # 有效信道 = H_k @ mrt_direction
            H_eff[:, k] = H[k] @ dir_k
        return H_eff

    # ================================================================
    # 新方法: 统一 MRT 发射预编码 (v2)
    # ================================================================

    def compute_precoding_v2(self, H, alpha, p_total, split_mask):
        """统一 MRT 发射预编码（公共/私有流共享 MRT 方向）

        上行 SIC 通过时域分离公共/私有流，无需空间正交。
        MRT 最大化信道增益，接收端 MMSE-SIC 处理干扰抑制。

        Args:
            H: (K, N_r, N_t) complex - 信道矩阵
            alpha: (K,) float ∈[0,1] - 公共/私有功率分割比
            p_total: (K,) float - 每用户总功率
            split_mask: (K,) float - 分裂掩码 (1=启用公共流)

        Returns:
            W_c: (K, N_t) complex - 公共流预编码 (||W_c[k]||² = alpha[k]*p[k]*split[k])
            W_p: (K, N_t) complex - 私有流预编码 (||W_p[k]||² = (1-alpha[k]*split[k])*p[k])
        """
        K = self.K
        alpha = np.asarray(alpha, dtype=float)
        p_total = np.asarray(p_total, dtype=float)
        split_mask = np.asarray(split_mask, dtype=float)

        W_c = np.zeros((K, self.N_t), dtype=complex)
        W_p = np.zeros((K, self.N_t), dtype=complex)

        for k in range(K):
            dir_k = self.mrt_direction(H[k])
            p_c = alpha[k] * p_total[k] * split_mask[k]
            p_p = (1.0 - alpha[k] * split_mask[k]) * p_total[k]
            W_c[k] = dir_k * np.sqrt(max(p_c, 0.0))
            W_p[k] = dir_k * np.sqrt(max(p_p, 0.0))

        return W_c, W_p

    # ================================================================
    # 旧方法 (保留向后兼容)
    # ================================================================

    def mrt_direction(self, H_k):
        """MRT方向：H_k的最大右奇异向量

        Args:
            H_k: (N_r, N_t) complex - 用户k的信道矩阵

        Returns:
            direction: (N_t,) complex, unit norm
        """
        # SVD分解: H_k = U @ diag(S) @ V^H
        # 主右奇异向量 = V的第一列
        _, S, Vh = np.linalg.svd(H_k, full_matrices=False)
        # Vh shape: (min(N_r,N_t), N_t), 第一行对应最大奇异值
        direction = Vh[0].conj()  # V的第一列 = Vh第一行的共轭
        # 确保单位范数
        norm = np.linalg.norm(direction)
        if norm < EPS:
            # 退化情况：返回随机单位向量
            direction = np.ones(self.N_t, dtype=complex) / np.sqrt(self.N_t)
        else:
            direction = direction / norm
        return direction

    def orthogonal_direction(self, H_k):
        """正交方向：H_k的第二右奇异向量
        用于公共流，与私有流MRT方向正交

        Args:
            H_k: (N_r, N_t) complex - 用户k的信道矩阵

        Returns:
            direction: (N_t,) complex, unit norm
        """
        _, S, Vh = np.linalg.svd(H_k, full_matrices=False)

        if Vh.shape[0] < 2 or self.N_t < 2:
            # N_t=1时无法正交，退回MRT方向
            return self.mrt_direction(H_k)

        # 第二右奇异向量
        direction = Vh[1].conj()
        norm = np.linalg.norm(direction)
        if norm < EPS:
            # 退化：生成与MRT正交的随机向量
            v1 = self.mrt_direction(H_k)
            direction = self._random_orthogonal(v1)
        else:
            direction = direction / norm
        return direction

    def _random_orthogonal(self, v):
        """生成与v正交的随机单位向量

        Args:
            v: (N_t,) complex, unit norm

        Returns:
            w: (N_t,) complex, unit norm, orthogonal to v
        """
        # Gram-Schmidt
        N_t = len(v)
        rand_vec = (np.random.randn(N_t) + 1j * np.random.randn(N_t)) / np.sqrt(2)
        # 减去v方向的投影
        proj = np.conj(v) @ rand_vec
        w = rand_vec - proj * v
        norm = np.linalg.norm(w)
        if norm < EPS:
            # 极端退化，重试
            rand_vec = np.zeros(N_t, dtype=complex)
            rand_vec[0] = 1.0
            proj = np.conj(v) @ rand_vec
            w = rand_vec - proj * v
            norm = np.linalg.norm(w)
        return w / (norm + EPS)

    def compute_directions(self, H, alpha, p_total):
        """根据信道和功率分配计算预编码方向 [Deprecated - 保留向后兼容]

        私有流: MRT方向 (主右奇异向量), 功率 = (1-alpha) * p_total
        公共流: 正交方向 (第二右奇异向量), 功率 = alpha * p_total

        预编码向量 = direction * sqrt(power)

        Args:
            H: (K, N_r, N_t) complex - 信道矩阵
            alpha: (K,) float ∈[0,1] - 公共/私有功率分割比
            p_total: (K,) float - 每用户总功率

        Returns:
            W_c: (K, N_t) complex - 公共流预编码 (||W_c[k]||^2 = alpha[k]*p_total[k])
            W_p: (K, N_t) complex - 私有流预编码 (||W_p[k]||^2 = (1-alpha[k])*p_total[k])
        """
        K = self.K
        alpha = np.asarray(alpha, dtype=float)
        p_total = np.asarray(p_total, dtype=float)

        W_c = np.zeros((K, self.N_t), dtype=complex)
        W_p = np.zeros((K, self.N_t), dtype=complex)

        for k in range(K):
            H_k = H[k]  # (N_r, N_t)

            # 私有流: MRT方向
            dir_p = self.mrt_direction(H_k)
            power_p = (1.0 - alpha[k]) * p_total[k]
            W_p[k] = dir_p * np.sqrt(max(power_p, 0.0))

            # 公共流: 正交方向（方案B）
            dir_c = self.orthogonal_direction(H_k)
            power_c = alpha[k] * p_total[k]
            W_c[k] = dir_c * np.sqrt(max(power_c, 0.0))

        return W_c, W_p
