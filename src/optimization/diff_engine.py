"""可微 RSMA 速率引擎（torch 影子实现）。

严格复刻 src/env/rate_calc.py 的 MMSE-SIC 速率计算与
src/env/beamforming.py 的 MRT 预编码，为逐槽专家求解器提供梯度通道。
数值一致性与容差由 tests/test_diff_engine.py 锁定。

与 numpy 原实现的对应关系（事实依据见
.trellis/tasks/09-02-solver-expert/research/solver-expert-design.md）：

- MRT 方向 = H_k 最大右奇异向量（SVD, Vh[0].conj()）。H 对专家是常量，
  方向在 no_grad 下计算即可与 numpy 精确一致；SVD 相位不唯一，但
  SINR（h^H R^{-1} h）与 outer(h, h^H) 均相位不变，速率不受影响。
- Phase 1（公共流）：R_int[k] = σ²I + robust + Σ_ALL h_p[j]h_p[j]^H
  （自身私有流计入干扰，与逐 j 求和合并后与 k 无关）
  + Σ_{j≠k} M1[j,k]·h_c[j]h_c[j]^H + ε²·Σ_{j≠k} R1[j,k]·p_c[j]·I。
  M1 = β² 若（同组且 rank[j]<rank[k]）否则 1；R1 为该条件的指示。
- Phase 2（私有流）：所有公共流按残余计（β² outer + ε² p_c I，
  h_c_norm<EPS 的用户整体跳过，含自身）；私有流干扰 M2 = β² 若
  rank[j]<rank[k]（无组条件）否则 1，外加 ε²·p_p[j]·I 残余项。
- outage 模式 = min(declared[H_hat, ε], actual[H_true, ε=0])。
"""
from __future__ import annotations

import torch

_EPS = 1e-12  # 与 rate_calc.EPS 一致


class DiffRSMAEngine:
    """可微速率引擎：precoding(H, α, p, split) → W；rates(H, W_c, W_p) → R。"""

    def __init__(self, K: int, N_r: int, N_t: int, sigma2: float,
                 device=None, dtype=torch.float64):
        self.K = int(K)
        self.N_r = int(N_r)
        self.N_t = int(N_t)
        self.sigma2 = float(sigma2)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"unsupported dtype {dtype}")
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    # ------------------------------------------------------------------
    # 预编码（beamforming.compute_precoding_v2 镜像）
    # ------------------------------------------------------------------

    def mrt_directions(self, H) -> torch.Tensor:
        """每用户 MRT 方向 = H_k 最大右奇异向量，(K, N_t) complex，已分离梯度。"""
        H = torch.as_tensor(H, dtype=self.cdtype, device=self.device)
        with torch.no_grad():
            _, _, Vh = torch.linalg.svd(H)
            dirs = Vh[:, 0, :].conj()
            norms = torch.linalg.vector_norm(dirs, dim=-1, keepdim=True)
            fallback = torch.ones_like(dirs) / (self.N_t ** 0.5)
            safe = torch.clamp(norms, min=_EPS)
            dirs = torch.where(norms < _EPS, fallback, dirs / safe)
        return dirs

    def precoding(self, H, alpha, p, split_mask):
        """W_c[k] = dir·√(α·p·split)，W_p[k] = dir·√((1−α·split)·p)。

        alpha/p/split_mask: (K,) 实数张量（p 需要梯度）。
        """
        dirs = self.mrt_directions(H)
        alpha = torch.as_tensor(alpha, dtype=self.dtype, device=self.device)
        p = torch.as_tensor(p, dtype=self.dtype, device=self.device)
        split = torch.as_tensor(split_mask, dtype=self.dtype, device=self.device)
        p_c = alpha * p * split
        p_p = (1.0 - alpha * split) * p
        # 1e-16 下限防 sqrt(0) 反向 NaN（功率为 0 的流，上游梯度为 0×inf）；
        # 前向误差 ≤1e-8（W 幅度），速率误差 ~1e-16，远低于一致性容差
        W_c = dirs * torch.sqrt(torch.clamp(p_c, min=1e-16)).unsqueeze(-1)
        W_p = dirs * torch.sqrt(torch.clamp(p_p, min=1e-16)).unsqueeze(-1)
        return W_c, W_p

    # ------------------------------------------------------------------
    # 速率计算（rate_calc.compute_rates_mmse_sic 镜像，向量化）
    # ------------------------------------------------------------------

    def _resolve_rank(self, decoding_order) -> torch.Tensor:
        """rank[k] = 用户 k 在解码序列中的位置（小 = 先解码）。"""
        if decoding_order is None:
            return torch.arange(self.K, device=self.device)
        order = torch.as_tensor(decoding_order, dtype=torch.float64, device=self.device)
        return torch.argsort(torch.argsort(order)).to(device=self.device)

    def rates(self, H, W_c, W_p, decoding_order=None, group_ids=None,
              error_bound=0.0, sic_imperfection=0.0) -> dict:
        """两阶段 MMSE-SIC 速率（对应 compute_rates_mmse_sic）。

        Args:
            H: (K, N_r, N_t) complex
            W_c/W_p: (K, N_t) complex（功率已嵌入，可带梯度）
            decoding_order: (K,) 优先级（值小先解码）
            group_ids: (K,) 分组
            error_bound: CSI 误差上界 ε（robust 正则）
            sic_imperfection: SIC 残余因子 β
        Returns:
            dict: R_c/R_p/R_total (K,) 张量、R_sum 标量、SINR_c/SINR_p
        """
        K, N_r = self.K, self.N_r
        dev, cd = self.device, self.cdtype
        H = torch.as_tensor(H, dtype=cd, device=dev)
        W_c = torch.as_tensor(W_c, dtype=cd, device=dev)
        W_p = torch.as_tensor(W_p, dtype=cd, device=dev)

        rank = self._resolve_rank(decoding_order)
        if group_ids is not None:
            g = torch.as_tensor(group_ids, device=dev).long()
        else:
            g = torch.zeros(K, dtype=torch.long, device=dev)

        h_c = torch.einsum("kij,kj->ki", H, W_c)
        h_p = torch.einsum("kij,kj->ki", H, W_p)
        p_c = (W_c * W_c.conj()).real.sum(-1)
        p_p = (W_p * W_p.conj()).real.sum(-1)
        h_c_norm = torch.linalg.vector_norm(h_c, dim=-1)
        h_p_norm = torch.linalg.vector_norm(h_p, dim=-1)

        hc_outer = torch.einsum("ka,kb->kab", h_c, h_c.conj())
        hp_outer = torch.einsum("ka,kb->kab", h_p, h_p.conj())

        eye = torch.eye(N_r, dtype=cd, device=dev)
        eye_r = torch.eye(N_r, dtype=self.dtype, device=dev)
        beta = float(sic_imperfection)
        eps = float(error_bound)
        robust = eps * eps * (p_c.sum() + p_p.sum()) * eye_r if eps > 0 else None

        same_group = g.unsqueeze(1) == g.unsqueeze(0)          # [j, k] = g[j]==g[k]
        rj_lt_rk = rank.unsqueeze(1) < rank.unsqueeze(0)       # [j, k] = rank[j] < rank[k]
        not_diag = ~torch.eye(K, dtype=torch.bool, device=dev)

        # ---- Phase 1: 公共流 ----
        # M1 对角线必须为 0：自身公共流是期望信号而非干扰（numpy 的 j==k continue）
        sic1 = same_group & rj_lt_rk & not_diag
        M1 = torch.where(sic1, torch.full_like(sic1, beta * beta, dtype=self.dtype),
                         torch.ones_like(sic1, dtype=self.dtype)) * not_diag.to(self.dtype)
        com1 = torch.einsum("jk,jab->kab", M1.to(cd), hc_outer)
        eps1 = eps * eps * (sic1.to(self.dtype) * p_c.unsqueeze(1)).sum(dim=0)
        R_int1 = self.sigma2 * eye_r
        if robust is not None:
            R_int1 = R_int1 + robust
        R_int1 = R_int1 + hp_outer.sum(dim=0).unsqueeze(0) + com1 \
            + eps1.view(K, 1, 1) * eye_r
        v1 = torch.linalg.solve(R_int1, h_c.unsqueeze(-1))
        SINR_c = torch.einsum("ka,ka->k", h_c.conj(), v1.squeeze(-1)).real
        SINR_c = torch.clamp(SINR_c, min=0.0)
        SINR_c = torch.where(h_c_norm < _EPS, torch.zeros_like(SINR_c), SINR_c)

        # ---- Phase 2: 私有流 ----
        # M2 对角线同样为 0：自身私有流是期望信号
        sic2 = rj_lt_rk & not_diag
        M2 = torch.where(sic2, torch.full_like(sic2, beta * beta, dtype=self.dtype),
                         torch.ones_like(sic2, dtype=self.dtype)) * not_diag.to(self.dtype)
        mask_c = (h_c_norm >= _EPS).to(cd)
        common_res = beta * beta * torch.einsum("j,jab->ab", mask_c, hc_outer) \
            + eps * eps * (mask_c.real * p_c).sum() * eye
        com2 = torch.einsum("jk,jab->kab", M2.to(cd), hp_outer)
        eps2 = eps * eps * (sic2.to(self.dtype) * p_p.unsqueeze(1)).sum(dim=0)
        R_int2 = self.sigma2 * eye_r + common_res.unsqueeze(0) + com2 \
            + eps2.view(K, 1, 1) * eye_r
        if robust is not None:
            R_int2 = R_int2 + robust
        v2 = torch.linalg.solve(R_int2, h_p.unsqueeze(-1))
        SINR_p = torch.einsum("ka,ka->k", h_p.conj(), v2.squeeze(-1)).real
        SINR_p = torch.clamp(SINR_p, min=0.0)
        SINR_p = torch.where(h_p_norm < _EPS, torch.zeros_like(SINR_p), SINR_p)

        R_c = torch.log2(1.0 + SINR_c)
        R_p = torch.log2(1.0 + SINR_p)
        R_total = R_c + R_p
        return {
            "R_c": R_c,
            "R_p": R_p,
            "R_total": R_total,
            "R_sum": R_total.sum(),
            "SINR_c": SINR_c,
            "SINR_p": SINR_p,
        }

    def outage_rates(self, H_hat, H_true, W_c, W_p, error_bound,
                     decoding_order=None, group_ids=None,
                     sic_imperfection=0.0) -> dict:
        """鲁棒速率 = min(declared[H_hat, ε], actual[H_true, ε=0])。"""
        declared = self.rates(H_hat, W_c, W_p, decoding_order=decoding_order,
                              group_ids=group_ids, error_bound=error_bound,
                              sic_imperfection=sic_imperfection)
        actual = self.rates(H_true, W_c, W_p, decoding_order=decoding_order,
                            group_ids=group_ids, error_bound=0.0,
                            sic_imperfection=sic_imperfection)
        R_c = torch.minimum(declared["R_c"], actual["R_c"])
        R_p = torch.minimum(declared["R_p"], actual["R_p"])
        R_total = R_c + R_p
        return {
            "R_c": R_c,
            "R_p": R_p,
            "R_total": R_total,
            "R_sum": R_total.sum(),
            "SINR_c": torch.minimum(declared["SINR_c"], actual["SINR_c"]),
            "SINR_p": torch.minimum(declared["SINR_p"], actual["SINR_p"]),
        }
