"""求解器专家：单槽 DGLCR surrogate 的在线梯度精修（expert-iteration 教师）。

长期优化问题的单槽分解：在每个时隙，以冻结对偶变量（Dinkelbach q*、
Y/X 虚拟队列、DPP 结构）求解 lyapunov.compute_reward DGLCR 活跃分支的
逐槽最大化问题，作为强教师策略产生演示轨迹（BC 预训练 → PPO 微调）。

决策变量：
  - p_logits (K 维)：softmax 功率分配（env use_softmax_power_share 路径下
    Worker 唯一的活功率自由度；logit 展宽经 clip(ratio,0,1) 可压总功率）
  - decoding_order：启发式候选集（gain_desc / queue_last），按 J 取最优
固定：α=split_mask（group_level_alpha=True 覆盖 Worker α 块，该块为死维）、
d_ratio=D_FRAC（奖励对 D 单调不减，见 tests/test_solver_expert.py 数值验证）

Surrogate 精确镜像 lyapunov.py DGLCR 分支（use_dinkelbach=True 强制
ee_gate=penalty_gate=1，tp_boost=1），包括两处「先更新后使用」的隐藏耦合：
  - q* EMA 更新（_dkl_served_ema/_dkl_power_ema，decay=0.99）先于
    dkl_value = served − q'·P_eff 计算 —— q' 是本步 (D, P_eff) 的函数
  - Y 虚拟队列更新（Y' = clip(Y + dvq_target·ΣA − ΣD, 0, y_max)）先于
    y_component = w_y·(Y'/y_scale)·ΣD 计算 —— Y' 依赖 ΣD
溢出惩罚以期望到达 λ 替代 Poisson 随机 A（同阈值同方向的次梯度代理）。

所有常数从 env.lyapunov 控制器实例运行时读取（天然对齐 config 覆盖）；
Y/X/EMA 只读，专家不触碰训练状态机。
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .diff_engine import DiffRSMAEngine

ACT_BOUND = 0.999          # 动作发射与 surrogate 内 clamp 的统一上界（与 bc_pretrain 一致）
D_FRAC = (ACT_BOUND + 1.0) / 2.0  # d 块发射 ACT_BOUND 时 env 实际执行的 d_ratio


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """priority[k] = k 在降序排名中的位置（值小 = 先解码）。"""
    return np.argsort(np.argsort(-np.asarray(values, dtype=np.float64)))


def _rank_asc(values: np.ndarray) -> np.ndarray:
    """priority[k] = k 在升序排名中的位置（值大 = 后解码，SIC 干扰最小）。"""
    return np.argsort(np.argsort(np.asarray(values, dtype=np.float64)))


class SolverExpert:
    """单槽 surrogate 梯度精修专家。

    用法（expert iteration 收集循环内，每时隙一次）::

        expert = SolverExpert(env)
        result = expert.solve(env)          # 优化 p_logits + 选 order
        obs, r, _, _, info = env.step(result["action"])   # 4K 动作
        # BC 目标: (obs, result["cont_action"], result["order"])
    """

    def __init__(self, env, n_steps: int = 25, lr: float = 0.1,
                 order_candidates: Sequence[str] = ("gain_desc", "queue_last"),
                 dtype: torch.dtype = torch.float64,
                 use_true_channel: bool = True):
        """``use_true_channel=False`` 使 surrogate 仅用估计信道 Ĥ（declared 速率），
        得到**因果**逐槽优化基线；默认 True 时 surrogate 的 outage 速率含真信道
        H_true 项（oracle 上界，用于信息-gap 分析，不可作为因果对照）。"""
        self.env_ref = env
        self.K = int(env.K)
        self.n_steps = int(n_steps)
        self.lr = float(lr)
        self.order_candidates = tuple(order_candidates)
        self.dtype = dtype
        self.use_true_channel = bool(use_true_channel)
        self.engine = DiffRSMAEngine(
            K=env.K, N_r=env.N_r, N_t=env.N_t,
            sigma2=env.rate_calc.sigma2, dtype=dtype,
        )

    # ------------------------------------------------------------------
    # 状态读取（每次 solve/evaluate 重新读，只读不写）
    # ------------------------------------------------------------------

    def _read_state(self, env) -> Dict:
        lya = env.lyapunov
        K = self.K
        cfg_sys = env.config.system

        if not (bool(lya.enable_delivery_vq) and bool(lya.use_dinkelbach)
                and not bool(lya.dgmw_enable)):
            raise ValueError(
                "SolverExpert 仅镜像 DGLCR+dinkelbach 活跃分支 "
                f"(delivery_vq={lya.enable_delivery_vq}, "
                f"dinkelbach={lya.use_dinkelbach}, dgmw={lya.dgmw_enable})"
            )
        if bool(lya.dinkelbach_outer_update) or float(lya.reward_v2_fixed_q) > 0.0:
            raise ValueError("SolverExpert 不支持 outer_update / fixed_q 变体")

        sigma2 = float(env.rate_calc.sigma2)
        self.engine.sigma2 = sigma2

        Q = np.asarray(env.traffic.queue_lengths, dtype=np.float64).copy()
        lam = np.asarray(env.traffic.lambda_per_user, dtype=np.float64).copy() \
            * float(getattr(env.traffic, "packet_size", 1.0))
        gains = np.asarray(
            env._channel_obs.get("channel_gains", np.ones(K)), dtype=np.float64
        )

        orders: Dict[str, np.ndarray] = {}
        for name in self.order_candidates:
            if name == "gain_desc":
                orders[name] = _rank_desc(gains)      # 强用户先解码（经典 SIC）
            elif name == "queue_last":
                orders[name] = _rank_asc(Q)           # 大队列后解码（SIC 干扰最小）
            else:
                raise ValueError(f"unknown order candidate {name!r}")

        if env.enable_hetero_qos:
            drift_w = np.asarray(env.qos_user_drift_weights, dtype=np.float64)
            ov_w = np.asarray(env.qos_user_overflow_weights, dtype=np.float64)
        else:
            drift_w = np.ones(K)
            ov_w = np.ones(K)

        return {
            "Q": torch.as_tensor(Q, dtype=self.dtype),
            "lam": torch.as_tensor(lam, dtype=self.dtype),
            "drift_w": torch.as_tensor(drift_w, dtype=self.dtype),
            "ov_w": torch.as_tensor(ov_w, dtype=self.dtype),
            "H_hat": torch.as_tensor(np.asarray(env._channel_obs["H_hat"]),
                                     dtype=self.engine.cdtype),
            "H_true": torch.as_tensor(np.asarray(env._channel_obs["H_true"]),
                                      dtype=self.engine.cdtype)
            if self.use_true_channel
            else torch.as_tensor(np.asarray(env._channel_obs["H_hat"]),
                                 dtype=self.engine.cdtype),
            "eps": float(env._channel_obs.get("sigma_est", float(cfg_sys.epsilon_csi))),
            "split": torch.as_tensor(
                np.clip(np.asarray(env.split_mask, dtype=np.float64), 0.0, 1.0),
                dtype=self.dtype),
            "group_ids": np.asarray(env.group_ids, dtype=np.int64),
            "orders": orders,
            "sic_imp": float(getattr(cfg_sys, "sic_imperfection", 0.0)),
            "tau": float(env.tau),
            "P_k_max": float(env.P_k_max),
            # lyapunov 常数（运行时真值）
            "V": float(lya.lyapunov_V),
            "ee_w": float(lya.ee_weight),
            "pa": float(lya.pa_efficiency),
            "pc": float(lya.circuit_power),
            "p_floor": float(lya.ee_power_floor),
            "ema_s": float(lya._dkl_served_ema),
            "ema_p": float(lya._dkl_power_ema),
            "dkl_decay": float(lya.dinkelbach_decay),
            "q_min": float(lya.dinkelbach_q_min),
            "q_max": float(lya.dinkelbach_q_max),
            "w_tp": float(lya.w_tp_dglcr),
            "w_y": float(lya.w_y_tp),
            "y_scale": float(lya.dvq_y_scale),
            "y_max": float(lya.dvq_y_max),
            "dvq_target": float(lya.dvq_target),
            "Y_t": float(lya._Y),
            "w_ov": float(lya.w_overflow),
            "enable_pvq": bool(lya.enable_power_vq),
            "w_pvq": float(lya.w_power_vq),
            "pvq_scale": float(lya.pvq_scale),
            "X_t": float(lya._X),
            "ldptrlq_q_w": float(lya.ldptrlq_q_weight),
            "Q_max": float(lya.Q_max),
            "w_early": float(lya.w_early_overflow),
            "warn_ratio": float(lya.overflow_warn_ratio),
            "reward_scale": float(lya.reward_scale),
            "v2": bool(lya.reward_v2_per_user_weight),
            "v2_lo": float(lya.reward_v2_weight_lo),
            "v2_hi": float(lya.reward_v2_weight_hi),
            "delivery_running": float(env.traffic.get_delivery_ratio()),
        }

    # ------------------------------------------------------------------
    # surrogate 镜像（可微；l 为功率 logits 叶子，order 为常量优先级）
    # ------------------------------------------------------------------

    def _mirror(self, st: Dict, l: torch.Tensor, order: np.ndarray,
                d_frac: Optional[float] = None) -> Dict:
        K = self.K
        d_frac = D_FRAC if d_frac is None else float(d_frac)

        # --- env 动作语义链：clip → softmax 份额 → clip ---
        l = torch.clamp(l, -ACT_BOUND, ACT_BOUND)
        e = torch.exp(l - torch.max(l))
        ratio = K * e / (torch.sum(e) + 1e-12)
        p = torch.clamp(ratio, 0.0, 1.0) * st["P_k_max"]

        # --- 物理链：MRT 预编码 + outage 速率（engine 已锁定与 numpy 一致）---
        W_c, W_p = self.engine.precoding(st["H_hat"], st["split"], p, st["split"])
        rr = self.engine.outage_rates(
            st["H_hat"], st["H_true"], W_c, W_p,
            error_bound=st["eps"],
            decoding_order=order,
            group_ids=st["group_ids"],
            sic_imperfection=st["sic_imp"],
        )
        R = rr["R_total"]
        Q = st["Q"]
        D = d_frac * torch.clamp(torch.minimum(R * st["tau"], Q), min=0.0)

        # --- 奖励镜像（lyapunov.compute_reward DGLCR 分支）---
        total_power = torch.sum(p)
        P_eff = torch.clamp(total_power / st["pa"] + st["pc"], min=st["p_floor"])
        served_total = torch.sum(D)
        if st["v2"]:
            w_user = torch.clamp(Q / torch.clamp(torch.mean(Q), min=1e-8),
                                 st["v2_lo"], st["v2_hi"])
            served_dkl = torch.sum(w_user * D)
        else:
            served_dkl = served_total

        # q* EMA 先更新后使用（lyapunov.py:388-396）
        beta = 1.0 - st["dkl_decay"]
        ema_s2 = st["dkl_decay"] * st["ema_s"] + beta * served_total
        ema_p2 = st["dkl_decay"] * st["ema_p"] + beta * P_eff
        q_new = torch.clamp(ema_s2 / torch.clamp(ema_p2, min=1e-8),
                            st["q_min"], st["q_max"])
        ee_comp = st["V"] * st["ee_w"] * (served_dkl - q_new * P_eff)

        backlog_w = (Q / st["Q_max"]) * st["drift_w"]
        tp_comp = st["w_tp"] * torch.sum(backlog_w * D)

        # Y 虚拟队列先更新后使用（lyapunov.py:612-619），A 用期望 λ
        A_total = torch.sum(st["lam"])
        Y2 = torch.clamp(st["Y_t"] + st["dvq_target"] * A_total - served_total,
                         min=0.0, max=st["y_max"])
        y_comp = st["w_y"] * (Y2 / st["y_scale"]) * served_total

        # X 惩罚用更新前的 X_t（lyapunov.py:605-607）
        if st["enable_pvq"]:
            pvq_pen = st["w_pvq"] * (st["X_t"] / st["pvq_scale"]) * total_power
        else:
            pvq_pen = torch.zeros((), dtype=self.dtype)

        # 溢出期望代理：OV = max(Q − D + λ − Q_max, 0)
        OV = torch.clamp(Q + st["lam"] - D - st["Q_max"], min=0.0)
        ov_pen = st["w_ov"] * torch.sum(st["ov_w"] * OV)

        # 常数项（对动作无梯度，用于 reward_pred 完整镜像）
        q_pen = st["ldptrlq_q_w"] * torch.sum(Q ** 2) / (st["Q_max"] ** 2)
        if st["w_early"] > 0.0 and st["warn_ratio"] < 1.0:
            excess = torch.clamp(Q / st["Q_max"] - st["warn_ratio"], min=0.0)
            early_pen = st["w_early"] * torch.sum(excess ** 2)
        else:
            early_pen = torch.zeros((), dtype=self.dtype)

        reward = st["reward_scale"] * (
            ee_comp + tp_comp + y_comp - q_pen - ov_pen - pvq_pen - early_pen
        )
        return {
            "reward": reward,
            "J": reward,  # 优化目标 = 完整镜像奖励
            "D": D,
            "p": p,
            "R": R,
            "P_eff": P_eff,
            "total_power": total_power,
            "q_new": q_new,
        }

    # ------------------------------------------------------------------
    # 求解：候选 order × Adam(p_logits) 取最优
    # ------------------------------------------------------------------

    def solve(self, env, warm_theta: Optional[np.ndarray] = None) -> Dict:
        st = self._read_state(env)
        best = None
        for name, order in st["orders"].items():
            theta = torch.zeros(
                self.K, dtype=self.dtype, requires_grad=True)
            if warm_theta is not None:
                with torch.no_grad():
                    theta.copy_(torch.as_tensor(
                        np.asarray(warm_theta, dtype=np.float64), dtype=self.dtype))
            opt = torch.optim.Adam([theta], lr=self.lr)
            with torch.enable_grad():
                for _ in range(self.n_steps):
                    opt.zero_grad()
                    out = self._mirror(st, theta, order)
                    loss = -out["reward"]
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                final = self._mirror(st, theta.detach(), order)
            cand = {
                "order_name": name,
                "order": np.asarray(order, dtype=np.int64),
                "theta": theta.detach().numpy().astype(np.float64).copy(),
                "J": float(final["reward"]),
                "p": final["p"].numpy().astype(np.float64).copy(),
                "D": final["D"].numpy().astype(np.float64).copy(),
                "R": final["R"].numpy().astype(np.float64).copy(),
                "total_power": float(final["total_power"]),
                "q_new": float(final["q_new"]),
            }
            if best is None or cand["J"] > best["J"]:
                best = cand

        p_act = np.clip(best["theta"], -ACT_BOUND, ACT_BOUND)
        cont_action = np.concatenate([
            np.full(self.K, ACT_BOUND, dtype=np.float64),   # α 块：死维，发射满格
            p_act,                                          # p logits（env 原始使用）
            np.full(self.K, ACT_BOUND, dtype=np.float64),   # d 块：d=D_FRAC
        ])
        full_action = np.concatenate([
            cont_action, best["order"].astype(np.float64)
        ])
        best["action"] = full_action
        best["cont_action"] = cont_action
        return best

    # ------------------------------------------------------------------
    # 评估（不优化）：给定 p_logits / order / d_frac 的镜像值
    # ------------------------------------------------------------------

    def evaluate(self, env, p_logits, order, d_frac: Optional[float] = None) -> Dict:
        st = self._read_state(env)
        l = torch.as_tensor(np.asarray(p_logits, dtype=np.float64), dtype=self.dtype)
        with torch.no_grad():
            out = self._mirror(st, l, np.asarray(order, dtype=np.int64), d_frac=d_frac)
        return {
            "reward": float(out["reward"]),
            "p": out["p"].numpy().astype(np.float64).copy(),
            "D": out["D"].numpy().astype(np.float64).copy(),
            "R": out["R"].numpy().astype(np.float64).copy(),
            "total_power": float(out["total_power"]),
        }
