"""SVEA：Shapley 值误差分摊的可解释特征子集选择（文献 1 核心）。

给定 (状态特征, 最终任务标签) 样本，构造分类博弈：

- 特征函数 ``v(S) = c̃ − tr_er(S)``，其中
  ``tr_er(S, m) = (1/m)·Σ ξ_i*`` 是子集 S 上 hinge-loss 线性 ERM 的最优平均松弛
  （线性规划，scipy ``linprog(method="highs")``），``c̃ = tr_er(∅)`` 为截距 only 误差；
- Shapley 值 φ_j（MC 排列采样近似）把 ``v(N)`` 分摊到各特征；
- SVEA ``e_j = c̃/n − φ_j``（文献 Theorem 1 的仿射变换）；
- **普适阈值 0**：``e_j < 0`` 的特征为支配性子集 SVEA_neg（``dominant_features``）；
- t-分布区间估计（分组 CLT）抗采样偏差；PSV 验证子集分类效力。

注：特征在**全集**上全局 z-score 标准化（每个子集沿用同一变换）。线性分类器
假设类在仿射特征变换下不变（w_j·x_j/σ_j 与 v_j·x_j 覆盖同一函数集），故博弈值精确不变。
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as sopt
import scipy.stats as sstats

# ---------------------------------------------------------------- 基础解算 --


def _hinge_lp(X_S: np.ndarray, y: np.ndarray):
    """解 hinge-loss 线性 ERM 的原始 LP。

    min  (1/m)·Σ ξ_i
    s.t. y_i(Σ_j w_j x_ij + b) ≥ 1 − ξ_i,  ξ_i ≥ 0

    变量列序 = [w_1..w_r, b, ξ_1..ξ_m]；返回 (w, b, err=最优平均松弛)。

    注：w/b 使用 **有界** 变量（±1e8，数值上等价于无界）——实测 hiGHS 在
    无界变量 + 高度不平衡标签（如 served 标签 99:1）下会以
    "HiGHS Status 0: Not Set" 数值失败；有界化后稳定收敛，不动摇博弈数学。
    """
    m = int(y.size)
    r = X_S.shape[1]
    A_ub = np.zeros((m, r + 1 + m))
    for i in range(m):
        A_ub[i, :r] = -y[i] * X_S[i]
        A_ub[i, r] = -y[i]
        A_ub[i, r + 1 + i] = -1.0
    b_ub = np.full(m, -1.0)
    c = np.zeros(r + 1 + m)
    c[r + 1 :] = 1.0 / m
    _L = 1e8
    bounds = [(-_L, _L)] * (r + 1) + [(0.0, None)] * m
    # hiGHS 默认解对高度不平衡/近退化实例偶发 "HiGHS Status 0: Not Set"：
    # 依次退化到 presolve=False 与内点法（三者数值解相同，仅求解器鲁棒性不同）。
    last = None
    for method, opts in (("highs", {}), ("highs", {"presolve": False}), ("highs-ipm", {})):
        try:
            res = sopt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method, options=opts)
            if res.success:
                w = res.x[:r]
                b = float(res.x[r])
                err = float(res.fun)  # (1/m)·Σξ*
                return w, b, err
            last = RuntimeError(f"hinge-LP 不可行（m={m}, r={r}, {method}/{opts}）：{res.message}")
        except Exception as ex:  # noqa: BLE001 —— 求解器异常统一按策略失败处理
            last = ex
    # 最后兜底：确定性 1e-9 级扰动打破近退化基（实测 "HiGHS Status 0: Not Set"
    # 全是纯数值退化，扰动后最优值稳定到 ~1e-4，不动摇博弈结论）。
    A_jit = A_ub + 1e-9 * np.random.default_rng(0).normal(size=A_ub.shape)
    for method in ("highs", "highs-ds"):
        try:
            res = sopt.linprog(c, A_ub=A_jit, b_ub=b_ub, bounds=bounds, method=method)
            if res.success:
                return res.x[:r], float(res.x[r]), float(res.fun)
            last = RuntimeError(f"hinge-LP 抖动后仍失败（m={m}, r={r}, {method}）：{res.message}")
        except Exception as ex:  # noqa: BLE001
            last = ex
    raise last


def tr_er(X_S: np.ndarray, y: np.ndarray) -> float:
    """子集 S 的 hinge-loss 训练误差 tr_er(S, m)（最优平均松弛）。"""
    _, _, err = _hinge_lp(np.asarray(X_S, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return err


def tr_er_empty(y: np.ndarray) -> float:
    """截距 only 分类器误差 c̃ = tr_er(∅, m)（闭式）。

    最优常数分类器 b*∈{−1,+1}：平均 hinge = 2·min(p, 1−p)，p=正类比例。
    """
    y = np.asarray(y, dtype=np.float64)
    p_pos = float((y > 0).mean())
    return 2.0 * min(p_pos, 1.0 - p_pos)


def hinge_lp_solution(X_S: np.ndarray, y: np.ndarray):
    """返回 (w, b, err)：用于 PSV 的预测/准确率。"""
    return _hinge_lp(np.asarray(X_S, dtype=np.float64), np.asarray(y, dtype=np.float64))


# ---------------------------------------------------------------- MC Shapley --

#: 并行求值上下文（ProcessPool initializer 注入；仅 workers>1 时使用）
_CTX: dict = {}


def _init_worker(Xs, ys: np.ndarray, c_tilde: float, tr_N: float, full_mask: int) -> None:
    """把博弈数据注入子进程全局（spawn 时随 initargs 序列化一次）。"""
    global _CTX
    _CTX = {
        "Xs": Xs, "ys": ys,
        "c_tilde": float(c_tilde), "tr_N": float(tr_N),
        "full_mask": int(full_mask), "n_feat": int(full_mask).bit_length(),
    }


def _eval_mask_par(mask: int) -> float:
    """子进程端的 mask→v 求值（与 svea_game.eval_subset 数学一致）。"""
    p = _CTX
    if mask == 0:
        return p["c_tilde"]
    if mask == p["full_mask"]:
        return p["tr_N"]
    cols = np.array(
        [j for j in range(p["n_feat"]) if mask & (1 << j)], dtype=np.int64
    )
    return p["c_tilde"] - tr_er(p["Xs"][:, cols], p["ys"])


def _mask_sequence(perms: np.ndarray, n: int) -> set[int]:
    """排列游走需要求值的全部前缀 mask（两阶段法第一阶段）。"""
    needed = {0}
    prefix = 0
    for row in perms:
        prefix = 0
        for j in row:
            prefix |= 1 << int(j)
            needed.add(prefix)
    return needed


def mc_permutation_shapley(eval_subset, n: int, B: int, rng: np.random.Generator,
                           workers: int = 1):
    """Maleki 式 MC 排列采样 Shapley 值（两阶段：先定 mask 集 → 求值 → 走游走）。

    Args:
        eval_subset: mask(bitmask int) -> v(S) 的特征函数（串行路径）。
        n: 玩家数。
        B: 排列样本数。
        rng: 确定性随机源。
        workers: >1 时用 ProcessPool 并行求值（值只算一次、顺序无关 →
            结果与串行逐位一致，仅墙钟不同）；≤1 串行。

    Returns:
        (phi (n,), n_solves): 均值 Shapley 与特征函数求值次数（= 去重 mask 数）。
    """
    from concurrent.futures import ProcessPoolExecutor

    perms = np.stack([rng.permutation(n) for _ in range(B)]).astype(np.int64)
    needed = _mask_sequence(perms, n)

    if workers > 1 and B * n >= 256:
        full_mask = (1 << n) - 1
        masks = sorted(needed)
        # initargs 需要 Xs/ys；eval_subset 是闭包不可序列化 → 由调用方预先
        # 注入 _CTX（见 svea_game workers>0 分支）；此处仅兜底断言。
        with ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_init_worker,
            initargs=(_CTX.get("Xs"), _CTX.get("ys"), _CTX.get("c_tilde"),
                      _CTX.get("tr_N"), full_mask),
        ) as ex:
            vals = list(ex.map(_eval_mask_par, masks))
        value = dict(zip(masks, vals))
    else:
        value: dict[int, float] = {}
        for mask in needed:
            if mask not in value:
                value[mask] = float(eval_subset(mask))

    phi = np.zeros(n, dtype=np.float64)
    for row in perms:
        prefix = 0
        for j in row:
            jj = int(j)
            nxt = prefix | (1 << jj)
            phi[jj] += value[nxt] - value[prefix]
            prefix = nxt
    phi /= max(B, 1)
    return phi, len(value)


def svea(phi: np.ndarray, c_tilde: float, n: int) -> np.ndarray:
    """SVEA_j = c̃/n − φ_j（文献 Theorem 1 的仿射变换）。"""
    return np.asarray(c_tilde, dtype=np.float64) / n - np.asarray(phi, dtype=np.float64)


def dominant_features(e: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """SVEA_neg = {j : e_j < 0}——普适阈值 0 的支配性特征子集。"""
    return np.flatnonzero(np.asarray(e, dtype=np.float64) < tol).astype(int)


def t_ci(e_matrix: np.ndarray):
    """对 G 组 SVEA 值矩阵 (G, n) 计算 t 分布 95% 区间。

    Returns:
        (mean (n,), lo (n,), hi (n,))：均值与 95% 置信区间下/上界。
    """
    e = np.asarray(e_matrix, dtype=np.float64)
    G = e.shape[0]
    mean = e.mean(axis=0)
    if G < 2:
        return mean, mean, mean
    se = e.std(axis=0, ddof=1) / np.sqrt(G)
    tcrit = float(sstats.t.ppf(0.975, G - 1))
    return mean, mean - tcrit * se, mean + tcrit * se


# ---------------------------------------------------------------- 分类博弈 --


def standardize(X: np.ndarray):
    """全局 z-score（拟合在全集上，结果对子集复用）。返回 (Xz, mu, sd)。"""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _stratified_subsample(
    X: np.ndarray, y: np.ndarray, m: int, rng: np.random.Generator, class_cap: float = 10.0
):
    """分层子采样（类不平衡缓解）。

    可平衡时取各半（各 m/2）；少数类过小（n_min < m/(cap+1)）时把样本数收缩
    到少数×cap 上下（如 9 个负样本 → 9:90 = 99 行），避免 majority-trick 与
    hiGHS 数值退化。返回 (X, y, idx)。
    """
    n = X.shape[0]
    if n <= m:
        return X, y, np.arange(n)
    pos = np.flatnonzero(y > 0)
    neg = np.flatnonzero(y <= 0)
    n_min = min(pos.size, neg.size)
    m_target = m
    if n_min < m / (class_cap + 1.0):
        m_target = max(2, int(n_min * (class_cap + 1.0)))
    n_neg_eff = min(neg.size, m_target // 2)
    n_pos_eff = min(pos.size, m_target - n_neg_eff)
    n_neg_eff = min(neg.size, m_target - n_pos_eff)  # 多数类短缺时回退
    if n_pos_eff < 1 and pos.size:
        n_pos_eff = 1
    idx = np.concatenate(
        [
            rng.choice(pos, size=int(n_pos_eff), replace=False),
            rng.choice(neg, size=int(n_neg_eff), replace=False),
        ]
    )
    rng.shuffle(idx)
    return X[idx], y[idx], idx


def svea_game(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    m_sv: int = 1000,
    B: int = 200,
    cache: dict | None = None,
    zscore: bool = True,
    class_cap: float = 10.0,
    exclude: np.ndarray | None = None,
    workers: int = 1,
):
    """对 (X, y) 跑完整 SVEA 分类博弈。

    Args:
        X: (n_samples, n_features)。
        y: (n_samples,) ∈ {−1,+1} 标签。
        rng: 确定性随机源。
        m_sv: 博弈样本数目标（实际行数受类别可用量约束，见树上采样）。
        B: 排列样本数。
        cache: 可选跨调用 bitmask→v 缓存（复用）。
        zscore: True 时内部在全集上 z-score（CI 分组传 zscore=False）。
        class_cap: 少数类比例上限（类不平衡缓解，见 _stratified_subsample）。
        exclude: 不参与博弈的特征下标（**零方差/确定性 null player**，见注）。

    Returns:
        dict: phi, sv, c_tilde, tr_er_N, sv_neg, n_solves, excluded, n_feat。

    注（方法学选择）：
    - 特征先全局 z-score、再按列 max-|·| 归一 —— 两者都是仿射变换，线性假设类
      不变，博弈值/Shapley 精确不变；列归一保证 hiGHS 数值稳定（实测 z-score
      后部分子集仍触发 "HiGHS Status 0: Not Set"，归一后消除）。
    - 类别过采样/收缩按类不平衡缓解处理（served 标签在某些 SNR 达 99:1）。
    - **零方差特征（固定 SNR 评估点内确定性的 arrivals/λ_bucket/Dinkelbach-η 块）
      = 合作博弈里的 null player**：边际恒为 0 → φ=0、e_j = c̃/n > 0 永不进入
       SVEA_neg（null-player 公理，其余特征的值精确不变）。从排列游走中排除它们
      避免退化 LP；SVEA 仿射变换仍用**原始 n=13**，阈值口径与全 13 特征博弈一致。
    """
    if zscore:
        Xz, _, _ = standardize(X)
    else:
        Xz = np.asarray(X, dtype=np.float64)
    n_feat = Xz.shape[1]
    exclude = np.asarray([], dtype=np.int64) if exclude is None else np.asarray(exclude, dtype=np.int64)
    active = np.array([j for j in range(n_feat) if j not in set(exclude.tolist())], dtype=np.int64)
    col_norm = np.maximum(np.abs(Xz).max(axis=0), 1e-9)
    Xc = Xz / col_norm
    Xs, ys, _ = _stratified_subsample(Xc, y, int(m_sv), rng, class_cap=class_cap)
    cache = {} if cache is None else cache
    c_tilde = tr_er_empty(ys)
    tr_N = tr_er(Xs[:, active], ys)
    n_active = len(active)
    full_mask = (1 << n_active) - 1

    def eval_subset(mask: int) -> float:
        if mask == 0:
            return c_tilde
        if mask == full_mask:
            return tr_N
        cols = active[[j for j in range(n_active) if mask & (1 << j)]]
        return c_tilde - tr_er(Xs[:, cols], ys)

    global _CTX
    if int(workers) > 1:
        # 并行路径：把样本与常数注入模块级上下文（子进程 initargs 用）
        _CTX.update(Xs=Xs, ys=ys, c_tilde=c_tilde, tr_N=tr_N)
    phi_active, n_solves = mc_permutation_shapley(eval_subset, n_active, B, rng,
                                                  workers=int(workers))
    phi = np.zeros(n_feat, dtype=np.float64)  # null player → 0
    phi[active] = phi_active
    sv = svea(phi, c_tilde, n_feat)  # 原始 n=13 的仿射变换
    return {
        "phi": phi,
        "sv": sv,
        "c_tilde": c_tilde,
        "tr_er_N": tr_N,
        "sv_neg": dominant_features(sv),
        "n_solves": n_solves,
        "excluded": exclude.tolist(),
        "n_feat": n_feat,
    }


def constant_feature_cols(X: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """零方差特征列下标（null player 检测）。"""
    Xz = np.asarray(X, dtype=np.float64)
    return np.flatnonzero(Xz.std(axis=0) < tol).astype(np.int64)


# ---------------------------------------------------------------- PSV 验证 --


def _subsample_indices(idx: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """从索引中确定性抽取 m 个（无放回），供 PSV fold 使用。"""
    if idx.size <= m:
        return idx
    return rng.choice(idx, size=m, replace=False)


def psv_linear(
    X: np.ndarray,
    y: np.ndarray,
    subset: np.ndarray,
    rng: np.random.Generator,
    exclude: np.ndarray | None = None,
):
    """PSV(K) = 子集 K 训练分类器测试准确率 / 全集分类器测试准确率（论文 Def. 2）。

    子集与全集各训练一个 hinge-loss 线性分类器（训练折叠子采样 ≤600 控 LP 规模），
    在同一独立测试折叠上比较准确率。零方差特征（exclude）从两个分类器的特征列
    中剔除（null player 不贡献判别力）。

    Returns:
        (psv, acc_subset, acc_full)，样本不足时 (None, None, None)。
    """
    Xz, _, _ = standardize(X)
    n = Xz.shape[0]
    te_idx = rng.choice(n, size=max(1, n // 5), replace=False)
    tr_idx = np.setdiff1d(np.arange(n), te_idx)
    if tr_idx.size < 40 or te_idx.size < 20:
        return None, None, None
    tr_idx = _subsample_indices(tr_idx, 600, rng)
    exc = set() if exclude is None else set(np.asarray(exclude, dtype=np.int64).tolist())
    all_cols = [j for j in range(Xz.shape[1]) if j not in exc]
    sub_cols = [int(j) for j in np.asarray(subset) if j not in exc]
    w_f, b_f, _ = hinge_lp_solution(Xz[tr_idx][:, all_cols], y[tr_idx])
    w_s, b_s, _ = hinge_lp_solution(Xz[tr_idx][:, sub_cols], y[tr_idx])
    acc_full = float((np.sign(Xz[te_idx][:, all_cols] @ w_f + b_f) == y[te_idx]).mean())
    acc_sub = float((np.sign(Xz[te_idx][:, sub_cols] @ w_s + b_s) == y[te_idx]).mean())
    if acc_full <= 0.0:
        return None, acc_sub, acc_full
    return acc_sub / acc_full, acc_sub, acc_full


def psv_mlp(
    X: np.ndarray,
    y: np.ndarray,
    subset: np.ndarray,
    rng: np.random.Generator,
    train_frac: float = 0.8,
    steps: int = 200,
    hidden: int = 32,
):
    """非线性 PSV（torch 小 MLP，论文 §4 的非线性分类器验证口径，flag 可选）。

    Returns:
        (psv, acc_subset, acc_full)；样本不足返回 (None, None, None)。
    """
    import torch
    from torch import nn

    Xz, _, _ = standardize(X)
    n = Xz.shape[0]
    te_n = max(1, int(n * (1 - train_frac)))
    te_idx = rng.choice(n, size=te_n, replace=False)
    tr_idx = np.setdiff1d(np.arange(n), te_idx)
    tr_idx = _subsample_indices(tr_idx, 800, rng)
    if tr_idx.size < 40 or te_idx.size < 20:
        return None, None, None
    full_cols = list(range(Xz.shape[1]))
    sub_cols = [int(j) for j in np.asarray(subset)]

    def fit_acc(cols: list[int]) -> float:
        Xtr = torch.from_numpy(Xz[tr_idx][:, cols]).float()
        ytr = torch.from_numpy((y[tr_idx] > 0).astype(np.float32))
        model = nn.Sequential(nn.Linear(len(cols), hidden), nn.ReLU(), nn.Linear(hidden, 1))
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(steps):
            perm = torch.randperm(Xtr.shape[0])[:128]
            opt.zero_grad()
            out = model(Xtr[perm]).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(out, ytr[perm])
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = model(torch.from_numpy(Xz[te_idx][:, cols]).float()).squeeze(-1) > 0
            acc = float((pred.numpy() == (y[te_idx] > 0)).mean())
        return acc

    acc_sub = fit_acc(sub_cols)
    acc_full = fit_acc(full_cols)
    if acc_full <= 0.0:
        return None, acc_sub, acc_full
    return acc_sub / acc_full, acc_sub, acc_full
