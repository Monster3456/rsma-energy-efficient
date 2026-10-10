"""文献 2 的验证协议：参照敏感性 + 模型无关代理对照。

- ``jaccard_topk``：两归因向量 top-k 特征集的 Jaccard 重叠；
- ``recall_at_k``：pred 的 top-k 中落在 ref top-k 的比例（|pred∩ref|/k）；
- ``build_validation``：从 kernel/eg 结果 dict 汇总 per-SNR 的
  参照对 Jaccard、EG-vs-KernelSHAP recall@k。
"""

from __future__ import annotations

import numpy as np


def top_k_indices(v: np.ndarray, k: int) -> np.ndarray:
    """按 |归因| 排序的前 k 个特征下标（符号无关，文献 2 不比较极性）。"""
    return np.argsort(np.abs(np.asarray(v, dtype=np.float64)))[::-1][:k].astype(int)


def jaccard_topk(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    """top-k 集合 Jaccard：|A∩B| / |A∪B|，k 等长时 = |A∩B| / (2k − |A∩B|)。"""
    if k <= 0:
        return 0.0
    sa = set(top_k_indices(a, k))
    sb = set(top_k_indices(b, k))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def recall_at_k(pred: np.ndarray, ref: np.ndarray, k: int = 5) -> float:
    """pred 的 top-k 特征中落在 ref top-k 的比例（文献 2 的 recall@k 口径）。"""
    if k <= 0:
        return 0.0
    sa = set(top_k_indices(pred, k))
    sb = set(top_k_indices(ref, k))
    return len(sa & sb) / k


def build_validation(kernel: dict, eg: dict, refs: list[str], k: int = 5) -> dict:
    """从 per-SNR 归因结果构造验证汇总（文献 2 协议）。

    - ``jaccard_refs``：**内省方法（EG）** 在参照变体间 top-k 重叠——参照敏感性
      （文献 Fig.1 的混淆矩阵口径）；
    - ``recall_k``：EG 的 top-k 落在模型无关代理（KernelSHAP）top-k 中的比例
      （文献 Table 1 的 LIME-代理 recall@k 口径）。

    Returns:
        {snr: {target: {"k": int, "jaccard_refs": {pair: float}, "recall_k": {ref: float}}}}
    """
    out: dict = {}
    for snr, eg_tar_map in eg.items():
        out[snr] = {}
        for target, eg_ref_map in eg_tar_map.items():
            ker_ref_map = kernel.get(snr, {}).get(target, {})
            jacc = {}
            for i, r1 in enumerate(refs):
                for r2 in refs[i + 1 :]:
                    if r1 in eg_ref_map and r2 in eg_ref_map:
                        jacc[f"{r1}_vs_{r2}"] = round(float(jaccard_topk(eg_ref_map[r1], eg_ref_map[r2], k)), 3)
            rec = {}
            for r in refs:
                if r in ker_ref_map and r in eg_ref_map:
                    rec[r] = round(float(recall_at_k(eg_ref_map[r], ker_ref_map[r], k)), 3)
            out[snr][target] = {"k": int(k), "jaccard_refs": jacc, "recall_k": rec}
    return out
