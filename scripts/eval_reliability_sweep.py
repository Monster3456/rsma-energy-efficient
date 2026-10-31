# -*- coding: utf-8 -*-
"""A: 逐 episode 可靠性跨基线扫描（零重训）。

对每个 SNR 点、每个基线（sdma / noma / equal），按官方口径逐 episode 跑
`_run_episode`，统计**逐点双门**（delivery >= ZETA_SDMA 且 ep 平均队列
<= 1.1*Q_ref）的 ep 违约率——把"均值达标 vs 逐回合可靠"的差异第一次填满基准表。

PPO 冠军（0dB 20%、5dB 4%、合计 3.4%）与 D3（22%/8%/>）已由 m4 桩测出并记录
（spec-stage1 §8）；本脚本只产出基线行，最终表在报告时合并。

用法:
  PYTHONPATH=. python scripts/eval_reliability_sweep.py [--n_episodes 50] \\
      [--baselines sdma,noma,equal] [--csi_mode imperfect]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.baselines.equal_power import EqualPowerBaseline  # noqa: E402
from scripts.baselines.noma_baseline import NOMABaseline  # noqa: E402
from scripts.baselines.sdma_baseline import SDMABaseline  # noqa: E402
from scripts.evaluate_snr_sweep import (  # noqa: E402
    SNR_POINTS,
    _run_episode,
    build_config_for_snr,
    make_baseline_action_fn,
)
from src.env.rsma_env import RSMAEnergyEnv  # noqa: E402

# 逐点交付门（SDMA 参照）与队列门（SDMA Q × 1.1）
ZETA_SDMA = {0: 0.856, 5: 0.979, 10: 0.961, 15: 0.887, 20: 0.795, 25: 0.724, 30: 0.664}
Q_SDMA = {0: 3.46, 5: 0.68, 10: 2.28, 15: 11.42, 20: 22.39, 25: 28.76, 30: 34.47}

BASELINE_FACTORY = {
    "sdma": SDMABaseline,
    "noma": NOMABaseline,
    "equal": EqualPowerBaseline,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--baselines", type=str, default="sdma,noma,equal")
    ap.add_argument("--csi_mode", default="imperfect")
    ap.add_argument("--out", default="logs/reliability_sweep.json")
    args = ap.parse_args(argv)

    bnames = [b for b in args.baselines.split(",") if b in BASELINE_FACTORY]
    if not bnames:
        raise SystemExit(f"无效基线 {args.baselines}；可用 {list(BASELINE_FACTORY)}")

    report = {}
    print("| SNR | 基线 | mean deliv | deliv ep-viol | mean Q | Q ep-viol | mean P |", flush=True)
    print("|---:|---|---:|---:|---:|---:|---:|", flush=True)
    for snr_db, sigma2 in SNR_POINTS:
        cfg = build_config_for_snr(sigma2, args.csi_mode)
        env = RSMAEnergyEnv(cfg)
        dref = ZETA_SDMA[int(snr_db)]
        qref = Q_SDMA[int(snr_db)] * 1.1
        for name in bnames:
            base = BASELINE_FACTORY[name](cfg)
            fn = make_baseline_action_fn(base)
            d_viol = q_viol = 0
            dls, qs, pws = [], [], []
            for ep in range(args.n_episodes):
                m = _run_episode(fn, env, seed=ep, meta_ac=None, T_s=20)
                dls.append(float(m["delivery_ratio"]))
                qs.append(float(m["mean_queue"]))
                pws.append(float(m["total_power"]))
                if m["delivery_ratio"] < dref:
                    d_viol += 1
                if m["mean_queue"] > qref:
                    q_viol += 1
            d_v = d_viol / args.n_episodes
            q_v = q_viol / args.n_episodes
            report.setdefault(str(int(snr_db)), {})[name] = {
                "deliver_ratio": float(np.mean(dls)),
                "deliv_ep_viol": d_v,
                "mean_queue": float(np.mean(qs)),
                "queue_ep_viol": q_v,
                "total_power": float(np.mean(pws)),
                "n": args.n_episodes,
            }
            print(f"| {snr_db:>2} | {name:>5} | {np.mean(dls):.4f} | {d_v:.3f} "
                  f"| {np.mean(qs):.2f} | {q_v:.3f} | {np.mean(pws):.2f} |", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    # 汇总：跨点 ep 违约（仅交付门）
    print("\n=== 可靠性汇总（交付门 ep 违约率，跨 7 点）===", flush=True)
    for name in bnames:
        rows = [report[p][name] for p in report]
        tot = sum(r["deliv_ep_viol"] * r["n"] for r in rows)
        n_all = sum(r["n"] for r in rows)
        mx = max(r["deliv_ep_viol"] for r in rows)
        print(f"  {name:>5}: total={tot}/{n_all}={tot / n_all:.3f}  最差点={mx:.3f}", flush=True)
    print(f"结果 -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())