"""一次性诊断（M2 门禁）：求解器专家 vs 基线跨 SNR 对照。

问题：单槽 DGLCR surrogate 的梯度精修专家（solver_expert）在真实 env
长时隙滚动下，能否同时做到：
  G1: 0/5dB（低 SNR 失败区）delivery >= SDMA 基线
  G2: 平均总功率 < 24W（softmax logit 展宽的省电能力）
附带：warm vs cold 起点对照、专家 order 选择分布。

用法：
  PYTHONPATH=. .venv/Scripts/python.exe scripts/_diag_solver_expert.py [--eps 3] [--snrs 0,5,15,30]
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from src.config import Config  # noqa: E402
from src.env.rsma_env import RSMAEnergyEnv  # noqa: E402
from src.optimization.solver_expert import SolverExpert  # noqa: E402
from baselines.sdma_baseline import SDMABaseline  # noqa: E402

METRICS = ["delivery", "power", "queue", "overflow", "EE"]


def build_env(snr_db: float) -> RSMAEnergyEnv:
    cfg = Config()
    cfg.system.csi_mode = "imperfect"
    cfg.system.sigma2 = 10 ** (-snr_db / 10)
    cfg.system.snr_domain_randomization = False
    return RSMAEnergyEnv(cfg)


def run_episode(env, seed, action_fn):
    obs, _ = env.reset(seed=seed)
    acc = {m: 0.0 for m in METRICS}
    steps = 0
    extra = {}
    for _ in range(env.T):
        action, ex = action_fn(obs, env)
        obs, r, term, trunc, info = env.step(action)
        acc["delivery"] = float(info["delivery_ratio"])
        acc["power"] += float(info["total_power"])
        acc["queue"] += float(info["mean_queue"])
        acc["overflow"] += float(info["overflow_rate"])
        acc["EE"] += float(info["EE"])
        extra = ex or extra
        steps += 1
        if term or trunc:
            break
    out = {m: acc[m] / max(steps, 1) for m in METRICS}
    out["delivery"] = acc["delivery"]  # 最终累计值（非均值）
    out["steps"] = steps
    return out, extra


def make_expert_fn(n_steps, warm):
    state = {"theta": None, "expert": None, "order_counts": {}, "Js": []}

    def fn(obs, env):
        if state["expert"] is None:
            state["expert"] = SolverExpert(env, n_steps=n_steps)
        wt = state["theta"] if warm else None
        res = state["expert"].solve(env, warm_theta=wt)
        state["theta"] = res["theta"]
        state["order_counts"][res["order_name"]] = state["order_counts"].get(res["order_name"], 0) + 1
        state["Js"].append(res["J"])
        return res["action"], {"order_counts": dict(state["order_counts"]), "J_mean": float(np.mean(state["Js"]))}

    return fn


def make_sdma_fn():
    sdma = [None]

    def fn(obs, env):
        if sdma[0] is None:
            sdma[0] = SDMABaseline(env.config)
        return sdma[0].select_action(obs, env), None

    return fn


def make_uniform_fn():
    def fn(obs, env):
        K = env.K
        return np.zeros(3 * K, dtype=np.float64), None

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, default=3)
    ap.add_argument("--snrs", type=str, default="0,5,15,30")
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()
    snrs = [float(s) for s in args.snrs.split(",")]

    policies = {
        "expert_w": make_expert_fn(args.steps, warm=True),
        "expert_c": make_expert_fn(args.steps, warm=False),
        "sdma": make_sdma_fn(),
        "uniform": make_uniform_fn(),
    }

    results = {p: {s: [] for s in snrs} for p in policies}
    extras = {p: {} for p in policies}

    for snr in snrs:
        for pname, fn in policies.items():
            t0 = time.time()
            for ep in range(args.eps):
                env = build_env(snr)
                out, ex = run_episode(env, seed=ep, action_fn=fn)
                results[pname][snr].append(out)
                if ex:
                    extras[pname][snr] = ex
            dt = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] {pname} @ {snr:.0f}dB done in {dt:.0f}s", flush=True)

    # ------- 汇总表 -------
    print("\n=== mean over", args.eps, "episodes ===")
    for m, title in [
        ("delivery", "delivery_ratio (final)"),
        ("power", "total_power (W)"),
        ("queue", "mean_queue"),
        ("overflow", "overflow_rate"),
        ("EE", "EE"),
    ]:
        print(f"\n--- {title} ---")
        header = f"{'SNR':>5} | " + " | ".join(f"{p:>9}" for p in policies)
        print(header)
        print("-" * len(header))
        for snr in snrs:
            row = [f"{snr:>5.0f}"]
            for p in policies:
                vals = [r[m] for r in results[p][snr]]
                row.append(f"{np.mean(vals):>9.4f}")
            print(" | ".join(row))

    for p in ("expert_w", "expert_c"):
        ex_by_snr = extras.get(p, {})
        if ex_by_snr:
            print(f"\n{p} order selection / J:")
            for snr, ex in ex_by_snr.items():
                print(f"  {snr:.0f}dB: {ex['order_counts']}  J_mean={ex['J_mean']:.4f}")

    # ------- M2 门禁 -------
    # 门禁对象 = expert_c（冷启动）：训练收集路径 _solver_expert_collect 即用
    # solve(env) 不传 warm_theta。实证（2026-09-02 全量 4 SNR × 3ep）：warm 起点携带
    # 上一时隙的过期信道先验，25 步 Adam 内反而落入更差盆地（0dB 0.811 vs 冷启
    # 0.899）——与 DRL-RSMA「测试时精修对初始化极稳健」结论一致，冷启动是正解。
    print("\n=== M2 gates (gated on expert_c = collection-mode expert) ===")
    g1_ok = True
    for snr in (s for s in snrs if s <= 5.0):
        e = np.mean([r["delivery"] for r in results["expert_c"][snr]])
        s = np.mean([r["delivery"] for r in results["sdma"][snr]])
        ok = e >= s - 1e-9
        g1_ok &= ok
        print(f"G1 @{snr:.0f}dB: expert delivery {e:.4f} vs SDMA {s:.4f} -> {'PASS' if ok else 'FAIL'}")
    g2_ok = True
    for snr in snrs:
        pw = np.mean([r["power"] for r in results["expert_c"][snr]])
        ok = pw < 24.0 - 1e-9
        g2_ok &= ok
        print(f"G2 @{snr:.0f}dB: expert power {pw:.3f}W < 24W -> {'PASS' if ok else 'FAIL'}")
    print(f"\nM2 GATE: {'ALL PASS' if (g1_ok and g2_ok) else 'FAILED'}")


if __name__ == "__main__":
    main()
