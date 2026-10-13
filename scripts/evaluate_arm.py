"""评估 + 裁决一键化：对某实验臂的 best_guard 跑官方评估并直接给出预注册裁决行。

用法（Windows venv，PYTHONPATH=.）：
    python scripts/evaluate_arm.py --run-dir logs/intp_P2a_s43 --prune
    python scripts/evaluate_arm.py --run-dir logs/xxx --ref-deliv 0.9526 \
        --append .trellis/tasks/10-09-interpretability/ablation_xxx.md

行为：委托 evaluate_snr_sweep_temporal（不可变脚本的官方口径 wrapper），解析
逐点 PPO-RSMA/SDMA 的 (Deliv, Q)，按两门（deliv≥SDMA 且 Q≤1.1×SDMA）计算
points_met / mean deliv / Δvs 参考；打印/追加一条 markdown 裁决行。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_eval(text: str):
    """解析官方评估输出：{snr: {"PPO-RSMA": (deliv, Q), "SDMA": (deliv, Q), ...}}。"""
    rows: dict[int, dict] = {}
    snr = None
    for line in text.splitlines():
        m = re.search(r"SNR= *(\d+) dB", line)
        if m:
            snr = int(m.group(1))
        m = re.search(r"(\S+) +P=([\d.]+) +Q=([\d.]+) +Deliv=([\d.]+)", line)
        if m and snr is not None:
            rows.setdefault(snr, {})[m.group(1)] = (float(m.group(4)), float(m.group(3)))
    return rows


def adjudicate(rows: dict, ref_deliv: float, snrs=None):
    """按两门计算 points_met / mean deliv / Δ，返回可打印行。"""
    snrs = snrs or [0, 5, 10, 15, 20, 25, 30]
    delivs, gates = [], []
    pdc_missing = False
    for s in snrs:
        p = rows.get(s, {}).get("PPO-RSMA")
        d = rows.get(s, {}).get("SDMA")
        if p is None or d is None:
            pdc_missing = True
            continue
        g = p[0] >= d[0] and p[1] <= 1.1 * d[1]
        delivs.append(p[0])
        gates.append(g)
    if not delivs:
        return None
    mean = sum(delivs) / len(delivs)
    return {
        "points_met": f"{sum(gates)}/{len(snrs)}" + (" (缺SDMA对比? 人工复核)" if pdc_missing else ""),
        "mean_deliv": round(mean, 4),
        "delta": round(abs(mean - ref_deliv), 4),
        "gate": "PASS" if (sum(gates) == len(snrs) and abs(mean - ref_deliv) <= 0.005) else "FAIL",
        "per_point": [(s, round(rows[s]["PPO-RSMA"][0], 3), rows[s]["PPO-RSMA"][1]) for s in snrs
                      if s in rows and "PPO-RSMA" in rows[s]],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="logs/<exp> 运行目录（含 checkpoints/）")
    p.add_argument("--checkpoint", default=None, help="默认 <run-dir>/checkpoints/best_guard.pt")
    p.add_argument("--ref-deliv", type=float, default=0.9526, help="裁决参考基线（默认 s43 A-prune）")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--obs-history-window", type=int, default=4)
    p.add_argument("--prune", action="store_true", help="剪枝观测检查点（A-prune 系）→ 加 --obs-prune-constant-blocks")
    p.add_argument("--snr-subset", default="", help="只评估指定 SNR（逗号分隔）")
    p.add_argument("--append", default="", help="追加大 markdown 行到该文件")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    ckpt = args.checkpoint or str(run_dir / "checkpoints" / "best_guard.pt")
    if not Path(ckpt).exists():
        raise SystemExit(f"checkpoint 不存在: {ckpt}")

    cmd = [sys.executable, "scripts/evaluate_snr_sweep_temporal.py",
           "--obs_history_window", str(args.obs_history_window),
           "--checkpoint", str(ckpt),
           "--n_episodes", str(args.n_episodes)]
    if args.prune:
        cmd.append("--obs-prune-constant-blocks")
    if args.snr_subset:
        cmd += ["--snr_subset", args.snr_subset]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        raise SystemExit(f"评估失败 rc={proc.returncode}:\n{text[-1500:]}")

    rows = parse_eval(text)
    req_snrs = [int(s) for s in args.snr_subset.split(",") if s] if args.snr_subset else None
    verdict = adjudicate(rows, args.ref_deliv, snrs=req_snrs)
    if verdict is None:
        raise SystemExit("未解析到评估结果（检查输出格式/--snr-subset 与解析器匹配）")
    line = (f"| {run_dir.name} | {verdict['points_met']} | {verdict['mean_deliv']} "
            f"| Δ{verdict['delta']} | {verdict['gate']} |")
    print(line)
    for s, d, q in verdict["per_point"]:
        print(f"  SNR {s:>2}: deliv {d:.3f} / Q {q:6.2f}")
    if args.append:
        Path(args.append).open("a", encoding="utf-8").write(line + "\n")
        print(f"appended -> {args.append}")

    raise SystemExit(0 if verdict["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()