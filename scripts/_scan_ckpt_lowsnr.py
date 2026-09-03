"""低 SNR 子集 checkpoint 扫描（诊断脚本，不进版本库）

对 temporal_worker_h4_trunk 的全部存档点在 {0, 5} dB 上跑 50ep 评估，
找能补齐两个失败点的候选。逐 checkpoint 顺序执行，输出汇总表。

用法: PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv/Scripts/python.exe scripts/_scan_ckpt_lowsnr.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO / "logs" / "temporal_worker_h4_trunk" / "checkpoints"
EVAL = REPO / "scripts" / "evaluate_snr_sweep_temporal.py"
PY = REPO / ".venv" / "Scripts" / "python.exe"

CKPTS = ["best.pt"] + [f"ckpt_{i}.pt" for i in range(100, 800, 100)] + ["final.pt"]

SNR_HDR = re.compile(r"=== SNR=\s*(\d+) dB")
PPO_ROW = re.compile(r"PPO-RSMA\s+P=([\d.]+)\s+Q=([\d.]+)\s+Deliv=([\d.]+)")
SDMA_ROW = re.compile(r"^\s+SDMA\s+P=([\d.]+)\s+Q=([\d.]+)\s+Deliv=([\d.]+)")


def scan_one(name: str) -> dict:
    ckpt = CKPT_DIR / name
    if not ckpt.is_file():
        return {"ckpt": name, "error": "missing"}
    cmd = [
        str(PY), str(EVAL),
        "--checkpoint", str(ckpt),
        "--n_episodes", "50",
        "--snr_subset", "0,5",
        "--obs_history_window", "4",
    ]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(REPO)
    r = subprocess.run(
        cmd, cwd=str(REPO), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    out = r.stdout + r.stderr
    ppo, sdma, cur = {}, {}, None
    for line in out.splitlines():
        m = SNR_HDR.search(line)
        if m:
            cur = int(m.group(1))
            continue
        if cur is None:
            continue
        m = PPO_ROW.search(line)
        if m:
            ppo[cur] = {"P": float(m.group(1)), "Q": float(m.group(2)),
                        "Deliv": float(m.group(3))}
            continue
        m = SDMA_ROW.search(line)
        if m:
            sdma[cur] = {"Q": float(m.group(2)), "Deliv": float(m.group(3))}
    return {"ckpt": name, "ppo": ppo, "sdma": sdma, "raw": out[-400:] if not ppo else ""}


def main():
    print(f"{'ckpt':<12} {'0dB Deliv':>9} {'0dB Q':>7} {'pass0':>5} "
          f"{'5dB Deliv':>9} {'5dB Q':>7} {'pass5':>5}", flush=True)
    sdma_ref = {}
    for name in CKPTS:
        res = scan_one(name)
        if "error" in res:
            print(f"{name:<12} ERROR: {res['error']}", flush=True)
            continue
        if not res["ppo"]:
            print(f"{name:<12} ERROR: no rows parsed\n{res['raw']}", flush=True)
            continue
        if res["sdma"]:
            sdma_ref = res["sdma"]  # SDMA 与 ckpt 无关，取最后一次
        line = f"{name:<12}"
        for snr in (0, 5):
            p = res["ppo"].get(snr)
            s = res["sdma"].get(snr)
            if p is None or s is None:
                line += f" {'n/a':>9} {'n/a':>7} {'?':>5}"
                continue
            ok = p["Deliv"] >= s["Deliv"] and p["Q"] <= 1.1 * s["Q"]
            line += f" {p['Deliv']:>9.3f} {p['Q']:>7.2f} {'PASS' if ok else 'fail':>5}"
        print(line, flush=True)
    if sdma_ref:
        print("\nSDMA reference:", flush=True)
        for snr in (0, 5):
            s = sdma_ref.get(snr)
            if s:
                print(f"  {snr}dB: Deliv={s['Deliv']:.3f} Q={s['Q']:.2f} "
                      f"(需 Deliv>={s['Deliv']:.3f}, Q<={1.1*s['Q']:.2f})", flush=True)


if __name__ == "__main__":
    main()
