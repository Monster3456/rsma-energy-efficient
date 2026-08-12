"""v2_powertax3 持续监控脚本（后台运行）

每 5 分钟抓取训练日志趋势，写入 logs/v2_powertax3_monitor.log；
检测异常（λ 触顶、delivery 跌破阈值、进程死亡、训练完成）时追加 ALERT 标记。
"""
import os
import re
import subprocess
import sys
import time
from datetime import datetime

LOG = "/Users/libo/PycharmProjects/rsma-energy-efficient/logs/v2_powertax3_train.log"
MON = "/Users/libo/PycharmProjects/rsma-energy-efficient/logs/v2_powertax3_monitor.log"
EXP_DIR = "/Users/libo/PycharmProjects/rsma-energy-efficient/logs/v2_powertax3"

DELIVERY_ALERT = 0.80   # AvgDeliv 低于此值告警
LAMBDA_ALERT = 95.0     # λ 逼近上限 100 告警
INTERVAL = 300          # 5 分钟


def tail_lines(path, n=50):
    try:
        with open(path, "r") as f:
            return f.readlines()[-n:]
    except FileNotFoundError:
        return []


def parse_step(line):
    m = re.search(r"Episodes=(\d+)", line)
    if not m:
        return None
    ep = int(m.group(1))
    lam = re.search(r"Lambda=([\d.]+)", line)
    deliv = re.search(r"AvgDeliv=([\d.]+)", line)
    fps = re.search(r"FPS=([\d.]+)", line)
    streak = re.search(r"streak=\[([\d, ]+)\]", line)
    return {
        "ep": ep,
        "lambda": float(lam.group(1)) if lam else None,
        "deliv": float(deliv.group(1)) if deliv else None,
        "fps": float(fps.group(1)) if fps else None,
        "streak": [int(x) for x in streak.group(1).split(",")] if streak else None,
    }


def is_alive():
    r = subprocess.run(
        ["ps", "aux"], capture_output=True, text=True
    )
    return "v2_powertax3" in r.stdout and "train.py" in r.stdout


def main():
    last_ep = -1
    with open(MON, "a") as mon:
        mon.write(f"=== monitor started {datetime.now().isoformat()} ===\n")
        while True:
            lines = tail_lines(LOG, 50)
            steps = [parse_step(l) for l in lines if "Step" in l]
            steps = [s for s in steps if s is not None]
            if not steps:
                time.sleep(INTERVAL)
                continue
            s = steps[-1]
            now = datetime.now().strftime("%H:%M:%S")
            alive = is_alive()
            finished = any("Training complete" in l for l in lines)
            alerts = []

            if not alive:
                alerts.append("PROCESS_DIED")
            if finished:
                alerts.append("TRAINING_COMPLETE")
            if s["deliv"] is not None and s["deliv"] < DELIVERY_ALERT:
                alerts.append(f"DELIVERY_LOW({s['deliv']:.3f})")
            if s["lambda"] is not None and s["lambda"] > LAMBDA_ALERT:
                alerts.append(f"LAMBDA_HIGH({s['lambda']:.1f})")
            if s["ep"] > last_ep:
                last_ep = s["ep"]
                line = (f"{now} ep={s['ep']} λ={s['lambda']:.2f} "
                        f"deliv={s['deliv']:.3f} fps={s['fps']:.0f} "
                        f"streak={s['streak']} {' '.join(alerts)}"
                        if s["streak"] else
                        f"{now} ep={s['ep']} λ={s['lambda']:.2f} "
                        f"deliv={s['deliv']:.3f} fps={s['fps']:.0f} {' '.join(alerts)}")
                print(line, flush=True)
                mon.write(line + "\n")
                mon.flush()

            if finished or (alerts and "PROCESS_DIED" in alerts and not alive):
                mon.write(f"=== monitor stopping {datetime.now().isoformat()} ===\n")
                mon.flush()
                sys.exit(0)

            time.sleep(INTERVAL)


if __name__ == "__main__":
    main()