"""轻量资源监控 — 每 N 秒打印一行 t=ISO cpu=__% gpu=__% vram=__/__GB ram=__GB

写到 .agent/monitor/<tag>_<ts>.log，方便 ralph loop 用 grep / tail 巡检。
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None


def gpu_query():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip().splitlines()
        if not out:
            return None
        util, used, total, temp = [s.strip() for s in out[0].split(",")]
        return {
            "gpu_util": float(util),
            "vram_used_gb": float(used) / 1024.0,
            "vram_total_gb": float(total) / 1024.0,
            "gpu_temp_c": float(temp),
        }
    except Exception as e:
        return {"error": str(e)}


def cpu_ram():
    if psutil is None:
        return {"cpu_pct": -1, "ram_used_gb": -1, "ram_total_gb": -1}
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "ram_used_gb": psutil.virtual_memory().used / (1024**3),
        "ram_total_gb": psutil.virtual_memory().total / (1024**3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="run")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--duration", type=float, default=0.0,
                   help="0 = run forever")
    p.add_argument("--out-dir", default=".agent/monitor")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"{args.tag}_{ts}.log"

    print(f"[monitor] tag={args.tag} log={log_path}")
    if psutil is None:
        print("[monitor] psutil not installed → cpu/ram = -1; pip install psutil")

    if psutil is not None:
        psutil.cpu_percent(interval=None)  # prime

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("t cpu_pct gpu_util vram_used_gb vram_total_gb ram_used_gb ram_total_gb gpu_temp_c\n")
        fh.flush()
        while True:
            now = dt.datetime.now().isoformat(timespec="seconds")
            cr = cpu_ram()
            g = gpu_query() or {}
            line = (
                f"{now} cpu={cr['cpu_pct']:.0f}% "
                f"gpu={g.get('gpu_util', float('nan')):.0f}% "
                f"vram={g.get('vram_used_gb', 0):.1f}/{g.get('vram_total_gb', 0):.1f}GB "
                f"ram={cr['ram_used_gb']:.1f}/{cr['ram_total_gb']:.1f}GB "
                f"gpu_t={g.get('gpu_temp_c', float('nan')):.0f}C"
            )
            print(line, flush=True)
            fh.write(line + "\n")
            fh.flush()
            if args.duration > 0 and time.time() - t0 > args.duration:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
