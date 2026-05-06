"""按 phase 1..N 顺序串行跑 verify_optimal.py。

每个 phase 用不同的 push 闸 (1: 3-12, 2: 4-15, 3: 6-20, 4: 10-30, 5: 12-35,
6: 15-40), 都用 18 worker, IDA* 60s 上限.

输出每个 phase 的 verified.json 到 assets/maps/phase{N}_verified.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

DEFAULT_PUSH_RANGES = {
    1: (3, 12),
    2: (4, 15),
    3: (6, 20),
    4: (10, 30),
    5: (12, 35),
    6: (15, 40),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phases", type=int, nargs="+", default=[5, 4, 3, 2, 1],
                   help="按此顺序串行跑 (默认 5→4→3→2→1)")
    p.add_argument("--ida-time", type=float, default=60.0)
    p.add_argument("--max-cost", type=int, default=200)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 137])
    p.add_argument("--num-workers", type=int, default=18)
    p.add_argument("--max-maps", type=int, default=0)
    p.add_argument("--push-ranges-json", default="",
                   help="JSON 字典覆盖默认 push 闸, 例如 '{\"1\":[2,10]}'")
    args = p.parse_args()

    push_ranges = dict(DEFAULT_PUSH_RANGES)
    if args.push_ranges_json:
        for k, v in json.loads(args.push_ranges_json).items():
            push_ranges[int(k)] = (int(v[0]), int(v[1]))

    out_dir = ROOT / "assets" / "maps"
    summary = []

    for phase in args.phases:
        if phase not in push_ranges:
            print(f"[skip] phase {phase} 无 push 闸配置")
            continue
        pmin, pmax = push_ranges[phase]
        out_path = out_dir / f"phase{phase}_verified.json"
        cmd = [
            PY,
            str(ROOT / "scripts" / "maps" / "verify_optimal.py"),
            "--phase", str(phase),
            "--seeds", *map(str, args.seeds),
            "--ida-time", str(args.ida_time),
            "--max-cost", str(args.max_cost),
            "--push-min", str(pmin),
            "--push-max", str(pmax),
            "--num-workers", str(args.num_workers),
            "--output", str(out_path),
        ]
        if args.max_maps > 0:
            cmd += ["--max-maps", str(args.max_maps)]
        print(f"\n[verify_all] phase={phase} pushes=[{pmin},{pmax}] → {out_path}",
              flush=True)
        t0 = time.time()
        ret = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = time.time() - t0
        if ret.returncode != 0:
            print(f"[verify_all] phase {phase} 失败 returncode={ret.returncode}")
            return ret.returncode
        # 读结果, 加到 summary
        try:
            with open(out_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            summary.append({
                "phase": phase,
                "n_input": data["n_input"],
                "n_passed": data["n_passed"],
                "pass_rate": data["pass_rate"],
                "median_pushes": data.get("median_pushes"),
                "elapsed_s": round(elapsed, 1),
            })
        except Exception as e:
            print(f"[verify_all] read summary failed: {e}")

    print("\n=== verify_all summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
