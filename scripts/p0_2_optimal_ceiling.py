"""P0.2 — 对 phase N 图集跑严格 IDA* 最优推数 (作为 BC 天花板).

对每张图：
  1. 加载、按 manifest seed (没有就用 base_seed)
  2. plan_exploration 取得已观测状态
  3. MultiBoxSolver(strategy='ida', time_limit=...) 求严格最优
  4. 输出 push_count, total_walk, status

并行：每张图一个进程, ProcessPoolExecutor.

用法：
  python scripts/p0_2_optimal_ceiling.py --phase 6 --max-maps 50 \
    --time-limit 180 --output .agent/baseline/phase6_optimal.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import glob
import io
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.paths import MAPS_ROOT, PROJECT_ROOT
from smartcar_sokoban.solver.explorer import plan_exploration
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid


def _solve_one(args: dict) -> dict:
    map_path = args["map_path"]
    seed = args["seed"]
    time_limit = args["time_limit"]
    max_cost = args["max_cost"]

    cfg = GameConfig()
    cfg.control_mode = "discrete"
    engine = GameEngine(cfg, str(PROJECT_ROOT))
    random.seed(seed)
    engine.reset(map_path)

    devnull = io.StringIO()
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(devnull):
            explore = plan_exploration(engine)
    except Exception as e:
        return {"map": os.path.basename(map_path), "seed": seed,
                "status": "explore_error", "error": str(e),
                "elapsed_s": time.perf_counter() - t0}

    state = engine.get_state()
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)

    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    try:
        with contextlib.redirect_stdout(devnull):
            sol = solver.solve(max_cost=max_cost, time_limit=time_limit,
                               strategy="ida")
    except Exception as e:
        return {"map": os.path.basename(map_path), "seed": seed,
                "status": "ida_error", "error": str(e),
                "elapsed_s": time.perf_counter() - t0}
    elapsed = time.perf_counter() - t0

    if sol is None:
        return {"map": os.path.basename(map_path), "seed": seed,
                "status": "ida_timeout_or_unsolved",
                "explore_steps": len(explore),
                "elapsed_s": round(elapsed, 2)}

    pushes = len(sol)
    walk = sum(wc + 1 for _, _, _, wc in sol)
    return {
        "map": os.path.basename(map_path),
        "seed": seed,
        "status": "ok",
        "pushes": pushes,
        "walk_plus_push": walk,
        "explore_steps": len(explore),
        "total_low_steps": len(explore) + walk,
        "elapsed_s": round(elapsed, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, required=True)
    p.add_argument("--max-maps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-limit", type=float, default=180.0)
    p.add_argument("--max-cost", type=int, default=200)
    p.add_argument("--output", required=True)
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 = (cpu_count-2)")
    args = p.parse_args()

    phase_dir = MAPS_ROOT / f"phase{args.phase}"
    map_list = sorted(glob.glob(str(phase_dir / "*.txt")))
    if args.max_maps > 0:
        map_list = map_list[:args.max_maps]
    if not map_list:
        raise SystemExit(f"no maps in {phase_dir}")

    workers = args.num_workers if args.num_workers > 0 else max(1, (os.cpu_count() or 4) - 2)

    print(f"[p0.2] phase={args.phase} maps={len(map_list)} workers={workers} "
          f"time_limit={args.time_limit}s max_cost={args.max_cost}",
          flush=True)

    jobs = [
        {"map_path": mp, "seed": args.seed,
         "time_limit": args.time_limit, "max_cost": args.max_cost}
        for mp in map_list
    ]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results: List[dict] = []
    t0 = time.time()
    done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_solve_one, j): j for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"map": "?", "status": "worker_crash", "error": str(e)}
            results.append(r)
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (len(jobs) - done)
            line = (f"[{done}/{len(jobs)}] {r.get('map','?'):20s} "
                    f"status={r.get('status','?'):24s} "
                    f"pushes={r.get('pushes','-'):>4} "
                    f"walk={r.get('walk_plus_push','-'):>4} "
                    f"t={r.get('elapsed_s','-'):>6}s | "
                    f"ETA={int(eta)}s")
            print(line, flush=True)

    # 写出
    ok = [r for r in results if r.get("status") == "ok"]
    summary = {
        "phase": args.phase,
        "n_maps": len(map_list),
        "n_solved": len(ok),
        "n_unsolved": len(map_list) - len(ok),
        "time_limit_s": args.time_limit,
        "median_pushes": (sorted(r["pushes"] for r in ok)[len(ok) // 2]
                          if ok else None),
        "median_walk_push": (sorted(r["walk_plus_push"] for r in ok)[len(ok) // 2]
                             if ok else None),
        "median_total_low": (sorted(r["total_low_steps"] for r in ok)[len(ok) // 2]
                             if ok else None),
        "max_pushes": max((r["pushes"] for r in ok), default=None),
        "max_walk_push": max((r["walk_plus_push"] for r in ok), default=None),
        "results": sorted(results, key=lambda r: r.get("map", "")),
        "elapsed_s": round(time.time() - t0, 2),
        "workers": workers,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
