"""P1.2 — 用 IDA* 验证地图，输出 verified manifest。

对每张图:
  1. 多 seed 测试 (默认 [7, 42, 137])
  2. plan_exploration → IDA*(strategy='ida', time_limit=N) 求最优
  3. 推数闸: 必须落在 [push_min, push_max]
  4. 至少 1 个 seed 通过即视为 verified, 写出第一个通过的 seed

输出 JSON:
  {
    "phase": N,
    "n_input": ..., "n_passed": ...,
    "push_min": ..., "push_max": ..., "ida_time_s": ...,
    "results": [
      {"map": "...", "status": "ok|push_too_low|push_too_high|ida_timeout",
       "verified_seed": int|None, "pushes": int|None,
       "walk_plus_push": int|None, "elapsed_s": float},
      ...
    ]
  }

用法:
  python scripts/maps/verify_optimal.py \
    --phase 6 --ida-time 60 --push-min 15 --push-max 40 \
    --seeds 7 42 137 --num-workers 18 \
    --output assets/maps/phase6_verified.json
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
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.paths import MAPS_ROOT, PROJECT_ROOT
from smartcar_sokoban.solver.explorer import plan_exploration
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid


def _verify_one(args: dict) -> dict:
    map_path = args["map_path"]
    seeds = list(args["seeds"])
    ida_time = args["ida_time"]
    max_cost = args["max_cost"]
    push_min = args["push_min"]
    push_max = args["push_max"]
    strategy = args.get("strategy", "ida")

    cfg = GameConfig()
    cfg.control_mode = "discrete"
    engine = GameEngine(cfg, str(PROJECT_ROOT))
    devnull = io.StringIO()

    last_status = "no_seed_passed"
    last_pushes = None
    last_walk = None
    last_elapsed = 0.0

    for seed in seeds:
        t0 = time.perf_counter()
        random.seed(seed)
        engine.reset(map_path)
        try:
            with contextlib.redirect_stdout(devnull):
                explore = plan_exploration(engine)
        except Exception as e:
            last_status = f"explore_error:{e}"
            continue
        state = engine.get_state()
        boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
        targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
        bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
        car = pos_to_grid(state.car_x, state.car_y)
        solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
        try:
            with contextlib.redirect_stdout(devnull):
                sol = solver.solve(max_cost=max_cost, time_limit=ida_time,
                                   strategy=strategy)
        except Exception as e:
            last_status = f"ida_error:{e}"
            continue
        elapsed = time.perf_counter() - t0
        last_elapsed = elapsed

        if sol is None:
            last_status = "ida_timeout"
            continue

        pushes = sum(1 for et, _, _, _ in sol if et == "box")
        bomb_pushes = sum(1 for et, _, _, _ in sol if et == "bomb")
        # 算总推数（含炸弹）作为闸；同时单独记录 box pushes
        total_pushes = pushes + bomb_pushes
        walk = sum(wc + 1 for _, _, _, wc in sol)

        if total_pushes < push_min:
            last_status = f"push_too_low:{total_pushes}"
            last_pushes = total_pushes
            last_walk = walk
            continue
        if total_pushes > push_max:
            last_status = f"push_too_high:{total_pushes}"
            last_pushes = total_pushes
            last_walk = walk
            continue

        return {
            "map": os.path.basename(map_path),
            "status": "ok",
            "verified_seed": seed,
            "pushes": total_pushes,
            "box_pushes": pushes,
            "bomb_pushes": bomb_pushes,
            "walk_plus_push": walk,
            "explore_steps": len(explore),
            "total_low_steps": len(explore) + walk,
            "elapsed_s": round(elapsed, 2),
        }

    return {
        "map": os.path.basename(map_path),
        "status": last_status,
        "verified_seed": None,
        "pushes": last_pushes,
        "walk_plus_push": last_walk,
        "elapsed_s": round(last_elapsed, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, required=True)
    p.add_argument("--max-maps", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 137])
    p.add_argument("--ida-time", type=float, default=60.0)
    p.add_argument("--max-cost", type=int, default=200)
    p.add_argument("--push-min", type=int, default=15)
    p.add_argument("--push-max", type=int, default=40)
    p.add_argument("--strategy", choices=["ida", "auto", "best_first"], default="ida",
                   help="solver strategy. ida=strict optimal (slow), auto/best_first=1.5x OPT (fast)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--filter-pattern", default="*.txt",
                   help="glob 过滤地图文件名 (默认 *.txt)")
    args = p.parse_args()

    phase_dir = MAPS_ROOT / f"phase{args.phase}"
    map_list = sorted(glob.glob(str(phase_dir / args.filter_pattern)))
    if args.max_maps > 0:
        map_list = map_list[:args.max_maps]
    if not map_list:
        raise SystemExit(f"no maps in {phase_dir}")

    workers = (args.num_workers if args.num_workers > 0
               else max(1, (os.cpu_count() or 4) - 2))

    print(f"[verify] phase={args.phase} maps={len(map_list)} workers={workers} "
          f"seeds={args.seeds} ida={args.ida_time}s pushes=[{args.push_min},{args.push_max}]",
          flush=True)

    jobs = [
        {
            "map_path": mp,
            "seeds": args.seeds,
            "ida_time": args.ida_time,
            "max_cost": args.max_cost,
            "push_min": args.push_min,
            "push_max": args.push_max,
            "strategy": args.strategy,
        }
        for mp in map_list
    ]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results: List[dict] = []
    t0 = time.time()
    done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_verify_one, j): j for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"map": "?", "status": f"crash:{e}"}
            results.append(r)
            done += 1
            elapsed = time.time() - t0
            eta = (elapsed / done * (len(jobs) - done)) if done else 0
            if done % 25 == 0 or done == len(jobs):
                pass_n = sum(1 for x in results if x.get("status") == "ok")
                print(f"[{done}/{len(jobs)}] passed={pass_n} "
                      f"last={r.get('map','?')} status={r.get('status','?')} "
                      f"pushes={r.get('pushes','-')} t={r.get('elapsed_s','-')}s | "
                      f"ETA={int(eta)}s", flush=True)

    ok = [r for r in results if r.get("status") == "ok"]
    pushes_all = [r["pushes"] for r in ok]
    summary = {
        "phase": args.phase,
        "n_input": len(map_list),
        "n_passed": len(ok),
        "n_failed": len(map_list) - len(ok),
        "pass_rate": round(len(ok) / max(1, len(map_list)), 4),
        "ida_time_s": args.ida_time,
        "push_min": args.push_min,
        "push_max": args.push_max,
        "seeds": args.seeds,
        "median_pushes": (sorted(pushes_all)[len(pushes_all) // 2]
                          if pushes_all else None),
        "min_pushes": min(pushes_all, default=None),
        "max_pushes": max(pushes_all, default=None),
        "results": sorted(results, key=lambda r: r.get("map", "")),
        "elapsed_s": round(time.time() - t0, 2),
        "workers": workers,
    }

    # status 直方图
    from collections import Counter
    status_hist = Counter(r.get("status", "?") for r in results)
    summary["status_histogram"] = dict(status_hist)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    summary_lite = {k: v for k, v in summary.items() if k != "results"}
    print(json.dumps(summary_lite, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
