"""把 verified.json 里 push_too_low 的 maps 用宽松的 push_min 救回.

输入: 旧 verified.json (含 push_too_low: pushes 数据)
逻辑:
  对每个 push_too_low entry, 重跑 [7,42,137] seeds, 用新 push_min 判断.
  把通过的 entries 写回 verified.json (合并到 ok 列表).
输出: 新 verified.json (覆盖或另存)

用法:
  python scripts/maps/recover_low_push_maps.py \
    --input assets/maps/phase5_verified.json \
    --output assets/maps/phase5_verified_relaxed.json \
    --new-push-min 3 \
    --num-workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.paths import MAPS_ROOT, PROJECT_ROOT
from smartcar_sokoban.solver.explorer import plan_exploration
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid


def _recover_one(args: dict) -> dict:
    map_path = args["map_path"]
    seeds = args["seeds"]
    ida_time = args["ida_time"]
    max_cost = args["max_cost"]
    push_min = args["push_min"]
    push_max = args["push_max"]

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
                                   strategy="ida")
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--new-push-min", type=int, required=True)
    p.add_argument("--push-max", type=int, default=999)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 137])
    p.add_argument("--ida-time", type=float, default=60.0)
    p.add_argument("--max-cost", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    phase = manifest.get("phase")
    phase_dir = MAPS_ROOT / f"phase{phase}"

    # 找 push_too_low entries (其中 pushes 落在 [new_push_min, push_max] 区间的)
    targets = []
    keep_existing = []
    for r in manifest["results"]:
        st = r.get("status", "")
        if st.startswith("push_too_low") and r.get("pushes") is not None:
            if args.new_push_min <= r["pushes"] <= args.push_max:
                targets.append(r["map"])
                continue
        keep_existing.append(r)  # 不动

    print(f"[recover] phase={phase} 原 entries={len(manifest['results'])} "
          f"待救回 push_too_low {len(targets)} (push_min {args.new_push_min}-{args.push_max})",
          flush=True)

    # 重跑这 N 张 maps
    jobs = [{
        "map_path": str(phase_dir / m),
        "seeds": args.seeds,
        "ida_time": args.ida_time,
        "max_cost": args.max_cost,
        "push_min": args.new_push_min,
        "push_max": args.push_max,
    } for m in targets]

    new_results = []
    t0 = time.time()
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(_recover_one, j): j for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"map": "?", "status": f"crash:{e}"}
            new_results.append(r)
            done += 1
            if done % 25 == 0 or done == len(jobs):
                ela = time.time() - t0
                eta = (ela / done * (len(jobs) - done)) if done else 0
                ok = sum(1 for x in new_results if x.get("status") == "ok")
                print(f"  [{done}/{len(jobs)}] recovered ok={ok} ETA={int(eta)}s", flush=True)

    # 合并
    all_results = keep_existing + new_results
    all_results.sort(key=lambda r: r.get("map", ""))
    ok = [r for r in all_results if r.get("status") == "ok"]
    pushes_all = [r["pushes"] for r in ok]

    summary = {
        "phase": phase,
        "n_input": manifest["n_input"],
        "n_passed": len(ok),
        "n_failed": manifest["n_input"] - len(ok),
        "pass_rate": round(len(ok) / max(1, manifest["n_input"]), 4),
        "ida_time_s": args.ida_time,
        "push_min": args.new_push_min,
        "push_max": args.push_max,
        "seeds": args.seeds,
        "median_pushes": (sorted(pushes_all)[len(pushes_all) // 2]
                          if pushes_all else None),
        "min_pushes": min(pushes_all, default=None),
        "max_pushes": max(pushes_all, default=None),
        "results": all_results,
        "elapsed_s": round(time.time() - t0, 2),
        "workers": args.num_workers,
        "recovered_from": args.input,
    }
    from collections import Counter
    summary["status_histogram"] = dict(Counter(r.get("status", "?") for r in all_results))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"[recover] saved → {args.output}")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
