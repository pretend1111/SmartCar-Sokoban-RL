"""对 verified_v4 里 ok 的 map, 尝试其余 9 个 seed 看哪些也能解.
输出: phase{N}_verified_all.json, 每张图带 all_seeds list (按解出来的速度排).
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.paths import MAPS_ROOT, PROJECT_ROOT
from smartcar_sokoban.solver.explorer import plan_exploration
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid


def _check_seed(args: dict) -> dict:
    map_path = args["map_path"]
    seed = args["seed"]
    time_limit = args["time_limit"]
    max_cost = args["max_cost"]
    push_min = args["push_min"]
    push_max = args["push_max"]

    cfg = GameConfig()
    cfg.control_mode = "discrete"
    engine = GameEngine(cfg, str(PROJECT_ROOT))
    devnull = io.StringIO()
    random.seed(seed)
    engine.reset(map_path)
    try:
        with contextlib.redirect_stdout(devnull):
            plan_exploration(engine)
    except Exception:
        return {"map": os.path.basename(map_path), "seed": seed, "ok": False}

    state = engine.get_state()
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(devnull):
            sol = solver.solve(max_cost=max_cost, time_limit=time_limit, strategy="auto")
    except Exception:
        return {"map": os.path.basename(map_path), "seed": seed, "ok": False}
    elapsed = time.perf_counter() - t0
    if sol is None:
        return {"map": os.path.basename(map_path), "seed": seed, "ok": False,
                "elapsed_s": round(elapsed, 2)}
    pushes = sum(1 for et,_,_,_ in sol if et == "box")
    bomb_pushes = sum(1 for et,_,_,_ in sol if et == "bomb")
    total = pushes + bomb_pushes
    if not (push_min <= total <= push_max):
        return {"map": os.path.basename(map_path), "seed": seed, "ok": False,
                "pushes": total, "elapsed_s": round(elapsed, 2)}
    return {"map": os.path.basename(map_path), "seed": seed, "ok": True,
            "pushes": total, "elapsed_s": round(elapsed, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="phase{N}_verified_v4.json")
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[0, 1, 7, 42, 100, 137, 200, 500, 999, 1234])
    p.add_argument("--time-limit", type=float, default=20.0)
    p.add_argument("--max-cost", type=int, default=200)
    p.add_argument("--push-min", type=int, default=3)
    p.add_argument("--push-max", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=12)
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    phase = manifest["phase"]
    phase_dir = MAPS_ROOT / f"phase{phase}"

    # 对所有 ok 的 map, 测全部 seeds
    targets_jobs = []
    for r in manifest["results"]:
        if r.get("status") != "ok":
            continue
        mp = str(phase_dir / r["map"])
        for s in args.seeds:
            targets_jobs.append({
                "map_path": mp, "seed": s,
                "time_limit": args.time_limit,
                "max_cost": args.max_cost,
                "push_min": args.push_min, "push_max": args.push_max,
            })

    print(f"[find_all_seeds] phase={phase} maps={sum(1 for r in manifest['results'] if r.get('status')=='ok')} "
          f"seeds={args.seeds} jobs={len(targets_jobs)}", flush=True)

    by_map: dict = {}  # map → list[(seed, elapsed)]
    t0 = time.time()
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(_check_seed, j): j for j in targets_jobs}
        for fut in concurrent.futures.as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"map": "?", "seed": -1, "ok": False, "err": str(e)}
            done += 1
            if r.get("ok"):
                by_map.setdefault(r["map"], []).append({
                    "seed": r["seed"], "pushes": r["pushes"],
                    "elapsed_s": r["elapsed_s"],
                })
            if done % 200 == 0:
                ela = time.time() - t0
                eta = ela / done * (len(targets_jobs) - done)
                print(f"  [{done}/{len(targets_jobs)}] maps_seen={len(by_map)} ETA={int(eta)}s",
                      flush=True)

    # 排序: 每 map 的 seeds 按 elapsed 升序 (秒解的优先)
    out_results = []
    for r in manifest["results"]:
        if r.get("status") != "ok":
            continue
        m = r["map"]
        seeds_info = sorted(by_map.get(m, []), key=lambda x: x["elapsed_s"])
        out_results.append({
            "map": m,
            "all_seeds": seeds_info,
            "n_solvable_seeds": len(seeds_info),
        })

    summary = {
        "phase": phase,
        "n_maps": len(out_results),
        "tested_seeds": args.seeds,
        "median_solvable_per_map": (
            sorted([r["n_solvable_seeds"] for r in out_results])[len(out_results)//2]
            if out_results else 0
        ),
        "min_solvable": min((r["n_solvable_seeds"] for r in out_results), default=0),
        "max_solvable": max((r["n_solvable_seeds"] for r in out_results), default=0),
        "results": out_results,
        "elapsed_s": round(time.time() - t0, 2),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in summary.items() if k != "results"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
