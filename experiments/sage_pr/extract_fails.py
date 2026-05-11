"""并行提取 V1 4-dir 重生成的失败地图清单."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _check_one(args):
    """检查 explore + solver. 用 v3 explorer (含拓扑配对)."""
    map_path, seed, phase = args
    from smartcar_sokoban.engine import GameEngine
    from smartcar_sokoban.solver.explorer import exploration_complete
    from smartcar_sokoban.solver.explorer_v3 import plan_exploration_v3
    from smartcar_sokoban.solver.pathfinder import pos_to_grid
    from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
    random.seed(seed)
    eng = GameEngine()
    try:
        eng.reset(map_path)
        with contextlib.redirect_stdout(io.StringIO()):
            plan_exploration_v3(eng, max_retries=15)
        state = eng.get_state()
        if not exploration_complete(state):
            return {"phase": phase, "map": map_path, "seed": seed, "reason": "explore_incomplete"}
        # check solver
        boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
        targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
        bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
        car = pos_to_grid(state.car_x, state.car_y)
        solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                moves = solver.solve(max_cost=300, time_limit=60.0, strategy="auto")
            except Exception:
                moves = None
        if not moves:
            return {"phase": phase, "map": map_path, "seed": seed, "reason": "solver_no_solution"}
        return None
    except Exception as e:
        return {"phase": phase, "map": map_path, "seed": seed, "reason": f"err:{type(e).__name__}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/sage_pr/v5_4dir_fails.json")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    from experiments.sage_pr.build_dataset_v3 import parse_phase456_seeds, list_phase_maps
    vmap = parse_phase456_seeds(os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json"))
    tasks = []
    for phase in [1, 2, 3, 4, 5, 6]:
        if phase in [1, 2, 3]:
            for m in list_phase_maps(phase):
                tasks.append((m, 0, phase))
        else:
            items_v = sorted([(k, v) for k, v in vmap.items() if f"phase{phase}/" in k])
            for m, seeds in items_v:
                for s in seeds[:5]:
                    tasks.append((m, s, phase))
    print(f"total tasks: {len(tasks)}, workers: {args.workers}")
    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        results = list(pool.imap_unordered(_check_one, tasks, chunksize=4))
    fails = [r for r in results if r is not None]
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.0f}s, total fails: {len(fails)}")
    import collections
    print(f"  by phase: {dict(collections.Counter(x['phase'] for x in fails))}")
    print(f"  by reason: {dict(collections.Counter(x['reason'] for x in fails))}")
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    with open(out_path, "w") as f:
        json.dump(fails, f, indent=2)
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
