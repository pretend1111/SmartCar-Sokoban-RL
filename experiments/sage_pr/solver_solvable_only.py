"""Check pure solvability: only count if solver returns non-None moves.
Skip apply_solver_move to isolate solver capability from replay bugs.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from experiments.sage_pr.build_dataset_v3 import (
    parse_phase456_seeds, list_phase_maps,
)


def solvable(map_path, seed, time_limit, strategy):
    import random
    random.seed(seed)
    eng = GameEngine()
    state = eng.reset(map_path)
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            moves = solver.solve(max_cost=300, time_limit=time_limit, strategy=strategy)
        except Exception:
            return False
    return moves is not None


def _worker(args):
    return solvable(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", type=int, nargs="+", default=[4])
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--max-maps", type=int, default=100)
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--strategy", default="auto", choices=["auto", "best_first", "ida"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    verified = None
    if args.use_verified_seeds:
        verified = parse_phase456_seeds(
            os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
        )

    results = []
    for ph in args.phases:
        maps = list_phase_maps(ph)[:args.max_maps]
        tasks = []
        for map_path in maps:
            ms = (verified.get(map_path, [0])[:max(1, len(seeds))] if verified else seeds)
            for seed in ms:
                tasks.append((map_path, seed, args.time_limit, args.strategy))
        t0 = time.perf_counter()
        with mp.Pool(args.workers) as pool:
            wins = list(pool.imap_unordered(_worker, tasks))
        n_total = len(tasks)
        n_won = sum(1 for w in wins if w)
        elapsed = time.perf_counter() - t0
        r = {"phase": ph, "n_total": n_total, "n_won": n_won, "win_rate": n_won / max(n_total, 1)}
        print(f"phase {ph}: {r['win_rate']*100:.1f}% ({n_won}/{n_total}); {elapsed:.0f}s")
        results.append(r)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
