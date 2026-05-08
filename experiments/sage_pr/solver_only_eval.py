"""纯求解器作为 agent: 从 start state 调 solver.solve() → 重放全程.

成本: 每图 0.5-15s (取决于难度).
Win rate 接近 100% 当 verified maps 上 solver 实际解出.

这不是神经模型的核心目标 (神经模型部署到 OpenART), 但可以作为 win-rate 上限验证.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from experiments.sage_pr.build_dataset_v3 import (
    apply_solver_move, parse_phase456_seeds, list_phase_maps,
)


def solver_only_episode(map_path: str, seed: int,
                         time_limit: float = 30.0,
                         strategy: str = "best_first") -> bool:
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
        moves = solver.solve(max_cost=300, time_limit=time_limit, strategy=strategy)
    if not moves:
        return False

    # 重新 reset 跟同 seed
    random.seed(seed)
    eng.reset(map_path)
    for m in moves:
        if not apply_solver_move(eng, m):
            return False
        if eng.get_state().won:
            return True
    return eng.get_state().won


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--max-maps", type=int, default=100)
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--strategy", default="best_first", choices=["auto", "best_first", "ida"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    verified_map = None
    if args.use_verified_seeds:
        verified_map = parse_phase456_seeds(
            os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
        )

    print(f"strategy={args.strategy}, time_limit={args.time_limit}")

    results = []
    for ph in args.phases:
        maps = list_phase_maps(ph)[:args.max_maps]
        n_total = 0
        n_won = 0
        t0 = time.perf_counter()
        for map_path in maps:
            ms = (verified_map.get(map_path, [0])[:max(1, len(seeds))]
                  if verified_map else seeds)
            for seed in ms:
                won = solver_only_episode(map_path, seed,
                                           time_limit=args.time_limit,
                                           strategy=args.strategy)
                n_total += 1
                if won: n_won += 1
        elapsed = time.perf_counter() - t0
        r = {
            "phase": ph,
            "n_total": n_total,
            "n_won": n_won,
            "win_rate": n_won / max(n_total, 1),
        }
        print(f"phase {ph}: {r['win_rate']*100:.2f}% ({n_won}/{n_total}); {elapsed:.0f}s")
        results.append(r)

    print("\n=== Summary ===")
    for r in results:
        print(f"phase {r['phase']}: {r['win_rate']*100:.2f}% ({r['n_won']}/{r['n_total']})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
