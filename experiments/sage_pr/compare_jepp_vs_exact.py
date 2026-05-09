"""对比 JEPP (Level B belief-IDA*) vs Exact (god-mode IDA* + plan_exploration) 步数.

对每张图分别跑两种 teacher, 测算:
  - JEPP: 总 macro 步数 (push + inspect)
  - Exact: 总 macro 步数 (explore_actions 拆分到 inspect 等价个数 + push 步数)

并计算 success rate 与平均步数差.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.explorer import plan_exploration
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from experiments.sage_pr.belief_ida_solver import belief_ida_solve
from experiments.sage_pr.build_dataset_v3 import (
    parse_phase456_seeds, list_phase_maps, apply_solver_move,
)


def run_jepp(map_path: str, seed: int, time_limit: float = 8.0
             ) -> Dict:
    """跑 belief-IDA* JEPP, 返回 (total_macro, n_inspect, n_push, won, status)."""
    t0 = time.perf_counter()
    samples, status = belief_ida_solve(map_path, seed,
                                         ida_time_limit=time_limit, step_limit=120)
    elapsed = time.perf_counter() - t0
    if not samples or status != "ok":
        return {"won": False, "macro": 0, "inspect": 0, "push": 0,
                "elapsed": elapsed, "status": status}
    n_inspect = sum(1 for s in samples if s["type"] == "inspect")
    n_push = sum(1 for s in samples if s["type"] == "push")
    return {"won": True, "macro": len(samples), "inspect": n_inspect,
            "push": n_push, "elapsed": elapsed, "status": status}


def run_exact(map_path: str, seed: int, solver_time: float = 30.0
              ) -> Dict:
    """跑 god-mode IDA* + plan_exploration, 返回 step counts.

    跟 JEPP 不同, exact 把探索全部拆成低层 actions (没"inspect macro"概念).
    为对比公平, 把 exact 的"inspect 阶段"近似计为 N_unidentified entities 次 macro,
    push 阶段 = solver moves 数.
    """
    random.seed(seed)
    eng = GameEngine()
    state = eng.reset(map_path)
    n_unid_init = (len(state.boxes) - len(state.seen_box_ids)) + \
                  (len(state.targets) - len(state.seen_target_ids))

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        explore_low_actions = plan_exploration(eng)

    state = eng.get_state()
    n_unid_after_explore = (len(state.boxes) - len(state.seen_box_ids)) + \
                           (len(state.targets) - len(state.seen_target_ids))
    # 排除推理推出的 = unid_init - unid_after_explore - len(explored_entities)
    # 估算 inspect macro 数 = 实际怼到的 entity 数
    inspect_macro = (n_unid_init - n_unid_after_explore) - \
                    max(0, n_unid_after_explore - 0)  # = identified 数 - 排除推理数
    inspect_macro = max(0, inspect_macro)
    # 简化: inspect_macro = 探索时实际怼到的 entity 数 ≈ N_unid_init - 1 (排除推理省 1)
    inspect_macro = max(0, n_unid_init - 1)

    # Phase 2: solver
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            moves = solver.solve(max_cost=300, time_limit=solver_time, strategy="auto")
        except Exception:
            moves = None

    elapsed = time.perf_counter() - t0
    if not moves:
        return {"won": False, "macro": 0, "inspect": inspect_macro,
                "push": 0, "explore_low": len(explore_low_actions),
                "elapsed": elapsed, "status": "solver_no_solution"}

    n_push = len(moves)

    # 真实跑通确认
    random.seed(seed)
    eng.reset(map_path)
    with contextlib.redirect_stdout(io.StringIO()):
        plan_exploration(eng)
    for m in moves:
        if not apply_solver_move(eng, m):
            return {"won": False, "macro": inspect_macro + n_push,
                    "inspect": inspect_macro, "push": n_push,
                    "explore_low": len(explore_low_actions),
                    "elapsed": elapsed, "status": "apply_fail"}

    won = eng.get_state().won
    return {
        "won": won, "macro": inspect_macro + n_push,
        "inspect": inspect_macro, "push": n_push,
        "explore_low": len(explore_low_actions),
        "elapsed": elapsed,
        "status": "ok" if won else "did_not_win",
    }


def compare_phase(phase: int, n_maps: int = 20, jepp_time: float = 8.0):
    """对比 phase 上 n_maps 张图的 JEPP vs exact 步数."""
    vmap = parse_phase456_seeds(
        os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
    )
    items = sorted([(k, v) for k, v in vmap.items() if f"phase{phase}/" in k])[:n_maps]

    jepp_wins = 0
    exact_wins = 0
    both_won_jepp_macro = []
    both_won_exact_macro = []

    print(f"\n=== Phase {phase} compare ({len(items)} maps) ===")
    for map_path, seeds in items:
        seed = seeds[0]
        full = os.path.join(ROOT, map_path)
        if not os.path.exists(full):
            continue

        j = run_jepp(map_path, seed, time_limit=jepp_time)
        e = run_exact(map_path, seed)

        if j["won"]:
            jepp_wins += 1
        if e["won"]:
            exact_wins += 1

        marker = ""
        if j["won"] and e["won"]:
            both_won_jepp_macro.append(j["macro"])
            both_won_exact_macro.append(e["macro"])
            diff = j["macro"] - e["macro"]
            marker = f"  Δ={diff:+d}"

        print(f"  {os.path.basename(map_path)} seed={seed}: "
              f"JEPP {'✓' if j['won'] else '✗'} {j['macro']}m {j['elapsed']:.1f}s | "
              f"exact {'✓' if e['won'] else '✗'} {e['macro']}m {e['elapsed']:.1f}s{marker}")

    print(f"\n  Win rate: JEPP {jepp_wins}/{len(items)}, exact {exact_wins}/{len(items)}")
    if both_won_jepp_macro:
        avg_j = sum(both_won_jepp_macro) / len(both_won_jepp_macro)
        avg_e = sum(both_won_exact_macro) / len(both_won_exact_macro)
        diff_avg = avg_j - avg_e
        print(f"  Avg macro (both won, n={len(both_won_jepp_macro)}): "
              f"JEPP {avg_j:.1f}, exact {avg_e:.1f}, Δ={diff_avg:+.1f}")
        # 配对差
        diffs = [j - e for j, e in zip(both_won_jepp_macro, both_won_exact_macro)]
        better = sum(1 for d in diffs if d < 0)
        worse = sum(1 for d in diffs if d > 0)
        equal = sum(1 for d in diffs if d == 0)
        print(f"  JEPP 比 exact 更短的图: {better}, 更长: {worse}, 持平: {equal}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--n-maps", type=int, default=15)
    parser.add_argument("--jepp-time", type=float, default=8.0)
    args = parser.parse_args()
    for ph in args.phases:
        compare_phase(ph, n_maps=args.n_maps, jepp_time=args.jepp_time)


if __name__ == "__main__":
    main()
