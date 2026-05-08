"""Hybrid v2: 模型 + 求解器 fallback 改进版.

策略:
    1. 调求解器从 START state, 拿全程 plan.
    2. Replay plan via apply_solver_move. Win!

对 phase 5 这种 solver 100% 可解的 phase, 直接 solver-only 完美.

注: 这是 win-rate-优先 推理. 实际部署仍用 model + rollout search.
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
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import (
    generate_candidates, candidates_legality_mask,
)
from smartcar_sokoban.symbolic.cand_features import encode_candidates
from smartcar_sokoban.symbolic.grid_tensor import (
    build_grid_tensor, build_global_features,
)
from experiments.sage_pr.model import build_default_model
from experiments.sage_pr.build_dataset_v3 import (
    apply_solver_move, parse_phase456_seeds, list_phase_maps,
    match_move_to_candidate,
)
from experiments.sage_pr.evaluate_sage_pr import (
    candidate_to_solver_move, _state_signature,
)
from experiments.sage_pr.rollout_search_eval import rollout_search_step


def eng_belief(eng, fully_observed=True):
    return BeliefState.from_engine_state(eng.get_state(), fully_observed=fully_observed)


def hybrid_v2_episode(model, device, map_path, seed,
                      *, step_limit=60,
                      stuck_threshold=4,
                      solver_time_limit=10.0,
                      beam_width=4, lookahead=12,
                      fully_observed=True):
    """先 rollout search, 卡住超过 stuck_threshold 步无进度 → 切 solver 全程接管."""
    import random
    import copy as _copy
    random.seed(seed)
    eng = GameEngine()
    eng.reset(map_path)
    visited = set()
    visited.add(_state_signature(eng.get_state()))
    inf_total = 0.0
    inf_calls = 0
    n_solver = 0

    no_progress = 0
    last_n_box = len(eng.get_state().boxes)
    using_solver = False

    for step in range(step_limit):
        s = eng.get_state()
        if s.won:
            return True, step, inf_total / max(inf_calls, 1), n_solver

        # 检查是否切换到 solver
        if not using_solver and (no_progress >= stuck_threshold):
            # rollout search 卡住 → solver 全程
            with contextlib.redirect_stdout(io.StringIO()):
                boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in s.boxes]
                targets = {t.num_id: pos_to_grid(t.x, t.y) for t in s.targets}
                bombs = [pos_to_grid(b.x, b.y) for b in s.bombs]
                car = pos_to_grid(s.car_x, s.car_y)
                solver = MultiBoxSolver(s.grid, car, boxes, targets, bombs)
                try:
                    moves = solver.solve(max_cost=300, time_limit=solver_time_limit, strategy="auto")
                except Exception:
                    moves = None
            if moves:
                using_solver = True
                n_solver += 1
                # apply ALL moves
                for mv in moves:
                    if not apply_solver_move(eng, mv):
                        return False, step, inf_total / max(inf_calls, 1), n_solver
                    if eng.get_state().won:
                        return True, step, inf_total / max(inf_calls, 1), n_solver
                continue
            # solver fail → 退回继续 rollout

        t0 = time.perf_counter()
        result = rollout_search_step(model, device, eng, visited,
                                       beam_width=beam_width, lookahead=lookahead,
                                       fully_observed=fully_observed)
        inf_total += time.perf_counter() - t0
        inf_calls += 1

        if result is None:
            no_progress += 1
            if no_progress >= stuck_threshold:
                continue
            else:
                return False, step, inf_total / max(inf_calls, 1), n_solver
        idx, cand = result

        bs = eng_belief(eng, fully_observed)
        move = candidate_to_solver_move(cand, bs)
        if move is None:
            return False, step, inf_total / max(inf_calls, 1), n_solver

        n_box_before = len(eng.get_state().boxes)
        if not apply_solver_move(eng, move):
            return False, step, inf_total / max(inf_calls, 1), n_solver
        n_box_after = len(eng.get_state().boxes)

        if n_box_after < n_box_before:
            no_progress = 0
        else:
            no_progress += 1

        visited.add(_state_signature(eng.get_state()))

    return eng.get_state().won, step_limit, inf_total / max(inf_calls, 1), n_solver


def _worker(args):
    map_path, seed, ckpt, step_limit, stuck, solver_time, beam, lookahead = args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=device)
    model = build_default_model().to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return hybrid_v2_episode(
        model, device, map_path, seed,
        step_limit=step_limit, stuck_threshold=stuck,
        solver_time_limit=solver_time,
        beam_width=beam, lookahead=lookahead,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--max-maps", type=int, default=100)
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--stuck", type=int, default=3)
    parser.add_argument("--solver-time-limit", type=float, default=10.0)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--lookahead", type=int, default=12)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device)
    model = build_default_model().to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"loaded {args.ckpt}, val_acc={ck.get('val_acc', '?'):.3f}")
    print(f"stuck={args.stuck}, solver_tl={args.solver_time_limit}, "
          f"beam={args.beam}, lookahead={args.lookahead}")

    verified_map = None
    if args.use_verified_seeds:
        verified_map = parse_phase456_seeds(
            os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
        )

    results = []
    for ph in args.phases:
        maps = list_phase_maps(ph)[:args.max_maps]
        n_total = 0
        n_won = 0
        total_solver = 0
        t0 = time.perf_counter()
        for map_path in maps:
            ms = (verified_map.get(map_path, [0])[:max(1, len(seeds))]
                  if verified_map else seeds)
            for seed in ms:
                won, steps, avg_inf, n_solver = hybrid_v2_episode(
                    model, device, map_path, seed,
                    stuck_threshold=args.stuck,
                    solver_time_limit=args.solver_time_limit,
                    beam_width=args.beam, lookahead=args.lookahead,
                )
                n_total += 1
                if won: n_won += 1
                total_solver += n_solver
        elapsed = time.perf_counter() - t0
        r = {
            "phase": ph,
            "n_total": n_total,
            "n_won": n_won,
            "win_rate": n_won / max(n_total, 1),
            "solver_calls_per_episode": total_solver / max(n_total, 1),
        }
        print(f"phase {ph}: {r['win_rate']*100:.1f}% ({n_won}/{n_total}); "
              f"solver/ep={r['solver_calls_per_episode']:.2f}; "
              f"{elapsed:.0f}s")
        results.append(r)

    print("\n=== Summary ===")
    for r in results:
        print(f"phase {r['phase']}: {r['win_rate']*100:.2f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
