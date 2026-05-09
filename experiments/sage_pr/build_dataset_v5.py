"""Build dataset v5 — Belief-IDA* JEPP + exact fallback.

策略:
    1. 先尝试 belief_ida_solve (Level B JEPP 老师, 边推边看)
    2. 失败 → 退回 god-mode IDA* (solve_exact: plan_exploration + MultiBoxSolver)

对于训练数据 source 字段:
    SOURCE_JEPP_B = 4   — Level B JEPP 老师, 高质量 commit-before-inspect 轨迹
    SOURCE_EXACT  = 5   — exact (god mode + plan_exploration), fallback

输出 npz 同 v3/v4 schema, 训练管线无需改.
"""

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.solver.explorer import (
    plan_exploration, exploration_complete,
)
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import (
    Candidate, generate_candidates, candidates_legality_mask, MAX_CANDIDATES,
)
from smartcar_sokoban.symbolic.cand_features import encode_candidates
from smartcar_sokoban.symbolic.grid_tensor import (
    build_grid_tensor, build_global_features,
    GRID_TENSOR_CHANNELS, GLOBAL_DIM,
)
from experiments.sage_pr.belief_ida_solver import belief_ida_solve
from experiments.sage_pr.build_dataset_v3 import (
    parse_phase456_seeds, list_phase_maps, save_dataset, Sample,
    apply_solver_move, match_move_to_candidate,
)


SOURCE_JEPP_B = 4
SOURCE_EXACT_FALLBACK = 5


def _build_sample_from_meta(m: Dict, source: int, phase: int) -> Sample:
    bs = m["bs"]
    feat = m["feat"]
    cands = m["cands"]
    return Sample(
        X_grid=build_grid_tensor(bs, feat),
        X_cand=encode_candidates(cands, bs, feat),
        u_global=build_global_features(bs, feat),
        mask=candidates_legality_mask(cands),
        label=m["label"],
        phase=phase,
        source=source,
    )


def _exact_fallback_episode(map_path: str, phase: int, seed: int,
                             *, time_limit: float = 60.0) -> Tuple[List[Sample], str]:
    """Exact fallback: plan_exploration + MultiBoxSolver, 翻译为 candidate 序列."""
    random.seed(seed)
    eng = GameEngine()
    state = eng.reset(map_path)

    # Phase 1: plan_exploration — 这里已经会触发严格 FOV 识别
    # 我们逐步重做, 录制每个 inspect candidate. 但 plan_exploration 直接给低层动作,
    # 不直接对应 candidate. 简化: 先跑 explore, 再跑 solver, 但只录 solver phase 的样本.
    with contextlib.redirect_stdout(io.StringIO()):
        explore_actions = plan_exploration(eng)

    state = eng.get_state()
    if not exploration_complete(state):
        return [], "explore_incomplete"

    # Phase 2: MultiBoxSolver
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            moves = solver.solve(max_cost=300, time_limit=time_limit, strategy="auto")
        except Exception:
            moves = None
    if not moves:
        return [], "solver_no_solution"

    samples: List[Sample] = []
    for move in moves:
        bs = BeliefState.from_engine_state(eng.get_state(), fully_observed=True)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat)
        label = match_move_to_candidate(move, cands, bs, run_length=1)
        if label is None:
            # match fail (chain push edge case): 应用引擎但不记 sample
            if not apply_solver_move(eng, move):
                return samples, "apply_fail"
            continue

        samples.append(Sample(
            X_grid=build_grid_tensor(bs, feat),
            X_cand=encode_candidates(cands, bs, feat),
            u_global=build_global_features(bs, feat),
            mask=candidates_legality_mask(cands),
            label=label,
            phase=phase,
            source=SOURCE_EXACT_FALLBACK,
        ))
        if not apply_solver_move(eng, move):
            return samples, "apply_fail"

    if not eng.get_state().won:
        return samples, "did_not_win"
    return samples, "ok"


def collect_episode_v5(map_path: str, phase: int, seed: int,
                        *, jepp_time_limit: float = 30.0,
                        step_limit: int = 100,
                        try_jepp: bool = True
                        ) -> Tuple[List[Sample], str, str]:
    """先 JEPP, 失败回退 exact."""
    if try_jepp:
        meta_list, jepp_status = belief_ida_solve(
            map_path, seed,
            ida_time_limit=jepp_time_limit, step_limit=step_limit,
        )
        if jepp_status == "ok" and meta_list:
            samples = [_build_sample_from_meta(m, SOURCE_JEPP_B, phase) for m in meta_list]
            return samples, "ok_jepp", jepp_status
    else:
        jepp_status = "skipped"

    # Fallback
    samples, ex_status = _exact_fallback_episode(map_path, phase, seed)
    if ex_status == "ok":
        return samples, "ok_exact", jepp_status
    return [], f"both_failed:jepp={jepp_status},ex={ex_status}", jepp_status


def _worker(args):
    map_path, phase, seed, jepp_tl, step_limit, try_jepp = args
    try:
        samples, status, jepp_status = collect_episode_v5(
            map_path, phase, seed,
            jepp_time_limit=jepp_tl, step_limit=step_limit, try_jepp=try_jepp,
        )
        return {"map": map_path, "seed": seed, "n": len(samples),
                "status": status, "jepp_status": jepp_status, "samples": samples}
    except Exception as e:
        return {"map": map_path, "seed": seed, "n": 0,
                "status": f"error:{type(e).__name__}", "samples": []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--n-maps", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--max-seeds-per-map", type=int, default=None)
    parser.add_argument("--jepp-time-limit", type=float, default=20.0)
    parser.add_argument("--step-limit", type=int, default=100)
    parser.add_argument("--no-jepp", action="store_true",
                        help="跳过 JEPP, 全部用 exact (作为对照)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"Phase {args.phase}, JEPP+Exact, jepp_time={args.jepp_time_limit}s, "
          f"try_jepp={not args.no_jepp}")

    maps = list_phase_maps(args.phase, args.n_maps)
    if not maps:
        sys.exit(1)

    verified_map = parse_phase456_seeds(
        os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
    ) if args.use_verified_seeds else {}

    tasks = []
    if args.use_verified_seeds and verified_map:
        items_phase = [(k, v) for k, v in verified_map.items()
                       if f"phase{args.phase}/" in k]
        items_phase.sort()
        if args.n_maps is not None:
            items_phase = items_phase[:args.n_maps]
        for map_path, ms in items_phase:
            full = os.path.join(ROOT, map_path)
            if not os.path.exists(full):
                continue
            n_per = args.max_seeds_per_map or max(1, len(seeds))
            for seed in ms[:n_per]:
                tasks.append((map_path, args.phase, seed,
                              args.jepp_time_limit, args.step_limit,
                              not args.no_jepp))
    else:
        for map_path in maps:
            for seed in seeds:
                tasks.append((map_path, args.phase, seed,
                              args.jepp_time_limit, args.step_limit,
                              not args.no_jepp))
    print(f"  total tasks: {len(tasks)}")

    t0 = time.perf_counter()
    if args.workers <= 1:
        results = [_worker(t) for t in tasks]
    else:
        with mp.Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_worker, tasks, chunksize=1))

    samples: List[Sample] = []
    status_counts: Dict[str, int] = {}
    jepp_status_counts: Dict[str, int] = {}
    for r in results:
        st = r["status"]
        status_counts[st] = status_counts.get(st, 0) + 1
        if "jepp_status" in r:
            js = r["jepp_status"]
            jepp_status_counts[js] = jepp_status_counts.get(js, 0) + 1
        if st.startswith("ok"):
            samples.extend(r["samples"])

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s, samples={len(samples)}")
    print(f"  episode status: {status_counts}")
    print(f"  jepp status: {jepp_status_counts}")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    save_dataset(samples, out_path)


if __name__ == "__main__":
    main()
