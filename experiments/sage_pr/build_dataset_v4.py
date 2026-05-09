"""Build dataset v4 — JEPP greedy commit, partial observation.

跟 v3 比:
    - belief 用 fully_observed=False (真实 partial obs, 含 K, Π)
    - 老师 = JEPP greedy commit (jepp_solver.jepp_pick_action)
    - 包含 inspect 候选和 inspect 标签
    - 每步 sample 同样输出 (X_grid, X_cand, u_global, mask, label, phase, source)

输出 npz 跟 v3 同 schema, 训练管线无需改动.
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.solver.explorer import compute_facing_actions
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
from experiments.sage_pr.jepp_solver import jepp_pick_action
from experiments.sage_pr.build_dataset_v3 import (
    parse_phase456_seeds, list_phase_maps, save_dataset,
    Sample, SOURCE_BF, match_move_to_candidate, solve_map,
)
from experiments.sage_pr.evaluate_sage_pr import _state_signature as _phys_signature


def _full_signature(state):
    """完整状态签名: 物理 + 朝向 + 已识别 ID 集合.

    inspect 动作不改变物理位置, 但改变 seen_*_ids 和角度. 必须算进 sig
    否则 inspect 后 sig 跟前一帧相同, 误判循环.
    """
    return (
        _phys_signature(state),
        round(state.car_angle * 4) / 4,   # 量化朝向 (0.25 弧度精度)
        frozenset(state.seen_box_ids),
        frozenset(state.seen_target_ids),
    )

import math


SOURCE_JEPP = 3   # 新老师标记


# ── 应用候选到 engine ────────────────────────────────────

def _heading_to_angle(heading: int) -> float:
    """heading 0..7 (E, SE, S, SW, W, NW, N, NE) → 弧度."""
    angles = {
        0: 0.0,
        1: math.pi / 4,
        2: math.pi / 2,
        3: 3 * math.pi / 4,
        4: math.pi,
        5: -3 * math.pi / 4,
        6: -math.pi / 2,
        7: -math.pi / 4,
    }
    return angles.get(heading, 0.0)


def apply_push_box_or_bomb(eng: GameEngine, cand: Candidate) -> bool:
    """把 push_box / push_bomb 候选展开为低层 discrete 动作并执行."""
    if cand.type not in ("push_box", "push_bomb"):
        return False

    eng.discrete_step(6)   # snap
    state = eng.get_state()
    dc, dr = cand.direction

    if cand.type == "push_box":
        if cand.box_idx >= len(state.boxes):
            return False
        b = state.boxes[cand.box_idx]
        ent_col, ent_row = pos_to_grid(b.x, b.y)
    else:
        if cand.bomb_idx >= len(state.bombs):
            return False
        bm = state.bombs[cand.bomb_idx]
        ent_col, ent_row = pos_to_grid(bm.x, bm.y)

    # macro 推送 = run_length 次 1-step push
    for k in range(cand.run_length):
        state = eng.get_state()
        car_target = (ent_col - dc, ent_row - dr)

        obstacles = set()
        for b2 in state.boxes:
            obstacles.add(pos_to_grid(b2.x, b2.y))
        for bm2 in state.bombs:
            obstacles.add(pos_to_grid(bm2.x, bm2.y))

        car_grid = pos_to_grid(state.car_x, state.car_y)
        if car_grid != car_target:
            path = bfs_path(car_grid, car_target, state.grid, obstacles)
            if path is None:
                return False
            for pdx, pdy in path:
                eng.discrete_step(direction_to_abs_action(pdx, pdy))

        eng.discrete_step(direction_to_abs_action(dc, dr))
        ent_col += dc
        ent_row += dr

    return True


def apply_inspect(eng: GameEngine, cand: Candidate) -> bool:
    """把 inspect 候选展开: 走到 viewpoint, 旋转面向 entity (引擎严格 FOV 自动识别)."""
    if cand.type != "inspect" or cand.viewpoint_col is None:
        return False
    eng.discrete_step(6)
    state = eng.get_state()

    obstacles = set()
    for b in state.boxes:
        obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs:
        obstacles.add(pos_to_grid(bm.x, bm.y))

    car_grid = pos_to_grid(state.car_x, state.car_y)
    target = (cand.viewpoint_col, cand.viewpoint_row)
    if car_grid != target:
        path = bfs_path(car_grid, target, state.grid, obstacles)
        if path is None:
            return False
        for pdx, pdy in path:
            eng.discrete_step(direction_to_abs_action(pdx, pdy))

    # 旋转
    state = eng.get_state()
    target_angle = _heading_to_angle(cand.inspect_heading or 0)
    rot_acts = compute_facing_actions(state.car_angle, target_angle)
    for a in rot_acts:
        eng.discrete_step(a)
    return True


def apply_candidate(eng: GameEngine, cand: Candidate) -> bool:
    if cand.type == "inspect":
        return apply_inspect(eng, cand)
    return apply_push_box_or_bomb(eng, cand)


# ── 单 episode 采集 ──────────────────────────────────────

def _solver_remaining_plan(eng: GameEngine,
                           bs: BeliefState,
                           cands: List[Candidate],
                           time_limit: float = 15.0) -> Optional[Tuple[int, Candidate]]:
    """全识别后, 用 MultiBoxSolver 规划剩余动作, 返回首步 (label, candidate)."""
    state = eng.get_state()
    moves = solve_map(state, max_cost=300, time_limit=time_limit, strategy="best_first")
    if not moves:
        return None
    label = match_move_to_candidate(moves[0], cands, bs, run_length=1)
    if label is None or not cands[label].legal:
        return None
    return label, cands[label]


def collect_episode_jepp(map_path: str, phase: int, seed: int,
                          *, step_limit: int = 80,
                          fully_observed: bool = False,
                          solver_time_limit: float = 15.0
                          ) -> Tuple[List[Sample], str]:
    """采集 JEPP partial-obs 单 episode.

    决策顺序:
        1. 未完全识别: jepp_pick_action (greedy commit / inspect)
        2. 完全识别后: 切到 MultiBoxSolver 严格规划 (避免 greedy 陷入循环)
    """
    import random
    random.seed(seed)

    eng = GameEngine()
    state = eng.reset(map_path)

    samples: List[Sample] = []
    sig_visit_count: Dict = {}
    sig_visit_count[_full_signature(state)] = 1

    for step in range(step_limit):
        state = eng.get_state()
        if state.won:
            return samples, "ok"

        bs = BeliefState.from_engine_state(state, fully_observed=fully_observed)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat)

        chosen: Optional[Candidate] = None
        label = -1
        if bs.fully_identified:
            res = _solver_remaining_plan(eng, bs, cands, solver_time_limit)
            if res is not None:
                label, chosen = res
        if chosen is None:
            chosen = jepp_pick_action(bs, feat, cands)
            if chosen is None:
                return samples, "no_action"
            label = -1
            for i, c in enumerate(cands):
                if c is chosen:
                    label = i
                    break
            if label < 0:
                return samples, "label_miss"

        # 录 sample
        X_grid = build_grid_tensor(bs, feat)
        X_cand = encode_candidates(cands, bs, feat)
        u_global = build_global_features(bs, feat)
        mask = candidates_legality_mask(cands)
        samples.append(Sample(
            X_grid=X_grid, X_cand=X_cand, u_global=u_global,
            mask=mask, label=label, phase=phase, source=SOURCE_JEPP,
        ))

        # 应用
        if not apply_candidate(eng, chosen):
            return samples, "apply_fail"

        new_sig = _full_signature(eng.get_state())
        sig_visit_count[new_sig] = sig_visit_count.get(new_sig, 0) + 1
        if sig_visit_count[new_sig] >= 3:
            # 严重循环 (同 state 第 3 次)
            return samples, "loop"

    return samples, "ok" if eng.get_state().won else "step_limit"


# ── worker / CLI ─────────────────────────────────────────

def _worker(args):
    map_path, phase, seed, step_limit, fully_observed = args
    try:
        samples, status = collect_episode_jepp(
            map_path, phase, seed,
            step_limit=step_limit, fully_observed=fully_observed,
        )
        return {"map": map_path, "seed": seed, "n": len(samples),
                "status": status, "samples": samples}
    except Exception as e:
        return {"map": map_path, "seed": seed, "n": 0,
                "status": f"error: {type(e).__name__}: {e}", "samples": []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--n-maps", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="0,42,137")
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--max-seeds-per-map", type=int, default=None)
    parser.add_argument("--step-limit", type=int, default=80)
    parser.add_argument("--fully-observed", action="store_true",
                        help="god mode (用作老 v3 行为对照)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"Phase {args.phase}, JEPP greedy commit, "
          f"fully_observed={args.fully_observed}, seeds={seeds}")

    maps = list_phase_maps(args.phase, args.n_maps)
    print(f"  found {len(maps)} maps")
    if not maps:
        sys.exit(1)

    verified_map = {}
    if args.use_verified_seeds:
        verified_map = parse_phase456_seeds(
            os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
        )
        print(f"  verified manifest entries: {len(verified_map)}")

    tasks: List = []
    if args.use_verified_seeds and verified_map:
        # 先按 phase 过滤, 再 n_maps 切
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
                              args.step_limit, args.fully_observed))
    else:
        for map_path in maps:
            for seed in seeds:
                tasks.append((map_path, args.phase, seed,
                              args.step_limit, args.fully_observed))
    print(f"  total tasks: {len(tasks)}")

    t0 = time.perf_counter()
    if args.workers <= 1:
        results = [_worker(t) for t in tasks]
    else:
        with mp.Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_worker, tasks, chunksize=1))

    samples: List[Sample] = []
    status_counts: Dict[str, int] = {}
    for r in results:
        st = r["status"]
        status_counts[st] = status_counts.get(st, 0) + 1
        if st == "ok" or st.startswith("ok"):
            samples.extend(r["samples"])

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s, samples={len(samples)}")
    print(f"  status: {status_counts}")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    save_dataset(samples, out_path)


if __name__ == "__main__":
    main()
