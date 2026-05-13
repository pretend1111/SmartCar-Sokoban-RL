"""Planner v5 — walk-first + JIT plan_exploration_v3 with single re-walk.

策略 (v3 升级版, 修复"绕路成本"问题):
  1. god plan
  2. 对每个 push:
     a. 直接 BFS walk → push_pos (FOV 沿路自动 reveal)
     b. 检查 σ:
        - 锁 → 推
        - 不锁 → 调 plan_exploration_v3 (从当前 car=push_pos 出发, 优化多 entity 扫描)
                 → walk_back to push_pos (push_walk_back)
                 → 检查 σ 二次, 锁了就推, 否则 fallback (强 push)
  3. 关键差异 vs v3: plan_exploration 只调一次 (不重试), 后续盲推, 避免连续 explore 退化.
"""

from __future__ import annotations

import contextlib
import io
from typing import List, Optional, Set, Tuple

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import generate_candidates

from experiments.sage_pr.build_dataset_v3 import match_move_to_candidate


def _god_plan(eng: GameEngine, time_limit: float = 30.0) -> Optional[List]:
    state = eng.get_state()
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return solver.solve(max_cost=300, time_limit=time_limit,
                                 strategy="auto")
        except Exception:
            return None


def _walk_to(eng: GameEngine, target: Tuple[int, int], tag: str) -> bool:
    state = eng.get_state()
    obstacles: Set[Tuple[int, int]] = set()
    for b in state.boxes:
        obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs:
        obstacles.add(pos_to_grid(bm.x, bm.y))
    car_grid = pos_to_grid(state.car_x, state.car_y)
    if car_grid == target:
        return True
    path = bfs_path(car_grid, target, state.grid, obstacles)
    if path is None:
        return False
    eng._step_tag = tag
    for pdx, pdy in path:
        eng.discrete_step(direction_to_abs_action(pdx, pdy))
    return True


def _get_push_pos(move) -> Tuple[int, int]:
    etype, eid, direction, _ = move
    dx, dy = direction
    if etype == "box":
        old_pos, _ = eid
        ec, er = old_pos
    elif etype == "bomb":
        ec, er = eid
    else:
        raise ValueError
    return (ec - dx, er - dy)


def _jit_explore(eng: GameEngine) -> None:
    from smartcar_sokoban.solver.explorer_v3 import plan_exploration_v3
    eng._step_tag = "explore_jit"
    with contextlib.redirect_stdout(io.StringIO()):
        plan_exploration_v3(eng, max_retries=15, verbose=False)


def planner_v5_walk_then_explore(eng: GameEngine,
                                   *, god_time_limit: float = 30.0) -> None:
    plan = _god_plan(eng, time_limit=god_time_limit)
    if plan is None:
        return

    explored_once = False
    for move in plan:
        state = eng.get_state()
        if state.won:
            return

        push_pos = _get_push_pos(move)

        # 1) walk to push_pos
        if not _walk_to(eng, push_pos, "push_walk"):
            return

        # 2) σ check
        state = eng.get_state()
        bs = BeliefState.from_engine_state(state, fully_observed=False)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat, enforce_sigma_lock=True)
        label = match_move_to_candidate(move, cands, bs, run_length=1)

        if label is None and not explored_once:
            # 全局 explore 一次 (FOV 已含 push_walk 累积, 不必扫已识别的)
            _jit_explore(eng)
            explored_once = True
            # walk back to push_pos
            state = eng.get_state()
            cur = pos_to_grid(state.car_x, state.car_y)
            if cur != push_pos:
                if not _walk_to(eng, push_pos, "push_walk_back"):
                    return

        # 3) push (即使 σ 仍不锁也推 — god plan 物理正确)
        eng._step_tag = "push"
        eng.discrete_step(direction_to_abs_action(*move[2]))
