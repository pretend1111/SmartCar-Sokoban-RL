"""Planner v2 — walk-first, check-later opportunistic.

跟 v1 的差异:
  v1: 先在当前位置 check σ → 不锁 → 选 inspect → walk to viewpoint → scan → 回来 walk to push_pos
  v2: 先 walk to push_pos (FOV 沿路自动更新 belief) → check σ → 仍不锁才 detour 一个最便宜的 inspect

期望: FOV 在 BFS-walk 时常常顺路扫到需要的 entity, 省掉单独 inspect.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
from typing import List, Optional, Set, Tuple

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import (
    generate_candidates, Candidate,
)

from experiments.sage_pr.build_dataset_v6 import pick_inspect_for_unlock
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


def _walk_to(eng: GameEngine, target: Tuple[int, int],
              tag_prefix: str) -> bool:
    """从当前 car_grid BFS 走到 target. obstacles = boxes ∪ bombs."""
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
    eng._step_tag = tag_prefix
    for pdx, pdy in path:
        eng.discrete_step(direction_to_abs_action(pdx, pdy))
    return True


def _rotate_to_heading(eng: GameEngine, heading: int, tag: str) -> None:
    """旋转到 4 方向 heading (0=right, 1=down, 2=left, 3=up).
    engine 坐标: angle 0=right, +π/2=down, -π/2=up."""
    HEADING_TO_ANGLE = {0: 0.0, 1: math.pi/2, 2: math.pi, 3: -math.pi/2}
    state = eng.get_state()
    tgt = HEADING_TO_ANGLE.get(heading, 0.0)
    diff = math.atan2(math.sin(tgt - state.car_angle),
                       math.cos(tgt - state.car_angle))
    n = round(diff / (math.pi / 2))
    eng._step_tag = tag
    if n == 2 or n == -2:
        eng.discrete_step(5); eng.discrete_step(5)
    elif n == 1:
        eng.discrete_step(5)
    elif n == -1:
        eng.discrete_step(4)


def _push_action(eng: GameEngine, direction: Tuple[int, int]) -> None:
    eng._step_tag = "push"
    eng.discrete_step(direction_to_abs_action(*direction))


def _get_push_pos(move) -> Tuple[int, int]:
    """从 move = (etype, eid, (dx,dy), _) 算出推位."""
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


def planner_v2_walk_first(eng: GameEngine,
                           *, max_inspect_retries: int = 3,
                           god_time_limit: float = 30.0) -> None:
    plan = _god_plan(eng, time_limit=god_time_limit)
    if plan is None:
        return

    for a_idx, move in enumerate(plan):
        state = eng.get_state()
        if state.won:
            return

        push_pos = _get_push_pos(move)
        # 1) 走到 push_pos (FOV 沿路自动更新 belief)
        if not _walk_to(eng, push_pos, "push_walk"):
            return

        # 2) 检查 belief 是否支持这个 push
        for retry in range(max_inspect_retries + 1):
            state = eng.get_state()
            bs = BeliefState.from_engine_state(state, fully_observed=False)
            feat = compute_domain_features(bs)
            cands = generate_candidates(bs, feat, enforce_sigma_lock=True)
            label = match_move_to_candidate(move, cands, bs, run_length=1)
            if label is not None:
                break

            # σ 不锁 → 找最便宜 inspect
            required_box = -1
            required_target = -1
            if move[0] == "box":
                old_pos, cid = move[1]
                for j, b in enumerate(bs.boxes):
                    if (b.col, b.row) == old_pos:
                        required_box = j
                        break
                for j, t in enumerate(bs.targets):
                    if t.num_id == cid:
                        required_target = j
                        break
            ins_label = pick_inspect_for_unlock(bs, cands, required_box,
                                                 required_target)
            if ins_label is None:
                break
            ins = cands[ins_label]
            # detour
            if not _walk_to(eng, (ins.viewpoint_col, ins.viewpoint_row),
                            "inspect_walk"):
                break
            _rotate_to_heading(eng, ins.inspect_heading or 0, "inspect_rot")
            # 回 push_pos
            if not _walk_to(eng, push_pos, "push_walk_back"):
                return

        # 3) 推 (无论 belief 是否锁定 — 物理上一定可行因为 god plan)
        _push_action(eng, move[2])
