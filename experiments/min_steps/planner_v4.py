"""Planner v4 — detour-cost-aware inspect.

跟 v1/v2 的差异: 选 inspect 时不按"车→viewpoint"代价, 而按"车→viewpoint→push_pos
真正绕路的 overhead"。在路上的 inspect 几乎 0 成本, 偏远的 inspect 才贵。

策略:
  1. god plan 拿 push 序列
  2. 走每个 push: 找还没识别但 σ 必要的 entity
      a. 对每个 candidate inspect viewpoint vp, 算 detour_cost(vp, push_pos)
         = walk(car, vp) + rot + walk(vp, push_pos) - walk(car, push_pos)
      b. 选 detour_cost 最小的, 如果 < 阈值就 detour, 否则 fallback 到 plan_exploration_v3
  3. 推
"""

from __future__ import annotations

import contextlib
import io
import math
from collections import deque
from typing import List, Optional, Set, Tuple

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features, INF
from smartcar_sokoban.symbolic.candidates import (
    generate_candidates, Candidate,
)

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


def _walk_cost(start: Tuple[int, int], end: Tuple[int, int],
                walls, obstacles: Set[Tuple[int, int]]) -> int:
    """BFS shortest-path cost. obstacles=set of (col,row). 返 INF 若不通."""
    if start == end:
        return 0
    rows = len(walls); cols = len(walls[0]) if rows else 0
    visited = {start}
    q = deque([(start, 0)])
    while q:
        (c, r), d = q.popleft()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc, nr = c + dc, r + dr
            if (nc, nr) in visited:
                continue
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if walls[nr][nc] == 1 or (nc, nr) in obstacles:
                continue
            visited.add((nc, nr))
            if (nc, nr) == end:
                return d + 1
            q.append(((nc, nr), d + 1))
    return INF


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


def _rotate_to_heading(eng: GameEngine, heading: int, tag: str) -> None:
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


def _heading_to_rot_cost(cur_a: float, heading: int) -> int:
    HEADING_TO_ANGLE = {0: 0.0, 1: math.pi/2, 2: math.pi, 3: -math.pi/2}
    tgt = HEADING_TO_ANGLE.get(heading, 0.0)
    diff = math.atan2(math.sin(tgt - cur_a), math.cos(tgt - cur_a))
    n = abs(round(diff / (math.pi / 2)))
    return min(n, 4 - n)


def _pick_min_detour_inspect(bs: BeliefState, cands: List[Candidate],
                              car: Tuple[int, int], push_pos: Tuple[int, int],
                              car_angle: float) -> Optional[Tuple[int, int]]:
    """挑 detour_cost 最小的 inspect.
    detour_cost = walk(car→vp) + rot + walk(vp→push_pos) - walk(car→push_pos)
    返回 (cand_idx, detour_cost) or None.
    """
    walls = bs.M  # 0=free,1=wall (list of list)
    obstacles: Set[Tuple[int, int]] = set()
    for b in bs.boxes:
        obstacles.add((b.col, b.row))
    for bm in bs.bombs:
        obstacles.add((bm.col, bm.row))

    direct = _walk_cost(car, push_pos, walls, obstacles)
    if direct == INF:
        direct = 0   # 兜底, 不影响排序

    best_idx = None
    best_cost = INF
    for k, c in enumerate(cands):
        if c.type != "inspect" or not c.legal:
            continue
        if c.viewpoint_col is None:
            continue
        vp = (c.viewpoint_col, c.viewpoint_row)
        c_to_vp = _walk_cost(car, vp, walls, obstacles)
        if c_to_vp == INF:
            continue
        vp_to_pp = _walk_cost(vp, push_pos, walls, obstacles)
        if vp_to_pp == INF:
            continue
        rot = _heading_to_rot_cost(car_angle, c.inspect_heading or 0)
        total = c_to_vp + rot + vp_to_pp
        detour = total - direct
        # 用绝对总代价排序 (而不是 detour), 这样如果原本走直线非常远, 顺路最近的还是赢
        if total < best_cost:
            best_cost = total
            best_idx = k
    return (best_idx, best_cost) if best_idx is not None else None


def planner_v4_detour_aware(eng: GameEngine,
                              *, max_inspect_retries: int = 3,
                              god_time_limit: float = 30.0) -> None:
    plan = _god_plan(eng, time_limit=god_time_limit)
    if plan is None:
        return

    for move in plan:
        state = eng.get_state()
        if state.won:
            return

        push_pos = _get_push_pos(move)
        # 不直接走到 push_pos. 先检查 σ, 必要时挑顺路 inspect, detour 完再去.
        for r in range(max_inspect_retries + 1):
            state = eng.get_state()
            bs = BeliefState.from_engine_state(state, fully_observed=False)
            feat = compute_domain_features(bs)
            cands = generate_candidates(bs, feat, enforce_sigma_lock=True)
            label = match_move_to_candidate(move, cands, bs, run_length=1)
            if label is not None:
                break

            # σ 不锁 → 挑 detour_cost 最小的 inspect
            car_grid = pos_to_grid(state.car_x, state.car_y)
            pick = _pick_min_detour_inspect(
                bs, cands, car_grid, push_pos, state.car_angle)
            if pick is None:
                break
            cand_idx, _ = pick
            ins = cands[cand_idx]
            vp = (ins.viewpoint_col, ins.viewpoint_row)

            if not _walk_to(eng, vp, "inspect_walk"):
                break
            _rotate_to_heading(eng, ins.inspect_heading or 0, "inspect_rot")

        # 走到 push_pos
        if not _walk_to(eng, push_pos, "push_walk"):
            return
        eng._step_tag = "push"
        eng.discrete_step(direction_to_abs_action(*move[2]))
