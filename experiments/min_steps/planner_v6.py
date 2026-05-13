"""Planner v6 — explore route biased to end at first push_pos.

策略:
  1. god plan, 拿 push_pos_1 = 首推位
  2. 列出需要扫描的 entity (排除律: N-1 个 box, N-1 个 target)
     + forced pair 跳过
  3. 给每个 entity 算 observation point (BFS-adjacent, line-of-sight 通)
     + heading + (cost(car_init → vp) + rot)
  4. 用小型 TSP 求 car_init → vp_a → vp_b → ... → push_pos_1 最优顺序
     (k ≤ 5, brute-force permutation)
  5. 执行 TSP 路径, 每点旋转 scan, 最后已经在 pp_1
  6. 用 god plan 推剩下
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
from collections import deque
from typing import List, Optional, Set, Tuple, Dict

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.solver.explorer import (
    find_observation_point, exploration_complete, get_entity_obstacles,
    get_all_entity_positions, _entity_scan_still_needed,
)
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs
from smartcar_sokoban.symbolic.features import INF


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
        eng.discrete_step(direction_to_abs_action(*[pdx, pdy]))
    return True


def _rotate_to_angle(eng: GameEngine, target_angle: float, tag: str) -> None:
    state = eng.get_state()
    diff = math.atan2(math.sin(target_angle - state.car_angle),
                       math.cos(target_angle - state.car_angle))
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


def _list_scan_entities(state, forced_box: Set[int], forced_tgt: Set[int]
                          ) -> List[Tuple[str, int, Tuple[int, int]]]:
    """需要扫的 entity: 排除已扫 + forced + 排除律最远的 1 个."""
    car_grid = pos_to_grid(state.car_x, state.car_y)
    todo: List[Tuple[str, int, Tuple[int, int]]] = []

    # box
    unseen_b = [i for i in range(len(state.boxes))
                if i not in state.seen_box_ids and i not in forced_box]
    if len(unseen_b) >= 2:
        # 跳最远的
        with_dist = []
        for i in unseen_b:
            b = state.boxes[i]
            bg = pos_to_grid(b.x, b.y)
            d = abs(car_grid[0]-bg[0]) + abs(car_grid[1]-bg[1])
            with_dist.append((d, i, bg))
        with_dist.sort()
        for _, i, bg in with_dist[:-1]:
            todo.append(("box", i, bg))

    # target
    unseen_t = [i for i in range(len(state.targets))
                if i not in state.seen_target_ids and i not in forced_tgt]
    if len(unseen_t) >= 2:
        with_dist = []
        for i in unseen_t:
            t = state.targets[i]
            tg = pos_to_grid(t.x, t.y)
            d = abs(car_grid[0]-tg[0]) + abs(car_grid[1]-tg[1])
            with_dist.append((d, i, tg))
        with_dist.sort()
        for _, i, tg in with_dist[:-1]:
            todo.append(("target", i, tg))
    return todo


def _plan_scan_route(eng: GameEngine, scan_entities: List[Tuple[str, int, Tuple[int, int]]],
                       endpoint: Tuple[int, int]
                       ) -> Optional[List[Tuple[Tuple[int, int], float]]]:
    """TSP-brute: 找 car → vp_perm → endpoint 最短.
    返回 [(vp, face_angle), ...] 路径上的 scan 点."""
    state = eng.get_state()
    car = pos_to_grid(state.car_x, state.car_y)
    obstacles = get_entity_obstacles(state)
    entity_pos = get_all_entity_positions(state)

    # 为每个 entity 找最优 viewpoint
    vps: List[Tuple[Tuple[int, int], float]] = []
    for etype, idx, eg in scan_entities:
        result = find_observation_point(
            car, eg, state.grid, obstacles, entity_pos,
            current_angle=state.car_angle)
        if result is None:
            continue
        vp, face_angle = result
        vps.append((vp, face_angle))

    if not vps:
        return []   # 没东西可扫 (forced + 排除律已经够)

    # TSP brute force (k ≤ 5, 120 permutations max)
    if len(vps) > 6:
        # 太多 — 退回贪心 nearest-first ending at endpoint
        vps = sorted(vps, key=lambda v:
                     _walk_cost(car, v[0], state.grid, obstacles))

    best_perm = None
    best_cost = INF
    for perm in itertools.permutations(range(len(vps))):
        c = car
        cur_angle = state.car_angle
        total = 0
        ok = True
        for idx in perm:
            vp, fa = vps[idx]
            wc = _walk_cost(c, vp, state.grid, obstacles)
            if wc == INF:
                ok = False; break
            # 旋转代价
            diff = math.atan2(math.sin(fa - cur_angle),
                               math.cos(fa - cur_angle))
            n = abs(round(diff / (math.pi / 2)))
            rot = min(n, 4 - n)
            total += wc + rot
            c = vp
            cur_angle = fa
            if total >= best_cost:
                ok = False; break
        if not ok:
            continue
        # 加最终 → endpoint
        wc_end = _walk_cost(c, endpoint, state.grid, obstacles)
        if wc_end == INF:
            continue
        total += wc_end
        if total < best_cost:
            best_cost = total
            best_perm = perm

    if best_perm is None:
        return None
    return [vps[i] for i in best_perm]


def planner_v6_tsp_explore(eng: GameEngine,
                            *, god_time_limit: float = 30.0) -> None:
    plan = _god_plan(eng, time_limit=god_time_limit)
    if not plan:
        return

    # 找 forced pair (可跳过扫)
    state = eng.get_state()
    forced = find_forced_pairs(state)
    forced_box = {i for i, _ in forced}
    forced_tgt = {j for _, j in forced}
    # mark seen 让 belief 不再要求扫
    state.seen_box_ids.update(forced_box)
    state.seen_target_ids.update(forced_tgt)

    # 列扫描表 + 首推位
    scan_list = _list_scan_entities(state, forced_box, forced_tgt)
    pp1 = _get_push_pos(plan[0])

    # TSP plan 扫描路径 (endpoint=pp1)
    route = _plan_scan_route(eng, scan_list, pp1)

    # 1) 执行扫描路径
    if route is not None:
        for vp, face_angle in route:
            if not _walk_to(eng, vp, "scan_walk"):
                return
            _rotate_to_angle(eng, face_angle, "scan_rot")

    # 2) 执行 god plan
    for move in plan:
        state = eng.get_state()
        if state.won:
            return
        push_pos = _get_push_pos(move)
        if not _walk_to(eng, push_pos, "push_walk"):
            return
        eng._step_tag = "push"
        eng.discrete_step(direction_to_abs_action(*move[2]))
