"""Planner v7 — DP over scan/push interleavings.

v6 用 TSP 把所有 scan 排在 push 之前. v7 允许 scan 穿插在 pushes 之间.

设:
  pushes = [pp_1, pp_2, ..., pp_n]   固定顺序 (god plan)
  scans  = [(vp_1, ang_1), ..., (vp_k, ang_k)]  无序

DP:
  state = (push_idx ∈ [0..n], scan_set ⊆ {0..k-1}, last_pos)
  其中 last_pos = car_init, 或 last 个 pp_i, 或 last 个 vp_j

  dp[state] = 最小累计 walk + rotation cost 走到 state

  transitions:
    1. 下推 (push_idx → push_idx+1): + walk(last_pos, pp_{push_idx+1})
    2. 插一个 scan (scan_set → scan_set ∪ {j}): + walk(last_pos, vp_j) + rot
"""

from __future__ import annotations

import contextlib
import io
import math
from collections import deque
from typing import List, Optional, Set, Tuple, Dict

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.solver.explorer import (
    find_observation_point, get_entity_obstacles, get_all_entity_positions,
)
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs
from smartcar_sokoban.symbolic.features import INF
from experiments.min_steps.planner_v6 import (
    _list_scan_entities,
)


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
        eng.discrete_step(direction_to_abs_action(pdx, pdy))
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


def _rot_steps(cur_angle: float, target_angle: float) -> int:
    diff = math.atan2(math.sin(target_angle - cur_angle),
                       math.cos(target_angle - cur_angle))
    n = abs(round(diff / (math.pi / 2)))
    return min(n, 4 - n)


def _solve_interleave(car_init: Tuple[int, int], car_angle: float,
                       push_positions: List[Tuple[int, int]],
                       scan_vps: List[Tuple[Tuple[int, int], float]],
                       walls, obstacles_per_step: List[Set[Tuple[int, int]]]
                       ) -> Optional[List[Tuple[str, int]]]:
    """DP 求 scan/push interleaving 最小 cost.

    Returns: action list, 每元素是 ("push", push_idx) 或 ("scan", scan_idx)
    顺序就是要执行的顺序.

    obstacles_per_step[i]: 推完 push i 之前的障碍集 (用于估 walk cost 时近似)
    """
    n_p = len(push_positions)
    n_s = len(scan_vps)
    if n_s > 6:
        return None   # 状态空间太大, 跳过

    # 节点编号: 0 = car_init, 1..n_p = push_pos, n_p+1..n_p+n_s = scan_vp
    # 但 last_pos 也可以是 scan_vp (旋转后还在原位)
    # 重新建模:
    #  状态 (push_idx, scan_set, last_node_id):
    #    push_idx ∈ [0, n_p] (已完成的 push 数)
    #    scan_set ⊆ {0..n_s-1}
    #    last_node_id: 0=car_init, 1..n_p=push_idx, n_p+1..n_p+n_s=scan
    #  注意 last_node 必须跟 (push_idx, scan_set) 一致 — last 个 action 是 push 还是 scan

    # 简单 DP, 用字典
    # state: (push_idx, scan_set_bitmap, last_kind, last_idx)
    # last_kind: 0=init, 1=push, 2=scan
    # last_idx: kind=1 → push_idx-1; kind=2 → scan idx

    # 取得每个节点的世界坐标
    def node_pos(kind, idx):
        if kind == 0:
            return car_init
        if kind == 1:
            return push_positions[idx]
        if kind == 2:
            return scan_vps[idx][0]
        raise ValueError

    def node_angle(kind, idx, prev_kind, prev_idx, prev_angle):
        """到达 node 后 car_angle. push/init 后 angle 跟移动方向相关, 暂忽略
           (假设 push 后 car_angle 不显式控制 — 走 absolute world move 不旋转).
           scan 后 angle = scan_vps[idx][1]."""
        if kind == 2:
            return scan_vps[idx][1]
        return prev_angle   # 移动不旋转

    # 用 dict[state] = (cost, parent_state) 反推 path
    INFV = float('inf')
    dp: Dict[Tuple[int, int, int, int], Tuple[float, Optional[Tuple]]] = {}
    init = (0, 0, 0, 0)
    dp[init] = (0.0, None)

    # BFS-like: 用列表 + 排序 (实际可用 priority queue, 但 N 小直接全展开)
    # 每个 state 的下一步: 推下一个 push, 或扫一个未扫的 scan.
    pending = [init]
    final_state = None
    final_cost = INFV

    # 用 stack iterative
    while pending:
        new_pending = []
        for state in pending:
            push_idx, scan_set, last_kind, last_idx = state
            cur_cost, _parent = dp[state]
            if push_idx == n_p and scan_set == (1 << n_s) - 1:
                if cur_cost < final_cost:
                    final_cost = cur_cost
                    final_state = state
                continue
            cur_pos = node_pos(last_kind, last_idx)
            # 当前 angle 重建很麻烦, 在 DP 里假设 angle 影响只在 scan 处 (rot 代价)
            # 为简化, 不在 cost 里加 rot for push (push 走 abs world 不旋转 angle)
            cur_angle = 0.0  # 初值
            # TODO: 严格地说应该从 state 重建 angle. 这里简化为按上次 scan 的 angle.
            # 简化: rot 代价只在 scan 时计 (从前一个状态的 angle 算)

            # 选合适的 obstacles: 用 push_idx 对应的 (push_idx == 推完 idx 个之后)
            obs = obstacles_per_step[min(push_idx, len(obstacles_per_step)-1)]

            # 选项 1: 推下一个 push
            if push_idx < n_p:
                target = push_positions[push_idx]
                wc = _walk_cost(cur_pos, target, walls, obs)
                if wc != INF:
                    new_state = (push_idx + 1, scan_set, 1, push_idx)
                    new_cost = cur_cost + wc + 1   # +1 for push action itself
                    if new_state not in dp or new_cost < dp[new_state][0]:
                        dp[new_state] = (new_cost, state)
                        new_pending.append(new_state)
            # 选项 2: 扫一个未扫的
            for s in range(n_s):
                if scan_set & (1 << s):
                    continue
                target_pos, target_angle = scan_vps[s]
                wc = _walk_cost(cur_pos, target_pos, walls, obs)
                if wc == INF:
                    continue
                rot = _rot_steps(cur_angle, target_angle)
                new_state = (push_idx, scan_set | (1 << s), 2, s)
                new_cost = cur_cost + wc + rot
                if new_state not in dp or new_cost < dp[new_state][0]:
                    dp[new_state] = (new_cost, state)
                    new_pending.append(new_state)
        pending = new_pending

    if final_state is None:
        return None

    # 回溯 path
    actions: List[Tuple[str, int]] = []
    cur = final_state
    while cur is not None:
        _push_idx, _scan_set, last_kind, last_idx = cur
        if last_kind == 1:
            actions.append(("push", last_idx))
        elif last_kind == 2:
            actions.append(("scan", last_idx))
        # init 不加
        parent = dp[cur][1]
        cur = parent
    actions.reverse()
    return actions


def planner_v7_dp_interleave(eng: GameEngine,
                              *, god_time_limit: float = 30.0) -> None:
    plan = _god_plan(eng, time_limit=god_time_limit)
    if not plan:
        return

    state = eng.get_state()
    forced = find_forced_pairs(state)
    forced_box = {i for i, _ in forced}
    forced_tgt = {j for _, j in forced}
    state.seen_box_ids.update(forced_box)
    state.seen_target_ids.update(forced_tgt)

    scan_list = _list_scan_entities(state, forced_box, forced_tgt)
    if not scan_list:
        # 没东西可扫 — 直接推
        for move in plan:
            if eng.get_state().won:
                return
            push_pos = _get_push_pos(move)
            if not _walk_to(eng, push_pos, "push_walk"):
                return
            eng._step_tag = "push"
            eng.discrete_step(direction_to_abs_action(*move[2]))
        return

    obstacles = get_entity_obstacles(state)
    entity_pos = get_all_entity_positions(state)

    scan_vps: List[Tuple[Tuple[int, int], float]] = []
    for etype, idx, eg in scan_list:
        result = find_observation_point(
            pos_to_grid(state.car_x, state.car_y), eg, state.grid, obstacles,
            entity_pos, current_angle=state.car_angle)
        if result is None:
            continue
        scan_vps.append(result)

    if not scan_vps:
        # 找不到 viewpoint, 退回 god push only
        for move in plan:
            push_pos = _get_push_pos(move)
            if not _walk_to(eng, push_pos, "push_walk"):
                return
            eng._step_tag = "push"
            eng.discrete_step(direction_to_abs_action(*move[2]))
        return

    push_positions = [_get_push_pos(m) for m in plan]
    car_init = pos_to_grid(state.car_x, state.car_y)
    # obstacles per step: 不严格模拟, 直接用当前 obstacles (简化)
    obstacles_per_step = [obstacles for _ in range(len(plan) + 1)]

    actions = _solve_interleave(car_init, state.car_angle, push_positions,
                                  scan_vps, state.grid, obstacles_per_step)
    if actions is None:
        # 退回 v6 行为
        for vp, fa in scan_vps:
            if not _walk_to(eng, vp, "scan_walk"):
                break
            _rotate_to_angle(eng, fa, "scan_rot")
        for move in plan:
            push_pos = _get_push_pos(move)
            if not _walk_to(eng, push_pos, "push_walk"):
                return
            eng._step_tag = "push"
            eng.discrete_step(direction_to_abs_action(*move[2]))
        return

    # 执行 action 列表
    push_iter = iter(plan)
    for kind, idx in actions:
        if eng.get_state().won:
            return
        if kind == "push":
            move = next(push_iter)
            push_pos = _get_push_pos(move)
            if not _walk_to(eng, push_pos, "push_walk"):
                return
            eng._step_tag = "push"
            eng.discrete_step(direction_to_abs_action(*move[2]))
        elif kind == "scan":
            vp, fa = scan_vps[idx]
            if not _walk_to(eng, vp, "scan_walk"):
                continue
            _rotate_to_angle(eng, fa, "scan_rot")
