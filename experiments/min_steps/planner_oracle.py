"""Planner oracle — exact min-step optimal under fixed god push plan.

逻辑:
  1. god MultiBoxSolver → push 序列 P (n_p 个 push)
  2. 列出需扫 entity → viewpoint vp_i + heading_i (n_s 个)
  3. 严格 clone engine, 跑一遍 P, 记录每个 push 之前/之后 engine 状态 (obstacles)
     → obstacle_per_step[0..n_p]
  4. 对每个 step k, BFS all-pairs 距离表 dist[k][p → q] (障碍 = obstacle_per_step[k])
  5. 状态 (push_idx, scan_set, last_kind, last_idx):
       last_kind ∈ {0=init, 1=after_push, 2=after_scan}
       last_idx 对应 push 或 scan index
     从 (last_kind, last_idx) 能推出 (last_pos, last_angle)
  6. DP / Dijkstra 找 (push_idx=n_p, scan_set=full) 最小 cost
  7. 重放最优 interleaving 拿到 oracle 步数

注: 假设 god plan 可执行 (物理上 won). engine 用真实 ID 判配对, 所以
  belief 不锁也照样配对成功; oracle 只关心步数最优.
"""

from __future__ import annotations

import contextlib
import copy
import heapq
import io
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.solver.explorer import (
    find_observation_point, get_entity_obstacles, get_all_entity_positions,
)
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs
from smartcar_sokoban.symbolic.features import INF

INFV = 10**9


def _fresh_engine_from_eng(eng: GameEngine) -> GameEngine:
    """从 best_best 共享的 map_path/seed 重建干净 engine (不含 monkey patch)."""
    import random
    from experiments.min_steps.planner_best import _BEST_MAP_PATH, _BEST_SEED
    if not _BEST_MAP_PATH:
        # 退回 deepcopy + 卸 monkey patch (尽力而为)
        return copy.deepcopy(eng)
    random.seed(_BEST_SEED)
    e = GameEngine()
    e.reset(_BEST_MAP_PATH)
    e.discrete_step(6)
    return e


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


def _god_plans(eng: GameEngine, k: int = 5,
                time_limit: float = 30.0,
                belief_weight: float = 2.0) -> List[List]:
    """K-best god plans — 用于 v18 enumerate (plan, alpha) 全部组合.

    belief_weight > 0 时, 把 state0 的 trigger_map 传给 solver, 让其优化 push 时
    顺路 walk-reveal entity 的方案. 用 cost = walk - belief_weight × bonus 调整,
    bonus = 路径 + 推后落点上 unique trigger 配置数.
    """
    state = eng.get_state()
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)

    # 构造 trigger_map (state0): (cell, q4) -> set of scan_indices
    trigger_map = None
    if belief_weight > 0:
        from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs
        from experiments.min_steps.planner_oracle_v4 import _build_entity_vps
        state_for_scan = eng.get_state()
        forced = find_forced_pairs(state_for_scan)
        fb = {i for i,_ in forced}; ft = {j for _,j in forced}
        state_for_scan.seen_box_ids.update(fb)
        state_for_scan.seen_target_ids.update(ft)
        scans = _list_scans(state_for_scan, fb, ft)
        vps_per_scan = _build_entity_vps(scans, state_for_scan)
        trigger_map = {}
        for j in range(len(scans)):
            for (vp, q) in vps_per_scan[j]:
                trigger_map.setdefault((vp, q), set()).add(j)

    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs,
                              trigger_map=trigger_map,
                              belief_weight=belief_weight)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return solver.solve_kbest(k=k, max_cost=300, time_limit=time_limit)
        except Exception:
            return []


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


def _bfs_from(start: Tuple[int, int], walls, obstacles: Set[Tuple[int, int]]
              ) -> Dict[Tuple[int, int], int]:
    """All distances 从 start 出发. 障碍 = walls ∪ obstacles."""
    rows = len(walls); cols = len(walls[0]) if rows else 0
    dist: Dict[Tuple[int, int], int] = {start: 0}
    q = deque([start])
    while q:
        c, r = q.popleft()
        d = dist[(c, r)]
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc, nr = c + dc, r + dr
            if (nc, nr) in dist:
                continue
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if walls[nr][nc] == 1 or (nc, nr) in obstacles:
                continue
            dist[(nc, nr)] = d + 1
            q.append((nc, nr))
    return dist


def _walk_path(start: Tuple[int, int], end: Tuple[int, int],
                walls, obstacles: Set[Tuple[int, int]]
                ) -> Optional[List[Tuple[int, int]]]:
    """返回 dx,dy 步序列."""
    if start == end:
        return []
    p = bfs_path(start, end, walls, obstacles)
    return p


def _rot_steps(cur_a: float, tgt_a: float) -> int:
    diff = math.atan2(math.sin(tgt_a - cur_a), math.cos(tgt_a - cur_a))
    n = abs(round(diff / (math.pi / 2)))
    return min(n, 4 - n)


def _heading_to_angle(h: int) -> float:
    """0=East(0), 1=South(π/2), 2=West(π), 3=North(-π/2). 与 engine.car_angle 一致."""
    return {0: 0.0, 1: math.pi/2, 2: math.pi, 3: -math.pi/2}.get(h, 0.0)


def _simulate_god_and_record(eng_init: GameEngine, god_plan: List):
    """跑 god plan, 记录每步前 snapshot. 返回 n_p+1 tuple:
        (walls, obstacles, car_pos, car_angle, box_pos_by_idx, tgt_pos_by_idx)
    box_pos_by_idx[i] = (pos, class_id) or None (consumed). 同理 target.
    walls 存全是因为 TNT 会动态改墙. boxes/targets 索引按 state0 顺序, 持久跟踪."""
    import copy as _cp
    e = _fresh_engine_from_eng(eng_init)
    snapshots = []
    state = e.get_state()
    n_box_init = len(state.boxes); n_tgt_init = len(state.targets)
    # 持久 box / target 跟踪: 用初始 class/num 作 stable id
    box_initial_classes = [b.class_id for b in state.boxes]
    tgt_initial_nums = [t.num_id for t in state.targets]

    def take_snap():
        s = e.get_state()
        obs = set()
        for b in s.boxes: obs.add(pos_to_grid(b.x, b.y))
        for bm in s.bombs: obs.add(pos_to_grid(bm.x, bm.y))
        cur_box_cls_to_pos = {b.class_id: pos_to_grid(b.x, b.y) for b in s.boxes}
        cur_tgt_num_to_pos = {t.num_id: pos_to_grid(t.x, t.y) for t in s.targets}
        box_pos_by_idx = [(cur_box_cls_to_pos.get(c), c) if c in cur_box_cls_to_pos else None
                          for c in box_initial_classes]
        tgt_pos_by_idx = [(cur_tgt_num_to_pos.get(n), n) if n in cur_tgt_num_to_pos else None
                          for n in tgt_initial_nums]
        return ([row[:] for row in s.grid], obs,
                pos_to_grid(s.car_x, s.car_y), s.car_angle,
                box_pos_by_idx, tgt_pos_by_idx)

    snapshots.append(take_snap())
    from experiments.sage_pr.build_dataset_v3 import apply_solver_move
    for move in god_plan:
        ok = apply_solver_move(e, move)
        if not ok:
            return snapshots
        snapshots.append(take_snap())
    return snapshots


def _list_scans(state, forced_box: Set[int], forced_tgt: Set[int]
                  ) -> List[Tuple[Tuple[int, int], float, str, int]]:
    """需扫 entity 的 (vp, scan_angle, etype, eidx) 列表.
    用排除律: N-1 即可. 跳过 forced pair."""
    obstacles = get_entity_obstacles(state)
    entity_pos = get_all_entity_positions(state)
    car_grid = pos_to_grid(state.car_x, state.car_y)
    todo = []

    # box: 跳已扫 / forced; >= 2 个时跳最远的 1 个
    unseen_b = [i for i in range(len(state.boxes))
                 if i not in state.seen_box_ids and i not in forced_box]
    if len(unseen_b) >= 2:
        with_dist = []
        for i in unseen_b:
            b = state.boxes[i]
            bg = pos_to_grid(b.x, b.y)
            d = abs(car_grid[0]-bg[0]) + abs(car_grid[1]-bg[1])
            with_dist.append((d, i, bg))
        with_dist.sort()
        for _, i, bg in with_dist[:-1]:
            r = find_observation_point(car_grid, bg, state.grid, obstacles,
                                          entity_pos, current_angle=state.car_angle)
            if r is None: continue
            vp, angle = r
            todo.append((vp, angle, "box", i))

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
            r = find_observation_point(car_grid, tg, state.grid, obstacles,
                                          entity_pos, current_angle=state.car_angle)
            if r is None: continue
            vp, angle = r
            todo.append((vp, angle, "target", i))
    return todo


@dataclass
class OracleResult:
    cost: int
    interleaving: List[Tuple[str, int]]   # ("push", k) or ("scan", j)


def oracle_min_steps(eng_init: GameEngine,
                       *, god_time_limit: float = 30.0,
                       verbose: bool = False) -> Optional[OracleResult]:
    """求 belief-aware 最小步数 (假设 god push 序列固定, scan 顺序/位置自由)."""
    plan = _god_plan(eng_init, time_limit=god_time_limit)
    if not plan:
        return None
    n_p = len(plan)

    # 找 forced pair (scan 跳过)
    state0 = eng_init.get_state()
    forced = find_forced_pairs(state0)
    forced_box = {i for i, _ in forced}
    forced_tgt = {j for _, j in forced}
    state0.seen_box_ids.update(forced_box)
    state0.seen_target_ids.update(forced_tgt)

    # 列扫描点 (在 t=0 belief 下)
    scans = _list_scans(state0, forced_box, forced_tgt)
    n_s = len(scans)

    if verbose:
        print(f"  god push count = {n_p}, scans needed = {n_s} (forced {len(forced)})")

    # 模拟 god plan, 记录每步 snapshot (n_p+1 个)
    snapshots = _simulate_god_and_record(eng_init, plan)
    if len(snapshots) < n_p + 1:
        # apply 中途失败 → oracle 不可解
        return None

    # 推 push k 自身的步数 = walk(car_before, push_pos[k]) + 1 (推动作)
    # 推完后 car = snapshot[k+1].car_grid (engine 已知)
    # snapshots[k] = (walls_k, obstacles_k, car_grid_k, car_angle_k)
    push_positions = [_get_push_pos(m) for m in plan]
    after_push_pos = [snapshots[k+1][2] for k in range(n_p)]
    after_push_angle = [snapshots[k+1][3] for k in range(n_p)]

    # 预计算: 每个 step k 的 walls 不同 (TNT 炸墙), obstacles 不同 (box 消失)
    bfs_cache: Dict[Tuple[int, Tuple[int, int]], Dict[Tuple[int, int], int]] = {}

    def walk(step_k: int, src: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        key = (step_k, src)
        if key not in bfs_cache:
            walls_k = snapshots[step_k][0]
            obs = snapshots[step_k][1] - {src}   # src 不算障碍
            bfs_cache[key] = _bfs_from(src, walls_k, obs)
        return bfs_cache[key]

    # DP / Dijkstra:
    #   state = (push_idx, scan_set, last_kind, last_idx)
    #   last_kind: 0=init, 1=after_push_k (k=last_idx), 2=after_scan_j (j=last_idx)
    #   cost = 累积 step count
    #
    # 初始: (0, 0, 0, 0), cost 0 (car at init)
    # 目标: push_idx=n_p, scan_set=full bitmask, last_*=任意
    # 转移:
    #   推下一个 (push_idx < n_p):
    #     src_pos = node_pos(state)
    #     d = walk(push_idx)[src_pos][push_positions[push_idx]] + 1 (push action)
    #     new state = (push_idx+1, scan_set, 1, push_idx)
    #   扫 entity j (j ∉ scan_set):
    #     src_pos = node_pos(state)
    #     vp_j, scan_angle_j = scans[j][0], scans[j][1]
    #     d = walk(push_idx)[src_pos][vp_j] + rot(last_angle → scan_angle_j)
    #     new state = (push_idx, scan_set|{j}, 2, j)

    def node_pos(push_idx: int, lk: int, li: int) -> Tuple[int, int]:
        if lk == 0:
            return snapshots[0][2]      # car init (snapshots[k] = walls, obs, car, angle)
        if lk == 1:
            return after_push_pos[li]
        if lk == 2:
            return scans[li][0]
        raise ValueError

    def node_angle(push_idx: int, lk: int, li: int) -> float:
        if lk == 0:
            return snapshots[0][3]
        if lk == 1:
            return after_push_angle[li]
        if lk == 2:
            return scans[li][1]
        raise ValueError

    full_mask = (1 << n_s) - 1 if n_s > 0 else 0
    init_state = (0, 0, 0, 0)
    dist: Dict[Tuple[int, int, int, int], int] = {init_state: 0}
    parent: Dict[Tuple[int, int, int, int], Optional[Tuple]] = {init_state: None}
    parent_action: Dict[Tuple[int, int, int, int], Optional[Tuple[str, int]]] = {init_state: None}

    pq = [(0, init_state)]
    best_final = INFV
    best_final_state = None
    while pq:
        c, st = heapq.heappop(pq)
        if c > dist.get(st, INFV):
            continue
        push_idx, scan_set, lk, li = st
        if push_idx == n_p and scan_set == full_mask:
            if c < best_final:
                best_final = c
                best_final_state = st
            continue

        src = node_pos(push_idx, lk, li)
        src_a = node_angle(push_idx, lk, li)
        d_tab = walk(push_idx, src)

        # 推下一个
        if push_idx < n_p:
            tgt = push_positions[push_idx]
            wc = d_tab.get(tgt, INFV)
            if wc < INFV:
                cost = c + wc + 1
                new_st = (push_idx + 1, scan_set, 1, push_idx)
                if cost < dist.get(new_st, INFV):
                    dist[new_st] = cost
                    parent[new_st] = st
                    parent_action[new_st] = ("push", push_idx)
                    heapq.heappush(pq, (cost, new_st))

        # 扫一个未扫的
        for j in range(n_s):
            if scan_set & (1 << j):
                continue
            vp = scans[j][0]
            angle = scans[j][1]
            wc = d_tab.get(vp, INFV)
            if wc >= INFV:
                continue
            rot = _rot_steps(src_a, angle)
            cost = c + wc + rot
            new_st = (push_idx, scan_set | (1 << j), 2, j)
            if cost < dist.get(new_st, INFV):
                dist[new_st] = cost
                parent[new_st] = st
                parent_action[new_st] = ("scan", j)
                heapq.heappush(pq, (cost, new_st))

    if best_final_state is None:
        return None

    # 回溯 action 列表
    actions = []
    cur = best_final_state
    while cur is not None and parent_action.get(cur) is not None:
        actions.append(parent_action[cur])
        cur = parent[cur]
    actions.reverse()
    return OracleResult(cost=best_final, interleaving=actions)


# ── 作为 planner 跑 (用 oracle interleaving 实际推动 engine) ───

def planner_oracle(eng: GameEngine) -> None:
    """oracle 既算最优 cost, 也实际重放 — 顺便验证 cost 跟实测吻合."""
    plan = _god_plan(eng, time_limit=30.0)
    if not plan:
        return

    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    forced_box = {i for i, _ in forced}
    forced_tgt = {j for _, j in forced}
    state0.seen_box_ids.update(forced_box)
    state0.seen_target_ids.update(forced_tgt)

    scans = _list_scans(state0, forced_box, forced_tgt)

    # 在 fresh engine 上求最优 interleaving (避免 monkey patch 污染)
    clone = _fresh_engine_from_eng(eng)
    result = oracle_min_steps(clone, god_time_limit=30.0, verbose=False)
    if result is None:
        return
    actions = result.interleaving

    # 在主 eng 上重放
    push_iter = iter(plan)
    for kind, idx in actions:
        if eng.get_state().won:
            return
        if kind == "push":
            move = next(push_iter)
            push_pos = _get_push_pos(move)
            _walk_to_executor(eng, push_pos, "push_walk")
            eng._step_tag = "push"
            eng.discrete_step(direction_to_abs_action(*move[2]))
        elif kind == "scan":
            vp, angle, _, _ = scans[idx]
            _walk_to_executor(eng, vp, "scan_walk")
            _rotate_executor(eng, angle, "scan_rot")


def _walk_to_executor(eng: GameEngine, target: Tuple[int, int], tag: str) -> bool:
    state = eng.get_state()
    obstacles: Set[Tuple[int, int]] = set()
    for b in state.boxes: obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs: obstacles.add(pos_to_grid(bm.x, bm.y))
    car_grid = pos_to_grid(state.car_x, state.car_y)
    if car_grid == target: return True
    path = bfs_path(car_grid, target, state.grid, obstacles)
    if path is None: return False
    eng._step_tag = tag
    for dx, dy in path:
        eng.discrete_step(direction_to_abs_action(dx, dy))
    return True


def _rotate_executor(eng: GameEngine, target_angle: float, tag: str) -> None:
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
