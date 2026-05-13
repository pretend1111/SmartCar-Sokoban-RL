"""Planner oracle v2 — FOV-during-walk auto-reveal + car_angle in state.

关键升级 vs v1:
  1. 状态加 car_angle (4 值: 0=E, π/2=S, π=W, -π/2=N)
  2. 每个 entity 有多个 (trigger_cell, angle) 配置 (engine FOV 严格识别条件)
  3. walk 时遍历路径 cell, 若 (cell, current_angle) 命中某 entity 的 trigger →
     scan_set 自动置位 (0 cost)
  4. explicit scan 仍可用 (walk + rotate)

期望大幅压缩 scan overhead: 车走 push_walk 时顺路识别多个 entity 不再需要专项扫.
"""

from __future__ import annotations

import contextlib
import copy
import heapq
import io
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.solver.explorer import (
    has_line_of_sight, get_entity_obstacles, get_all_entity_positions,
)
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs

INFV = 10**9

# 4 个量化朝向 → angle_q4 ∈ {0,1,2,3} → (dx, dy) trigger 偏移
# angle 0 (E): cell east of car → trigger = entity - (1, 0)
# angle 1 (S, +π/2): cell south of car (engine y+) → trigger = entity - (0, 1)
# angle 2 (W, π): trigger = entity - (-1, 0) = entity + (1, 0)
# angle 3 (N, -π/2): trigger = entity - (0, -1) = entity + (0, 1)
ANGLE_Q4_TO_RAD = {0: 0.0, 1: math.pi/2, 2: math.pi, 3: -math.pi/2}
ANGLE_Q4_TO_UNIT = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def _rad_to_q4(a: float) -> int:
    """engine.car_angle rad → q4 ∈ {0,1,2,3}."""
    n = round((a / (math.pi / 2))) % 4
    # n=0 → E, n=1 → S, n=2 → W or -2, n=-1 or 3 → N
    if n == -1: n = 3
    if n == -2: n = 2
    return n


def _fresh_engine_from_eng(eng: GameEngine) -> GameEngine:
    from experiments.min_steps.planner_best import _BEST_MAP_PATH, _BEST_SEED
    if not _BEST_MAP_PATH:
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


def _bfs_path_with_steps(start: Tuple[int, int], end: Tuple[int, int],
                          walls, obstacles: Set[Tuple[int, int]]
                          ) -> Optional[List[Tuple[int, int]]]:
    """BFS 返回从 start 到 end 经过的中间格子序列 (不含 start, 含 end)."""
    if start == end:
        return []
    rows = len(walls); cols = len(walls[0]) if rows else 0
    parent = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        if c == end:
            break
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc, nr = c[0]+dc, c[1]+dr
            if (nc, nr) in parent: continue
            if not (0 <= nr < rows and 0 <= nc < cols): continue
            if walls[nr][nc] == 1 or (nc, nr) in obstacles: continue
            parent[(nc, nr)] = c
            q.append((nc, nr))
    if end not in parent:
        return None
    # 回溯
    path = []
    cur = end
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _simulate_god_and_record(eng_init: GameEngine, god_plan: List):
    """跑 god plan, 记录每个 push 之前的 (walls, obstacles, car_pos, car_angle,
    seen_box_set, seen_target_set, entity_index_map). 后两个用于"模拟 walk 时
    哪些 entity 还没被引擎识别(那些就是 oracle 待 scan)."""
    e = _fresh_engine_from_eng(eng_init)
    snapshots = []
    state = e.get_state()
    obs = set()
    for b in state.boxes: obs.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs: obs.add(pos_to_grid(bm.x, bm.y))
    # entity index map (snapshots[k]['box_at_pos'] = idx) — 但 box 推动后位置变
    snapshots.append({
        "walls": [row[:] for row in state.grid],
        "obstacles": obs,
        "car": pos_to_grid(state.car_x, state.car_y),
        "angle": state.car_angle,
        "boxes": [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes],
        "targets": [(pos_to_grid(t.x, t.y), t.num_id) for t in state.targets],
    })
    from experiments.sage_pr.build_dataset_v3 import apply_solver_move
    for move in god_plan:
        if not apply_solver_move(e, move):
            return snapshots
        state = e.get_state()
        obs = set()
        for b in state.boxes: obs.add(pos_to_grid(b.x, b.y))
        for bm in state.bombs: obs.add(pos_to_grid(bm.x, bm.y))
        snapshots.append({
            "walls": [row[:] for row in state.grid],
            "obstacles": obs,
            "car": pos_to_grid(state.car_x, state.car_y),
            "angle": state.car_angle,
            "boxes": [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes],
            "targets": [(pos_to_grid(t.x, t.y), t.num_id) for t in state.targets],
        })
    return snapshots


def _build_entity_triggers(initial_state, forced_box, forced_tgt):
    """返回 scan 表: 每个待识别 entity 的 (4 个 trigger 配置).

    Returns: List of dicts {
        "etype": "box"/"target",
        "entity_pos": (col, row),   # entity 自身位置 (用于排除)
        "triggers": [(trigger_cell, angle_q4), ...],   # 4 个候选
    }
    """
    car_grid = pos_to_grid(initial_state.car_x, initial_state.car_y)
    obstacles = get_entity_obstacles(initial_state)
    entity_pos = get_all_entity_positions(initial_state)
    walls = initial_state.grid

    entities = []

    # box — 排除律: N-1 个 (跳最远)
    unseen_b = [i for i in range(len(initial_state.boxes))
                 if i not in initial_state.seen_box_ids and i not in forced_box]
    if len(unseen_b) >= 2:
        with_dist = []
        for i in unseen_b:
            b = initial_state.boxes[i]
            bg = pos_to_grid(b.x, b.y)
            d = abs(car_grid[0]-bg[0]) + abs(car_grid[1]-bg[1])
            with_dist.append((d, i, bg))
        with_dist.sort()
        for _, i, bg in with_dist[:-1]:
            entities.append({"etype": "box", "eidx": i, "entity_pos": bg})

    # target
    unseen_t = [i for i in range(len(initial_state.targets))
                 if i not in initial_state.seen_target_ids and i not in forced_tgt]
    if len(unseen_t) >= 2:
        with_dist = []
        for i in unseen_t:
            t = initial_state.targets[i]
            tg = pos_to_grid(t.x, t.y)
            d = abs(car_grid[0]-tg[0]) + abs(car_grid[1]-tg[1])
            with_dist.append((d, i, tg))
        with_dist.sort()
        for _, i, tg in with_dist[:-1]:
            entities.append({"etype": "target", "eidx": i, "entity_pos": tg})

    # 为每个 entity 计算 4 个 trigger 配置 (车在哪个格 + 朝哪个角度)
    rows = len(walls); cols = len(walls[0]) if rows else 0
    for ent in entities:
        ec, er = ent["entity_pos"]
        triggers = []
        for q in range(4):
            dx, dy = ANGLE_Q4_TO_UNIT[q]
            # 车要在 entity - unit_q 位置, 面向 q. 即 trigger_cell = (ec-dx, er-dy)
            tc = (ec - dx, er - dy)
            tc_col, tc_row = tc
            # trigger_cell 必须是非墙非该 entity 的格子, 且有视线
            if not (0 <= tc_row < rows and 0 <= tc_col < cols):
                continue
            if walls[tc_row][tc_col] == 1:
                continue
            if tc == (ec, er):
                continue
            # 视线检查 (排除 entity 自身)
            ent_set = entity_pos - {(ec, er)}
            if not has_line_of_sight(tc_col, tc_row, ec, er, walls, ent_set):
                continue
            triggers.append((tc, q))
        ent["triggers"] = triggers
    return entities


def _walk_reveals(path: List[Tuple[int, int]], car_angle_q4: int,
                   trigger_map: Dict[Tuple[Tuple[int, int], int], Set[int]]
                   ) -> Set[int]:
    """模拟一段 walk 的 cell 序列, 检查每个 cell 是否触发某 entity 的 trigger
    (angle_q4 不变, 因为 abs translation 不旋转).
    返回 walk 中被自动识别的 entity_set_indices."""
    revealed = set()
    for cell in path:
        ent_indices = trigger_map.get((cell, car_angle_q4))
        if ent_indices:
            revealed.update(ent_indices)
    return revealed


@dataclass
class OracleResult2:
    cost: int
    interleaving: List[Tuple[str, int]]


def oracle_v2_min_steps(eng_init: GameEngine,
                          *, god_time_limit: float = 30.0) -> Optional[OracleResult2]:
    plan = _god_plan(eng_init, time_limit=god_time_limit)
    if not plan:
        return None
    n_p = len(plan)

    state0 = eng_init.get_state()
    forced = find_forced_pairs(state0)
    forced_box = {i for i, _ in forced}
    forced_tgt = {j for _, j in forced}
    state0.seen_box_ids.update(forced_box)
    state0.seen_target_ids.update(forced_tgt)

    entities = _build_entity_triggers(state0, forced_box, forced_tgt)
    n_s = len(entities)

    snapshots = _simulate_god_and_record(eng_init, plan)
    if len(snapshots) < n_p + 1:
        return None

    push_positions = [_get_push_pos(m) for m in plan]
    # after_push 状态从 snapshots[k+1]
    # angle 不变 (push 是 abs translation)
    # car_pos 变到 snapshots[k+1]["car"]

    # trigger 表: 每个 step_k 都有可能不一样 (entity 推走后, trigger 失效).
    # 简化: 用初始 state 算 trigger, 但在每 step_k 检查 entity 是否还在原位.
    # 进一步简化: forced entity 已识别, 没被推走; 其他 entity 是 target / 还没被
    # 推到的 box. target 永不动; box 在推完特定 step 后才动. 这里都按初始位置.
    # 对 box scan: 若被推走, 该 scan 自动失效, oracle 不能再选它. 处理:
    # 给每个 entity 计算"何时该 entity 还在原位"的最大 push_idx.

    # 简化: 假设 entity 在 oracle 决策的时间窗内不变 (box 在被推前都在原位).
    # box 第一次被推的 step = ? 找 god_plan 里第一个 move 涉及该 box.

    box_first_push_step = {}   # entity idx → step it first moved
    for i, m in enumerate(plan):
        etype, eid, dir_, _ = m
        if etype == "box":
            old_pos, _ = eid
            for ent in entities:
                if ent["etype"] == "box" and ent["entity_pos"] == old_pos:
                    if ent["eidx"] not in box_first_push_step:
                        box_first_push_step[ent["eidx"]] = i
                    break

    # 对于 box entity j, 在 push_idx > first_push_step 之后, entity 位置变化,
    # scan trigger / 显式 scan 都失效. 但我们仍可以在 push_idx <= first_push_step
    # 时识别它. (实际上 push 走 walk 也是基于 push_idx 之前的状态.)

    # trigger map: (cell, angle_q4) → set of entity indices (in `entities` list).
    # 但 trigger 的有效性 depends on step_k. Simpler: 让 DP 用 step_k-specific
    # filter.

    # 先建一个完全的 trigger map (忽略 step_k), 后续在 walk 模拟时 filter.
    full_trigger_map: Dict[Tuple[Tuple[int, int], int], Set[int]] = {}
    for j, ent in enumerate(entities):
        for (tc, q) in ent["triggers"]:
            full_trigger_map.setdefault((tc, q), set()).add(j)

    # 帮助函数: 给定 step_k, src 位置, 车 angle, 目标位置, 返回 (walk_len, revealed_set).
    bfs_path_cache: Dict[Tuple[int, Tuple[int, int], Tuple[int, int]], Optional[List]] = {}

    def get_walk(step_k: int, src: Tuple[int, int], dst: Tuple[int, int]):
        key = (step_k, src, dst)
        if key not in bfs_path_cache:
            walls = snapshots[step_k]["walls"]
            obs = snapshots[step_k]["obstacles"] - {src, dst}
            bfs_path_cache[key] = _bfs_path_with_steps(src, dst, walls, obs)
        return bfs_path_cache[key]

    def walk_reveals_step(step_k: int, path: List[Tuple[int, int]], car_q4: int) -> Set[int]:
        """模拟 walk, 返回 (revealed entity indices), 注意 box entity 在
        push_idx > first_push_step 之后位置变了, 不再有效 trigger."""
        revealed = set()
        for cell in path:
            ents = full_trigger_map.get((cell, car_q4))
            if not ents: continue
            for j in ents:
                ent = entities[j]
                if ent["etype"] == "box":
                    fps = box_first_push_step.get(ent["eidx"], -1)
                    if step_k > fps:  # box 已被推走
                        continue
                revealed.add(j)
        return revealed

    # explicit scan: 每个 entity 列 ALL triggers, DP 选最便宜的.
    scans_simple = []   # 每元素 = list of (vp, q4) options
    for ent in entities:
        scans_simple.append(ent["triggers"])   # list of (cell, q4) tuples

    # DP state = (push_idx, scan_set, last_pos, car_q4)
    full_mask = (1 << n_s) - 1 if n_s > 0 else 0
    init_q4 = _rad_to_q4(snapshots[0]["angle"])
    init_pos = snapshots[0]["car"]

    # 初始时 car 站着可能就触发 trigger (0-step reveal)
    init_reveals = walk_reveals_step(0, [init_pos], init_q4)
    init_mask = 0
    for j in init_reveals:
        init_mask |= (1 << j)

    init_state = (0, init_mask, init_pos, init_q4)
    dist = {init_state: 0}
    parent: Dict[Tuple, Optional[Tuple]] = {init_state: None}
    parent_act: Dict[Tuple, Optional[Tuple[str, int]]] = {init_state: None}

    pq = [(0, init_state)]
    best_cost = INFV
    best_final = None

    while pq:
        c, st = heapq.heappop(pq)
        if c > dist.get(st, INFV):
            continue
        push_idx, scan_set, last_pos, q4 = st
        if push_idx == n_p and scan_set == full_mask:
            if c < best_cost:
                best_cost = c
                best_final = st
            continue

        # 转移 1: 推下一个 push
        if push_idx < n_p:
            tgt = push_positions[push_idx]
            path = get_walk(push_idx, last_pos, tgt)
            if path is not None:
                walk_len = len(path)
                reveals = walk_reveals_step(push_idx, path, q4)
                new_mask = scan_set
                for j in reveals:
                    new_mask |= (1 << j)
                # 推完后 car 位置: god 模拟下是 snapshots[push_idx+1]["car"]
                new_pos = snapshots[push_idx+1]["car"]
                new_cost = c + walk_len + 1
                new_st = (push_idx+1, new_mask, new_pos, q4)
                if new_cost < dist.get(new_st, INFV):
                    dist[new_st] = new_cost
                    parent[new_st] = st
                    parent_act[new_st] = ("push", push_idx)
                    heapq.heappush(pq, (new_cost, new_st))

        # 转移 2: 显式 scan entity j (每个 entity 可能多个 trigger 选项)
        for j in range(n_s):
            if scan_set & (1 << j): continue
            ent = entities[j]
            # box scan 在被推走后失效
            if ent["etype"] == "box":
                fps = box_first_push_step.get(ent["eidx"], -1)
                if push_idx > fps: continue
            options = scans_simple[j]
            if not options: continue
            for (vp, scan_q4) in options:
                path = get_walk(push_idx, last_pos, vp)
                if path is None: continue
                walk_len = len(path)
                reveals = walk_reveals_step(push_idx, path, q4)
                new_mask = scan_set
                for k in reveals:
                    new_mask |= (1 << k)
                # rotate
                rot_n = abs((scan_q4 - q4) % 4)
                rot_n = min(rot_n, 4 - rot_n)
                new_mask |= (1 << j)
                new_cost = c + walk_len + rot_n
                new_st = (push_idx, new_mask, vp, scan_q4)
                if new_cost < dist.get(new_st, INFV):
                    dist[new_st] = new_cost
                    parent[new_st] = st
                    parent_act[new_st] = ("scan", j)
                    heapq.heappush(pq, (new_cost, new_st))

    if best_final is None:
        return None

    actions = []
    cur = best_final
    while cur is not None and parent_act.get(cur) is not None:
        actions.append(parent_act[cur])
        cur = parent[cur]
    actions.reverse()
    return OracleResult2(cost=best_cost, interleaving=actions)


# Planner 包装
def planner_oracle_v2(eng: GameEngine) -> None:
    plan = _god_plan(eng, time_limit=30.0)
    if not plan:
        return
    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    forced_box = {i for i, _ in forced}
    forced_tgt = {j for _, j in forced}
    state0.seen_box_ids.update(forced_box)
    state0.seen_target_ids.update(forced_tgt)
    entities = _build_entity_triggers(state0, forced_box, forced_tgt)

    clone = _fresh_engine_from_eng(eng)
    res = oracle_v2_min_steps(clone)
    if res is None:
        # fallback: 退到 v1 oracle 行为 (无 reveal)
        from experiments.min_steps.planner_oracle import planner_oracle
        planner_oracle(eng)
        return

    # 重放 (跟 v1 的 planner_oracle 一样, 用 scans_simple 的 vp/angle 执行 scan)
    scans_exec = []
    for ent in entities:
        if ent["triggers"]:
            tc, q = ent["triggers"][0]
            scans_exec.append((tc, ANGLE_Q4_TO_RAD[q]))
        else:
            scans_exec.append(None)

    push_iter = iter(plan)
    for kind, idx in res.interleaving:
        if eng.get_state().won:
            return
        if kind == "push":
            move = next(push_iter)
            push_pos = _get_push_pos(move)
            _walk_to(eng, push_pos, "push_walk")
            eng._step_tag = "push"
            eng.discrete_step(direction_to_abs_action(*move[2]))
        elif kind == "scan":
            sc = scans_exec[idx]
            if sc is None: continue
            vp, scan_angle = sc
            _walk_to(eng, vp, "scan_walk")
            _rotate(eng, scan_angle, "scan_rot")


def _walk_to(eng: GameEngine, target: Tuple[int, int], tag: str) -> bool:
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


def _rotate(eng: GameEngine, target_angle: float, tag: str) -> None:
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
