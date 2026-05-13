"""Planner oracle v3 — minimal patch over v1: walk-reveal during push walks.

跟 v1 不同: 推每个 push 时, 模拟 BFS path 的每个 cell, 检查 car_angle 下是否
触发某个 entity 的 4-adj trigger. 若是, scan_set 自动置位. 显式 scan 仅在剩余
没被 auto-revealed 的 entity 上做.

跟 v2 不同 (避免回归):
  - 仅 1 个 viewpoint per entity (沿用 find_observation_point)
  - 维持 v1 的 state 结构 (push_idx, scan_set, last_kind, last_idx)
  - 把 car_q4 加进 state — 但 last_kind=1 (push) 后 q4 不变,
                        last_kind=2 (scan) 后 q4=scan_q4

实际上 car_q4 可从 (last_kind, last_idx) 推: init=init_q4, push 后保持 prior_q4,
scan 后 = scans[li].q4. 在 DP 转移时显式传播.
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
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs

from experiments.min_steps.planner_oracle import (
    _god_plan, _bfs_from, _list_scans, _get_push_pos,
    _heading_to_angle, _rot_steps, _simulate_god_and_record,
    _fresh_engine_from_eng,
)

INFV = 10**9


def _rad_to_q4(a: float) -> int:
    n = round((a / (math.pi / 2))) % 4
    if n == -1: n = 3
    if n == -2: n = 2
    return n


# Walk-reveal trigger map: for each (cell, q4_facing) → set of scan indices revealed.
# We build it from the entity positions + scan list. A scan = (vp, angle, etype, eidx).
# The scan's own (vp, q4(angle)) is ONE trigger. But we also add the OTHER 3 angle
# triggers (entity ± unit_vec) so that walks at different angles can still pick up
# the entity for free.
def _build_walk_trigger_map(scans, state):
    """Returns dict[(cell, q4)] -> set of scan indices. Plus an "alt cell" indicator
    so we don't 重复 detect the explicit scan."""
    triggers: Dict[Tuple[Tuple[int, int], int], Set[int]] = {}
    walls = state.grid
    rows = len(walls); cols = len(walls[0]) if rows else 0
    from smartcar_sokoban.solver.explorer import get_all_entity_positions, has_line_of_sight
    entity_pos = get_all_entity_positions(state)

    for j, scan in enumerate(scans):
        vp, angle, etype, eidx = scan
        # Find entity world position
        if etype == "box":
            b = state.boxes[eidx]
            ep = pos_to_grid(b.x, b.y)
        else:
            t = state.targets[eidx]
            ep = pos_to_grid(t.x, t.y)

        # 4 possible (vp, q4) configurations to identify this entity
        for q in range(4):
            unit_map = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
            dx, dy = unit_map[q]
            tc = (ep[0] - dx, ep[1] - dy)
            # validate
            if not (0 <= tc[1] < rows and 0 <= tc[0] < cols):
                continue
            if walls[tc[1]][tc[0]] == 1:
                continue
            if tc == ep:
                continue
            ent_set = entity_pos - {ep}
            if not has_line_of_sight(tc[0], tc[1], ep[0], ep[1], walls, ent_set):
                continue
            triggers.setdefault((tc, q), set()).add(j)
    return triggers


@dataclass
class OracleResult3:
    cost: int
    interleaving: List[Tuple[str, int]]


def oracle_v3_min_steps(eng_init: GameEngine, plan: Optional[List] = None,
                          *, god_time_limit: float = 30.0
                          ) -> Optional[OracleResult3]:
    if plan is None:
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
    scans = _list_scans(state0, forced_box, forced_tgt)
    n_s = len(scans)

    snapshots = _simulate_god_and_record(eng_init, plan)
    if len(snapshots) < n_p + 1:
        return None

    push_positions = [_get_push_pos(m) for m in plan]
    after_push_pos = [snapshots[k+1][2] for k in range(n_p)]
    after_push_angle = [snapshots[k+1][3] for k in range(n_p)]

    # walk-reveal trigger map (uses initial state, valid for boxes only before they move)
    trigger_map = _build_walk_trigger_map(scans, state0)

    # box first-push step: scan invalid (entity moved) after this step
    box_first_push_step: Dict[int, int] = {}
    for i, m in enumerate(plan):
        etype, eid, dir_, _ = m
        if etype == "box":
            old_pos, _ = eid
            for j, sc in enumerate(scans):
                _, _, sc_etype, sc_eidx = sc
                if sc_etype == "box":
                    box = state0.boxes[sc_eidx]
                    if (box.col if hasattr(box, 'col') else pos_to_grid(box.x, box.y)[0],
                        box.row if hasattr(box, 'row') else pos_to_grid(box.x, box.y)[1]
                        ) == old_pos or pos_to_grid(box.x, box.y) == old_pos:
                        if j not in box_first_push_step:
                            box_first_push_step[j] = i

    bfs_path_cache: Dict[Tuple[int, Tuple[int, int], Tuple[int, int]], Optional[List[Tuple[int, int]]]] = {}

    def get_walk_path(step_k: int, src: Tuple[int, int], dst: Tuple[int, int]):
        key = (step_k, src, dst)
        if key in bfs_path_cache:
            return bfs_path_cache[key]
        walls_k = snapshots[step_k][0]
        obs = snapshots[step_k][1] - {src, dst}
        # BFS path that returns intermediate + dst cells (no src)
        if src == dst:
            bfs_path_cache[key] = []
            return []
        rows = len(walls_k); cols = len(walls_k[0]) if rows else 0
        parent = {src: None}
        q = deque([src])
        while q:
            c = q.popleft()
            if c == dst:
                break
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc, nr = c[0]+dc, c[1]+dr
                if (nc, nr) in parent: continue
                if not (0 <= nr < rows and 0 <= nc < cols): continue
                if walls_k[nr][nc] == 1 or (nc, nr) in obs: continue
                parent[(nc, nr)] = c
                q.append((nc, nr))
        if dst not in parent:
            bfs_path_cache[key] = None
            return None
        path = []
        cur = dst
        while cur != src:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        bfs_path_cache[key] = path
        return path

    def walk_reveals(path, q4, step_k):
        revealed = set()
        for cell in path:
            ents = trigger_map.get((cell, q4))
            if not ents: continue
            for j in ents:
                # box scan invalid after first push of that box
                if scans[j][2] == "box":
                    fps = box_first_push_step.get(j, -1)
                    if step_k > fps:
                        continue
                revealed.add(j)
        return revealed

    # DP state: (push_idx, scan_set, last_kind, last_idx, car_q4)
    # last_kind: 0=init 1=after_push 2=after_scan
    # car_q4: 当前车朝向
    init_pos = snapshots[0][2]
    init_q4 = _rad_to_q4(snapshots[0][3])
    # 初始 reveal at init position (0-len walk)
    init_reveals = walk_reveals([init_pos], init_q4, 0)
    init_mask = 0
    for j in init_reveals:
        init_mask |= (1 << j)
    init_state = (0, init_mask, 0, 0, init_q4)
    dist = {init_state: 0}
    parent: Dict[Tuple, Optional[Tuple]] = {init_state: None}
    parent_act: Dict[Tuple, Optional[Tuple[str, int]]] = {init_state: None}
    full_mask = (1 << n_s) - 1 if n_s > 0 else 0

    def node_pos(lk, li):
        if lk == 0: return init_pos
        if lk == 1: return after_push_pos[li]
        if lk == 2: return scans[li][0]
        return None

    pq = [(0, init_state)]
    best_cost = INFV
    best_final = None
    while pq:
        c, st = heapq.heappop(pq)
        if c > dist.get(st, INFV):
            continue
        push_idx, scan_set, lk, li, q4 = st
        if push_idx == n_p and scan_set == full_mask:
            if c < best_cost:
                best_cost = c
                best_final = st
            continue

        src = node_pos(lk, li)
        if src is None: continue

        # 转移 1: 推下一个 push
        if push_idx < n_p:
            tgt = push_positions[push_idx]
            path = get_walk_path(push_idx, src, tgt)
            if path is not None:
                walk_len = len(path)
                reveals = walk_reveals(path, q4, push_idx)
                new_mask = scan_set
                for j in reveals:
                    new_mask |= (1 << j)
                new_cost = c + walk_len + 1
                # push action 自身在 push_pos → after_push_pos, 也可能 trigger reveal
                # (推动 box 后 car 移动 1 步, 经过 push_pos 已计入 path). 推后位置
                # 是 after_push_pos[push_idx], 这是新位置.
                # 推后的位置 = after_push_pos. 是否也算 trigger? 推后 angle 不变.
                # 把 after_push_pos 也加进 reveal check.
                after_reveals = walk_reveals([after_push_pos[push_idx]], q4, push_idx+1)
                for j in after_reveals:
                    new_mask |= (1 << j)
                new_st = (push_idx+1, new_mask, 1, push_idx, q4)
                if new_cost < dist.get(new_st, INFV):
                    dist[new_st] = new_cost
                    parent[new_st] = st
                    parent_act[new_st] = ("push", push_idx)
                    heapq.heappush(pq, (new_cost, new_st))

        # 转移 2: 显式 scan
        for j in range(n_s):
            if scan_set & (1 << j): continue
            # box scan 已被推走 → invalid
            if scans[j][2] == "box":
                fps = box_first_push_step.get(j, -1)
                if push_idx > fps: continue
            vp = scans[j][0]
            scan_q4 = _rad_to_q4(scans[j][1])
            path = get_walk_path(push_idx, src, vp)
            if path is None: continue
            walk_len = len(path)
            reveals = walk_reveals(path, q4, push_idx)
            new_mask = scan_set
            for k in reveals:
                new_mask |= (1 << k)
            # 旋转 cost
            rot_n = abs((scan_q4 - q4) % 4)
            rot_n = min(rot_n, 4 - rot_n)
            new_mask |= (1 << j)
            new_cost = c + walk_len + rot_n
            new_st = (push_idx, new_mask, 2, j, scan_q4)
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
    return OracleResult3(cost=best_cost, interleaving=actions)


def planner_oracle_v3(eng: GameEngine) -> None:
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

    clone = _fresh_engine_from_eng(eng)
    res = oracle_v3_min_steps(clone, plan=plan)
    if res is None:
        from experiments.min_steps.planner_oracle import planner_oracle
        planner_oracle(eng)
        return

    from experiments.min_steps.planner_oracle import (
        _walk_to_executor, _rotate_executor,
    )

    push_iter = iter(plan)
    for kind, idx in res.interleaving:
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
