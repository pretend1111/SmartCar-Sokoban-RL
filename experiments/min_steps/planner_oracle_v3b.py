"""Planner oracle v3b — 极简增量补丁: 完全沿用 v1 oracle 的 DP 结构,
只在每次 push transition 时检查 walk 中是否经过 init_q4 朝向的 trigger cell,
若有, scan_set 置位. 不在 state 加 q4 字段.

预期: v3b 至少 == v1 (因为它只是减少需要的 scan 数), 而且经常 < v1.
"""

from __future__ import annotations

import contextlib
import copy
import heapq
import io
import math
import random
from typing import Dict, List, Optional, Set, Tuple

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs

from experiments.min_steps.planner_oracle import (
    _god_plan, _bfs_from, _list_scans, _get_push_pos,
    _rot_steps, _simulate_god_and_record, _fresh_engine_from_eng,
)
from experiments.min_steps.planner_oracle_v3 import (
    _build_walk_trigger_map, _rad_to_q4,
)
from collections import deque

INFV = 10**9


def _bfs_path_cells(start, end, walls, obstacles):
    """Return [intermediate, ..., end] (no start). None if unreachable."""
    if start == end:
        return []
    rows = len(walls); cols = len(walls[0]) if rows else 0
    parent = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        if c == end: break
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc, nr = c[0]+dc, c[1]+dr
            if (nc, nr) in parent: continue
            if not (0 <= nr < rows and 0 <= nc < cols): continue
            if walls[nr][nc] == 1 or (nc, nr) in obstacles: continue
            parent[(nc, nr)] = c
            q.append((nc, nr))
    if end not in parent: return None
    path = []
    cur = end
    while cur != start:
        path.append(cur); cur = parent[cur]
    path.reverse()
    return path


def oracle_v3b_min_steps(eng_init: GameEngine, plan: Optional[List] = None,
                          *, god_time_limit: float = 30.0):
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

    init_q4 = _rad_to_q4(snapshots[0][3])
    trigger_map = _build_walk_trigger_map(scans, state0)

    # box first-push step
    box_first_push_step: Dict[int, int] = {}
    for i, m in enumerate(plan):
        etype, eid, dir_, _ = m
        if etype == "box":
            old_pos, _ = eid
            for j, sc in enumerate(scans):
                _, _, sc_et, sc_ei = sc
                if sc_et == "box":
                    b = state0.boxes[sc_ei]
                    if pos_to_grid(b.x, b.y) == old_pos:
                        if j not in box_first_push_step:
                            box_first_push_step[j] = i

    def walk_reveals(path, step_k):
        """ALWAYS use init_q4 — sound approximation: walks at non-init angle
        might reveal MORE entities, but we ignore that (no harm to v1's optimum)."""
        revealed = set()
        for cell in path:
            ents = trigger_map.get((cell, init_q4))
            if not ents: continue
            # 不做 box-moved 检查 — 跟 v1 oracle 行为一致 (v1 也允许 "无效" scan)
            revealed.update(ents)
        return revealed

    walk_cache: Dict[Tuple[int, Tuple[int, int], Tuple[int, int]],
                     Tuple[int, Set[int]]] = {}

    def walk(step_k, src, dst):
        """Returns (walk_len, revealed_scans). None if unreachable."""
        key = (step_k, src, dst)
        if key in walk_cache:
            return walk_cache[key]
        if src == dst:
            walk_cache[key] = (0, set())
            return walk_cache[key]
        walls_k = snapshots[step_k][0]
        # 跟 replay BFS 一致: 仅排除 src (dst 必须真的可达, 不能假装它不是障碍)
        obs = snapshots[step_k][1] - {src}
        path = _bfs_path_cells(src, dst, walls_k, obs)
        if path is None:
            walk_cache[key] = None
            return None
        walk_len = len(path)
        revealed = walk_reveals(path, step_k)
        walk_cache[key] = (walk_len, revealed)
        return walk_cache[key]

    # v1's state structure: (push_idx, scan_set, last_kind, last_idx)
    # last_kind: 0=init 1=after_push 2=after_scan
    # car angle: 0=init_angle (init), engine init angle after pushes (= init_angle anyway in v1's incorrect model), scan_angle after scan
    init_pos = snapshots[0][2]
    # 初始 reveals (0-len walk)
    init_reveals = walk_reveals([init_pos], 0)
    init_mask = 0
    for j in init_reveals: init_mask |= (1 << j)

    init_state = (0, init_mask, 0, 0)
    dist = {init_state: 0}
    parent: Dict[Tuple, Optional[Tuple]] = {init_state: None}
    parent_act: Dict[Tuple, Optional[Tuple[str, int]]] = {init_state: None}
    full_mask = (1 << n_s) - 1 if n_s > 0 else 0

    def node_pos(lk, li):
        if lk == 0: return init_pos
        if lk == 1: return after_push_pos[li]
        if lk == 2: return scans[li][0]
        return None

    def node_angle(lk, li):
        if lk == 0: return snapshots[0][3]
        if lk == 1: return after_push_angle[li]
        if lk == 2: return scans[li][1]
        return 0.0

    pq = [(0, init_state)]
    best = INFV; best_st = None
    while pq:
        c, st = heapq.heappop(pq)
        if c > dist.get(st, INFV): continue
        push_idx, scan_set, lk, li = st
        if push_idx == n_p and scan_set == full_mask:
            if c < best: best, best_st = c, st
            continue
        src = node_pos(lk, li)
        if src is None: continue
        src_a = node_angle(lk, li)

        # 推下一个
        if push_idx < n_p:
            res = walk(push_idx, src, push_positions[push_idx])
            if res is not None:
                wlen, reveals = res
                new_mask = scan_set
                for j in reveals: new_mask |= (1 << j)
                # 推完位置也 check trigger
                after_reveals = walk_reveals([after_push_pos[push_idx]], push_idx+1)
                for j in after_reveals: new_mask |= (1 << j)
                cost = c + wlen + 1
                new_st = (push_idx+1, new_mask, 1, push_idx)
                if cost < dist.get(new_st, INFV):
                    dist[new_st] = cost
                    parent[new_st] = st
                    parent_act[new_st] = ("push", push_idx)
                    heapq.heappush(pq, (cost, new_st))

        # 显式 scan (跟 v1 一致, 不做 box-moved 检查)
        for j in range(n_s):
            if scan_set & (1 << j): continue
            vp = scans[j][0]
            res = walk(push_idx, src, vp)
            if res is None: continue
            wlen, reveals = res
            rot = _rot_steps(src_a, scans[j][1])
            new_mask = scan_set
            for k in reveals: new_mask |= (1 << k)
            new_mask |= (1 << j)
            cost = c + wlen + rot
            new_st = (push_idx, new_mask, 2, j)
            if cost < dist.get(new_st, INFV):
                dist[new_st] = cost
                parent[new_st] = st
                parent_act[new_st] = ("scan", j)
                heapq.heappush(pq, (cost, new_st))

    if best_st is None: return None
    actions = []
    cur = best_st
    while cur is not None and parent_act.get(cur) is not None:
        actions.append(parent_act[cur])
        cur = parent[cur]
    actions.reverse()
    from experiments.min_steps.planner_oracle_v3 import OracleResult3
    return OracleResult3(cost=best, interleaving=actions)


def planner_oracle_v3b(eng: GameEngine) -> None:
    plan = _god_plan(eng, time_limit=30.0)
    if not plan: return
    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    state0.seen_box_ids.update(fb); state0.seen_target_ids.update(ft)
    scans = _list_scans(state0, fb, ft)

    clone = _fresh_engine_from_eng(eng)
    res = oracle_v3b_min_steps(clone, plan=plan)
    if res is None:
        from experiments.min_steps.planner_oracle import planner_oracle
        planner_oracle(eng); return

    from experiments.min_steps.planner_oracle import (
        _walk_to_executor, _rotate_executor,
    )
    push_iter = iter(plan)
    for kind, idx in res.interleaving:
        if eng.get_state().won: return
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
