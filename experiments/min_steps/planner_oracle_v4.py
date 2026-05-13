"""Planner oracle v4 — full state tracking + multi-vp + proactive rotation.

State: (push_idx, scan_set, last_pos, car_q4)
  push_idx: god plan pointer 0..n_p
  scan_set: bitmask of scanned entities
  last_pos: (col, row)
  car_q4: ∈ {0,1,2,3} = E/S/W/N

Actions:
  - Push next: walk to push_pos[k] at current q4 → reveal during walk. cost = walk + 1.
    After push, car at after_push_pos[k], q4 unchanged.
  - Scan entity j via trigger (vp, vp_q4): walk to vp at current q4 → reveal during walk +
    rotate to vp_q4. cost = walk + rot. After, car at vp, q4 = vp_q4.
  - Rotate ±90 (no walk): cost 1, new q4 = (q4+1) or (q4-1) mod 4.
    After rotation, check 0-len reveal at current cell.

Replay uses chosen vp from interleaving — no mismatch with DP.
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
from smartcar_sokoban.solver.explorer import (
    has_line_of_sight, get_all_entity_positions,
)

from experiments.min_steps.planner_oracle import (
    _god_plan, _list_scans, _get_push_pos, _simulate_god_and_record,
    _fresh_engine_from_eng, _walk_to_executor, _rotate_executor,
)
from experiments.min_steps.planner_oracle_v3 import _rad_to_q4

INFV = 10**9
Q4_RAD = {0: 0.0, 1: math.pi/2, 2: math.pi, 3: -math.pi/2}
Q4_UNIT = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def _build_entity_vps(scans, state0):
    """For each scan, list ALL valid trigger configs (vp_cell, vp_q4).
    Returns list of lists: vps_per_scan[j] = [(cell, q4), ...]."""
    walls = state0.grid
    rows = len(walls); cols = len(walls[0]) if rows else 0
    entity_pos = get_all_entity_positions(state0)

    vps_per_scan = []
    for scan in scans:
        _, _, etype, eidx = scan
        if etype == "box":
            b = state0.boxes[eidx]; ep = pos_to_grid(b.x, b.y)
        else:
            t = state0.targets[eidx]; ep = pos_to_grid(t.x, t.y)
        opts = []
        for q in range(4):
            dx, dy = Q4_UNIT[q]
            vp = (ep[0] - dx, ep[1] - dy)
            if not (0 <= vp[1] < rows and 0 <= vp[0] < cols): continue
            if walls[vp[1]][vp[0]] == 1: continue
            if vp == ep: continue
            ent_set = entity_pos - {ep}
            if not has_line_of_sight(vp[0], vp[1], ep[0], ep[1], walls, ent_set):
                continue
            opts.append((vp, q))
        vps_per_scan.append(opts)
    return vps_per_scan


def _bfs_path_cells(start, end, walls, obstacles):
    """Returns [intermediate, ..., end] (no start). None if unreachable."""
    if start == end: return []
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
    path = []; cur = end
    while cur != start:
        path.append(cur); cur = parent[cur]
    path.reverse()
    return path


# (dx, dy, q4) — q4: 0=east, 1=south, 2=west, 3=north
_DIRS_Q4 = [(1, 0, 0), (0, 1, 1), (-1, 0, 2), (0, -1, 3)]


def _bfs_path_cells_optimal(start, end, walls, obstacles,
                              trigger_map=None, scan_set=0,
                              prefer_end_q4=None):
    """最短路径 + 三层 tiebreak:
      (1) 路径上 walk-reveal 新增 entity 数最大 (相对于 scan_set, 用 bitmask 去重)
      (2) 抵达 end 时朝向匹配 prefer_end_q4 (省 rotate 步)
      (3) 字典序最小 (确定性)

    返回 [intermediate, ..., end] (含 end, 不含 start). 不可达返回 None.
    """
    if start == end: return []
    rows = len(walls); cols = len(walls[0]) if rows else 0
    # ── BFS 计算最短距离
    dist = {start: 0}
    q = deque([start])
    while q:
        c = q.popleft()
        if c == end: continue
        d = dist[c]
        for dx, dy, _ in _DIRS_Q4:
            nc = (c[0]+dx, c[1]+dy)
            if not (0 <= nc[1] < rows and 0 <= nc[0] < cols): continue
            if walls[nc[1]][nc[0]] == 1: continue
            if nc in obstacles: continue
            if nc in dist: continue
            dist[nc] = d + 1
            q.append(nc)
    if end not in dist: return None
    L = dist[end]
    # ── 分层 DP: 在 (cell, last_q4) 状态空间中找 max (reveal_bits popcount, end_match)
    # State key: (cell, last_q4). last_q4 = -1 表示起点 (无方向)
    best = {(start, -1): (0, 0, None)}   # (reveal_bits, end_match_flag, parent_state)
    layer = [(start, -1)]
    for d in range(L):
        next_states = {}
        for st in layer:
            cell, _ = st
            rev_acc, _, _ = best[st]
            for dx, dy, q4 in _DIRS_Q4:
                nc = (cell[0]+dx, cell[1]+dy)
                if nc not in dist: continue
                if dist[nc] != d + 1: continue
                add = 0
                if trigger_map is not None:
                    ents = trigger_map.get((nc, q4))
                    if ents:
                        for j in ents:
                            add |= (1 << j)
                    add &= ~scan_set
                new_rev = rev_acc | add
                em = 0
                if nc == end and d + 1 == L and prefer_end_q4 is not None and q4 == prefer_end_q4:
                    em = 1
                new_st = (nc, q4)
                cand = (bin(new_rev).count('1'), em)
                exist = next_states.get(new_st)
                if exist is None or cand > (bin(exist[0]).count('1'), exist[1]):
                    next_states[new_st] = (new_rev, em, st)
        for k, v in next_states.items():
            best[k] = v
        layer = list(next_states.keys())
        if not layer: return None
    # ── 取 end 上最佳终态
    end_states = [(st, v) for st, v in best.items() if st[0] == end]
    if not end_states: return None
    end_states.sort(key=lambda x: (-bin(x[1][0]).count('1'), -x[1][1]))
    cur = end_states[0][0]
    path = []
    while best[cur][2] is not None:
        path.append(cur[0])
        cur = best[cur][2]
    path.reverse()
    return path


@dataclass
class OracleResult4:
    cost: int
    # interleaving 中:
    #   ("push", k)
    #   ("scan", j, vp_idx)   <- which vp option used (replay matches)
    #   ("rot", new_q4)        <- proactive rotate (replay)
    interleaving: List[Tuple]


def oracle_v4_min_steps(eng_init: GameEngine, plan: Optional[List] = None,
                          *, god_time_limit: float = 30.0,
                          allow_proactive_rotation: bool = True
                          ) -> Optional[OracleResult4]:
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

    vps_per_scan = _build_entity_vps(scans, state0)

    # trigger_map[(cell, q4)] = set of scan indices that get auto-revealed
    trigger_map: Dict[Tuple[Tuple[int, int], int], Set[int]] = {}
    for j in range(n_s):
        for (vp, q) in vps_per_scan[j]:
            trigger_map.setdefault((vp, q), set()).add(j)

    walk_cache: Dict[Tuple[int, Tuple[int, int], Tuple[int, int]],
                      Tuple[int, List[Tuple[int, int]]]] = {}

    def walk_path(step_k, src, dst):
        key = (step_k, src, dst)
        if key in walk_cache: return walk_cache[key]
        if src == dst:
            walk_cache[key] = (0, [])
            return walk_cache[key]
        walls_k = snapshots[step_k][0]
        obs = snapshots[step_k][1] - {src}
        path = _bfs_path_cells(src, dst, walls_k, obs)
        if path is None:
            walk_cache[key] = None
            return None
        walk_cache[key] = (len(path), path)
        return walk_cache[key]

    def reveals_along(path, q4):
        """Each cell visited at this q4 — what scans get revealed?"""
        revealed = 0
        for cell in path:
            ents = trigger_map.get((cell, q4))
            if not ents: continue
            for j in ents:
                revealed |= (1 << j)
        return revealed

    # 初始 reveals at init position
    init_pos = snapshots[0][2]
    init_q4 = _rad_to_q4(snapshots[0][3])
    init_mask = reveals_along([init_pos], init_q4)

    # DP
    init_state = (0, init_mask, init_pos, init_q4)
    dist = {init_state: 0}
    parent: Dict[Tuple, Optional[Tuple]] = {init_state: None}
    parent_act: Dict[Tuple, Optional[Tuple]] = {init_state: None}
    full_mask = (1 << n_s) - 1 if n_s > 0 else 0

    pq = [(0, init_state)]
    best_cost = INFV
    best_st = None

    while pq:
        c, st = heapq.heappop(pq)
        if c > dist.get(st, INFV): continue
        push_idx, scan_set, lp, q4 = st
        if push_idx == n_p and scan_set == full_mask:
            if c < best_cost:
                best_cost = c
                best_st = st
            continue

        # Action 1: Push next
        if push_idx < n_p:
            res = walk_path(push_idx, lp, push_positions[push_idx])
            if res is not None:
                wlen, path = res
                # 沿 path 的 reveals (current q4)
                new_mask = scan_set | reveals_along(path, q4)
                # 推完后 car 在 after_push_pos[push_idx], q4 不变. 也 check reveal at after_push.
                new_mask |= reveals_along([after_push_pos[push_idx]], q4)
                new_cost = c + wlen + 1
                new_st = (push_idx+1, new_mask, after_push_pos[push_idx], q4)
                if new_cost < dist.get(new_st, INFV):
                    dist[new_st] = new_cost
                    parent[new_st] = st
                    parent_act[new_st] = ("push", push_idx)
                    heapq.heappush(pq, (new_cost, new_st))

        # Action 2: Scan entity j via specific vp option
        for j in range(n_s):
            if scan_set & (1 << j): continue
            for vp_idx, (vp, vp_q4) in enumerate(vps_per_scan[j]):
                res = walk_path(push_idx, lp, vp)
                if res is None: continue
                wlen, path = res
                # reveals during walk (current q4)
                new_mask = scan_set | reveals_along(path, q4)
                # rotate to vp_q4
                rot_n = abs((vp_q4 - q4) % 4)
                rot_n = min(rot_n, 4 - rot_n)
                # after arrival + rotation, check reveal at vp with vp_q4 (this includes j by definition)
                new_mask |= reveals_along([vp], vp_q4)
                new_cost = c + wlen + rot_n
                new_st = (push_idx, new_mask, vp, vp_q4)
                if new_cost < dist.get(new_st, INFV):
                    dist[new_st] = new_cost
                    parent[new_st] = st
                    parent_act[new_st] = ("scan", j, vp_idx)
                    heapq.heappush(pq, (new_cost, new_st))

        # Action 3: Proactive rotation (only if useful — has unseen triggers at curr cell)
        if allow_proactive_rotation:
            for delta in (1, -1):
                new_q4 = (q4 + delta) % 4
                # check if this rotation reveals something at curr cell
                potential = reveals_along([lp], new_q4) & ~scan_set
                # 仅允许有用的旋转 (避免无限 state)
                if potential == 0: continue
                new_mask = scan_set | potential
                new_cost = c + 1
                new_st = (push_idx, new_mask, lp, new_q4)
                if new_cost < dist.get(new_st, INFV):
                    dist[new_st] = new_cost
                    parent[new_st] = st
                    parent_act[new_st] = ("rot", new_q4)
                    heapq.heappush(pq, (new_cost, new_st))

    if best_st is None:
        return None
    actions = []
    cur = best_st
    while cur is not None and parent_act.get(cur) is not None:
        actions.append(parent_act[cur])
        cur = parent[cur]
    actions.reverse()
    return OracleResult4(cost=best_cost, interleaving=actions)


def planner_oracle_v4(eng: GameEngine) -> None:
    plan = _god_plan(eng, time_limit=30.0)
    if not plan: return
    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    state0.seen_box_ids.update(fb); state0.seen_target_ids.update(ft)
    scans = _list_scans(state0, fb, ft)

    clone = _fresh_engine_from_eng(eng)
    res = oracle_v4_min_steps(clone, plan=plan)
    if res is None:
        from experiments.min_steps.planner_oracle_v3b import planner_oracle_v3b
        planner_oracle_v3b(eng); return

    vps_per_scan = _build_entity_vps(scans, state0)

    push_iter = iter(plan)
    for action in res.interleaving:
        if eng.get_state().won: return
        if action[0] == "push":
            _, k = action
            move = next(push_iter)
            push_pos = _get_push_pos(move)
            _walk_to_executor(eng, push_pos, "push_walk")
            eng._step_tag = "push"
            eng.discrete_step(direction_to_abs_action(*move[2]))
        elif action[0] == "scan":
            _, j, vp_idx = action
            if vp_idx >= len(vps_per_scan[j]): continue
            vp, vp_q4 = vps_per_scan[j][vp_idx]
            _walk_to_executor(eng, vp, "scan_walk")
            _rotate_executor(eng, Q4_RAD[vp_q4], "scan_rot")
        elif action[0] == "rot":
            _, new_q4 = action
            _rotate_executor(eng, Q4_RAD[new_q4], "proactive_rot")
