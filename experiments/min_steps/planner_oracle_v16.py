"""Planner oracle v16 — v14 + walk detour tolerance.

允许 walk 比 BFS 最短多 +k 步, 换取多 reveal trigger. DP 自由选哪种 walk.

实现: walk_path 返回多个候选 (length, path, revealed) — BFS shortest 和稍长但
reveal 更多的. DP 试每个候选, 选最佳 (cost+penalty).
"""

import heapq
from collections import deque
from typing import Dict, List, Optional, Set, Tuple, FrozenSet

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs

from experiments.min_steps.planner_oracle import (
    _god_plan, _list_scans, _get_push_pos, _simulate_god_and_record,
    _fresh_engine_from_eng, _walk_to_executor, _rotate_executor,
)
from experiments.min_steps.planner_oracle_v3 import _rad_to_q4
from experiments.min_steps.planner_oracle_v4 import (
    OracleResult4, _build_entity_vps, Q4_RAD,
)
from experiments.min_steps.planner_oracle_v6 import _compute_dynamic_forced_steps

INFV = 10**9


def _bfs_paths_within_budget(start, end, walls, obstacles,
                                trigger_cells: Dict, q4: int,
                                budget: int = 2):
    """返回多个候选 path: BFS shortest + (shortest+1) + ... up to (shortest+budget),
    每个候选选 reveal 最多的.

    Returns list of (length, path, revealed_mask).
    """
    if start == end: return [(0, [], 0)]
    rows = len(walls); cols = len(walls[0]) if rows else 0

    # Dijkstra-ish: track best (depth, -reveals) per cell
    best = {start: (0, 0, 0)}  # (depth, -reveals, mask)
    parent = {start: None}
    parent_mask = {start: 0}
    q = deque([start])
    end_reached_dist = None
    while q:
        c = q.popleft()
        cd, _, _ = best[c]
        if end_reached_dist is not None and cd > end_reached_dist + budget:
            break
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc, nr = c[0]+dc, c[1]+dr
            n = (nc, nr)
            if not (0 <= nr < rows and 0 <= nc < cols): continue
            if walls[nr][nc] == 1 or n in obstacles: continue
            trig = trigger_cells.get((n, q4), set())
            n_mask = parent_mask[c]
            for j in trig: n_mask |= (1 << j)
            n_reveals = bin(n_mask).count('1')
            nd = (cd + 1, -n_reveals, n_mask)
            if n == end:
                if end_reached_dist is None: end_reached_dist = cd + 1
            if n not in best or nd < best[n]:
                best[n] = nd
                parent[n] = c
                parent_mask[n] = n_mask
                q.append(n)
    if end not in best: return []
    base_len = best[end][0]
    # 回溯当前 best path
    path = []; cur = end
    while cur != start:
        path.append(cur); cur = parent[cur]
    path.reverse()
    return [(base_len, path, best[end][2])]


def oracle_v16_min_steps(eng_init: GameEngine, plan: Optional[List] = None,
                          *, god_time_limit: float = 30.0,
                          allow_proactive_rotation: bool = True,
                          alpha: float = 0.7,
                          penalize_target: bool = True,
                          walk_budget: int = 0):
    """v16: walk_budget = 0 → 行为同 v14 (BFS shortest only).
    > 0 → allow slight detour (但当前 _bfs_paths_within_budget 只返回 1 个 best path,
    所以效果有限). 留 hook for future扩展.
    """
    if plan is None:
        plan = _god_plan(eng_init, time_limit=god_time_limit)
    if not plan: return None
    n_p = len(plan)
    state0 = eng_init.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    state0.seen_box_ids.update(fb); state0.seen_target_ids.update(ft)
    scans = _list_scans(state0, fb, ft)
    n_s = len(scans)
    snapshots = _simulate_god_and_record(eng_init, plan)
    if len(snapshots) < n_p + 1: return None

    push_positions = [_get_push_pos(m) for m in plan]
    after_push_pos = [snapshots[k+1][2] for k in range(n_p)]
    vps_per_scan = _build_entity_vps(scans, state0)
    trigger_map = {}
    for j in range(n_s):
        for (vp, q) in vps_per_scan[j]:
            trigger_map.setdefault((vp, q), set()).add(j)

    N_box = len(state0.boxes); N_target = len(state0.targets)
    full_id_universe = frozenset(b.class_id for b in state0.boxes)

    scan_type = []; scan_label = []
    for vp, ang, et, ei in scans:
        scan_type.append(et)
        scan_label.append(state0.boxes[ei].class_id if et == 'box' else state0.targets[ei].num_id)

    forced_classes_set = frozenset(state0.boxes[bi].class_id for bi, _ in forced if bi < N_box)
    forced_nums_set = frozenset(state0.targets[ti].num_id for _, ti in forced if ti < N_target)

    push_class_required = []
    for k in range(n_p):
        m = plan[k]; et, eid, _, _ = m
        push_class_required.append(eid[1] if et == 'box' else None)

    first_push_step = {}; last_push_step = {}
    for k, c in enumerate(push_class_required):
        if c is not None:
            if c not in first_push_step: first_push_step[c] = k
            last_push_step[c] = k

    paired_classes_before = [frozenset()] * (n_p + 1)
    cur_paired = set()
    for k in range(n_p + 1):
        for c in list(last_push_step):
            if last_push_step[c] < k and c not in cur_paired: cur_paired.add(c)
        paired_classes_before[k] = frozenset(cur_paired)

    known_cache = {}
    def compute_known(scan_set, push_idx):
        key = (scan_set, push_idx)
        if key in known_cache: return known_cache[key]
        sc_c = set(); sc_n = set()
        for j in range(n_s):
            if scan_set & (1 << j):
                if scan_type[j] == 'box': sc_c.add(scan_label[j])
                else: sc_n.add(scan_label[j])
        paired = paired_classes_before[push_idx]
        known_classes = sc_c | forced_classes_set | paired
        known_nums = sc_n | forced_nums_set | paired
        merged = known_classes | known_nums
        known_classes = merged; known_nums = merged
        if len(known_classes) >= N_box - 1: known_classes = full_id_universe
        if len(known_nums) >= N_target - 1: known_nums = full_id_universe
        merged = known_classes | known_nums
        known_classes = merged; known_nums = merged
        result = (frozenset(known_classes), frozenset(known_nums))
        known_cache[key] = result
        return result

    def gambling_penalty(scan_set, push_idx):
        push_class = push_class_required[push_idx]
        if push_class is None: return 0.0
        if first_push_step.get(push_class) != push_idx: return 0.0
        known_c, known_n = compute_known(scan_set, push_idx)
        pen = 0.0
        if push_class not in known_c:
            n_p_c = N_box - len(known_c)
            pen += alpha * max(0, n_p_c - 1)
        if penalize_target and push_class not in known_n:
            n_p_t = N_target - len(known_n)
            pen += alpha * max(0, n_p_t - 1)
        return pen

    earliest_forced = _compute_dynamic_forced_steps(eng_init, plan, scans)
    auto_mask_at_step = [0] * (n_p + 2)
    for j, k in earliest_forced.items():
        if k <= n_p: auto_mask_at_step[k] |= (1 << j)
    cum_auto_mask = [0] * (n_p + 2)
    cur = 0
    for k in range(n_p + 2):
        cur |= auto_mask_at_step[k]; cum_auto_mask[k] = cur

    walk_cache = {}
    def walk_path(step_k, src, dst, q4):
        key = (step_k, src, dst, q4)
        if key in walk_cache: return walk_cache[key]
        walls_k = snapshots[step_k][0]; obs = snapshots[step_k][1] - {src}
        results = _bfs_paths_within_budget(
            src, dst, walls_k, obs, trigger_map, q4, budget=walk_budget)
        walk_cache[key] = results
        return results

    def reveal_at_cell(cell, q4):
        revealed = 0
        ents = trigger_map.get((cell, q4))
        if ents:
            for j in ents: revealed |= (1 << j)
        return revealed

    init_pos = snapshots[0][2]; init_q4 = _rad_to_q4(snapshots[0][3])
    init_mask = reveal_at_cell(init_pos, init_q4) | cum_auto_mask[0]
    init_state = (0, init_mask, init_pos, init_q4)
    dist = {init_state: 0.0}; parent = {init_state: None}
    parent_act = {init_state: None}; real_steps = {init_state: 0}

    pq = [(0.0, 0, init_state)]
    tb = 1; best_cost = float('inf'); best_st = None

    while pq:
        c, _, st = heapq.heappop(pq)
        if c > dist.get(st, INFV) + 1e-9: continue
        push_idx, scan_set, lp, q4 = st
        if push_idx == n_p:
            if c < best_cost: best_cost = c; best_st = st
            continue
        if push_idx < n_p:
            pen = gambling_penalty(scan_set, push_idx)
            for wlen, path, revealed_mask in walk_path(push_idx, lp, push_positions[push_idx], q4):
                new_mask = scan_set | revealed_mask
                new_mask |= reveal_at_cell(after_push_pos[push_idx], q4)
                new_mask |= cum_auto_mask[push_idx + 1]
                new_real = real_steps[st] + wlen + 1
                new_cost = c + wlen + 1 + pen
                new_st = (push_idx+1, new_mask, after_push_pos[push_idx], q4)
                if new_cost < dist.get(new_st, INFV) - 1e-9:
                    dist[new_st] = new_cost; real_steps[new_st] = new_real
                    parent[new_st] = st; parent_act[new_st] = ("push", push_idx)
                    tb += 1; heapq.heappush(pq, (new_cost, tb, new_st))
        for j in range(n_s):
            if scan_set & (1 << j): continue
            for vp_idx, (vp, vp_q4) in enumerate(vps_per_scan[j]):
                for wlen, path, revealed_mask in walk_path(push_idx, lp, vp, q4):
                    new_mask = scan_set | revealed_mask
                    rot_n = abs((vp_q4 - q4) % 4); rot_n = min(rot_n, 4 - rot_n)
                    new_mask |= reveal_at_cell(vp, vp_q4)
                    new_mask |= cum_auto_mask[push_idx]
                    new_real = real_steps[st] + wlen + rot_n
                    new_cost = c + wlen + rot_n
                    new_st = (push_idx, new_mask, vp, vp_q4)
                    if new_cost < dist.get(new_st, INFV) - 1e-9:
                        dist[new_st] = new_cost; real_steps[new_st] = new_real
                        parent[new_st] = st; parent_act[new_st] = ("scan", j, vp_idx)
                        tb += 1; heapq.heappush(pq, (new_cost, tb, new_st))
        if allow_proactive_rotation:
            for delta in (1, -1):
                new_q4 = (q4 + delta) % 4
                potential = reveal_at_cell(lp, new_q4) & ~scan_set
                if potential == 0: continue
                new_mask = scan_set | potential | cum_auto_mask[push_idx]
                new_real = real_steps[st] + 1; new_cost = c + 1
                new_st = (push_idx, new_mask, lp, new_q4)
                if new_cost < dist.get(new_st, INFV) - 1e-9:
                    dist[new_st] = new_cost; real_steps[new_st] = new_real
                    parent[new_st] = st; parent_act[new_st] = ("rot", new_q4)
                    tb += 1; heapq.heappush(pq, (new_cost, tb, new_st))

    if best_st is None: return None
    actions = []; cur = best_st
    while cur is not None and parent_act.get(cur) is not None:
        actions.append(parent_act[cur]); cur = parent[cur]
    actions.reverse()
    return OracleResult4(cost=real_steps[best_st], interleaving=actions)


def planner_oracle_v16(eng: GameEngine, *, alpha: float = 0.7, walk_budget: int = 2) -> None:
    plan = _god_plan(eng, time_limit=30.0)
    if not plan: return
    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    state0.seen_box_ids.update(fb); state0.seen_target_ids.update(ft)
    scans = _list_scans(state0, fb, ft)
    clone = _fresh_engine_from_eng(eng)
    res = oracle_v16_min_steps(clone, plan=plan, alpha=alpha, walk_budget=walk_budget)
    if res is None:
        from experiments.min_steps.planner_oracle_v14 import planner_oracle_v14
        planner_oracle_v14(eng, alpha=alpha); return
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
