"""Planner oracle v15 — v14 + extended scan list (all N entities).

之前 _list_scans 给 N-1 个 entity (跳最远那个, 排除律可推).
v15: 给所有 N 个, 让 DP 自由选最便宜的 N-1 子集.

效果: 可能选到便宜 vp + 跳过贵的 farthest, 节省 scan walk.
"""

import heapq
from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from collections import deque

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs
from smartcar_sokoban.solver.explorer import (
    find_observation_point, get_entity_obstacles, get_all_entity_positions,
)

from experiments.min_steps.planner_oracle import (
    _god_plan, _get_push_pos, _simulate_god_and_record,
    _fresh_engine_from_eng, _walk_to_executor, _rotate_executor,
)
from experiments.min_steps.planner_oracle_v3 import _rad_to_q4
from experiments.min_steps.planner_oracle_v4 import (
    OracleResult4, _bfs_path_cells, _build_entity_vps, Q4_RAD,
)
from experiments.min_steps.planner_oracle_v6 import _compute_dynamic_forced_steps

INFV = 10**9


def _list_scans_full(state, forced_box: Set[int], forced_tgt: Set[int]):
    """所有非 forced 非 seen 的 box/target. NOT 排除律 trimmed."""
    car_grid = pos_to_grid(state.car_x, state.car_y)
    obstacles = get_entity_obstacles(state)
    entity_pos = get_all_entity_positions(state)
    todo = []

    for i in range(len(state.boxes)):
        if i in state.seen_box_ids or i in forced_box: continue
        b = state.boxes[i]
        bg = pos_to_grid(b.x, b.y)
        r = find_observation_point(car_grid, bg, state.grid, obstacles,
                                      entity_pos, current_angle=state.car_angle)
        if r is None: continue
        vp, angle = r
        todo.append((vp, angle, 'box', i))

    for i in range(len(state.targets)):
        if i in state.seen_target_ids or i in forced_tgt: continue
        t = state.targets[i]
        tg = pos_to_grid(t.x, t.y)
        r = find_observation_point(car_grid, tg, state.grid, obstacles,
                                      entity_pos, current_angle=state.car_angle)
        if r is None: continue
        vp, angle = r
        todo.append((vp, angle, 'target', i))
    return todo


def oracle_v15_min_steps(eng_init: GameEngine, plan: Optional[List] = None,
                          *, god_time_limit: float = 30.0,
                          allow_proactive_rotation: bool = True,
                          alpha: float = 0.7,
                          penalize_target: bool = True):
    if plan is None:
        plan = _god_plan(eng_init, time_limit=god_time_limit)
    if not plan: return None
    n_p = len(plan)
    state0 = eng_init.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    state0.seen_box_ids.update(fb); state0.seen_target_ids.update(ft)
    scans = _list_scans_full(state0, fb, ft)   # ALL N entities
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
    def walk_path(step_k, src, dst):
        key = (step_k, src, dst)
        if key in walk_cache: return walk_cache[key]
        if src == dst:
            walk_cache[key] = (0, []); return walk_cache[key]
        walls_k = snapshots[step_k][0]; obs = snapshots[step_k][1] - {src}
        path = _bfs_path_cells(src, dst, walls_k, obs)
        if path is None:
            walk_cache[key] = None; return None
        walk_cache[key] = (len(path), path); return walk_cache[key]

    def reveals_along(path, q4):
        revealed = 0
        for cell in path:
            ents = trigger_map.get((cell, q4))
            if not ents: continue
            for j in ents: revealed |= (1 << j)
        return revealed

    init_pos = snapshots[0][2]; init_q4 = _rad_to_q4(snapshots[0][3])
    init_mask = reveals_along([init_pos], init_q4) | cum_auto_mask[0]
    init_state = (0, init_mask, init_pos, init_q4)
    dist = {init_state: 0.0}; parent = {init_state: None}
    parent_act = {init_state: None}; real_steps = {init_state: 0}

    pq = [(0.0, 0, init_state)]
    tb = 1; best_cost = float('inf'); best_st = None

    while pq:
        c, _, st = heapq.heappop(pq)
        if c > dist.get(st, INFV) + 1e-9: continue
        push_idx, scan_set, lp, q4 = st
        # 仅要求 push_idx == n_p, 不强制 scan_set 满
        if push_idx == n_p:
            if c < best_cost: best_cost = c; best_st = st
            continue
        if push_idx < n_p:
            pen = gambling_penalty(scan_set, push_idx)
            res = walk_path(push_idx, lp, push_positions[push_idx])
            if res is not None:
                wlen, path = res
                new_mask = scan_set | reveals_along(path, q4)
                new_mask |= reveals_along([after_push_pos[push_idx]], q4)
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
                res = walk_path(push_idx, lp, vp)
                if res is None: continue
                wlen, path = res
                new_mask = scan_set | reveals_along(path, q4)
                rot_n = abs((vp_q4 - q4) % 4); rot_n = min(rot_n, 4 - rot_n)
                new_mask |= reveals_along([vp], vp_q4)
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
                potential = reveals_along([lp], new_q4) & ~scan_set
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


def planner_oracle_v15(eng: GameEngine, *, alpha: float = 0.7) -> None:
    plan = _god_plan(eng, time_limit=30.0)
    if not plan: return
    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    state0.seen_box_ids.update(fb); state0.seen_target_ids.update(ft)
    scans = _list_scans_full(state0, fb, ft)
    clone = _fresh_engine_from_eng(eng)
    res = oracle_v15_min_steps(clone, plan=plan, alpha=alpha)
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
