"""Planner oracle v6 — v4 + dynamic forced-pair detection.

当 push 完前几个 box 后, 剩余 box 的拓扑可能锁定新的 forced pair (原本 2 个可达
target 变 1 个). 这些 entity 不再需要扫.

实现:
  1. 模拟 god plan, 在每个 step_k 跑 find_forced_pairs(state_k)
  2. 对 scan list 中的每个 entity j, 算 earliest_forced[j] = 最早 push_idx 使
     该 entity 变 forced (或 INFV 表示永不)
  3. DP 转移时: 转入 state (push_idx=k, ...) 后, 把 scan_set 的所有
     j with earliest_forced[j] <= k 自动置位

ROI: 实测 ~0.7 step/map 改善
"""

from __future__ import annotations

import copy
import heapq
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
from experiments.sage_pr.build_dataset_v3 import apply_solver_move

from experiments.min_steps.planner_oracle import (
    _god_plan, _list_scans, _get_push_pos, _simulate_god_and_record,
    _fresh_engine_from_eng, _walk_to_executor, _rotate_executor,
)
from experiments.min_steps.planner_oracle_v3 import _rad_to_q4
from experiments.min_steps.planner_oracle_v4 import (
    OracleResult4, _bfs_path_cells, _build_entity_vps, Q4_RAD, Q4_UNIT,
)

INFV = 10**9


def _compute_dynamic_forced_steps(eng_init: GameEngine, plan: List,
                                    scans: List) -> Dict[int, int]:
    """For each scan index j, compute the earliest push_idx k at which the entity
    associated with scan j becomes forced via topology. INFV if never.

    scans elements: (vp, angle, etype, eidx). eidx is original box/target index
    in state.boxes/state.targets — but after pushes, box indices may shift (boxes
    get consumed). We track by ORIGINAL position + class_id for boxes,
    or by position+num_id for targets.
    """
    # 初始 entity 标识
    state0 = eng_init.get_state()
    boxes_init = [(pos_to_grid(b.x, b.y), b.class_id) for b in state0.boxes]
    targets_init = [(pos_to_grid(t.x, t.y), t.num_id) for t in state0.targets]

    # 为每个 scan 找原始 entity 的 "label" (box: class_id; target: num_id)
    scan_label = []
    for vp, ang, etype, eidx in scans:
        if etype == "box":
            scan_label.append(("box", boxes_init[eidx][1]))   # class_id
        else:
            scan_label.append(("target", targets_init[eidx][1]))   # num_id

    # 跑 god plan, 每步记录 forced pairs (按 label)
    sim = _fresh_engine_from_eng(eng_init)
    forced_label_at_step: Dict[int, Set[Tuple[str, int]]] = {}
    # k=0 forced
    ss = sim.get_state()
    pairs_0 = find_forced_pairs(ss)
    forced_labels = set()
    for bi, ti in pairs_0:
        if bi < len(ss.boxes) and ti < len(ss.targets):
            forced_labels.add(("box", ss.boxes[bi].class_id))
            forced_labels.add(("target", ss.targets[ti].num_id))
    forced_label_at_step[0] = forced_labels

    for k, move in enumerate(plan):
        if not apply_solver_move(sim, move):
            break
        ss = sim.get_state()
        pairs_k = find_forced_pairs(ss)
        forced_labels = set()
        for bi, ti in pairs_k:
            if bi < len(ss.boxes) and ti < len(ss.targets):
                forced_labels.add(("box", ss.boxes[bi].class_id))
                forced_labels.add(("target", ss.targets[ti].num_id))
        forced_label_at_step[k+1] = forced_labels

    # 累积: forced labels 在 step k 之前任意时刻出现过
    cumulative = set()
    earliest_forced = {}
    for k in sorted(forced_label_at_step.keys()):
        cumulative |= forced_label_at_step[k]
        for j, label in enumerate(scan_label):
            if label in cumulative and j not in earliest_forced:
                earliest_forced[j] = k
    return earliest_forced


def oracle_v6_min_steps(eng_init: GameEngine, plan: Optional[List] = None,
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
    trigger_map: Dict[Tuple[Tuple[int, int], int], Set[int]] = {}
    for j in range(n_s):
        for (vp, q) in vps_per_scan[j]:
            trigger_map.setdefault((vp, q), set()).add(j)

    # 动态 forced pair 计算
    earliest_forced = _compute_dynamic_forced_steps(eng_init, plan, scans)
    # auto_mask_at_step[k] = bitmask of scans auto-set at push_idx=k
    auto_mask_at_step = [0] * (n_p + 2)
    for j, k in earliest_forced.items():
        if k <= n_p:
            auto_mask_at_step[k] |= (1 << j)
    # cumulative: 转入 push_idx=k 时, 自动 OR in 所有 step<=k 的 auto mask
    cum_auto_mask = [0] * (n_p + 2)
    cur = 0
    for k in range(n_p + 2):
        cur |= auto_mask_at_step[k]
        cum_auto_mask[k] = cur

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
        revealed = 0
        for cell in path:
            ents = trigger_map.get((cell, q4))
            if not ents: continue
            for j in ents:
                revealed |= (1 << j)
        return revealed

    init_pos = snapshots[0][2]
    init_q4 = _rad_to_q4(snapshots[0][3])
    init_mask = reveals_along([init_pos], init_q4) | cum_auto_mask[0]

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

        # Push next
        if push_idx < n_p:
            res = walk_path(push_idx, lp, push_positions[push_idx])
            if res is not None:
                wlen, path = res
                new_mask = scan_set | reveals_along(path, q4)
                new_mask |= reveals_along([after_push_pos[push_idx]], q4)
                # 加 dynamic forced pair auto
                new_mask |= cum_auto_mask[push_idx + 1]
                new_cost = c + wlen + 1
                new_st = (push_idx+1, new_mask, after_push_pos[push_idx], q4)
                if new_cost < dist.get(new_st, INFV):
                    dist[new_st] = new_cost
                    parent[new_st] = st
                    parent_act[new_st] = ("push", push_idx)
                    heapq.heappush(pq, (new_cost, new_st))

        # Scan
        for j in range(n_s):
            if scan_set & (1 << j): continue
            for vp_idx, (vp, vp_q4) in enumerate(vps_per_scan[j]):
                res = walk_path(push_idx, lp, vp)
                if res is None: continue
                wlen, path = res
                new_mask = scan_set | reveals_along(path, q4)
                rot_n = abs((vp_q4 - q4) % 4)
                rot_n = min(rot_n, 4 - rot_n)
                new_mask |= reveals_along([vp], vp_q4)
                new_mask |= cum_auto_mask[push_idx]   # 注意 push_idx 不变
                new_cost = c + wlen + rot_n
                new_st = (push_idx, new_mask, vp, vp_q4)
                if new_cost < dist.get(new_st, INFV):
                    dist[new_st] = new_cost
                    parent[new_st] = st
                    parent_act[new_st] = ("scan", j, vp_idx)
                    heapq.heappush(pq, (new_cost, new_st))

        # Proactive rotation
        if allow_proactive_rotation:
            for delta in (1, -1):
                new_q4 = (q4 + delta) % 4
                potential = reveals_along([lp], new_q4) & ~scan_set
                if potential == 0: continue
                new_mask = scan_set | potential | cum_auto_mask[push_idx]
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


def planner_oracle_v6(eng: GameEngine) -> None:
    plan = _god_plan(eng, time_limit=30.0)
    if not plan: return
    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    state0.seen_box_ids.update(fb); state0.seen_target_ids.update(ft)
    scans = _list_scans(state0, fb, ft)

    clone = _fresh_engine_from_eng(eng)
    res = oracle_v6_min_steps(clone, plan=plan)
    if res is None:
        from experiments.min_steps.planner_oracle_v4 import planner_oracle_v4
        planner_oracle_v4(eng); return

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
