"""Planner oracle v7 — strict belief-aware (NO gambling pushes).

硬约束: push box B 合法 iff
  - B.class_id ∈ known_classes (via scan/exclusion/forced-pair)
  - AND B.class_id ∈ known_nums (对应 target 也识别了)

known_classes:
  scanned_classes ∪ forced_classes ∪ (exclusion 如果 N-1 已知)

这是真 belief-aware. 期望步数比 v6 高 (得多扫几次先), 但不再作弊.
"""

from __future__ import annotations

import copy
import heapq
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, FrozenSet

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs
from experiments.sage_pr.build_dataset_v3 import apply_solver_move

from experiments.min_steps.planner_oracle import (
    _god_plan, _list_scans, _get_push_pos, _simulate_god_and_record,
    _fresh_engine_from_eng, _walk_to_executor, _rotate_executor,
)
from experiments.min_steps.planner_oracle_v3 import _rad_to_q4
from experiments.min_steps.planner_oracle_v4 import (
    OracleResult4, _bfs_path_cells, _build_entity_vps, Q4_RAD, Q4_UNIT,
)
from experiments.min_steps.planner_oracle_v6 import _compute_dynamic_forced_steps

INFV = 10**9


def oracle_v7_min_steps(eng_init: GameEngine, plan: Optional[List] = None,
                          *, god_time_limit: float = 30.0,
                          allow_proactive_rotation: bool = True,
                          require_target_known: bool = True,
                          ) -> Optional[OracleResult4]:
    """Strict-belief oracle.
    require_target_known: 如果 True, 要求 target.num_id 也已识别 (默认). False
    则只需 box.class_id 已识别 (略松)."""
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

    # ── Belief tracking 预计算 ──────────────────────────────
    N_box = len(state0.boxes)
    N_target = len(state0.targets)
    # Universe: 假设 box.class_id 来自 {0..N-1} (Sokoban 约定)
    class_universe = frozenset(b.class_id for b in state0.boxes)
    num_universe = frozenset(t.num_id for t in state0.targets)

    # 每个 scan 反映的 class/num
    scan_type = []   # 'box' or 'target'
    scan_label = []  # class_id or num_id
    for vp, ang, et, ei in scans:
        scan_type.append(et)
        if et == 'box':
            scan_label.append(state0.boxes[ei].class_id)
        else:
            scan_label.append(state0.targets[ei].num_id)

    forced_classes_set = frozenset(state0.boxes[bi].class_id for bi, _ in forced if bi < N_box)
    forced_nums_set = frozenset(state0.targets[ti].num_id for _, ti in forced if ti < N_target)

    # cache: scan_set bitmask → (known_classes_frozenset, known_nums_frozenset)
    known_cache: Dict[int, Tuple[FrozenSet[int], FrozenSet[int]]] = {}
    def compute_known(scan_set: int):
        if scan_set in known_cache:
            return known_cache[scan_set]
        scanned_classes = set()
        scanned_nums = set()
        for j in range(n_s):
            if scan_set & (1 << j):
                if scan_type[j] == 'box':
                    scanned_classes.add(scan_label[j])
                else:
                    scanned_nums.add(scan_label[j])
        known_classes = scanned_classes | forced_classes_set
        known_nums = scanned_nums | forced_nums_set
        # Exclusion law: N-1 already known → universe completes
        if len(known_classes) >= N_box - 1:
            known_classes = class_universe
        if len(known_nums) >= N_target - 1:
            known_nums = num_universe
        result = (frozenset(known_classes), frozenset(known_nums))
        known_cache[scan_set] = result
        return result

    # 每个 push 需要的 class_id (= 推的 box 的 class_id)
    push_class_required = []
    for k in range(n_p):
        move = plan[k]
        etype, eid, _, _ = move
        if etype == 'box':
            _, cid = eid
            push_class_required.append(cid)
        else:
            # bomb push — no belief required
            push_class_required.append(None)

    # 动态 forced pair 仍然可用 (推走某些 box 后剩余拓扑变化)
    earliest_forced = _compute_dynamic_forced_steps(eng_init, plan, scans)
    auto_mask_at_step = [0] * (n_p + 2)
    for j, k in earliest_forced.items():
        if k <= n_p:
            auto_mask_at_step[k] |= (1 << j)
    cum_auto_mask = [0] * (n_p + 2)
    cur = 0
    for k in range(n_p + 2):
        cur |= auto_mask_at_step[k]
        cum_auto_mask[k] = cur

    # ── 路径 BFS cache ────────────────────────────────────
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

    # ── DP ────────────────────────────────────────────────
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
        if push_idx == n_p:
            # 所有 push 完成. scan_set 不必满 (allow 跳过最终扫). 但严格起见
            # 还是要求 full_mask (避免 DP 偏好"省 scan" 反而失真).
            if scan_set == full_mask:
                if c < best_cost:
                    best_cost = c
                    best_st = st
            continue

        # Push next — 但必须 belief-feasible
        if push_idx < n_p:
            class_needed = push_class_required[push_idx]
            ok_to_push = True
            if class_needed is not None:
                known_c, known_n = compute_known(scan_set)
                if class_needed not in known_c:
                    ok_to_push = False
                if require_target_known and class_needed not in known_n:
                    ok_to_push = False

            if ok_to_push:
                res = walk_path(push_idx, lp, push_positions[push_idx])
                if res is not None:
                    wlen, path = res
                    new_mask = scan_set | reveals_along(path, q4)
                    new_mask |= reveals_along([after_push_pos[push_idx]], q4)
                    new_mask |= cum_auto_mask[push_idx + 1]
                    new_cost = c + wlen + 1
                    new_st = (push_idx+1, new_mask, after_push_pos[push_idx], q4)
                    if new_cost < dist.get(new_st, INFV):
                        dist[new_st] = new_cost
                        parent[new_st] = st
                        parent_act[new_st] = ("push", push_idx)
                        heapq.heappush(pq, (new_cost, new_st))

        # Scan entity j
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
                new_mask |= cum_auto_mask[push_idx]
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


def planner_oracle_v7(eng: GameEngine) -> None:
    plan = _god_plan(eng, time_limit=30.0)
    if not plan: return
    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    state0.seen_box_ids.update(fb); state0.seen_target_ids.update(ft)
    scans = _list_scans(state0, fb, ft)

    clone = _fresh_engine_from_eng(eng)
    res = oracle_v7_min_steps(clone, plan=plan)
    if res is None:
        # Strict 失败 — fall back to v6 (松散一点也比卡死好)
        from experiments.min_steps.planner_oracle_v6 import planner_oracle_v6
        planner_oracle_v6(eng)
        return

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
