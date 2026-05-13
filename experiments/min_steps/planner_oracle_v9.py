"""Planner oracle v9 — v8 + pair-push as implicit scan.

观察: god plan 里每个 box 的"最后一次推"(consume push) 是 pair-attempt.
engine 自动 pair → 双方 ID 揭示 (即使没专门 scan).

实现:
  - 预算 last_push_step[class_id] = 该 class 最后一次被推的 step idx
  - DP 计算 known_classes 时, 加入 {c : last_push_step[c] < current_push_idx}
  - 这样, 推过的 box 自动算"已知" 在后续 penalty 计算中

效果: 同一个 plan, 但后续 push 的 unknown count 可能减少 → 更少 penalty →
       敢更激进 push, 步数更接近 lb.
"""

from __future__ import annotations

import heapq
import math
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
    OracleResult4, _bfs_path_cells, _build_entity_vps, Q4_RAD,
)
from experiments.min_steps.planner_oracle_v6 import _compute_dynamic_forced_steps

INFV = 10**9


def oracle_v9_min_steps(eng_init: GameEngine, plan: Optional[List] = None,
                          *, god_time_limit: float = 30.0,
                          allow_proactive_rotation: bool = True,
                          alpha: float = 1.0,
                          penalize_target: bool = True,
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

    # ── Belief tracking ─────────────────────────────────
    N_box = len(state0.boxes)
    N_target = len(state0.targets)
    class_universe = frozenset(b.class_id for b in state0.boxes)
    num_universe = frozenset(t.num_id for t in state0.targets)

    scan_type = []
    scan_label = []
    for vp, ang, et, ei in scans:
        scan_type.append(et)
        if et == 'box':
            scan_label.append(state0.boxes[ei].class_id)
        else:
            scan_label.append(state0.targets[ei].num_id)

    forced_classes_set = frozenset(state0.boxes[bi].class_id for bi, _ in forced if bi < N_box)
    forced_nums_set = frozenset(state0.targets[ti].num_id for _, ti in forced if ti < N_target)

    push_class_required = []
    for k in range(n_p):
        move = plan[k]
        etype, eid, _, _ = move
        if etype == 'box':
            _, cid = eid
            push_class_required.append(cid)
        else:
            push_class_required.append(None)

    # First/last push step per class (last = consume push)
    first_push_step: Dict[int, int] = {}
    last_push_step: Dict[int, int] = {}
    for k, c in enumerate(push_class_required):
        if c is not None:
            if c not in first_push_step:
                first_push_step[c] = k
            last_push_step[c] = k

    # 提前 build: 在 push_idx=p 开始时, 已通过 pair-push 揭示的 class_ids
    # = {c : last_push_step[c] < p}
    # 同样 target.num_id 通过 pair 也会揭示 (= same set since c=num)
    paired_classes_before: List[FrozenSet[int]] = [frozenset()] * (n_p + 1)
    cur_paired = set()
    for k in range(n_p + 1):
        # 在到达 k 之前已 pair 的: last_push_step[c] < k
        for c in list(last_push_step):
            if last_push_step[c] < k and c not in cur_paired:
                cur_paired.add(c)
        paired_classes_before[k] = frozenset(cur_paired)

    known_cache: Dict[Tuple[int, int], Tuple[FrozenSet[int], FrozenSet[int]]] = {}
    def compute_known(scan_set: int, push_idx: int):
        key = (scan_set, push_idx)
        if key in known_cache:
            return known_cache[key]
        sc_c = set()
        sc_n = set()
        for j in range(n_s):
            if scan_set & (1 << j):
                if scan_type[j] == 'box':
                    sc_c.add(scan_label[j])
                else:
                    sc_n.add(scan_label[j])
        # paired-via-implicit-scan 加入
        paired = paired_classes_before[push_idx]
        known_classes = sc_c | forced_classes_set | paired
        known_nums = sc_n | forced_nums_set | paired   # pair 同时揭示 box class 和 target num
        if len(known_classes) >= N_box - 1:
            known_classes = class_universe
        if len(known_nums) >= N_target - 1:
            known_nums = num_universe
        result = (frozenset(known_classes), frozenset(known_nums))
        known_cache[key] = result
        return result

    def gambling_penalty(scan_set: int, push_idx: int) -> float:
        push_class = push_class_required[push_idx]
        if push_class is None:
            return 0.0
        # 仅在首次推该 box 时计 penalty (commitment)
        if first_push_step.get(push_class) != push_idx:
            return 0.0
        known_c, known_n = compute_known(scan_set, push_idx)
        pen = 0.0
        if push_class not in known_c:
            n_possible = N_box - len(known_c)
            pen += alpha * max(0, n_possible - 1)
        if penalize_target and push_class not in known_n:
            n_possible_t = N_target - len(known_n)
            pen += alpha * max(0, n_possible_t - 1)
        return pen

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
    dist: Dict[Tuple, float] = {init_state: 0.0}
    parent: Dict[Tuple, Optional[Tuple]] = {init_state: None}
    parent_act: Dict[Tuple, Optional[Tuple]] = {init_state: None}
    real_steps: Dict[Tuple, int] = {init_state: 0}
    full_mask = (1 << n_s) - 1 if n_s > 0 else 0

    pq: List[Tuple[float, int, Tuple]] = [(0.0, 0, init_state)]
    tiebreak = 1
    best_cost = float('inf')
    best_st = None

    while pq:
        c, _tb, st = heapq.heappop(pq)
        if c > dist.get(st, INFV) + 1e-9: continue
        push_idx, scan_set, lp, q4 = st
        if push_idx == n_p and scan_set == full_mask:
            if c < best_cost:
                best_cost = c
                best_st = st
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
                    dist[new_st] = new_cost
                    real_steps[new_st] = new_real
                    parent[new_st] = st
                    parent_act[new_st] = ("push", push_idx)
                    tiebreak += 1
                    heapq.heappush(pq, (new_cost, tiebreak, new_st))

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
                new_real = real_steps[st] + wlen + rot_n
                new_cost = c + wlen + rot_n
                new_st = (push_idx, new_mask, vp, vp_q4)
                if new_cost < dist.get(new_st, INFV) - 1e-9:
                    dist[new_st] = new_cost
                    real_steps[new_st] = new_real
                    parent[new_st] = st
                    parent_act[new_st] = ("scan", j, vp_idx)
                    tiebreak += 1
                    heapq.heappush(pq, (new_cost, tiebreak, new_st))

        if allow_proactive_rotation:
            for delta in (1, -1):
                new_q4 = (q4 + delta) % 4
                potential = reveals_along([lp], new_q4) & ~scan_set
                if potential == 0: continue
                new_mask = scan_set | potential | cum_auto_mask[push_idx]
                new_real = real_steps[st] + 1
                new_cost = c + 1
                new_st = (push_idx, new_mask, lp, new_q4)
                if new_cost < dist.get(new_st, INFV) - 1e-9:
                    dist[new_st] = new_cost
                    real_steps[new_st] = new_real
                    parent[new_st] = st
                    parent_act[new_st] = ("rot", new_q4)
                    tiebreak += 1
                    heapq.heappush(pq, (new_cost, tiebreak, new_st))

    if best_st is None:
        return None
    actions = []
    cur = best_st
    while cur is not None and parent_act.get(cur) is not None:
        actions.append(parent_act[cur])
        cur = parent[cur]
    actions.reverse()
    return OracleResult4(cost=real_steps[best_st], interleaving=actions)


def planner_oracle_v9(eng: GameEngine, *, alpha: float = 0.4) -> None:
    plan = _god_plan(eng, time_limit=30.0)
    if not plan: return
    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    state0.seen_box_ids.update(fb); state0.seen_target_ids.update(ft)
    scans = _list_scans(state0, fb, ft)

    clone = _fresh_engine_from_eng(eng)
    res = oracle_v9_min_steps(clone, plan=plan, alpha=alpha)
    if res is None:
        from experiments.min_steps.planner_oracle_v6 import planner_oracle_v6
        planner_oracle_v6(eng); return

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
