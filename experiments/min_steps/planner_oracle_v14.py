"""Planner oracle v14 — v12 + relaxed goal (no full_mask requirement).

v12 强制 DP 达到 scan_set == full_mask. 但某些 entity 可能通过 pair-consumption
/ forced-pair / exclusion 隐式识别 — 不需要显式 scan.

v14: 仅要求 push_idx == n_p (所有 push 完成). Scan 不再强制 — 但赌博 penalty
仍累积. DP 自动 trade-off scan-cost vs gambling-penalty.

效果: 删 redundant scans → 步数减少. Penalty 可能略增 (允许更多 gambling).
"""

import heapq
from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from collections import deque

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
    OracleResult4, _bfs_path_cells, _bfs_path_cells_optimal,
    _build_entity_vps, Q4_RAD,
)
from experiments.min_steps.planner_oracle_v6 import _compute_dynamic_forced_steps

INFV = 10**9


def oracle_v14_min_steps(eng_init: GameEngine, plan: Optional[List] = None,
                          *, god_time_limit: float = 30.0,
                          allow_proactive_rotation: bool = True,
                          alpha: float = 0.4,
                          penalize_target: bool = True,
                          trust_walk_reveal: bool = False):
    if plan is None:
        plan = _god_plan(eng_init, time_limit=god_time_limit)
    if not plan: return None
    n_p = len(plan)
    state0 = eng_init.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    # 不污染 engine state (旧 bug: 把 forced-pair entity 标进 engine.seen_*_ids,
    # 导致 viz 把"几何配对推理"误绘成"FOV 已识别 + class id 已知").
    # _list_scans 通过 forced 参数过滤; v14 DP 走 forced_classes_set / forced_nums_set.
    scans = _list_scans(state0, fb, ft)
    n_s = len(scans)
    snapshots = _simulate_god_and_record(eng_init, plan)
    if len(snapshots) < n_p + 1: return None

    push_positions = [_get_push_pos(m) for m in plan]
    after_push_pos = [snapshots[k+1][2] for k in range(n_p)]
    vps_per_scan = _build_entity_vps(scans, state0)
    trigger_map = {}   # static (state0): vp -> set of scan-indices, 用于 vp-based explicit scan
    for j in range(n_s):
        for (vp, q) in vps_per_scan[j]:
            trigger_map.setdefault((vp, q), set()).add(j)

    # ── 动态 trigger_map_per_step (新): 每步 k 重建, 基于当前 entity 位置.
    # walk-reveal 在执行时由 engine FOV 触发, 而 entity 位置随推动作变化.
    # 静态 trigger_map (state0-based) 在 k>0 时模型失真, 是之前 trust_walk_reveal=False 的根因.
    from experiments.min_steps.planner_oracle_v4 import Q4_UNIT
    scan_idx_for_box = {}; scan_idx_for_tgt = {}
    for j, (vp, ang, et, ei) in enumerate(scans):
        if et == 'box': scan_idx_for_box.setdefault(ei, []).append(j)
        else: scan_idx_for_tgt.setdefault(ei, []).append(j)

    def _build_trigger_map_at_step(k):
        snap = snapshots[k]
        walls_k = snap[0]
        obs_k = snap[1]
        box_info = snap[4] if len(snap) >= 5 else None
        tgt_info = snap[5] if len(snap) >= 6 else None
        if box_info is None: return {}
        rows = len(walls_k); cols = len(walls_k[0]) if rows else 0
        tmap = {}
        # 收集还活着的 entity (位置, 对应 scan_indices)
        actives = []
        for i, bi in enumerate(box_info):
            if bi is None: continue
            pos, _cls = bi
            if i in scan_idx_for_box:
                actives.append((pos, scan_idx_for_box[i]))
        for i, ti in enumerate(tgt_info):
            if ti is None: continue
            pos, _num = ti
            if i in scan_idx_for_tgt:
                actives.append((pos, scan_idx_for_tgt[i]))
        for ep, sc_list in actives:
            for q in range(4):
                dx, dy = Q4_UNIT[q]
                vp_x = ep[0] - dx; vp_y = ep[1] - dy
                if not (0 <= vp_y < rows and 0 <= vp_x < cols): continue
                if walls_k[vp_y][vp_x] == 1: continue
                vp = (vp_x, vp_y)
                if vp == ep: continue
                if vp in obs_k: continue  # vp 被 box/bomb 挡住, 无法在该格 walk-reveal
                # 相邻格 (dist=1), LOS 平凡 (无墙在中间)
                tmap.setdefault((vp, q), set()).update(sc_list)
        return tmap

    trigger_maps_per_step = [_build_trigger_map_at_step(k) for k in range(len(snapshots))]

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
        """返回 (K_box, K_tgt): 在 push_idx 推之前 agent 能识别的 box-class / target-num 集合.

        重要修复 (相对旧 v14): 不再把 known_classes 和 known_nums 合并,
        因为 scan box 只识别该 box 的 class, 并不告诉 agent 哪个 target 是这个 num.
        Forced pair 的箱/目标 a priori 计入 (几何上强制匹配, 不需 scan 信息).
        Paired (已 consume) class 通过排除法计入 (假设 prior consumption 是 informed).
        """
        key = (scan_set, push_idx)
        if key in known_cache: return known_cache[key]
        sc_c = set(); sc_n = set()
        for j in range(n_s):
            if scan_set & (1 << j):
                if scan_type[j] == 'box': sc_c.add(scan_label[j])
                else: sc_n.add(scan_label[j])
        paired = paired_classes_before[push_idx]
        known_classes = sc_c | set(forced_classes_set) | set(paired)
        known_nums = sc_n | set(forced_nums_set) | set(paired)
        # 同侧 bijective universe propagation: N-1 知 → 全知
        if len(known_classes) >= N_box - 1:
            known_classes = set(full_id_universe)
        if len(known_nums) >= N_target - 1:
            known_nums = set(full_id_universe)
        result = (frozenset(known_classes), frozenset(known_nums))
        known_cache[key] = result
        return result

    def gambling_penalty(scan_set, push_idx):
        """Consumption-side penalty: 在该 class 的 LAST push (推到 target 消除) 时检查.

        重要修复 (相对旧 v14): penalty 在 last_push 而非 first_push 触发, 因为
        盲推真正的成本在 consumption 时刻 (commit (box.class, target.num) 配对).
        Box-side / target-side 分别累加 — 任一未知都算赌博.
        """
        push_class = push_class_required[push_idx]
        if push_class is None: return 0.0
        if last_push_step.get(push_class) != push_idx: return 0.0
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

    # 旧版基础 BFS, 只算最短距离 (用于早期可达性测试和回退)
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

    # 新: walk-reveal aware + end-orient 优选. scan_set / prefer_end_q4 状态依赖, 不缓存.
    # 用 dynamic trigger_maps_per_step[step_k] 准确建模当前 entity 位置.
    def walk_path_optimal(step_k, src, dst, scan_set, prefer_end_q4):
        if src == dst: return (0, [])
        walls_k = snapshots[step_k][0]; obs = snapshots[step_k][1] - {src}
        path = _bfs_path_cells_optimal(src, dst, walls_k, obs,
                                           trigger_map=trigger_maps_per_step[step_k],
                                           scan_set=scan_set,
                                           prefer_end_q4=prefer_end_q4)
        if path is None: return None
        return (len(path), path)

    _DIR_TO_Q4 = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}

    def reveals_along(path, q4, step_k=None):
        """单格 reveal: 已 rotate 到指定 q4 后在某 cell 触发. 用于 scan vp/init/after-push.
        step_k != None 时用 dynamic trigger_map; 否则 static (state0)."""
        tmap = trigger_maps_per_step[step_k] if step_k is not None else trigger_map
        revealed = 0
        for cell in path:
            ents = tmap.get((cell, q4))
            if not ents: continue
            for j in ents: revealed |= (1 << j)
        return revealed

    def reveals_along_walk(start_pos, path, step_k=None):
        """走路 reveal: 每步走完车头 = 该步方向。orient = dir(prev→cell).
        当传入 step_k 时, 用 trigger_maps_per_step[k] 反映当前 entity 实际位置;
        否则退回静态 trigger_map (state0-based, 仅 k=0 准确)."""
        tmap = trigger_maps_per_step[step_k] if step_k is not None else trigger_map
        revealed = 0
        prev = start_pos
        for cell in path:
            dx = cell[0] - prev[0]
            dy = cell[1] - prev[1]
            q = _DIR_TO_Q4.get((dx, dy))
            if q is not None:
                ents = tmap.get((cell, q))
                if ents:
                    for j in ents: revealed |= (1 << j)
            prev = cell
        return revealed

    if not trust_walk_reveal:
        # 老 fallback (dynamic trigger_map 启用后通常不需要).
        def reveals_along_walk(_start_pos, _path, step_k=None):
            return 0

    init_pos = snapshots[0][2]; init_q4 = _rad_to_q4(snapshots[0][3])
    init_mask = reveals_along([init_pos], init_q4) | cum_auto_mask[0]
    init_state = (0, init_mask, init_pos, init_q4)
    dist = {init_state: 0.0}; parent = {init_state: None}
    parent_act = {init_state: None}; real_steps = {init_state: 0}

    # 推完之后车头朝向 = 推方向 (engine 中 discrete_step 推动作会把 car_angle 设为 push 方向)
    after_push_q4 = [_rad_to_q4(snapshots[k+1][3]) for k in range(n_p)]
    # 最后一格走完车头 = 该步方向 (q4_after_walk); 同样用于 scan 路径的 last-step orient
    def _walk_end_q4(start_pos, path, fallback_q4):
        if not path: return fallback_q4
        last = path[-1]
        prev = path[-2] if len(path) >= 2 else start_pos
        dx = last[0] - prev[0]; dy = last[1] - prev[1]
        return _DIR_TO_Q4.get((dx, dy), fallback_q4)

    pq = [(0.0, 0, init_state)]
    tb = 1; best_cost = float('inf'); best_st = None

    while pq:
        c, _, st = heapq.heappop(pq)
        if c > dist.get(st, INFV) + 1e-9: continue
        push_idx, scan_set, lp, q4 = st
        # v14: 仅要求 push_idx == n_p, 不强制 scan_set 满
        if push_idx == n_p:
            if c < best_cost: best_cost = c; best_st = st
            continue
        if push_idx < n_p:
            pen = gambling_penalty(scan_set, push_idx)
            # Push 不需 end_q4 偏好 (push 自带旋转), 但 walk-reveal 仍要最大化.
            res = walk_path_optimal(push_idx, lp, push_positions[push_idx],
                                       scan_set=scan_set, prefer_end_q4=None)
            if res is not None:
                wlen, path = res
                new_mask = scan_set | reveals_along_walk(lp, path, step_k=push_idx)
                push_q4 = after_push_q4[push_idx]
                new_mask |= reveals_along([after_push_pos[push_idx]], push_q4)
                new_mask |= cum_auto_mask[push_idx + 1]
                new_real = real_steps[st] + wlen + 1
                new_cost = c + wlen + 1 + pen
                new_st = (push_idx+1, new_mask, after_push_pos[push_idx], push_q4)
                if new_cost < dist.get(new_st, INFV) - 1e-9:
                    dist[new_st] = new_cost; real_steps[new_st] = new_real
                    parent[new_st] = st
                    parent_act[new_st] = ("push", push_idx, tuple(path))
                    tb += 1; heapq.heappush(pq, (new_cost, tb, new_st))
        for j in range(n_s):
            if scan_set & (1 << j): continue
            for vp_idx, (vp, vp_q4) in enumerate(vps_per_scan[j]):
                # 走到 vp 时尽量让最后一步方向 == vp_q4, 省 1-2 步 rotate.
                res = walk_path_optimal(push_idx, lp, vp,
                                           scan_set=scan_set, prefer_end_q4=vp_q4)
                if res is None: continue
                wlen, path = res
                new_mask = scan_set | reveals_along_walk(lp, path, step_k=push_idx)
                arrive_q4 = _walk_end_q4(lp, path, q4)
                rot_n = abs((vp_q4 - arrive_q4) % 4); rot_n = min(rot_n, 4 - rot_n)
                new_mask |= reveals_along([vp], vp_q4)
                new_mask |= cum_auto_mask[push_idx]
                new_real = real_steps[st] + wlen + rot_n
                new_cost = c + wlen + rot_n
                new_st = (push_idx, new_mask, vp, vp_q4)
                if new_cost < dist.get(new_st, INFV) - 1e-9:
                    dist[new_st] = new_cost; real_steps[new_st] = new_real
                    parent[new_st] = st
                    parent_act[new_st] = ("scan", j, vp_idx, tuple(path))
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


def planner_oracle_v14(eng: GameEngine, *, alpha: float = 0.4) -> None:
    plan = _god_plan(eng, time_limit=30.0)
    if not plan: return
    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    # 不污染 engine state, 见 oracle_v14_min_steps 同位说明.
    scans = _list_scans(state0, fb, ft)
    clone = _fresh_engine_from_eng(eng)
    res = oracle_v14_min_steps(clone, plan=plan, alpha=alpha)
    if res is None:
        from experiments.min_steps.planner_oracle_v12 import planner_oracle_v12
        planner_oracle_v12(eng, alpha=alpha); return
    vps_per_scan = _build_entity_vps(scans, state0)
    _execute_actions_v14(eng, res.interleaving, plan, vps_per_scan, scans=scans)


def _execute_actions_v14(eng: GameEngine, interleaving, plan, vps_per_scan,
                          scans=None) -> None:
    """执行 v14 interleaving. Action 形式:
       ("push", k, path)  ("scan", j, vp_idx, path)  ("rot", new_q4)
    旧 3-tuple 形式 fallback 到 _walk_to_executor.

    优化: scan 目标 entity 已被 engine FOV 识别 (walk-reveal 顺路) → 跳过该 scan + 它的 walk.
    """
    push_iter = iter(plan)
    for action in interleaving:
        if eng.get_state().won: return
        kind = action[0]
        if kind == "push":
            k = action[1]
            path = action[2] if len(action) >= 3 else None
            move = next(push_iter)
            push_pos = _get_push_pos(move)
            if path:
                _walk_path_executor(eng, path, "push_walk")
            else:
                _walk_to_executor(eng, push_pos, "push_walk")
            eng._step_tag = "push"
            eng.discrete_step(direction_to_abs_action(*move[2]))
        elif kind == "scan":
            j = action[1]; vp_idx = action[2]
            path = action[3] if len(action) >= 4 else None
            if vp_idx >= len(vps_per_scan[j]): continue
            # 跳过冗余 scan: 目标 entity 已通过 walk-reveal 识别
            if scans is not None and j < len(scans):
                et, ei = scans[j][2], scans[j][3]
                s = eng.get_state()
                if et == 'box' and ei in s.seen_box_ids: continue
                if et == 'target' and ei in s.seen_target_ids: continue
            vp, vp_q4 = vps_per_scan[j][vp_idx]
            if path:
                _walk_path_executor(eng, path, "scan_walk")
            else:
                _walk_to_executor(eng, vp, "scan_walk")
            _rotate_executor(eng, Q4_RAD[vp_q4], "scan_rot")
        elif kind == "rot":
            new_q4 = action[1]
            _rotate_executor(eng, Q4_RAD[new_q4], "proactive_rot")


def _walk_path_executor(eng: GameEngine, path, tag: str) -> bool:
    """沿着 planner 已选的具体 path 走 (每步 = 走相邻格子). 返回 True 表示成功抵达 path[-1].

    如果某步因 engine state 与 planner 预测不符 (车没动) 而失败, 立即回退到标准 BFS.
    """
    from smartcar_sokoban.solver.pathfinder import pos_to_grid as _pos2g
    eng._step_tag = tag
    for cell in path:
        state = eng.get_state()
        cur = _pos2g(state.car_x, state.car_y)
        if cur == cell: continue
        dx = cell[0] - cur[0]; dy = cell[1] - cur[1]
        if abs(dx) + abs(dy) != 1:
            # 跳格 — planner 路径与实际位置已发散, fallback BFS 到 path 终点
            return _walk_to_executor(eng, path[-1], tag)
        eng.discrete_step(direction_to_abs_action(dx, dy))
        state = eng.get_state()
        new_pos = _pos2g(state.car_x, state.car_y)
        if new_pos != cell:
            # 这一步被 engine 拒绝, fallback
            return _walk_to_executor(eng, path[-1], tag)
    return True
