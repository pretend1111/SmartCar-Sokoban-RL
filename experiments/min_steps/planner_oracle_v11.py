"""Planner oracle v11 — best-of-many α per map.

每张图试 v10 with multiple α 值, 选最佳 (step + lambda × gambling_count) 的 plan.

不同 map 可能需要不同 α: 简单图低 α 足够, 复杂图需高 α 强制 scan.
"""

import copy
from typing import List, Tuple
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs

from experiments.min_steps.planner_oracle import (
    _god_plan, _list_scans, _get_push_pos, _fresh_engine_from_eng,
    _walk_to_executor, _rotate_executor,
)
from experiments.min_steps.planner_oracle_v4 import (
    OracleResult4, _build_entity_vps, Q4_RAD,
)
from experiments.min_steps.planner_oracle_v10 import oracle_v10_min_steps


def _count_gambling(plan: List, scan_interleave: List[Tuple]) -> int:
    """计数 interleave 中, 首次 push 该 box 之前是否已 scanned 该 box (或它的 target).
    简化: 数 ('push', k) 在 ('scan', j, ...) 出现之前的次数."""
    scanned_boxes_by_idx = set()
    seen_first_push = set()
    g = 0
    for action in scan_interleave:
        if action[0] == 'scan':
            # 标记该 scan 涉及的 box_class (但 v10 接口下 scan idx 对应 scan list)
            pass   # 简化, 这个不准. 用直接 gambling count from oracle
        elif action[0] == 'push':
            k = action[1]
            etype, eid, _, _ = plan[k]
            if etype != 'box': continue
            _, cid = eid
            if cid in seen_first_push: continue
            seen_first_push.add(cid)
            # Was scanned before? Approx: check ('scan' ... ) entries earlier
            g_potential = True
            # without precise lookup, assume "not yet scanned" → counts as gambling
            # 这里返回 0 是因为 oracle 内部已优化 — 不是这个函数的正确实现
    return g


def planner_oracle_v11(eng: GameEngine,
                        *, alphas: Tuple[float, ...] = (0.1, 0.4, 1.0, 2.0),
                        eval_bias: float = 2.0) -> None:
    """对每个 α 跑 v10, 评估 plan 的 (step + bias × gambling_count), 取最低."""
    plan = _god_plan(eng, time_limit=30.0)
    if not plan: return
    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    state0.seen_box_ids.update(fb); state0.seen_target_ids.update(ft)
    scans = _list_scans(state0, fb, ft)

    # 评估每个 α 对应的 plan score
    best_score = float('inf')
    best_res = None
    for alpha in alphas:
        clone = _fresh_engine_from_eng(eng)
        res = oracle_v10_min_steps(clone, plan=plan, alpha=alpha)
        if res is None: continue
        # 数 gambling: 找 first-push 是否被 scan 在前
        # 一个 push k 之前的 scans 包含哪些 entities?
        # 直接用 plan 反推: 找 first-push 时 scan_set 状态
        scan_set = 0
        first_push_seen = set()
        gambling_count = 0
        forced_classes_set = {state0.boxes[bi].class_id for bi, _ in forced if bi < len(state0.boxes)}
        consumed_classes = set()
        N = len(state0.boxes); universe = {b.class_id for b in state0.boxes}

        # Process interleave in order
        for action in res.interleaving:
            if action[0] == 'scan':
                _, j, _ = action
                scan_set |= (1 << j)
            elif action[0] == 'push':
                _, k = action
                etype, eid, _, _ = plan[k]
                if etype != 'box': continue
                _, cid = eid
                if cid in first_push_seen: continue
                # 计算 effective known
                explicit = forced_classes_set | consumed_classes
                # 从 scan_set 加 scanned class_ids
                for j in range(len(scans)):
                    if scan_set & (1 << j):
                        vp, ang, et, ei = scans[j]
                        if et == 'box':
                            explicit.add(state0.boxes[ei].class_id)
                        # also target's num
                        # (skip num for simplicity)
                eff = universe if len(explicit) >= N - 1 else explicit
                first_push_seen.add(cid)
                if cid not in eff:
                    gambling_count += 1
                # 标记 consumed (if last push of this box)
                # — 简化省略
        score = res.cost + eval_bias * gambling_count
        if score < best_score:
            best_score = score
            best_res = res

    if best_res is None:
        from experiments.min_steps.planner_oracle_v6 import planner_oracle_v6
        planner_oracle_v6(eng); return

    vps_per_scan = _build_entity_vps(scans, state0)
    push_iter = iter(plan)
    for action in best_res.interleaving:
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
