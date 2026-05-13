"""Planner oracle v17 — v14 + chain-permutation best-of.

枚举 god plan 的 box-chain 排列, 每个 feasible 排列跑 v14 oracle, 取最低 cost.

ROI: 不同 chain 顺序 god cost 相同, 但 oracle 看到的 belief 状态不同 → 不同
gambling 模式. 找最匹配的.
"""

import contextlib, io, itertools, math, copy as _cp, random
from typing import Dict, List, Optional, Tuple
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.action_defs import direction_to_abs_action

from experiments.min_steps.planner_oracle import (
    _god_plan, _fresh_engine_from_eng,
)
from experiments.min_steps.planner_oracle_v4 import OracleResult4
from experiments.min_steps.planner_oracle_v14 import oracle_v14_min_steps


def _group_chains(plan: List) -> List[List]:
    """按 box class_id 分组 chains."""
    by_class = {}
    for m in plan:
        et, eid, _, _ = m
        if et == 'box':
            _, cid = eid
            by_class.setdefault(cid, []).append(m)
        else:
            by_class.setdefault(f'__bomb_{id(m)}', []).append(m)
    return list(by_class.values())


def _is_plan_feasible(eng_init: GameEngine, plan: List) -> bool:
    """跑一遍 plan in fresh engine, 检查 engine.won."""
    from experiments.sage_pr.build_dataset_v3 import apply_solver_move
    e = _fresh_engine_from_eng(eng_init)
    for m in plan:
        if not apply_solver_move(e, m): return False
    return e.get_state().won


def oracle_v17_min_steps(eng_init: GameEngine,
                          *, god_time_limit: float = 30.0,
                          alpha: float = 0.7,
                          max_perms: int = 6
                          ) -> Optional[OracleResult4]:
    plan0 = _god_plan(eng_init, time_limit=god_time_limit)
    if not plan0: return None
    chains = _group_chains(plan0)
    n_chains = len(chains)
    if n_chains <= 1:
        return oracle_v14_min_steps(eng_init, plan=plan0, alpha=alpha)
    n_perms = math.factorial(n_chains)
    if n_perms > max_perms:
        # 随机采样 max_perms-1 + 原顺序
        seen = {tuple(range(n_chains))}
        perms = [tuple(range(n_chains))]
        while len(perms) < max_perms:
            p = list(range(n_chains))
            random.shuffle(p)
            t = tuple(p)
            if t not in seen:
                seen.add(t); perms.append(t)
    else:
        perms = list(itertools.permutations(range(n_chains)))

    best_res = None
    best_score = float('inf')
    for perm in perms:
        var_plan = []
        for idx in perm:
            var_plan.extend(chains[idx])
        if not _is_plan_feasible(eng_init, var_plan): continue
        try:
            res = oracle_v14_min_steps(eng_init, plan=var_plan, alpha=alpha)
        except Exception:
            continue
        if res is None: continue
        # 用 res.cost 作 score (real steps, already含 penalty 是 minimized indirectly)
        if res.cost < best_score:
            best_score = res.cost
            best_res = res
            best_res._plan = var_plan
    return best_res


def planner_oracle_v17(eng: GameEngine, *, alpha: float = 0.7, max_perms: int = 6) -> None:
    from experiments.min_steps.planner_oracle import (
        _list_scans, _get_push_pos, _walk_to_executor, _rotate_executor,
    )
    from experiments.min_steps.planner_oracle_v4 import (
        _build_entity_vps, Q4_RAD,
    )
    from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs

    plan0 = _god_plan(eng, time_limit=30.0)
    if not plan0: return
    clone = _fresh_engine_from_eng(eng)
    res = oracle_v17_min_steps(clone, alpha=alpha, max_perms=max_perms)
    if res is None:
        from experiments.min_steps.planner_oracle_v14 import planner_oracle_v14
        planner_oracle_v14(eng, alpha=alpha); return
    plan = getattr(res, '_plan', plan0)

    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    state0.seen_box_ids.update(fb); state0.seen_target_ids.update(ft)
    scans = _list_scans(state0, fb, ft)
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
