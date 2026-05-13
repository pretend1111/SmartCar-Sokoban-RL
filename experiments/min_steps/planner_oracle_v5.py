"""Planner oracle v5 — multi god plan via box-chain permutation.

观察: god plan 把 (box A, target T_A) 推送拆成 box-chain (连续推 box A 直到 target).
不同 box 之间没有依赖 (假设无 bomb / 推链), chain 顺序可自由排列, 总 push count
和 walk cost 略有变化但 god 仍可行.

策略:
  1. 跑 god plan → push moves
  2. 把 moves 按 (box_class_id) 分组成 chains
  3. 枚举 chains 的排列 (n! 个, n=box count)
  4. 对每个排列, 重建 push sequence (重新算 walk_cost), 跑 oracle_v4
  5. 取 oracle cost 最低的

约束: 仅对 n ≤ 5 启用 (n!=120 上限)
"""

from __future__ import annotations

import itertools
import math
import random
from typing import List, Optional, Tuple, Dict

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action

from experiments.min_steps.planner_oracle import (
    _god_plan, _fresh_engine_from_eng,
)
from experiments.min_steps.planner_oracle_v4 import (
    oracle_v4_min_steps, OracleResult4, _bfs_path_cells,
    _build_entity_vps, _walk_to_executor, _rotate_executor,
    Q4_RAD,
)
from experiments.min_steps.planner_oracle import (
    _list_scans, _get_push_pos,
)
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs


def _group_into_chains(plan: List) -> List[List]:
    """把 god plan 按 box 分组成 chains (同一 box 的连续推送序列保持原顺序).

    Returns: list of chains, each chain = list of moves (preserved order).
    """
    if not plan:
        return []
    # 推 box 序列: 每个 chain = 推一个 box 从 start 到 final target
    chains_by_class: Dict[int, List] = {}
    for move in plan:
        etype, eid, direction, _ = move
        if etype == "box":
            old_pos, class_id = eid
            chains_by_class.setdefault(class_id, []).append(move)
        elif etype == "bomb":
            # bomb 单独 chain (key 用 negative)
            bomb_pos = eid
            chains_by_class.setdefault(-100 - bomb_pos[0]*16 - bomb_pos[1], []).append(move)
    return list(chains_by_class.values())


def _enumerate_plans(plan: List, max_perms: int = 24) -> List[List]:
    """生成 plan 的所有 chain 排列变体 (上限 max_perms 个).

    限制: 仅对 n ≤ 5 的 chain count 枚举全部, 否则随机采样.
    """
    chains = _group_into_chains(plan)
    n = len(chains)
    if n <= 1:
        return [plan]
    if math.factorial(n) <= max_perms:
        perms = list(itertools.permutations(range(n)))
    else:
        # 太多 — 随机采样 + 始终包含原顺序
        seen = {tuple(range(n))}
        perms = [tuple(range(n))]
        for _ in range(max_perms - 1):
            p = list(range(n))
            random.shuffle(p)
            t = tuple(p)
            if t not in seen:
                seen.add(t); perms.append(t)
            if len(perms) >= max_perms: break

    variants = []
    for perm in perms:
        variant = []
        for idx in perm:
            variant.extend(chains[idx])
        variants.append(variant)
    return variants


def _validate_and_compute_cost(eng_init: GameEngine, plan: List) -> Optional[int]:
    """在 fresh engine 上模拟 plan, 看是否可行.
    Returns total engine step count, None if infeasible."""
    import copy as _cp
    from experiments.sage_pr.build_dataset_v3 import apply_solver_move
    e = _fresh_engine_from_eng(eng_init)
    steps = 0
    orig = e.discrete_step
    def wrapped(a):
        nonlocal steps
        steps += 1
        return orig(a)
    e.discrete_step = wrapped   # type: ignore
    for move in plan:
        if not apply_solver_move(e, move):
            return None
    if not e.get_state().won:
        return None
    return steps


def oracle_v5_min_steps(eng_init: GameEngine,
                          *, god_time_limit: float = 30.0,
                          max_perms: int = 12
                          ) -> Optional[OracleResult4]:
    """跑所有 box-chain 排列, 选 oracle cost 最低的."""
    plan = _god_plan(eng_init, time_limit=god_time_limit)
    if not plan:
        return None

    variants = _enumerate_plans(plan, max_perms=max_perms)

    best_result = None
    best_cost = float('inf')
    best_plan = plan
    for var_plan in variants:
        # 验证可行
        feas = _validate_and_compute_cost(eng_init, var_plan)
        if feas is None: continue
        # oracle on this variant
        clone = _fresh_engine_from_eng(eng_init)
        res = oracle_v4_min_steps(clone, plan=var_plan)
        if res is None: continue
        if res.cost < best_cost:
            best_cost = res.cost
            best_result = res
            best_plan = var_plan
    if best_result is None:
        return None
    # 返回 result 但需要把 best_plan 一起传给 caller (这是 replay 用)
    best_result.cost = best_cost
    # hack: 把 plan 挂在 result 上
    best_result._plan = best_plan   # type: ignore[attr-defined]
    return best_result


def planner_oracle_v5(eng: GameEngine) -> None:
    """跑多 plan 候选, 取最佳, 在主 eng replay."""
    res = oracle_v5_min_steps(_fresh_engine_from_eng(eng))
    if res is None:
        from experiments.min_steps.planner_oracle_v4 import planner_oracle_v4
        planner_oracle_v4(eng); return

    plan = getattr(res, '_plan', None)
    if plan is None:
        from experiments.min_steps.planner_oracle_v4 import planner_oracle_v4
        planner_oracle_v4(eng); return

    # 计算 scans (跟 oracle v4 中一样)
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
