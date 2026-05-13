"""Planner v1 — opportunistic inspect.

思路:
  1. 先用 god-mode MultiBoxSolver 拿完整 push 序列 P (作为执行模板)
  2. partial-obs 重放: 维护 belief, 推每一步前检查
      a. 当前 push 是否 belief-feasible (=> enforce_sigma_lock 给出 legal)
      b. 不可 → 选 'pick_inspect_for_unlock' 最便宜的 inspect, 走过去, 旋转扫描
      c. 可 → BFS 到推位, 顺路 FOV 自动更新 belief, 推
  3. 重复直到 P 走完, engine.won 应为 True

期望:
  - 大多数 push 的 BFS 路径会经过 box / target 旁边, FOV 自动揭示
  - 只有少数 (typically 0-1 个) 需要专门 detour 去 inspect
  - 总步数 ≈ god push_only + (少量 inspect detour)
"""

from __future__ import annotations

import contextlib
import io
import random
from typing import List, Optional

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import (
    generate_candidates, Candidate,
)

from experiments.sage_pr.build_dataset_v6 import pick_inspect_for_unlock
from experiments.sage_pr.belief_ida_solver import apply_inspect
from experiments.sage_pr.build_dataset_v3 import match_move_to_candidate


def _god_plan(eng: GameEngine, time_limit: float = 30.0) -> Optional[List]:
    """跑 god-mode multi-box solver, 拿 push 序列."""
    state = eng.get_state()
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return solver.solve(max_cost=300, time_limit=time_limit,
                                 strategy="auto")
        except Exception:
            return None


def _apply_push_move(eng: GameEngine, move) -> bool:
    """重新实现 apply_solver_move 但显式分 tag (BFS-walk vs 推).
    注: 不做 snap, 它是 no-op."""
    etype, eid, direction, _ = move
    state = eng.get_state()
    dx, dy = direction
    if etype == "box":
        old_pos, _cid = eid
        ec, er = old_pos
    elif etype == "bomb":
        ec, er = eid
    else:
        return False
    car_target = (ec - dx, er - dy)

    obstacles = set()
    for b in state.boxes:
        obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs:
        obstacles.add(pos_to_grid(bm.x, bm.y))
    car_grid = pos_to_grid(state.car_x, state.car_y)

    if car_grid != car_target:
        path = bfs_path(car_grid, car_target, state.grid, obstacles)
        if path is None:
            return False
        eng._step_tag = "push_walk"
        for pdx, pdy in path:
            eng.discrete_step(direction_to_abs_action(pdx, pdy))

    eng._step_tag = "push"
    eng.discrete_step(direction_to_abs_action(dx, dy))
    return True


def _apply_inspect_tagged(eng: GameEngine, cand: Candidate) -> bool:
    """跟 belief_ida_solver.apply_inspect 一致, 但 tag 分类. 不做 snap."""
    state = eng.get_state()
    obstacles = set()
    for b in state.boxes:
        obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs:
        obstacles.add(pos_to_grid(bm.x, bm.y))
    car_grid = pos_to_grid(state.car_x, state.car_y)
    target = (cand.viewpoint_col, cand.viewpoint_row)
    if car_grid != target:
        path = bfs_path(car_grid, target, state.grid, obstacles)
        if path is None:
            return False
        eng._step_tag = "inspect_walk"
        for pdx, pdy in path:
            eng.discrete_step(direction_to_abs_action(pdx, pdy))
    state = eng.get_state()

    # 旋转面向 viewpoint
    import math
    HEADING_TO_ANGLE = {0: 0, 1: math.pi/2, 2: math.pi, 3: -math.pi/2,
                        4: math.pi/4, 5: 3*math.pi/4, 6: -3*math.pi/4,
                        7: -math.pi/4}
    tgt_a = HEADING_TO_ANGLE.get(cand.inspect_heading or 0, 0.0)
    diff = math.atan2(math.sin(tgt_a - state.car_angle),
                      math.cos(tgt_a - state.car_angle))
    n = round(diff / (math.pi / 2))
    eng._step_tag = "inspect_rot"
    if n == 2 or n == -2:
        eng.discrete_step(5); eng.discrete_step(5)
    elif n == 1:
        eng.discrete_step(5)
    elif n == -1:
        eng.discrete_step(4)
    return True


def planner_v1_opportunistic(eng: GameEngine,
                              *, max_inspects_per_push: int = 8,
                              god_time_limit: float = 30.0) -> None:
    """主 planner."""
    plan = _god_plan(eng, time_limit=god_time_limit)
    if plan is None:
        return

    a_idx = 0
    inspect_streak = 0

    while a_idx < len(plan):
        state = eng.get_state()
        if state.won:
            return

        bs = BeliefState.from_engine_state(state, fully_observed=False)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat, enforce_sigma_lock=True)

        move = plan[a_idx]
        label = match_move_to_candidate(move, cands, bs, run_length=1)

        if label is not None:
            # push belief-feasible → 直接推. BFS-walk 时 FOV 自动更新 belief.
            ok = _apply_push_move(eng, move)
            if not ok:
                return
            a_idx += 1
            inspect_streak = 0
            continue

        # push 不合法 (大多是 σ 未锁) → 找最便宜 inspect
        if inspect_streak >= max_inspects_per_push:
            # 兜底: 用无抑制的 cand 推 (相当于"赌博")
            cands_loose = generate_candidates(bs, feat, enforce_sigma_lock=False)
            label2 = match_move_to_candidate(move, cands_loose, bs, run_length=1)
            if label2 is None:
                # 真没法 — apply 引擎层 push (不再用 candidate)
                eng._step_tag = "fallback"
                ok = _apply_push_move(eng, move)
                if not ok:
                    return
            else:
                ok = _apply_push_move(eng, move)
                if not ok:
                    return
            a_idx += 1
            inspect_streak = 0
            continue

        # 选 inspect (沿用 V2 的 pick)
        required_box = -1
        required_target = -1
        if move[0] == "box":
            old_pos, cid = move[1]
            for j, b in enumerate(bs.boxes):
                if (b.col, b.row) == old_pos:
                    required_box = j
                    break
            for j, t in enumerate(bs.targets):
                if t.num_id == cid:
                    required_target = j
                    break

        ins_label = pick_inspect_for_unlock(bs, cands, required_box, required_target)
        if ins_label is None:
            # 没 inspect 可做 → fallback
            eng._step_tag = "fallback"
            ok = _apply_push_move(eng, move)
            if not ok:
                return
            a_idx += 1
            inspect_streak = 0
            continue

        ins_cand = cands[ins_label]
        ok = _apply_inspect_tagged(eng, ins_cand)
        if not ok:
            return
        inspect_streak += 1
