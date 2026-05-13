"""Planner v3 — JIT plan_exploration.

策略:
  1. 先用 god plan 拿 push 序列
  2. 走每个 push_pos (FOV 沿路自动更新 belief)
  3. 走到后检查 σ. 不锁 → 调 plan_exploration (它的全局调度比单 inspect 强):
        - plan_exploration 自动用排除律: N-1 个就够
        - explorer_v3 自动跳 forced pairs
        - explore 从当前 car 位置出发, 不强行回到 t=0 起点
     探完再 try push, 若 σ 还不锁 (罕见) → 再来一次或 fallback

  与 v1_explore_first 的差异: 不在 t=0 一次性 explore 完, 而是 push 路上 FOV 顺路扫,
     真不够才 JIT 触发 explore. 期望省掉那些"FOV 顺路就能扫到"的 entity 的专项扫描.
"""

from __future__ import annotations

import contextlib
import io
from typing import List, Optional, Set, Tuple

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import generate_candidates
from smartcar_sokoban.solver.explorer import exploration_complete

from experiments.sage_pr.build_dataset_v3 import match_move_to_candidate


def _god_plan(eng: GameEngine, time_limit: float = 30.0) -> Optional[List]:
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


def _walk_to(eng: GameEngine, target: Tuple[int, int], tag: str) -> bool:
    state = eng.get_state()
    obstacles: Set[Tuple[int, int]] = set()
    for b in state.boxes:
        obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs:
        obstacles.add(pos_to_grid(bm.x, bm.y))
    car_grid = pos_to_grid(state.car_x, state.car_y)
    if car_grid == target:
        return True
    path = bfs_path(car_grid, target, state.grid, obstacles)
    if path is None:
        return False
    eng._step_tag = tag
    for pdx, pdy in path:
        eng.discrete_step(direction_to_abs_action(pdx, pdy))
    return True


def _get_push_pos(move) -> Tuple[int, int]:
    etype, eid, direction, _ = move
    dx, dy = direction
    if etype == "box":
        old_pos, _ = eid
        ec, er = old_pos
    elif etype == "bomb":
        ec, er = eid
    else:
        raise ValueError
    return (ec - dx, er - dy)


def _jit_explore(eng: GameEngine) -> None:
    """从当前 engine 状态调 plan_exploration_v3, 应用它的 action list."""
    from smartcar_sokoban.solver.explorer_v3 import plan_exploration_v3
    eng._step_tag = "explore_jit"
    with contextlib.redirect_stdout(io.StringIO()):
        plan_exploration_v3(eng, max_retries=15, verbose=False)


def planner_v3_jit_explore(eng: GameEngine,
                            *, god_time_limit: float = 30.0,
                            max_explore_rounds: int = 2) -> None:
    plan = _god_plan(eng, time_limit=god_time_limit)
    if plan is None:
        return

    for move in plan:
        state = eng.get_state()
        if state.won:
            return

        push_pos = _get_push_pos(move)
        # 1) 走到 push_pos (FOV 沿路免费扫)
        if not _walk_to(eng, push_pos, "push_walk"):
            return

        # 2) 检查这一步 push 是否 belief-feasible
        for r in range(max_explore_rounds + 1):
            state = eng.get_state()
            bs = BeliefState.from_engine_state(state, fully_observed=False)
            feat = compute_domain_features(bs)
            cands = generate_candidates(bs, feat, enforce_sigma_lock=True)
            label = match_move_to_candidate(move, cands, bs, run_length=1)
            if label is not None:
                break
            # 不锁 → JIT explore
            _jit_explore(eng)
            # explore 后, 车可能不在 push_pos. 重新走过去
            state = eng.get_state()
            cur = pos_to_grid(state.car_x, state.car_y)
            if cur != push_pos:
                if not _walk_to(eng, push_pos, "push_walk_back"):
                    return

        # 3) 推
        eng._step_tag = "push"
        eng.discrete_step(direction_to_abs_action(*move[2]))
