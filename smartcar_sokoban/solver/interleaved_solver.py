"""交错探索 + 推送的求解器 — 真正的 partial observability 路径规划.

每个 outer step:
    1. 用 BeliefState (FOV-seen + ID 排除推理) 算 Π 矩阵
    2. 若所有 box 的 Π 行唯一 (= 目标已确定): 调 MultiBoxSolver 全程求解, 一次性推完
    3. 若至少 1 个 box 的 target 唯一: 推这一个 (然后回外圈循环 — 推送过程的 FOV 更新可能让更多 box 变 determined)
    4. 否则: 扫描下一个 entity (find_observation_point + 走过去 + 旋转面对)

对比旧 2-phase (`solve_exact`):
- 旧: 先全部探索 (~45 步) → 再一次性求解 (~81 步) = 126 步
- 新: 边探索边推, 推送过程顺路触发 FOV 识别, 排除推理立即生效, 节约不必要的扫描.
"""

from __future__ import annotations

import contextlib
import io
from typing import List, Optional, Tuple

from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.explorer import (
    compute_facing_actions,
    find_observation_point,
    get_all_entity_positions,
    get_entity_obstacles,
    get_scan_targets,
    direction_to_action,
    exploration_complete,
)
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.solver.push_solver import bfs_push
from smartcar_sokoban.symbolic.belief import BeliefState


# ── helpers ─────────────────────────────────────────────────

def _compute_pi(state) -> "np.ndarray":
    """从 engine state 算 Π 矩阵 (含 ID 排除推理)."""
    bs = BeliefState.from_engine_state(state, fully_observed=False)
    return bs.Pi


def _determined_pairs(Pi) -> List[Tuple[int, int]]:
    """返回 [(box_idx, target_idx)], 仅对 Π 行唯一的 box."""
    pairs = []
    n_box, n_tgt = Pi.shape
    for i in range(n_box):
        ones = [j for j in range(n_tgt) if Pi[i, j] > 0.5]
        if len(ones) == 1:
            pairs.append((i, ones[0]))
    return pairs


# ── apply solver_move (跟 build_dataset_v3 一致) ───────────

def _apply_solver_move(engine: GameEngine, move) -> List[int]:
    """复制 build_dataset_v3.apply_solver_move 的逻辑, 返回低层动作 list."""
    etype, eid, direction, _ = move
    actions = [6]   # snap
    engine.discrete_step(6)

    state = engine.get_state()
    dx, dy = direction
    if etype == "box":
        old_pos, _ = eid
        ec, er = old_pos
    elif etype == "bomb":
        ec, er = eid
    else:
        return []
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
            return None
        for pdx, pdy in path:
            a = direction_to_abs_action(pdx, pdy)
            engine.discrete_step(a)
            actions.append(a)

    a = direction_to_abs_action(dx, dy)
    engine.discrete_step(a)
    actions.append(a)
    return actions


# ── 单个 box 推到 target ─────────────────────────────────

def _push_one_determined_box(engine: GameEngine,
                             box_idx: int,
                             tgt_idx: int) -> Optional[List[int]]:
    """对单个已确定 box 跑 push BFS (其他 box / bomb 当障碍)."""
    state = engine.get_state()
    box = state.boxes[box_idx]
    tgt = state.targets[tgt_idx]

    car_grid = pos_to_grid(state.car_x, state.car_y)
    box_grid = pos_to_grid(box.x, box.y)
    tgt_grid = pos_to_grid(tgt.x, tgt.y)

    if box_grid == tgt_grid:
        return []  # 已在目标

    obstacles = set()
    for i, b in enumerate(state.boxes):
        if i == box_idx:
            continue
        obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs:
        obstacles.add(pos_to_grid(bm.x, bm.y))

    directions = bfs_push(car_grid, box_grid, tgt_grid, state.grid, obstacles)
    if directions is None:
        return None   # 单箱 BFS 解不出 (可能要炸弹)

    # 转成低层动作 (跟 build_dataset_v3 apply_solver_move 一致 — 由车移动一格直接撞箱)
    actions = []
    for dx, dy in directions:
        a = direction_to_abs_action(dx, dy)
        engine.discrete_step(a)
        actions.append(a)
    return actions


# ── 单次扫描迭代 ────────────────────────────────────────

def _execute_one_scan(engine: GameEngine) -> Optional[List[int]]:
    """从当前状态选最近未识别 entity, 走过去 + 旋转面对. 返回低层动作 list."""
    state = engine.get_state()
    targets = get_scan_targets(state)
    if not targets:
        return None
    car_grid = pos_to_grid(state.car_x, state.car_y)
    obstacles = get_entity_obstacles(state)
    entity_pos = get_all_entity_positions(state)

    best_entity = None
    best_actions_list = None
    best_cost = float('inf')

    for ex, ey, etype, eidx in targets:
        eg = pos_to_grid(ex, ey)
        result = find_observation_point(
            car_grid, eg, state.grid, obstacles, entity_pos,
            current_angle=state.car_angle,
        )
        if result is None:
            continue
        obs_pos, face_angle = result
        path = bfs_path(car_grid, obs_pos, state.grid, obstacles)
        if path is None:
            continue
        face_acts = compute_facing_actions(state.car_angle, face_angle)
        cost = len(path) + len(face_acts)
        if cost < best_cost:
            best_cost = cost
            best_entity = (etype, eidx)
            best_actions_list = (path, face_angle)

    if best_entity is None:
        # 兜底 360° 扫描
        actions = []
        for _ in range(8):
            engine.discrete_step(5)
            actions.append(5)
            if exploration_complete(engine.get_state()):
                break
        return actions if actions else None

    path, face_angle = best_actions_list
    actions = []

    # 走过去
    for dx, dy in path:
        a = direction_to_action(dx, dy)
        engine.discrete_step(a)
        actions.append(a)

    # 旋转面对
    face_acts = compute_facing_actions(engine.get_state().car_angle, face_angle)
    for a in face_acts:
        engine.discrete_step(a)
        actions.append(a)

    return actions


# ── 全求解 ──────────────────────────────────────────────

def _run_full_solver_apply_all(engine: GameEngine,
                                time_limit: float) -> Optional[List[int]]:
    """所有 box 的 target 已确定 → 调 MultiBoxSolver, 用 solution_to_actions
    把整段路径合并成 directions list (不在每次推前 snap, 跟 preview_failed 等价)."""
    state = engine.get_state()
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    with contextlib.redirect_stdout(io.StringIO()):
        solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
        moves = solver.solve(max_cost=1000, time_limit=time_limit, strategy='auto')
    if moves is None:
        return None

    # solution_to_actions: 整段 directions, no inter-push snaps
    directions = solver.solution_to_actions(moves)

    # 头部一个 snap (跟探索结尾衔接)
    actions = [6]
    engine.discrete_step(6)

    for dx, dy in directions:
        a = direction_to_abs_action(dx, dy)
        engine.discrete_step(a)
        actions.append(a)
        if engine.get_state().won:
            break
    return actions


# ── 主入口 ──────────────────────────────────────────────

def solve_exact_interleaved(engine: GameEngine,
                             time_limit: float = 60.0,
                             allow_single_push: bool = False,
                             verbose: bool = True) -> Optional[Tuple[List[int], dict]]:
    """交错求解 + 探索, 返回 (低层动作 list, 统计 dict).

    Args:
        allow_single_push: 若 True, 在部分 box 已 determined 时尝试单箱 BFS 推送
            (实测在 phase 4 多箱协调上反而更差因为不全局优化, 默认关).
    """
    all_actions: List[int] = []
    stats = {
        "outer_iterations": 0,
        "scan_iterations": 0,
        "single_push_iterations": 0,
        "full_solver_runs": 0,
        "total_explore_steps": 0,
        "total_push_steps": 0,
    }
    max_outer = 60

    for outer in range(max_outer):
        stats["outer_iterations"] = outer + 1
        state = engine.get_state()
        if state.won:
            break

        Pi = _compute_pi(state)
        n_box = len(state.boxes)
        if n_box == 0:
            break

        determined = _determined_pairs(Pi)
        all_determined = len(determined) == n_box

        if all_determined:
            if verbose:
                print(f"  [iter {outer + 1}] all {n_box} boxes determined → 全求解")
            push_actions = _run_full_solver_apply_all(engine, time_limit=time_limit)
            if push_actions is None:
                if verbose:
                    print("    全求解失败")
                return None
            all_actions.extend(push_actions)
            stats["total_push_steps"] += len(push_actions)
            stats["full_solver_runs"] += 1
            continue

        if allow_single_push and determined:
            box_idx, tgt_idx = determined[0]
            if verbose:
                print(f"  [iter {outer + 1}] {len(determined)}/{n_box} 已 determined "
                      f"→ 推 box_{box_idx} 到 target_{tgt_idx}")
            push_actions = _push_one_determined_box(engine, box_idx, tgt_idx)
            if push_actions:
                all_actions.extend(push_actions)
                stats["total_push_steps"] += len(push_actions)
                stats["single_push_iterations"] += 1
                continue
            elif verbose:
                print(f"    单箱 BFS 解不出 (可能要炸弹), 改扫描")

        if verbose:
            print(f"  [iter {outer + 1}] 还有 unknown ({n_box - len(determined)} 未定) → 扫描")
        scan_actions = _execute_one_scan(engine)
        if scan_actions is None:
            if verbose:
                print("    无可扫描 entity, 停止")
            break
        all_actions.extend(scan_actions)
        stats["total_explore_steps"] += len(scan_actions)
        stats["scan_iterations"] += 1

    if not engine.get_state().won:
        return None

    return all_actions, stats
