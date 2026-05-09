"""探索 + 推 交织求解器.

策略:
    1. 用 god-mode MultiBoxSolver 拿最优 push plan (假设 ID 全已知).
    2. 用 排除法 算需要 identify 的 entity 集合.
    3. 把 identify events 见缝插针插入 push events 之间, 使总动作步数最短.

    Greedy 插入: 对每个 identify event, 遍历所有插入点 (push 之前/之间/之后),
    选 (额外 walk cost) 最小的. 多个 identify 选完后按 (插入点, 距离) 渲染成
    完整低层动作序列.

接口:
    plan_interleaved(engine, time_limit=60.0) -> Tuple[List[low_level_actions], int]
"""

from __future__ import annotations

import contextlib
import io
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver, MBState, DIRS
from smartcar_sokoban.solver.explorer import (
    get_scan_targets, has_line_of_sight,
    get_entity_obstacles, get_all_entity_positions,
    compute_facing_actions, ANGLE_UP,
)
from smartcar_sokoban.solver.pathfinder import pos_to_grid, bfs_path
from smartcar_sokoban.action_defs import direction_to_abs_action


Pos = Tuple[int, int]
DIRS_8 = [(1, 0), (-1, 0), (0, 1), (0, -1),
          (1, 1), (1, -1), (-1, 1), (-1, -1)]


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class IdentTask:
    etype: str             # 'box' or 'target'
    eidx: int
    entity_grid: Pos
    obs_options: List[Tuple[int, int, float]]   # (col, row, face_angle_quantized)


@dataclass
class Insertion:
    """一个 identify 任务被插入到某个 push 之前的位置."""
    insert_before_push_k: int   # k ∈ [0, M]; k=0 表示在第一个 push 之前; k=M 表示最后
    ident: IdentTask
    obs_pos: Pos
    face_angle: float
    extra_cost: int             # 估算的额外步数


# ── BFS 辅助 ──────────────────────────────────────────────

def _bfs_dist(start: Pos, grid, occupied: Set[Pos]) -> Dict[Pos, int]:
    """从 start 出发 4 邻 BFS, 返回 dest → 最短步数."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if not (0 <= start[0] < cols and 0 <= start[1] < rows):
        return {}
    visited = {start}
    dist = {start: 0}
    q = deque([start])
    while q:
        c, r = q.popleft()
        d = dist[(c, r)]
        for dc, dr in DIRS:
            nc, nr = c + dc, r + dr
            if not (0 <= nc < cols and 0 <= nr < rows):
                continue
            if grid[nr][nc] == 1:
                continue
            if (nc, nr) in occupied:
                continue
            if (nc, nr) in visited:
                continue
            visited.add((nc, nr))
            dist[(nc, nr)] = d + 1
            q.append((nc, nr))
    return dist


def _bfs_path(start: Pos, goal: Pos, grid, occupied: Set[Pos]
              ) -> Optional[List[Tuple[int, int]]]:
    """4 邻 BFS, 返回方向列表 (dx, dy)."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if start == goal:
        return []
    visited = {start}
    q = deque([(start, [])])
    while q:
        (c, r), path = q.popleft()
        for dc, dr in DIRS:
            nc, nr = c + dc, r + dr
            new_path = path + [(dc, dr)]
            if (nc, nr) == goal:
                return new_path
            if not (0 <= nc < cols and 0 <= nr < rows):
                continue
            if grid[nr][nc] == 1:
                continue
            if (nc, nr) in occupied:
                continue
            if (nc, nr) in visited:
                continue
            visited.add((nc, nr))
            q.append(((nc, nr), new_path))
    return None


# ── identify 候选 obs 点 ──────────────────────────────────

def _adj_obs_cells(entity_grid: Pos, grid,
                   obstacles: Set[Pos], entity_positions: Set[Pos]
                   ) -> List[Tuple[int, int, float]]:
    """8 邻紧贴 entity 且有视线的格子, 返回 (col, row, face_angle_quantized)."""
    ec, er = entity_grid
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    out = []
    for dc, dr in DIRS_8:
        nc, nr = ec + dc, er + dr
        if not (0 <= nc < cols and 0 <= nr < rows):
            continue
        if grid[nr][nc] == 1:
            continue
        if (nc, nr) in obstacles:
            continue
        if not has_line_of_sight(nc, nr, ec, er, grid, entity_positions):
            continue
        face_angle = math.atan2(er - nr, ec - nc)
        face_q = round(face_angle / (math.pi / 4)) * (math.pi / 4)
        face_q = math.atan2(math.sin(face_q), math.cos(face_q))
        out.append((nc, nr, face_q))
    return out


def _angle_diff_steps(a1: float, a2: float) -> int:
    """两个 8-向量化角度之间最短旋转步数 (每步 π/4)."""
    diff = a1 - a2
    diff = math.atan2(math.sin(diff), math.cos(diff))
    steps = round(abs(diff) / (math.pi / 4))
    return min(steps, 8 - steps)


# ── 主流程 ────────────────────────────────────────────────

def plan_interleaved(engine, time_limit: float = 60.0,
                     verbose: bool = False
                     ) -> Optional[Tuple[List[int], int]]:
    """求解探索 + 推 交织 plan.

    Returns:
        (low_level_actions, total_steps) 或 None 失败.
    """
    state0 = engine.get_state()

    # ── 1. god-mode push 求解 ────────────────────────────
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state0.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state0.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state0.bombs]
    car_pos = pos_to_grid(state0.car_x, state0.car_y)

    solver = MultiBoxSolver(state0.grid, car_pos, boxes, targets, bombs)
    with contextlib.redirect_stdout(io.StringIO()):
        push_plan = solver.solve(strategy='auto', time_limit=time_limit)
    if push_plan is None:
        if verbose:
            print("  push solver 无解")
        return None
    if verbose:
        print(f"  push 求解: {len(push_plan)} 推操作")

    # ── 2. 构建 checkpoint (每次 push 前/后的状态) ──────
    M = len(push_plan)
    checkpoints: List[MBState] = [solver.initial]
    cur = solver.initial
    for etype, eid, direction, _ in push_plan:
        new_state = solver._apply_push(cur, etype, eid, direction)
        if new_state is None:
            if verbose:
                print(f"  apply_push 失败 (push={etype} {eid} {direction})")
            return None
        checkpoints.append(new_state)
        cur = new_state

    # ── 3. 收集需要 identify 的任务 ──────────────────────
    scan_data = get_scan_targets(state0)   # 已应用排除法 (N-1 of N)
    if verbose:
        print(f"  scan targets: {len(scan_data)}")

    obstacles0 = get_entity_obstacles(state0)
    entity_positions0 = get_all_entity_positions(state0)
    ident_tasks: List[IdentTask] = []
    for ex, ey, etype, eidx in scan_data:
        entity_grid = pos_to_grid(ex, ey)
        opts = _adj_obs_cells(entity_grid, state0.grid, obstacles0, entity_positions0)
        if not opts:
            if verbose:
                print(f"  ⚠️ {etype}_{eidx} at {entity_grid} 无有效 obs 点")
            continue
        ident_tasks.append(IdentTask(
            etype=etype, eidx=eidx, entity_grid=entity_grid, obs_options=opts,
        ))

    # ── 4. 预算每个 checkpoint 的 walk_dist 表 ─────────
    walk_dist_per_cp: List[Dict[Pos, int]] = []
    grid_per_cp: List[Any] = []
    occ_per_cp: List[Set[Pos]] = []
    for cp in checkpoints:
        g = solver._get_grid(cp.destroyed)
        occ = solver._get_occupied(cp)
        walk_dist_per_cp.append(_bfs_dist(cp.car, g, occ))
        grid_per_cp.append(g)
        occ_per_cp.append(occ)

    # ── 5. 贪心插入: 每个 ident 选 cheapest 插入点 ──────
    # 插入到 "checkpoint k 之前" = 在 push k 之前 (推 plan 索引).
    # 插入位置 k = M 表示最后 push 之后 (无下一段 walk).
    # 多 ident 同一 k 时按 "顺路" 排, 这里简化为按邻接顺序.

    insertions: Dict[int, List[Insertion]] = {}

    for task in ident_tasks:
        best: Optional[Insertion] = None
        best_cost = float('inf')

        for k in range(M + 1):
            wd = walk_dist_per_cp[k]
            grid = grid_per_cp[k]
            occ = occ_per_cp[k]

            for (acol, arow, face_q) in task.obs_options:
                # adj cell 在该 checkpoint 下是否仍可用
                if (acol, arow) in occ:
                    continue
                if grid[arow][acol] == 1:
                    continue
                walk_in = wd.get((acol, arow))
                if walk_in is None:
                    continue

                # walk back 估算: 从 adj_pos 到下一 checkpoint.car (若有)
                if k < M:
                    next_car = checkpoints[k + 1].car
                    # BFS 临时算
                    back_path = _bfs_path((acol, arow), next_car, grid, occ)
                    if back_path is None:
                        continue
                    walk_back = len(back_path)
                    base = walk_in_baseline = wd.get(next_car, walk_in)
                    extra_walk = walk_in + walk_back - base
                else:
                    extra_walk = walk_in

                # 旋转成本估算 (4 平均, 用 2)
                rot_cost = 2

                cost = max(0, extra_walk) + rot_cost
                if cost < best_cost:
                    best_cost = cost
                    best = Insertion(
                        insert_before_push_k=k,
                        ident=task,
                        obs_pos=(acol, arow),
                        face_angle=face_q,
                        extra_cost=cost,
                    )

        if best is None:
            if verbose:
                print(f"  ⚠️ {task.etype}_{task.eidx} 无可插入点")
            return None
        insertions.setdefault(best.insert_before_push_k, []).append(best)
        if verbose:
            print(f"  insert {task.etype}_{task.eidx} → before push {best.insert_before_push_k} "
                  f"@ {best.obs_pos}, extra={best.extra_cost}")

    # ── 6. 渲染交织事件序列 → 低层动作 ──────────────────
    actions: List[int] = [6]   # 初始 snap

    cur_state = solver.initial
    cur_angle = state0.car_angle

    for k in range(M + 1):
        # 6.1 当前 checkpoint 有 ident insertion → 按 nearest-neighbor 顺序执行
        ins_list = list(insertions.get(k, []))
        while ins_list:
            grid = solver._get_grid(cur_state.destroyed)
            occ = solver._get_occupied(cur_state)
            walk_dist = _bfs_dist(cur_state.car, grid, occ)
            # 选下一个最近的 ident
            best_idx = -1
            best_d = float('inf')
            for i, ins in enumerate(ins_list):
                d = walk_dist.get(ins.obs_pos)
                if d is not None and d < best_d:
                    best_d = d
                    best_idx = i
            if best_idx < 0:
                if verbose:
                    print(f"  ⚠️ cp{k}: 剩余 ident 都走不到")
                return None
            ins = ins_list.pop(best_idx)

            walk = _bfs_path(cur_state.car, ins.obs_pos, grid, occ)
            if walk is None:
                return None
            for dx, dy in walk:
                actions.append(direction_to_abs_action(dx, dy))

            # 旋转面对 entity
            face_acts = compute_facing_actions(cur_angle, ins.face_angle)
            actions.extend(face_acts)

            # 更新当前姿态
            cur_state = MBState(
                car=ins.obs_pos,
                boxes=cur_state.boxes,
                bombs=cur_state.bombs,
                destroyed=cur_state.destroyed,
                norm_car=ins.obs_pos,
            )
            cur_angle = ins.face_angle

        # 6.2 执行 push k (若有)
        if k < M:
            etype, eid, direction, _ = push_plan[k]
            grid = solver._get_grid(cur_state.destroyed)
            occ = solver._get_occupied(cur_state)
            if etype == 'box':
                old_pos = eid[0]
            else:
                old_pos = eid
            push_from = (old_pos[0] - direction[0],
                         old_pos[1] - direction[1])

            walk = _bfs_path(cur_state.car, push_from, grid, occ)
            if walk is None:
                if verbose:
                    print(f"  ⚠️ push {k}: 走不到推位 {push_from}")
                return None
            for dx, dy in walk:
                actions.append(direction_to_abs_action(dx, dy))

            # 推一步
            actions.append(direction_to_abs_action(*direction))

            # 更新状态
            new_state = solver._apply_push(cur_state, etype, eid, direction)
            if new_state is None:
                return None
            cur_state = new_state
            # cur_angle 不更新 (推送本身不旋转)

    return actions, len(actions)


def plan_best(engine, time_limit: float = 60.0, verbose: bool = False
              ) -> Optional[Tuple[List[int], int]]:
    """同时跑 sequential (plan_exploration + solver) 和 interleaved, 选短的.

    交织算法是 greedy heuristic, 偶尔会比单纯 sequential 长. 此 wrapper 取
    保底.
    """
    from smartcar_sokoban.preview_failed import solve_exact   # 延迟导入避免循环
    import random

    state0 = engine.get_state()
    # 我们要在两次 reset 之间保持同样的 seed 决定的 ID, 所以通过克隆引擎.
    # 这里直接调用; 假设外部已 seed + reset.

    # interleaved
    interleaved = plan_interleaved(engine, time_limit=time_limit, verbose=verbose)

    # sequential 之前要把 engine 重新 reset 到 state0 (因为 plan_interleaved 不改 engine state)
    # 实际上 plan_interleaved 也没改, 所以 engine 还在 state0. solve_exact 会从当前 state0 开始.
    sequential = solve_exact(engine)

    cands = []
    if interleaved is not None:
        cands.append(("interleaved", interleaved[0], interleaved[1]))
    if sequential is not None:
        cands.append(("sequential", sequential[0], sequential[1]))
    if not cands:
        return None

    cands.sort(key=lambda c: c[2])
    if verbose:
        for n, _, c in cands:
            print(f"  {n}: {c} 步")
    return (cands[0][1], cands[0][2])
