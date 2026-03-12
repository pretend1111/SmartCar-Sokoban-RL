"""炸弹规划 — 计算最优爆破位置, 然后将 TNT 当箱子推到目标位置.

核心思路:
    1. 找出被墙阻挡的 box→target 配对 (bfs_push 无解)
    2. 枚举所有内墙, 对每面墙模拟 3×3 爆炸, 验证爆炸后推箱可行
       → 得到「候选爆破墙壁」列表
    3. 对每面候选墙壁, 遍历所有 TNT:
       - 用 bfs_push 把 TNT 当箱子推到墙壁旁的空地
       - 计算车重新定位 + 最后一步推入墙壁触发爆炸
    4. 选总成本最小的方案, 返回 BombTask 列表

关键原则:
    TNT = 箱子, 目标墙壁旁的空地 = 目标位置,
    计算出爆破位置后, 剩下的就是普通推箱子问题。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid, DIRS_4
from smartcar_sokoban.solver.push_solver import bfs_push


# ── 数据结构 ───────────────────────────────────────────────

@dataclass
class BombTask:
    """一次炸弹引爆任务."""
    bomb_idx: int                        # 炸弹在 state.bombs 中的索引
    bomb_grid: Tuple[int, int]           # 炸弹当前位置 (col, row)
    wall_grid: Tuple[int, int]           # 需要爆破的目标墙壁
    tnt_dest: Tuple[int, int]            # TNT 推送目标 (墙壁旁空地)
    approach_dir: Tuple[int, int]        # 最后推入墙壁的方向 (dx, dy)
    target_pairs: List[Tuple[int, int]]  # 此次爆炸能解锁的配对
    priority: float                      # 总成本估算 (越小越优先)


# ── 模拟工具 ──────────────────────────────────────────────

def simulate_explosion(grid, wall_col: int, wall_row: int):
    """模拟 TNT 在接触墙壁 (wall_col, wall_row) 时的 3×3 爆炸.

    返回 grid 的深拷贝（被炸的墙变空地）, 不炸最外圈。
    """
    new_grid = copy.deepcopy(grid)
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            r = wall_row + dy
            c = wall_col + dx
            if r <= 0 or r >= rows - 1:
                continue
            if c <= 0 or c >= cols - 1:
                continue
            if new_grid[r][c] == 1:
                new_grid[r][c] = 0

    return new_grid


def _trace_push(car_start, box_start, directions):
    """模拟推箱方向序列, 返回最终 (car_pos, box_pos)."""
    cx, cy = car_start
    bx, by = box_start
    for dx, dy in directions:
        ncx, ncy = cx + dx, cy + dy
        if (ncx, ncy) == (bx, by):
            bx, by = bx + dx, by + dy
        cx, cy = ncx, ncy
    return (cx, cy), (bx, by)


# ── 核心算法 ──────────────────────────────────────────────

def _find_candidate_walls(grid, car_grid, box_grid, target_grid,
                          obstacles):
    """枚举所有内墙, 找到爆破后能解锁推箱的墙壁.

    Returns:
        [(wall_col, wall_row, push_cost), ...] 按 push_cost 排序
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    candidates = []

    for r in range(1, rows - 1):  # 跳过最外圈 (不可炸)
        for c in range(1, cols - 1):
            if grid[r][c] != 1:
                continue

            # 模拟爆炸
            new_grid = simulate_explosion(grid, c, r)

            # 爆炸后检查推箱是否可行
            result = bfs_push(car_grid, box_grid, target_grid,
                              new_grid, obstacles)
            if result is not None:
                candidates.append((c, r, len(result)))

    candidates.sort(key=lambda x: x[2])
    return candidates


def _find_tnt_route(car_grid, bomb_grid, wall_grid,
                    grid, obstacles):
    """规划将 TNT 推到墙壁旁并引爆的完整路径.

    TNT 当箱子推, 目标 = 墙壁旁的空地,
    最后一步推入墙壁触发爆炸。

    Args:
        car_grid:   车当前位置
        bomb_grid:  TNT 当前位置
        wall_grid:  目标墙壁位置
        grid:       地图网格
        obstacles:  所有障碍物 (含其他炸弹和箱子)

    Returns:
        (approach_dir, tnt_dest, full_directions, cost) 或 None
    """
    wc, wr = wall_grid
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    best = None
    best_cost = float('inf')

    for dx, dy in DIRS_4:
        # TNT 最终要在墙壁的反方向一格 (空地)
        tnt_dest = (wc - dx, wr - dy)
        tc, tr = tnt_dest

        if not (0 <= tr < rows and 0 <= tc < cols):
            continue
        if grid[tr][tc] == 1:
            continue
        # TNT 目标位已被其他实体占据 (除非就是 TNT 当前位置)
        if tnt_dest in obstacles and tnt_dest != bomb_grid:
            continue

        # 最后推 TNT 入墙时, 车要站在 (tc-dx, tr-dy)
        final_push_from = (tc - dx, tr - dy)
        fc, fr = final_push_from
        if not (0 <= fr < rows and 0 <= fc < cols):
            continue
        if grid[fr][fc] == 1:
            continue

        # ── 用 bfs_push 把 TNT 当箱子推到 tnt_dest ──
        push_obs = obstacles.copy()
        push_obs.discard(bomb_grid)  # TNT 本身不是障碍

        if bomb_grid == tnt_dest:
            tnt_dirs = []  # TNT 已经在目标位
        else:
            tnt_dirs = bfs_push(car_grid, bomb_grid, tnt_dest,
                                grid, push_obs)
            if tnt_dirs is None:
                continue

        # 模拟推送后计算车的位置
        if tnt_dirs:
            car_after, _ = _trace_push(car_grid, bomb_grid, tnt_dirs)
        else:
            car_after = car_grid

        # ── 车走到最终推位 ──
        reposition = []
        if car_after != final_push_from:
            # 走位时不能穿过 TNT (它现在在 tnt_dest)
            repo_obs = push_obs.copy()
            repo_obs.add(tnt_dest)
            repo_path = bfs_path(car_after, final_push_from,
                                 grid, repo_obs)
            if repo_path is None:
                continue
            reposition = repo_path

        # 完整路径: 推 TNT 到位 + 车走位 + 最后一推入墙
        full_dirs = tnt_dirs + reposition + [(dx, dy)]
        cost = len(full_dirs)

        if cost < best_cost:
            best_cost = cost
            best = ((dx, dy), tnt_dest, full_dirs, cost)

    return best


# ── 全局分析 ───────────────────────────────────────────────

def analyze_bomb_tasks(state,
                       pairs: List[Tuple[int, int]]
                       ) -> List[BombTask]:
    """分析配对, 找出需要炸弹开路的目标, 计算最优爆破位置.

    算法:
        1. 找被墙阻挡的配对 (bfs_push 无解)
        2. 枚举所有内墙, 模拟爆炸, 找能解锁推箱的候选墙壁
        3. 对每面候选墙壁, 用 bfs_push 把 TNT 当箱子推过去
        4. 选总成本最低的方案

    Returns:
        BombTask 列表, 按优先级排序
    """
    grid = state.grid

    # 收集所有实体作为障碍物
    all_obstacles: Set[Tuple[int, int]] = set()
    for b in state.boxes:
        all_obstacles.add(pos_to_grid(b.x, b.y))
    for b in state.bombs:
        all_obstacles.add(pos_to_grid(b.x, b.y))

    car_grid = pos_to_grid(state.car_x, state.car_y)

    # ── Step 1: 找被墙阻挡的配对 ──
    # 注意: 仅当去掉炸弹障碍后仍无解时才视为"被墙阻挡",
    # 否则说明只是炸弹挡路, 可以通过推炸弹解决, 不需要爆破。
    blocked_pairs: List[Tuple[int, int]] = []

    # 仅含箱子的障碍集 (无炸弹)
    box_only_obstacles: Set[Tuple[int, int]] = set()
    for b in state.boxes:
        box_only_obstacles.add(pos_to_grid(b.x, b.y))

    for bi, ti in pairs:
        box = state.boxes[bi]
        target = state.targets[ti]
        box_grid = pos_to_grid(box.x, box.y)
        target_grid = pos_to_grid(target.x, target.y)

        obs = all_obstacles.copy()
        obs.discard(box_grid)

        if bfs_push(car_grid, box_grid, target_grid, grid, obs) is None:
            # 再试一次: 去掉炸弹障碍
            obs_no_bombs = box_only_obstacles.copy()
            obs_no_bombs.discard(box_grid)
            if bfs_push(car_grid, box_grid, target_grid,
                        grid, obs_no_bombs) is None:
                # 真正被墙阻挡, 需要炸弹开路
                blocked_pairs.append((bi, ti))

    if not blocked_pairs:
        return []

    # ── Step 2+3: 对每个被阻挡的配对, 找最优爆破方案 ──
    # 注: 炸弹规划全程使用 box_only_obstacles (不含炸弹),
    # 因为炸弹可被引擎推开, 不应影响候选墙壁搜索和 TNT 路由。
    tasks: List[BombTask] = []
    used_bombs: Set[int] = set()

    for bi, ti in blocked_pairs:
        box = state.boxes[bi]
        target = state.targets[ti]
        box_grid = pos_to_grid(box.x, box.y)
        target_grid = pos_to_grid(target.x, target.y)

        obs_no_bombs = box_only_obstacles.copy()
        obs_no_bombs.discard(box_grid)

        # 枚举候选墙壁: 炸哪面墙能解锁推箱?
        # 使用不含炸弹的障碍集, 避免炸弹误判导致候选墙壁被排除
        candidate_walls = _find_candidate_walls(
            grid, car_grid, box_grid, target_grid, obs_no_bombs)

        if not candidate_walls:
            continue  # 没有任何单面墙能解决

        # 对每面候选墙壁, 找最近的 TNT 推过去
        best_task = None
        best_cost = float('inf')

        for wc, wr, push_cost in candidate_walls:
            for bomb_idx, bomb in enumerate(state.bombs):
                if bomb_idx in used_bombs:
                    continue
                bomb_grid = pos_to_grid(bomb.x, bomb.y)

                # TNT 路由也不含炸弹障碍 (其他炸弹可被推开)
                route = _find_tnt_route(
                    car_grid, bomb_grid, (wc, wr),
                    grid, box_only_obstacles)
                if route is None:
                    continue

                approach_dir, tnt_dest, full_dirs, tnt_cost = route
                total_cost = tnt_cost + push_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_task = BombTask(
                        bomb_idx=bomb_idx,
                        bomb_grid=bomb_grid,
                        wall_grid=(wc, wr),
                        tnt_dest=tnt_dest,
                        approach_dir=approach_dir,
                        target_pairs=[(bi, ti)],
                        priority=total_cost,
                    )

        if best_task is not None:
            # 检查同一次爆炸是否还能解锁其他被阻挡配对
            wc, wr = best_task.wall_grid
            new_grid = simulate_explosion(grid, wc, wr)

            for bi2, ti2 in blocked_pairs:
                if (bi2, ti2) == (bi, ti):
                    continue
                box2 = state.boxes[bi2]
                target2 = state.targets[ti2]
                box2_grid = pos_to_grid(box2.x, box2.y)
                target2_grid = pos_to_grid(target2.x, target2.y)
                obs2 = box_only_obstacles.copy()
                obs2.discard(box2_grid)
                r2 = bfs_push(car_grid, box2_grid, target2_grid,
                              new_grid, obs2)
                if r2 is not None:
                    best_task.target_pairs.append((bi2, ti2))

            tasks.append(best_task)
            used_bombs.add(best_task.bomb_idx)

    # 去重 & 排序
    seen: Set[int] = set()
    unique: List[BombTask] = []
    for t in tasks:
        if t.bomb_idx not in seen:
            seen.add(t.bomb_idx)
            unique.append(t)
    unique.sort(key=lambda t: t.priority)

    return unique


# ── 执行引爆 ───────────────────────────────────────────────

def plan_bomb_execution(car_pos: Tuple[int, int],
                        task: BombTask,
                        grid,
                        obstacles: Set[Tuple[int, int]]
                        ) -> Optional[List[Tuple[int, int]]]:
    """规划执行一次炸弹引爆的完整路径.

    用 bfs_push 把 TNT 当箱子推到 wall_grid 旁的空地,
    然后车走位, 最后一步把 TNT 推入墙壁引爆。

    Returns:
        [(dx, dy), ...] 方向序列, 或 None
    """
    route = _find_tnt_route(
        car_pos, task.bomb_grid, task.wall_grid,
        grid, obstacles)
    if route is None:
        return None
    _, _, full_dirs, _ = route
    return full_dirs
