"""纯路径规划 — 车在网格上的 BFS 移动（不推箱子）."""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Set, Tuple

# 4 个基本方向: (dx, dy) — 右/左/下/上
DIRS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


# ── 坐标转换 ───────────────────────────────────────────────

def pos_to_grid(x: float, y: float) -> Tuple[int, int]:
    """引擎浮点位置 → 网格坐标 (col, row)."""
    return int(round(x - 0.5)), int(round(y - 0.5))


def grid_to_pos(col: int, row: int) -> Tuple[float, float]:
    """网格坐标 → 引擎浮点位置."""
    return col + 0.5, row + 0.5


# ── 通行检测 ───────────────────────────────────────────────

def is_walkable(col: int, row: int, grid,
                obstacles: Set[Tuple[int, int]]) -> bool:
    """格子是否可通行（不是墙、不是障碍物、不越界）."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return False
    if grid[row][col] == 1:
        return False
    if (col, row) in obstacles:
        return False
    return True


# ── BFS 最短路径 ───────────────────────────────────────────

def bfs_path(start: Tuple[int, int],
             goal: Tuple[int, int],
             grid,
             obstacles: Optional[Set[Tuple[int, int]]] = None
             ) -> Optional[List[Tuple[int, int]]]:
    """BFS 求最短路径（4 方向）。

    Args:
        start: (col, row) 起点
        goal:  (col, row) 终点
        grid:  二维数组, 1=墙 0=空
        obstacles: 额外障碍物坐标集合

    Returns:
        方向列表 [(dx, dy), ...] 或 None（不可达）
    """
    if obstacles is None:
        obstacles = set()

    if start == goal:
        return []

    queue: deque = deque()
    queue.append((start[0], start[1], []))
    visited = {start}

    while queue:
        col, row, path = queue.popleft()
        for dx, dy in DIRS_4:
            nc, nr = col + dx, row + dy
            if (nc, nr) == goal:
                return path + [(dx, dy)]
            if (nc, nr) not in visited and \
               is_walkable(nc, nr, grid, obstacles):
                visited.add((nc, nr))
                queue.append((nc, nr, path + [(dx, dy)]))

    return None


# ── 可达区域 ───────────────────────────────────────────────

def get_reachable(start: Tuple[int, int], grid,
                  obstacles: Optional[Set[Tuple[int, int]]] = None
                  ) -> Set[Tuple[int, int]]:
    """BFS 获取从 start 可达的所有空地格子."""
    if obstacles is None:
        obstacles = set()

    queue: deque = deque([start])
    visited = {start}

    while queue:
        col, row = queue.popleft()
        for dx, dy in DIRS_4:
            nc, nr = col + dx, row + dy
            if (nc, nr) not in visited and \
               is_walkable(nc, nr, grid, obstacles):
                visited.add((nc, nr))
                queue.append((nc, nr))

    return visited
