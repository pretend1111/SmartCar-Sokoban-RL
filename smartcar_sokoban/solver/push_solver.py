"""BFS 推箱子求解器 — 求解将单个箱子推到目标位置的最短动作序列."""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Set, Tuple

# 4 方向
DIRS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def bfs_push(car_start: Tuple[int, int],
             box_start: Tuple[int, int],
             box_goal: Tuple[int, int],
             grid,
             obstacles: Optional[Set[Tuple[int, int]]] = None
             ) -> Optional[List[Tuple[int, int]]]:
    """BFS 求解将一个箱子从 box_start 推到 box_goal。

    状态空间: (car_col, car_row, box_col, box_row)
    动作: 4 方向车移动；若车移入箱子位置则箱子同方向推动。

    Args:
        car_start: 车初始位置 (col, row)
        box_start: 箱子初始位置 (col, row)
        box_goal:  箱子目标位置 (col, row)
        grid:      二维网格, 1=墙 0=空
        obstacles: 其他不可通行实体（其他箱子/炸弹）坐标集

    Returns:
        方向列表 [(dx, dy), ...] 或 None（无解）
    """
    if obstacles is None:
        obstacles = set()

    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    if box_start == box_goal:
        return []

    def in_bounds(c, r):
        return 0 <= c < cols and 0 <= r < rows

    def cell_free(c, r, current_box):
        """格子是否可通行（排除当前箱子位置，因为箱子会被推走）."""
        if not in_bounds(c, r):
            return False
        if grid[r][c] == 1:
            return False
        if (c, r) in obstacles:
            return False
        return True

    sc, sr = car_start
    bc, br = box_start
    start_state = (sc, sr, bc, br)

    queue: deque = deque([(start_state, [])])
    visited = {start_state}

    while queue:
        (cc, cr, bc, br), path = queue.popleft()

        for dx, dy in DIRS_4:
            ncc, ncr = cc + dx, cr + dy  # 车的新位置

            # 车不能出界或撞墙
            if not in_bounds(ncc, ncr):
                continue
            if grid[ncr][ncc] == 1:
                continue

            if ncc == bc and ncr == br:
                # ── 推箱子 ──
                nbc, nbr = bc + dx, br + dy

                # 箱子新位置必须合法
                if not cell_free(nbc, nbr, (bc, br)):
                    continue

                new_state = (ncc, ncr, nbc, nbr)
                if new_state not in visited:
                    new_path = path + [(dx, dy)]
                    # 检查是否到达目标
                    if (nbc, nbr) == box_goal:
                        return new_path
                    visited.add(new_state)
                    queue.append((new_state, new_path))
            else:
                # ── 车自由移动 ──
                # 不能走到障碍物上
                if (ncc, ncr) in obstacles:
                    continue

                new_state = (ncc, ncr, bc, br)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [(dx, dy)]))

    return None  # 无解


def estimate_push_cost(box_pos: Tuple[int, int],
                       target_pos: Tuple[int, int]) -> int:
    """曼哈顿距离启发估算推箱成本."""
    return abs(box_pos[0] - target_pos[0]) + \
           abs(box_pos[1] - target_pos[1])
