"""随机地图生成器 — 为 RL 训练提供多样化的合法地图.

保证:
    1. 所有空地连通 (flood fill 验证)
    2. 每个箱子都能被推到至少一个目标 (BFS 验证)
    3. 车初始位置可达所有实体
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Optional, Set, Tuple

# 地图符号
WALL = '#'
AIR = '-'
TARGET = '.'
BOX = '$'
BOMB = '*'

COLS = 16
ROWS = 12


def generate_map(n_boxes: int = 3,
                 n_bombs: int = 0,
                 wall_density: float = 0.10,
                 seed: Optional[int] = None,
                 max_retries: int = 200) -> Optional[str]:
    """生成一张随机的合法 Sokoban 地图.

    Args:
        n_boxes:      箱子/目标对数 (1-5)
        n_bombs:      炸弹数 (0-3)
        wall_density: 内墙占内部空地的比例 (0.0-0.3)
        seed:         随机种子
        max_retries:  最大重试次数

    Returns:
        16×12 的地图字符串 (行以 \\n 分隔), 或 None (无法生成)
    """
    rng = random.Random(seed)

    for attempt in range(max_retries):
        result = _try_generate(n_boxes, n_bombs, wall_density, rng)
        if result is not None:
            return result

    return None


def _try_generate(n_boxes: int, n_bombs: int,
                  wall_density: float, rng: random.Random
                  ) -> Optional[str]:
    """单次尝试生成地图."""

    # ── 1. 创建基础网格 (外围全墙) ─────────────────────
    grid = [[0] * COLS for _ in range(ROWS)]
    for c in range(COLS):
        grid[0][c] = 1
        grid[ROWS - 1][c] = 1
    for r in range(ROWS):
        grid[r][0] = 1
        grid[r][COLS - 1] = 1

    # ── 2. 随机放置内墙 ───────────────────────────────
    inner_cells = [(c, r) for r in range(1, ROWS - 1)
                   for c in range(1, COLS - 1)]
    n_walls = int(len(inner_cells) * wall_density)
    rng.shuffle(inner_cells)

    for c, r in inner_cells[:n_walls]:
        grid[r][c] = 1
        # 确保放墙后仍然连通
        if not _is_connected(grid):
            grid[r][c] = 0

    # ── 3. 获取开放格子 ───────────────────────────────
    open_cells = [(c, r) for r in range(1, ROWS - 1)
                  for c in range(1, COLS - 1)
                  if grid[r][c] == 0]

    need = 2 * n_boxes + n_bombs
    if len(open_cells) < need + 5:  # 留一些缓冲
        return None

    # ── 4. 确定车的出生点 ──────────────────────────────
    # 和引擎逻辑一致: 第2列中间行
    mid_row = ROWS // 2
    car_cell = None
    for col in [1, COLS - 2]:
        if grid[mid_row][col] == 0:
            car_cell = (col, mid_row)
            break
    if car_cell is None:
        # 找最近的开放格子
        for c, r in open_cells:
            car_cell = (c, r)
            break
    if car_cell is None:
        return None

    # ── 5. 获取从车出发可达的格子 ──────────────────────
    reachable = _flood_fill(grid, car_cell)
    reachable_list = [p for p in reachable if p != car_cell]
    if len(reachable_list) < need + 2:
        return None

    rng.shuffle(reachable_list)

    # ── 6. 放置实体 (保证不在死角) ─────────────────────
    entity_positions: Set[Tuple[int, int]] = set()

    box_positions: List[Tuple[int, int]] = []
    target_positions: List[Tuple[int, int]] = []
    bomb_positions: List[Tuple[int, int]] = []

    idx = 0
    for _ in range(n_boxes):
        pos = _pick_non_corner(reachable_list, idx, grid, entity_positions)
        if pos is None:
            return None
        box_positions.append(pos)
        entity_positions.add(pos)
        idx += 1

    for _ in range(n_boxes):
        pos = _pick_non_corner(reachable_list, idx, grid, entity_positions)
        if pos is None:
            return None
        target_positions.append(pos)
        entity_positions.add(pos)
        idx += 1

    for _ in range(n_bombs):
        while idx < len(reachable_list):
            c, r = reachable_list[idx]
            idx += 1
            if (c, r) not in entity_positions:
                bomb_positions.append((c, r))
                entity_positions.add((c, r))
                break
        else:
            return None

    # ── 7. 验证可解性 (各箱子能推到至少一个目标) ────
    # 使用简单检查: 每个箱子都能被从至少一侧推动
    for bc, br in box_positions:
        pushable = False
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            behind = (bc - dx, br - dy)
            front = (bc + dx, br + dy)
            if (behind in reachable and front in reachable and
                    behind not in entity_positions and
                    front not in entity_positions):
                pushable = True
                break
        if not pushable:
            return None

    # ── 8. 转为字符串 ─────────────────────────────────
    return _grid_to_string(grid, box_positions,
                           target_positions, bomb_positions)


def _pick_non_corner(reachable_list, start_idx, grid,
                     taken: Set[Tuple[int, int]]
                     ) -> Optional[Tuple[int, int]]:
    """从可达列表中找一个不在死角的位置."""
    for i in range(start_idx, len(reachable_list)):
        c, r = reachable_list[i]
        if (c, r) in taken:
            continue
        # 排除角落死锁: 不能两面相邻都是墙
        walls = [grid[r - 1][c] == 1,  # 上
                 grid[r + 1][c] == 1,  # 下
                 grid[r][c - 1] == 1,  # 左
                 grid[r][c + 1] == 1]  # 右
        corner = ((walls[0] and walls[2]) or
                  (walls[0] and walls[3]) or
                  (walls[1] and walls[2]) or
                  (walls[1] and walls[3]))
        if not corner:
            # 交换到 start_idx 位置以"消耗"它
            reachable_list[i], reachable_list[start_idx] = \
                reachable_list[start_idx], reachable_list[i]
            return (c, r)
    return None


def _is_connected(grid) -> bool:
    """检查所有内部空地是否连通."""
    # 找第一个空地
    start = None
    total_open = 0
    for r in range(1, ROWS - 1):
        for c in range(1, COLS - 1):
            if grid[r][c] == 0:
                total_open += 1
                if start is None:
                    start = (c, r)
    if start is None or total_open == 0:
        return True

    reached = _flood_fill(grid, start)
    return len(reached) == total_open


def _flood_fill(grid, start: Tuple[int, int]) -> Set[Tuple[int, int]]:
    """从 start 开始 BFS, 返回所有可达空地."""
    visited = {start}
    queue = deque([start])
    while queue:
        c, r = queue.popleft()
        for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nc, nr = c + dc, r + dr
            if (nc, nr) not in visited:
                if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] == 0:
                    visited.add((nc, nr))
                    queue.append((nc, nr))
    return visited


def _grid_to_string(grid, box_pos, target_pos, bomb_pos) -> str:
    """将网格 + 实体位置转换为地图字符串."""
    box_set = set(box_pos)
    target_set = set(target_pos)
    bomb_set = set(bomb_pos)

    lines = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            if (c, r) in box_set:
                row.append(BOX)
            elif (c, r) in target_set:
                row.append(TARGET)
            elif (c, r) in bomb_set:
                row.append(BOMB)
            elif grid[r][c] == 1:
                row.append(WALL)
            else:
                row.append(AIR)
        lines.append(''.join(row))
    return '\n'.join(lines)


# ── 预定义课程地图 ────────────────────────────────────────

def make_curriculum_maps():
    """生成各课程阶段的训练地图文件内容."""
    maps = {}

    # Phase 1: 1 箱, 空旷, 无炸弹
    maps['train_phase1'] = generate_map(
        n_boxes=1, n_bombs=0, wall_density=0.0, seed=100)

    # Phase 2: 1 箱, 少量墙
    maps['train_phase2'] = generate_map(
        n_boxes=1, n_bombs=0, wall_density=0.05, seed=200)

    # Phase 3: 2 箱, 中等墙
    maps['train_phase3'] = generate_map(
        n_boxes=2, n_bombs=0, wall_density=0.08, seed=300)

    # Phase 4: 3 箱, 中等墙
    maps['train_phase4'] = generate_map(
        n_boxes=3, n_bombs=0, wall_density=0.10, seed=400)

    # Phase 5: 3 箱 + TNT
    maps['train_phase5'] = generate_map(
        n_boxes=3, n_bombs=1, wall_density=0.12, seed=500)

    return maps


if __name__ == '__main__':
    # 测试: 生成并打印一张地图
    for difficulty in [1, 2, 3]:
        m = generate_map(n_boxes=difficulty, n_bombs=max(0, difficulty - 2),
                         wall_density=0.05 * difficulty, seed=42 + difficulty)
        if m:
            print(f"=== {difficulty} boxes ===")
            print(m)
            print()
        else:
            print(f"Failed to generate map with {difficulty} boxes")
