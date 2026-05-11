"""plan_exploration_v3 — V2 + 拓扑预先配对.

新策略:
  策略1: 检测 "wall-locked + 唯一可达 target" 的箱子, 拓扑上直接配对,
         从 scan 集合中剔除 → 探索目标变少.
  策略2: 若当前 scan target X 不可达, 不放弃, 试下一个可达的 Y; X 用排除法兜底.

注: 这版假设箱子的"可达 target"由当前墙布局决定 (忽略 bomb 炸墙的潜在扩展).
    保守做法: 仅在该箱子的锁定墙均为 wall_init=不可炸 (本游戏内墙都可炸, 但外圈墙不可)
    时才认 wall-locked. 外圈是固定墙, 4 邻里 2 个相邻墙都是外圈 → 一定永久锁.
"""

from __future__ import annotations

import contextlib
import io
from typing import Dict, List, Optional, Set, Tuple

from smartcar_sokoban.solver.explorer import (
    plan_exploration, exploration_complete,
)
from smartcar_sokoban.solver.explorer_v2 import plan_exploration_v2
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid


DIRS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _is_wall(grid, c: int, r: int) -> bool:
    rows = len(grid); cols = len(grid[0]) if rows else 0
    if not (0 <= r < rows and 0 <= c < cols):
        return True   # 越界=墙
    return grid[r][c] == 1


def _box_axis_lock(grid, c: int, r: int) -> Optional[str]:
    """返回 'horizontal'(只能左右动) / 'vertical'(只能上下动) / None (自由)."""
    wn = _is_wall(grid, c, r - 1)
    ws = _is_wall(grid, c, r + 1)
    we = _is_wall(grid, c + 1, r)
    ww = _is_wall(grid, c - 1, r)
    h_locked = wn and ws    # 上下都墙, 只能水平
    v_locked = we and ww    # 左右都墙, 只能垂直
    if h_locked and not v_locked:
        return "horizontal"
    if v_locked and not h_locked:
        return "vertical"
    if h_locked and v_locked:
        return "fully_stuck"   # 4 边都墙, 不可推
    return None


def _push_reachable_along_axis(grid, start: Tuple[int, int], axis: str,
                                 obstacles: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """从 start 沿轴方向 (horizontal=col 方向 / vertical=row 方向) 找 push 可达格.

    保守模型: 无 bomb 炸墙. 障碍 = 墙 + 其他 entity.
    箱子从 (c, r) 推到 (c+k, r) 需要 (c+1..c+k, r) 全部非障碍.
    """
    visited = {start}
    if axis == "horizontal":
        dirs = [(1, 0), (-1, 0)]
    else:
        dirs = [(0, 1), (0, -1)]
    queue = [start]
    while queue:
        c, r = queue.pop()
        for dc, dr in dirs:
            nc, nr = c + dc, r + dr
            if (nc, nr) in visited: continue
            if _is_wall(grid, nc, nr): continue
            if (nc, nr) in obstacles: continue
            visited.add((nc, nr))
            queue.append((nc, nr))
    return visited


def find_forced_pairs(state) -> List[Tuple[int, int]]:
    """找拓扑强制配对 (box_idx, target_idx).

    条件:
      1. box 被外圈/内墙在 2 个垂直方向死锁 (axis-locked)
      2. 沿该轴 push 可达的 target 恰好 1 个
    """
    grid = state.grid
    # 障碍: 其他 box / bomb (不含 target — target 是地面)
    other_obstacles = set()
    box_pos_list = []
    for i, b in enumerate(state.boxes):
        p = pos_to_grid(b.x, b.y)
        box_pos_list.append((i, p))
        other_obstacles.add(p)
    for bm in state.bombs:
        other_obstacles.add(pos_to_grid(bm.x, bm.y))

    target_pos = {i: pos_to_grid(t.x, t.y) for i, t in enumerate(state.targets)}

    pairs: List[Tuple[int, int]] = []
    used_targets: Set[int] = set()

    for i, bp in box_pos_list:
        axis = _box_axis_lock(grid, *bp)
        if axis not in ("horizontal", "vertical"):
            continue
        # 沿轴可达 (排除其他 box / bomb)
        obs_no_self = other_obstacles - {bp}
        reach = _push_reachable_along_axis(grid, bp, axis, obs_no_self)
        # 看哪些 target 在 reach 集合里
        reachable_tgts = [j for j, tp in target_pos.items()
                          if tp in reach and j not in used_targets]
        if len(reachable_tgts) == 1:
            j = reachable_tgts[0]
            pairs.append((i, j))
            used_targets.add(j)
    return pairs


def plan_exploration_v3(engine, max_retries: int = 15,
                         verbose: bool = False) -> List[int]:
    """V2 + 拓扑强制配对预处理.

    步骤:
      1. 计算 forced_pairs (wall-locked + 唯一可达 target)
      2. 验证 engine 真实 class_id == num_id (否则 pair 错, 跳过)
      3. 把 forced_pair 的 box / target mark seen (BeliefState 会暴露真实 ID)
      4. 跑 V2 探索 (含推开障碍, scan_targets 自动不含 forced)
    """
    state = engine.get_state()
    candidate_pairs = find_forced_pairs(state)
    # 验证 engine 实际 ID 匹配 (理论上 wall-locked + unique reachable 必匹配,
    # 但兜底防御)
    forced = []
    for i, j in candidate_pairs:
        if i >= len(state.boxes) or j >= len(state.targets):
            continue
        if state.boxes[i].class_id == state.targets[j].num_id:
            forced.append((i, j))

    if not forced:
        return plan_exploration_v2(engine, max_retries=max_retries, verbose=verbose)

    s = engine.state
    forced_box = {i for i, _ in forced}
    forced_tgt = {j for _, j in forced}
    # mark seen — class_id / num_id 真实值已在 state.boxes/targets 里, 这步只是
    # 把它们暴露给 BeliefState
    s.seen_box_ids.update(forced_box)
    s.seen_target_ids.update(forced_tgt)

    if verbose:
        print(f"  [v3] forced pairs: {forced} (skip scanning {len(forced)} boxes & targets)")

    return plan_exploration_v2(engine, max_retries=max_retries, verbose=verbose)
