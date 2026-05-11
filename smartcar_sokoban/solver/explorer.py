"""探索策略 — 高效巡游扫描，发现所有箱子/目标的编号信息.

核心优化:
    1. 排除法: N 个箱子只需扫 N-1 个，最后一个的编号通过排除法推出
    2. 就近原则: 每次选最近的未知实体扫描
    3. 远距离扫描: 只要 90° FOV 内无遮挡即可扫描，无需贴近实体
    4. 精准移动: BFS 到最优观察点，最小旋转扫描
    5. 不碰箱子: 箱子/炸弹作为障碍物绕行
    6. 实时中断: 每步都检查途中 FOV 发现，避免走冤枉路
"""

from __future__ import annotations

import math
from typing import List, Optional, Set, Tuple

from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.solver.pathfinder import (
    bfs_path, pos_to_grid, grid_to_pos, DIRS_4, is_walkable,
)

# 车的固定移动朝向（朝上）
ANGLE_UP = -math.pi / 2


# ── 排除法逻辑 ─────────────────────────────────────────────

def exploration_complete(state) -> bool:
    """用排除法判断: 是否已经能确定所有配对关系.

    规则:
    - N 个箱子扫了 N-1 个 → 最后一个编号 = 剩余未出现的编号
    - N 个目标扫了 N-1 个 → 同理
    - 因此只要 unseen_boxes <= 1 AND unseen_targets <= 1 就够了
    """
    n_boxes = len(state.boxes)
    n_targets = len(state.targets)
    unseen_boxes = n_boxes - len(state.seen_box_ids)
    unseen_targets = n_targets - len(state.seen_target_ids)
    return unseen_boxes <= 1 and unseen_targets <= 1


def get_scan_targets(state) -> List[Tuple[float, float, str, int]]:
    """获取需要扫描的实体列表（排除法优化后）.

    对箱子和目标分别:
    - 如果未扫 >= 2: 需要扫 (未扫数 - 1) 个, 跳过最远的那个
    - 如果未扫 == 1: 排除法已知, 不需要扫
    - 如果未扫 == 0: 全部已知

    返回的列表按到车的曼哈顿距离排序（最近优先）。
    """
    car_grid = pos_to_grid(state.car_x, state.car_y)
    required = []

    # ── 箱子 ──
    unseen_box_indices = [i for i in range(len(state.boxes))
                          if i not in state.seen_box_ids]
    if len(unseen_box_indices) >= 2:
        # 计算每个未扫箱子到车的距离
        with_dist = []
        for i in unseen_box_indices:
            box = state.boxes[i]
            bg = pos_to_grid(box.x, box.y)
            dist = abs(car_grid[0] - bg[0]) + abs(car_grid[1] - bg[1])
            with_dist.append((dist, i, box.x, box.y))
        with_dist.sort()  # 最近的在前

        # 跳过最远的那个（排除法推出）
        for dist, i, x, y in with_dist[:-1]:
            required.append((x, y, 'box', i, dist))

    # ── 目标 ──
    unseen_tgt_indices = [i for i in range(len(state.targets))
                          if i not in state.seen_target_ids]
    if len(unseen_tgt_indices) >= 2:
        with_dist = []
        for i in unseen_tgt_indices:
            tgt = state.targets[i]
            tg = pos_to_grid(tgt.x, tgt.y)
            dist = abs(car_grid[0] - tg[0]) + abs(car_grid[1] - tg[1])
            with_dist.append((dist, i, tgt.x, tgt.y))
        with_dist.sort()

        for dist, i, x, y in with_dist[:-1]:
            required.append((x, y, 'target', i, dist))

    # 按距离排序, 最近优先
    required.sort(key=lambda t: t[4])
    return [(x, y, tp, idx) for x, y, tp, idx, _ in required]


# ── 状态查询 ───────────────────────────────────────────────

def get_entity_obstacles(state) -> Set[Tuple[int, int]]:
    """所有实体的网格坐标作为障碍物集合 (用于移动避让)."""
    obs = set()
    for b in state.boxes:
        obs.add(pos_to_grid(b.x, b.y))
    for b in state.bombs:
        obs.add(pos_to_grid(b.x, b.y))
    return obs


def get_all_entity_positions(state) -> Set[Tuple[int, int]]:
    """所有实体的网格坐标 (箱子+炸弹+目标), 用于视线遮挡检测."""
    pos = set()
    for b in state.boxes:
        pos.add(pos_to_grid(b.x, b.y))
    for b in state.bombs:
        pos.add(pos_to_grid(b.x, b.y))
    for t in state.targets:
        pos.add(pos_to_grid(t.x, t.y))
    return pos


# ── 视线检测 ───────────────────────────────────────────────

def has_line_of_sight(from_col: int, from_row: int,
                      to_col: int, to_row: int,
                      grid,
                      entity_positions: Optional[Set[Tuple[int, int]]] = None
                      ) -> bool:
    """检查两点之间是否有视线（不被墙壁或实体遮挡）.

    entity_positions: 可选的实体网格坐标集合（箱子/炸弹/目标），
                      目标自身的位置会自动排除。
    """
    x0, y0 = from_col + 0.5, from_row + 0.5
    x1, y1 = to_col + 0.5, to_row + 0.5

    dx = x1 - x0
    dy = y1 - y0
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.1:
        return True

    steps = int(dist * 4) + 1
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for i in range(1, steps):
        t = i / steps
        px = x0 + dx * t
        py = y0 + dy * t
        col = int(px)
        row = int(py)
        if 0 <= row < rows and 0 <= col < cols:
            if grid[row][col] == 1:
                return False
        # 检查是否被其他实体遮挡 (排除目标自身)
        if entity_positions and (col, row) != (to_col, to_row):
            if (col, row) in entity_positions:
                return False
    return True


# ── 观察点选择 ─────────────────────────────────────────────

def _compute_face_cost(nc: int, nr: int,
                       ec: int, er: int,
                       current_angle: float) -> Tuple[float, int]:
    """计算从 (nc,nr) 面向 (ec,er) 所需的量化角度和旋转步数 (4 方向系统)."""
    angle = math.atan2((er + 0.5) - (nr + 0.5),
                       (ec + 0.5) - (nc + 0.5))
    angle_q = round(angle / (math.pi / 2)) * (math.pi / 2)
    angle_q = math.atan2(math.sin(angle_q), math.cos(angle_q))

    angle_diff = angle_q - current_angle
    angle_diff = math.atan2(math.sin(angle_diff),
                            math.cos(angle_diff))
    rot_steps = abs(round(angle_diff / (math.pi / 2)))
    rot_steps = min(rot_steps, 4 - rot_steps)
    return angle_q, rot_steps


def find_observation_point(car_grid: Tuple[int, int],
                           entity_grid: Tuple[int, int],
                           grid,
                           obstacles: Set[Tuple[int, int]],
                           entity_positions: Optional[Set[Tuple[int, int]]] = None,
                           current_angle: float = ANGLE_UP,
                           ) -> Optional[Tuple[Tuple[int, int], float]]:
    """找到扫描某个实体的最优观察点.

    新规则 (匹配 engine 严格 FOV): 必须 BFS 到 entity 的 8 邻紧贴格子
    (距离 ≤ √2), 然后朝向 entity. "怼一下" 才算识别.

    返回 (obs_pos, face_angle), face_angle 已量化到 π/4 倍数.
    """
    from collections import deque

    ec, er = entity_grid
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    # 4-邻紧贴格子 (上下左右), 必须跟 entity 有视线 (避开墙角遮挡)
    DIRS_VP = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    adjacent: Set[Tuple[int, int]] = set()
    for dc, dr in DIRS_VP:
        nc, nr = ec + dc, er + dr
        if not (0 <= nc < cols and 0 <= nr < rows):
            continue
        if grid[nr][nc] == 1:
            continue
        if (nc, nr) in obstacles:
            continue
        # 视线检查: (nc, nr) 到 entity 之间不能有墙 / 其他实体阻挡
        # (entity 自己的格子会自动排除)
        if not has_line_of_sight(nc, nr, ec, er, grid, entity_positions):
            continue
        adjacent.add((nc, nr))

    if not adjacent:
        return None

    best = None
    best_cost = float('inf')

    # BFS 从车当前位置开始
    visited = {car_grid}
    queue = deque()
    queue.append((car_grid, 0))   # (pos, move_steps)

    while queue:
        (nc, nr), move_steps = queue.popleft()

        # 剪枝
        if move_steps >= best_cost:
            break

        if (nc, nr) in adjacent:
            angle_q, rot_steps = _compute_face_cost(
                nc, nr, ec, er, current_angle)
            total_cost = move_steps + rot_steps
            if total_cost < best_cost:
                best_cost = total_cost
                best = ((nc, nr), angle_q)

        # 扩展 (4 邻 BFS, 因为车走动只能 4 邻 + 离散动作)
        for dx, dy in DIRS_4:
            nx, ny = nc + dx, nr + dy
            if (nx, ny) in visited:
                continue
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if grid[ny][nx] == 1 or (nx, ny) in obstacles:
                continue
            visited.add((nx, ny))
            queue.append(((nx, ny), move_steps + 1))

    return best


# ── 动作转换 ───────────────────────────────────────────────

def direction_to_action(dx: int, dy: int) -> int:
    """世界方向 → 求解器使用的绝对平移动作."""
    return direction_to_abs_action(dx, dy)


def compute_facing_actions(current_angle: float,
                           target_angle: float) -> List[int]:
    """最短旋转路径: current → target (4 方向系统, 每步 ±90°)."""
    def norm(a):
        return math.atan2(math.sin(a), math.cos(a))

    diff = norm(target_angle - current_angle)
    # 4 方向: 每步 π/2. steps ∈ {-2, -1, 0, 1, 2}
    steps = round(diff / (math.pi / 2))

    if steps == 2 or steps == -2:
        return [5, 5]   # 180° = 2 个右转
    elif steps == 1:
        return [5]
    elif steps == -1:
        return [4]
    return []


def restore_angle_actions(current_angle: float) -> List[int]:
    """回到 ANGLE_UP 的旋转动作."""
    return compute_facing_actions(current_angle, ANGLE_UP)


# ── 实时排除检查 ──────────────────────────────────────────

def _entity_scan_still_needed(state, etype: str, eidx: int) -> bool:
    """判断指定实体是否仍需扫描（考虑排除法）.

    排除法: N 个同类实体只需扫 N-1 个.
    当某类型未扫描数 ≤ 1 时, 最后一个的编号可通过排除法推出,
    不需要再专门去扫描了.
    """
    if etype == 'box':
        if eidx in state.seen_box_ids:
            return False  # 已被 FOV 发现
        unseen_count = len(state.boxes) - len(state.seen_box_ids)
        return unseen_count >= 2  # 未扫 ≥ 2 才还需要继续扫
    elif etype == 'target':
        if eidx in state.seen_target_ids:
            return False
        unseen_count = len(state.targets) - len(state.seen_target_ids)
        return unseen_count >= 2
    return False


# ── 主探索流程 ─────────────────────────────────────────────

def plan_exploration(engine) -> List[int]:
    """高效探索: 排除法 + BFS 精准移动 + 最小旋转 + 实时中断.

    策略:
    1. 获取需要扫描的实体列表（排除法：跳过最远的）
    2. 贪心选最近的，BFS 走到观察点
    3. 每走一步都检查: 途中 FOV 是否已发现目标或触发排除法
       → 是: 立即中断当前路径, 重新评估
    4. 仅在观察时最小旋转面向实体
    5. 直到排除法可以确定所有配对
    """
    all_actions: List[int] = []
    state = engine.get_state()

    # 初始 snap
    state = engine.discrete_step(6)
    all_actions.append(6)

    # 报告初始状态
    n = len(state.boxes)
    print(f"  共 {n} 个箱子, {len(state.targets)} 个目标")
    print(f"  排除法: 每类只需扫 {max(0, n-1)} 个")

    max_iterations = 30
    iteration = 0

    while not exploration_complete(state) and iteration < max_iterations:
        iteration += 1

        # 获取还需要扫描的实体（排除法优化后）
        targets = get_scan_targets(state)
        if not targets:
            break

        car_grid = pos_to_grid(state.car_x, state.car_y)
        obstacles = get_entity_obstacles(state)
        entity_pos = get_all_entity_positions(state)

        # 选最近的实体
        best_entity = None
        best_actions_list = None
        best_cost = float('inf')

        for ex, ey, etype, eidx in targets:
            entity_grid = pos_to_grid(ex, ey)
            result = find_observation_point(
                car_grid, entity_grid, state.grid, obstacles,
                entity_pos, current_angle=state.car_angle)
            if result is None:
                continue

            obs_pos, face_angle = result
            path = bfs_path(car_grid, obs_pos, state.grid, obstacles)
            if path is None:
                continue

            # 计算总成本
            face_acts = compute_facing_actions(state.car_angle, face_angle)
            cost = len(path) + len(face_acts)

            if cost < best_cost:
                best_cost = cost
                best_entity = (ex, ey, etype, eidx)
                best_actions_list = (path, face_angle)

        if best_entity is None:
            # 兜底: 360° 扫描, 每步也检查
            for _ in range(8):
                state = engine.discrete_step(5)
                all_actions.append(5)
                if exploration_complete(state):
                    break
            continue

        ex, ey, etype, eidx = best_entity
        path, face_angle = best_actions_list

        # 1) 移动到观察点
        #    每步检查: 当前目标是否已被途中 FOV 发现或排除法不再需要
        #    注意: 不在走路时检查 exploration_complete,
        #    因为中断走路会把车扔在路中间（远离实体的坏位置）,
        #    导致后续求解器搜索空间爆炸. 走完路到观察点后,
        #    外层 while 循环的 exploration_complete 自然会终止.
        for dx, dy in path:
            a = direction_to_action(dx, dy)
            state = engine.discrete_step(a)
            all_actions.append(a)

        # 走完路或提前中断后, 检查是否还需要旋转
        if exploration_complete(state):
            continue
        if not _entity_scan_still_needed(state, etype, eidx):
            continue  # 回到 while 循环重新选目标

        # 2) 旋转面向实体 —— 旋转时也检查
        face_acts = compute_facing_actions(state.car_angle, face_angle)
        for a in face_acts:
            state = engine.discrete_step(a)
            all_actions.append(a)
            if exploration_complete(state):
                break

    # 报告结果
    unseen_b = len(state.boxes) - len(state.seen_box_ids)
    unseen_t = len(state.targets) - len(state.seen_target_ids)
    deduced_b = max(0, unseen_b)
    deduced_t = max(0, unseen_t)
    print(f"  扫描了 {len(state.seen_box_ids)} 箱子 + "
          f"{len(state.seen_target_ids)} 目标 "
          f"(排除推导 {deduced_b} 箱 + {deduced_t} 目标)")

    return all_actions
