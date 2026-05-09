"""领域特征预计算 — SAGE-PR 状态层的"经典算法"部分.

参考 docs/FINAL_ARCH_DESIGN.md §2.3.

事件触发: 玩家或箱子位置改变就重算. 单次预算 ≤ 4 ms.

所有输出张量都是 full grid 尺寸 [12, 16] (跟 BeliefState.M 对齐).
裁到 playable [10, 14] 在网络输入端做.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from smartcar_sokoban.symbolic.belief import BeliefState, GRID_ROWS, GRID_COLS

INF = np.iinfo(np.int32).max

# 4 邻接 (dc, dr) — east / west / south / north
DIRS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
# 跟 push_dir_field 4 通道顺序: 0=E, 1=W, 2=S, 3=N
DIR_NAMES = ("E", "W", "S", "N")


@dataclass
class DomainFeatures:
    """所有派生特征的容器.

    Shape 规约:
        [rows, cols] = [12, 16] (full grid).
        push_dist_field / push_dir_field 按 box 索引列表存储.
    """
    player_bfs_dist: np.ndarray         # int32 [12,16], 不可达 = INF
    reachable_mask: np.ndarray          # bool  [12,16]
    push_dist_field: List[np.ndarray]   # list of int32 [12,16] per box (匹配 target)
    push_dir_field: np.ndarray          # int8 [12,16,4], 0/1, 4 方向 best step
    deadlock_mask: np.ndarray           # bool [12,16]
    info_gain_heatmap: np.ndarray       # float32 [12,16] in [0,1]
    box_target_match: List[Optional[int]]  # box_i → target_j (按 Pi argmax, None 若不确定)


# ── 玩家 BFS / 可达性 ─────────────────────────────────────

def _build_player_obstacle_grid(bs: BeliefState) -> np.ndarray:
    """墙 + 箱子 + 炸弹都作障碍物 (玩家走不过)."""
    obs = bs.M.astype(bool).copy()
    for b in bs.boxes:
        if 0 <= b.row < GRID_ROWS and 0 <= b.col < GRID_COLS:
            obs[b.row, b.col] = True
    for bm in bs.bombs:
        if 0 <= bm.row < GRID_ROWS and 0 <= bm.col < GRID_COLS:
            obs[bm.row, bm.col] = True
    return obs


def _bfs_dist_full(start_col: int, start_row: int,
                   obstacles: np.ndarray) -> np.ndarray:
    """4 邻接 BFS, 输出每个空格到 start 的最短距离 (步数). 不可达 = INF."""
    rows, cols = obstacles.shape
    dist = np.full((rows, cols), INF, dtype=np.int32)
    if not (0 <= start_row < rows and 0 <= start_col < cols):
        return dist
    if obstacles[start_row, start_col]:
        return dist
    dist[start_row, start_col] = 0
    q = deque()
    q.append((start_col, start_row))
    while q:
        c, r = q.popleft()
        d = dist[r, c]
        for dc, dr in DIRS_4:
            nc, nr = c + dc, r + dr
            if 0 <= nr < rows and 0 <= nc < cols:
                if not obstacles[nr, nc] and dist[nr, nc] == INF:
                    dist[nr, nc] = d + 1
                    q.append((nc, nr))
    return dist


def compute_player_bfs(bs: BeliefState) -> Tuple[np.ndarray, np.ndarray]:
    """车 BFS 距离 + 可达 mask."""
    obs = _build_player_obstacle_grid(bs)
    # 玩家自己的格子是 free (虽然 obs 里其他箱视为障碍, 但玩家自己不视障碍)
    pc, pr = bs.player_col, bs.player_row
    if 0 <= pr < GRID_ROWS and 0 <= pc < GRID_COLS:
        obs[pr, pc] = False  # 起点必须可走
    dist = _bfs_dist_full(pc, pr, obs)
    reachable = dist != INF
    return dist, reachable


# ── 推送距离场 (reverse-push BFS) ────────────────────────

def _reverse_push_bfs(target_col: int, target_row: int,
                      walls: np.ndarray) -> np.ndarray:
    """从目标格反向推 BFS — 不考虑车位置约束 (启发式).

    每一步: 从当前格 (c, r), 推送方向 d, 反推到 (c-d).
    要求:
      - (c, r) 当前是空格 (因为是箱可在的位置)
      - (c-d) 也是空格 (车反推前所在的位置)
      - (c-2d) 不需要 (反向 BFS 只关心 box 落点)

    简化: 只看墙 (无视箱/炸弹), 因为推送距离是"如果场上只有这一个箱"的下界.
    """
    rows, cols = walls.shape
    dist = np.full((rows, cols), INF, dtype=np.int32)
    if not (0 <= target_row < rows and 0 <= target_col < cols):
        return dist
    if walls[target_row, target_col]:
        return dist
    dist[target_row, target_col] = 0
    q = deque()
    q.append((target_col, target_row))
    while q:
        c, r = q.popleft()
        d = dist[r, c]
        for dc, dr in DIRS_4:
            # 反推: 箱原本在 (c-dc, r-dr), 沿 (dc,dr) 推一步到 (c, r).
            # 要求: (c-dc, r-dr) 是空格 (箱原位); (c, r) 是空格 (推后落点, 已保证); (c-2dc, r-2dr) 是空格 (车原位)
            box_prev_c = c - dc
            box_prev_r = r - dr
            car_prev_c = c - 2 * dc
            car_prev_r = r - 2 * dr
            # 边界 + 墙检查
            if not (0 <= box_prev_r < rows and 0 <= box_prev_c < cols):
                continue
            if walls[box_prev_r, box_prev_c]:
                continue
            if not (0 <= car_prev_r < rows and 0 <= car_prev_c < cols):
                continue
            if walls[car_prev_r, car_prev_c]:
                continue
            if dist[box_prev_r, box_prev_c] == INF:
                dist[box_prev_r, box_prev_c] = d + 1
                q.append((box_prev_c, box_prev_r))
    return dist


def compute_box_target_match(bs: BeliefState) -> List[Optional[int]]:
    """从 Π 矩阵给每个 box 选一个最可能 target.

    若有唯一兼容 target → 返回该 idx; 多个 → argmax (这里因为是 0/1, 选第一个);
    无兼容 → None.
    """
    Pi = bs.Pi
    matches: List[Optional[int]] = []
    for i in range(len(bs.boxes)):
        compat = np.where(Pi[i] > 0.5)[0]
        if len(compat) == 0:
            matches.append(None)
        else:
            matches.append(int(compat[0]))
    return matches


def compute_push_dist_fields(bs: BeliefState,
                             matches: List[Optional[int]]) -> List[np.ndarray]:
    """每个 box 到其 (假定) 目标的反向推 BFS 距离场."""
    walls = bs.M.astype(bool)
    fields: List[np.ndarray] = []
    for i, b in enumerate(bs.boxes):
        tgt_idx = matches[i]
        if tgt_idx is None or tgt_idx >= len(bs.targets):
            fields.append(np.full((GRID_ROWS, GRID_COLS), INF, dtype=np.int32))
            continue
        t = bs.targets[tgt_idx]
        fields.append(_reverse_push_bfs(t.col, t.row, walls))
    return fields


def compute_push_dir_field(push_dist_fields: List[np.ndarray]) -> np.ndarray:
    """流场 [12, 16, 4]: 每格 4 方向 one-hot, 推哪个方向距离最小.

    取所有 box 的距离场聚合 (min): 表示"任意 box 在这里, 推哪个方向最划算".
    实际上 SAGE-PR 用每个候选自己计算, 这里只生成全局聚合作为输入通道.
    """
    if not push_dist_fields:
        return np.zeros((GRID_ROWS, GRID_COLS, 4), dtype=np.int8)
    # 聚合距离场: min over boxes
    stack = np.stack(push_dist_fields, axis=0)
    agg = stack.min(axis=0)
    out = np.zeros((GRID_ROWS, GRID_COLS, 4), dtype=np.int8)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if agg[r, c] == INF or agg[r, c] == 0:
                continue
            best_d = INF
            best_k = -1
            for k, (dc, dr) in enumerate(DIRS_4):
                nc, nr = c + dc, r + dr
                if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                    nd = agg[nr, nc]
                    if nd < best_d:
                        best_d = nd
                        best_k = k
            if best_k >= 0 and best_d < agg[r, c]:
                out[r, c, best_k] = 1
    return out


# ── 死锁检测 (静态) ────────────────────────────────────────

def compute_destructible_walls(bs: BeliefState) -> Set[Tuple[int, int]]:
    """估算可被炸弹炸毁的墙集合.

    上限近似: 把每个炸弹通过 4-邻 BFS 能到达的所有格作为潜在引爆点.
        对每个引爆点 (c, r), 它若被推入相邻墙格则在 (c, r) 处引爆, 3×3 范围内的墙都被清除.
        所以可被炸毁的墙 = ⋃ {3×3 around (c, r) | (c, r) 是潜在引爆点 且 它有相邻墙}.

    注: 这只是 admissible 上限 (会过度乐观), 但对 deadlock 检测来说更宽松 = 更安全
        (避免假死锁误标 candidate 非法). 真正不可解的 deadlock 仍由 solver 自己排除.
    """
    walls = bs.M.astype(bool)
    rows, cols = walls.shape
    destructible: Set[Tuple[int, int]] = set()

    if not bs.bombs:
        return destructible

    # 障碍 = 墙 + 箱 (其他炸弹也算, 但简化先不区分; 反正 BFS 会越过)
    box_obstacles = {(b.col, b.row) for b in bs.boxes}
    bomb_positions = {(bm.col, bm.row) for bm in bs.bombs}

    for bomb in bs.bombs:
        # BFS 从炸弹位置, 4-邻可达 (墙/箱阻挡, 其他炸弹也阻挡)
        start = (bomb.col, bomb.row)
        visited: Set[Tuple[int, int]] = {start}
        q = deque([start])
        other_bombs = bomb_positions - {start}
        while q:
            c, r = q.popleft()
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc, nr = c + dc, r + dr
                if (nc, nr) in visited:
                    continue
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if walls[nr, nc]:
                    continue
                if (nc, nr) in box_obstacles or (nc, nr) in other_bombs:
                    continue
                visited.add((nc, nr))
                q.append((nc, nr))

        # 对每个可达格 (c, r), 若它有相邻墙, 它就是潜在引爆点
        for c, r in visited:
            has_adj_wall = False
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                wc, wr = c + dc, r + dr
                if 0 <= wr < rows and 0 <= wc < cols and walls[wr, wc]:
                    has_adj_wall = True
                    break
            if not has_adj_wall:
                continue
            # 3×3 周围所有墙格都可被炸毁
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    ec, er = c + dx, r + dy
                    if 0 <= er < rows and 0 <= ec < cols and walls[er, ec]:
                        destructible.add((ec, er))

    return destructible


def compute_deadlock_mask(bs: BeliefState) -> np.ndarray:
    """静态死锁: 角落 + 边缘线 (考虑炸弹炸墙).

    Corner deadlock: 格 (r, c) 是空格, 但 (r±1, c) 至少一个是墙 + (r, c±1) 至少
        一个是墙, 且 (r, c) 不是任何 target.

    若导致 corner 的墙能被炸弹炸毁 (compute_destructible_walls), 则不算 deadlock.

    注: 这是箱子推到该格后无法推出的格. 对玩家无影响, 仅用作候选合法性 mask.
    """
    walls = bs.M.astype(bool)
    deadlock = np.zeros_like(walls, dtype=bool)

    # target 集合
    target_cells = {(t.col, t.row) for t in bs.targets}
    destructible = compute_destructible_walls(bs)

    def is_solid_wall(c: int, r: int) -> bool:
        """是真墙 (不可被炸毁)."""
        if not (0 <= r < GRID_ROWS and 0 <= c < GRID_COLS):
            return True   # 越界 = 实墙
        return walls[r, c] and (c, r) not in destructible

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if walls[r, c]:
                continue
            if (c, r) in target_cells:
                continue

            # 检查 4 个 corner 配置 (用 effective wall: 不含可炸墙)
            wall_n = is_solid_wall(c, r - 1)
            wall_s = is_solid_wall(c, r + 1)
            wall_e = is_solid_wall(c + 1, r)
            wall_w = is_solid_wall(c - 1, r)

            corner_ne = wall_n and wall_e
            corner_nw = wall_n and wall_w
            corner_se = wall_s and wall_e
            corner_sw = wall_s and wall_w

            if corner_ne or corner_nw or corner_se or corner_sw:
                deadlock[r, c] = True

    return deadlock


# ── 信息增益热度图 (粗粒度近似) ──────────────────────────

def compute_info_gain_heatmap(bs: BeliefState,
                              reachable_mask: np.ndarray) -> np.ndarray:
    """每格作为观测点的预期信息增益 (粗粒度).

    定义: 从该格做 4 方向射线, 每条命中一个 unidentified entity 就 +1.
    阻挡: 墙 / 箱子.
    """
    n_unid = bs.n_unidentified_boxes + bs.n_unidentified_targets
    out = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    if n_unid == 0:
        return out

    walls = bs.M.astype(bool)

    # entity → (col, row) 集合 (按 unidentified)
    unid_cells: Set[Tuple[int, int]] = set()
    for b in bs.boxes:
        if b.class_id is None:
            unid_cells.add((b.col, b.row))
    for t in bs.targets:
        if t.num_id is None:
            unid_cells.add((t.col, t.row))

    if not unid_cells:
        return out

    # 箱炸弹位置 (作为遮挡, 但不算"未识别 entity 自身")
    blockers: Set[Tuple[int, int]] = set()
    for b in bs.boxes:
        blockers.add((b.col, b.row))
    for bm in bs.bombs:
        blockers.add((bm.col, bm.row))

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if walls[r, c] or not reachable_mask[r, c]:
                continue
            count = 0
            for dc, dr in DIRS_4:
                # 沿 (dc, dr) 射线最多 14 格, 命中第一个 entity 或墙
                tc, tr = c + dc, r + dr
                steps = 0
                while 0 <= tr < GRID_ROWS and 0 <= tc < GRID_COLS and steps < 14:
                    if walls[tr, tc]:
                        break
                    if (tc, tr) in unid_cells:
                        count += 1
                        break
                    if (tc, tr) in blockers:
                        # 遮挡 (但不是未识别)
                        break
                    tc += dc
                    tr += dr
                    steps += 1
            out[r, c] = count / max(1, n_unid)
    return out


# ── 主入口 ────────────────────────────────────────────────

def compute_domain_features(bs: BeliefState) -> DomainFeatures:
    """计算所有派生特征."""
    pdist, reach = compute_player_bfs(bs)
    matches = compute_box_target_match(bs)
    push_fields = compute_push_dist_fields(bs, matches)
    push_dir = compute_push_dir_field(push_fields)
    deadlock = compute_deadlock_mask(bs)
    ig = compute_info_gain_heatmap(bs, reach)

    return DomainFeatures(
        player_bfs_dist=pdist,
        reachable_mask=reach,
        push_dist_field=push_fields,
        push_dir_field=push_dir,
        deadlock_mask=deadlock,
        info_gain_heatmap=ig,
        box_target_match=matches,
    )


# ── 归一化辅助 ────────────────────────────────────────────

def normalize_dist_field(dist: np.ndarray, scale: float = 30.0) -> np.ndarray:
    """INF → 1.0, 否则 tanh(d / scale).

    输出 float32 ∈ [0, 1].
    """
    out = np.zeros_like(dist, dtype=np.float32)
    finite = dist != INF
    out[finite] = np.tanh(dist[finite].astype(np.float32) / scale)
    out[~finite] = 1.0
    return out
