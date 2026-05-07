"""候选动作生成器 (符号) — SAGE-PR §3.

每个候选 = 一条合法 macro action.

类型:
    push_box(box_idx, dir, run_length)
    push_bomb(bomb_idx, dir)              # dir 可能正交或对角
    inspect(viewpoint, heading)
    return_garage                         # 终局, 暂不实现 (留 padding)
    pad                                    # padding (合法 mask=0)

输出结构:
    [Candidate, ...]  长度 ≤ 64.
    生成器自动 padding 到 64 (`pad` 类型, mask=0).

合法性 mask 由 `Candidate.legal=True` 标记; 非法直接 `legal=False`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from smartcar_sokoban.symbolic.belief import (
    BeliefState, GRID_ROWS, GRID_COLS,
)
from smartcar_sokoban.symbolic.features import (
    DIRS_4, INF, DomainFeatures,
    compute_domain_features,
)


MAX_CANDIDATES = 64
MAX_BOXES = 5
MAX_BOMBS = 3
MAX_INSPECT = 8
MAX_PUSH_RUN = 3   # macro 上限 (k=1, 2, 3)


# ── 候选数据类 ────────────────────────────────────────────

@dataclass
class Candidate:
    type: str = "pad"
    legal: bool = False

    # 推箱
    box_idx: Optional[int] = None

    # 炸弹
    bomb_idx: Optional[int] = None

    # 推方向 (col_delta, row_delta), 8 维 (含对角)
    direction: Optional[Tuple[int, int]] = None
    is_diagonal: bool = False

    # macro 推送步数
    run_length: int = 1

    # 探索观察
    viewpoint_col: Optional[int] = None
    viewpoint_row: Optional[int] = None
    inspect_heading: Optional[int] = None  # 0..7 朝向

    # 调试用
    note: str = ""


# ── 辅助 ─────────────────────────────────────────────────

def _in_playable(col: int, row: int) -> bool:
    """是否在 14×10 playable 区域内 (col ∈ [1,14], row ∈ [1,10])."""
    return 1 <= col <= GRID_COLS - 2 and 1 <= row <= GRID_ROWS - 2


def _build_obstacle_set(bs: BeliefState,
                        exclude_box_idx: Optional[int] = None,
                        exclude_bomb_idx: Optional[int] = None) -> Set[Tuple[int, int]]:
    """墙 + 箱 + 炸弹 (排除指定的). 返回 (col, row) 集合.
    注: 墙的 wall=1 的格子不会进 obstacle (BFS 自己处理), 这里只放可移动实体.
    """
    obs: Set[Tuple[int, int]] = set()
    for i, b in enumerate(bs.boxes):
        if i != exclude_box_idx:
            obs.add((b.col, b.row))
    for k, bm in enumerate(bs.bombs):
        if k != exclude_bomb_idx:
            obs.add((bm.col, bm.row))
    return obs


def _is_free(col: int, row: int, walls: np.ndarray,
             obstacles: Set[Tuple[int, int]]) -> bool:
    """格子可通行 (不是墙, 不是其他实体, 在 bound 内)."""
    if not (0 <= row < GRID_ROWS and 0 <= col < GRID_COLS):
        return False
    if walls[row, col]:
        return False
    if (col, row) in obstacles:
        return False
    return True


def _bfs_from_player(bs: BeliefState, walls: np.ndarray,
                     obstacles: Set[Tuple[int, int]]) -> np.ndarray:
    """玩家可达性 BFS, 输出 dist int32 [12,16]. 不可达 = INF."""
    dist = np.full((GRID_ROWS, GRID_COLS), INF, dtype=np.int32)
    pc, pr = bs.player_col, bs.player_row
    if not _in_playable(pc, pr) and not (0 <= pr < GRID_ROWS and 0 <= pc < GRID_COLS):
        return dist
    if walls[pr, pc]:
        return dist
    dist[pr, pc] = 0
    q = deque()
    q.append((pc, pr))
    while q:
        c, r = q.popleft()
        d = dist[r, c]
        for dc, dr in DIRS_4:
            nc, nr = c + dc, r + dr
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                if walls[nr, nc] or (nc, nr) in obstacles:
                    continue
                if dist[nr, nc] == INF:
                    dist[nr, nc] = d + 1
                    q.append((nc, nr))
    return dist


# ── 推箱候选 ──────────────────────────────────────────────

def _gen_push_box_candidates(bs: BeliefState,
                             feat: DomainFeatures) -> List[Candidate]:
    """枚举每箱 × 4 方向 (含 1-3 步 macro)."""
    out: List[Candidate] = []
    walls = bs.M.astype(bool)

    for i, b in enumerate(bs.boxes):
        if i >= MAX_BOXES:
            break
        # 排除自己作为障碍 (车要在 anti-side, 箱要被推走)
        # 推方向 (col_delta=dc, row_delta=dr)
        for dc, dr in DIRS_4:
            cand = Candidate(
                type="push_box",
                box_idx=i,
                direction=(dc, dr),
                is_diagonal=False,
                run_length=1,
                legal=False,
            )

            # 推位 (车需要站的位置) = 箱反方向一格
            push_pos_col = b.col - dc
            push_pos_row = b.row - dr

            # 推后箱位
            box_next_col = b.col + dc
            box_next_row = b.row + dr

            obstacles = _build_obstacle_set(bs, exclude_box_idx=i)

            # 1) 推位必须可达
            #    把推位作为 obstacle 临时去掉 (车要站这儿) — 但实际 obstacle set 不含
            #    箱 i, 所以推位可能就是 free.
            #    BFS 已经过滤了墙和其他实体, 所以只要 dist < INF 就行.
            #    注意: 若推位本身是个 box/bomb (除了被排除的), 那不行.
            if not _is_free(push_pos_col, push_pos_row, walls, obstacles):
                cand.note = "push_pos blocked"
                out.append(cand)
                continue
            # 玩家 BFS 距离 (不算其他箱炸弹的话)
            dist_to_push = feat.player_bfs_dist[push_pos_row, push_pos_col]
            if dist_to_push == INF:
                cand.note = "push_pos unreachable"
                out.append(cand)
                continue

            # 2) 推后箱位必须可推 (不是墙, 不是其他实体)
            if not _is_free(box_next_col, box_next_row, walls, obstacles):
                cand.note = "box_next blocked"
                out.append(cand)
                continue

            # 3) 不能推入死锁 (除非该格是 target 且 ID 兼容)
            target_cells_compat = set()
            for j, t in enumerate(bs.targets):
                # 用 Pi 检查 box i 是否可能配 target j
                if bs.Pi[i, j] > 0.5:
                    target_cells_compat.add((t.col, t.row))
            if feat.deadlock_mask[box_next_row, box_next_col] and \
                    (box_next_col, box_next_row) not in target_cells_compat:
                cand.note = "deadlock after push"
                out.append(cand)
                continue

            # 至此 1-step 合法
            cand.legal = True
            out.append(cand)

            # macro (run_length=2,3)
            cur_box_col, cur_box_row = box_next_col, box_next_row
            cur_push_pos_col, cur_push_pos_row = b.col, b.row  # 车跟着箱走
            for k in range(2, MAX_PUSH_RUN + 1):
                next_box_col = cur_box_col + dc
                next_box_row = cur_box_row + dr
                if not _is_free(next_box_col, next_box_row, walls, obstacles):
                    break
                if feat.deadlock_mask[next_box_row, next_box_col] and \
                        (next_box_col, next_box_row) not in target_cells_compat:
                    break
                # macro 候选合法
                macro = Candidate(
                    type="push_box",
                    box_idx=i,
                    direction=(dc, dr),
                    is_diagonal=False,
                    run_length=k,
                    legal=True,
                )
                out.append(macro)
                cur_box_col, cur_box_row = next_box_col, next_box_row

    return out


# ── 推炸弹候选 ────────────────────────────────────────────

DIRS_8 = (
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)


def _gen_push_bomb_candidates(bs: BeliefState,
                              feat: DomainFeatures) -> List[Candidate]:
    """每炸弹 × 8 方向 (含 4 对角)."""
    out: List[Candidate] = []
    walls = bs.M.astype(bool)

    for k, bm in enumerate(bs.bombs):
        if k >= MAX_BOMBS:
            break
        for dc, dr in DIRS_8:
            is_diag = dc != 0 and dr != 0
            cand = Candidate(
                type="push_bomb",
                bomb_idx=k,
                direction=(dc, dr),
                is_diagonal=is_diag,
                run_length=1,
                legal=False,
            )

            # 推位 = 炸弹反方向一格
            push_pos_col = bm.col - dc
            push_pos_row = bm.row - dr

            obstacles = _build_obstacle_set(bs, exclude_bomb_idx=k)

            if not _is_free(push_pos_col, push_pos_row, walls, obstacles):
                out.append(cand)
                continue
            if feat.player_bfs_dist[push_pos_row, push_pos_col] == INF:
                out.append(cand)
                continue

            # 推后炸弹位
            bomb_next_col = bm.col + dc
            bomb_next_row = bm.row + dr
            if not (0 <= bomb_next_row < GRID_ROWS and 0 <= bomb_next_col < GRID_COLS):
                out.append(cand)
                continue

            if walls[bomb_next_row, bomb_next_col]:
                # 推入墙 → 引爆 (这里不区分对角/正交, 引擎已支持对角推炸进墙)
                cand.legal = True
                cand.note = "explode on wall"
                out.append(cand)
                continue

            if (bomb_next_col, bomb_next_row) in obstacles:
                out.append(cand)
                continue

            # 对角推 + 炸弹未撞墙 → 引擎规则: 炸弹仅允许"对角推入墙特例", 否则
            # 对角推非法.
            if is_diag:
                # 引擎只允许对角推炸弹入墙. 推到空地 → 不合法.
                out.append(cand)
                continue

            cand.legal = True
            out.append(cand)

    return out


# ── 探索候选 ──────────────────────────────────────────────

def _gen_inspect_candidates(bs: BeliefState,
                            feat: DomainFeatures) -> List[Candidate]:
    """挑 top-K info_gain > 0 的可达格作为视点."""
    if bs.fully_identified:
        return []
    ig = feat.info_gain_heatmap
    candidates: List[Tuple[float, int, int]] = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if ig[r, c] > 0 and feat.reachable_mask[r, c]:
                candidates.append((float(ig[r, c]), c, r))
    if not candidates:
        return []
    candidates.sort(reverse=True)
    out: List[Candidate] = []
    for score, c, r in candidates[:MAX_INSPECT]:
        # heading: 选 4 邻接里 IG 最高的方向作为朝向
        best_heading = 0
        best_neighbor_score = -1.0
        for k, (dc, dr) in enumerate(DIRS_4):
            nc, nr = c + dc, r + dr
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                # heading 8 朝向, 0=E,1=SE,...
                heading_8 = {0: 0, 1: 4, 2: 2, 3: 6}[k]  # E,W,S,N → 0,4,2,6
                if ig[nr, nc] > best_neighbor_score:
                    best_neighbor_score = ig[nr, nc]
                    best_heading = heading_8
        out.append(Candidate(
            type="inspect",
            viewpoint_col=c,
            viewpoint_row=r,
            inspect_heading=best_heading,
            legal=True,
        ))
    return out


# ── 主入口 ────────────────────────────────────────────────

def generate_candidates(bs: BeliefState,
                        feat: Optional[DomainFeatures] = None,
                        max_total: int = MAX_CANDIDATES) -> List[Candidate]:
    """生成 ≤ max_total 个候选 (含 padding)."""
    if feat is None:
        feat = compute_domain_features(bs)

    cands: List[Candidate] = []
    cands.extend(_gen_push_box_candidates(bs, feat))
    cands.extend(_gen_push_bomb_candidates(bs, feat))
    cands.extend(_gen_inspect_candidates(bs, feat))

    # 截断到 max_total - 1, 留 1 位给 return_garage / 全局 fallback
    if len(cands) > max_total:
        # 优先级: legal > illegal; type 内: 1-step > macro
        legal_cands = [c for c in cands if c.legal]
        illegal_cands = [c for c in cands if not c.legal]
        cands = legal_cands[:max_total] + illegal_cands[:max(0, max_total - len(legal_cands))]
        cands = cands[:max_total]

    # padding 到 max_total
    while len(cands) < max_total:
        cands.append(Candidate(type="pad", legal=False))

    return cands


def candidates_legality_mask(cands: List[Candidate]) -> np.ndarray:
    """提取合法性 mask: float32 [N], 合法 = 1, 非法 = 0."""
    return np.array([1.0 if c.legal else 0.0 for c in cands], dtype=np.float32)
