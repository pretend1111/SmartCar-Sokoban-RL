"""JEPP — Joint Explore-Push Planner (greedy commit version).

形式化:
    POMDP 状态: (s_phys, K, Π).
    动作: push_box / push_bomb (确定性) / inspect (改 K, Π).
    目标: 最小化 E[Σ c(a_t)].

Greedy commit 启发:
    1. 找 "对所有 σ ∈ Σ(Π) 都减距离" 的 push (committable). 取 commit value 最大者.
    2. 无 committable push → 选 cost 最低的 inspect (走到 viewpoint + 旋转).
    3. 都没有 → fallback: 取兼容 σ 数最多的 push (软 commit).

输出: 在线 (state, candidate_label) 序列, 喂数据生成.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from smartcar_sokoban.symbolic.belief import BeliefState, GRID_ROWS, GRID_COLS
from smartcar_sokoban.symbolic.features import (
    DomainFeatures, INF, _reverse_push_bfs, DIRS_4,
)
from smartcar_sokoban.symbolic.candidates import Candidate


def all_target_dist_fields(bs: BeliefState) -> List[np.ndarray]:
    """对每个 target 计算 reverse-push BFS 距离场 [GRID_ROWS, GRID_COLS]."""
    walls = bs.M.astype(bool)
    fields: List[np.ndarray] = []
    for t in bs.targets:
        fields.append(_reverse_push_bfs(t.col, t.row, walls))
    return fields


def push_committable(cand: Candidate, bs: BeliefState,
                     target_fields: List[np.ndarray],
                     ) -> Tuple[bool, float]:
    """检查 push 候选是否 committable: 对每个 j∈{Π[i,j]=1}, 推后距离都减小.

    返回 (是否 commit, commit_value=min_j (d_old - d_new)).
    """
    if cand.type != "push_box" or not cand.legal:
        return False, 0.0
    if cand.box_idx is None or cand.direction is None:
        return False, 0.0

    box = bs.boxes[cand.box_idx]
    dc, dr = cand.direction
    end_col = box.col + cand.run_length * dc
    end_row = box.row + cand.run_length * dr
    if not (0 <= end_row < GRID_ROWS and 0 <= end_col < GRID_COLS):
        return False, 0.0

    Pi = bs.Pi   # [N_box, N_target]
    n_target = len(bs.targets)
    if Pi.shape[1] == 0:
        return False, 0.0

    compatible_js = [j for j in range(n_target) if Pi[cand.box_idx, j] > 0.5]
    if not compatible_js:
        return False, 0.0

    min_improvement = float("inf")
    for j in compatible_js:
        if j >= len(target_fields):
            return False, 0.0
        field = target_fields[j]
        d_old = field[box.row, box.col]
        d_new = field[end_row, end_col]
        if d_old == INF or d_new == INF:
            return False, 0.0    # 不可达 → 不可 commit
        if d_new >= d_old:
            return False, 0.0    # 这个 σ 的 target 没改进 → 不能 commit
        min_improvement = min(min_improvement, d_old - d_new)

    return True, float(min_improvement)


def push_soft_commit_score(cand: Candidate, bs: BeliefState,
                           target_fields: List[np.ndarray]
                           ) -> float:
    """软 commit: 兼容 σ 中改进的比例 (用于 fallback)."""
    if cand.type != "push_box" or not cand.legal:
        return -1.0
    box = bs.boxes[cand.box_idx]
    dc, dr = cand.direction
    end_col = box.col + cand.run_length * dc
    end_row = box.row + cand.run_length * dr
    if not (0 <= end_row < GRID_ROWS and 0 <= end_col < GRID_COLS):
        return -1.0

    Pi = bs.Pi
    compatible_js = [j for j in range(Pi.shape[1]) if Pi[cand.box_idx, j] > 0.5]
    if not compatible_js:
        return -1.0

    n_improved = 0
    total_improve = 0.0
    for j in compatible_js:
        if j >= len(target_fields):
            continue
        d_old = target_fields[j][box.row, box.col]
        d_new = target_fields[j][end_row, end_col]
        if d_old == INF or d_new == INF:
            continue
        if d_new < d_old:
            n_improved += 1
            total_improve += (d_old - d_new)
    if not compatible_js:
        return -1.0
    return (n_improved / len(compatible_js)) * 100 + (total_improve / max(1, n_improved))


def _bfs_walk_cost(start: Tuple[int, int], end: Tuple[int, int],
                   walls: np.ndarray, obstacles: set) -> int:
    """4 邻 BFS 路径长度 (车走路成本). 不可达返回 INF."""
    if start == end:
        return 0
    rows, cols = walls.shape
    visited = {start}
    q = deque()
    q.append((start, 0))
    while q:
        (c, r), d = q.popleft()
        for dc, dr in DIRS_4:
            nc, nr = c + dc, r + dr
            if (nc, nr) in visited:
                continue
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if walls[nr, nc]:
                continue
            if (nc, nr) in obstacles:
                continue
            visited.add((nc, nr))
            if (nc, nr) == end:
                return d + 1
            q.append(((nc, nr), d + 1))
    return INF


def inspect_cost(cand: Candidate, bs: BeliefState) -> int:
    """估算 inspect 的低层动作成本 = 走路 + 旋转."""
    if cand.type != "inspect" or cand.viewpoint_col is None:
        return INF
    walls = bs.M.astype(bool)
    obstacles = {(b.col, b.row) for b in bs.boxes}
    obstacles.update({(bm.col, bm.row) for bm in bs.bombs})
    car = (bs.player_col, bs.player_row)
    end = (cand.viewpoint_col, cand.viewpoint_row)
    walk = _bfs_walk_cost(car, end, walls, obstacles)
    if walk == INF:
        return INF
    # 旋转成本: 当前朝向 → cand.inspect_heading. theta_player ∈ [0,7]
    cur = bs.theta_player
    tgt = cand.inspect_heading or 0
    diff = (tgt - cur) % 8
    rot = min(diff, 8 - diff)
    return walk + rot


def jepp_pick_action(bs: BeliefState, feat: DomainFeatures,
                     candidates: List[Candidate]) -> Optional[Candidate]:
    """JEPP 主决策: 选 commitable push, 否则选 cheapest inspect, 否则 soft commit."""

    target_fields = all_target_dist_fields(bs)

    # 1. Committable push (硬 commit)
    commit_pushes: List[Tuple[Candidate, float]] = []
    for c in candidates:
        if c.type != "push_box" or not c.legal:
            continue
        ok, value = push_committable(c, bs, target_fields)
        if ok:
            commit_pushes.append((c, value))

    if commit_pushes:
        # 取 commit value 最大者 (减距离最多), 同分按 run_length 大优先
        commit_pushes.sort(key=lambda x: (-x[1], -x[0].run_length))
        return commit_pushes[0][0]

    # 1b. Committable bomb? bomb push 引爆墙也属于 commit 类 (墙必爆才能继续).
    # 简化: 任何合法的 push_bomb 都视为 commitable (墙是 σ-无关的)
    bomb_cands = [c for c in candidates if c.type == "push_bomb" and c.legal]
    if bomb_cands and not bs.fully_identified:
        # 只有炸弹推入墙才算 commit (改变物理), 推到空地不 commit
        walls = bs.M.astype(bool)
        for c in bomb_cands:
            bm = bs.bombs[c.bomb_idx]
            dc, dr = c.direction
            new_col = bm.col + dc
            new_row = bm.row + dr
            if 0 <= new_row < GRID_ROWS and 0 <= new_col < GRID_COLS:
                if walls[new_row, new_col]:
                    return c   # 推炸弹入墙, 不依赖 σ
    elif bomb_cands and bs.fully_identified:
        return bomb_cands[0]   # 完全识别后任何合法 bomb 都可执行

    # 2. 已完全识别 → 不需要 inspect, 但也没 commit 的 push (僵局)
    #    → 取最大 push_dist 减少的 push
    if bs.fully_identified:
        best = None
        best_score = -INF
        for c in candidates:
            if c.type == "push_box" and c.legal:
                score = push_soft_commit_score(c, bs, target_fields)
                if score > best_score:
                    best_score = score
                    best = c
        return best

    # 3. 否则 → cheapest inspect
    inspect_cands = [c for c in candidates if c.type == "inspect" and c.legal]
    if inspect_cands:
        scored: List[Tuple[int, Candidate]] = []
        for c in inspect_cands:
            cost = inspect_cost(c, bs)
            if cost == INF:
                continue
            scored.append((cost, c))
        if scored:
            scored.sort(key=lambda x: x[0])
            return scored[0][1]

    # 4. 兜底: soft commit push (每个 σ 中改进比例最高)
    best = None
    best_score = -INF
    for c in candidates:
        if c.type == "push_box" and c.legal:
            score = push_soft_commit_score(c, bs, target_fields)
            if score > best_score:
                best_score = score
                best = c
    if best is not None:
        return best

    # 5. 终极兜底: 任意合法 push (避免 None 卡死). 优先 push_box, 其次 push_bomb.
    legal_pushes = [c for c in candidates
                    if c.legal and c.type == "push_box" and c.run_length == 1]
    if legal_pushes:
        return legal_pushes[0]
    legal_bombs = [c for c in candidates if c.legal and c.type == "push_bomb"]
    if legal_bombs:
        return legal_bombs[0]
    return None
