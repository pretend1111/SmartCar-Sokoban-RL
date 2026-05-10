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
    inspect_heading: Optional[int] = None         # 0..7 朝向 (东/SE/南/SW/西/NW/北/NE)
    inspect_target_type: Optional[str] = None     # "box" / "target" — JEPP 老师选 inspect 时指向的 entity 类型
    inspect_target_idx: Optional[int] = None      # bs.boxes / bs.targets 里的索引

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


def _box_at(bs: BeliefState, col: int, row: int,
            exclude_idx: Optional[int] = None) -> Optional[int]:
    """在 (col, row) 找箱子索引, 排除 exclude_idx."""
    for j, b in enumerate(bs.boxes):
        if j == exclude_idx:
            continue
        if b.col == col and b.row == row:
            return j
    return None


def _bomb_at(bs: BeliefState, col: int, row: int,
             exclude_idx: Optional[int] = None) -> Optional[int]:
    for k, bm in enumerate(bs.bombs):
        if k == exclude_idx:
            continue
        if bm.col == col and bm.row == row:
            return k
    return None


def _can_chain_push(bs: BeliefState, walls: np.ndarray,
                    box_idx: int, dc: int, dr: int,
                    already_in_chain: Set[int]) -> bool:
    """递归检查 box_idx 沿 (dc, dr) 是否可推. 撞另一箱 / 炸弹 → 递归.

    撞墙 (本箱要推到墙位) → 不可推 (箱不能炸).
    """
    if box_idx in already_in_chain:
        return False
    b = bs.boxes[box_idx]
    new_col = b.col + dc
    new_row = b.row + dr
    if not (0 <= new_row < GRID_ROWS and 0 <= new_col < GRID_COLS):
        return False
    if walls[new_row, new_col]:
        return False
    next_bomb = _bomb_at(bs, new_col, new_row)
    if next_bomb is not None:
        # box → bomb → ... 链
        return _can_chain_bomb_push(bs, walls, next_bomb, dc, dr,
                                      bombs_in_chain=set(),
                                      boxes_in_chain=already_in_chain | {box_idx})
    next_box = _box_at(bs, new_col, new_row, exclude_idx=box_idx)
    if next_box is not None:
        return _can_chain_push(bs, walls, next_box, dc, dr,
                                already_in_chain | {box_idx})
    return True


def _can_chain_bomb_push(bs: BeliefState, walls: np.ndarray,
                          bomb_idx: int, dc: int, dr: int,
                          bombs_in_chain: Set[int],
                          boxes_in_chain: Set[int]) -> bool:
    """递归检查 bomb_idx 沿 (dc, dr) 是否可推. 撞墙 → 引爆 OK; 撞另炸弹 → 递归;
    撞箱 → 看箱能否再推.
    """
    if bomb_idx in bombs_in_chain:
        return False
    bm = bs.bombs[bomb_idx]
    new_col = bm.col + dc
    new_row = bm.row + dr
    if not (0 <= new_row < GRID_ROWS and 0 <= new_col < GRID_COLS):
        return False
    if walls[new_row, new_col]:
        # 撞墙引爆, 链终止
        return True
    next_bomb = _bomb_at(bs, new_col, new_row, exclude_idx=bomb_idx)
    if next_bomb is not None:
        return _can_chain_bomb_push(bs, walls, next_bomb, dc, dr,
                                      bombs_in_chain | {bomb_idx},
                                      boxes_in_chain)
    next_box = _box_at(bs, new_col, new_row)
    if next_box is not None:
        if next_box in boxes_in_chain:
            return False
        # 不要预先把 next_box 加进 already_in_chain — _can_chain_push 内部第一行
        # 会做 cycle check, 如果 next_box 已在 set 里它会 return False (误判 chain blocked).
        return _can_chain_push(bs, walls, next_box, dc, dr,
                                already_in_chain=boxes_in_chain)
    return True


# ── 推箱候选 ──────────────────────────────────────────────

def _gen_push_box_candidates(bs: BeliefState,
                             feat: DomainFeatures,
                             enforce_sigma_lock: bool = False) -> List[Candidate]:
    """枚举每箱 × 4 方向 (含 1-3 步 macro).

    enforce_sigma_lock=True 时 (V2 训练用): 若 push 落到 target cell, 要求 σ 已锁定
        (Π 在 box-target 这格上是单射). 否则标 illegal — 强制模型先 inspect.
    """
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

            # 2) 推后箱位必须可推. 允许链式推 (撞另一箱/炸弹 → 它也被同向推).
            if not _is_free(box_next_col, box_next_row, walls, obstacles):
                # 检查 box_next 是箱 / 炸弹 / 墙
                hit_box_idx = _box_at(bs, box_next_col, box_next_row, exclude_idx=i)
                hit_bomb_idx = _bomb_at(bs, box_next_col, box_next_row)
                if hit_box_idx is not None:
                    # 箱推箱链
                    if not _can_chain_push(bs, walls, hit_box_idx, dc, dr,
                                            already_in_chain={i}):
                        cand.note = "chain blocked"
                        out.append(cand)
                        continue
                elif hit_bomb_idx is not None:
                    # 箱推炸弹链 (炸弹被推到墙 → 引爆 OK)
                    if not _can_chain_bomb_push(bs, walls, hit_bomb_idx, dc, dr,
                                                  bombs_in_chain=set(),
                                                  boxes_in_chain={i}):
                        cand.note = "bomb chain blocked"
                        out.append(cand)
                        continue
                else:
                    # 撞墙
                    cand.note = "box_next blocked (wall)"
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

            # 3.5) σ-lock 抑制场: 若 box_next 是 target cell 且 σ 未锁, 拒绝 commit
            if enforce_sigma_lock:
                target_at_dest = None
                for j, t in enumerate(bs.targets):
                    if (t.col, t.row) == (box_next_col, box_next_row):
                        target_at_dest = j
                        break
                if target_at_dest is not None:
                    pi_row_sum = int(bs.Pi[i].sum())
                    pi_col_sum = int(bs.Pi[:, target_at_dest].sum())
                    sigma_locked = (bs.Pi[i, target_at_dest] > 0.5
                                    and pi_row_sum == 1 and pi_col_sum == 1)
                    if not sigma_locked:
                        cand.note = "uncertain σ-commit"
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
                # σ-lock 抑制 (macro 落地是 target 也要锁)
                macro_block_sigma = False
                if enforce_sigma_lock:
                    for j2, t2 in enumerate(bs.targets):
                        if (t2.col, t2.row) == (next_box_col, next_box_row):
                            pr_s = int(bs.Pi[i].sum())
                            pc_s = int(bs.Pi[:, j2].sum())
                            if not (bs.Pi[i, j2] > 0.5 and pr_s == 1 and pc_s == 1):
                                macro_block_sigma = True
                            break
                if macro_block_sigma:
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
                # 撞实体: 是箱链 / 炸弹链 → 递归看链能否成立
                if is_diag:
                    # 对角推炸弹仅支持入墙特例, 撞实体不行
                    out.append(cand)
                    continue
                # 链头是 bomb 还是 box?
                hit_bomb = _bomb_at(bs, bomb_next_col, bomb_next_row, exclude_idx=k)
                hit_box = _box_at(bs, bomb_next_col, bomb_next_row)
                if hit_bomb is not None:
                    if _can_chain_bomb_push(bs, walls, hit_bomb, dc, dr,
                                             bombs_in_chain={k}, boxes_in_chain=set()):
                        cand.legal = True
                        cand.note = "bomb chain"
                        out.append(cand)
                        continue
                if hit_box is not None:
                    # bomb 推 box: 把 box 当链头检查
                    if _can_chain_push(bs, walls, hit_box, dc, dr,
                                        already_in_chain=set()):
                        cand.legal = True
                        cand.note = "bomb→box chain"
                        out.append(cand)
                        continue
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
    """对每个未识别 entity, 枚举其 8 邻 + LOS 通的 viewpoint, 朝向 entity.

    匹配 engine 严格 FOV 规则: 必须距离 ≤ √2 + 朝向 ≤ ±22.5° + 视线无遮挡.
    """
    if bs.fully_identified:
        return []

    walls = bs.M.astype(bool)

    obstacles: Set[Tuple[int, int]] = set()
    for b in bs.boxes:
        obstacles.add((b.col, b.row))
    for bm in bs.bombs:
        obstacles.add((bm.col, bm.row))

    # 用于 has_line_of_sight 的 entity_positions (含 targets)
    entity_pos: Set[Tuple[int, int]] = set(obstacles)
    for t in bs.targets:
        entity_pos.add((t.col, t.row))

    DIRS_8 = (
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    )
    # heading 量化: dc, dr → 0..7 (东=0, SE=1, 南=2, ..., NE=7)
    DIR_TO_HEADING = {
        (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
        (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7,
    }

    out: List[Candidate] = []
    seen_keys: Set[Tuple[int, int, str, int]] = set()

    def _has_los(fc: int, fr: int, tc: int, tr: int) -> bool:
        # 内联 has_line_of_sight 简化版 (只看墙 + 实体 (排除 entity 自身))
        x0, y0 = fc + 0.5, fr + 0.5
        x1, y1 = tc + 0.5, tr + 0.5
        import math as _m
        dx, dy = x1 - x0, y1 - y0
        dist = _m.sqrt(dx * dx + dy * dy)
        if dist < 0.1:
            return True
        steps = int(dist * 4) + 1
        for i in range(1, steps):
            t = i / steps
            px = x0 + dx * t
            py = y0 + dy * t
            cc, rr = int(px), int(py)
            if 0 <= rr < GRID_ROWS and 0 <= cc < GRID_COLS:
                if walls[rr, cc]:
                    return False
                if (cc, rr) != (tc, tr) and (cc, rr) in entity_pos:
                    return False
        return True

    def _enum_for_entity(etype: str, eidx: int, ec: int, er: int):
        for dc, dr in DIRS_8:
            nc, nr = ec + dc, er + dr
            if not (0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS):
                continue
            if walls[nr, nc]:
                continue
            if (nc, nr) in obstacles:
                continue
            if not feat.reachable_mask[nr, nc]:
                continue
            if not _has_los(nc, nr, ec, er):
                continue
            # 朝向: 从 viewpoint 看 entity 的方向 (反 dc, dr)
            head_dc, head_dr = -dc, -dr
            heading = DIR_TO_HEADING.get((head_dc, head_dr), 0)
            key = (nc, nr, etype, eidx)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(Candidate(
                type="inspect",
                viewpoint_col=nc,
                viewpoint_row=nr,
                inspect_heading=heading,
                inspect_target_type=etype,
                inspect_target_idx=eidx,
                legal=True,
            ))

    for i, b in enumerate(bs.boxes):
        if b.class_id is None:
            _enum_for_entity("box", i, b.col, b.row)
    for j, t in enumerate(bs.targets):
        if t.num_id is None:
            _enum_for_entity("target", j, t.col, t.row)

    # 兜底: 若某未识别 entity 没有任何严格 8 邻 viewpoint, 加 "宽松" 候选 —
    # 找最近可达 cell 任意方向, heading=0 占位 (实战这种 cell 旁边推一推就能产生空隙).
    # 这种候选的 legal=True 但实际可能识别不了 entity, 算占位让老师有动作可选.
    coverage: Set[Tuple[str, int]] = set()
    for c in out:
        coverage.add((c.inspect_target_type, c.inspect_target_idx))

    def _enum_loose(etype: str, eidx: int, ec: int, er: int):
        # 找距 entity ≤ 3 的可达 cell, 任意方向
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nc, nr = ec + dc, er + dr
                if not (0 <= nc < GRID_COLS and 0 <= nr < GRID_ROWS):
                    continue
                if walls[nr, nc] or (nc, nr) in obstacles:
                    continue
                if not feat.reachable_mask[nr, nc]:
                    continue
                # heading 朝 entity 大致方向 (取符号化 dc, dr)
                hdc = -1 if dc > 0 else (1 if dc < 0 else 0)
                hdr = -1 if dr > 0 else (1 if dr < 0 else 0)
                if (hdc, hdr) == (0, 0):
                    continue
                heading = DIR_TO_HEADING.get((hdc, hdr), 0)
                key = (nc, nr, etype, eidx)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append(Candidate(
                    type="inspect",
                    viewpoint_col=nc,
                    viewpoint_row=nr,
                    inspect_heading=heading,
                    inspect_target_type=etype,
                    inspect_target_idx=eidx,
                    legal=True,
                    note="loose-fallback",
                ))
                return

    for i, b in enumerate(bs.boxes):
        if b.class_id is None and ("box", i) not in coverage:
            _enum_loose("box", i, b.col, b.row)
    for j, t in enumerate(bs.targets):
        if t.num_id is None and ("target", j) not in coverage:
            _enum_loose("target", j, t.col, t.row)

    return out[:MAX_INSPECT]


# ── 主入口 ────────────────────────────────────────────────

def generate_candidates(bs: BeliefState,
                        feat: Optional[DomainFeatures] = None,
                        max_total: int = MAX_CANDIDATES,
                        enforce_sigma_lock: bool = False) -> List[Candidate]:
    """生成 ≤ max_total 个候选 (含 padding).

    enforce_sigma_lock=True (V2): 推到 target cell 的 push 必须 σ 锁定 (Π 单射)
        否则标 illegal — 强制 inspect 优先.
    """
    if feat is None:
        feat = compute_domain_features(bs)

    cands: List[Candidate] = []
    cands.extend(_gen_push_box_candidates(bs, feat, enforce_sigma_lock=enforce_sigma_lock))
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
