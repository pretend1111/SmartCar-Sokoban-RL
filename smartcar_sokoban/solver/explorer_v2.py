"""plan_exploration_v2 — V1 探索器扩展, 卡住时推开障碍再继续.

逻辑:
    1. 跑标准 plan_exploration
    2. 仍有未识别 entity → 枚举 (其他箱/炸弹, 4 方向) 试推:
       a. push_pos 可达
       b. box_next 非墙/非其他实体
       c. box_next 不是 target cell (避免误消)
       d. box_next 不是角落死格 (防止推到不可解死角)
       e. 推后用 clone engine 跑 plan_exploration, 验证某 unid entity 变成 observed
    3. 选最便宜的推 (BFS 路径 + 推 1 步) 应用到真 engine, 再调 plan_exploration
    4. 最多 retry K 次, 每次只解锁 1 entity

输出: 跟 plan_exploration 同样的 low-level action 列表.
"""

from __future__ import annotations

import contextlib
import copy
import io
from typing import List, Optional, Set, Tuple

from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.solver.explorer import (
    plan_exploration, exploration_complete, direction_to_action,
)
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid


DIRS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIRS_8 = DIRS_4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _compute_deadlock_corners(grid) -> Set[Tuple[int, int]]:
    """简单 corner-deadlock: 格 (c, r) 非墙, 但有相邻墙 (r±1) + (c±1) 任意组合."""
    rows = len(grid); cols = len(grid[0])
    out = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                continue
            wn = (r == 0) or grid[r - 1][c] == 1
            ws = (r == rows - 1) or grid[r + 1][c] == 1
            we = (c == cols - 1) or grid[r][c + 1] == 1
            ww = (c == 0) or grid[r][c - 1] == 1
            if (wn and we) or (wn and ww) or (ws and we) or (ws and ww):
                out.add((c, r))
    return out


def _apply_push_recorded(eng, ent_pos: Tuple[int, int], direction: Tuple[int, int],
                          actions_out: List[int]) -> bool:
    """把车导航到推位 + 推一步, 把 low-level actions 记入 actions_out."""
    ec, er = ent_pos
    dc, dr = direction
    car_target = (ec - dc, er - dr)

    eng.discrete_step(6); actions_out.append(6)   # snap
    state = eng.get_state()
    obstacles = set()
    for b in state.boxes: obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs: obstacles.add(pos_to_grid(bm.x, bm.y))
    car_grid = pos_to_grid(state.car_x, state.car_y)
    if car_grid != car_target:
        path = bfs_path(car_grid, car_target, state.grid, obstacles)
        if path is None:
            return False
        for pdx, pdy in path:
            a = direction_to_abs_action(pdx, pdy)
            eng.discrete_step(a); actions_out.append(a)
    a = direction_to_abs_action(dc, dr)
    eng.discrete_step(a); actions_out.append(a)
    return True


def _count_unidentified(state) -> int:
    return ((len(state.boxes) - len(state.seen_box_ids))
            + (len(state.targets) - len(state.seen_target_ids)))


def _find_first_unid(state):
    """返回 (entity_type, idx, (col, row)) 或 None."""
    for i, b in enumerate(state.boxes):
        if i not in state.seen_box_ids:
            return ("box", i, pos_to_grid(b.x, b.y))
    for i, t in enumerate(state.targets):
        if i not in state.seen_target_ids:
            return ("target", i, pos_to_grid(t.x, t.y))
    return None


def _bfs_reach_count(start: Tuple[int, int], grid, obstacles) -> int:
    """车从 start 4-邻 BFS 能到的格子数 (不含起点)."""
    from collections import deque
    rows = len(grid); cols = len(grid[0])
    visited = {start}
    q = deque([start])
    while q:
        c, r = q.popleft()
        for dc, dr in DIRS_4:
            nc, nr = c + dc, r + dr
            if (nc, nr) in visited: continue
            if not (0 <= nc < cols and 0 <= nr < rows): continue
            if grid[nr][nc] == 1 or (nc, nr) in obstacles: continue
            visited.add((nc, nr))
            q.append((nc, nr))
    return len(visited) - 1


def _try_clear_one_push(engine, deadlock: Set[Tuple[int, int]]
                          ) -> Optional[Tuple[int, List[int], str]]:
    """枚举所有 (entity, direction) 推, 优先选:
       1. 让 unid 立即减少
       2. 否则让 car BFS 可达格子增加 (打开后续探索通路)
    """
    state = engine.get_state()
    grid = state.grid
    rows = len(grid); cols = len(grid[0])
    target_cells = {pos_to_grid(t.x, t.y) for t in state.targets}
    n_unid_before = _count_unidentified(state)

    car_grid = pos_to_grid(state.car_x, state.car_y)

    box_pos = [(i, pos_to_grid(b.x, b.y), b.class_id) for i, b in enumerate(state.boxes)]
    bomb_pos = [(k, pos_to_grid(bm.x, bm.y)) for k, bm in enumerate(state.bombs)]
    all_ent_pos = {p for _, p, *_ in box_pos} | {p for _, p in bomb_pos}

    reach_before = _bfs_reach_count(car_grid, grid, all_ent_pos)
    candidates = []   # (priority, cost, ent_kind, idx, dir, ent_pos, info)

    # 试推 box
    for i, (c, r), cid in box_pos:
        for dc, dr in DIRS_4:
            next_pos = (c + dc, r + dr)
            push_pos = (c - dc, r - dr)
            # 边界
            if not (0 <= next_pos[0] < cols and 0 <= next_pos[1] < rows):
                continue
            if not (0 <= push_pos[0] < cols and 0 <= push_pos[1] < rows):
                continue
            # 推后不能撞墙
            if grid[next_pos[1]][next_pos[0]] == 1:
                continue
            # 推后不能撞其他实体
            if next_pos in (all_ent_pos - {(c, r)}):
                continue
            # 不推到 target cell (避免误消除)
            if next_pos in target_cells:
                continue
            # 不推到死角
            if next_pos in deadlock:
                continue
            # push_pos 不能是墙 / 其他实体
            if grid[push_pos[1]][push_pos[0]] == 1:
                continue
            if push_pos in (all_ent_pos - {(c, r)}):
                continue
            # 车能到 push_pos
            obstacles = all_ent_pos - {(c, r)}
            path = bfs_path(car_grid, push_pos, grid, obstacles)
            if path is None:
                continue
            cost = len(path) + 1   # walk + push

            # clone engine, 模拟推
            eng_clone = copy.deepcopy(engine)
            tmp_actions: List[int] = []
            if not _apply_push_recorded(eng_clone, (c, r), (dc, dr), tmp_actions):
                continue
            # 评估 (a) unid 减少 (跑 plan_exploration), (b) 否则 BFS 可达增加
            with contextlib.redirect_stdout(io.StringIO()):
                plan_exploration(eng_clone)
            sc = eng_clone.get_state()
            n_unid_after = _count_unidentified(sc)
            # 推后新 obstacles set (entity 推到 next_pos)
            new_ent_pos = (all_ent_pos - {(c, r)}) | {next_pos}
            new_car_grid = pos_to_grid(sc.car_x, sc.car_y)
            reach_after = _bfs_reach_count(new_car_grid, grid, new_ent_pos)
            unid_drop = n_unid_before - n_unid_after
            reach_gain = reach_after - reach_before
            if unid_drop > 0:
                priority = 0   # immediate progress
                candidates.append((priority, cost, "box", i, (dc, dr), (c, r),
                                   {"unid_drop": unid_drop, "reach_gain": reach_gain}))
            elif reach_gain > 0:
                priority = 1   # path opening
                candidates.append((priority, cost, "box", i, (dc, dr), (c, r),
                                   {"unid_drop": 0, "reach_gain": reach_gain}))

    # 试推 bomb (4 方向; 推炸弹入墙会引爆 — 这里限制不撞墙)
    for k, (c, r) in bomb_pos:
        for dc, dr in DIRS_4:
            next_pos = (c + dc, r + dr)
            push_pos = (c - dc, r - dr)
            if not (0 <= next_pos[0] < cols and 0 <= next_pos[1] < rows): continue
            if not (0 <= push_pos[0] < cols and 0 <= push_pos[1] < rows): continue
            if grid[next_pos[1]][next_pos[0]] == 1: continue   # 不推入墙 (避免引爆)
            if next_pos in (all_ent_pos - {(c, r)}): continue
            if next_pos in deadlock: continue
            if grid[push_pos[1]][push_pos[0]] == 1: continue
            if push_pos in (all_ent_pos - {(c, r)}): continue
            obstacles = all_ent_pos - {(c, r)}
            path = bfs_path(car_grid, push_pos, grid, obstacles)
            if path is None: continue
            cost = len(path) + 1
            eng_clone = copy.deepcopy(engine)
            tmp_actions: List[int] = []
            if not _apply_push_recorded(eng_clone, (c, r), (dc, dr), tmp_actions):
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                plan_exploration(eng_clone)
            sc = eng_clone.get_state()
            n_unid_after = _count_unidentified(sc)
            new_ent_pos = (all_ent_pos - {(c, r)}) | {next_pos}
            new_car_grid = pos_to_grid(sc.car_x, sc.car_y)
            reach_after = _bfs_reach_count(new_car_grid, grid, new_ent_pos)
            unid_drop = n_unid_before - n_unid_after
            reach_gain = reach_after - reach_before
            if unid_drop > 0:
                candidates.append((0, cost, "bomb", k, (dc, dr), (c, r),
                                   {"unid_drop": unid_drop, "reach_gain": reach_gain}))
            elif reach_gain > 0:
                candidates.append((1, cost, "bomb", k, (dc, dr), (c, r),
                                   {"unid_drop": 0, "reach_gain": reach_gain}))

    if not candidates:
        return None
    # 排序: priority 0 (unid drop) 优先, 然后 -unid_drop, 然后 -reach_gain, 然后 cost
    candidates.sort(key=lambda x: (
        x[0],
        -x[6].get("unid_drop", 0),
        -x[6].get("reach_gain", 0),
        x[1],
    ))
    best = candidates[0]
    priority, cost, ent_kind, idx, dir, ent_pos, info = best

    actions: List[int] = []
    if not _apply_push_recorded(engine, ent_pos, dir, actions):
        return None
    detail = (f"push {ent_kind}{idx}@{ent_pos} dir={dir} "
              f"(prio={priority}, unid={info.get('unid_drop',0)}, "
              f"reach+={info.get('reach_gain',0)}, cost={cost})")
    return cost, actions, detail


def plan_exploration_v2(engine, max_retries: int = 5,
                         verbose: bool = False) -> List[int]:
    """V1 explore + 推开障碍补丁."""
    all_actions: List[int] = []

    # 标准 explore
    with contextlib.redirect_stdout(io.StringIO()):
        actions = plan_exploration(engine)
    all_actions.extend(actions)

    state = engine.get_state()
    if exploration_complete(state):
        return all_actions

    deadlock = _compute_deadlock_corners(state.grid)

    for retry in range(max_retries):
        if exploration_complete(engine.get_state()):
            break
        result = _try_clear_one_push(engine, deadlock)
        if result is None:
            if verbose:
                print(f"  [v2 retry {retry}] no push helps, stop")
            break
        cost, push_actions, detail = result
        all_actions.extend(push_actions)
        if verbose:
            print(f"  [v2 retry {retry}] {detail}, +{len(push_actions)} low-level")
        with contextlib.redirect_stdout(io.StringIO()):
            new_actions = plan_exploration(engine)
        all_actions.extend(new_actions)

    return all_actions
