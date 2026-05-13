"""多箱联合 IDA* 求解器 — 推箱操作级搜索, 找全局最优推序.

和当前的"逐个箱子 BFS"不同, 这里把所有箱子+炸弹看作一个整体,
在"推操作"空间上做 IDA* 搜索.  每个"推操作" = 车走到推位 + 推一步.
搜索自然包含链式推、顺路推、炸弹开路等策略.

状态 = (车位置, 箱子位置集合, 炸弹位置集合, 被炸毁的墙集合)
操作 = Push(实体, 方向)
代价 = 车走到推位的步数 + 1

启发式 = Σ 每个箱子到匹配目标的最小推送距离
死锁检测 = 角落死锁 + 边缘死锁
"""

from __future__ import annotations

import copy
import heapq
import itertools
import time
from collections import deque
from itertools import permutations
from typing import (
    Dict, FrozenSet, List, Optional, Set, Tuple,
)

Pos = Tuple[int, int]
Dir = Tuple[int, int]
BoxEntry = Tuple[Pos, int]   # ((col, row), class_id)

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIRS_8 = [(1, 0), (-1, 0), (0, 1), (0, -1),
          (1, 1), (1, -1), (-1, 1), (-1, -1)]
DIRS_DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


# ── 状态 ───────────────────────────────────────────────────

class MBState:
    """不可变状态, 用于哈希和搜索."""
    __slots__ = ('car', 'boxes', 'bombs', 'destroyed', '_h', 'norm_car')

    def __init__(self, car: Pos,
                 boxes: FrozenSet[BoxEntry],
                 bombs: FrozenSet[Pos],
                 destroyed: FrozenSet[Pos],
                 norm_car: Pos = None):
        self.car = car
        self.boxes = boxes
        self.bombs = bombs
        self.destroyed = destroyed
        # norm_car: 车可达区域的规范化代表 (最小坐标)
        self.norm_car = norm_car if norm_car is not None else car
        self._h = hash((self.norm_car, boxes, bombs, destroyed))

    def __hash__(self):
        return self._h

    def __eq__(self, other):
        return (self.norm_car == other.norm_car and
                self.boxes == other.boxes and
                self.bombs == other.bombs and
                self.destroyed == other.destroyed)


# ── 求解器 ─────────────────────────────────────────────────

class MultiBoxSolver:
    """多箱联合 IDA* 求解器.

    Args:
        grid:     2D list, 1=墙 0=空
        car_pos:  (col, row)
        boxes:    [((col, row), class_id), ...]
        targets:  {class_id: (col, row)}
        bombs:    [(col, row), ...]
    """

    def __init__(self, grid, car_pos: Pos,
                 boxes: List[BoxEntry],
                 targets: Dict[int, Pos],
                 bombs: List[Pos],
                 trigger_map: Optional[Dict] = None,
                 belief_weight: float = 0.0):
        self.base_grid = [row[:] for row in grid]
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0
        self.targets = dict(targets)

        # 预计算初始地图的推送距离表 (reverse BFS)
        self._push_dist: Dict[int, Dict[Pos, int]] = {}
        for cid, tpos in self.targets.items():
            self._push_dist[cid] = self._min_push_distances(tpos, self.base_grid)

        # 缓存: destroyed → {cid → {pos → dist}}
        self._dist_cache: Dict[FrozenSet[Pos], Dict[int, Dict[Pos, int]]] = {
            frozenset(): self._push_dist,
        }

        # 计算初始状态的规范化车位置
        init_occupied = set()
        for pos, _ in boxes:
            init_occupied.add(pos)
        for pos in bombs:
            init_occupied.add(pos)
        norm_car_init = self._normalize_car(car_pos, self.base_grid, init_occupied)

        self.initial = MBState(
            car=car_pos,
            boxes=frozenset(boxes),
            bombs=frozenset(bombs),
            destroyed=frozenset(),
            norm_car=norm_car_init,
        )

        # ── Phase 1: 炸弹目标规划 ──
        self.target_walls: List[Pos] = []       # 需要被炸的墙
        self._wall_trigger_dist: Dict[Pos, Dict[Pos, int]] = {}  # wall → {pos → 推送距离}
        self._plan_bomb_targets()

        # 统计
        self.nodes = 0
        self.best_solution = None

        # Memoization caches (跨 solve_kbest 子调用共享, 大幅加速)
        self._heuristic_cache: Dict[MBState, int] = {}
        self._deadlock_cache: Dict[MBState, bool] = {}

        # ── Belief-aware cost augmentation ──
        # trigger_map: dict (cell, q4) -> set of scan_indices revealed
        # belief_weight: bonus 系数. push 走的路径上每出现一个未识别 trigger config -> 走路 cost 减 belief_weight
        # 这让 solver 主动选 walk-reveal 友好的推法, 减少后续 explicit scan.
        self._trigger_map = trigger_map or {}
        self._belief_weight = belief_weight
        self._bonus_cache: Dict[Tuple[Pos, Pos], int] = {}

    # ── 网格工具 ───────────────────────────────────────────

    def _get_grid(self, destroyed: FrozenSet[Pos]):
        if not destroyed:
            return self.base_grid
        g = [row[:] for row in self.base_grid]
        for c, r in destroyed:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                g[r][c] = 0
        return g

    def _in_bounds(self, c, r):
        return 0 <= c < self.cols and 0 <= r < self.rows

    def _is_wall(self, grid, c, r):
        if not self._in_bounds(c, r):
            return True
        return grid[r][c] == 1

    def _is_diagonal_dir(self, direction: Dir) -> bool:
        dx, dy = direction
        return dx != 0 and dy != 0

    def _can_step(self, grid, start: Pos, direction: Dir,
                  allow_dest_wall: bool = False) -> bool:
        """检查一步移动是否符合引擎的墙碰撞规则."""
        sc, sr = start
        dx, dy = direction
        dc, dr = sc + dx, sr + dy

        if not self._in_bounds(sc, sr):
            return False
        if not self._in_bounds(dc, dr):
            return False

        if self._is_diagonal_dir(direction):
            if self._is_wall(grid, sc + dx, sr):
                return False
            if self._is_wall(grid, sc, sr + dy):
                return False

        if self._is_wall(grid, dc, dr):
            return allow_dest_wall
        return True

    def _is_special_diagonal_bomb_push(self, grid, bomb_pos: Pos,
                                       direction: Dir,
                                       occupied: Optional[Set[Pos]] = None) -> bool:
        """仅允许炸弹斜推入对角墙，且两侧正交格都为空."""
        if not self._is_diagonal_dir(direction):
            return False

        bx, by = bomb_pos
        dx, dy = direction
        wall = (bx + dx, by + dy)
        if not self._in_bounds(*wall):
            return False
        if not self._is_wall(grid, wall[0], wall[1]):
            return False

        occupied = occupied or set()
        side_cells = ((bx + dx, by), (bx, by + dy))
        for cell in side_cells:
            if not self._in_bounds(*cell):
                return False
            if self._is_wall(grid, cell[0], cell[1]):
                return False
            if cell in occupied:
                return False

        return True

    def _normalize_car(self, car: Pos, grid, occupied: Set[Pos]) -> Pos:
        """车位置规范化: 找到车可达区域中坐标最小的位置.

        两个状态如果箱子/炸弹位置相同, 车只要在同一个连通区域,
        那么它们本质上是等价的. 用最小坐标代表整个区域.
        """
        # 规范化能减少等价状态，但在高分支图上每个子节点都跑一次 BFS
        # 代价过高；phase6_11 的热点几乎全在这里。先退回精确车位表示，
        # 用更低的单节点成本换取更高的扩展速度。
        return car

    # ── Phase 1: 炸弹目标规划 ──────────────────────────────

    def _plan_bomb_targets(self):
        """逆向推导: 分析哪些箱子到不了目标, 然后确定炸弹引爆位置.

        1. 对每个箱子做反向拉 BFS, 找出不可达的 (box, target) 对
        2. 遍历所有内部墙, 模拟 3×3 爆炸, 检查哪些爆炸能打通路径
        3. 筛选炸弹实际可达的引爆点, 按推送距离选最优方案
        """
        # 找不可达的箱子
        unreachable = []
        for (bc, br), cid in self.initial.boxes:
            if cid in self._push_dist:
                if (bc, br) not in self._push_dist[cid]:
                    unreachable.append(((bc, br), cid))

        if not unreachable:
            return  # 所有箱子都可达, 不需要炸弹

        needed_cids = {cid for _, cid in unreachable}
        print(f"  Phase1: 不可达箱子 cids={needed_cids}")

        # 收集所有内部墙壁 (边框不可摧毁)
        wall_cells = []
        for r in range(1, self.rows - 1):
            for c in range(1, self.cols - 1):
                if self.base_grid[r][c] == 1:
                    wall_cells.append((c, r))

        # 对每面墙, 模拟 3×3 爆炸, 检查哪些不可达箱子变可达
        # 同时计算炸弹推送距离, 过滤掉炸弹到不了的墙
        bomb_positions = list(self.initial.bombs)
        # candidates: [(wall, solved_cids, trigger_dist, best_bomb_dist)]
        candidates = []

        for wc, wr in wall_cells:
            destroyed = frozenset(
                (wc + dx, wr + dy)
                for dy in range(-1, 2) for dx in range(-1, 2)
                if 1 <= wr + dy < self.rows - 1 and 1 <= wc + dx < self.cols - 1
            )
            test_grid = self._get_grid(destroyed)

            solved = set()
            for (bc, br), cid in unreachable:
                target = self.targets[cid]
                pd = self._min_push_distances(target, test_grid)
                if (bc, br) in pd:
                    solved.add(cid)

            if not solved:
                continue

            # 计算炸弹到此墙的推送距离
            wdist = self._compute_wall_trigger_dist((wc, wr), self.base_grid)

            # 找最近的可达炸弹
            best_bomb_d = 999999
            for bp in bomb_positions:
                d = wdist.get(bp)
                if d is not None and d < best_bomb_d:
                    best_bomb_d = d

            if best_bomb_d < 999999:  # 至少有一颗炸弹能到达
                candidates.append(((wc, wr), solved, wdist, best_bomb_d))

        if not candidates:
            print(f"  Phase1: ⚠️ 没有炸弹可达的有效引爆点")
            return

        # 按覆盖 cid 数降序, 炸弹距离升序排列
        candidates.sort(key=lambda x: (-len(x[1]), x[3]))

        # 贪心选覆盖所有 needed_cids 的最优方案
        selected_walls: List[Pos] = []
        selected_dists: Dict[Pos, Dict[Pos, int]] = {}
        covered: Set[int] = set()

        for wall, solved, wdist, _ in candidates:
            new_covered = solved - covered
            if not new_covered:
                continue
            selected_walls.append(wall)
            selected_dists[wall] = wdist
            covered |= new_covered
            if covered == needed_cids:
                break

        if covered != needed_cids:
            print(f"  Phase1: ⚠️ 无法找到爆破方案覆盖 {needed_cids - covered}")
            return

        self.target_walls = selected_walls
        self._wall_trigger_dist = selected_dists

        # 分配炸弹 (选推送距离最短的)
        available_bombs = list(bomb_positions)
        for wall in selected_walls:
            wdist = selected_dists[wall]
            if not available_bombs:
                print(f"  Phase1: ⚠️ 炸弹不足")
                break
            best_bomb = min(available_bombs,
                            key=lambda b: wdist.get(b, 999999))
            bd = wdist.get(best_bomb, None)
            available_bombs.remove(best_bomb)
            print(f"  Phase1: 炸弹 {best_bomb} → 引爆墙 {wall} "
                  f"(推送距离={bd})")

    def _compute_wall_trigger_dist(self, wall_pos: Pos, grid) -> Dict[Pos, int]:
        """计算从任意位置到目标墙触发位的最小推送距离.

        炸弹要引爆墙 wall_pos, 必须从墙的相邻空格被推入墙中.
        触发位 = wall_pos 四个方向上的非墙相邻格.
        返回: {pos: min_pushes_to_any_trigger + 1(引爆推)}
        """
        wc, wr = wall_pos
        combined: Dict[Pos, int] = {}

        for dx, dy in DIRS:
            tc, tr = wc - dx, wr - dy  # 触发位 (炸弹在此, 被推向墙)
            if not self._in_bounds(tc, tr) or grid[tr][tc] == 1:
                continue
            # 从触发位反向 BFS, 得到推送距离表
            pd = self._min_push_distances((tc, tr), grid)
            for pos, d in pd.items():
                total = d + 1  # +1 是最后推入墙的那一步
                if pos not in combined or total < combined[pos]:
                    combined[pos] = total

        for dx, dy in DIRS_DIAG:
            tc, tr = wc - dx, wr - dy
            if not self._in_bounds(tc, tr) or grid[tr][tc] == 1:
                continue
            if not self._is_special_diagonal_bomb_push(grid, (tc, tr), (dx, dy)):
                continue
            pd = self._min_push_distances((tc, tr), grid)
            for pos, d in pd.items():
                total = d + 1
                if pos not in combined or total < combined[pos]:
                    combined[pos] = total

        return combined

    # ── 最小推送距离 (reverse BFS) ─────────────────────────

    def _min_push_distances(self, target: Pos, grid):
        """从 target 反向 BFS, 计算每个位置到 target 的最小推送次数.

        只考虑地形 (墙壁), 忽略其他箱子/炸弹.
        推一次 = 箱子从 (c,r) 移到 (c+dx, r+dy), 要求车能站在 (c-dx, r-dy).
        """
        dist: Dict[Pos, int] = {target: 0}
        queue = deque([target])

        while queue:
            c, r = queue.popleft()
            d = dist[(c, r)]

            for dx, dy in DIRS:
                # 箱子从 (c-dx, r-dy) 被推到 (c, r)
                # 即: 箱子原来在 prev, 车在 prev 的反方向
                prev = (c - dx, r - dy)
                car_behind = (c - 2 * dx, r - 2 * dy)

                if not self._can_step(grid, prev, (dx, dy)):
                    continue
                if not self._can_step(grid, car_behind, (dx, dy)):
                    continue

                if prev not in dist:
                    dist[prev] = d + 1
                    queue.append(prev)

        return dist

    def _get_push_dist(self, destroyed: FrozenSet[Pos]) -> Dict[int, Dict[Pos, int]]:
        """获取指定 destroyed 状态下的推送距离表 (带缓存)."""
        if destroyed in self._dist_cache:
            return self._dist_cache[destroyed]

        grid = self._get_grid(destroyed)
        push_dist: Dict[int, Dict[Pos, int]] = {}
        for cid, tpos in self.targets.items():
            push_dist[cid] = self._min_push_distances(tpos, grid)

        self._dist_cache[destroyed] = push_dist
        return push_dist

    # ── 启发式 ─────────────────────────────────────────────

    def _heuristic(self, state: MBState) -> int:
        """下界估计: 箱子到目标距离 + 炸弹到引爆点距离.

        使用当前 destroyed 状态对应的地图计算推送距离.
        如果有目标墙还未被炸, 加上炸弹到引爆点的最小推送距离.
        """
        cached = self._heuristic_cache.get(state)
        if cached is not None: return cached
        push_dist = self._get_push_dist(state.destroyed)
        h = 0
        for (bc, br), cid in state.boxes:
            if cid in push_dist:
                d = push_dist[cid].get((bc, br))
                if d is not None:
                    h += d
                elif state.bombs:
                    tc, tr = self.targets[cid]
                    h += abs(bc - tc) + abs(br - tr)
                else:
                    return 999999

        # 炸弹引导: 计算未炸墙的最小推送代价
        if self.target_walls and state.bombs:
            grid = self._get_grid(state.destroyed)
            undestroyed = []
            for wall in self.target_walls:
                wc, wr = wall
                if grid[wr][wc] == 1:  # 墙还在
                    undestroyed.append(wall)

            if undestroyed:
                bombs_list = list(state.bombs)
                if len(bombs_list) >= len(undestroyed):
                    # 最优分配: 每面墙分一颗最近的炸弹
                    if len(undestroyed) == 1:
                        wall = undestroyed[0]
                        wdist = self._wall_trigger_dist.get(wall, {})
                        min_d = min((wdist.get(bp, 999) for bp in bombs_list),
                                    default=999)
                        if min_d < 999:
                            h += min_d
                    else:
                        # 枚举炸弹排列找最小总距离 (≤3! = 6)
                        min_total = 999999
                        for perm in permutations(bombs_list, len(undestroyed)):
                            total = 0
                            for wall, bomb in zip(undestroyed, perm):
                                wdist = self._wall_trigger_dist.get(wall, {})
                                total += wdist.get(bomb, 999)
                            min_total = min(min_total, total)
                        if min_total < 999999:
                            h += min_total

        self._heuristic_cache[state] = h
        return h

    # ── 死锁检测 ───────────────────────────────────────────

    def _is_deadlock(self, state: MBState) -> bool:
        cached = self._deadlock_cache.get(state)
        if cached is not None: return cached
        grid = self._get_grid(state.destroyed)
        push_dist = self._get_push_dist(state.destroyed)

        result = False
        for (bc, br), cid in state.boxes:
            target = self.targets.get(cid)
            if target and (bc, br) == target:
                continue  # 已在目标上

            # 角落死锁: 两面相邻墙 + 不在目标位
            # 如果还有炸弹, 墙可能被炸掉, 跳过角落检测
            if not state.bombs:
                wu = self._is_wall(grid, bc, br - 1)
                wd = self._is_wall(grid, bc, br + 1)
                wl = self._is_wall(grid, bc - 1, br)
                wr = self._is_wall(grid, bc + 1, br)

                if (wu and wl) or (wu and wr) or (wd and wl) or (wd and wr):
                    result = True; break

            # 检查箱子是否到不了目标 (当前地图)
            if cid in push_dist:
                if (bc, br) not in push_dist[cid]:
                    if not state.bombs:
                        result = True; break

        self._deadlock_cache[state] = result
        return result

    # ── 车的 BFS 移动 ─────────────────────────────────────

    def _car_bfs(self, start: Pos, goal: Pos,
                 grid, occupied: Set[Pos]) -> Optional[int]:
        """车从 start 走到 goal 的最短步数 (避开墙+占位实体).

        返回步数, 或 None 不可达.
        """
        dists = self._car_bfs_all(start, grid, occupied)
        return dists.get(goal)

    def _car_bfs_all(self, start: Pos, grid,
                     occupied: Set[Pos]) -> Dict[Pos, int]:
        """车从 start 出发到所有可达空地的最短步数 (仅上下左右)."""
        visited = {start}
        dist: Dict[Pos, int] = {start: 0}
        queue = deque([start])

        while queue:
            c, r = queue.popleft()
            d = dist[(c, r)]
            for dc, dr in DIRS:
                nc, nr = c + dc, r + dr
                if ((nc, nr) not in visited and
                        self._can_step(grid, (c, r), (dc, dr)) and
                        self._in_bounds(nc, nr) and
                        grid[nr][nc] == 0 and
                        (nc, nr) not in occupied):
                    visited.add((nc, nr))
                    dist[(nc, nr)] = d + 1
                    queue.append((nc, nr))

        return dist

    def _car_bfs_path(self, start: Pos, goal: Pos,
                      grid, occupied: Set[Pos]
                      ) -> Optional[List[Dir]]:
        """车的 BFS, 返回方向列表 (仅上下左右)."""
        if start == goal:
            return []
        if goal in occupied:
            return None

        visited = {start}
        queue = deque([(start, [])])

        while queue:
            (c, r), path = queue.popleft()
            for dc, dr in DIRS:
                nc, nr = c + dc, r + dr
                new_path = path + [(dc, dr)]
                if (nc, nr) == goal:
                    if self._can_step(grid, (c, r), (dc, dr)):
                        return new_path
                    continue
                if ((nc, nr) not in visited and
                        self._can_step(grid, (c, r), (dc, dr)) and
                        self._in_bounds(nc, nr) and
                        grid[nr][nc] == 0 and
                        (nc, nr) not in occupied):
                    visited.add((nc, nr))
                    queue.append(((nc, nr), new_path))

        return None

    # ── 推操作枚举 ─────────────────────────────────────────

    def _get_occupied(self, state: MBState) -> Set[Pos]:
        occ = set()
        for pos, _ in state.boxes:
            occ.add(pos)
        for pos in state.bombs:
            occ.add(pos)
        return occ

    def _check_chain(self, pos: Pos, dx: int, dy: int,
                     grid, box_map: Dict[Pos, BoxEntry],
                     bomb_set: Set[Pos]) -> bool:
        """检查从 pos 沿 (dx,dy) 方向的链式推是否合法."""
        if self._is_diagonal_dir((dx, dy)):
            return False
        cur = pos
        while True:
            allow_dest_wall = cur in bomb_set
            if not self._can_step(grid, cur, (dx, dy),
                                  allow_dest_wall=allow_dest_wall):
                return False
            nxt = (cur[0] + dx, cur[1] + dy)
            if self._is_wall(grid, nxt[0], nxt[1]):
                return cur in bomb_set
            if nxt in box_map or nxt in bomb_set:
                cur = nxt
                continue
            return True

    _DIR_Q4_LOOKUP = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}

    def _compute_belief_bonus(self, start: Pos, end: Pos,
                                push_dir: Optional[Tuple[int, int]],
                                grid, occupied: Set[Pos]) -> int:
        """Belief-aware bonus: 走从 start 到 end 的最短路径上, 经过多少 unique 触发配置
        (cell, q4) 命中 trigger_map. 再算 push 后落点的额外 reveal.
        每个 unique scan_idx 算 1.
        """
        if not self._trigger_map: return 0
        path = self._car_bfs_path(start, end, grid, occupied)
        if path is None: return 0
        revealed = 0
        cur = start
        for dx, dy in path:
            cell = (cur[0] + dx, cur[1] + dy)
            q4 = self._DIR_Q4_LOOKUP.get((dx, dy))
            if q4 is not None:
                ents = self._trigger_map.get((cell, q4))
                if ents:
                    for j in ents: revealed |= (1 << j)
            cur = cell
        # 加 push 落点 reveal: push 后车在 push_pos+push_dir, 朝 push_dir
        if push_dir is not None:
            after_pos = (cur[0] + push_dir[0], cur[1] + push_dir[1])
            q4 = self._DIR_Q4_LOOKUP.get(push_dir)
            if q4 is not None:
                ents = self._trigger_map.get((after_pos, q4))
                if ents:
                    for j in ents: revealed |= (1 << j)
        return bin(revealed).count('1')

    def _enum_pushes(self, state: MBState):
        """枚举所有合法推操作 (支持 8 方向 + 链式推).

        Yields: (entity_type, entity_id, direction, walk_cost)
           entity_type: 'box' 或 'bomb'
           entity_id: BoxEntry 或 Pos
           direction: (dx, dy)
           walk_cost: 车走到推位的步数 (可能被 belief-bonus 调整)
        """
        grid = self._get_grid(state.destroyed)
        occupied = self._get_occupied(state)
        walk_dist = self._car_bfs_all(state.car, grid, occupied)
        box_positions = {pos for pos, _ in state.boxes}
        box_map = {pos: entry for entry in state.boxes for pos in [entry[0]]}
        bomb_set = set(state.bombs)

        # ── 推箱子 (仅正交方向 + 链式推) ──
        for (bx, by), cid in state.boxes:
            for dx, dy in DIRS:
                push_from = (bx - dx, by - dy)
                push_to = (bx + dx, by + dy)

                pfx, pfy = push_from
                if not self._in_bounds(pfx, pfy):
                    continue
                if grid[pfy][pfx] == 1:
                    continue
                if push_from in occupied:
                    continue
                if not self._can_step(grid, push_from, (dx, dy)):
                    continue

                ptx, pty = push_to
                if not self._can_step(grid, (bx, by), (dx, dy)):
                    continue

                if ((ptx, pty) in box_positions and (ptx, pty) != (bx, by)) or \
                   (ptx, pty) in bomb_set:
                    if not self._check_chain((bx, by), dx, dy,
                                             grid, box_map, bomb_set):
                        continue

                walk = walk_dist.get(push_from)
                if walk is None:
                    continue

                yield ('box', ((bx, by), cid), (dx, dy), walk)

        # ── 推炸弹 (正交方向 + 对角墙引爆特例) ──
        has_bomb_targets = bool(self.target_walls)
        undestroyed_walls = set()
        if has_bomb_targets:
            for wall in self.target_walls:
                wc, wr = wall
                if grid[wr][wc] == 1:
                    undestroyed_walls.add(wall)

        for bx, by in state.bombs:
            for dx, dy in DIRS_8:
                push_from = (bx - dx, by - dy)
                push_to = (bx + dx, by + dy)

                pfx, pfy = push_from
                if not self._in_bounds(pfx, pfy):
                    continue
                if grid[pfy][pfx] == 1:
                    continue
                if push_from in occupied:
                    continue
                if not self._can_step(grid, push_from, (dx, dy)):
                    continue

                ptx, pty = push_to
                if self._is_diagonal_dir((dx, dy)):
                    if not self._is_special_diagonal_bomb_push(
                            grid, (bx, by), (dx, dy), occupied - {(bx, by)}):
                        continue
                    is_detonation = True
                else:
                    if not self._can_step(grid, (bx, by), (dx, dy),
                                          allow_dest_wall=True):
                        continue

                    is_detonation = self._is_wall(grid, ptx, pty)
                    if not is_detonation:
                        if (ptx, pty) in box_positions or (ptx, pty) in bomb_set:
                            if not self._check_chain((bx, by), dx, dy,
                                                     grid, box_map, bomb_set):
                                continue

                if has_bomb_targets and undestroyed_walls:
                    if is_detonation:
                        covers_target = False
                        for ddy in range(-1, 2):
                            for ddx in range(-1, 2):
                                if (ptx + ddx, pty + ddy) in undestroyed_walls:
                                    covers_target = True
                                    break
                            if covers_target:
                                break
                        if not covers_target:
                            continue
                    else:
                        bomb_pos = (bx, by)
                        useful = False
                        for wall in undestroyed_walls:
                            wdist = self._wall_trigger_dist.get(wall, {})
                            d_before = wdist.get(bomb_pos, 999)
                            d_after = wdist.get(push_to, 999)
                            if d_after < d_before:
                                useful = True
                                break
                        if not useful:
                            continue

                walk = walk_dist.get(push_from)
                if walk is None:
                    continue

                yield ('bomb', (bx, by), (dx, dy), walk)

    # ── 推操作执行 ─────────────────────────────────────────

    def _apply_push(self, state: MBState,
                    etype: str, eid, direction: Dir
                    ) -> Optional[MBState]:
        """执行一次推操作 (支持链式推), 返回新状态."""
        dx, dy = direction
        grid = self._get_grid(state.destroyed)

        new_boxes = set(state.boxes)
        new_bombs = set(state.bombs)
        new_destroyed = set(state.destroyed)

        if etype == 'box':
            first_pos = eid[0]
            car_after = first_pos
        elif etype == 'bomb':
            first_pos = eid
            car_after = first_pos
        else:
            return None

        box_map = {pos: entry for entry in new_boxes for pos in [entry[0]]}
        bomb_pos_set = set(new_bombs)

        chain = []
        pos = first_pos
        while True:
            if pos in box_map:
                entry = box_map[pos]
                chain.append(('box', entry, pos))
                pos = (pos[0] + dx, pos[1] + dy)
                continue
            if pos in bomb_pos_set:
                chain.append(('bomb', pos, pos))
                pos = (pos[0] + dx, pos[1] + dy)
                continue
            break

        for i in range(len(chain) - 1, -1, -1):
            ctype, cid_val, old_p = chain[i]
            new_p = (old_p[0] + dx, old_p[1] + dy)
            allow_dest_wall = ctype == 'bomb'
            if self._is_diagonal_dir(direction):
                if ctype != 'bomb':
                    return None
                remaining = {pos for pos, _ in new_boxes}
                remaining |= set(new_bombs)
                remaining.discard(old_p)
                if not self._is_special_diagonal_bomb_push(
                        grid, old_p, direction, remaining):
                    return None
            if not self._can_step(grid, old_p, direction,
                                  allow_dest_wall=allow_dest_wall):
                return None

            if ctype == 'box':
                old_box_pos, box_cid = cid_val
                new_boxes.discard((old_box_pos, box_cid))

                if self._is_wall(grid, new_p[0], new_p[1]):
                    return None

                target = self.targets.get(box_cid)
                if target and new_p == target:
                    pass
                else:
                    new_boxes.add((new_p, box_cid))
            else:
                old_bomb_pos = cid_val
                new_bombs.discard(old_bomb_pos)

                npx, npy = new_p
                if self._is_wall(grid, npx, npy):
                    for ddy in range(-1, 2):
                        for ddx in range(-1, 2):
                            wr = npy + ddy
                            wc = npx + ddx
                            if (1 <= wr < self.rows - 1 and
                                    1 <= wc < self.cols - 1):
                                new_destroyed.add((wc, wr))
                else:
                    new_bombs.add(new_p)

        # 计算新状态的 grid (destroyed 可能已变)
        new_destroyed_fs = frozenset(new_destroyed)
        new_boxes_fs = frozenset(new_boxes)
        new_bombs_fs = frozenset(new_bombs)

        # 计算规范化车位置
        new_grid = self._get_grid(new_destroyed_fs)
        new_occupied = set()
        for pos, _ in new_boxes_fs:
            new_occupied.add(pos)
        for pos in new_bombs_fs:
            new_occupied.add(pos)
        norm_car = self._normalize_car(car_after, new_grid, new_occupied)

        return MBState(
            car=car_after,
            boxes=new_boxes_fs,
            bombs=new_bombs_fs,
            destroyed=new_destroyed_fs,
            norm_car=norm_car,
        )

    # ── IDA* 搜索 ─────────────────────────────────────────

    def _solve_best_first(self, max_cost: int,
                          time_limit: float,
                          weight: float = 1.5,
                          start_state: Optional['MBState'] = None,
                          prefix: Optional[List] = None,
                          quiet: bool = False) -> Optional[List]:
        """Weighted A*: 优先找可行解, 适合高分支复杂图.

        K-best 支持:
          start_state: 从指定 state 开始搜索 (默认 self.initial).
          prefix: 已经执行的 move 序列 (前缀). 返回 plan 会拼上前缀.
          quiet: 抑制日志输出.
        """
        start = start_state if start_state is not None else self.initial
        prefix = list(prefix) if prefix else []
        start_h = self._heuristic(start)
        if start_h >= 999999 and not start.bombs:
            return None

        t0 = time.perf_counter()
        counter = itertools.count()
        open_heap = []
        heapq.heappush(
            open_heap,
            (start_h * weight, start_h, 0, next(counter), start),
        )
        parent: Dict[MBState, Tuple[Optional[MBState], Optional[Tuple]]] = {
            start: (None, None),
        }
        best_g: Dict[MBState, int] = {start: 0}
        nodes = 0

        while open_heap:
            if time.perf_counter() - t0 > time_limit:
                print(f"  BestFirst: ⏱️ 超时 ({time_limit}s)")
                return None

            _, _, g, _, state = heapq.heappop(open_heap)
            if g != best_g.get(state):
                continue

            nodes += 1
            if self._is_goal(state):
                path: List = []
                cur = state
                while parent[cur][0] is not None:
                    prev, move = parent[cur]
                    path.append(move)
                    cur = prev
                path.reverse()
                elapsed = time.perf_counter() - t0
                if not quiet:
                    print(f"  BestFirst: ✅ 找到解! 推操作={len(prefix) + len(path)}, "
                          f"总步数={sum(wc + 1 for _, _, _, wc in prefix + path)}, "
                          f"节点={nodes}, 耗时={elapsed:.2f}s")
                self.nodes = nodes
                return prefix + path

            if self._is_deadlock(state):
                continue

            pushes = list(self._enum_pushes(state))
            scored_pushes = []
            for etype, eid, direction, walk_cost in pushes:
                new_state = self._apply_push(state, etype, eid, direction)
                if new_state is None:
                    continue
                cost = walk_cost + 1
                new_g = g + cost
                if new_g >= best_g.get(new_state, 999999999):
                    continue
                if new_g > max_cost:
                    continue
                h = self._heuristic(new_state)
                scored_pushes.append(
                    (new_g + weight * h, h, new_g,
                     etype, eid, direction, walk_cost, new_state)
                )

            scored_pushes.sort(key=lambda x: (x[0], x[1], x[2]))
            for _, h, new_g, etype, eid, direction, walk_cost, new_state in scored_pushes:
                best_g[new_state] = new_g
                parent[new_state] = (
                    state, (etype, eid, direction, walk_cost)
                )
                heapq.heappush(
                    open_heap,
                    (new_g + weight * h, h, new_g,
                     next(counter), new_state)
                )

        if not quiet:
            print("  BestFirst: 无解")
        return None

    def solve_kbest(self, k: int = 5,
                    max_cost: int = 300,
                    time_limit: float = 30.0) -> List[List]:
        """枚举 K 个 (可能多样化) god plan, 按 cost 排序返回.

        策略:
          1. 先解出 base plan (无约束, optimal).
          2. 对每个能合法 first-push 的箱子 / 推方向, 强制 first move = 该 push,
             从 state-after-first-move 继续 A*. 得到 first-push 多样的备选 plan.
          3. 去重 (按 plan tuple), 按 cost 排序, 返回 top-K.

        time_limit 分摊到 base + 每个变体. quiet 模式下不打印 base solver 日志.
        """
        if k <= 0: return []
        plans: List[List] = []
        seen = set()

        def cost(p): return sum(wc + 1 for _, _, _, wc in p)

        def add_plan(p):
            if p is None: return
            key = tuple(p)
            if key in seen: return
            seen.add(key)
            plans.append(p)

        t0 = time.perf_counter()
        # base
        base_budget = max(0.5, time_limit * 0.4)
        base = self._solve_best_first(max_cost, base_budget, quiet=False)
        add_plan(base)
        if base is None and not self.initial.bombs:
            return []

        # 枚举所有合法 first-push (state.initial → 推不同 box)
        initial_pushes = list(self._enum_pushes(self.initial))
        # 按 (entity_id pos) 分组, 每个 entity 取所有方向作为候选 first-move
        by_pos = {}
        for move in initial_pushes:
            etype, eid, direction, wc = move
            pos_key = eid[0] if etype == 'box' else eid
            by_pos.setdefault(pos_key, []).append(move)

        remaining_budget = max(0.5, time_limit - (time.perf_counter() - t0))
        per_variant = max(0.3, remaining_budget / max(len(by_pos), 1))

        # 每个 entity 取 1 个方向 (top-1) — swap variant 补充多样性
        first_move_candidates = []
        for pos_key, moves in by_pos.items():
            first_move_candidates.append(min(moves, key=lambda m: m[3]))
        # 跳过 base plan 已有的 first push
        base_first = tuple(base[0]) if base else None
        for first_move in first_move_candidates:
            if len(plans) >= k * 2: break
            if time.perf_counter() - t0 > time_limit: break
            if base_first == tuple(first_move): continue  # 已是 base
            etype, eid, direction, walk_cost = first_move
            state1 = self._apply_push(self.initial, etype, eid, direction)
            if state1 is None: continue
            if self._is_deadlock(state1): continue
            rest_max_cost = max_cost - (walk_cost + 1)
            if rest_max_cost <= 0: continue
            variant = self._solve_best_first(
                rest_max_cost, per_variant,
                start_state=state1, prefix=[first_move], quiet=True
            )
            add_plan(variant)

        # ── push-swap permutation 变体 (Plan A.11) ──
        # 对每个 base/variant plan, 尝试交换相邻 commutative push 对.
        # 同 push 顺序但 box-level 等价 → 不同 walk 路径 → 可能 belief-aware 更友好.
        swap_variants = []
        for plan in list(plans):   # 拷贝因为 add_plan 会 mutate
            swap_variants.extend(self._generate_swap_variants(plan))
        for v in swap_variants:
            add_plan(v)

        # 按 cost 排序, 截断 (留更宽空间给 swap 变体)
        plans.sort(key=cost)
        return plans[:max(k, k * 2)]

    def _generate_swap_variants(self, plan: List) -> List[List]:
        """生成交换相邻 commutative 推对的 plan 变体.
        两个 push 在 box-level 等价 (boxes/bombs/destroyed 相同, car_pos 可不同) 视为可交换.
        车位置不同→ 不同 walk 路径 → 给 v18 belief-aware 评估更多多样性。
        """
        if not plan or len(plan) < 2: return []
        states_orig = [self.initial]
        cur = self.initial
        for move in plan:
            cur = self._apply_push(cur, *move[:3])
            if cur is None: return []
            states_orig.append(cur)
        variants = []
        seen_keys = set()
        for i in range(len(plan) - 1):
            if plan[i][:3] == plan[i+1][:3]: continue
            s = self._apply_push(states_orig[i], *plan[i+1][:3])
            if s is None: continue
            s = self._apply_push(s, *plan[i][:3])
            if s is None: continue
            target = states_orig[i+2]
            if (s.boxes != target.boxes or s.bombs != target.bombs
                or s.destroyed != target.destroyed): continue
            swapped = list(plan)
            swapped[i], swapped[i+1] = swapped[i+1], swapped[i]
            key = tuple((m[0], m[1], m[2]) for m in swapped)
            if key in seen_keys: continue
            seen_keys.add(key)
            variants.append(swapped)
        return variants

    def solve(self, max_cost: int = 300,
              time_limit: float = 30.0,
              strategy: str = "auto") -> Optional[List]:
        """求解.

        Returns:
            推操作列表 [(etype, eid, direction, walk_cost), ...]
            或 None 无解/超时
        """
        t0_global = time.perf_counter()
        time_limit = max(0.01, float(time_limit))
        use_ida_fallback = strategy == "auto" and time_limit >= 2.0

        if strategy in ("auto", "best_first"):
            # Respect the caller's budget. Online teacher queries rely on
            # tight per-state limits and should not be expanded silently.
            bf_limit = time_limit if not use_ida_fallback else time_limit * 0.5
            result = self._solve_best_first(max_cost=max_cost,
                                            time_limit=bf_limit)
            if result is not None or strategy == "best_first" or not use_ida_fallback:
                return result

        # IDA* 用剩余时间
        elapsed = time.perf_counter() - t0_global
        remaining = time_limit - elapsed
        if remaining < 2.0:
            print(f"  IDA*: 剩余时间不足 ({remaining:.1f}s)")
            return None

        h0 = self._heuristic(self.initial)
        if h0 >= 999999:
            if self.initial.bombs:
                print("  IDA*: 初始状态有箱子到不了目标, 但有炸弹可开路")
                h0 = 0  # 用 0 作为起始 bound, 让搜索尝试炸弹操作
                for (bc, br), cid in self.initial.boxes:
                    tc, tr = self.targets[cid]
                    h0 += abs(bc - tc) + abs(br - tr)
            else:
                print("  IDA*: 初始状态有箱子到不了目标")
                return None

        self.nodes = 0
        self._t0 = time.perf_counter()
        self._time_limit = remaining
        self._timed_out = False
        self._transposition: Dict[MBState, int] = {}  # state → best g
        bound = h0
        path: List = []

        print(f"  IDA*: 初始启发值={h0}, 箱子={len(self.initial.boxes)}, "
              f"炸弹={len(self.initial.bombs)}")

        while bound <= max_cost and not self._timed_out:
            self._transposition.clear()  # 每个 iteration 清空
            result = self._search(self.initial, 0, bound, path)
            if isinstance(result, list):
                elapsed = time.perf_counter() - self._t0
                print(f"  IDA*: ✅ 找到解! 推操作={len(result)}, "
                      f"总步数={sum(wc+1 for _,_,_,wc in result)}, "
                      f"节点={self.nodes}, 耗时={elapsed:.2f}s")
                return result
            if result == float('inf'):
                print(f"  IDA*: 无解 (bound={bound})")
                return None
            old_bound = bound
            bound = result
            elapsed = time.perf_counter() - self._t0
            tt_size = len(self._transposition)
            print(f"  IDA*: bound {old_bound}→{bound}, "
                  f"nodes={self.nodes}, tt={tt_size}, {elapsed:.1f}s")

        if self._timed_out:
            print(f"  IDA*: ⏱️ 超时 ({self._time_limit}s)")
        return None

    def _search(self, state: MBState, g: int, bound: int,
                path: List) -> any:
        """IDA* 递归搜索.

        Returns:
            list  — 找到解 (path 的副本)
            int   — 下一个 bound
            inf   — 这个分支无解
        """
        if self._timed_out:
            return float('inf')

        # 超时检测
        if self.nodes % 5000 == 0:
            if time.perf_counter() - self._t0 > self._time_limit:
                self._timed_out = True
                return float('inf')

        self.nodes += 1

        h = self._heuristic(state)
        f = g + h

        if f > bound:
            return f

        if self._is_goal(state):
            return list(path)

        # 传递表剪枝: 如果此状态已以更低 g 值访问过, 跳过
        prev_g = self._transposition.get(state)
        if prev_g is not None and prev_g <= g:
            return float('inf')
        self._transposition[state] = g

        if self._is_deadlock(state):
            return float('inf')

        min_t = float('inf')

        # 枚举所有推操作, 先生成并计算子状态的 f 值用于排序
        pushes = list(self._enum_pushes(state))

        # 预计算每个推操作的子状态启发值, 用于智能排序
        scored_pushes = []
        for etype, eid, direction, walk_cost in pushes:
            new_state = self._apply_push(state, etype, eid, direction)
            if new_state is None:
                continue
            cost = walk_cost + 1
            child_h = self._heuristic(new_state)
            child_f = g + cost + child_h
            scored_pushes.append((child_f, etype, eid, direction, walk_cost, new_state))

        # 按子节点的 f 值排序 (最有希望的先搜)
        scored_pushes.sort(key=lambda x: x[0])

        for child_f, etype, eid, direction, walk_cost, new_state in scored_pushes:
            # 如果子节点 f 值已超 bound, 后面的都更大, 直接 break
            cost = walk_cost + 1
            if child_f > bound:
                min_t = min(min_t, child_f)
                break

            move = (etype, eid, direction, walk_cost)
            path.append(move)

            t = self._search(new_state, g + cost, bound, path)
            path.pop()

            if isinstance(t, list):
                return t
            min_t = min(min_t, t)

        return min_t

    def _is_goal(self, state: MBState) -> bool:
        return len(state.boxes) == 0

    # ── 解→动作序列 ───────────────────────────────────────

    def solution_to_actions(self, solution: List) -> List[Dir]:
        """将推操作序列转换为车移动方向序列."""
        actions: List[Dir] = []
        state = self.initial

        for etype, eid, direction, walk_cost in solution:
            grid = self._get_grid(state.destroyed)
            occupied = self._get_occupied(state)

            # 车走到推位
            if etype == 'box':
                old_pos = eid[0]
            else:
                old_pos = eid
            push_from = (old_pos[0] - direction[0],
                         old_pos[1] - direction[1])

            walk_dirs = self._car_bfs_path(
                state.car, push_from, grid, occupied)
            if walk_dirs:
                actions.extend(walk_dirs)

            # 推一步
            actions.append(direction)

            # 更新状态
            state = self._apply_push(state, etype, eid, direction)

        return actions
