"""自动玩家 — 整合探索、推箱求解与炸弹规划，全自动通关.

流程:
    Phase 1 — 探索:  巡游地图, FOV 扫描获取所有箱子/目标编号
    Phase 2 — 分析:  匹配配对, 检测被墙阻挡的目标, 规划 TNT 引爆方案
    Phase 3 — 执行:  按最优顺序逐个推送:
                     - 优先处理不需要炸弹的配对
                     - 在不干扰后续操作的时机穿插执行 TNT 引爆
                     - 引爆后重新规划剩余配对的路线
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from smartcar_sokoban.action_defs import is_translation_action
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from smartcar_sokoban.solver.push_solver import bfs_push, estimate_push_cost
from smartcar_sokoban.solver.explorer import (
    plan_exploration, direction_to_action,
)
from smartcar_sokoban.solver.bomb_planner import (
    analyze_bomb_tasks, plan_bomb_execution, simulate_explosion,
    BombTask,
)


class AutoPlayer:
    """推箱子全自动求解器.

    三阶段执行:
        Phase 1 — 探索: 巡游地图, FOV 扫描获取所有箱子/目标编号
        Phase 2 — 分析: 识别被墙阻挡的配对, 规划 TNT 开路方案
        Phase 3 — 执行: 智能交错推箱和引爆操作, 引爆后重规划

    用法:
        player = AutoPlayer(engine)
        actions = player.solve()
    """

    def __init__(self, engine):
        self.engine = engine

    def solve(self) -> List[int]:
        """全自动求解，返回完整动作序列."""
        all_actions: List[int] = []

        # ── Phase 1: 探索 ──────────────────────────────────
        print("═══ Phase 1: 探索扫描 ═══")
        explore_actions = plan_exploration(self.engine)
        all_actions.extend(explore_actions)

        state = self.engine.get_state()
        print(f"  探索完成, 用了 {len(explore_actions)} 步")
        print(f"  发现 {len(state.seen_box_ids)}/{len(state.boxes)} "
              f"箱子, {len(state.seen_target_ids)}/{len(state.targets)} 目标")

        # ── Phase 2: 分析配对与炸弹需求 ────────────────────
        print("═══ Phase 2: 分析配对 ═══")
        state = self.engine.get_state()
        pairs = self._match_pairs(state)
        print(f"  匹配到 {len(pairs)} 个配对")

        bomb_tasks = []
        if state.bombs:
            bomb_tasks = analyze_bomb_tasks(state, pairs)
            if bomb_tasks:
                blocked_pair_ids = set()
                for task in bomb_tasks:
                    for bi, ti in task.target_pairs:
                        blocked_pair_ids.add((bi, ti))
                print(f"  发现 {len(blocked_pair_ids)} 个被墙阻挡的配对")
                print(f"  规划了 {len(bomb_tasks)} 次 TNT 引爆:")
                for i, task in enumerate(bomb_tasks):
                    bc, br = task.bomb_grid
                    wc, wr = task.wall_grid
                    print(f"    [{i+1}] 炸弹@({bc},{br}) → "
                          f"墙@({wc},{wr}), "
                          f"可解锁 {len(task.target_pairs)} 个配对")
            else:
                print("  所有配对均可直接推送, 无需炸弹")
        else:
            print("  地图无炸弹")

        # ── Phase 3: 执行推箱 + 炸弹 ──────────────────────
        print("═══ Phase 3: 推箱求解 ═══")
        push_actions = self._solve_with_bombs(bomb_tasks)
        all_actions.extend(push_actions)

        state = self.engine.get_state()
        if state.won:
            print(f"🎉 通关！总步数: {len(all_actions)}")
        else:
            remaining = len(state.boxes)
            print(f"⚠️ 未完全通关, 剩余 {remaining} 个箱子")

        return all_actions

    # ── Phase 3: 智能执行 ──────────────────────────────────

    def _solve_with_bombs(self, bomb_tasks: List[BombTask]) -> List[int]:
        """智能交错执行推箱和炸弹引爆.

        策略:
        1. 每轮循环: 先尝试所有可直接推送的配对
        2. 如果没有可直接推送的 → 执行最优的 TNT 引爆
        3. 引爆后重新分析并规划
        4. 重复直到全部完成或无解
        """
        all_actions: List[int] = []
        pending_bombs = list(bomb_tasks)  # 待执行的炸弹任务
        push_count = 0
        bomb_count = 0
        max_rounds = 50  # 安全上限
        # 防止无限循环: 记录已引爆的 (bomb_grid, wall_grid) 对
        detonated: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()

        for round_num in range(max_rounds):
            state = self.engine.get_state()
            if not state.boxes or state.won:
                break

            # 匹配当前可用的配对
            pairs = self._match_pairs(state)
            if not pairs:
                print("  ❌ 无法匹配任何箱子-目标对!")
                break

            # ── 尝试直接推送 ──
            solved_one = False
            sorted_pairs = self._sort_pairs(pairs, state)

            for box_idx, target_idx in sorted_pairs:
                box = state.boxes[box_idx]
                target = state.targets[target_idx]

                actions = self._solve_one_push(state, box_idx, target_idx)
                if actions is not None:
                    push_count += 1
                    print(f"  [{push_count}] 推箱子 #{box_idx}"
                          f"(id={box.class_id}) → "
                          f"目标 #{target_idx}"
                          f"(id={target.num_id})")

                    # 逐步执行, 检测卡死后提前中断
                    executed = 0
                    stuck = False
                    for a in actions:
                        old_x, old_y = state.car_x, state.car_y
                        state = self.engine.discrete_step(a)
                        all_actions.append(a)
                        executed += 1
                        # 任意平移动作都应改变车位置, 否则说明被阻
                        if is_translation_action(a):
                            if (abs(state.car_x - old_x) < 0.01 and
                                    abs(state.car_y - old_y) < 0.01):
                                print(f"    ⚠️ 第 {executed}/{len(actions)}"
                                      f" 步车被阻, 重新规划")
                                stuck = True
                                break
                    if not stuck:
                        print(f"    ✅ 完成, {len(actions)} 步")
                    solved_one = True
                    break  # 重新评估

            if solved_one:
                continue

            # ── 没有可直接推送的 → 重新分析炸弹需求 ──
            if state.bombs:
                print("    重新分析炸弹需求...")
                new_tasks = analyze_bomb_tasks(state, pairs)
                if new_tasks:
                    # 过滤掉已引爆过的方案
                    fresh_tasks = [
                        nt for nt in new_tasks
                        if (nt.bomb_grid, nt.wall_grid) not in detonated
                    ]
                    if fresh_tasks:
                        # 替换 pending (不做合并, 用最新分析结果)
                        pending_bombs = fresh_tasks
                        blocked_count = sum(len(t.target_pairs)
                                            for t in fresh_tasks)
                        print(f"    发现 {blocked_count} 个被阻挡配对, "
                              f"{len(fresh_tasks)} 个新炸弹任务")
                    else:
                        pending_bombs = []
                        print("    所有可用炸弹方案已尝试过")

            # ── 执行炸弹引爆 ──
            if pending_bombs:
                bomb_result = self._execute_best_bomb(
                    state, pending_bombs)
                if bomb_result is not None:
                    actions, used_task = bomb_result
                    bomb_count += 1
                    bc, br = used_task.bomb_grid
                    wc, wr = used_task.wall_grid
                    print(f"  💣[{bomb_count}] 引爆炸弹@({bc},{br})"
                          f" → 墙@({wc},{wr})")

                    # 逐步执行引爆路径, 检测卡死
                    executed = 0
                    stuck = False
                    for a in actions:
                        old_x, old_y = state.car_x, state.car_y
                        state = self.engine.discrete_step(a)
                        all_actions.append(a)
                        executed += 1
                        if is_translation_action(a):
                            if (abs(state.car_x - old_x) < 0.01 and
                                    abs(state.car_y - old_y) < 0.01):
                                print(f"    ⚠️ 第 {executed}/{len(actions)}"
                                      f" 步车被阻, 重新规划")
                                stuck = True
                                break
                    if not stuck:
                        print(f"    ✅ 引爆完成, {len(actions)} 步")

                    # 记录已引爆, 防止重复
                    detonated.add((used_task.bomb_grid,
                                   used_task.wall_grid))
                    # 从 pending 移除 (按位置匹配)
                    pending_bombs = [
                        t for t in pending_bombs
                        if t.bomb_grid != used_task.bomb_grid
                    ]
                    continue

            if not solved_one:
                print("  ❌ 所有配对均无解, 停止")
                break

        return all_actions

    # ── 炸弹执行 ───────────────────────────────────────────

    def _execute_best_bomb(self, state,
                           pending_bombs: List[BombTask]
                           ) -> Optional[Tuple[List[int], BombTask]]:
        """选择并执行最优的炸弹引爆任务.

        选择标准: 车到推送位的距离最短的任务优先。
        """
        car_grid = pos_to_grid(state.car_x, state.car_y)
        obstacles = self._get_obstacles(state)
        obstacles_no_bombs = self._get_obstacles_no_bombs(state)

        best_actions = None
        best_task = None
        best_cost = float('inf')

        for task in pending_bombs:
            # 检查炸弹是否还存在 (可能已被其他爆炸消灭)
            bomb_still_exists = False
            for b in state.bombs:
                bg = pos_to_grid(b.x, b.y)
                if bg == task.bomb_grid:
                    bomb_still_exists = True
                    break

            if not bomb_still_exists:
                continue

            # 规划路径: 先含炸弹障碍, 失败则去掉炸弹重试
            dirs = plan_bomb_execution(
                car_grid, task, state.grid, obstacles)
            if dirs is None:
                dirs = plan_bomb_execution(
                    car_grid, task, state.grid, obstacles_no_bombs)
            if dirs is None:
                continue

            cost = len(dirs)
            if cost < best_cost:
                best_cost = cost
                best_task = task
                best_actions = [direction_to_action(dx, dy)
                                for dx, dy in dirs]

        if best_actions is None:
            return None

        return best_actions, best_task

    # ── 推箱求解 ───────────────────────────────────────────

    def _solve_one_push(self, state, box_idx: int,
                        target_idx: int
                        ) -> Optional[List[int]]:
        """求解将指定箱子推到指定目标的动作序列."""
        box = state.boxes[box_idx]
        target = state.targets[target_idx]

        car_grid = pos_to_grid(state.car_x, state.car_y)
        box_grid = pos_to_grid(box.x, box.y)
        target_grid = pos_to_grid(target.x, target.y)

        # 障碍物 = 所有其他箱子 + 所有炸弹
        obstacles = self._get_obstacles(state, exclude_box=box_idx)

        # BFS 推箱求解
        directions = bfs_push(car_grid, box_grid, target_grid,
                              state.grid, obstacles)

        if directions is None:
            # 回退: 把炸弹视为空气 (可被推开), 重新搜索
            obstacles_no_bombs = self._get_obstacles_no_bombs(
                state, exclude_box=box_idx)
            directions = bfs_push(car_grid, box_grid, target_grid,
                                  state.grid, obstacles_no_bombs)
            if directions is not None:
                print(f"    💡 忽略炸弹障碍后找到路径"
                      f" ({len(directions)} 步)")

        if directions is None:
            return None

        actions = [direction_to_action(dx, dy) for dx, dy in directions]
        return actions



    # ── 匹配与排序 ─────────────────────────────────────────

    def _match_pairs(self, state) -> List[Tuple[int, int]]:
        """匹配箱子与目标: class_id == num_id."""
        pairs = []
        for bi, box in enumerate(state.boxes):
            for ti, target in enumerate(state.targets):
                if box.class_id == target.num_id:
                    pairs.append((bi, ti))
        return pairs

    def _sort_pairs(self, pairs: List[Tuple[int, int]],
                    state) -> List[Tuple[int, int]]:
        """按推送成本排序配对."""
        def cost(pair):
            bi, ti = pair
            box = state.boxes[bi]
            target = state.targets[ti]
            c = estimate_push_cost(
                pos_to_grid(box.x, box.y),
                pos_to_grid(target.x, target.y)
            )
            car_grid = pos_to_grid(state.car_x, state.car_y)
            box_grid = pos_to_grid(box.x, box.y)
            c += abs(car_grid[0] - box_grid[0]) + \
                 abs(car_grid[1] - box_grid[1])
            return c

        return sorted(pairs, key=cost)

    # ── 辅助 ───────────────────────────────────────────────

    def _get_obstacles(self, state,
                       exclude_box: int = -1
                       ) -> Set[Tuple[int, int]]:
        """获取障碍物集合（其他箱子 + 炸弹）."""
        obstacles: Set[Tuple[int, int]] = set()
        for i, b in enumerate(state.boxes):
            if i != exclude_box:
                obstacles.add(pos_to_grid(b.x, b.y))
        for b in state.bombs:
            obstacles.add(pos_to_grid(b.x, b.y))
        return obstacles

    def _get_obstacles_no_bombs(self, state,
                                exclude_box: int = -1
                                ) -> Set[Tuple[int, int]]:
        """获取障碍物集合（仅其他箱子, 不含炸弹）.

        用于回退场景: 炸弹可被引擎推开, 不应阻断推箱规划。
        """
        obstacles: Set[Tuple[int, int]] = set()
        for i, b in enumerate(state.boxes):
            if i != exclude_box:
                obstacles.add(pos_to_grid(b.x, b.y))
        return obstacles

