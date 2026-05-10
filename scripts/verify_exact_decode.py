"""验证脚本: 对比 exact solver 的理论路径 vs 实际通过高层环境回放执行的路径.

目的:
    1. 用 MultiBoxSolver 求解 phase6_11 (或指定地图)
    2. 把 solver 的 solution 逐步解码成高层动作
    3. 通过 SokobanHLEnv 的 _execute_single_push 实际执行
    4. 对比两者: 步数一不一样, 有没有执行失败

用法:
    python scripts/verify_exact_decode.py
    python scripts/verify_exact_decode.py --map assets/maps/phase6/phase6_11.txt
    python scripts/verify_exact_decode.py --all   # 跑所有 phase 4-6 地图
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import random
import sys
from contextlib import redirect_stdout

# 确保项目根在 path 里
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from smartcar_sokoban.solver.explorer import (
    plan_exploration, direction_to_action, compute_facing_actions,
    exploration_complete, find_observation_point,
    get_entity_obstacles, get_all_entity_positions,
)
from smartcar_sokoban.solver.high_level_teacher import (
    map_solver_move_to_high_level_action,
    PUSH_BOX_START, PUSH_BOMB_START, EXPLORE_BOX_START, EXPLORE_TGT_START,
    N_DIRS, N_BOMB_DIRS, BOX_DIR_DELTAS, BOMB_DIR_DELTAS,
    MAX_BOXES, MAX_TARGETS,
)
from smartcar_sokoban.solver.pathfinder import bfs_path


def decode_high_level_action(action: int):
    """把高层动作 ID 解码成人类可读的描述."""
    if action < EXPLORE_BOX_START + MAX_BOXES:
        return f"EXPLORE_BOX[{action - EXPLORE_BOX_START}]"
    if action < EXPLORE_TGT_START + MAX_TARGETS:
        return f"EXPLORE_TGT[{action - EXPLORE_TGT_START}]"
    if action < PUSH_BOMB_START:
        offset = action - PUSH_BOX_START
        box_idx = offset // N_DIRS
        dir_idx = offset % N_DIRS
        dir_name = ["上", "下", "左", "右"][dir_idx]
        return f"PUSH_BOX[{box_idx}]_{dir_name}"
    offset = action - PUSH_BOMB_START
    bomb_idx = offset // N_BOMB_DIRS
    dir_idx = offset % N_BOMB_DIRS
    return f"PUSH_BOMB[{bomb_idx}]_DIR{dir_idx}"


def execute_push_manually(engine, entity_idx, dir_idx, etype):
    """手动执行一次推操作 (模拟 SokobanHLEnv._execute_single_push 的逻辑).

    返回 (步数, 是否成功, 失败原因)
    """
    state = engine.get_state()

    if etype == 'box':
        if entity_idx >= len(state.boxes):
            return 0, False, f"box idx {entity_idx} 越界 (共{len(state.boxes)})"
        ex, ey = state.boxes[entity_idx].x, state.boxes[entity_idx].y
    else:
        if entity_idx >= len(state.bombs):
            return 0, False, f"bomb idx {entity_idx} 越界 (共{len(state.bombs)})"
        ex, ey = state.bombs[entity_idx].x, state.bombs[entity_idx].y

    ec, er = pos_to_grid(ex, ey)
    dir_deltas = BOX_DIR_DELTAS if etype == 'box' else BOMB_DIR_DELTAS
    if not (0 <= dir_idx < len(dir_deltas)):
        return 0, False, f"dir_idx {dir_idx} 越界"
    dx, dy = dir_deltas[dir_idx]

    # 车需要站在实体的反方向
    car_target = (ec - dx, er - dy)
    car_grid = pos_to_grid(state.car_x, state.car_y)

    # 障碍物: 所有实体
    obstacles = set()
    for i, b in enumerate(state.boxes):
        obstacles.add(pos_to_grid(b.x, b.y))
    for i, b in enumerate(state.bombs):
        obstacles.add(pos_to_grid(b.x, b.y))

    steps = 0

    if car_grid != car_target:
        path = bfs_path(car_grid, car_target, state.grid, obstacles)
        if path is None:
            return 0, False, (
                f"BFS 导航失败: 车@{car_grid} -> 站位点@{car_target}, "
                f"实体@({ec},{er}), 方向({dx},{dy}), "
                f"障碍物={sorted(obstacles)}"
            )

        for pdx, pdy in path:
            a = direction_to_action(pdx, pdy)
            engine.discrete_step(a)
            steps += 1

    # 推: 车移动 → 撞到实体 → 推动
    a = direction_to_action(dx, dy)
    old_entity_x, old_entity_y = ex, ey

    new_state = engine.discrete_step(a)
    steps += 1

    # 检查是否推动了
    success = False
    if etype == 'box':
        if entity_idx < len(new_state.boxes):
            nb = new_state.boxes[entity_idx]
            success = (abs(nb.x - old_entity_x) > 0.01 or
                       abs(nb.y - old_entity_y) > 0.01)
        else:
            success = True  # 箱子到达目标被消除
    else:
        if entity_idx < len(new_state.bombs):
            nb = new_state.bombs[entity_idx]
            success = (abs(nb.x - old_entity_x) > 0.01 or
                       abs(nb.y - old_entity_y) > 0.01)
        else:
            success = True  # 炸弹引爆

    return steps, success, ("" if success else "推动未生效")


def verify_one_map(map_path: str, seed: int = 42, verbose: bool = True):
    """验证一张地图: exact solver -> 解码 -> 回放."""
    base_dir = PROJECT_ROOT
    cfg = GameConfig()
    rel_path = os.path.relpath(map_path, base_dir).replace('\\', '/')
    map_name = os.path.basename(map_path)

    # ── Step 1: 探索阶段 ──
    engine = GameEngine(cfg, base_dir)
    random.seed(seed)
    engine.reset(rel_path)

    devnull = io.StringIO()
    with redirect_stdout(devnull):
        explore_actions = plan_exploration(engine)
    explore_steps = len(explore_actions)

    state = engine.get_state()
    if verbose:
        print(f"\n{'='*60}")
        print(f"地图: {map_name}  seed={seed}")
        print(f"探索步数: {explore_steps}")
        print(f"箱子: {len(state.boxes)}, 目标: {len(state.targets)}, "
              f"炸弹: {len(state.bombs)}")

    # ── Step 2: exact solver 求解 ──
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)

    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    with redirect_stdout(devnull):
        solution = solver.solve(max_cost=1000, time_limit=60.0)

    if solution is None:
        if verbose:
            print(f"  ❌ exact solver 无解!")
        return {"map": map_name, "status": "solver_failed"}

    # solver 理论步数
    solver_pushes = len(solution)
    solver_total_steps = sum(wc + 1 for _, _, _, wc in solution)

    if verbose:
        print(f"\nsolver 输出: {solver_pushes} 次推操作, "
              f"理论总步数={solver_total_steps}")
        for i, (etype, eid, direction, wc) in enumerate(solution):
            if etype == "box":
                pos, cid = eid
                print(f"  [{i+1}] 推箱子 @{pos} (class={cid}) "
                      f"方向{direction} 导航{wc}步")
            else:
                print(f"  [{i+1}] 推炸弹 @{eid} "
                      f"方向{direction} 导航{wc}步")

    # ── Step 3 + 4: 重新初始化引擎, 逐步解码 + 实际回放 ──
    # 关键: 必须用实时状态来解码, 因为每次推操作都会改变实体位置
    engine2 = GameEngine(cfg, base_dir)
    random.seed(seed)
    engine2.reset(rel_path)

    # 先执行探索
    for a in explore_actions:
        engine2.discrete_step(a)

    actual_steps = explore_steps
    push_results = []

    if verbose:
        print(f"\n逐步解码 + 回放执行:")

    for i, move in enumerate(solution):
        # 用当前实时状态来解码
        current_state = engine2.get_state()
        hl = map_solver_move_to_high_level_action(current_state, move)
        etype_name, eid, direction, wc = move

        if hl is None:
            push_results.append(("decode_fail", 0, move))
            if verbose:
                print(f"  [{i+1}] ❌ 解码失败! "
                      f"move=({etype_name}, {eid}, {direction}, {wc})")
                # 打印调试信息
                if etype_name == "box":
                    old_pos, class_id = eid
                    print(f"        期望: box @{old_pos} class={class_id}")
                    print(f"        当前 boxes:")
                    for bi, b in enumerate(current_state.boxes):
                        bg = pos_to_grid(b.x, b.y)
                        print(f"          [{bi}] @{bg} class={b.class_id}")
                elif etype_name == "bomb":
                    print(f"        期望: bomb @{eid}")
                    print(f"        当前 bombs:")
                    for bi, b in enumerate(current_state.bombs):
                        bg = pos_to_grid(b.x, b.y)
                        print(f"          [{bi}] @{bg}")
            continue

        if verbose:
            print(f"  [{i+1}] 解码: {decode_high_level_action(hl)} (id={hl})")

        # 从高层动作反解 entity_idx, dir_idx, etype
        if hl < PUSH_BOX_START:
            push_results.append(("unexpected_explore", 0, move))
            if verbose:
                print(f"        ❌ 意外的探索动作")
            continue

        if hl < PUSH_BOMB_START:
            offset = hl - PUSH_BOX_START
            entity_idx = offset // N_DIRS
            dir_idx = offset % N_DIRS
            etype = 'box'
        else:
            offset = hl - PUSH_BOMB_START
            entity_idx = offset // N_BOMB_DIRS
            dir_idx = offset % N_BOMB_DIRS
            etype = 'bomb'

        steps, success, reason = execute_push_manually(
            engine2, entity_idx, dir_idx, etype)

        actual_steps += steps
        status = "ok" if success else "fail"
        push_results.append((status, steps, move))

        if verbose:
            mark = "✅" if success else "❌"
            expected_steps = wc + 1
            diff = steps - expected_steps
            diff_str = f" (差{diff:+d}步)" if diff != 0 else ""
            print(f"        {mark} 实际{steps}步 vs 理论{expected_steps}步"
                  f"{diff_str}"
                  f"{'  原因: ' + reason if reason else ''}")

    # ── 总结 ──
    final_state = engine2.get_state()
    won = final_state.won

    ok_count = sum(1 for s, _, _ in push_results if s == "ok")
    fail_count = sum(1 for s, _, _ in push_results if s == "fail")
    decode_fail_count = sum(1 for s, _, _ in push_results if s == "decode_fail")

    solver_expected_total = explore_steps + solver_total_steps

    result = {
        "map": map_name,
        "status": "pass" if (won and fail_count == 0 and decode_fail_count == 0) else "FAIL",
        "won": won,
        "solver_pushes": solver_pushes,
        "solver_total_steps": solver_expected_total,
        "actual_total_steps": actual_steps,
        "step_diff": actual_steps - solver_expected_total,
        "push_ok": ok_count,
        "push_fail": fail_count,
        "decode_fail": decode_fail_count,
    }

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  通关: {'✅ 是' if won else '❌ 否'}")
        print(f"  推操作: {ok_count} 成功 / {fail_count} 失败 / "
              f"{decode_fail_count} 解码失败")
        print(f"  solver 理论总步: {solver_expected_total}")
        print(f"  实际回放总步:    {actual_steps}")
        print(f"  步数差异:        {actual_steps - solver_expected_total:+d}")
        if result["status"] == "pass":
            print(f"  结论: ✅ 完美匹配!")
        else:
            print(f"  结论: ❌ 存在差异!")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='验证 exact solver -> 高层动作 + BFS 的解码正确性')
    parser.add_argument('--map', type=str, default=None,
                        help='指定地图路径')
    parser.add_argument('--all', action='store_true',
                        help='跑所有 phase 4-6 地图')
    parser.add_argument('--phase', type=int, default=None,
                        help='只跑指定 phase')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--quiet', action='store_true',
                        help='只输出总结')
    args = parser.parse_args()

    maps_root = os.path.join(PROJECT_ROOT, "assets", "maps")

    if args.map:
        map_path = os.path.join(PROJECT_ROOT, args.map) if not os.path.isabs(args.map) else args.map
        verify_one_map(map_path, seed=args.seed, verbose=not args.quiet)
    elif args.all or args.phase:
        phases = [args.phase] if args.phase else [4, 5, 6]
        all_results = []
        for phase in phases:
            phase_dir = os.path.join(maps_root, f"phase{phase}")
            map_files = sorted(glob.glob(os.path.join(phase_dir, "*.txt")))
            map_files = [f for f in map_files
                         if 'verify_' not in os.path.basename(f)]
            print(f"\n{'='*60}")
            print(f"Phase {phase}: {len(map_files)} 张地图")
            print(f"{'='*60}")

            for fpath in map_files:
                result = verify_one_map(fpath, seed=args.seed,
                                        verbose=not args.quiet)
                all_results.append(result)

        # 总结
        print(f"\n{'='*60}")
        print(f"总结")
        print(f"{'='*60}")
        pass_count = sum(1 for r in all_results if r["status"] == "pass")
        fail_count = sum(1 for r in all_results if r["status"] == "FAIL")
        solver_fail = sum(1 for r in all_results if r["status"] == "solver_failed")
        print(f"  总计: {len(all_results)} 张")
        print(f"  ✅ 完美匹配: {pass_count}")
        print(f"  ❌ 存在差异: {fail_count}")
        print(f"  ⚠️  solver 无解: {solver_fail}")

        if fail_count > 0:
            print(f"\n差异详情:")
            for r in all_results:
                if r["status"] == "FAIL":
                    print(f"  {r['map']}: "
                          f"通关={'是' if r.get('won') else '否'}, "
                          f"步数差={r.get('step_diff', '?')}, "
                          f"推失败={r.get('push_fail', 0)}, "
                          f"解码失败={r.get('decode_fail', 0)}")
    else:
        # 默认跑 phase6_11
        default_map = os.path.join(maps_root, "phase6", "phase6_11.txt")
        if os.path.exists(default_map):
            verify_one_map(default_map, seed=args.seed)
        else:
            print(f"默认地图不存在: {default_map}")
            print("请用 --map 指定地图路径, 或 --all 跑所有地图")


if __name__ == "__main__":
    main()
