"""地图预览 — 观看 AutoPlayer 自动求解任意地图.

用法:
    python preview_failed.py --phase 1          # Phase 1
    python preview_failed.py --phase 3 --idx 5  # Phase 3 第 6 张
    python preview_failed.py --phase 2 --map phase2_0042.txt

快捷键:
    SPACE  — 播放/暂停
    N / P  — 下一张/上一张
    R      — 重新求解
    ←/→    — 调整回放速度
    Tab    — 切换渲染 (简易/3D)
    ESC    — 退出
"""

from __future__ import annotations

import argparse
import copy
import glob
import io
import json
import math
import os
import random
import time
from contextlib import redirect_stdout

import pygame

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine, GameState
from smartcar_sokoban.paths import MAPS_ROOT, PROJECT_ROOT
from smartcar_sokoban.renderer import Renderer
from smartcar_sokoban.solver.auto_player import AutoPlayer
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.explorer import direction_to_action
from smartcar_sokoban.solver.pathfinder import pos_to_grid


def load_seed_manifest():
    path = MAPS_ROOT / "phase456_seed_manifest.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get('phases', {})
    except Exception:
        return {}


def directions_to_discrete_actions(directions):
    """将网格方向序列转换成求解器回放动作序列."""
    actions = [6]  # 首步用于把车吸附到网格中心

    for dx, dy in directions:
        actions.append(direction_to_action(dx, dy))

    return actions


def solve_exact(engine):
    from smartcar_sokoban.solver.explorer import plan_exploration

    # ── Phase 1: 先探索（和 auto 一样，不是上帝视角） ──────
    explore_actions = plan_exploration(engine)
    print(f"    探索完成, {len(explore_actions)} 步")

    # ── Phase 2: 探索完毕后，取当前状态给精确求解器 ────────
    state = engine.get_state()
    boxes = [((int(box.x), int(box.y)), box.class_id) for box in state.boxes]
    targets = {target.num_id: (int(target.x), int(target.y)) for target in state.targets}
    bombs = [(int(bomb.x), int(bomb.y)) for bomb in state.bombs]
    car_pos = pos_to_grid(state.car_x, state.car_y)
    print(f"    车位={car_pos}, {len(boxes)}箱 {len(targets)}目标 {len(bombs)}炸弹")

    solver = MultiBoxSolver(
        grid=state.grid,
        car_pos=car_pos,
        boxes=boxes,
        targets=targets,
        bombs=bombs,
    )
    solution = solver.solve(max_cost=1000, time_limit=60.0)
    if solution is None:
        print("    ❌ MultiBoxSolver 无解")
        return None
    directions = solver.solution_to_actions(solution)
    # 合并探索动作 + 求解动作 (探索结束后车已在网格上，无需再 snap)
    solve_actions = [direction_to_action(dx, dy) for dx, dy in directions]
    all_actions = list(explore_actions) + solve_actions
    solve_steps = sum(wc + 1 for _, _, _, wc in solution)
    total_steps = len(explore_actions) + solve_steps
    print(f"    ✅ 求解成功: 探索{len(explore_actions)}步 + 求解{solve_steps}步 = 总{total_steps}步")
    return all_actions, total_steps


def lerp_state(prev: GameState, cur: GameState, t: float) -> GameState:
    s = copy.deepcopy(cur)
    s.car_x = prev.car_x + (cur.car_x - prev.car_x) * t
    s.car_y = prev.car_y + (cur.car_y - prev.car_y) * t
    ad = cur.car_angle - prev.car_angle
    ad = math.atan2(math.sin(ad), math.cos(ad))
    s.car_angle = prev.car_angle + ad * t
    for i in range(min(len(prev.boxes), len(cur.boxes))):
        s.boxes[i].x = prev.boxes[i].x + (cur.boxes[i].x - prev.boxes[i].x) * t
        s.boxes[i].y = prev.boxes[i].y + (cur.boxes[i].y - prev.boxes[i].y) * t
    for i in range(min(len(prev.bombs), len(cur.bombs))):
        s.bombs[i].x = prev.bombs[i].x + (cur.bombs[i].x - prev.bombs[i].x) * t
        s.bombs[i].y = prev.bombs[i].y + (cur.bombs[i].y - prev.bombs[i].y) * t
    return s


def main():
    parser = argparse.ArgumentParser(description='预览地图 (AutoPlayer 自动求解)')
    parser.add_argument('--phase', type=int, default=1, help='阶段 (1-6)')
    parser.add_argument('--idx', type=int, default=0, help='起始地图索引')
    parser.add_argument('--map', type=str, default=None, help='指定地图文件名')
    parser.add_argument('--seed', type=int, default=None, help='指定地图编号随机种子；默认优先读取 manifest')
    parser.add_argument('--solver', choices=['auto', 'exact'], default='auto', help='回放使用的求解器')
    args = parser.parse_args()

    phase_dir = MAPS_ROOT / f"phase{args.phase}"
    map_list = sorted([f for f in glob.glob(str(phase_dir / "*.txt"))
                       if 'verify_' not in os.path.basename(f)])
    if not map_list:
        print(f"❌ assets/maps/phase{args.phase}/ 无地图"); return

    idx = args.idx
    if args.map:
        for i, f in enumerate(map_list):
            if os.path.basename(f) == args.map:
                idx = i; break
    idx = min(idx, len(map_list) - 1)

    print(f"Phase {args.phase}: {len(map_list)} 张地图")
    print("SPACE=播放/暂停  N/P=翻页  R=重新求解  ←/→=调速  Tab=渲染  ESC=退出")

    cfg = GameConfig()
    cfg.render_mode = "simple"
    cfg.control_mode = "discrete"
    engine = GameEngine(cfg, str(PROJECT_ROOT))
    renderer = Renderer(cfg, str(PROJECT_ROOT))
    renderer.init()
    clock = pygame.time.Clock()
    seed_manifest = load_seed_manifest()

    actions = []
    act_idx = 0
    playing = False
    animating = False
    anim_progress = 0.0
    prev_state = None
    state = None
    map_name = ""
    current_seed = None
    current_solver = args.solver
    step_delay = 0.10
    last_step_time = 0.0

    def solve_and_load():
        nonlocal actions, act_idx, playing, animating, anim_progress
        nonlocal prev_state, state, map_name, idx, current_seed, current_solver

        fpath = map_list[idx]
        map_name = os.path.basename(fpath)
        rel = os.path.relpath(fpath, PROJECT_ROOT).replace('\\', '/')
        manifest_seed = None
        phase_key = f'phase{args.phase}'
        if phase_key in seed_manifest:
            item = seed_manifest[phase_key].get(map_name)
            if item:
                seeds = item.get('verified_seeds') or []
                if seeds:
                    manifest_seed = int(seeds[0])
        current_seed = args.seed if args.seed is not None else (manifest_seed if manifest_seed is not None else 42)
        current_solver = args.solver

        # 求解
        random.seed(current_seed)
        engine.reset(rel)
        if args.solver == 'exact':
            result = solve_exact(engine)
            actions = result[0] if result is not None else []
        else:
            devnull = io.StringIO()
            with redirect_stdout(devnull):
                player = AutoPlayer(engine)
                actions_result = player.solve()
            actions = actions_result or []
            if not engine.get_state().won:
                random.seed(current_seed)
                engine.reset(rel)
                exact_result = solve_exact(engine)
                if exact_result is not None:
                    actions = exact_result[0]
                    current_solver = 'auto->exact'
                else:
                    current_solver = 'auto'

        # 重置用于回放
        random.seed(current_seed)
        engine.reset(rel)
        state = engine.get_state()
        prev_state = None
        act_idx = 0
        playing = False
        animating = False
        anim_progress = 0.0

        print(
            f"\n  [{idx+1}/{len(map_list)}] {map_name} | "
            f"seed={current_seed} | solver={current_solver} | {len(actions)} 步"
        )

    solve_and_load()

    running = True
    while running:
        dt = clock.tick(cfg.fps) / 1000.0
        dt = min(dt, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_n:
                    idx = (idx + 1) % len(map_list); solve_and_load()
                elif event.key == pygame.K_p:
                    idx = (idx - 1) % len(map_list); solve_and_load()
                elif event.key == pygame.K_r:
                    solve_and_load()
                elif event.key == pygame.K_LEFT:
                    step_delay = min(1.0, step_delay + 0.03)
                elif event.key == pygame.K_RIGHT:
                    step_delay = max(0.02, step_delay - 0.03)
                elif event.key == pygame.K_TAB:
                    m = "simple" if cfg.render_mode == "full" else "full"
                    renderer.switch_mode(m)

        # 回放
        now = time.perf_counter()
        if playing and not animating and act_idx < len(actions):
            if now - last_step_time >= step_delay:
                prev_state = copy.deepcopy(state)
                state = engine.discrete_step(actions[act_idx])
                act_idx += 1
                animating = True
                anim_progress = 0.0
                last_step_time = now

        if act_idx >= len(actions) and not animating and playing:
            playing = False
            print("  🎉 通关！" if state.won else f"  ❌ 失败 (剩余 {len(state.boxes)} 箱)")

        # 动画
        if animating and prev_state is not None:
            anim_progress += cfg.discrete_anim_speed * dt
            if anim_progress >= 1.0:
                anim_progress = 1.0; animating = False
            render_state = lerp_state(prev_state, state, anim_progress)
        else:
            render_state = state

        if render_state:
            renderer.render(render_state, engine.get_fov_rays())

        pygame.display.set_caption(
            f"预览 | Phase {args.phase} | {map_name} "
            f"[{idx+1}/{len(map_list)}] "
            f"seed={current_seed} "
            f"solver={current_solver} "
            f"{'▶' if playing else '⏸'} {act_idx}/{len(actions)}")

    renderer.close()


if __name__ == "__main__":
    main()
