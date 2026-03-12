"""自动求解入口 — 加载地图, 运行 BFS 求解器, 可视化过程."""

from __future__ import annotations

import copy
import glob
import math
import os
import random
import sys
import time

import pygame

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine, GameState
from smartcar_sokoban.paths import MAPS_ROOT, PROJECT_ROOT
from smartcar_sokoban.renderer import Renderer
from smartcar_sokoban.solver.auto_player import AutoPlayer


def lerp_state(prev: GameState, cur: GameState, t: float) -> GameState:
    """在 prev 和 cur 之间线性插值（用于动画）."""
    s = copy.deepcopy(cur)
    s.car_x = prev.car_x + (cur.car_x - prev.car_x) * t
    s.car_y = prev.car_y + (cur.car_y - prev.car_y) * t

    angle_diff = cur.car_angle - prev.car_angle
    angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
    s.car_angle = prev.car_angle + angle_diff * t

    n = min(len(prev.boxes), len(cur.boxes))
    for i in range(n):
        s.boxes[i].x = prev.boxes[i].x + \
            (cur.boxes[i].x - prev.boxes[i].x) * t
        s.boxes[i].y = prev.boxes[i].y + \
            (cur.boxes[i].y - prev.boxes[i].y) * t

    n = min(len(prev.bombs), len(cur.bombs))
    for i in range(n):
        s.bombs[i].x = prev.bombs[i].x + \
            (cur.bombs[i].x - prev.bombs[i].x) * t
        s.bombs[i].y = prev.bombs[i].y + \
            (cur.bombs[i].y - prev.bombs[i].y) * t

    return s


def main():
    base_dir = str(PROJECT_ROOT)

    # ── 配置 ──────────────────────────────────────────
    cfg = GameConfig()
    cfg.render_mode = "simple"      # 使用简易渲染
    cfg.control_mode = "discrete"   # 离散模式

    # 命令行参数: python auto_solve.py [地图编号]
    map_idx = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0

    map_files = sorted(glob.glob(str(MAPS_ROOT / "*.txt")))
    if not map_files:
        print("错误: assets/maps/ 目录中没有找到地图文件")
        sys.exit(1)

    # 过滤掉 _temp 开头的文件
    map_files = [f for f in map_files
                 if not os.path.basename(f).startswith('_')]

    map_idx = min(map_idx, len(map_files) - 1)
    map_name = os.path.basename(map_files[map_idx])
    print(f"加载地图: {map_name}")

    # ── 初始化引擎 ─────────────────────────────────────
    engine = GameEngine(cfg, base_dir)
    rel_path = os.path.relpath(map_files[map_idx], base_dir)

    # 固定随机种子，确保求解和回放时编号一致
    map_seed = random.randint(0, 2**32 - 1)
    random.seed(map_seed)
    engine.reset(rel_path)

    # ── Phase 1+2: 求解 ────────────────────────────────
    print()
    player = AutoPlayer(engine)
    t0 = time.perf_counter()
    actions = player.solve()
    solve_time = time.perf_counter() - t0
    print(f"\n求解耗时: {solve_time*1000:.1f} ms")
    print(f"总动作数: {len(actions)}")

    # ── 重放可视化 ─────────────────────────────────────
    print("\n按 SPACE 开始回放, ESC 退出, ←/→ 调速")
    random.seed(map_seed)   # 恢复相同种子
    engine.reset(rel_path)  # 重置引擎（编号与求解时一致）

    renderer = Renderer(cfg, base_dir)
    renderer.init()
    clock = pygame.time.Clock()

    # 回放状态
    action_idx = 0
    playing = False
    step_delay = 0.15  # 每步延时（秒）
    last_step_time = 0.0

    # 动画插值
    animating = False
    anim_progress = 0.0
    prev_state = None
    state = engine.get_state()

    running = True
    while running:
        dt = clock.tick(cfg.fps) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                    print("▶ 播放" if playing else "⏸ 暂停")
                elif event.key == pygame.K_LEFT:
                    step_delay = min(1.0, step_delay + 0.05)
                    print(f"速度: {1/step_delay:.1f} 步/秒")
                elif event.key == pygame.K_RIGHT:
                    step_delay = max(0.02, step_delay - 0.05)
                    print(f"速度: {1/step_delay:.1f} 步/秒")
                elif event.key == pygame.K_r:
                    # 重置回放（使用相同种子）
                    random.seed(map_seed)
                    engine.reset(rel_path)
                    state = engine.get_state()
                    action_idx = 0
                    playing = False
                    animating = False
                    print("↺ 重置回放")

        # ── 回放逻辑 ──────────────────────────────────
        now = time.perf_counter()
        if playing and not animating and action_idx < len(actions):
            if now - last_step_time >= step_delay:
                prev_state = copy.deepcopy(state)
                state = engine.discrete_step(actions[action_idx])
                action_idx += 1
                animating = True
                anim_progress = 0.0
                last_step_time = now

        if animating and prev_state is not None:
            anim_progress += cfg.discrete_anim_speed * dt
            if anim_progress >= 1.0:
                anim_progress = 1.0
                animating = False
            render_state = lerp_state(prev_state, state, anim_progress)
        else:
            render_state = state

        if action_idx >= len(actions) and not animating and playing:
            playing = False
            if state.won:
                print("🎉 回放完成 - 通关成功!")
            else:
                print("⚠️ 回放完成 - 未通关")

        # ── 渲染 ──────────────────────────────────────
        fov_rays = engine.get_fov_rays()
        renderer.render(render_state, fov_rays)

        # 显示进度
        pygame.display.set_caption(
            f"推箱子 BFS 求解器 | {map_name} | "
            f"步骤 {action_idx}/{len(actions)} | "
            f"{'▶' if playing else '⏸'}"
        )

    renderer.close()


if __name__ == "__main__":
    main()
