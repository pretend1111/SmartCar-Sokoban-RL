"""手动游玩地图 — 简易渲染 + 离散操控.

用法:
    python play_failed.py --phase 1          # Phase 1
    python play_failed.py --phase 3 --idx 5  # Phase 3 第 6 张
    python play_failed.py --phase 2 --map phase2_0042.txt

快捷键:
    W/A/S/D — 前进/左移/后退/右移
    ←/→     — 左转/右转
    N / P   — 下一张/上一张
    R       — 重置当前地图
    Tab     — 切换渲染 (简易/3D)
    ESC     — 退出
"""

from __future__ import annotations

import argparse
import copy
import glob
import math
import os
import random
import sys

import pygame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine, GameState
from smartcar_sokoban.renderer import Renderer


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
    parser = argparse.ArgumentParser(description='手动游玩地图')
    parser.add_argument('--phase', type=int, default=1, help='阶段 (1-6)')
    parser.add_argument('--idx', type=int, default=0, help='起始地图索引')
    parser.add_argument('--map', type=str, default=None, help='指定地图文件名')
    args = parser.parse_args()

    phase_dir = os.path.join(ROOT, 'assets', 'maps', f'phase{args.phase}')
    map_list = sorted([f for f in glob.glob(os.path.join(phase_dir, '*.txt'))
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
    print("W/A/S/D=移动  ←/→=转向  N/P=翻页  R=重置  Tab=渲染  ESC=退出")

    cfg = GameConfig()
    cfg.render_mode = "simple"
    cfg.control_mode = "discrete"
    engine = GameEngine(cfg, ROOT)
    renderer = Renderer(cfg, ROOT)
    renderer.init()
    clock = pygame.time.Clock()

    animating = False
    anim_progress = 0.0
    prev_state = None
    state = None
    map_name = ""
    step_count = 0
    won = False

    def load():
        nonlocal state, prev_state, animating, anim_progress
        nonlocal map_name, step_count, won

        fpath = map_list[idx]
        map_name = os.path.basename(fpath)
        rel = os.path.relpath(fpath, ROOT).replace('\\', '/')
        random.seed(42)
        engine.reset(rel)
        state = engine.get_state()
        prev_state = None
        animating = False
        anim_progress = 0.0
        step_count = 0
        won = False
        print(f"\n  [{idx+1}/{len(map_list)}] {map_name}")

    load()

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
                elif event.key == pygame.K_r:
                    load()
                elif event.key == pygame.K_n:
                    idx = (idx + 1) % len(map_list); load()
                elif event.key == pygame.K_p:
                    idx = (idx - 1) % len(map_list); load()
                elif event.key == pygame.K_TAB:
                    m = "simple" if cfg.render_mode == "full" else "full"
                    renderer.switch_mode(m)

                elif not animating and not won:
                    action = None
                    if event.key == pygame.K_w: action = 0
                    elif event.key == pygame.K_s: action = 1
                    elif event.key == pygame.K_a: action = 2
                    elif event.key == pygame.K_d: action = 3
                    elif event.key == pygame.K_LEFT: action = 4
                    elif event.key == pygame.K_RIGHT: action = 5

                    if action is not None:
                        prev_state = copy.deepcopy(state)
                        state = engine.discrete_step(action)
                        animating = True
                        anim_progress = 0.0
                        step_count += 1

        # 动画
        if animating and prev_state is not None:
            anim_progress += cfg.discrete_anim_speed * dt
            if anim_progress >= 1.0:
                anim_progress = 1.0; animating = False
            render_state = lerp_state(prev_state, state, anim_progress)
        else:
            render_state = state

        # 通关检测
        if state and state.won and not won:
            won = True
            print(f"  🎉 通关！{step_count} 步")

        if render_state:
            renderer.render(render_state, engine.get_fov_rays())

        status = f"🎉通关" if won else f"步数:{step_count}"
        pygame.display.set_caption(
            f"游玩 | Phase {args.phase} | {map_name} "
            f"[{idx+1}/{len(map_list)}] {status}")

    renderer.close()


if __name__ == "__main__":
    main()
