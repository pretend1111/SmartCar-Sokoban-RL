"""人工游玩入口 — 支持连续 / 离散两种操控模式.

新增功能:
    --phase N       指定阶段 (1-6), 只加载该阶段的地图
    --idx N         起始地图索引 (0-based)
    --map FILE      直接指定地图文件名 (如 phase6_11.txt)
    --seed N        随机种子 (默认 42)
    --god           上帝模式: 开局显示所有编号, 无需探索
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import random
import sys

import pygame

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine, GameState
from smartcar_sokoban.paths import MAPS_ROOT, PROJECT_ROOT
from smartcar_sokoban.renderer import Renderer


# ── 离散模式动画插值 ──────────────────────────────────────

def lerp_state(prev: GameState, cur: GameState, t: float) -> GameState:
    """在 prev 和 cur 之间线性插值生成过渡帧状态."""
    s = copy.deepcopy(cur)

    s.car_x = prev.car_x + (cur.car_x - prev.car_x) * t
    s.car_y = prev.car_y + (cur.car_y - prev.car_y) * t

    angle_diff = cur.car_angle - prev.car_angle
    angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
    s.car_angle = prev.car_angle + angle_diff * t

    n = min(len(prev.boxes), len(cur.boxes))
    for i in range(n):
        s.boxes[i].x = prev.boxes[i].x + (cur.boxes[i].x - prev.boxes[i].x) * t
        s.boxes[i].y = prev.boxes[i].y + (cur.boxes[i].y - prev.boxes[i].y) * t

    n = min(len(prev.bombs), len(cur.bombs))
    for i in range(n):
        s.bombs[i].x = prev.bombs[i].x + (cur.bombs[i].x - prev.bombs[i].x) * t
        s.bombs[i].y = prev.bombs[i].y + (cur.bombs[i].y - prev.bombs[i].y) * t

    return s


def reveal_all(state: GameState):
    """上帝模式: 将所有箱子和目标标记为已看到."""
    state.seen_box_ids = set(range(len(state.boxes)))
    state.seen_target_ids = set(range(len(state.targets)))


# ── 主函数 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="推箱子 — 人工游玩")
    parser.add_argument("--phase", type=int, default=0,
                        help="阶段 (1-6), 0=加载所有地图")
    parser.add_argument("--idx", type=int, default=0,
                        help="起始地图索引 (0-based)")
    parser.add_argument("--map", type=str, default="",
                        help="直接指定地图文件名 (如 phase6_11.txt)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认 42)")
    parser.add_argument("--god", action="store_true",
                        help="上帝模式: 开局显示所有编号")
    args = parser.parse_args()

    base_dir = str(PROJECT_ROOT)
    cfg = GameConfig()
    engine = GameEngine(cfg, base_dir)
    renderer = Renderer(cfg, base_dir)

    # ── 收集地图文件 ──
    if args.map:
        # 直接指定文件名
        candidates = glob.glob(str(MAPS_ROOT / "**" / args.map), recursive=True)
        if not candidates:
            candidates = glob.glob(str(MAPS_ROOT / args.map))
        if not candidates:
            print(f"错误: 找不到地图 {args.map}")
            sys.exit(1)
        map_files = sorted(candidates)
    elif args.phase > 0:
        pattern = str(MAPS_ROOT / f"phase{args.phase}" / "*.txt")
        map_files = sorted(glob.glob(pattern))
        if not map_files:
            print(f"错误: Phase {args.phase} 没有地图")
            sys.exit(1)
    else:
        # 全部地图, 按 phase 排序
        map_files = []
        for p in range(1, 7):
            pattern = str(MAPS_ROOT / f"phase{p}" / "*.txt")
            map_files.extend(sorted(glob.glob(pattern)))
        if not map_files:
            map_files = sorted(glob.glob(str(MAPS_ROOT / "*.txt")))
        if not map_files:
            print("错误: 没有找到地图文件")
            sys.exit(1)

    # ── 加载 seed manifest ──
    manifest_path = str(MAPS_ROOT / "phase456_seed_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    current_map_idx = min(args.idx, len(map_files) - 1)
    god_mode = args.god
    steps = 0

    def load_map(idx: int):
        nonlocal current_map_idx, animating, steps
        current_map_idx = idx % len(map_files)
        rel_path = os.path.relpath(map_files[current_map_idx], base_dir).replace("\\", "/")
        map_name = os.path.basename(map_files[current_map_idx])

        # seed: 优先 manifest, 其次命令行
        seed = args.seed
        if map_name in manifest:
            seed = manifest[map_name].get("seed", seed)
        random.seed(seed)

        state = engine.reset(rel_path)
        animating = False
        steps = 0

        if god_mode:
            reveal_all(state)

        phase_name = os.path.basename(os.path.dirname(
            map_files[current_map_idx]))
        god_tag = " [GOD]" if god_mode else ""
        print(f"已加载地图: {phase_name}/{map_name} (seed={seed}){god_tag}"
              f"  箱子={len(state.boxes)} 目标={len(state.targets)}"
              f" 炸弹={len(state.bombs)}")
        return state

    state = load_map(current_map_idx)
    renderer.init()
    clock = pygame.time.Clock()

    # 离散模式动画状态
    animating = False
    anim_progress = 0.0
    prev_state: GameState = None

    print("\n=== 推箱子游戏 ===")
    print("操控: W/A/S/D 移动, ←/→ 转向")
    print("切图: N=下一张, P=上一张, 1-9=跳转")
    print("功能: R=重新开始, G=切换上帝模式")
    print("      Tab=切换渲染, M=切换操控, ESC=退出")
    god_str = "ON ✅" if god_mode else "OFF"
    print(f"当前: 渲染={cfg.render_mode} | 操控={cfg.control_mode}"
          f" | 上帝模式={god_str}")
    print(f"地图: {len(map_files)} 张可用\n")

    running = True
    while running:
        dt = clock.tick(cfg.fps) / 1000.0
        dt = min(dt, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # ── 全局快捷键 ──
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    state = load_map(current_map_idx)

                elif event.key == pygame.K_n:
                    state = load_map(current_map_idx + 1)

                elif event.key == pygame.K_p:
                    state = load_map(current_map_idx - 1)

                elif event.key == pygame.K_g:
                    god_mode = not god_mode
                    tag = "ON ✅" if god_mode else "OFF"
                    print(f"上帝模式: {tag}")
                    if god_mode:
                        reveal_all(state)

                elif event.key == pygame.K_TAB:
                    new_mode = ("simple" if cfg.render_mode == "full"
                                else "full")
                    renderer.switch_mode(new_mode)
                    print(f"渲染模式: {new_mode}")

                elif event.key == pygame.K_m:
                    cfg.control_mode = ("discrete"
                                        if cfg.control_mode == "continuous"
                                        else "continuous")
                    animating = False
                    print(f"操控模式: {cfg.control_mode}")

                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3,
                                   pygame.K_4, pygame.K_5, pygame.K_6,
                                   pygame.K_7, pygame.K_8, pygame.K_9):
                    idx = event.key - pygame.K_1
                    if idx < len(map_files):
                        state = load_map(idx)

                # ── 离散模式: 按键触发动作 ──
                if cfg.control_mode == "discrete" and not animating:
                    action = None
                    if event.key == pygame.K_w:
                        action = 0
                    elif event.key == pygame.K_s:
                        action = 1
                    elif event.key == pygame.K_a:
                        action = 2
                    elif event.key == pygame.K_d:
                        action = 3
                    elif event.key == pygame.K_LEFT:
                        action = 4
                    elif event.key == pygame.K_RIGHT:
                        action = 5

                    if action is not None:
                        prev_state = copy.deepcopy(state)
                        state = engine.discrete_step(action)
                        steps += 1
                        animating = True
                        anim_progress = 0.0

                        if god_mode:
                            reveal_all(state)

                        if state.won:
                            print(f"🎉 通关! 总步数: {steps}")

        # ── 更新逻辑 ──
        if cfg.control_mode == "continuous":
            keys = pygame.key.get_pressed()
            forward = strafe = turn = 0.0
            if keys[pygame.K_w]: forward += 1.0
            if keys[pygame.K_s]: forward -= 1.0
            if keys[pygame.K_a]: strafe -= 1.0
            if keys[pygame.K_d]: strafe += 1.0
            if keys[pygame.K_LEFT]: turn -= 1.0
            if keys[pygame.K_RIGHT]: turn += 1.0
            state = engine.step(forward, strafe, turn, dt)
            if god_mode:
                reveal_all(state)
            render_state = state
        else:
            if animating and prev_state is not None:
                anim_progress += cfg.discrete_anim_speed * dt
                if anim_progress >= 1.0:
                    anim_progress = 1.0
                    animating = False
                render_state = lerp_state(prev_state, state, anim_progress)
            else:
                render_state = state

        # ── 渲染 ──
        fov_rays = (engine.get_fov_rays()
                    if cfg.render_mode == "simple" else None)
        renderer.render(render_state, fov_rays)

        # 窗口标题显示步数和地图名
        map_name = os.path.basename(map_files[current_map_idx])
        god_tag = " [GOD]" if god_mode else ""
        title = (f"{map_name} | 步数: {steps}"
                 f" | {current_map_idx+1}/{len(map_files)}{god_tag}")
        pygame.display.set_caption(title)

    renderer.close()
    print(f"\n最终步数: {steps}")


if __name__ == "__main__":
    main()
