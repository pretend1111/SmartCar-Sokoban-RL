"""可视化对比: exact solver 原始路径 vs 高层决策+BFS 解码路径.

左半屏: exact solver 通过 solution_to_actions() 直接回放
右半屏: solver 方案逐步解码为高层动作, 再通过 BFS 执行

操作:
    SPACE       播放/暂停
    LEFT/RIGHT  上一张/下一张地图
    UP/DOWN     加速/减速
    R           重置当前地图
    S           跳过当前 (solver无解时)
    ESC         退出

用法:
    python scripts/visualize_decode_comparison.py
    python scripts/visualize_decode_comparison.py --phase 6
    python scripts/visualize_decode_comparison.py --timeout 30
"""

from __future__ import annotations

import argparse
import copy
import glob
import io
import json
import os
import random
import sys
import time
from contextlib import redirect_stdout
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pygame

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid, bfs_path
from smartcar_sokoban.solver.explorer import (
    plan_exploration, direction_to_action, compute_facing_actions,
    find_observation_point, get_entity_obstacles, get_all_entity_positions,
)
from smartcar_sokoban.solver.high_level_teacher import (
    map_solver_move_to_high_level_action,
    PUSH_BOX_START, PUSH_BOMB_START,
    N_DIRS, N_BOMB_DIRS, BOX_DIR_DELTAS, BOMB_DIR_DELTAS,
    MAX_BOXES, MAX_TARGETS,
)
from smartcar_sokoban.action_defs import ABS_WORLD_MOVE_TO_ACTION


# ── 颜色 ──────────────────────────────────────────────────
BLACK      = (20, 20, 30)
WALL_COLOR = (60, 65, 80)
FLOOR_COLOR = (200, 205, 215)
GRID_LINE  = (170, 175, 185)
CAR_COLOR  = (50, 180, 80)
BOX_COLORS = [
    (230, 100, 100), (100, 150, 230), (230, 200, 60),
    (180, 100, 230), (100, 230, 180),
]
TARGET_COLORS = [
    (255, 150, 150), (150, 190, 255), (255, 230, 120),
    (220, 160, 255), (150, 255, 220),
]
BOMB_COLOR = (255, 140, 50)
PATH_EXACT = (50, 200, 100, 120)   # 绿色半透明
PATH_DECODE = (100, 150, 255, 120)  # 蓝色半透明
PANEL_BG   = (30, 32, 40)
TEXT_COLOR  = (220, 225, 235)
ACCENT     = (100, 200, 255)
SUCCESS_COLOR = (80, 230, 120)
FAIL_COLOR = (255, 100, 100)
WARN_COLOR = (255, 200, 80)


def direction_to_engine_action(dx: int, dy: int) -> int:
    """方向 -> 引擎绝对动作 ID."""
    return ABS_WORLD_MOVE_TO_ACTION.get((dx, dy), 6)


def record_exact_path(engine, explore_actions, solution, solver):
    """用 exact solver 的 solution_to_actions 直接回放, 记录车轨迹."""
    positions = []
    state = engine.get_state()
    positions.append(pos_to_grid(state.car_x, state.car_y))

    # 先执行探索
    for a in explore_actions:
        state = engine.discrete_step(a)
        positions.append(pos_to_grid(state.car_x, state.car_y))

    # 用 solver 内部的 solution_to_actions 得到原始方向序列
    devnull = io.StringIO()
    with redirect_stdout(devnull):
        dirs = solver.solution_to_actions(solution)

    for dx, dy in dirs:
        a = direction_to_engine_action(dx, dy)
        state = engine.discrete_step(a)
        positions.append(pos_to_grid(state.car_x, state.car_y))

    return positions, state.won


def record_decoded_path(engine, explore_actions, solution):
    """用解码后的高层动作 + BFS 回放, 记录车轨迹."""
    positions = []
    state = engine.get_state()
    positions.append(pos_to_grid(state.car_x, state.car_y))

    # 先执行探索
    for a in explore_actions:
        state = engine.discrete_step(a)
        positions.append(pos_to_grid(state.car_x, state.car_y))

    decode_ok = 0
    decode_fail = 0
    push_ok = 0
    push_fail = 0

    for move in solution:
        current_state = engine.get_state()
        hl = map_solver_move_to_high_level_action(current_state, move)

        if hl is None:
            decode_fail += 1
            continue

        decode_ok += 1

        # 解析高层动作
        if hl < PUSH_BOX_START:
            continue  # 不应该出现
        if hl < PUSH_BOMB_START:
            offset = hl - PUSH_BOX_START
            entity_idx = offset // N_DIRS
            dir_idx = offset % N_DIRS
            etype = 'box'
            dir_deltas = BOX_DIR_DELTAS
        else:
            offset = hl - PUSH_BOMB_START
            entity_idx = offset // N_BOMB_DIRS
            dir_idx = offset % N_BOMB_DIRS
            etype = 'bomb'
            dir_deltas = BOMB_DIR_DELTAS

        state = engine.get_state()

        # 获取实体位置
        if etype == 'box':
            if entity_idx >= len(state.boxes):
                push_fail += 1
                continue
            ex, ey = state.boxes[entity_idx].x, state.boxes[entity_idx].y
        else:
            if entity_idx >= len(state.bombs):
                push_fail += 1
                continue
            ex, ey = state.bombs[entity_idx].x, state.bombs[entity_idx].y

        ec, er = pos_to_grid(ex, ey)
        dx, dy = dir_deltas[dir_idx]
        car_target = (ec - dx, er - dy)
        car_grid = pos_to_grid(state.car_x, state.car_y)

        # BFS 导航到站位点
        obstacles = set()
        for b in state.boxes:
            obstacles.add(pos_to_grid(b.x, b.y))
        for b in state.bombs:
            obstacles.add(pos_to_grid(b.x, b.y))

        if car_grid != car_target:
            path = bfs_path(car_grid, car_target, state.grid, obstacles)
            if path is None:
                push_fail += 1
                continue
            for pdx, pdy in path:
                a = direction_to_engine_action(pdx, pdy)
                state = engine.discrete_step(a)
                positions.append(pos_to_grid(state.car_x, state.car_y))

        # 推
        a = direction_to_engine_action(dx, dy)
        old_ex, old_ey = ex, ey
        state = engine.discrete_step(a)
        positions.append(pos_to_grid(state.car_x, state.car_y))

        # 检查推动是否成功
        if etype == 'box':
            if entity_idx < len(state.boxes):
                nb = state.boxes[entity_idx]
                if abs(nb.x - old_ex) > 0.01 or abs(nb.y - old_ey) > 0.01:
                    push_ok += 1
                else:
                    push_fail += 1
            else:
                push_ok += 1
        else:
            if entity_idx < len(state.bombs):
                nb = state.bombs[entity_idx]
                if abs(nb.x - old_ex) > 0.01 or abs(nb.y - old_ey) > 0.01:
                    push_ok += 1
                else:
                    push_fail += 1
            else:
                push_ok += 1

    final = engine.get_state()
    return positions, final.won, decode_ok, decode_fail, push_ok, push_fail


def get_seed_manifest():
    manifest_path = os.path.join(PROJECT_ROOT, "assets", "maps",
                                 "phase456_seed_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f).get("phases", {})
    return {}


def get_seed_for_map(manifest, phase, map_name, default=42):
    phase_key = f"phase{phase}"
    if phase_key in manifest:
        item = manifest[phase_key].get(map_name)
        if item:
            seeds = item.get("verified_seeds", [])
            if seeds:
                return int(seeds[0])
    return default


def precompute_map(map_path, seed, timeout):
    """预计算一张地图: 探索 + solver + 两条路径."""
    cfg = GameConfig()
    base_dir = PROJECT_ROOT
    rel = os.path.relpath(map_path, base_dir).replace('\\', '/')
    name = os.path.basename(map_path)

    # ── 探索 ──
    engine = GameEngine(cfg, base_dir)
    random.seed(seed)
    engine.reset(rel)

    devnull = io.StringIO()
    with redirect_stdout(devnull):
        explore_actions = plan_exploration(engine)

    state = engine.get_state()
    grid_snapshot = copy.deepcopy(state.grid)

    # ── solver ──
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)

    # 保存初始实体位置 (用于绘制)
    init_boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    init_targets = [(pos_to_grid(t.x, t.y), t.num_id) for t in state.targets]
    init_bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]

    solver = MultiBoxSolver(grid_snapshot, car, boxes, targets, bombs)
    with redirect_stdout(devnull):
        solution = solver.solve(max_cost=1000, time_limit=timeout)

    if solution is None:
        return {
            "name": name, "grid": grid_snapshot,
            "init_boxes": init_boxes, "init_targets": init_targets,
            "init_bombs": init_bombs, "init_car": car,
            "solved": False,
        }

    solver_steps = sum(wc + 1 for _, _, _, wc in solution)
    solver_pushes = len(solution)

    # ── 录 exact 路径 ──
    engine1 = GameEngine(cfg, base_dir)
    random.seed(seed)
    engine1.reset(rel)
    exact_path, exact_won = record_exact_path(
        engine1, explore_actions, solution, solver)

    # ── 录 decoded 路径 ──
    engine2 = GameEngine(cfg, base_dir)
    random.seed(seed)
    engine2.reset(rel)
    decode_path, decode_won, d_ok, d_fail, p_ok, p_fail = \
        record_decoded_path(engine2, explore_actions, solution)

    return {
        "name": name,
        "grid": grid_snapshot,
        "init_boxes": init_boxes,
        "init_targets": init_targets,
        "init_bombs": init_bombs,
        "init_car": car,
        "solved": True,
        "solver_pushes": solver_pushes,
        "solver_steps": solver_steps,
        "explore_steps": len(explore_actions),
        "exact_path": exact_path,
        "exact_won": exact_won,
        "exact_total": len(exact_path) - 1,
        "decode_path": decode_path,
        "decode_won": decode_won,
        "decode_total": len(decode_path) - 1,
        "decode_ok": d_ok,
        "decode_fail": d_fail,
        "push_ok": p_ok,
        "push_fail": p_fail,
    }


# ── 渲染 ──────────────────────────────────────────────────

def draw_grid(surface, grid, x_off, y_off, cell_size,
              init_boxes, init_targets, init_bombs, init_car,
              path, path_color, step_idx,
              label, stats_lines, won):
    """绘制一个地图面板 (含路径动画)."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    # 地图背景
    for r in range(rows):
        for c in range(cols):
            x = x_off + c * cell_size
            y = y_off + r * cell_size
            if grid[r][c] == 1:
                pygame.draw.rect(surface, WALL_COLOR,
                                 (x, y, cell_size, cell_size))
            else:
                pygame.draw.rect(surface, FLOOR_COLOR,
                                 (x, y, cell_size, cell_size))
            # 网格线
            pygame.draw.rect(surface, GRID_LINE,
                             (x, y, cell_size, cell_size), 1)

    # 目标
    for (tc, tr), tid in init_targets:
        x = x_off + tc * cell_size + cell_size // 4
        y = y_off + tr * cell_size + cell_size // 4
        color = TARGET_COLORS[tid % len(TARGET_COLORS)]
        s = cell_size // 2
        pygame.draw.rect(surface, color, (x, y, s, s), 2)
        # 目标编号
        if cell_size >= 16:
            font_small = pygame.font.SysFont("consolas", max(10, cell_size // 3))
            txt = font_small.render(str(tid), True, color)
            surface.blit(txt, (x + s // 2 - txt.get_width() // 2,
                               y + s // 2 - txt.get_height() // 2))

    # 路径轨迹 (到当前 step)
    trail_surf = pygame.Surface((cols * cell_size, rows * cell_size),
                                pygame.SRCALPHA)
    n = min(step_idx + 1, len(path))
    for i in range(n):
        pc, pr = path[i]
        x = pc * cell_size + cell_size // 4
        y = pr * cell_size + cell_size // 4
        # 越新的点越不透明
        alpha = max(40, int(path_color[3] * (0.3 + 0.7 * i / max(1, n - 1))))
        color = (*path_color[:3], alpha)
        pygame.draw.rect(trail_surf, color,
                         (x, y, cell_size // 2, cell_size // 2))
    surface.blit(trail_surf, (x_off, y_off))

    # 箱子 (初始位置, 半透明)
    for (bc, br), cid in init_boxes:
        x = x_off + bc * cell_size + 2
        y = y_off + br * cell_size + 2
        s = cell_size - 4
        color = BOX_COLORS[cid % len(BOX_COLORS)]
        pygame.draw.rect(surface, color, (x, y, s, s))
        if cell_size >= 16:
            font_small = pygame.font.SysFont("consolas", max(10, cell_size // 3))
            txt = font_small.render(str(cid), True, BLACK)
            surface.blit(txt, (x + s // 2 - txt.get_width() // 2,
                               y + s // 2 - txt.get_height() // 2))

    # 炸弹
    for bc, br in init_bombs:
        cx = x_off + bc * cell_size + cell_size // 2
        cy = y_off + br * cell_size + cell_size // 2
        pygame.draw.circle(surface, BOMB_COLOR, (cx, cy), cell_size // 3)
        if cell_size >= 16:
            font_small = pygame.font.SysFont("consolas", max(10, cell_size // 3))
            txt = font_small.render("💣", True, BLACK)
            surface.blit(txt, (cx - txt.get_width() // 2,
                               cy - txt.get_height() // 2))

    # 车当前位置
    if step_idx < len(path):
        cc, cr = path[step_idx]
        cx = x_off + cc * cell_size + cell_size // 2
        cy = y_off + cr * cell_size + cell_size // 2
        pygame.draw.circle(surface, CAR_COLOR, (cx, cy), cell_size // 3)

    # 车初始位置 标记
    ic, ir = init_car
    ix = x_off + ic * cell_size + cell_size // 2
    iy = y_off + ir * cell_size + cell_size // 2
    pygame.draw.circle(surface, (200, 200, 200), (ix, iy), 3)

    # 标签和统计
    font = pygame.font.SysFont("consolas", 16)
    font_bold = pygame.font.SysFont("consolas", 18, bold=True)

    txt = font_bold.render(label, True, ACCENT)
    surface.blit(txt, (x_off, y_off - 25))

    # 统计信息
    for i, line in enumerate(stats_lines):
        color = TEXT_COLOR
        if "✅" in line:
            color = SUCCESS_COLOR
        elif "❌" in line:
            color = FAIL_COLOR
        elif "⚠" in line:
            color = WARN_COLOR
        txt = font.render(line, True, color)
        surface.blit(txt, (x_off, y_off + rows * cell_size + 5 + i * 18))


def main():
    parser = argparse.ArgumentParser(
        description='可视化对比 exact solver 路径 vs 高层解码+BFS 路径')
    parser.add_argument('--phase', type=int, default=None,
                        help='只看指定 phase')
    parser.add_argument('--timeout', type=float, default=30.0,
                        help='solver 超时 (秒)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    maps_root = os.path.join(PROJECT_ROOT, "assets", "maps")
    manifest = get_seed_manifest()

    # 收集所有地图
    phases = [args.phase] if args.phase else list(range(1, 7))
    all_maps = []
    for phase in phases:
        phase_dir = os.path.join(maps_root, f"phase{phase}")
        files = sorted(glob.glob(os.path.join(phase_dir, "*.txt")))
        files = [f for f in files if 'verify_' not in os.path.basename(f)]
        for f in files:
            name = os.path.basename(f)
            seed = get_seed_for_map(manifest, phase, name, args.seed)
            all_maps.append((f, phase, seed))

    total = len(all_maps)
    print(f"共 {total} 张地图, 开始预计算...")

    # 预计算所有地图
    results = []
    for i, (fpath, phase, seed) in enumerate(all_maps):
        name = os.path.basename(fpath)
        print(f"  [{i+1}/{total}] Phase {phase} / {name} ...", end="",
              flush=True)
        t0 = time.perf_counter()
        data = precompute_map(fpath, seed, args.timeout)
        data["phase"] = phase
        elapsed = time.perf_counter() - t0
        if data["solved"]:
            print(f"  ✅ {elapsed:.1f}s  "
                  f"exact={data['exact_total']}步 "
                  f"decode={data['decode_total']}步 "
                  f"差{data['decode_total']-data['exact_total']:+d}")
        else:
            print(f"  ⚠️ solver 无解 ({elapsed:.1f}s)")
        results.append(data)

    solved = [r for r in results if r["solved"]]
    unsolved = [r for r in results if not r["solved"]]
    print(f"\n预计算完成: {len(solved)} 已解 / {len(unsolved)} 无解")

    if not solved:
        print("没有可显示的地图!")
        return

    # ── Pygame 可视化 ──
    pygame.init()

    # 计算窗口大小
    max_rows = max(len(r["grid"]) for r in solved)
    max_cols = max(len(r["grid"][0]) for r in solved)

    # 自适应 cell_size
    screen_w = min(1800, pygame.display.Info().current_w - 100)
    screen_h = min(1000, pygame.display.Info().current_h - 100)
    panel_w = (screen_w - 40) // 2
    info_h = 120  # 底部信息区高度
    top_margin = 40
    cell_size = min(panel_w // max_cols, (screen_h - info_h - top_margin) // max_rows)
    cell_size = max(12, min(cell_size, 40))

    actual_w = cell_size * max_cols
    actual_h = cell_size * max_rows
    win_w = actual_w * 2 + 60
    win_h = actual_h + info_h + top_margin + 20

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Exact vs Decoded 路径对比")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 16)
    font_big = pygame.font.SysFont("consolas", 22, bold=True)

    current_idx = 0
    step_idx = 0
    playing = False
    speed = 8  # 步/秒
    last_step_time = 0.0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        data = solved[current_idx]

        max_steps = max(data["exact_total"], data["decode_total"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_RIGHT:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        speed = min(120, speed + 4)
                    else:
                        current_idx = (current_idx + 1) % len(solved)
                        step_idx = 0
                        playing = False
                elif event.key == pygame.K_LEFT:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        speed = max(1, speed - 4)
                    else:
                        current_idx = (current_idx - 1) % len(solved)
                        step_idx = 0
                        playing = False
                elif event.key == pygame.K_UP:
                    speed = min(120, speed + 4)
                elif event.key == pygame.K_DOWN:
                    speed = max(1, speed - 4)
                elif event.key == pygame.K_r:
                    step_idx = 0
                    playing = False
                elif event.key == pygame.K_END:
                    step_idx = max_steps

        # 自动步进
        now = time.perf_counter()
        if playing and step_idx < max_steps:
            if now - last_step_time >= 1.0 / speed:
                step_idx += 1
                last_step_time = now
        elif playing and step_idx >= max_steps:
            playing = False

        # ── 渲染 ──
        screen.fill(PANEL_BG)

        grid = data["grid"]
        rows = len(grid)
        cols = len(grid[0])

        # 标题
        title = (f"[{current_idx+1}/{len(solved)}]  "
                 f"Phase {data['phase']} / {data['name']}  "
                 f"#{step_idx}/{max_steps}  "
                 f"速度: {speed}步/s  "
                 f"{'▶' if playing else '⏸'}")
        txt = font_big.render(title, True, ACCENT)
        screen.blit(txt, (15, 8))

        # 左面板: exact
        left_x = 15
        top_y = top_margin
        exact_stats = [
            f"步数: {data['exact_total']}",
            f"通关: {'✅' if data['exact_won'] else '❌'}",
            f"推操作: {data['solver_pushes']}  探索: {data['explore_steps']}",
        ]
        draw_grid(screen, grid, left_x, top_y, cell_size,
                  data["init_boxes"], data["init_targets"],
                  data["init_bombs"], data["init_car"],
                  data["exact_path"], PATH_EXACT,
                  min(step_idx, len(data["exact_path"]) - 1),
                  "EXACT (原始路径)", exact_stats, data["exact_won"])

        # 右面板: decoded
        right_x = left_x + cols * cell_size + 30
        step_diff = data["decode_total"] - data["exact_total"]
        decode_stats = [
            f"步数: {data['decode_total']}  (差{step_diff:+d})",
            f"通关: {'✅' if data['decode_won'] else '❌'}",
            f"解码: {data['decode_ok']}ok/{data['decode_fail']}fail  "
            f"推: {data['push_ok']}ok/{data['push_fail']}fail",
        ]
        draw_grid(screen, grid, right_x, top_y, cell_size,
                  data["init_boxes"], data["init_targets"],
                  data["init_bombs"], data["init_car"],
                  data["decode_path"], PATH_DECODE,
                  min(step_idx, len(data["decode_path"]) - 1),
                  "DECODED (高层+BFS)", decode_stats, data["decode_won"])

        # 底部操作提示
        help_y = win_h - 22
        help_txt = "SPACE=播放  ←→=切换  ↑↓/SHIFT+←→=变速  R=重置  END=跳到末尾  ESC=退出"
        txt = font.render(help_txt, True, (120, 125, 140))
        screen.blit(txt, (15, help_y))

        pygame.display.flip()

    pygame.quit()

    # 打印总结
    print(f"\n{'='*60}")
    print(f"总结 ({len(solved)} 张已解地图)")
    print(f"{'='*60}")
    perfect = sum(1 for r in solved
                  if r["decode_won"] and r["decode_fail"] == 0
                  and r["push_fail"] == 0)
    step_diffs = [r["decode_total"] - r["exact_total"] for r in solved]
    avg_diff = sum(step_diffs) / len(step_diffs) if step_diffs else 0
    print(f"  完美解码通关: {perfect}/{len(solved)}")
    print(f"  步数差异: 平均 {avg_diff:+.1f}步  "
          f"最小 {min(step_diffs):+d}  最大 {max(step_diffs):+d}")

    if unsolved:
        print(f"\n  ⚠️ {len(unsolved)} 张地图 solver 无解:")
        for r in unsolved:
            print(f"    Phase {r['phase']} / {r['name']}")


if __name__ == "__main__":
    main()
