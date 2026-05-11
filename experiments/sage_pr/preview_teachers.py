"""V1 老师 vs V2 老师 双视图对比, 支持箭头切换地图.

V1 老师: plan_exploration (探索阶段) + MultiBoxSolver (god-mode after explore)
V2 老师: god-mode A + 抑制场 + 嵌入 inspect (build_dataset_v6 同逻辑)

用法:
    python experiments/sage_pr/preview_teachers.py --map assets/maps/phase5/phase5_0001.txt

快捷键:
    SPACE 播放/暂停    R 重置    ←/→ 调速    N / Page Down 下张    P / Page Up 上张    ESC 退出
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pygame

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.explorer import (
    plan_exploration, direction_to_action,
)
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import generate_candidates
from experiments.sage_pr.build_dataset_v3 import (
    parse_phase456_seeds, match_move_to_candidate,
)
from experiments.sage_pr.belief_ida_solver import apply_inspect, _heading_to_angle
from experiments.sage_pr.build_dataset_v6 import (
    plan_god_mode, pick_inspect_for_unlock,
)
from smartcar_sokoban.solver.explorer import compute_facing_actions
from experiments.sage_pr.preview_compare import (
    record_apply_solver_move, record_apply_inspect,
)
from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.renderer import Renderer
from smartcar_sokoban.paths import PROJECT_ROOT


def make_headless_renderer():
    """构造一个 Renderer 实例但不开窗口, 用于渲染到任意 surface."""
    cfg = GameConfig()
    cfg.render_mode = "simple"
    r = Renderer(cfg, str(PROJECT_ROOT))
    pygame.font.init()
    r._font = pygame.font.SysFont("Arial", 16, bold=True)
    r._big_font = pygame.font.SysFont("Arial", 24, bold=True)
    return r


# ── V1 老师轨迹 ────────────────────────────────────────────

def record_teacher_v1(map_path: str, seed: int, *, time_limit: float = 30.0):
    """plan_exploration + MultiBoxSolver (god-mode after explore). 录 low-level actions."""
    random.seed(seed)
    eng = GameEngine(); eng.reset(map_path)
    action_log: List[int] = []
    info = {"won": False, "n_explore": 0, "n_push_macros": 0}

    with contextlib.redirect_stdout(io.StringIO()):
        explore_actions = plan_exploration(eng)
    info["n_explore"] = len(explore_actions)
    # plan_exploration 已经在 eng 上跑过了 (副作用), 重置然后重做录制
    random.seed(seed); eng.reset(map_path)
    # 录探索 actions
    for a in explore_actions:
        eng.discrete_step(a); action_log.append(a)

    state = eng.get_state()
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            solution = solver.solve(max_cost=300, time_limit=time_limit, strategy="auto")
        except Exception:
            solution = None
    if not solution:
        return action_log, info
    directions = solver.solution_to_actions(solution)
    # 转 low-level
    eng.discrete_step(6); action_log.append(6)
    for dx, dy in directions:
        a = direction_to_action(dx, dy)
        eng.discrete_step(a); action_log.append(a)
        info["n_push_macros"] += 1
    info["won"] = eng.get_state().won
    return action_log, info


# ── V2 老师轨迹 (god-mode A + 抑制场 + 嵌入 inspect) ────────

def record_teacher_v2(map_path: str, seed: int, *, time_limit: float = 30.0):
    random.seed(seed)
    eng = GameEngine(); eng.reset(map_path)
    plan = plan_god_mode(map_path, seed, time_limit=time_limit)
    action_log: List[int] = []
    info = {"won": False, "n_push": 0, "n_inspect": 0}
    if plan is None:
        return action_log, info
    a_idx = 0; inspect_streak = 0
    while a_idx < len(plan):
        bs = BeliefState.from_engine_state(eng.get_state(), fully_observed=False)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat, enforce_sigma_lock=True)
        move = plan[a_idx]
        label = match_move_to_candidate(move, cands, bs, run_length=1)
        if label is not None:
            ok = record_apply_solver_move(eng, move, action_log)
            if not ok: return action_log, info
            info["n_push"] += 1
            a_idx += 1; inspect_streak = 0; continue
        if inspect_streak >= 8:
            ok = record_apply_solver_move(eng, move, action_log)
            if not ok: return action_log, info
            a_idx += 1; inspect_streak = 0; continue
        rb, rt = -1, -1
        if move[0] == "box":
            op, cid = move[1]
            for j, b in enumerate(bs.boxes):
                if (b.col, b.row) == op: rb = j; break
            for j, t in enumerate(bs.targets):
                if t.num_id == cid: rt = j; break
        il = pick_inspect_for_unlock(bs, cands, rb, rt)
        if il is None:
            ok = record_apply_solver_move(eng, move, action_log)
            if not ok: return action_log, info
            a_idx += 1; inspect_streak = 0; continue
        ok = record_apply_inspect(eng, cands[il], action_log)
        if not ok: return action_log, info
        info["n_inspect"] += 1
        inspect_streak += 1
    info["won"] = eng.get_state().won
    return action_log, info


# ── 同 phase 地图列表 ─────────────────────────────────────

def list_phase_maps_local(map_path: str) -> List[str]:
    abs_path = os.path.join(ROOT, map_path) if not os.path.isabs(map_path) else map_path
    folder = os.path.dirname(abs_path)
    rel_folder = os.path.relpath(folder, ROOT).replace("\\", "/")
    return sorted([
        f"{rel_folder}/{fn}" for fn in os.listdir(folder)
        if fn.endswith(".txt") and not fn.startswith("_")
    ])


def lookup_seed(map_rel: str, vmap: dict, default: int = 0) -> int:
    rel = map_rel.replace("\\", "/")
    return vmap.get(rel, [default])[0]


# ── 主可视化 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, help="例: assets/maps/phase5/phase5_0001.txt")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=20.0)
    args = parser.parse_args()

    vmap = parse_phase456_seeds(
        os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
    )

    all_maps = list_phase_maps_local(args.map)
    rel = args.map.replace("\\", "/")
    try:
        cur_idx = all_maps.index(rel)
    except ValueError:
        cur_idx = 0
        all_maps = [rel] + all_maps
    print(f"phase folder has {len(all_maps)} maps, current idx={cur_idx}")

    pygame.init()
    pygame.display.set_caption("V1 teacher vs V2 teacher")
    renderer = make_headless_renderer()
    W = renderer.w   # 640
    H = renderer.h   # 480
    TITLE_H = 30
    screen = pygame.display.set_mode((W * 2 + 20, H + TITLE_H))
    clock = pygame.time.Clock()
    surf_l = pygame.Surface((W, H))
    surf_r = pygame.Surface((W, H))
    big_font = pygame.font.SysFont("Arial", 18, bold=True)
    small_font = pygame.font.SysFont("Consolas", 14)

    # 状态
    eng_l = GameEngine()
    eng_r = GameEngine()
    log_l: List[int] = []
    log_r: List[int] = []
    info_l: Dict = {}
    info_r: Dict = {}
    cur_seed = 0
    step = 0
    playing = True
    step_delay = 0.05
    last_step = 0.0

    def load_map(idx: int):
        nonlocal eng_l, eng_r, log_l, log_r, info_l, info_r, cur_seed, step
        map_rel = all_maps[idx]
        cur_seed = args.seed if args.seed is not None else lookup_seed(map_rel, vmap)
        print(f"\n==== [{idx+1}/{len(all_maps)}] {map_rel} seed={cur_seed} ====")
        t0 = time.perf_counter()
        log_l, info_l = record_teacher_v1(map_rel, cur_seed, time_limit=args.time_limit)
        t1 = time.perf_counter()
        log_r, info_r = record_teacher_v2(map_rel, cur_seed, time_limit=args.time_limit)
        t2 = time.perf_counter()
        print(f"  V1: {len(log_l)} low-level, explore={info_l['n_explore']} push_macros={info_l['n_push_macros']} "
              f"won={info_l['won']} ({t1-t0:.1f}s)")
        print(f"  V2: {len(log_r)} low-level, push={info_r['n_push']} inspect={info_r['n_inspect']} "
              f"won={info_r['won']} ({t2-t1:.1f}s)")
        eng_l = GameEngine(); random.seed(cur_seed); eng_l.reset(map_rel)
        eng_r = GameEngine(); random.seed(cur_seed); eng_r.reset(map_rel)
        step = 0

    load_map(cur_idx)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_r:
                    eng_l = GameEngine(); random.seed(cur_seed); eng_l.reset(all_maps[cur_idx])
                    eng_r = GameEngine(); random.seed(cur_seed); eng_r.reset(all_maps[cur_idx])
                    step = 0
                elif event.key in (pygame.K_LEFT,):
                    step_delay = min(0.5, step_delay + 0.02)
                elif event.key in (pygame.K_RIGHT,):
                    step_delay = max(0.01, step_delay - 0.02)
                elif event.key in (pygame.K_n, pygame.K_PAGEDOWN):
                    cur_idx = (cur_idx + 1) % len(all_maps); load_map(cur_idx)
                elif event.key in (pygame.K_p, pygame.K_PAGEUP):
                    cur_idx = (cur_idx - 1) % len(all_maps); load_map(cur_idx)

        now = time.perf_counter()
        if playing and now - last_step >= step_delay:
            if step < max(len(log_l), len(log_r)):
                if step < len(log_l):
                    eng_l.discrete_step(log_l[step])
                if step < len(log_r):
                    eng_r.discrete_step(log_r[step])
                step += 1
                last_step = now

        info_l_t = (f"step {step}/{len(log_l)} | explore={info_l.get('n_explore',0)} "
                    f"push_macros={info_l.get('n_push_macros',0)} "
                    f"won={'YES' if eng_l.get_state().won else 'no'}")
        info_r_t = (f"step {step}/{len(log_r)} | push={info_r.get('n_push',0)} "
                    f"inspect={info_r.get('n_inspect',0)} "
                    f"won={'YES' if eng_r.get_state().won else 'no'}")

        # 用 Renderer._render_2d 渲染到 surface
        renderer._render_2d(surf_l, eng_l.get_state(), None, show_labels=True)
        renderer._render_2d(surf_r, eng_r.get_state(), None, show_labels=True)

        # 标题栏
        screen.fill((24, 24, 28))
        title_l = big_font.render(
            f"V1 老师 (explore→push) | {os.path.basename(all_maps[cur_idx])}",
            True, (220, 220, 240))
        title_r = big_font.render(
            f"V2 老师 (god-A+抑制场) | seed={cur_seed}",
            True, (220, 220, 240))
        screen.blit(title_l, (8, 6))
        screen.blit(title_r, (W + 28, 6))

        # 主渲染
        screen.blit(surf_l, (0, TITLE_H))
        screen.blit(surf_r, (W + 20, TITLE_H))

        # info 文字 (画在 surface 上)
        info_text_l = small_font.render(info_l_t, True, (200, 240, 200))
        info_text_r = small_font.render(info_r_t, True, (200, 240, 200))
        screen.blit(info_text_l, (8, H + TITLE_H - 18))
        screen.blit(info_text_r, (W + 28, H + TITLE_H - 18))

        # 分隔条
        pygame.draw.rect(screen, (60, 60, 70), (W, 0, 20, H + TITLE_H))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
