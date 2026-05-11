"""模型 vs 老师 双视图对比可视化.

左: SAGE-PR 模型 (V2 mode + rollout search), 纯神经
右: 老师 (god-mode A + 抑制场 + 嵌入 inspect, 跟 build_dataset_v6 同逻辑)

用法:
    python experiments/sage_pr/preview_compare.py \
        --ckpt .agent/sage_pr/runs/v12_dag8d/best.pt \
        --map assets/maps/phase5/phase5_0001.txt

快捷键:
    SPACE 播放/暂停    R 重置    ←/→ 调速    N/P 翻图    ESC 退出
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

import numpy as np
import pygame
import torch

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.explorer import compute_facing_actions
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import (
    Candidate, generate_candidates, candidates_legality_mask,
)
from smartcar_sokoban.symbolic.cand_features import encode_candidates
from smartcar_sokoban.symbolic.grid_tensor import build_grid_tensor, build_global_features
from experiments.sage_pr.model import build_default_model
from experiments.sage_pr.build_dataset_v3 import (
    apply_solver_move, parse_phase456_seeds, match_move_to_candidate,
)
from experiments.sage_pr.evaluate_sage_pr import candidate_to_solver_move, _state_signature
from experiments.sage_pr.rollout_search_eval import rollout_search_step
from experiments.sage_pr.belief_ida_solver import apply_inspect, _heading_to_angle
from experiments.sage_pr.build_dataset_v6 import (
    plan_god_mode, pick_inspect_for_unlock,
)


# ── 记录 action 序列 ────────────────────────────────────────

def record_apply_solver_move(eng: GameEngine, move, log: List):
    eng.discrete_step(6); log.append(6)
    state = eng.get_state()
    etype, eid, direction, _ = move
    dx, dy = direction
    if etype == "box":
        old_pos, _ = eid; ec, er = old_pos
    elif etype == "bomb":
        ec, er = eid
    else:
        return False
    car_target = (ec - dx, er - dy)
    obstacles = set()
    for b in state.boxes: obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs: obstacles.add(pos_to_grid(bm.x, bm.y))
    car_grid = pos_to_grid(state.car_x, state.car_y)
    if car_grid != car_target:
        path = bfs_path(car_grid, car_target, state.grid, obstacles)
        if path is None: return False
        for pdx, pdy in path:
            a = direction_to_abs_action(pdx, pdy)
            eng.discrete_step(a); log.append(a)
    a = direction_to_abs_action(dx, dy)
    eng.discrete_step(a); log.append(a)
    return True


def record_apply_inspect(eng: GameEngine, cand: Candidate, log: List):
    state = eng.get_state()
    obstacles = set()
    for b in state.boxes: obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs: obstacles.add(pos_to_grid(bm.x, bm.y))
    car_grid = pos_to_grid(state.car_x, state.car_y)
    target = (cand.viewpoint_col, cand.viewpoint_row)
    eng.discrete_step(6); log.append(6)
    if car_grid != target:
        path = bfs_path(car_grid, target, state.grid, obstacles)
        if path is None: return False
        for pdx, pdy in path:
            a = direction_to_abs_action(pdx, pdy)
            eng.discrete_step(a); log.append(a)
    state = eng.get_state()
    rot_acts = compute_facing_actions(state.car_angle, _heading_to_angle(cand.inspect_heading or 0))
    for a in rot_acts:
        eng.discrete_step(a); log.append(a)
    return True


# ── 模型 (V2 + rollout search) trajectory ──────────────────

def record_model(model, device, map_path: str, seed: int,
                 *, step_limit=150, beam=8, lookahead=25):
    random.seed(seed)
    eng = GameEngine(); eng.reset(map_path)
    action_log: List[int] = []
    visited = {_state_signature(eng.get_state())}
    info = {"won": False, "n_macros": 0}

    for step in range(step_limit):
        if eng.get_state().won:
            info["won"] = True; return action_log, info
        result = rollout_search_step(model, device, eng, visited,
                                      beam_width=beam, lookahead=lookahead,
                                      fully_observed=False, enforce_sigma_lock=True)
        if result is None: return action_log, info
        idx, cand = result
        bs = BeliefState.from_engine_state(eng.get_state(), fully_observed=False)
        if cand.type == "inspect":
            ok = record_apply_inspect(eng, cand, action_log)
        else:
            move = candidate_to_solver_move(cand, bs)
            ok = record_apply_solver_move(eng, move, action_log) if move else False
        if not ok: return action_log, info
        info["n_macros"] += 1
        visited.add(_state_signature(eng.get_state()))
    info["won"] = eng.get_state().won
    return action_log, info


# ── 老师 (god-mode A + 抑制场) trajectory ──────────────────

def record_teacher(map_path: str, seed: int, *, step_limit=200, time_limit=30.0):
    random.seed(seed)
    eng = GameEngine(); eng.reset(map_path)
    plan = plan_god_mode(map_path, seed, time_limit=time_limit)
    if plan is None: return [], {"won": False, "n_macros": 0}
    action_log: List[int] = []
    info = {"won": False, "n_macros": 0, "n_inspect": 0, "n_push": 0}
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
            info["n_push"] += 1; info["n_macros"] += 1
            a_idx += 1; inspect_streak = 0; continue
        if inspect_streak >= 8:
            ok = record_apply_solver_move(eng, move, action_log)
            if not ok: return action_log, info
            info["n_macros"] += 1
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
            info["n_macros"] += 1
            a_idx += 1; inspect_streak = 0; continue
        ok = record_apply_inspect(eng, cands[il], action_log)
        if not ok: return action_log, info
        info["n_inspect"] += 1; info["n_macros"] += 1
        inspect_streak += 1
    info["won"] = eng.get_state().won
    return action_log, info


# ── 简单网格渲染 (避免依赖 Renderer 单窗口约束) ────────────

CELL = 28
GRID_ROWS, GRID_COLS = 12, 16


def render_engine(surf: pygame.Surface, eng: GameEngine,
                  title: str, info_text: str):
    """把一个 engine 状态画到 surf 上 (左上角原点)."""
    s = eng.get_state()
    grid = s.grid
    surf.fill((20, 20, 24))
    is_np = hasattr(grid, "shape")
    # walls
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x = c * CELL; y = r * CELL + 40   # title bar offset
            cell = grid[r, c] if is_np else grid[r][c]
            if cell:
                pygame.draw.rect(surf, (80, 80, 90), (x, y, CELL, CELL))
            else:
                pygame.draw.rect(surf, (44, 44, 48), (x, y, CELL, CELL))
            pygame.draw.rect(surf, (28, 28, 32), (x, y, CELL, CELL), 1)
    # targets
    for t in s.targets:
        gc, gr = pos_to_grid(t.x, t.y)
        x = gc * CELL + CELL // 2; y = gr * CELL + 40 + CELL // 2
        col = (210, 180, 60) if t.num_id is not None else (110, 90, 30)
        pygame.draw.circle(surf, col, (x, y), CELL // 3, 2)
        if t.num_id is not None:
            font = pygame.font.SysFont("consolas", 14)
            txt = font.render(str(t.num_id), True, col)
            surf.blit(txt, (x - 4, y - 7))
    # bombs (red)
    for bm in s.bombs:
        gc, gr = pos_to_grid(bm.x, bm.y)
        x = gc * CELL + CELL // 2; y = gr * CELL + 40 + CELL // 2
        pygame.draw.circle(surf, (220, 80, 60), (x, y), CELL // 3 - 2)
    # boxes (purple if id known, gray if unknown)
    for b in s.boxes:
        gc, gr = pos_to_grid(b.x, b.y)
        x = gc * CELL; y = gr * CELL + 40
        col = (140, 110, 200) if b.class_id is not None else (120, 120, 130)
        pygame.draw.rect(surf, col, (x + 4, y + 4, CELL - 8, CELL - 8))
        if b.class_id is not None:
            font = pygame.font.SysFont("consolas", 14)
            txt = font.render(str(b.class_id), True, (240, 240, 240))
            surf.blit(txt, (x + CELL // 2 - 4, y + CELL // 2 - 7))
    # car (triangle pointing in direction)
    cx = s.car_x * CELL; cy = s.car_y * CELL + 40
    angle = s.car_angle
    p1 = (cx + math.cos(angle) * CELL * 0.4, cy + math.sin(angle) * CELL * 0.4)
    p2 = (cx + math.cos(angle + 2.5) * CELL * 0.3, cy + math.sin(angle + 2.5) * CELL * 0.3)
    p3 = (cx + math.cos(angle - 2.5) * CELL * 0.3, cy + math.sin(angle - 2.5) * CELL * 0.3)
    pygame.draw.polygon(surf, (80, 200, 80), [p1, p2, p3])
    # title
    font = pygame.font.SysFont("consolas", 18, bold=True)
    title_t = font.render(title, True, (240, 240, 240))
    surf.blit(title_t, (8, 8))
    font2 = pygame.font.SysFont("consolas", 14)
    info_t = font2.render(info_text, True, (180, 220, 180))
    surf.blit(info_t, (8, 24))


def play_compare(map_path: str, seed: int, action_log_m: List[int],
                  action_log_t: List[int], info_m: dict, info_t: dict):
    pygame.init()
    pygame.display.set_caption("SAGE-PR vs Teacher")
    W = CELL * GRID_COLS
    H = CELL * GRID_ROWS + 40
    screen = pygame.display.set_mode((W * 2 + 20, H))
    clock = pygame.time.Clock()

    eng_m = GameEngine(); random.seed(seed); eng_m.reset(map_path)
    eng_t = GameEngine(); random.seed(seed); eng_t.reset(map_path)

    surf_m = pygame.Surface((W, H))
    surf_t = pygame.Surface((W, H))

    step = 0
    playing = True
    step_delay = 0.05
    last_step = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_SPACE: playing = not playing
                elif event.key == pygame.K_r:
                    eng_m = GameEngine(); random.seed(seed); eng_m.reset(map_path)
                    eng_t = GameEngine(); random.seed(seed); eng_t.reset(map_path)
                    step = 0
                elif event.key == pygame.K_LEFT:
                    step_delay = min(0.5, step_delay + 0.02)
                elif event.key == pygame.K_RIGHT:
                    step_delay = max(0.01, step_delay - 0.02)

        now = time.perf_counter()
        if playing and now - last_step >= step_delay:
            if step < max(len(action_log_m), len(action_log_t)):
                if step < len(action_log_m):
                    eng_m.discrete_step(action_log_m[step])
                if step < len(action_log_t):
                    eng_t.discrete_step(action_log_t[step])
                step += 1
                last_step = now

        model_won = eng_m.get_state().won
        teacher_won = eng_t.get_state().won
        info_m_text = (f"step {step}/{len(action_log_m)} | "
                        f"{info_m['n_macros']} macros | "
                        f"won={'YES' if model_won else 'no'}")
        info_t_text = (f"step {step}/{len(action_log_t)} | "
                        f"{info_t['n_macros']} macros | "
                        f"won={'YES' if teacher_won else 'no'}")
        render_engine(surf_m, eng_m, "MODEL (V2 + rollout)", info_m_text)
        render_engine(surf_t, eng_t, "TEACHER (god-mode A)", info_t_text)
        screen.blit(surf_m, (0, 0))
        screen.blit(surf_t, (W + 20, 0))
        # separator
        pygame.draw.rect(screen, (60, 60, 70), (W, 0, 20, H))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=25)
    parser.add_argument("--step-limit", type=int, default=150)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    model = build_default_model().to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"loaded {args.ckpt}, val_acc={ck.get('val_acc', '?'):.3f}")

    seed = args.seed
    if seed is None:
        vmap = parse_phase456_seeds(os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json"))
        rel = args.map.replace("\\", "/")
        seed = vmap.get(rel, [0])[0]
    print(f"map={args.map}, seed={seed}")

    print("\n[1/2] recording MODEL trajectory...")
    t0 = time.perf_counter()
    log_m, info_m = record_model(model, device, args.map, seed,
                                   step_limit=args.step_limit,
                                   beam=args.beam, lookahead=args.lookahead)
    print(f"  {len(log_m)} low-level, {info_m['n_macros']} macros, "
          f"won={info_m['won']}, {time.perf_counter()-t0:.1f}s")

    print("\n[2/2] recording TEACHER trajectory...")
    t0 = time.perf_counter()
    log_t, info_t = record_teacher(args.map, seed, step_limit=args.step_limit)
    print(f"  {len(log_t)} low-level, {info_t['n_macros']} macros "
          f"(push={info_t.get('n_push', 0)} inspect={info_t.get('n_inspect', 0)}), "
          f"won={info_t['won']}, {time.perf_counter()-t0:.1f}s")

    if args.no_render:
        return

    play_compare(args.map, seed, log_m, log_t, info_m, info_t)


if __name__ == "__main__":
    main()
