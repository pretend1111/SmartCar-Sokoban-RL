"""SAGE-PR 推理可视化 — 录制 hybrid_v2 trajectory 后 pygame 回放.

用法:
    python experiments/sage_pr/preview_sage_pr.py \
        --ckpt .agent/sage_pr/runs/dl3_r1_train/best.pt \
        --map assets/maps/phase6/phase6_11.txt --seed 0

快捷键 (pygame):
    SPACE — 播放/暂停
    R     — 重置
    ESC   — 退出
    ←/→   — 调整回放速度
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import os
import random
import sys
import time
from typing import List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine, GameState
from smartcar_sokoban.paths import PROJECT_ROOT
from smartcar_sokoban.renderer import Renderer
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import (
    Candidate, generate_candidates, candidates_legality_mask,
)
from smartcar_sokoban.symbolic.cand_features import encode_candidates
from smartcar_sokoban.symbolic.grid_tensor import (
    build_grid_tensor, build_global_features,
)
from experiments.sage_pr.model import build_default_model
from experiments.sage_pr.build_dataset_v3 import (
    parse_phase456_seeds, match_move_to_candidate,
)
from experiments.sage_pr.evaluate_sage_pr import (
    candidate_to_solver_move, _state_signature,
)
from experiments.sage_pr.rollout_search_eval import rollout_search_step

import pygame


# ── 录制版 apply_solver_move ──────────────────────────────

def apply_solver_move_recorded(eng: GameEngine, move,
                                action_log: List[Tuple[int, str]]):
    """跟 build_dataset_v3.apply_solver_move 一样, 但把每个 discrete_step 动作记入 action_log."""
    etype, eid, direction, _ = move
    eng.discrete_step(6)   # snap
    action_log.append((6, "snap"))

    state = eng.get_state()
    dx, dy = direction
    if etype == "box":
        old_pos, _ = eid
        ec, er = old_pos
    elif etype == "bomb":
        ec, er = eid
    else:
        return False
    car_target = (ec - dx, er - dy)

    obstacles = set()
    for b in state.boxes:
        obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs:
        obstacles.add(pos_to_grid(bm.x, bm.y))

    car_grid = pos_to_grid(state.car_x, state.car_y)
    if car_grid != car_target:
        path = bfs_path(car_grid, car_target, state.grid, obstacles)
        if path is None:
            return False
        for pdx, pdy in path:
            a = direction_to_abs_action(pdx, pdy)
            eng.discrete_step(a)
            action_log.append((a, f"walk ({pdx},{pdy})"))

    a = direction_to_abs_action(dx, dy)
    eng.discrete_step(a)
    tag = f"PUSH {etype}@({ec},{er}) dir=({dx},{dy})"
    action_log.append((a, tag))
    return True


# ── Hybrid v2 inference 录制 ──────────────────────────────

def hybrid_v2_record(model, device, map_path: str, seed: int,
                     *, step_limit: int = 60,
                     stuck_threshold: int = 1,
                     solver_time_limit: float = 30.0,
                     beam_width: int = 4, lookahead: int = 12,
                     fully_observed: bool = True,
                     enforce_sigma_lock: bool = False,
                     pure_neural: bool = False,
                     ) -> Tuple[List[Tuple[int, str]], bool, dict]:
    """跑 hybrid_v2 inference 并录所有 discrete actions."""
    random.seed(seed)
    eng = GameEngine()
    eng.reset(map_path)

    action_log: List[Tuple[int, str]] = []
    visited = set()
    visited.add(_state_signature(eng.get_state()))

    no_progress = 0
    using_solver = False
    info = {
        "n_macro_steps": 0,
        "n_solver_calls": 0,
        "n_model_steps": 0,
        "won": False,
    }

    for step in range(step_limit):
        s = eng.get_state()
        if s.won:
            info["won"] = True
            return action_log, True, info

        if not pure_neural and not using_solver and (no_progress >= stuck_threshold):
            with contextlib.redirect_stdout(io.StringIO()):
                boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in s.boxes]
                targets = {t.num_id: pos_to_grid(t.x, t.y) for t in s.targets}
                bombs = [pos_to_grid(b.x, b.y) for b in s.bombs]
                car = pos_to_grid(s.car_x, s.car_y)
                solver = MultiBoxSolver(s.grid, car, boxes, targets, bombs)
                try:
                    moves = solver.solve(max_cost=300, time_limit=solver_time_limit, strategy="auto")
                except Exception:
                    moves = None
            if moves:
                using_solver = True
                info["n_solver_calls"] += 1
                for mv in moves:
                    if not apply_solver_move_recorded(eng, mv, action_log):
                        return action_log, False, info
                    info["n_macro_steps"] += 1
                    if eng.get_state().won:
                        info["won"] = True
                        return action_log, True, info
                continue

        result = rollout_search_step(model, device, eng, visited,
                                      beam_width=beam_width, lookahead=lookahead,
                                      fully_observed=fully_observed,
                                      enforce_sigma_lock=enforce_sigma_lock)
        if result is None:
            no_progress += 1
            if no_progress >= stuck_threshold:
                continue
            return action_log, False, info
        idx, cand = result

        bs = BeliefState.from_engine_state(eng.get_state(), fully_observed=fully_observed)

        n_box_before = len(eng.get_state().boxes)

        if cand.type == "inspect":
            # 录 inspect: 把 apply_inspect 的所有低层动作录下来
            from experiments.sage_pr.belief_ida_solver import _heading_to_angle
            from smartcar_sokoban.solver.explorer import compute_facing_actions
            state = eng.get_state()
            obstacles = set()
            for b in state.boxes:
                obstacles.add(pos_to_grid(b.x, b.y))
            for bm in state.bombs:
                obstacles.add(pos_to_grid(bm.x, bm.y))
            car_grid = pos_to_grid(state.car_x, state.car_y)
            target = (cand.viewpoint_col, cand.viewpoint_row)
            eng.discrete_step(6)
            action_log.append((6, "snap"))
            if car_grid != target:
                path = bfs_path(car_grid, target, state.grid, obstacles)
                if path is None:
                    return action_log, False, info
                for pdx, pdy in path:
                    a = direction_to_abs_action(pdx, pdy)
                    eng.discrete_step(a)
                    action_log.append((a, f"INSPECT walk ({pdx},{pdy})"))
            state = eng.get_state()
            rot_acts = compute_facing_actions(
                state.car_angle, _heading_to_angle(cand.inspect_heading or 0))
            for a in rot_acts:
                eng.discrete_step(a)
                action_log.append((a, f"INSPECT rotate -> heading {cand.inspect_heading}"))
        else:
            move = candidate_to_solver_move(cand, bs)
            if move is None:
                return action_log, False, info
            if not apply_solver_move_recorded(eng, move, action_log):
                return action_log, False, info
        info["n_macro_steps"] += 1
        info["n_model_steps"] += 1
        n_box_after = len(eng.get_state().boxes)

        if n_box_after < n_box_before:
            no_progress = 0
        else:
            no_progress += 1

        visited.add(_state_signature(eng.get_state()))

    info["won"] = eng.get_state().won
    return action_log, info["won"], info


# ── 状态插值动画 ──────────────────────────────────────────

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


# ── 主可视化 ──────────────────────────────────────────────

def list_phase_maps_local(map_path: str) -> List[str]:
    """从 --map 路径推 phase 文件夹, 列出该文件夹所有地图 (sorted)."""
    abs_path = os.path.join(ROOT, map_path) if not os.path.isabs(map_path) else map_path
    folder = os.path.dirname(abs_path)
    rel_folder = os.path.relpath(folder, ROOT).replace("\\", "/")
    files = sorted([
        f"{rel_folder}/{fn}"
        for fn in os.listdir(folder)
        if fn.endswith(".txt") and not fn.startswith("_")
    ])
    return files


def lookup_seed(map_rel: str, vmap: dict, default: int = 0) -> int:
    """从 manifest 取首个 verified seed, 没就用 default."""
    rel = map_rel.replace("\\", "/")
    return vmap.get(rel, [default])[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=".agent/sage_pr/runs/dl3_r1_train/best.pt")
    parser.add_argument("--map", required=True, help="例: assets/maps/phase6/phase6_11.txt")
    parser.add_argument("--seed", type=int, default=None,
                        help="若不指定且 manifest 有 verified seed 则用 manifest")
    parser.add_argument("--stuck", type=int, default=1)
    parser.add_argument("--solver-time-limit", type=float, default=30.0)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--lookahead", type=int, default=12)
    parser.add_argument("--mode", choices=["v1", "v2"], default="v1",
                        help="v1=fully_obs+solver fallback; v2=partial-obs+抑制场+纯神经")
    parser.add_argument("--pure-neural", action="store_true",
                        help="禁用 solver fallback, 纯模型决策")
    parser.add_argument("--step-limit", type=int, default=60)
    parser.add_argument("--no-render", action="store_true",
                        help="只跑推理打印 trace, 不开 pygame")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    model = build_default_model().to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"loaded {args.ckpt}, val_acc={ck.get('val_acc', '?'):.3f}")

    vmap = parse_phase456_seeds(
        os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
    )

    # 列出同 phase 所有地图, 并定位当前索引
    all_maps = list_phase_maps_local(args.map)
    rel = args.map.replace("\\", "/")
    try:
        cur_idx = all_maps.index(rel)
    except ValueError:
        cur_idx = 0
        all_maps = [rel] + all_maps   # fallback: 把 --map 加到列表头
    print(f"phase folder has {len(all_maps)} maps, current idx={cur_idx}")

    # 推理一张图 → 返回 (action_log, won, info, seed_used)
    def infer_map(map_rel: str):
        seed = args.seed if args.seed is not None else lookup_seed(map_rel, vmap)
        print(f"\n==== {map_rel} (seed={seed}) ====")
        print(f"running hybrid_v2 inference (stuck={args.stuck}, "
              f"solver_tl={args.solver_time_limit}s)...")
        t0 = time.perf_counter()
        action_log, won, info = hybrid_v2_record(
            model, device, map_rel, seed,
            step_limit=args.step_limit,
            stuck_threshold=args.stuck,
            solver_time_limit=args.solver_time_limit,
            beam_width=args.beam, lookahead=args.lookahead,
            fully_observed=(args.mode == "v1"),
            enforce_sigma_lock=(args.mode == "v2"),
            pure_neural=args.pure_neural,
        )
        elapsed = time.perf_counter() - t0
        n_low = len(action_log)
        print(f"  inference: {elapsed:.1f}s, {info['n_macro_steps']} macro / "
              f"{n_low} low-level, won={won}")
        print(f"  model steps={info['n_model_steps']}, solver calls={info['n_solver_calls']}")
        return action_log, won, info, seed

    action_log, won, info, seed = infer_map(rel)
    n_low = len(action_log)

    # 打印前 30 trace
    print("\n动作 trace (前 30):")
    for i, (a, tag) in enumerate(action_log[:30]):
        print(f"  [{i:3d}] action={a:2d} {tag}")
    if n_low > 30:
        print(f"  ... ({n_low-30} more)")

    if args.no_render:
        return

    # ── pygame 回放 ──────────────────────────────────────
    cfg = GameConfig()
    cfg.render_mode = "simple"
    cfg.control_mode = "discrete"
    eng_play = GameEngine(cfg, str(PROJECT_ROOT))
    renderer = Renderer(cfg, str(PROJECT_ROOT))
    renderer.init()
    clock = pygame.time.Clock()

    cur_map = rel
    cur_seed = seed
    cur_action_log = action_log
    cur_info = info

    random.seed(cur_seed)
    eng_play.reset(cur_map)

    state = eng_play.get_state()
    prev_state = None
    act_idx = 0
    playing = True
    animating = False
    anim_progress = 0.0
    step_delay = 0.1
    last_step_time = 0.0

    def reload_map(new_idx: int):
        """切换到 new_idx 对应的地图: 重跑推理 + 重置回放."""
        nonlocal cur_idx, cur_map, cur_seed, cur_action_log, cur_info
        nonlocal state, prev_state, act_idx, playing, animating, anim_progress
        cur_idx = new_idx % len(all_maps)
        cur_map = all_maps[cur_idx]
        cur_action_log, _, cur_info, cur_seed = infer_map(cur_map)
        random.seed(cur_seed)
        eng_play.reset(cur_map)
        state = eng_play.get_state()
        prev_state = None
        act_idx = 0
        playing = True
        animating = False
        anim_progress = 0.0

    def reset_current():
        nonlocal state, prev_state, act_idx, animating, anim_progress, playing
        random.seed(cur_seed)
        eng_play.reset(cur_map)
        state = eng_play.get_state()
        prev_state = None
        act_idx = 0
        animating = False
        anim_progress = 0.0
        playing = True

    print("\n[pygame] SPACE=暂停 R=重置 ←/→=切地图 +/-=调速 Tab=切渲染 ESC=退出")
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
                elif event.key == pygame.K_r:
                    reset_current()
                elif event.key == pygame.K_LEFT:
                    reload_map(cur_idx - 1)
                elif event.key == pygame.K_RIGHT:
                    reload_map(cur_idx + 1)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    step_delay = max(0.02, step_delay - 0.03)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    step_delay = min(1.0, step_delay + 0.03)
                elif event.key == pygame.K_TAB:
                    m = "simple" if cfg.render_mode == "full" else "full"
                    renderer.switch_mode(m)

        n_low = len(cur_action_log)
        now = time.perf_counter()
        if playing and not animating and act_idx < n_low:
            if now - last_step_time >= step_delay:
                prev_state = copy.deepcopy(state)
                a, _ = cur_action_log[act_idx]
                state = eng_play.discrete_step(a)
                act_idx += 1
                animating = True
                anim_progress = 0.0
                last_step_time = now

        if animating and prev_state is not None:
            anim_progress += dt * cfg.discrete_anim_speed
            if anim_progress >= 1.0:
                anim_progress = 1.0
                animating = False
            display_state = lerp_state(prev_state, state, anim_progress)
        else:
            display_state = state

        renderer.render(display_state)

        # HUD
        map_name = os.path.basename(cur_map)
        title_lines = [
            f"[{cur_idx+1}/{len(all_maps)}] {map_name}  seed={cur_seed}",
            f"step {act_idx}/{n_low} ({cur_info['n_macro_steps']} macro)",
            f"{'WON ✓' if state.won else 'in progress' if act_idx < n_low else 'FAIL'}",
            f"speed={1/step_delay:.1f}/s  ←→ map  +/- speed  R reset",
        ]
        try:
            font = pygame.font.SysFont("consolas", 16)
            for j, txt in enumerate(title_lines):
                surf = font.render(txt, True, (255, 255, 255))
                renderer.screen.blit(surf, (8, 8 + 20 * j))
        except Exception:
            pass

        pygame.display.flip()

        if act_idx >= n_low and not animating and playing:
            playing = False
            if state.won:
                print(f"  🎉 played to end: WON in {n_low} actions")
            else:
                print(f"  ❌ trace exhausted, didn't win")

    pygame.quit()


if __name__ == "__main__":
    main()
