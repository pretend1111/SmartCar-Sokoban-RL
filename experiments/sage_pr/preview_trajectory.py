"""通用 trajectory 预览脚本.

把"录制 trajectory"逻辑跟"渲染回放"逻辑解耦, 后续要加新 recorder 只改 RECORDERS dict.

用法 (单 panel):
    python experiments/sage_pr/preview_trajectory.py \
        --recorders v1_v3 \
        --map assets/maps/phase5/phase5_0001.txt

用法 (双 panel 对比):
    python experiments/sage_pr/preview_trajectory.py \
        --recorders v1_orig,v1_v3 \
        --map assets/maps/phase5/phase5_0001.txt

用法 (从失败清单批量看):
    python experiments/sage_pr/preview_trajectory.py \
        --recorders v1_v3 \
        --fails-list runs/sage_pr/v5_v3_fails.json

快捷键:
    SPACE 播放/暂停    R 重置    ←/→ 调速    N/PgDn 下张    P/PgUp 上张    ESC 退出
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
from typing import Callable, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pygame

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.paths import PROJECT_ROOT
from smartcar_sokoban.renderer import Renderer
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.explorer import (
    plan_exploration, exploration_complete, direction_to_action,
)
from smartcar_sokoban.solver.explorer_v2 import plan_exploration_v2
from smartcar_sokoban.solver.explorer_v3 import plan_exploration_v3
from experiments.sage_pr.build_dataset_v3 import parse_phase456_seeds


# ── 通用 low-level 记录 helper ──────────────────────────

def _record_apply_solver_move(eng: GameEngine, move, log: List[int]) -> bool:
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


def _record_solver_phase(eng: GameEngine, log: List[int],
                          *, time_limit: float = 30.0) -> bool:
    """跑 MultiBoxSolver, 录每一步 push 的 low-level. 返回是否解出."""
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
        return False
    directions = solver.solution_to_actions(solution)
    eng.discrete_step(6); log.append(6)
    for dx, dy in directions:
        a = direction_to_action(dx, dy)
        eng.discrete_step(a); log.append(a)
    return True


# ── Recorder 函数: (map_path, seed) → (action_log, info_dict) ────

def _recorder_v1_orig(map_path: str, seed: int):
    """V1 原始 (plan_exploration + MultiBoxSolver, 无任何补丁)."""
    random.seed(seed)
    eng = GameEngine(); eng.reset(map_path)
    log: List[int] = []
    info = {"won": False, "n_explore": 0, "n_solve": 0, "solver_ok": False}
    with contextlib.redirect_stdout(io.StringIO()):
        explore = plan_exploration(eng)
    info["n_explore"] = len(explore)
    random.seed(seed); eng.reset(map_path)
    for a in explore:
        eng.discrete_step(a); log.append(a)
    if not exploration_complete(eng.get_state()):
        info["err"] = "explore_incomplete"
        return log, info
    ok = _record_solver_phase(eng, log)
    info["solver_ok"] = ok
    info["n_solve"] = len(log) - info["n_explore"]
    info["won"] = eng.get_state().won
    return log, info


def _recorder_v1_v2(map_path: str, seed: int):
    """V1 + plan_exploration_v2 (推开障碍补丁) + MultiBoxSolver."""
    random.seed(seed)
    eng = GameEngine(); eng.reset(map_path)
    log: List[int] = []
    info = {"won": False, "n_explore": 0, "n_solve": 0, "solver_ok": False}
    with contextlib.redirect_stdout(io.StringIO()):
        explore = plan_exploration_v2(eng, max_retries=15)
    info["n_explore"] = len(explore)
    random.seed(seed); eng.reset(map_path)
    for a in explore:
        eng.discrete_step(a); log.append(a)
    if not exploration_complete(eng.get_state()):
        info["err"] = "explore_incomplete"
        return log, info
    ok = _record_solver_phase(eng, log)
    info["solver_ok"] = ok
    info["n_solve"] = len(log) - info["n_explore"]
    info["won"] = eng.get_state().won
    return log, info


def _recorder_v1_v3(map_path: str, seed: int):
    """V1 + plan_exploration_v3 (推开 + 拓扑配对) + MultiBoxSolver (当前正式版)."""
    random.seed(seed)
    eng = GameEngine(); eng.reset(map_path)
    log: List[int] = []
    info = {"won": False, "n_explore": 0, "n_solve": 0,
            "solver_ok": False, "forced_pairs": None}
    # 先看 v3 forced pairs
    from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs
    candidate = find_forced_pairs(eng.get_state())
    state_init = eng.get_state()
    real = [(i, j) for i, j in candidate
            if i < len(state_init.boxes) and j < len(state_init.targets)
            and state_init.boxes[i].class_id == state_init.targets[j].num_id]
    info["forced_pairs"] = real
    with contextlib.redirect_stdout(io.StringIO()):
        explore = plan_exploration_v3(eng, max_retries=15)
    info["n_explore"] = len(explore)
    random.seed(seed); eng.reset(map_path)
    # Re-apply: 注意 plan_exploration_v3 内部会 mutate state.seen_box_ids;
    # 重置后这些 seen 没了, 所以 record 时也要重做 mark.
    # 简化: 直接 deepcopy 第一次跑后的 final state... 但回放需要从头开始.
    # 这里取巧: 直接在 fresh engine 上跑 actions, 因为 seen_box_ids 在 engine FOV 中自然更新.
    # 但 v3 强制配对的 entity 在低层 actions 跑完后 FOV 不会真的识别 (车没真过去).
    # 处理: 完整应用 actions 后, 手动 mark seen 那些 forced.
    for a in explore:
        eng.discrete_step(a); log.append(a)
    s = eng.get_state()
    if real:
        for i, j in real:
            s.seen_box_ids.add(i)
            s.seen_target_ids.add(j)
    if not exploration_complete(s):
        info["err"] = "explore_incomplete"
        return log, info
    ok = _record_solver_phase(eng, log)
    info["solver_ok"] = ok
    info["n_solve"] = len(log) - info["n_explore"]
    info["won"] = eng.get_state().won
    return log, info


RECORDERS: Dict[str, Callable] = {
    "v1_orig": _recorder_v1_orig,
    "v1_v2": _recorder_v1_v2,
    "v1_v3": _recorder_v1_v3,
}


# ── headless Renderer (复用 Renderer 2D 渲染到任意 surface) ──

def _make_headless_renderer():
    cfg = GameConfig()
    cfg.render_mode = "simple"
    r = Renderer(cfg, str(PROJECT_ROOT))
    pygame.font.init()
    r._font = pygame.font.SysFont("Arial", 16, bold=True)
    r._big_font = pygame.font.SysFont("Arial", 24, bold=True)
    return r


# ── 主循环 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorders", default="v1_v3",
                        help=f"逗号分隔, 可选: {','.join(RECORDERS)}")
    parser.add_argument("--map", default=None, help="单图模式 (跟 --fails-list 二选一)")
    parser.add_argument("--fails-list", default=None,
                        help="按清单批量看 (JSON 列表, 每个 item 含 'map' 和 'seed')")
    parser.add_argument("--phase-only", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    recorders_list = [r.strip() for r in args.recorders.split(",") if r.strip()]
    for r in recorders_list:
        if r not in RECORDERS:
            print(f"unknown recorder: {r}"); sys.exit(1)
    print(f"recorders: {recorders_list}")

    # 构造任务列表
    if args.fails_list:
        path = args.fails_list if os.path.isabs(args.fails_list) else os.path.join(ROOT, args.fails_list)
        with open(path) as f:
            items = json.load(f)
        if args.phase_only is not None:
            items = [x for x in items if x.get("phase") == args.phase_only]
        task_list = [(x["map"], x.get("seed", 0)) for x in items]
    elif args.map:
        rel = args.map.replace("\\", "/")
        # 同 phase 文件夹列出 + 当前指针
        abs_path = os.path.join(ROOT, args.map) if not os.path.isabs(args.map) else args.map
        folder = os.path.dirname(abs_path)
        rel_folder = os.path.relpath(folder, ROOT).replace("\\", "/")
        files = sorted([
            f"{rel_folder}/{fn}" for fn in os.listdir(folder)
            if fn.endswith(".txt") and not fn.startswith("_")
        ])
        if args.seed is not None:
            task_list = [(m, args.seed) for m in files]
        else:
            vmap = parse_phase456_seeds(
                os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json"))
            task_list = [(m, vmap.get(m, [0])[0]) for m in files]
        # start idx
        try:
            start_i = files.index(rel)
        except ValueError:
            start_i = 0
        args.start = start_i
    else:
        print("need --map or --fails-list"); sys.exit(1)

    if not task_list:
        print("no maps"); sys.exit(1)

    # pygame setup
    pygame.init()
    renderer = _make_headless_renderer()
    W = renderer.w; H = renderer.h
    TITLE_H = 30
    GAP = 20
    n_panels = len(recorders_list)
    screen_w = W * n_panels + GAP * (n_panels - 1)
    screen = pygame.display.set_mode((screen_w, H + TITLE_H + 20))
    pygame.display.set_caption(f"trajectory preview: {','.join(recorders_list)}")
    clock = pygame.time.Clock()
    big_font = pygame.font.SysFont("Arial", 18, bold=True)
    small_font = pygame.font.SysFont("Consolas", 13)
    surfs = [pygame.Surface((W, H)) for _ in recorders_list]

    # state
    cur_idx = max(0, min(args.start, len(task_list) - 1))
    engines: List[GameEngine] = [GameEngine() for _ in recorders_list]
    logs: List[List[int]] = [[] for _ in recorders_list]
    infos: List[dict] = [{} for _ in recorders_list]
    step = 0
    playing = True
    step_delay = 0.05
    last_step = 0.0

    def load_map(idx: int):
        nonlocal step
        map_rel, seed = task_list[idx]
        print(f"\n[{idx+1}/{len(task_list)}] {map_rel} seed={seed}")
        for i, rname in enumerate(recorders_list):
            t0 = time.perf_counter()
            log, info = RECORDERS[rname](map_rel, seed)
            print(f"  {rname}: {len(log)} low-level, info={info} ({time.perf_counter()-t0:.1f}s)")
            logs[i] = log; infos[i] = info
            random.seed(seed)
            engines[i] = GameEngine(); engines[i].reset(map_rel)
        step = 0

    load_map(cur_idx)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_SPACE: playing = not playing
                elif event.key == pygame.K_r:
                    for i in range(len(recorders_list)):
                        map_rel, seed = task_list[cur_idx]
                        random.seed(seed)
                        engines[i] = GameEngine(); engines[i].reset(map_rel)
                    step = 0
                elif event.key == pygame.K_LEFT:
                    step_delay = min(0.5, step_delay + 0.02)
                elif event.key == pygame.K_RIGHT:
                    step_delay = max(0.01, step_delay - 0.02)
                elif event.key in (pygame.K_n, pygame.K_PAGEDOWN):
                    cur_idx = (cur_idx + 1) % len(task_list); load_map(cur_idx)
                elif event.key in (pygame.K_p, pygame.K_PAGEUP):
                    cur_idx = (cur_idx - 1) % len(task_list); load_map(cur_idx)

        now = time.perf_counter()
        if playing and now - last_step >= step_delay:
            max_len = max(len(l) for l in logs)
            if step < max_len:
                for i, log in enumerate(logs):
                    if step < len(log):
                        engines[i].discrete_step(log[step])
                step += 1
                last_step = now

        # render
        screen.fill((24, 24, 28))
        map_rel, seed = task_list[cur_idx]
        for i, rname in enumerate(recorders_list):
            renderer._render_2d(surfs[i], engines[i].get_state(), None, show_labels=True)
            x_off = i * (W + GAP)
            title = big_font.render(f"{rname} | {os.path.basename(map_rel)} seed={seed}",
                                     True, (220, 220, 240))
            screen.blit(title, (x_off + 8, 6))
            screen.blit(surfs[i], (x_off, TITLE_H))
            info = infos[i]
            won = engines[i].get_state().won
            info_text = (f"step {step}/{len(logs[i])} | "
                         f"explore={info.get('n_explore', 0)} "
                         f"solve={info.get('n_solve', 0)} "
                         f"won={'YES' if won else info.get('err', 'no')}")
            if info.get("forced_pairs"):
                info_text += f" | forced={info['forced_pairs']}"
            t = small_font.render(info_text, True, (200, 240, 200))
            screen.blit(t, (x_off + 8, H + TITLE_H + 2))
            if i > 0:
                pygame.draw.rect(screen, (60, 60, 70),
                                 (x_off - GAP, 0, GAP, H + TITLE_H + 20))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
