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


def _record_apply_inspect(eng: GameEngine, cand, log: List[int]) -> bool:
    """inspect 候选展开 + 记 log."""
    from experiments.sage_pr.belief_ida_solver import _heading_to_angle
    from smartcar_sokoban.solver.explorer import compute_facing_actions
    if cand.viewpoint_col is None:
        return False
    eng.discrete_step(6); log.append(6)
    state = eng.get_state()
    obstacles = set()
    for b in state.boxes: obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs: obstacles.add(pos_to_grid(bm.x, bm.y))
    car_grid = pos_to_grid(state.car_x, state.car_y)
    target = (cand.viewpoint_col, cand.viewpoint_row)
    if car_grid != target:
        path = bfs_path(car_grid, target, state.grid, obstacles)
        if path is None: return False
        for pdx, pdy in path:
            a = direction_to_abs_action(pdx, pdy)
            eng.discrete_step(a); log.append(a)
    state = eng.get_state()
    rot_acts = compute_facing_actions(state.car_angle,
                                       _heading_to_angle(cand.inspect_heading or 0))
    for a in rot_acts:
        eng.discrete_step(a); log.append(a)
    return True


def _record_solver_phase(eng: GameEngine, log: List[int],
                          *, time_limit: float = 60.0) -> bool:
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


_MODEL_CKPT_GLOBAL: Dict[str, object] = {"model": None, "device": None, "ckpt": None}


def _recorder_model(map_path: str, seed: int):
    """模型 rollout — 用 v3_large9 (或 _MODEL_CKPT_GLOBAL["ckpt"] 覆盖) 跑 top-1 greedy, 录每一步 low-level."""
    import numpy as np
    import torch
    from experiments.sage_pr.model import build_model_from_ckpt
    from experiments.sage_pr.evaluate_sage_pr import (
        candidate_to_solver_move, _state_signature,
    )
    from experiments.sage_pr.belief_ida_solver import apply_inspect
    from experiments.sage_pr.build_dataset_v3 import apply_solver_move
    from smartcar_sokoban.symbolic.belief import BeliefState
    from smartcar_sokoban.symbolic.features import compute_domain_features
    from smartcar_sokoban.symbolic.candidates import (
        generate_candidates, candidates_legality_mask,
    )
    from smartcar_sokoban.symbolic.cand_features import encode_candidates
    from smartcar_sokoban.symbolic.grid_tensor import (
        build_grid_tensor, build_global_features,
    )

    ckpt = _MODEL_CKPT_GLOBAL["ckpt"] or ".agent/sage_pr/runs/v3_large9/best.pt"
    if _MODEL_CKPT_GLOBAL["model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model_from_ckpt(ckpt, device=device)
        model.eval()
        _MODEL_CKPT_GLOBAL["model"] = model
        _MODEL_CKPT_GLOBAL["device"] = device
    model = _MODEL_CKPT_GLOBAL["model"]
    device = _MODEL_CKPT_GLOBAL["device"]

    random.seed(seed)
    eng = GameEngine(); eng.reset(map_path)
    log: List[int] = []
    info = {"won": False, "n_macros": 0}

    visited = {_state_signature(eng.get_state())}
    step_limit = 80
    top_k = 1   # 纯 greedy; top_k>1 会引入 visited-skip 启发, 跟 eval (beam search) 不一致
    for step in range(step_limit):
        s = eng.get_state()
        if s.won:
            info["won"] = True
            info["n_macros"] = step
            return log, info
        bs = BeliefState.from_engine_state(s, fully_observed=True)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat, enforce_sigma_lock=False)
        legal = [c.legal for c in cands]
        if not any(legal):
            info["n_macros"] = step
            return log, info

        X_grid = build_grid_tensor(bs, feat).transpose(2, 0, 1)
        X_cand = encode_candidates(cands, bs, feat)
        u_global = build_global_features(bs, feat)
        mask = candidates_legality_mask(cands)
        xg = torch.from_numpy(X_grid).unsqueeze(0).to(device)
        xc = torch.from_numpy(X_cand).unsqueeze(0).to(device)
        ug = torch.from_numpy(u_global).unsqueeze(0).to(device)
        mk = torch.from_numpy(mask).unsqueeze(0).to(device)
        with torch.no_grad():
            score, _, _, _, _ = model(xg, xc, ug, mk)
        sn = score.cpu().numpy().squeeze(0)
        sn[~np.array(legal)] = -1e9
        order = np.argsort(-sn)

        # visited check on clone (用未 wrap 的 apply_solver_move / apply_inspect, 不会污染 log)
        chosen = None
        for k in range(min(top_k, len(order))):
            idx = int(order[k])
            cand = cands[idx]
            if not cand.legal: continue
            eng_clone = copy.deepcopy(eng)
            if cand.type == "inspect":
                ok = apply_inspect(eng_clone, cand)
            else:
                m = candidate_to_solver_move(cand, bs)
                ok = m is not None and apply_solver_move(eng_clone, m)
            if not ok: continue
            sig = _state_signature(eng_clone.get_state())
            if sig in visited: continue
            chosen = idx; break
        if chosen is None:
            chosen = int(order[0])
        cand = cands[chosen]

        # 真正应用到 eng — 用 _record_apply_solver_move 手动记 log (避免 monkey-patch closure bug)
        if cand.type == "inspect":
            ok = _record_apply_inspect(eng, cand, log)
        else:
            m = candidate_to_solver_move(cand, bs)
            ok = m is not None and _record_apply_solver_move(eng, m, log)
        if not ok:
            info["n_macros"] = step
            return log, info
        visited.add(_state_signature(eng.get_state()))

    info["n_macros"] = step_limit
    info["won"] = eng.get_state().won
    return log, info


def _recorder_model_search(map_path: str, seed: int):
    """模型 beam search 推理 (beam=8 lookahead=25) — 这才是 eval 报告 96-100% 的真实路径."""
    import torch
    from experiments.sage_pr.model import build_model_from_ckpt
    from experiments.sage_pr.rollout_search_eval import (
        rollout_search_step, _apply_any,
    )
    from experiments.sage_pr.evaluate_sage_pr import _state_signature
    from smartcar_sokoban.symbolic.belief import BeliefState

    ckpt = _MODEL_CKPT_GLOBAL["ckpt"] or ".agent/sage_pr/runs/v3_large9/best.pt"
    if _MODEL_CKPT_GLOBAL["model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model_from_ckpt(ckpt, device=device)
        model.eval()
        _MODEL_CKPT_GLOBAL["model"] = model
        _MODEL_CKPT_GLOBAL["device"] = device
    model = _MODEL_CKPT_GLOBAL["model"]
    device = _MODEL_CKPT_GLOBAL["device"]

    from experiments.sage_pr.evaluate_sage_pr import candidate_to_solver_move
    random.seed(seed)
    eng = GameEngine(); eng.reset(map_path)
    log: List[int] = []
    info = {"won": False, "n_macros": 0, "n_explore_actions": 0}

    # 部署架构: 先跑 BFS explorer 算法识别全部 entity, 然后 NN 接管 push
    # 这样可视化里能看到完整流程: 探索阶段 (车转头/平移) → 推箱阶段 (NN)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            explore_actions = plan_exploration_v3(eng, max_retries=15)
        except Exception:
            explore_actions = []
    log.extend(explore_actions)
    info["n_explore_actions"] = len(explore_actions)
    if not exploration_complete(eng.get_state()):
        # explorer 失败 → reset + NN 从初始接管 (fallback)
        random.seed(seed)
        eng = GameEngine(); eng.reset(map_path)
        log.clear()
        info["n_explore_actions"] = 0

    visited = {_state_signature(eng.get_state())}
    beam_width = 8
    lookahead = 25
    step_limit = 60
    for step in range(step_limit):
        s = eng.get_state()
        if s.won:
            info["won"] = True; info["n_macros"] = step
            return log, info
        res = rollout_search_step(
            model, device, eng, visited,
            beam_width=beam_width, lookahead=lookahead,
            fully_observed=True, enforce_sigma_lock=False,
        )
        if res is None:
            info["n_macros"] = step; return log, info
        chosen_idx, cand = res
        bs_now = BeliefState.from_engine_state(eng.get_state(), fully_observed=True)
        if cand.type == "inspect":
            ok = _record_apply_inspect(eng, cand, log)
        else:
            m = candidate_to_solver_move(cand, bs_now)
            ok = m is not None and _record_apply_solver_move(eng, m, log)
        if not ok:
            info["n_macros"] = step; return log, info
        visited.add(_state_signature(eng.get_state()))
    info["n_macros"] = step_limit
    info["won"] = eng.get_state().won
    return log, info


def _recorder_training_data(map_path: str, seed: int):
    """直接复现 build_dataset_v5._exact_fallback_episode 的逻辑 — 实际丢给模型训练的轨迹.

    plan_exploration_v3 + MultiBoxSolver (time_limit=60s), 跟 v1_v3 等价,
    但名字标明 = "训练数据原始轨迹", 方便诊断模型 vs 训练数据偏差.
    """
    random.seed(seed)
    eng = GameEngine(); eng.reset(map_path)
    log: List[int] = []
    info = {"won": False, "n_explore": 0, "n_solve": 0,
            "solver_ok": False, "_label": "training_data (build_dataset_v5 等价)"}
    with contextlib.redirect_stdout(io.StringIO()):
        explore = plan_exploration_v3(eng, max_retries=15)
    info["n_explore"] = len(explore)
    random.seed(seed); eng.reset(map_path)
    for a in explore:
        eng.discrete_step(a); log.append(a)
    s = eng.get_state()
    if not exploration_complete(s):
        info["err"] = "explore_incomplete"
        return log, info
    ok = _record_solver_phase(eng, log, time_limit=60.0)
    info["solver_ok"] = ok
    info["n_solve"] = len(log) - info["n_explore"]
    info["won"] = eng.get_state().won
    return log, info


RECORDERS: Dict[str, Callable] = {
    "v1_orig": _recorder_v1_orig,
    "v1_v2": _recorder_v1_v2,
    "v1_v3": _recorder_v1_v3,
    "training_data": _recorder_training_data,
    "model": _recorder_model,
    "model_search": _recorder_model_search,
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
    parser.add_argument("--ckpt", default=".agent/sage_pr/runs/v5_push_only/best.pt",
                        help="模型 ckpt 路径 (model / model_search recorder 用)")
    args = parser.parse_args()

    # 推到 GLOBAL 让 recorder 读到
    _MODEL_CKPT_GLOBAL["ckpt"] = args.ckpt

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
