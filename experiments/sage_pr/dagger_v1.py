"""V1 路线 DAgger: 模型 rollout (含外挂 explorer) + 老师 (god-mode 解当前状态) 重打 label.

每步:
  1. 状态 s (post-explore): 模型选 cand a_m
  2. 老师 = MultiBoxSolver(s).solve() 第一步 → match 到 candidate a_t
  3. 录 sample (s 特征, label=a_t)
  4. 用 a_m 推进 (DAgger: 模型自己的分布)
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import multiprocessing as mp
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.explorer_v3 import plan_exploration_v3
from smartcar_sokoban.solver.explorer import exploration_complete
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import (
    generate_candidates, candidates_legality_mask,
)
from smartcar_sokoban.symbolic.cand_features import encode_candidates
from smartcar_sokoban.symbolic.grid_tensor import build_grid_tensor, build_global_features
from experiments.sage_pr.build_dataset_v3 import (
    apply_solver_move, parse_phase456_seeds, list_phase_maps, save_dataset, Sample,
    match_move_to_candidate, SOURCE_AUTO,
)
from experiments.sage_pr.evaluate_sage_pr import candidate_to_solver_move, _state_signature
from experiments.sage_pr.model import build_default_model, build_model_from_ckpt


def teacher_label_v1(eng: GameEngine, cands, bs: BeliefState,
                      time_limit: float = 5.0) -> Optional[int]:
    """老师 = MultiBoxSolver from 当前 engine state."""
    state = eng.get_state()
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    if not boxes:
        return None
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            moves = solver.solve(max_cost=200, time_limit=time_limit, strategy="auto")
        except Exception:
            moves = None
    if not moves:
        return None
    return match_move_to_candidate(moves[0], cands, bs, run_length=1)


def collect_v1_dagger_episode(model, device, map_path: str, seed: int, phase: int,
                                *, step_limit: int = 80, top_k: int = 4,
                                teacher_time_limit: float = 3.0
                                ) -> Tuple[List[Sample], Dict]:
    info = {"steps": 0, "won": False, "n_disagree": 0,
            "n_teacher_no_label": 0, "phase": phase}

    random.seed(seed)
    eng = GameEngine()
    eng.reset(map_path)
    # 跑 explorer 让 entities identified
    with contextlib.redirect_stdout(io.StringIO()):
        plan_exploration_v3(eng, max_retries=15)
    if not exploration_complete(eng.get_state()):
        return [], info

    samples: List[Sample] = []
    visited = {_state_signature(eng.get_state())}

    for step in range(step_limit):
        s = eng.get_state()
        if s.won:
            info["won"] = True; break
        bs = BeliefState.from_engine_state(s, fully_observed=True)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat, enforce_sigma_lock=False)
        legal = [c.legal for c in cands]
        if not any(legal): break

        t_label = teacher_label_v1(eng, cands, bs, teacher_time_limit)
        if t_label is None:
            info["n_teacher_no_label"] += 1
            break

        samples.append(Sample(
            X_grid=build_grid_tensor(bs, feat),
            X_cand=encode_candidates(cands, bs, feat),
            u_global=build_global_features(bs, feat),
            mask=candidates_legality_mask(cands),
            label=t_label, phase=phase, source=SOURCE_AUTO,
        ))

        # 若启用 rollout-advance, 用 1-step lookahead 评估每个 cand 进度 (用 push_dist 减少)
        # 否则用 model top-1
        # 模型推理
        xg = torch.from_numpy(samples[-1].X_grid.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)
        xc = torch.from_numpy(samples[-1].X_cand.astype(np.float32)).unsqueeze(0).to(device)
        ug = torch.from_numpy(samples[-1].u_global.astype(np.float32)).unsqueeze(0).to(device)
        mk = torch.from_numpy(samples[-1].mask.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            score, _, _, _, _ = model(xg, xc, ug, mk)
        sn = score.cpu().numpy().squeeze(0)
        sn[~np.array(legal)] = -1e9
        order = np.argsort(-sn)
        if int(order[0]) != t_label:
            info["n_disagree"] += 1

        # 模型 top-k 推进
        chosen = None
        for k in range(min(top_k, len(order))):
            idx = int(order[k])
            cand = cands[idx]
            if not cand.legal: continue
            eng_clone = copy.deepcopy(eng)
            m = candidate_to_solver_move(cand, bs)
            if not m: continue
            if not apply_solver_move(eng_clone, m): continue
            sig = _state_signature(eng_clone.get_state())
            if sig in visited: continue
            chosen = idx; break
        if chosen is None:
            chosen = int(order[0])
        cand = cands[chosen]
        m = candidate_to_solver_move(cand, bs)
        if not m or not apply_solver_move(eng, m):
            break
        visited.add(_state_signature(eng.get_state()))
        info["steps"] += 1

    return samples, info


_MODEL_GLOBAL = {"model": None, "device": None}


def _worker_init(ckpt_path: str):
    """子进程初始化: 加载模型到 CPU (避免多 process 抢 GPU)."""
    device = torch.device("cpu")
    model = build_model_from_ckpt(ckpt_path, device=device)
    model.eval()
    _MODEL_GLOBAL["model"] = model
    _MODEL_GLOBAL["device"] = device


def _worker_episode(args):
    map_path, seed, phase, step_limit, top_k, teacher_tl = args
    model = _MODEL_GLOBAL["model"]
    device = _MODEL_GLOBAL["device"]
    samples, info = collect_v1_dagger_episode(
        model, device, map_path, seed, phase,
        step_limit=step_limit, top_k=top_k,
        teacher_time_limit=teacher_tl,
    )
    return samples, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--phases", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--max-maps", type=int, default=200)
    parser.add_argument("--max-seeds-per-map", type=int, default=1)
    parser.add_argument("--step-limit", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--teacher-time-limit", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_ckpt(args.ckpt, device=device)
    model.eval()
    print(f"loaded {args.ckpt}")

    verified = parse_phase456_seeds(
        os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
    ) if args.use_verified_seeds else None

    all_tasks = []
    for phase in args.phases:
        if args.use_verified_seeds and verified:
            items = sorted([(k, v) for k, v in verified.items() if f"phase{phase}/" in k])[:args.max_maps]
            for map_path, seeds in items:
                for seed in seeds[:args.max_seeds_per_map]:
                    all_tasks.append((map_path, seed, phase, args.step_limit,
                                      args.top_k, args.teacher_time_limit))
        else:
            maps = list_phase_maps(phase, args.max_maps)
            for m in maps:
                all_tasks.append((m, 0, phase, args.step_limit,
                                  args.top_k, args.teacher_time_limit))

    print(f"total tasks: {len(all_tasks)}, workers: {args.workers}")
    all_samples: List[Sample] = []
    phase_stats: Dict[int, Dict] = {p: {"n_won": 0, "n_total": 0, "samples": 0, "disagree": 0}
                                     for p in args.phases}
    t0 = time.perf_counter()
    if args.workers <= 1:
        for tsk in all_tasks:
            samples, info = _worker_episode(tsk)
            _MODEL_GLOBAL["model"] = model
            _MODEL_GLOBAL["device"] = device
            p = tsk[2]
            phase_stats[p]["n_total"] += 1
            if info["won"]: phase_stats[p]["n_won"] += 1
            phase_stats[p]["samples"] += len(samples)
            phase_stats[p]["disagree"] += info["n_disagree"]
            all_samples.extend(samples)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_worker_init,
                       initargs=(args.ckpt,)) as pool:
            for i, (samples, info) in enumerate(
                    pool.imap_unordered(_worker_episode, all_tasks, chunksize=2)):
                # find phase by sample.phase
                p = info.get("phase")
                if p is None and samples:
                    p = samples[0].phase
                if p is None:
                    p = -1
                if p in phase_stats:
                    phase_stats[p]["n_total"] += 1
                    if info["won"]: phase_stats[p]["n_won"] += 1
                    phase_stats[p]["samples"] += len(samples)
                    phase_stats[p]["disagree"] += info["n_disagree"]
                all_samples.extend(samples)
                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{len(all_tasks)} ({time.perf_counter()-t0:.0f}s)")

    elapsed = time.perf_counter() - t0
    print(f"\ndone in {elapsed:.0f}s")
    for p, st in sorted(phase_stats.items()):
        if st["n_total"] > 0:
            print(f"  phase {p}: won {st['n_won']}/{st['n_total']} "
                  f"({100*st['n_won']/st['n_total']:.1f}%) "
                  f"samples={st['samples']} disagree={st['disagree']}")

    print(f"\nTotal samples: {len(all_samples)}")
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    save_dataset(all_samples, out_path)


if __name__ == "__main__":
    main()
