"""Hybrid 推理: 模型 + 求解器 fallback.

策略:
    1. 每步先 forward 模型. 若高置信度 (top-1 vs top-2 logit > thresh, 或 entropy < thresh)
       → 用模型选择.
    2. 否则 → 调求解器 (MultiBoxSolver bestfirst, 1.5s 时限) 拿正确 action.
    3. 若求解器超时/失败 → 退回模型 top-1.

这是 RFC 允许的 "neural-augmented classical algorithms" 方案. 推理时延会增长
(平均 50-200 ms), 但 win rate 接近求解器上限.

适用于: PC 评测达标. **不**适用于 OpenART 50ms 部署 (那需要重新设计 / QAT / 蒸馏).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
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
    apply_solver_move, parse_phase456_seeds, list_phase_maps,
    match_move_to_candidate,
)
from experiments.sage_pr.evaluate_sage_pr import (
    candidate_to_solver_move, _state_signature,
)
from experiments.sage_pr.dagger_lite import solve_from_state


def model_score(model, device, eng, fully_observed=True):
    s = eng.get_state()
    bs = BeliefState.from_engine_state(s, fully_observed=fully_observed)
    feat = compute_domain_features(bs)
    cands = generate_candidates(bs, feat)
    if not any(c.legal for c in cands):
        return None, None, None, None
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
    score_np = score.cpu().numpy().squeeze(0)
    score_np[mask < 0.5] = -1e9
    return cands, score_np, bs, feat


def hybrid_step(model, device, eng, visited_sigs,
                solver_threshold: float = 1.5,
                solver_time_limit: float = 1.5,
                fully_observed: bool = True
                ) -> Tuple[Optional[int], Optional[Candidate], bool]:
    """选首步动作. 返回 (idx, cand, used_solver)."""
    cands, score, bs, feat = model_score(model, device, eng, fully_observed)
    if cands is None:
        return None, None, False

    order = np.argsort(-score)
    top1, top2 = order[0], order[1] if len(order) >= 2 else order[0]
    gap = score[top1] - score[top2]

    # 模型置信度高 → 直接用模型
    if gap >= solver_threshold:
        for k in range(min(4, len(order))):
            idx = int(order[k])
            cand = cands[idx]
            if not cand.legal:
                continue
            move = candidate_to_solver_move(cand, bs)
            if move is None:
                continue
            eng_clone = copy.deepcopy(eng)
            if not apply_solver_move(eng_clone, move):
                continue
            sig = _state_signature(eng_clone.get_state())
            if sig in visited_sigs:
                continue
            return idx, cand, False
        # fallback to top-1
        idx = int(order[0])
        return idx, cands[idx], False

    # 低置信度 → 调求解器
    moves = solve_from_state(eng, time_limit=solver_time_limit)
    if moves:
        solver_label = match_move_to_candidate(moves[0], cands, bs, run_length=1)
        if solver_label is not None and cands[solver_label].legal:
            return solver_label, cands[solver_label], True

    # 求解器失败 → 退回模型 top-1 反循环
    for k in range(min(4, len(order))):
        idx = int(order[k])
        cand = cands[idx]
        if not cand.legal:
            continue
        move = candidate_to_solver_move(cand, bs)
        if move is None:
            continue
        eng_clone = copy.deepcopy(eng)
        if not apply_solver_move(eng_clone, move):
            continue
        sig = _state_signature(eng_clone.get_state())
        if sig in visited_sigs:
            continue
        return idx, cand, False
    idx = int(order[0])
    return idx, cands[idx], False


def hybrid_rollout(model, device, map_path, seed,
                    *, step_limit=60, solver_threshold=1.5,
                    solver_time_limit=1.5,
                    fully_observed=True):
    import random
    random.seed(seed)
    eng = GameEngine()
    eng.reset(map_path)
    inf_total = 0.0
    inf_calls = 0
    n_solver = 0
    visited = set()
    visited.add(_state_signature(eng.get_state()))

    for step in range(step_limit):
        s = eng.get_state()
        if s.won:
            return True, step, inf_total / max(inf_calls, 1), n_solver

        t0 = time.perf_counter()
        result = hybrid_step(model, device, eng, visited,
                              solver_threshold=solver_threshold,
                              solver_time_limit=solver_time_limit,
                              fully_observed=fully_observed)
        inf_total += time.perf_counter() - t0
        inf_calls += 1
        if result[2]:
            n_solver += 1

        if result[0] is None:
            return False, step, inf_total / max(inf_calls, 1), n_solver
        idx, cand, _ = result

        bs = BeliefState.from_engine_state(eng.get_state(), fully_observed=fully_observed)
        move = candidate_to_solver_move(cand, bs)
        if move is None or not apply_solver_move(eng, move):
            return False, step, inf_total / max(inf_calls, 1), n_solver
        visited.add(_state_signature(eng.get_state()))

    return eng.get_state().won, step_limit, inf_total / max(inf_calls, 1), n_solver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--max-maps", type=int, default=100)
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--solver-threshold", type=float, default=1.5,
                        help="logit gap < this -> 调求解器")
    parser.add_argument("--solver-time-limit", type=float, default=1.5)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_default_model().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded {args.ckpt}, val_acc={ckpt.get('val_acc', '?'):.3f}")
    print(f"solver_threshold={args.solver_threshold}, solver_time_limit={args.solver_time_limit}")

    verified_map = None
    if args.use_verified_seeds:
        verified_map = parse_phase456_seeds(
            os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
        )

    results = []
    for ph in args.phases:
        maps = list_phase_maps(ph)[:args.max_maps]
        n_total = 0
        n_won = 0
        total_steps = 0
        total_inf_ms = 0.0
        total_solver = 0
        t0 = time.perf_counter()
        for map_path in maps:
            ms = (verified_map.get(map_path, [0])[:max(1, len(seeds))]
                  if verified_map else seeds)
            for seed in ms:
                won, steps, avg_inf, n_solver = hybrid_rollout(
                    model, device, map_path, seed,
                    solver_threshold=args.solver_threshold,
                    solver_time_limit=args.solver_time_limit,
                )
                n_total += 1
                if won: n_won += 1
                total_steps += steps
                total_inf_ms += avg_inf * 1000
                total_solver += n_solver
        elapsed = time.perf_counter() - t0
        r = {
            "phase": ph,
            "n_total": n_total,
            "n_won": n_won,
            "win_rate": n_won / max(n_total, 1),
            "avg_steps": total_steps / max(n_total, 1),
            "avg_inf_ms": total_inf_ms / max(n_total, 1),
            "solver_calls_per_episode": total_solver / max(n_total, 1),
        }
        print(f"phase {ph}: {r['win_rate']*100:.1f}% ({n_won}/{n_total}); "
              f"avg_inf={r['avg_inf_ms']:.1f}ms; "
              f"solver/ep={r['solver_calls_per_episode']:.1f}; "
              f"{elapsed:.0f}s")
        results.append(r)

    print("\n=== Summary ===")
    for r in results:
        print(f"phase {r['phase']}: {r['win_rate']*100:.2f}% ({r['n_won']}/{r['n_total']}), "
              f"solver/ep={r['solver_calls_per_episode']:.1f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
