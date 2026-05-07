"""DAgger lite — 一轮 (rollout 失败 → solver 标签 → 加入数据).

简化策略:
    1. 用当前模型 rollout 每张图.
    2. 失败的 (won=False) 收集 trajectory 中所有状态.
    3. 对每个状态调 solver, 找正确 action.
    4. 保存为 npz 加入下一轮训练.

输出: .agent/sage_pr/dagger_<tag>.npz
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import (
    Candidate, generate_candidates, candidates_legality_mask, MAX_CANDIDATES,
)
from smartcar_sokoban.symbolic.cand_features import encode_candidates
from smartcar_sokoban.symbolic.grid_tensor import (
    build_grid_tensor, build_global_features, GRID_TENSOR_CHANNELS, GLOBAL_DIM,
)
from experiments.sage_pr.model import build_default_model
from experiments.sage_pr.build_dataset_v3 import (
    apply_solver_move, parse_phase456_seeds, list_phase_maps,
    save_dataset, Sample, match_move_to_candidate, SOURCE_BF,
)
from experiments.sage_pr.evaluate_sage_pr import (
    candidate_to_solver_move, _state_signature,
)


def solve_from_state(eng: GameEngine, time_limit: float = 10.0) -> Optional[List]:
    """从当前 engine state 开 solver. 返回 moves 或 None."""
    state = eng.get_state()
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    with contextlib.redirect_stdout(io.StringIO()):
        return solver.solve(max_cost=300, time_limit=time_limit, strategy="best_first")


def collect_dagger_episode(model, device, map_path: str, seed: int,
                           *, step_limit: int = 30,
                           top_k: int = 4,
                           solver_time_limit: float = 2.0) -> List[Sample]:
    """Rollout 一个 episode, 返回 model-failed 状态的 sample (用 solver 重标)."""
    import copy, random
    random.seed(seed)

    eng = GameEngine()
    state = eng.reset(map_path)
    samples: List[Sample] = []
    visited_sigs = set()
    visited_sigs.add(_state_signature(eng.get_state()))

    for step in range(step_limit):
        s = eng.get_state()
        if s.won:
            return samples
        bs = BeliefState.from_engine_state(s, fully_observed=True)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat)
        legal = [c.legal for c in cands]
        if not any(legal):
            return samples

        X_grid = build_grid_tensor(bs, feat)
        X_cand = encode_candidates(cands, bs, feat)
        u_global = build_global_features(bs, feat)
        mask = candidates_legality_mask(cands)

        xg_t = torch.from_numpy(X_grid.transpose(2, 0, 1)).unsqueeze(0).to(device)
        xc_t = torch.from_numpy(X_cand).unsqueeze(0).to(device)
        ug_t = torch.from_numpy(u_global).unsqueeze(0).to(device)
        mk_t = torch.from_numpy(mask).unsqueeze(0).to(device)

        with torch.no_grad():
            score, _, _, _, _ = model(xg_t, xc_t, ug_t, mk_t)
        score_np = score.cpu().numpy().squeeze(0)
        score_np[~np.array(legal)] = -1e9
        order = np.argsort(-score_np)
        model_idx = int(order[0])

        # 调 solver 找正确 action
        moves = solve_from_state(eng, time_limit=solver_time_limit)
        if not moves:
            # 当前状态不可解 → 跳过 (可能是死锁了)
            return samples
        solver_label = match_move_to_candidate(moves[0], cands, bs, run_length=1)
        if solver_label is None:
            # solver move 在我们 candidates 里没匹配 → 跳过
            return samples

        # DAgger 标准做法: 把 solver_label 加入数据 (无论 model 同不同意).
        # 这能修正模型在自身 rollout 路径上的所有错误信念.
        samples.append(Sample(
            X_grid=X_grid, X_cand=X_cand, u_global=u_global, mask=mask,
            label=solver_label, phase=0, source=SOURCE_BF,
        ))

        # 选个候选执行 (用 top-k 防循环, 用 model 的选择)
        chosen_idx = None
        for k in range(min(top_k, len(order))):
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
            chosen_idx = idx
            break
        if chosen_idx is None:
            chosen_idx = int(order[0]) if cands[int(order[0])].legal else None
        if chosen_idx is None:
            return samples

        cand = cands[chosen_idx]
        move = candidate_to_solver_move(cand, bs)
        if not apply_solver_move(eng, move):
            return samples
        visited_sigs.add(_state_signature(eng.get_state()))

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--phases", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--max-maps", type=int, default=200)
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--step-limit", type=int, default=30,
                        help="rollout 步数上限. 越小越快, 但收集少.")
    parser.add_argument("--solver-time-limit", type=float, default=2.0,
                        help="每步 solver 调用时限.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_default_model().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded {args.ckpt}, val_acc={ckpt.get('val_acc', '?'):.3f}")

    verified = parse_phase456_seeds(
        os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
    ) if args.use_verified_seeds else None

    all_samples: List[Sample] = []
    t0 = time.perf_counter()
    for ph in args.phases:
        maps = list_phase_maps(ph)[:args.max_maps]
        n_maps = 0
        n_samples_phase = 0
        for map_path in maps:
            ms = (verified.get(map_path, [0])[:max(1, len(seeds))]
                  if verified else seeds)
            for seed in ms:
                ss = collect_dagger_episode(model, device, map_path, seed,
                                             step_limit=args.step_limit)
                for s in ss:
                    s.phase = ph
                all_samples.extend(ss)
                n_samples_phase += len(ss)
            n_maps += 1
            if n_maps % 20 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  phase {ph}: {n_maps} maps, {n_samples_phase} samples, {elapsed:.0f}s")
        print(f"phase {ph}: {n_maps} maps -> {n_samples_phase} dagger samples")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    save_dataset(all_samples, out_path)


if __name__ == "__main__":
    main()
