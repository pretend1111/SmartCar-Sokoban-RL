"""V2-aware DAgger: 模型 V2 mode rollout + 老师 (god-mode + 抑制场) 重打 label.

每步:
  1. 模型 V2 mode 选 action a_m (基于 partial-obs + enforce_sigma_lock)
  2. 老师在当前 engine 上重新 god-mode 求解 → 给出正确 action a_t (push 或 inspect)
  3. 录 sample (state, label=a_t)
  4. 用 a_m 推进 engine (DAgger: 在模型自己的轨迹上学)

输出:
  .agent/sage_pr/dagger_v2_rN.npz (跟 v6 同 schema)
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

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import (
    Candidate, generate_candidates, candidates_legality_mask, MAX_CANDIDATES,
)
from smartcar_sokoban.symbolic.cand_features import encode_candidates
from smartcar_sokoban.symbolic.grid_tensor import build_grid_tensor, build_global_features
from experiments.sage_pr.build_dataset_v3 import (
    apply_solver_move, parse_phase456_seeds, list_phase_maps, save_dataset, Sample,
    match_move_to_candidate,
)
from experiments.sage_pr.belief_ida_solver import apply_inspect
from experiments.sage_pr.evaluate_sage_pr import candidate_to_solver_move
from experiments.sage_pr.build_dataset_v6 import pick_inspect_for_unlock, SOURCE_V2
from experiments.sage_pr.model import build_default_model


def teacher_label(eng: GameEngine, cands: List[Candidate], bs: BeliefState,
                   *, time_limit: float = 5.0
                   ) -> Tuple[Optional[int], str]:
    """老师 (god-mode + 抑制场) 在当前 engine state 上重新求解, 给出 label.

    Returns (label_idx, kind) where kind ∈ {"push", "inspect", "no_solve", "no_match"}.
    """
    state = eng.get_state()
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    if not boxes:
        return None, "won"
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
        return None, "no_solve"

    move = moves[0]
    label = match_move_to_candidate(move, cands, bs, run_length=1)
    if label is not None:
        return label, "push"

    # push 被抑制 → 选 inspect
    rb, rt = -1, -1
    if move[0] == "box":
        op, cid = move[1]
        for j, b in enumerate(bs.boxes):
            if (b.col, b.row) == op:
                rb = j
                break
        for j, t in enumerate(bs.targets):
            if t.num_id == cid:
                rt = j
                break
    ins_label = pick_inspect_for_unlock(bs, cands, rb, rt)
    if ins_label is not None:
        return ins_label, "inspect"
    return None, "no_match"


def _state_signature(state):
    return (
        round(state.car_x * 2),
        round(state.car_y * 2),
        frozenset((round(b.x * 2), round(b.y * 2)) for b in state.boxes),
        frozenset((round(b.x * 2), round(b.y * 2)) for b in state.bombs),
    )


def collect_v2_dagger_episode(model, device, map_path: str, seed: int, phase: int,
                                *, step_limit: int = 80,
                                top_k: int = 4,
                                teacher_time_limit: float = 5.0
                                ) -> Tuple[List[Sample], Dict]:
    info = {"steps": 0, "won": False, "n_disagree": 0,
            "n_teacher_no_solve": 0, "n_teacher_no_match": 0}

    random.seed(seed)
    eng = GameEngine()
    eng.reset(map_path)
    samples: List[Sample] = []
    visited_sigs = {_state_signature(eng.get_state())}

    for step in range(step_limit):
        s = eng.get_state()
        if s.won:
            info["won"] = True
            break
        bs = BeliefState.from_engine_state(s, fully_observed=False)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat, enforce_sigma_lock=True)
        legal = [c.legal for c in cands]
        if not any(legal):
            break

        # 老师标签
        t_label, t_kind = teacher_label(eng, cands, bs, time_limit=teacher_time_limit)
        if t_label is None:
            if t_kind == "no_solve":
                info["n_teacher_no_solve"] += 1
            elif t_kind == "no_match":
                info["n_teacher_no_match"] += 1
            break

        # 录 sample
        samples.append(Sample(
            X_grid=build_grid_tensor(bs, feat),
            X_cand=encode_candidates(cands, bs, feat),
            u_global=build_global_features(bs, feat),
            mask=candidates_legality_mask(cands),
            label=t_label, phase=phase, source=SOURCE_V2,
        ))

        # 模型推理 (用于决定下一步 engine 推进)
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

        # DAgger: 用模型 top-k (避循环) 推进 engine
        chosen = None
        for k in range(min(top_k, len(order))):
            idx = int(order[k])
            cand = cands[idx]
            if not cand.legal:
                continue
            eng_clone = copy.deepcopy(eng)
            if cand.type == "inspect":
                ok = apply_inspect(eng_clone, cand)
            else:
                m = candidate_to_solver_move(cand, bs)
                ok = apply_solver_move(eng_clone, m) if m else False
            if not ok:
                continue
            sig = _state_signature(eng_clone.get_state())
            if sig in visited_sigs:
                continue
            chosen = idx
            break
        if chosen is None:
            chosen = int(order[0])

        cand = cands[chosen]
        if cand.type == "inspect":
            ok = apply_inspect(eng, cand)
        else:
            m = candidate_to_solver_move(cand, bs)
            ok = apply_solver_move(eng, m) if m else False
        if not ok:
            break
        visited_sigs.add(_state_signature(eng.get_state()))
        info["steps"] += 1

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
    parser.add_argument("--teacher-time-limit", type=float, default=5.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    model = build_default_model().to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"loaded {args.ckpt}, val_acc={ck.get('val_acc', '?'):.3f}")

    verified = parse_phase456_seeds(
        os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
    ) if args.use_verified_seeds else None

    all_samples: List[Sample] = []
    summary: Dict[int, Dict] = {}
    t0 = time.perf_counter()

    for phase in args.phases:
        print(f"\n=== Phase {phase} DAgger ===")
        if args.use_verified_seeds and verified:
            items = sorted([(k, v) for k, v in verified.items() if f"phase{phase}/" in k])[:args.max_maps]
            tasks = []
            for map_path, seeds in items:
                for seed in seeds[:args.max_seeds_per_map]:
                    tasks.append((map_path, seed))
        else:
            maps = list_phase_maps(phase, args.max_maps)
            tasks = [(m, 0) for m in maps]

        n_won = 0; n_total = 0; n_samples = 0; n_disagree = 0
        for map_path, seed in tasks:
            samples, info = collect_v2_dagger_episode(
                model, device, map_path, seed, phase,
                step_limit=args.step_limit, top_k=args.top_k,
                teacher_time_limit=args.teacher_time_limit,
            )
            n_total += 1
            if info["won"]:
                n_won += 1
            n_samples += len(samples)
            n_disagree += info["n_disagree"]
            all_samples.extend(samples)
        elapsed = time.perf_counter() - t0
        win_rate = n_won / max(1, n_total)
        print(f"  phase {phase}: won {n_won}/{n_total} ({win_rate*100:.1f}%) "
              f"samples={n_samples} disagree={n_disagree} elapsed={elapsed:.0f}s")
        summary[phase] = {"won": n_won, "total": n_total, "samples": n_samples, "disagree": n_disagree}

    print(f"\nTotal samples: {len(all_samples)}")
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    save_dataset(all_samples, out_path)


if __name__ == "__main__":
    main()
