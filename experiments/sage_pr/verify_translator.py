"""验证 translator 100% 准确.

对每张图, 跑 exact teacher (plan_exploration + MultiBoxSolver), 然后:
  1. 双 engine 并行: eng_ref 和 eng_cand
  2. 每步 exact 给一个 move
  3. eng_ref: 直接 apply_solver_move(move) (即数据生成时录的实际行为)
  4. eng_cand: 在状态上跑 generate_candidates, match_move_to_candidate 得 label
              然后用 candidate_to_solver_move(cand[label]) → mock_move,
              apply_solver_move(mock_move) (即模型推理时的行为)
  5. 比对两个 engine 的 (car pos, boxes, bombs, won) 是否一致

输出:
  - per-map: total_steps, label_miss, divergence_step, status
  - aggregate: 100% 一致率
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from smartcar_sokoban.solver.explorer import plan_exploration, exploration_complete
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import (
    generate_candidates, candidates_legality_mask, Candidate,
)
from experiments.sage_pr.build_dataset_v3 import (
    apply_solver_move, match_move_to_candidate,
    parse_phase456_seeds, list_phase_maps,
)
from experiments.sage_pr.evaluate_sage_pr import candidate_to_solver_move


def _engine_signature(eng: GameEngine) -> Tuple:
    """车位姿 + 箱炸弹位 + 胜利位."""
    s = eng.get_state()
    return (
        round(s.car_x * 4) / 4,
        round(s.car_y * 4) / 4,
        round(s.car_angle * 16) / 16,
        tuple(sorted((pos_to_grid(b.x, b.y), b.class_id) for b in s.boxes)),
        tuple(sorted(pos_to_grid(b.x, b.y) for b in s.bombs)),
        tuple(sorted((pos_to_grid(t.x, t.y), t.num_id) for t in s.targets)),
        s.won,
    )


def verify_episode(map_path: str, seed: int,
                    *, solver_time: float = 30.0
                    ) -> Dict[str, Any]:
    """验证一张图的 translator 准确性."""
    random.seed(seed)
    eng_ref = GameEngine()
    eng_ref.reset(map_path)

    # Phase 1: explore (相同的低层 actions 同步两个 engine)
    with contextlib.redirect_stdout(io.StringIO()):
        explore_actions = plan_exploration(eng_ref)
    if not exploration_complete(eng_ref.get_state()):
        return {"map": map_path, "seed": seed, "status": "explore_incomplete",
                "n_steps": 0, "n_label_miss": 0, "n_diverge": 0,
                "first_diverge_step": None}

    # 复制 explore 后状态到 eng_cand
    eng_cand = copy.deepcopy(eng_ref)

    # Phase 2: solver
    state = eng_ref.get_state()
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            moves = solver.solve(max_cost=300, time_limit=solver_time, strategy="auto")
        except Exception:
            moves = None
    if not moves:
        return {"map": map_path, "seed": seed, "status": "solver_no_solution",
                "n_steps": 0, "n_label_miss": 0, "n_diverge": 0,
                "first_diverge_step": None}

    n_label_miss = 0
    n_diverge = 0
    first_diverge_step: Optional[int] = None
    diverge_detail: Optional[str] = None

    for step_i, move in enumerate(moves):
        # ── 录数据时的逻辑: match → label → 检查 cand → 应用 ──
        bs = BeliefState.from_engine_state(eng_cand.get_state(), fully_observed=True)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat)
        label = match_move_to_candidate(move, cands, bs, run_length=1)

        if label is None:
            n_label_miss += 1
            # 跳过 sample 但同步两边 engine
            ok_r = apply_solver_move(eng_ref, move)
            ok_c = apply_solver_move(eng_cand, move)
            if not (ok_r and ok_c):
                return {"map": map_path, "seed": seed, "status": "apply_fail",
                        "n_steps": step_i, "n_label_miss": n_label_miss,
                        "n_diverge": n_diverge, "first_diverge_step": first_diverge_step}
            continue

        # ── 推理时的逻辑: cand[label] → candidate_to_solver_move → apply ──
        cand = cands[label]
        # 验证 cand 性质
        if cand.run_length != 1:
            n_diverge += 1
            if first_diverge_step is None:
                first_diverge_step = step_i
                diverge_detail = f"run_length={cand.run_length} != 1"
        if not cand.legal:
            n_diverge += 1
            if first_diverge_step is None:
                first_diverge_step = step_i
                diverge_detail = f"cand.legal=False"

        mock_move = candidate_to_solver_move(cand, bs)
        if mock_move is None:
            n_diverge += 1
            if first_diverge_step is None:
                first_diverge_step = step_i
                diverge_detail = "candidate_to_solver_move returned None"
            apply_solver_move(eng_ref, move)
            apply_solver_move(eng_cand, move)
            continue

        # eng_ref: 用 exact 的 move
        ok_r = apply_solver_move(eng_ref, move)
        # eng_cand: 用 cand 翻译出的 mock_move
        ok_c = apply_solver_move(eng_cand, mock_move)

        if not ok_r:
            return {"map": map_path, "seed": seed, "status": "ref_apply_fail",
                    "n_steps": step_i, "n_label_miss": n_label_miss,
                    "n_diverge": n_diverge, "first_diverge_step": first_diverge_step}
        if not ok_c:
            n_diverge += 1
            if first_diverge_step is None:
                first_diverge_step = step_i
                diverge_detail = "cand apply_solver_move failed"
            # 让 cand 也跟上, 用 ref 的 move
            apply_solver_move(eng_cand, move)
            continue

        # 比对状态
        sig_r = _engine_signature(eng_ref)
        sig_c = _engine_signature(eng_cand)
        if sig_r != sig_c:
            n_diverge += 1
            if first_diverge_step is None:
                first_diverge_step = step_i
                diverge_detail = f"state mismatch: ref={sig_r[:3]} cand={sig_c[:3]}"
            # resync cand to ref
            eng_cand = copy.deepcopy(eng_ref)

    won_ref = eng_ref.get_state().won
    won_cand = eng_cand.get_state().won

    return {
        "map": map_path, "seed": seed,
        "status": "ok" if (won_ref and won_cand and n_diverge == 0) else (
            "ref_did_not_win" if not won_ref else (
            "cand_did_not_win" if not won_cand else f"diverge_n={n_diverge}")),
        "n_steps": len(moves),
        "n_label_miss": n_label_miss,
        "n_diverge": n_diverge,
        "first_diverge_step": first_diverge_step,
        "diverge_detail": diverge_detail,
        "won_ref": won_ref,
        "won_cand": won_cand,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--n-maps", type=int, default=20)
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--solver-time", type=float, default=30.0)
    parser.add_argument("--out", type=str, default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    verified_map = parse_phase456_seeds(
        os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
    ) if args.use_verified_seeds else {}

    all_results: List[Dict[str, Any]] = []
    for phase in args.phases:
        print(f"\n=== Phase {phase} translator verify ===")
        items: List[Tuple[str, int]] = []
        if args.use_verified_seeds and verified_map:
            phase_items = sorted([(k, v) for k, v in verified_map.items()
                                   if f"phase{phase}/" in k])[:args.n_maps]
            for map_path, seeds in phase_items:
                items.append((map_path, seeds[0]))
        else:
            maps = list_phase_maps(phase, args.n_maps)
            for map_path in maps:
                items.append((map_path, 0))

        ok_count = 0
        diverge_count = 0
        miss_total = 0
        for map_path, seed in items:
            full = os.path.join(ROOT, map_path)
            if not os.path.exists(full):
                continue
            res = verify_episode(map_path, seed, solver_time=args.solver_time)
            res["phase"] = phase
            all_results.append(res)
            tag = "✓" if res["status"] == "ok" else "✗"
            print(f"  {tag} {os.path.basename(map_path)} seed={seed}: "
                  f"steps={res['n_steps']} miss={res['n_label_miss']} "
                  f"diverge={res['n_diverge']} status={res['status']}")
            if res.get("diverge_detail"):
                print(f"      ↳ first_diverge@{res['first_diverge_step']}: {res['diverge_detail']}")
            if res["status"] == "ok":
                ok_count += 1
            if res["n_diverge"] > 0:
                diverge_count += 1
            miss_total += res["n_label_miss"]
        print(f"\n  Phase {phase}: ok {ok_count}/{len(items)}, "
              f"diverge {diverge_count}/{len(items)}, label_miss total {miss_total}")

    n = len(all_results)
    n_ok = sum(1 for r in all_results if r["status"] == "ok")
    n_div = sum(1 for r in all_results if r["n_diverge"] > 0)
    miss = sum(r["n_label_miss"] for r in all_results)
    total_steps = sum(r["n_steps"] for r in all_results)
    print(f"\n========================================")
    print(f"Total: {n} maps, {total_steps} steps")
    print(f"  Translator round-trip ok: {n_ok}/{n} ({n_ok/max(n,1)*100:.1f}%)")
    print(f"  Diverge maps: {n_div}/{n}")
    print(f"  label_miss total: {miss} ({miss/max(total_steps,1)*100:.2f}% of steps)")

    if args.out:
        out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
