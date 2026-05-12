"""Rollout search 推理 — 不依赖 value head, 用真实进度评估.

每步决策:
    1. 当前状态 score top-K 候选.
    2. 对每个候选: 应用 → greedy rollout R 步 → 测量进度 (推送距离场总和的减少).
    3. 选最佳候选作为本步动作.

这是简化版 MCTS / lookahead. 比 beam search 更鲁棒 — 不依赖 value head.
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
from smartcar_sokoban.symbolic.features import compute_domain_features, INF
from smartcar_sokoban.symbolic.candidates import (
    Candidate, generate_candidates, candidates_legality_mask,
)
from smartcar_sokoban.symbolic.cand_features import encode_candidates
from smartcar_sokoban.symbolic.grid_tensor import (
    build_grid_tensor, build_global_features,
)
from experiments.sage_pr.model import build_model_from_ckpt
from experiments.sage_pr.build_dataset_v3 import (
    apply_solver_move, parse_phase456_seeds, list_phase_maps,
)
from experiments.sage_pr.evaluate_sage_pr import (
    candidate_to_solver_move, _state_signature,
)


def measure_progress(eng: GameEngine) -> float:
    """状态进度 ∈ [0, 1]. 1 = 全部消除 (won), 0 = 起始.

    定义: 1 - normalized(剩余推送距离总和).
    """
    s = eng.get_state()
    if s.won:
        return 1.0
    bs = BeliefState.from_engine_state(s, fully_observed=True)
    feat = compute_domain_features(bs)
    total = 0.0
    for i, b in enumerate(bs.boxes):
        if i >= len(feat.push_dist_field):
            continue
        d = feat.push_dist_field[i][b.row, b.col]
        if d != INF:
            total += float(d)
        else:
            total += 50.0  # 不可达惩罚
    # n_box_remaining 越多 progress 越低
    n_remaining = len(bs.boxes)
    base = total + n_remaining * 5.0  # base offset to reward removal
    return -base   # 直接返回负成本 (越大越好)


def model_score(model, device, eng: GameEngine,
                fully_observed: bool = True,
                enforce_sigma_lock: bool = False,
                ) -> Tuple[Optional[List[Candidate]], Optional[np.ndarray]]:
    s = eng.get_state()
    bs = BeliefState.from_engine_state(s, fully_observed=fully_observed)
    feat = compute_domain_features(bs)
    cands = generate_candidates(bs, feat, enforce_sigma_lock=enforce_sigma_lock)
    if not any(c.legal for c in cands):
        return None, None
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
    return cands, score_np


def _apply_any(eng_target, cand: Candidate, bs_at_step: BeliefState) -> bool:
    """Apply candidate (push or inspect) to engine."""
    from experiments.sage_pr.belief_ida_solver import apply_inspect
    if cand.type == "inspect":
        return apply_inspect(eng_target, cand)
    move = candidate_to_solver_move(cand, bs_at_step)
    if move is None:
        return False
    return apply_solver_move(eng_target, move)


def greedy_rollout_n(model, device, eng: GameEngine,
                     n: int, visited_sigs: set,
                     fully_observed: bool = True,
                     enforce_sigma_lock: bool = False) -> Tuple[GameEngine, bool]:
    """Greedy rollout n steps on a clone. Return (final_eng, won)."""
    eng = copy.deepcopy(eng)
    for _ in range(n):
        s = eng.get_state()
        if s.won:
            return eng, True
        cands, score = model_score(model, device, eng, fully_observed, enforce_sigma_lock)
        if cands is None:
            return eng, False
        order = np.argsort(-score)
        chosen = None
        for k in range(min(4, len(order))):
            idx = int(order[k])
            cand = cands[idx]
            if not cand.legal:
                continue
            bs_now = BeliefState.from_engine_state(eng.get_state(), fully_observed=fully_observed)
            eng_clone = copy.deepcopy(eng)
            if not _apply_any(eng_clone, cand, bs_now):
                continue
            sig = _state_signature(eng_clone.get_state())
            if sig in visited_sigs:
                continue
            chosen = idx
            break
        if chosen is None:
            chosen = int(order[0]) if cands[int(order[0])].legal else None
        if chosen is None:
            return eng, False
        bs_now = BeliefState.from_engine_state(eng.get_state(), fully_observed=fully_observed)
        if not _apply_any(eng, cands[chosen], bs_now):
            return eng, False
    return eng, eng.get_state().won


def rollout_search_step(model, device, eng: GameEngine,
                         visited_sigs: set,
                         beam_width: int = 4, lookahead: int = 4,
                         fully_observed: bool = True,
                         enforce_sigma_lock: bool = False
                         ) -> Optional[Tuple[int, Candidate]]:
    """对 top-B 候选, 模拟 lookahead 步 greedy rollout, 选进度最大的."""
    cands, score = model_score(model, device, eng, fully_observed, enforce_sigma_lock)
    if cands is None:
        return None
    legal_idx = [i for i, c in enumerate(cands) if c.legal]
    legal_idx.sort(key=lambda i: -score[i])
    top_b = legal_idx[:beam_width]
    if not top_b:
        return None

    if len(top_b) == 1:
        return top_b[0], cands[top_b[0]]

    best_score = -1e18
    best_idx = None
    for idx in top_b:
        cand = cands[idx]
        bs_now = BeliefState.from_engine_state(eng.get_state(), fully_observed=fully_observed)
        eng_clone = copy.deepcopy(eng)
        if not _apply_any(eng_clone, cand, bs_now):
            continue
        eng_after_rollout, won = greedy_rollout_n(
            model, device, eng_clone, lookahead, visited_sigs,
            fully_observed, enforce_sigma_lock,
        )
        progress = measure_progress(eng_after_rollout)
        if won:
            progress += 1000.0  # huge bonus
        score_total = progress + 0.1 * score[idx]
        if score_total > best_score:
            best_score = score_total
            best_idx = idx

    if best_idx is None:
        best_idx = top_b[0]
    return best_idx, cands[best_idx]


def rollout_search_episode(model, device, map_path: str, seed: int,
                            *, step_limit: int = 60,
                            beam_width: int = 3, lookahead: int = 4,
                            fully_observed: bool = True,
                            enforce_sigma_lock: bool = False,
                            ) -> Tuple[bool, int, float]:
    import random
    random.seed(seed)

    eng = GameEngine()
    eng.reset(map_path)
    inf_total = 0.0
    inf_calls = 0
    visited = set()
    visited.add(_state_signature(eng.get_state()))

    for step in range(step_limit):
        s = eng.get_state()
        if s.won:
            return True, step, inf_total / max(inf_calls, 1)

        t0 = time.perf_counter()
        result = rollout_search_step(model, device, eng, visited,
                                       beam_width=beam_width,
                                       lookahead=lookahead,
                                       fully_observed=fully_observed,
                                       enforce_sigma_lock=enforce_sigma_lock)
        inf_total += time.perf_counter() - t0
        inf_calls += 1

        if result is None:
            return False, step, inf_total / max(inf_calls, 1)
        idx, cand = result

        bs_now = BeliefState.from_engine_state(eng.get_state(), fully_observed=fully_observed)
        if not _apply_any(eng, cand, bs_now):
            return False, step, inf_total / max(inf_calls, 1)
        visited.add(_state_signature(eng.get_state()))

    return eng.get_state().won, step_limit, inf_total / max(inf_calls, 1)


def evaluate_phase_rollout(model, device, phase: int, seeds_per_map: List[int],
                            *, step_limit: int = 60,
                            beam_width: int = 3, lookahead: int = 4,
                            max_maps: Optional[int] = None,
                            verified_seeds_map: Optional[Dict[str, List[int]]] = None,
                            fully_observed: bool = True,
                            enforce_sigma_lock: bool = False,
                            ) -> Dict:
    maps = list_phase_maps(phase)
    if max_maps is not None:
        maps = maps[:max_maps]
    n_total = 0
    n_won = 0
    total_steps = 0
    total_inf_ms = 0.0
    failed: List[Tuple[str, int]] = []
    for map_path in maps:
        if verified_seeds_map is not None and map_path in verified_seeds_map:
            ms = verified_seeds_map[map_path][:max(1, len(seeds_per_map))]
        else:
            ms = seeds_per_map
        for seed in ms:
            won, steps, avg_inf = rollout_search_episode(
                model, device, map_path, seed,
                step_limit=step_limit,
                beam_width=beam_width, lookahead=lookahead,
                fully_observed=fully_observed,
                enforce_sigma_lock=enforce_sigma_lock,
            )
            n_total += 1
            if won:
                n_won += 1
            else:
                failed.append((map_path, seed))
            total_steps += steps
            total_inf_ms += avg_inf * 1000
    return {
        "phase": phase,
        "n_total": n_total,
        "n_won": n_won,
        "win_rate": n_won / max(n_total, 1),
        "avg_steps": total_steps / max(n_total, 1),
        "avg_inf_ms": total_inf_ms / max(n_total, 1),
        "failed": failed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--step-limit", type=int, default=60)
    parser.add_argument("--max-maps", type=int, default=100)
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--mode", choices=["v1", "v2"], default="v1",
                        help="v1=fully_observed; v2=partial-obs + enforce_sigma_lock")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    fully_observed = (args.mode == "v1")
    enforce_sigma_lock = (args.mode == "v2")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_model_from_ckpt(args.ckpt, device=device)
    model.eval()
    print(f"loaded {args.ckpt}, val_acc={ckpt.get('val_acc', '?'):.3f}")
    print(f"beam={args.beam}, lookahead={args.lookahead}")

    verified_map = None
    if args.use_verified_seeds:
        verified_map = parse_phase456_seeds(
            os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
        )

    results = []
    for ph in args.phases:
        print(f"\n=== Phase {ph} ===")
        t0 = time.perf_counter()
        r = evaluate_phase_rollout(
            model, device, ph, seeds,
            step_limit=args.step_limit,
            beam_width=args.beam, lookahead=args.lookahead,
            max_maps=args.max_maps,
            verified_seeds_map=verified_map,
            fully_observed=fully_observed,
            enforce_sigma_lock=enforce_sigma_lock,
        )
        elapsed = time.perf_counter() - t0
        print(f"  win_rate = {r['win_rate']*100:.2f}% ({r['n_won']}/{r['n_total']}); "
              f"avg_steps={r['avg_steps']:.1f}; "
              f"avg_inf={r['avg_inf_ms']:.1f}ms; "
              f"elapsed={elapsed:.1f}s")
        results.append(r)

    print("\n=== Summary ===")
    for r in results:
        print(f"phase {r['phase']}: win_rate={r['win_rate']*100:.2f}% "
              f"({r['n_won']}/{r['n_total']}), avg_inf={r['avg_inf_ms']:.1f}ms")

    if args.out:
        for r in results:
            failed = r.get("failed", [])
            if failed:
                print(f"\nFailed phase {r['phase']} ({len(failed)}):")
                for mp, seed in failed:
                    print(f"  {mp} seed={seed}")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=list)


if __name__ == "__main__":
    main()
