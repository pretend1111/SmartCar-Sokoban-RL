"""神经引导 beam search 推理 — P7.4.

每步决策:
    1. 从当前状态获取候选 + 网络分数.
    2. 取 top-B 候选, 各自模拟 1 步 (apply_solver_move) 到克隆 engine.
    3. 在 s' 上递归 beam (depth - 1).
    4. 每条路径打分 = Σ α·log π(a_t) + λ_v·V(s_D).
    5. 取最高分路径的 a_0 作为本步动作.

时间复杂度: O(B^D) 推理 / 决策. 默认 B=3, D=2 → 9 calls/decision (~14 ms cuda).
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
    Candidate, generate_candidates, candidates_legality_mask, MAX_CANDIDATES,
)
from smartcar_sokoban.symbolic.cand_features import encode_candidates
from smartcar_sokoban.symbolic.grid_tensor import (
    build_grid_tensor, build_global_features,
)
from experiments.sage_pr.model import build_default_model
from experiments.sage_pr.build_dataset_v3 import (
    apply_solver_move, parse_phase456_seeds, list_phase_maps,
)
from experiments.sage_pr.evaluate_sage_pr import (
    candidate_to_solver_move, _state_signature,
)


def model_forward_state(model, device, eng: GameEngine,
                        fully_observed: bool = True
                        ) -> Tuple[Optional[List[Candidate]], Optional[np.ndarray], Optional[float]]:
    """单次前向, 返回 (candidates, score_logits, value)."""
    s = eng.get_state()
    bs = BeliefState.from_engine_state(s, fully_observed=fully_observed)
    feat = compute_domain_features(bs)
    cands = generate_candidates(bs, feat)
    if not any(c.legal for c in cands):
        return None, None, None

    X_grid = build_grid_tensor(bs, feat).transpose(2, 0, 1)
    X_cand = encode_candidates(cands, bs, feat)
    u_global = build_global_features(bs, feat)
    mask = candidates_legality_mask(cands)

    xg_t = torch.from_numpy(X_grid).unsqueeze(0).to(device)
    xc_t = torch.from_numpy(X_cand).unsqueeze(0).to(device)
    ug_t = torch.from_numpy(u_global).unsqueeze(0).to(device)
    mk_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    with torch.no_grad():
        score, value, _, _, _ = model(xg_t, xc_t, ug_t, mk_t)
    score_np = score.cpu().numpy().squeeze(0)
    score_np[mask < 0.5] = -1e9
    value_f = float(value.cpu().numpy().squeeze(0))
    return cands, score_np, value_f


def beam_search_step(model, device, eng: GameEngine,
                     visited_sigs: set,
                     beam_width: int = 3, depth: int = 2,
                     log_alpha: float = 1.0, lambda_v: float = 0.5,
                     fully_observed: bool = True
                     ) -> Optional[Tuple[int, Candidate]]:
    """选最佳第一步动作.

    Returns:
        (action_idx, cand) or None if no legal action.
    """
    cands, score, _ = model_forward_state(model, device, eng, fully_observed)
    if cands is None:
        return None

    # log softmax (with temperature 1)
    log_probs = score - np.logaddexp.reduce(score, axis=-1, keepdims=False)

    # Top-B legal candidates
    legal_idx = [i for i, c in enumerate(cands) if c.legal]
    legal_idx.sort(key=lambda i: -score[i])
    top_b = legal_idx[:beam_width]
    if not top_b:
        return None

    if depth <= 1:
        # 只用当前 logπ + value 选首动作 (退化为 greedy + 反循环)
        for idx in top_b:
            cand = cands[idx]
            move = candidate_to_solver_move(cand, eng_belief(eng, fully_observed))
            if move is None:
                continue
            eng_clone = copy.deepcopy(eng)
            if not apply_solver_move(eng_clone, move):
                continue
            sig = _state_signature(eng_clone.get_state())
            if sig in visited_sigs:
                continue
            return idx, cand
        # fallback: top-1 若全不通过反循环
        idx = top_b[0]
        return idx, cands[idx]

    # Beam: 对每个 top-B, 模拟一步 → 递归 beam (depth - 1)
    best_score = -1e18
    best_first = None
    for idx in top_b:
        cand = cands[idx]
        bs_now = BeliefState.from_engine_state(eng.get_state(), fully_observed=fully_observed)
        move = candidate_to_solver_move(cand, bs_now)
        if move is None:
            continue
        eng_clone = copy.deepcopy(eng)
        if not apply_solver_move(eng_clone, move):
            continue
        sig = _state_signature(eng_clone.get_state())
        # 不严格禁止重访 (beam 允许), 但加 penalty
        revisit_penalty = -3.0 if sig in visited_sigs else 0.0
        # 检查是否赢
        if eng_clone.get_state().won:
            path_score = log_alpha * log_probs[idx] + 100.0
        else:
            # 递归: 在 s' 上跑 1-step beam, 累加 logπ + value
            sub_cands, sub_score, sub_value = model_forward_state(model, device, eng_clone, fully_observed)
            if sub_cands is None or not any(c.legal for c in sub_cands):
                # 死锁
                path_score = log_alpha * log_probs[idx] - 5.0
            else:
                # 取下一步 top-1 logπ
                sub_log_probs = sub_score - np.logaddexp.reduce(sub_score, axis=-1)
                next_max_logp = float(sub_log_probs.max())
                path_score = (
                    log_alpha * log_probs[idx]
                    + log_alpha * next_max_logp
                    + lambda_v * (sub_value if sub_value is not None else 0.0)
                    + revisit_penalty
                )

        if path_score > best_score:
            best_score = path_score
            best_first = (idx, cand)

    return best_first


def eng_belief(eng, fully_observed: bool = True) -> BeliefState:
    """quick helper."""
    return BeliefState.from_engine_state(eng.get_state(), fully_observed=fully_observed)


def rollout_beam(model, device, map_path: str, seed: int,
                 *, step_limit: int = 60,
                 beam_width: int = 3, depth: int = 2,
                 fully_observed: bool = True) -> Tuple[bool, int, float]:
    """单图 beam search rollout."""
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
        result = beam_search_step(model, device, eng, visited,
                                   beam_width=beam_width, depth=depth,
                                   fully_observed=fully_observed)
        inf_total += time.perf_counter() - t0
        inf_calls += 1

        if result is None:
            return False, step, inf_total / max(inf_calls, 1)
        idx, cand = result

        bs_now = eng_belief(eng, fully_observed)
        move = candidate_to_solver_move(cand, bs_now)
        if move is None or not apply_solver_move(eng, move):
            return False, step, inf_total / max(inf_calls, 1)
        visited.add(_state_signature(eng.get_state()))

    return eng.get_state().won, step_limit, inf_total / max(inf_calls, 1)


def evaluate_phase_beam(model, device, phase: int, seeds_per_map: List[int],
                         *, step_limit: int = 60,
                         beam_width: int = 3, depth: int = 2,
                         max_maps: Optional[int] = None,
                         verified_seeds_map: Optional[Dict[str, List[int]]] = None,
                         ) -> Dict:
    maps = list_phase_maps(phase)
    if max_maps is not None:
        maps = maps[:max_maps]
    n_total = 0
    n_won = 0
    total_steps = 0
    total_inf_ms = 0.0
    for map_path in maps:
        if verified_seeds_map is not None and map_path in verified_seeds_map:
            ms = verified_seeds_map[map_path][:max(1, len(seeds_per_map))]
        else:
            ms = seeds_per_map
        for seed in ms:
            won, steps, avg_inf = rollout_beam(
                model, device, map_path, seed,
                step_limit=step_limit,
                beam_width=beam_width, depth=depth,
            )
            n_total += 1
            if won:
                n_won += 1
            total_steps += steps
            total_inf_ms += avg_inf * 1000
    return {
        "phase": phase,
        "n_total": n_total,
        "n_won": n_won,
        "win_rate": n_won / max(n_total, 1),
        "avg_steps": total_steps / max(n_total, 1),
        "avg_inf_ms": total_inf_ms / max(n_total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--step-limit", type=int, default=60)
    parser.add_argument("--max-maps", type=int, default=None)
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_default_model().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded {args.ckpt}, val_acc={ckpt.get('val_acc', '?'):.3f}")
    print(f"beam={args.beam}, depth={args.depth}")

    verified_map = None
    if args.use_verified_seeds:
        verified_map = parse_phase456_seeds(
            os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
        )

    results = []
    for ph in args.phases:
        print(f"\n=== Phase {ph} ===")
        t0 = time.perf_counter()
        r = evaluate_phase_beam(model, device, ph, seeds,
                                 step_limit=args.step_limit,
                                 beam_width=args.beam, depth=args.depth,
                                 max_maps=args.max_maps,
                                 verified_seeds_map=verified_map)
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
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
