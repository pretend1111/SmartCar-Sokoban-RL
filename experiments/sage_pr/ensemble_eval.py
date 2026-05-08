"""集成评估: 多 ckpt score 平均.

试图通过集成 BC + DAgger 不同阶段的 ckpt 减少 variance, 提升 win rate.
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
)
from experiments.sage_pr.evaluate_sage_pr import (
    candidate_to_solver_move, _state_signature,
)


def load_models(ckpt_paths: List[str], device) -> List:
    models = []
    for p in ckpt_paths:
        ck = torch.load(p, map_location=device)
        m = build_default_model().to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        models.append(m)
        print(f"  loaded {p}, val_acc={ck.get('val_acc', '?'):.3f}")
    return models


def ensemble_score(models, device, bs, feat, cands) -> Optional[np.ndarray]:
    X_grid = build_grid_tensor(bs, feat).transpose(2, 0, 1)
    X_cand = encode_candidates(cands, bs, feat)
    u_global = build_global_features(bs, feat)
    mask = candidates_legality_mask(cands)
    xg = torch.from_numpy(X_grid).unsqueeze(0).to(device)
    xc = torch.from_numpy(X_cand).unsqueeze(0).to(device)
    ug = torch.from_numpy(u_global).unsqueeze(0).to(device)
    mk = torch.from_numpy(mask).unsqueeze(0).to(device)

    scores = []
    with torch.no_grad():
        for m in models:
            s, _, _, _, _ = m(xg, xc, ug, mk)
            scores.append(s.cpu().numpy().squeeze(0))
    avg = np.mean(scores, axis=0)
    avg[mask < 0.5] = -1e9
    return avg


def rollout_one(models, device, map_path: str, seed: int, *,
                step_limit: int = 60, top_k: int = 4,
                fully_observed: bool = True) -> Tuple[bool, int, float]:
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
        bs = BeliefState.from_engine_state(s, fully_observed=fully_observed)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat)
        if not any(c.legal for c in cands):
            return False, step, inf_total / max(inf_calls, 1)
        t0 = time.perf_counter()
        score = ensemble_score(models, device, bs, feat, cands)
        inf_total += time.perf_counter() - t0
        inf_calls += 1

        order = np.argsort(-score)
        chosen = None
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
            if sig in visited:
                continue
            chosen = idx
            break
        if chosen is None:
            for k in range(min(top_k, len(order))):
                idx = int(order[k])
                if cands[idx].legal:
                    chosen = idx
                    break
        if chosen is None:
            return False, step, inf_total / max(inf_calls, 1)
        cand = cands[chosen]
        move = candidate_to_solver_move(cand, bs)
        if move is None or not apply_solver_move(eng, move):
            return False, step, inf_total / max(inf_calls, 1)
        visited.add(_state_signature(eng.get_state()))

    return eng.get_state().won, step_limit, inf_total / max(inf_calls, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--max-maps", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {len(args.ckpts)} models...")
    models = load_models(args.ckpts, device)

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
        t0 = time.perf_counter()
        for map_path in maps:
            ms = (verified_map.get(map_path, [0])[:max(1, len(seeds))]
                  if verified_map else seeds)
            for seed in ms:
                won, steps, avg_inf = rollout_one(
                    models, device, map_path, seed,
                    top_k=args.top_k,
                )
                n_total += 1
                if won: n_won += 1
                total_steps += steps
                total_inf_ms += avg_inf * 1000
        elapsed = time.perf_counter() - t0
        r = {
            "phase": ph,
            "n_total": n_total,
            "n_won": n_won,
            "win_rate": n_won / max(n_total, 1),
            "avg_steps": total_steps / max(n_total, 1),
            "avg_inf_ms": total_inf_ms / max(n_total, 1),
        }
        print(f"phase {ph}: {r['win_rate']*100:.1f}% ({n_won}/{n_total}), "
              f"avg_inf={r['avg_inf_ms']:.1f}ms, {elapsed:.0f}s")
        results.append(r)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
