"""SAGE-PR Stage A 评估 — deterministic rollout per phase.

每张图: 用 BeliefState (god mode) + candidate generator + SAGE-PR 神经评分,
        argmax 选 candidate, 应用 push_box / push_bomb 到引擎. 重复直到通关
        或 step_limit.

输出每 phase 的:
    win_rate, avg_steps, avg_inf_time
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from dataclasses import dataclass
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


def candidate_to_solver_move(cand: Candidate, bs: BeliefState):
    """把网络选的 candidate 翻成 solver move 格式 (供 apply_solver_move 用)."""
    if cand.type == "push_box":
        b = bs.boxes[cand.box_idx]
        # eid = ((col, row), class_id). 若 ID 未知, 用 -1 凑数 (apply_solver_move 在 eid 匹配时
        # 用 (pos, class_id), 但 apply 不严格检查 cid).
        cid = b.class_id if b.class_id is not None else 0
        return ("box", ((b.col, b.row), cid), cand.direction, 0)
    elif cand.type == "push_bomb":
        bm = bs.bombs[cand.bomb_idx]
        return ("bomb", (bm.col, bm.row), cand.direction, 0)
    return None


def _state_signature(state):
    """简易状态哈希: (car_pos, frozenset boxes, frozenset bombs)."""
    return (
        round(state.car_x * 2),
        round(state.car_y * 2),
        frozenset((round(b.x * 2), round(b.y * 2)) for b in state.boxes),
        frozenset((round(b.x * 2), round(b.y * 2)) for b in state.bombs),
    )


def rollout_one(model, device, map_path: str, seed: int, *,
                step_limit: int = 60,
                top_k: int = 1,
                fully_observed: bool = True,
                enforce_sigma_lock: bool = False,
                use_external_explorer: bool = False
                ) -> Tuple[bool, int, float, int]:
    """单图 deterministic rollout.

    返回 (won, n_macros, avg_inf_ms_per_call, n_lowlevel).

    use_external_explorer=True: 跑 plan_exploration_v3 后再让模型接管 push 阶段
        (V1 路线, fully_observed=True 自动激活).
    fully_observed=False + enforce_sigma_lock=True → V2 纯神经推理.
    """
    import random
    from experiments.sage_pr.belief_ida_solver import apply_inspect
    random.seed(seed)

    eng = GameEngine()
    state = eng.reset(map_path)
    inf_total = 0.0
    inf_calls = 0
    lowlevel_count = [0]   # closure box

    # 包装 engine.discrete_step 统计 low-level
    orig_step = eng.discrete_step
    def counted_step(a):
        lowlevel_count[0] += 1
        return orig_step(a)
    eng.discrete_step = counted_step

    # 可选: 先跑外挂 plan_exploration_v3 (V1 路线)
    if use_external_explorer:
        from smartcar_sokoban.solver.explorer_v3 import plan_exploration_v3
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            plan_exploration_v3(eng, max_retries=15)

    visited_sigs = set()
    visited_sigs.add(_state_signature(eng.get_state()))

    def _apply_cand(eng_target, cand, bs_at_step):
        if cand.type == "inspect":
            return apply_inspect(eng_target, cand)
        move = candidate_to_solver_move(cand, bs_at_step)
        if move is None:
            return False
        return apply_solver_move(eng_target, move)

    for step in range(step_limit):
        s = eng.get_state()
        if s.won:
            return True, step, inf_total / max(inf_calls, 1), lowlevel_count[0]

        bs = BeliefState.from_engine_state(s, fully_observed=fully_observed)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat, enforce_sigma_lock=enforce_sigma_lock)

        legal = [c.legal for c in cands]
        if not any(legal):
            return False, step, inf_total / max(inf_calls, 1), lowlevel_count[0]

        X_grid = build_grid_tensor(bs, feat).transpose(2, 0, 1)
        X_cand = encode_candidates(cands, bs, feat)
        u_global = build_global_features(bs, feat)
        mask = candidates_legality_mask(cands)

        xg_t = torch.from_numpy(X_grid).unsqueeze(0).to(device)
        xc_t = torch.from_numpy(X_cand).unsqueeze(0).to(device)
        ug_t = torch.from_numpy(u_global).unsqueeze(0).to(device)
        mk_t = torch.from_numpy(mask).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            score, _, _, _, _ = model(xg_t, xc_t, ug_t, mk_t)
        inf_total += time.perf_counter() - t0
        inf_calls += 1

        score_np = score.cpu().numpy().squeeze(0)
        score_np[~np.array(legal)] = -1e9
        order = np.argsort(-score_np)

        # 找第一个: legal + apply 后状态未访问过
        chosen_idx = None
        for k in range(min(top_k, len(order))):
            idx = int(order[k])
            cand = cands[idx]
            if not cand.legal:
                continue
            import copy
            eng_clone = copy.deepcopy(eng)
            if not _apply_cand(eng_clone, cand, bs):
                continue
            sig = _state_signature(eng_clone.get_state())
            if sig in visited_sigs:
                continue
            chosen_idx = idx
            break

        if chosen_idx is None:
            # 所有 top-k 都被访问过 / 不合法 → 回退到 top-1 (允许重访)
            for k in range(min(top_k, len(order))):
                idx = int(order[k])
                if cands[idx].legal:
                    chosen_idx = idx
                    break
        if chosen_idx is None:
            return False, step, inf_total / max(inf_calls, 1), lowlevel_count[0]

        cand = cands[chosen_idx]
        if not _apply_cand(eng, cand, bs):
            return False, step, inf_total / max(inf_calls, 1), lowlevel_count[0]
        visited_sigs.add(_state_signature(eng.get_state()))

    return eng.get_state().won, step_limit, inf_total / max(inf_calls, 1), lowlevel_count[0]


def evaluate_phase(model, device, phase: int, seeds_per_map: List[int],
                   *, step_limit: int = 60, top_k: int = 1,
                   max_maps: Optional[int] = None,
                   verified_seeds_map: Optional[Dict[str, List[int]]] = None,
                   fully_observed: bool = True,
                   enforce_sigma_lock: bool = False,
                   use_external_explorer: bool = False,
                   record_teacher: bool = False,
                   ) -> Dict[str, float]:
    maps = list_phase_maps(phase)
    if max_maps is not None:
        maps = maps[:max_maps]
    n_total = 0
    n_won = 0
    total_steps = 0
    total_inf_ms = 0.0
    total_lowlevel = 0
    total_teacher_low = 0
    teacher_won = 0
    for map_path in maps:
        if verified_seeds_map is not None and map_path in verified_seeds_map:
            ms = verified_seeds_map[map_path][:max(1, len(seeds_per_map))]
        else:
            ms = seeds_per_map
        for seed in ms:
            won, steps, avg_inf, n_low = rollout_one(
                model, device, map_path, seed,
                step_limit=step_limit, top_k=top_k,
                fully_observed=fully_observed,
                enforce_sigma_lock=enforce_sigma_lock,
                use_external_explorer=use_external_explorer)
            n_total += 1
            if won:
                n_won += 1
                total_lowlevel += n_low
            total_steps += steps
            total_inf_ms += avg_inf * 1000

            if record_teacher and won:
                # 老师 (v1_v3) 在同图上的 trajectory
                from experiments.sage_pr.preview_trajectory import _recorder_v1_v3
                t_log, t_info = _recorder_v1_v3(map_path, seed)
                if t_info.get("won"):
                    total_teacher_low += len(t_log)
                    teacher_won += 1

    return {
        "phase": phase,
        "n_total": n_total,
        "n_won": n_won,
        "win_rate": n_won / max(n_total, 1),
        "avg_steps": total_steps / max(n_total, 1),
        "avg_inf_ms": total_inf_ms / max(n_total, 1),
        "avg_model_lowlevel": total_lowlevel / max(n_won, 1),
        "avg_teacher_lowlevel": total_teacher_low / max(teacher_won, 1),
        "ratio_model_teacher": (total_lowlevel / max(n_won, 1)) /
                                max(1, total_teacher_low / max(teacher_won, 1)),
        "teacher_won": teacher_won,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="path to best.pt")
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seeds", type=str, default="0",
                        help="CSV seeds, default '0'")
    parser.add_argument("--step-limit", type=int, default=60)
    parser.add_argument("--max-maps", type=int, default=None,
                        help="limit per phase (None=all)")
    parser.add_argument("--use-verified-seeds", action="store_true",
                        help="使用 phase456_seed_manifest 中每图的 verified seed")
    parser.add_argument("--top-k", type=int, default=1,
                        help="rollout 时尝试 top-k 候选, 跳过会重访状态的 (避免循环). top_k=1 = 纯 greedy.")
    parser.add_argument("--mode", choices=["v1", "v2"], default="v1",
                        help="v1 = fully_observed (跑前需 plan_exploration); v2 = partial-obs + 抑制场 (纯神经)")
    parser.add_argument("--external-explorer", action="store_true",
                        help="V1 路线: 跑前先 plan_exploration_v3 把 entity 都识别")
    parser.add_argument("--record-teacher", action="store_true",
                        help="同时录 teacher (v1_v3) trajectory, 计算 model vs teacher 低层步数比")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_default_model().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded {args.ckpt}, val_acc={ckpt.get('val_acc', '?'):.3f}")

    verified_map = None
    if args.use_verified_seeds:
        verified_map = parse_phase456_seeds(
            os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
        )
        print(f"verified manifest entries: {len(verified_map)}")

    fully_observed = (args.mode == "v1")
    enforce_sigma_lock = (args.mode == "v2")
    print(f"mode={args.mode}, fully_observed={fully_observed}, "
          f"enforce_sigma_lock={enforce_sigma_lock}")

    results = []
    for ph in args.phases:
        print(f"\n=== Phase {ph} ===")
        t0 = time.perf_counter()
        r = evaluate_phase(model, device, ph, seeds,
                           step_limit=args.step_limit, top_k=args.top_k,
                           max_maps=args.max_maps,
                           verified_seeds_map=verified_map,
                           fully_observed=fully_observed,
                           enforce_sigma_lock=enforce_sigma_lock,
                           use_external_explorer=args.external_explorer,
                           record_teacher=args.record_teacher)
        elapsed = time.perf_counter() - t0
        print(f"  win_rate = {r['win_rate']*100:.2f}% "
              f"({r['n_won']}/{r['n_total']}); "
              f"avg_steps={r['avg_steps']:.1f}; "
              f"avg_inf={r['avg_inf_ms']:.1f}ms; "
              f"elapsed={elapsed:.1f}s")
        results.append(r)

    print("\n=== Summary ===")
    for r in results:
        line = (f"phase {r['phase']}: win_rate={r['win_rate']*100:.2f}% "
                f"({r['n_won']}/{r['n_total']})")
        if r.get("avg_model_lowlevel", 0) > 0:
            line += (f" | model_low={r['avg_model_lowlevel']:.1f}"
                     f" teacher_low={r['avg_teacher_lowlevel']:.1f}"
                     f" ratio={r['ratio_model_teacher']:.2f}x")
        line += f" | avg_inf={r['avg_inf_ms']:.1f}ms"
        print(line)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
