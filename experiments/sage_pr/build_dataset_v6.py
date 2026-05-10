"""V2 Dataset: god-mode exact A + 抑制场 + 插入 inspect.

算法:
  1. 跑 god-mode exact 拿 push 计划 A (一定可行, 物理路径定死)
  2. 重置, partial-obs 重放:
     a. 用 enforce_sigma_lock=True 生成 candidates (推到 target 必须 σ 锁定)
     b. 拿 A 的下一步 push, match 到 candidate
     c. 若 match 失败 (因为 σ 未锁导致 push illegal) → 选 best inspect, 录 inspect sample,
        应用 inspect (改 K), 不前进 A 指针, 回到 (a)
     d. 否则 (push legal) → 录 push sample, apply, 前进 A 指针
  3. 验证: 走完 A 后 engine.won = True

样本结构 (与 v3/v5 同 schema):
  X_grid, X_cand, u_global, mask, label, phase, source=SOURCE_V2 (=6)

输出 npz 同 v3 schema, 训练管线无需改.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.solver.explorer import plan_exploration, exploration_complete
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features, INF
from smartcar_sokoban.symbolic.candidates import (
    Candidate, generate_candidates, candidates_legality_mask, MAX_CANDIDATES,
)
from smartcar_sokoban.symbolic.cand_features import encode_candidates
from smartcar_sokoban.symbolic.grid_tensor import (
    build_grid_tensor, build_global_features,
)
from experiments.sage_pr.build_dataset_v3 import (
    parse_phase456_seeds, list_phase_maps, save_dataset, Sample,
    apply_solver_move, match_move_to_candidate,
)
from experiments.sage_pr.belief_ida_solver import apply_inspect


SOURCE_V2 = 6


# ── god-mode A planner ────────────────────────────────────

def plan_god_mode(map_path: str, seed: int, time_limit: float = 60.0
                   ) -> Optional[List]:
    """跑 god-mode exact: 直接 reset + 不跑 explore + 拿全 ID 解 plan."""
    random.seed(seed)
    eng = GameEngine()
    state = eng.reset(map_path)
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return solver.solve(max_cost=300, time_limit=time_limit, strategy="auto")
        except Exception:
            return None


# ── inspect 选择 ──────────────────────────────────────────

def _walk_cost(start: Tuple[int, int], end: Tuple[int, int],
               walls: np.ndarray, obstacles: set) -> int:
    if start == end:
        return 0
    rows, cols = walls.shape
    visited = {start}
    q = deque([(start, 0)])
    while q:
        (c, r), d = q.popleft()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc, nr = c + dc, r + dr
            if (nc, nr) in visited:
                continue
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if walls[nr, nc] or (nc, nr) in obstacles:
                continue
            visited.add((nc, nr))
            if (nc, nr) == end:
                return d + 1
            q.append(((nc, nr), d + 1))
    return INF


def pick_inspect_for_unlock(bs: BeliefState, cands: List[Candidate],
                              required_box: int, required_target: int) -> Optional[int]:
    """挑能 'unlock σ' 的最便宜 inspect.

    需要 unlock 的 box / target 中至少一个 — 任意识别一个就能让 Π 进一步收敛.
    打分: cost (走 + 旋转), 优先 cost 低且能识别 required_box/target 的.
    """
    walls = bs.M.astype(bool)
    obstacles = {(b.col, b.row) for b in bs.boxes}
    obstacles.update({(bm.col, bm.row) for bm in bs.bombs})
    car = (bs.player_col, bs.player_row)

    def _cost(c: Candidate) -> int:
        if c.viewpoint_col is None:
            return INF
        wc = _walk_cost(car, (c.viewpoint_col, c.viewpoint_row), walls, obstacles)
        if wc == INF:
            return INF
        cur = bs.theta_player
        tgt = c.inspect_heading or 0
        diff = (tgt - cur) % 8
        rot = min(diff, 8 - diff)
        return wc + rot

    best_idx = None
    best_score = (INF, INF)   # (优先级 0=直接 unlock 1=任意 unlock, cost)
    for k, c in enumerate(cands):
        if c.type != "inspect" or not c.legal:
            continue
        if c.inspect_target_type is None or c.inspect_target_idx is None:
            continue
        cost = _cost(c)
        if cost == INF:
            continue
        # 优先级: 直接 unlock required (0), 否则任意未识别 (1)
        if c.inspect_target_type == "box" and c.inspect_target_idx == required_box:
            prio = 0
        elif c.inspect_target_type == "target" and c.inspect_target_idx == required_target:
            prio = 0
        else:
            prio = 1
        score = (prio, cost)
        if score < best_score:
            best_score = score
            best_idx = k
    return best_idx


# ── 单 episode V2 ──────────────────────────────────────────

def collect_episode_v2(map_path: str, phase: int, seed: int,
                        *, time_limit: float = 60.0,
                        max_inspects_per_push: int = 8,
                        verify: bool = False,
                        ) -> Tuple[List[Sample], str, Dict]:
    """V2: god-mode A + suppress + insert inspect.

    verify=True 时, 每个 sample 在 clone engine 上跑 cand[label] 的自然执行路径
        (push: candidate_to_solver_move + apply_solver_move; inspect: apply_inspect),
        跟数据生成时实际 apply 后的 engine state 比对.
    """
    from experiments.sage_pr.evaluate_sage_pr import candidate_to_solver_move

    info = {"n_inspect": 0, "n_push": 0, "n_label_miss": 0,
            "n_force_apply_unsupp": 0, "trajectory_len": 0,
            "n_diverge": 0, "first_diverge_step": None,
            "diverge_detail": None}

    plan = plan_god_mode(map_path, seed, time_limit=time_limit)
    if plan is None:
        return [], "no_god_plan", info

    # 重置 (相同 seed 保 ID 一致), partial-obs 重放
    random.seed(seed)
    eng = GameEngine()
    eng.reset(map_path)

    eng_cand = copy.deepcopy(eng) if verify else None

    samples: List[Sample] = []
    a_idx = 0
    inspect_streak = 0   # 防止单步多次 inspect 死循环
    step_global = 0

    def _state_sig(e: GameEngine):
        s = e.get_state()
        return (
            round(s.car_x * 4) / 4, round(s.car_y * 4) / 4,
            round(s.car_angle * 16) / 16,
            tuple(sorted((pos_to_grid(b.x, b.y), b.class_id) for b in s.boxes)),
            tuple(sorted(pos_to_grid(b.x, b.y) for b in s.bombs)),
            tuple(sorted(s.seen_box_ids)),
            tuple(sorted(s.seen_target_ids)),
        )

    def _record_diverge(label_step, detail):
        info["n_diverge"] += 1
        if info["first_diverge_step"] is None:
            info["first_diverge_step"] = label_step
            info["diverge_detail"] = detail

    while a_idx < len(plan):
        bs = BeliefState.from_engine_state(eng.get_state(), fully_observed=False)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat, enforce_sigma_lock=True)

        move = plan[a_idx]
        label = match_move_to_candidate(move, cands, bs, run_length=1)

        if label is not None:
            # push 在 partial-obs + 抑制场下合法 → 录 push sample
            samples.append(Sample(
                X_grid=build_grid_tensor(bs, feat),
                X_cand=encode_candidates(cands, bs, feat),
                u_global=build_global_features(bs, feat),
                mask=candidates_legality_mask(cands),
                label=label, phase=phase, source=SOURCE_V2,
            ))
            if verify and eng_cand is not None:
                # cand[label] 的自然执行 (push) on clone
                cand_picked = cands[label]
                mock_move = candidate_to_solver_move(cand_picked, bs)
                ok_c = apply_solver_move(eng_cand, mock_move) if mock_move else False
                if not ok_c:
                    _record_diverge(step_global, "cand push apply failed")
                    eng_cand = copy.deepcopy(eng)
                if not apply_solver_move(eng, move):
                    return samples, "apply_fail", info
                # 比对 ref vs cand
                if _state_sig(eng) != _state_sig(eng_cand):
                    _record_diverge(step_global, "push state mismatch")
                    eng_cand = copy.deepcopy(eng)
            else:
                if not apply_solver_move(eng, move):
                    return samples, "apply_fail", info
            info["n_push"] += 1
            a_idx += 1
            inspect_streak = 0
            step_global += 1
            continue

        # push illegal — 多半因 σ 未锁. 找需要 unlock 的 box / target
        if inspect_streak >= max_inspects_per_push:
            # 兜底: 抑制场卡死, 强行用无抑制 cands 找 push (产模型不该见的样本但保数据)
            cands_nosup = generate_candidates(bs, feat, enforce_sigma_lock=False)
            label2 = match_move_to_candidate(move, cands_nosup, bs, run_length=1)
            if label2 is None:
                info["n_label_miss"] += 1
                if not apply_solver_move(eng, move):
                    return samples, "apply_fail", info
                if verify and eng_cand is not None:
                    apply_solver_move(eng_cand, move)   # 同步 cand engine
                a_idx += 1
                inspect_streak = 0
                step_global += 1
                continue
            # apply but DON'T record (会污染数据)
            info["n_force_apply_unsupp"] += 1
            if not apply_solver_move(eng, move):
                return samples, "apply_fail", info
            if verify and eng_cand is not None:
                apply_solver_move(eng_cand, move)
            a_idx += 1
            inspect_streak = 0
            step_global += 1
            continue

        # 选 inspect
        # required_box / required_target = move 指向的 entity 涉及的 box & 它该去的 target
        required_box = -1
        required_target = -1
        if move[0] == "box":
            old_pos, cid = move[1]
            for j, b in enumerate(bs.boxes):
                if (b.col, b.row) == old_pos:
                    required_box = j
                    break
            # 找匹配的 target
            for j, t in enumerate(bs.targets):
                if t.num_id == cid:
                    required_target = j
                    break

        ins_label = pick_inspect_for_unlock(bs, cands, required_box, required_target)
        if ins_label is None:
            # 没法 inspect — 兜底
            info["n_label_miss"] += 1
            if not apply_solver_move(eng, move):
                return samples, "apply_fail", info
            if verify and eng_cand is not None:
                apply_solver_move(eng_cand, move)
            a_idx += 1
            inspect_streak = 0
            step_global += 1
            continue

        ins_cand = cands[ins_label]
        samples.append(Sample(
            X_grid=build_grid_tensor(bs, feat),
            X_cand=encode_candidates(cands, bs, feat),
            u_global=build_global_features(bs, feat),
            mask=candidates_legality_mask(cands),
            label=ins_label, phase=phase, source=SOURCE_V2,
        ))
        if verify and eng_cand is not None:
            ok_c = apply_inspect(eng_cand, ins_cand)
            if not ok_c:
                _record_diverge(step_global, "cand inspect apply failed")
                eng_cand = copy.deepcopy(eng)
            if not apply_inspect(eng, ins_cand):
                return samples, "inspect_apply_fail", info
            if _state_sig(eng) != _state_sig(eng_cand):
                _record_diverge(step_global, "inspect state mismatch")
                eng_cand = copy.deepcopy(eng)
        else:
            if not apply_inspect(eng, ins_cand):
                return samples, "inspect_apply_fail", info
        info["n_inspect"] += 1
        inspect_streak += 1
        step_global += 1

    info["trajectory_len"] = info["n_push"] + info["n_inspect"]
    won = eng.get_state().won
    if not won:
        return samples, "did_not_win", info
    return samples, "ok", info


def _worker(args):
    map_path, phase, seed, time_limit, verify = args
    try:
        samples, status, info = collect_episode_v2(
            map_path, phase, seed, time_limit=time_limit, verify=verify)
        return {"map": map_path, "seed": seed, "n": len(samples),
                "status": status, "samples": samples, "info": info}
    except Exception as e:
        return {"map": map_path, "seed": seed, "n": 0,
                "status": f"error:{type(e).__name__}", "samples": [], "info": {}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--n-maps", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--max-seeds-per-map", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify", action="store_true",
                        help="每步在 clone engine 跑 cand[label] 自然路径, 跟实际 apply 比对")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"V2 Phase {args.phase}, time_limit={args.time_limit}s")

    maps = list_phase_maps(args.phase, args.n_maps)
    if not maps:
        sys.exit(1)

    verified_map = parse_phase456_seeds(
        os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
    ) if args.use_verified_seeds else {}

    tasks = []
    if args.use_verified_seeds and verified_map:
        items_phase = [(k, v) for k, v in verified_map.items()
                       if f"phase{args.phase}/" in k]
        items_phase.sort()
        if args.n_maps is not None:
            items_phase = items_phase[:args.n_maps]
        for map_path, ms in items_phase:
            full = os.path.join(ROOT, map_path)
            if not os.path.exists(full):
                continue
            n_per = args.max_seeds_per_map or max(1, len(seeds))
            for seed in ms[:n_per]:
                tasks.append((map_path, args.phase, seed, args.time_limit, args.verify))
    else:
        for map_path in maps:
            for seed in seeds:
                tasks.append((map_path, args.phase, seed, args.time_limit, args.verify))
    print(f"  total tasks: {len(tasks)}")

    t0 = time.perf_counter()
    if args.workers <= 1:
        results = [_worker(t) for t in tasks]
    else:
        with mp.Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_worker, tasks, chunksize=1))

    samples: List[Sample] = []
    status_counts: Dict[str, int] = {}
    n_inspect_total = 0
    n_push_total = 0
    n_miss_total = 0
    n_force_total = 0
    n_diverge_total = 0
    diverge_episodes = 0
    diverge_log: List[Dict] = []
    for r in results:
        st = r["status"]
        status_counts[st] = status_counts.get(st, 0) + 1
        if st == "ok":
            samples.extend(r["samples"])
        info = r.get("info", {})
        n_inspect_total += info.get("n_inspect", 0)
        n_push_total += info.get("n_push", 0)
        n_miss_total += info.get("n_label_miss", 0)
        n_force_total += info.get("n_force_apply_unsupp", 0)
        nd = info.get("n_diverge", 0)
        n_diverge_total += nd
        if nd > 0:
            diverge_episodes += 1
            diverge_log.append({
                "map": r["map"], "seed": r["seed"],
                "first_diverge_step": info.get("first_diverge_step"),
                "diverge_detail": info.get("diverge_detail"),
                "n_diverge": nd,
            })

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s, samples={len(samples)}")
    print(f"  episode status: {status_counts}")
    print(f"  trajectory: {n_push_total} push samples + {n_inspect_total} inspect samples")
    print(f"  inspect ratio: {n_inspect_total/max(1, n_push_total+n_inspect_total)*100:.1f}%")
    print(f"  label_miss (fallthrough): {n_miss_total}, force_apply_unsupp: {n_force_total}")
    if args.verify:
        print(f"  verify: diverge_steps={n_diverge_total} diverge_episodes={diverge_episodes}")
        if diverge_log:
            print(f"  --- first 10 diverge episodes ---")
            for d in diverge_log[:10]:
                print(f"    {d}")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    save_dataset(samples, out_path)


if __name__ == "__main__":
    main()
