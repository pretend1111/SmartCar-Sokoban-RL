"""候选感知数据生成 — SAGE-PR P3.1.

每个样本格式 (npz keys):
    X_grid:    [10, 14, 30] float32
    X_cand:    [64, 128]    float32
    u_global:  [16]         float32
    mask:      [64]         float32 (合法 1 / 非法 0)
    label:     int64        (≤ 64, 老师选的候选索引)
    phase:     int8 (1..6)
    source:    int8 (0=ida, 1=best_first, 2=auto_player)

设计:
    1. 用 MultiBoxSolver 求解每张图 (god mode), 拿移动序列.
    2. 重新模拟引擎, 每次执行 1 移动前: 捕获 belief / 算特征 / 生成候选 /
       匹配 (box_idx, dir, run=1) 找 label.
    3. 保存 [N_samples] 张样本到 npz.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    build_grid_tensor, build_global_features,
    GRID_TENSOR_CHANNELS, GLOBAL_DIM,
)


SOURCE_IDA = 0
SOURCE_BF = 1
SOURCE_AUTO = 2

SOURCE_NAMES = {SOURCE_IDA: "ida", SOURCE_BF: "best_first", SOURCE_AUTO: "auto"}


# ── 求解 ──────────────────────────────────────────────────

def solve_map(state, max_cost: int, time_limit: float, strategy: str) -> Optional[List]:
    """用 MultiBoxSolver 求解, 返回 moves 列表 (None = 失败)."""
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)

    with contextlib.redirect_stdout(io.StringIO()):
        return solver.solve(max_cost=max_cost, time_limit=time_limit, strategy=strategy)


# ── 引擎重放 ──────────────────────────────────────────────

def apply_solver_move(eng: GameEngine, move) -> bool:
    """通过 high_level_env 等价语义把一步 solver move 应用到 engine.

    返回 True 表示成功. False 表示失败 (跳过样本).

    注: engine.discrete_step 第一次调用若未对齐网格会消耗为 snap. 调用前先做
    一次 no-op 来强制 snap, 避免后续 path 第一步被吞掉.
    """
    from smartcar_sokoban.solver.pathfinder import bfs_path
    from smartcar_sokoban.action_defs import direction_to_abs_action as direction_to_action

    # 强制 snap (action=6 是 no-op, 但触发 snap 逻辑)
    eng.discrete_step(6)

    etype, eid, direction, _ = move
    state = eng.get_state()
    dx, dy = direction

    # 找实体
    if etype == "box":
        old_pos, cid = eid
        ec, er = old_pos
        # 推位 = 实体反方向
        car_target = (ec - dx, er - dy)
    elif etype == "bomb":
        ec, er = eid
        car_target = (ec - dx, er - dy)
    else:
        return False

    # 障碍物 = 所有实体
    obstacles = set()
    for b in state.boxes:
        obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs:
        obstacles.add(pos_to_grid(bm.x, bm.y))

    car_grid = pos_to_grid(state.car_x, state.car_y)

    # 导航到推位
    if car_grid != car_target:
        path = bfs_path(car_grid, car_target, state.grid, obstacles)
        if path is None:
            return False
        for pdx, pdy in path:
            a = direction_to_action(pdx, pdy)
            eng.discrete_step(a)

    # 推
    a = direction_to_action(dx, dy)
    eng.discrete_step(a)
    return True


# ── 候选 label 匹配 ──────────────────────────────────────

def match_move_to_candidate(move, cands: List[Candidate],
                            bs: BeliefState, run_length: int = 1) -> Optional[int]:
    """找 candidate index 使 (type, box_idx 或 bomb_idx, direction, run_length) 匹配 move."""
    etype, eid, direction, _ = move

    if etype == "box":
        old_pos, cid = eid
        for i, b in enumerate(bs.boxes):
            if (b.col, b.row) == old_pos and b.class_id == cid:
                for k, c in enumerate(cands):
                    if c.type != "push_box" or not c.legal:
                        continue
                    if c.box_idx == i and c.direction == direction and c.run_length == run_length:
                        return k
                return None
    elif etype == "bomb":
        old_pos = eid
        for i, bm in enumerate(bs.bombs):
            if (bm.col, bm.row) == old_pos:
                for k, c in enumerate(cands):
                    if c.type != "push_bomb" or not c.legal:
                        continue
                    if c.bomb_idx == i and c.direction == direction:
                        return k
                return None
    return None


def compute_macro_run_length(moves: List, idx: int, max_k: int = 3) -> int:
    """看 moves[idx:] 头部连续多少步是 same (etype, direction) 且 entity 顺序串联."""
    if idx >= len(moves):
        return 0
    m_i = moves[idx]
    etype_i, eid_i, dir_i, _ = m_i
    if etype_i != "box":
        return 1   # 炸弹不做 macro
    pos_i, cid_i = eid_i
    cur_pos = pos_i
    k = 1
    while k < max_k and idx + k < len(moves):
        m_k = moves[idx + k]
        etype_k, eid_k, dir_k, _ = m_k
        if etype_k != "box" or dir_k != dir_i:
            break
        pos_k, cid_k = eid_k
        if cid_k != cid_i:
            break
        expected = (cur_pos[0] + dir_i[0], cur_pos[1] + dir_i[1])
        if pos_k != expected:
            break
        cur_pos = pos_k
        k += 1
    return k


# ── 单 episode 采集 ──────────────────────────────────────

@dataclass
class Sample:
    X_grid: np.ndarray
    X_cand: np.ndarray
    u_global: np.ndarray
    mask: np.ndarray
    label: int
    phase: int
    source: int


def collect_episode(map_path: str, phase: int, seed: int,
                    *, strategy: str, max_cost: int, time_limit: float,
                    fully_observed: bool = True,
                    use_macro: bool = True,
                    pre_explorer: bool = True) -> Tuple[List[Sample], str]:
    """采集单张图的所有样本.

    Returns:
        (samples, status)
        status in {"ok", "no_solve", "label_miss", "step_fail", "no_samples", "explore_fail"}

    Args:
        use_macro: 若 True, 把连续 same (entity, dir) 推送压成 macro action
            (run_length=1..3), 缩短轨迹长度.
        pre_explorer: 若 True (默认), 先跑 plan_exploration_v3 让车走到部署起点,
            然后从 post-explorer 状态求解 + 采样. 跟部署架构对齐 (explorer 跑在板上,
            NN 接管推箱).
    """
    eng = GameEngine()
    import smartcar_sokoban.map_loader as map_loader_module
    # 设置 seed (random.shuffle 用全局 random)
    import random
    random.seed(seed)

    state = eng.reset(map_path)

    # 部署对齐: 跑 explorer, 让 engine 推进到 NN 接管的真实起点
    if pre_explorer:
        from smartcar_sokoban.solver.explorer_v3 import plan_exploration_v3
        from smartcar_sokoban.solver.explorer import exploration_complete
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                plan_exploration_v3(eng, max_retries=15)
            except Exception:
                return [], "explore_fail"
        if not exploration_complete(eng.get_state()):
            return [], "explore_fail"
        state = eng.get_state()

    moves = solve_map(state, max_cost=max_cost, time_limit=time_limit, strategy=strategy)
    if not moves:
        return [], "no_solve"

    source = SOURCE_IDA if strategy == "ida" else (
        SOURCE_BF if strategy == "best_first" else SOURCE_AUTO
    )

    # NOTE: 不再二次 reset. engine 当前状态 = post-explorer (或初始, 取决于 pre_explorer),
    # 是 solver 算 moves 的起点, 直接从这开始采样.

    samples: List[Sample] = []
    label_miss = 0
    i = 0

    while i < len(moves):
        # 捕获状态
        bs = BeliefState.from_engine_state(eng.state, fully_observed=fully_observed)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat)

        # 计算 macro run_length
        if use_macro:
            k_macro = compute_macro_run_length(moves, i, max_k=3)
        else:
            k_macro = 1

        # 优先匹配 macro, 失败回落到 1-step
        label = None
        actual_k = 1
        for try_k in range(k_macro, 0, -1):
            label = match_move_to_candidate(moves[i], cands, bs, run_length=try_k)
            if label is not None:
                actual_k = try_k
                break

        if label is None:
            label_miss += 1
            # 推进引擎 (跳过该 sample)
            if not apply_solver_move(eng, moves[i]):
                return samples, "step_fail"
            i += 1
            continue

        X_grid = build_grid_tensor(bs, feat)
        X_cand = encode_candidates(cands, bs, feat)
        u_global = build_global_features(bs, feat)
        mask = candidates_legality_mask(cands)

        samples.append(Sample(
            X_grid=X_grid,
            X_cand=X_cand,
            u_global=u_global,
            mask=mask,
            label=label,
            phase=phase,
            source=source,
        ))

        # 应用 actual_k 次 (macro)
        for j in range(actual_k):
            if not apply_solver_move(eng, moves[i + j]):
                return samples, "step_fail"
        i += actual_k

    if not samples:
        return [], "no_samples"
    return samples, ("ok" if label_miss == 0 else f"ok_miss_{label_miss}")


# ── 多图 worker ──────────────────────────────────────────

def _worker_collect(args) -> Dict[str, Any]:
    map_path, phase, seed, strategy, max_cost, time_limit, use_macro = args
    try:
        samples, status = collect_episode(
            map_path=map_path, phase=phase, seed=seed,
            strategy=strategy, max_cost=max_cost, time_limit=time_limit,
            use_macro=use_macro,
        )
    except Exception as e:
        return {"map": map_path, "seed": seed, "n": 0, "status": f"error: {e}", "samples": []}
    return {
        "map": map_path,
        "seed": seed,
        "n": len(samples),
        "status": status,
        "samples": samples,
    }


# ── 数据集 IO ────────────────────────────────────────────

def save_dataset(samples: List[Sample], out_path: str) -> None:
    n = len(samples)
    if n == 0:
        print("⚠️  no samples to save")
        return
    X_grid = np.zeros((n, 10, 14, GRID_TENSOR_CHANNELS), dtype=np.float32)
    X_cand = np.zeros((n, MAX_CANDIDATES, 128), dtype=np.float32)
    u_global = np.zeros((n, GLOBAL_DIM), dtype=np.float32)
    mask = np.zeros((n, MAX_CANDIDATES), dtype=np.float32)
    label = np.zeros(n, dtype=np.int64)
    phase = np.zeros(n, dtype=np.int8)
    source = np.zeros(n, dtype=np.int8)

    for i, s in enumerate(samples):
        X_grid[i] = s.X_grid
        X_cand[i] = s.X_cand
        u_global[i] = s.u_global
        mask[i] = s.mask
        label[i] = s.label
        phase[i] = s.phase
        source[i] = s.source

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        X_grid=X_grid, X_cand=X_cand, u_global=u_global,
        mask=mask, label=label, phase=phase, source=source,
    )
    print(f"  saved {n} samples → {out_path}")


# ── CLI ──────────────────────────────────────────────────

def parse_phase456_seeds(manifest_path: str) -> Dict[str, List[int]]:
    """从 phase456_seed_manifest.json 读 (map_rel_path → [seed_list]).

    Manifest 结构: {"phases": {"phase4": {"phase4_01.txt": {"verified_seeds": [...], ...}}}}
    返回: {"assets/maps/phase4/phase4_01.txt": [0, 5, 12, ...]}
    """
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, List[int]] = {}
    phases = data.get("phases", {})
    for phase_key, phase_dict in phases.items():
        # phase_key e.g. "phase4"
        for fname, info in phase_dict.items():
            seeds = info.get("verified_seeds", [])
            map_path = f"assets/maps/{phase_key}/{fname}"
            out[map_path] = list(seeds)
    return out


def list_phase_maps(phase: int, n_max: Optional[int] = None) -> List[str]:
    """枚举 assets/maps/phase{N}/*.txt. 返回 forward-slash paths (跨 OS 一致)."""
    folder = os.path.join(ROOT, f"assets/maps/phase{phase}")
    if not os.path.isdir(folder):
        return []
    files = sorted([
        f"assets/maps/phase{phase}/{fn}"   # 强制 forward-slash, 与 manifest 一致
        for fn in os.listdir(folder)
        if fn.endswith(".txt") and fn.startswith(f"phase{phase}_")
    ])
    if n_max is not None:
        files = files[:n_max]
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--n-maps", type=int, default=None)
    parser.add_argument("--seeds", type=str, default="0,42,137",
                        help="CSV of seeds to use per map")
    parser.add_argument("--strategy", choices=["auto", "best_first", "ida"], default="best_first")
    parser.add_argument("--max-cost", type=int, default=300)
    parser.add_argument("--time-limit", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=str, required=True,
                        help="输出 npz 路径 (如 .agent/sage_pr/phase4.npz)")
    parser.add_argument("--use-verified-seeds", action="store_true",
                        help="按 phase456_seed_manifest 选 seed")
    parser.add_argument("--max-seeds-per-map", type=int, default=None,
                        help="对 verified maps 的最大 seed 数 (默认与 --seeds 数相同)")
    parser.add_argument("--no-macro", action="store_true",
                        help="只用 1-step labels (默认: macro labels 1-3 步合并)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"Phase {args.phase}, strategy={args.strategy}, "
          f"seeds={seeds}, time_limit={args.time_limit}s, workers={args.workers}")

    maps = list_phase_maps(args.phase, args.n_maps)
    print(f"  found {len(maps)} maps")
    if not maps:
        sys.exit(1)

    verified_map = {}
    if args.use_verified_seeds:
        verified_map = parse_phase456_seeds(
            os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
        )
        print(f"  verified manifest entries: {len(verified_map)}")

    # 任务列表
    tasks = []
    if args.use_verified_seeds and verified_map:
        # 用 manifest 直接列任务 (只跑 verified maps), 限制数量 = n_maps 或全部
        items = sorted(verified_map.items())
        if args.n_maps is not None:
            items = items[:args.n_maps]
        for map_path, ms in items:
            full_path = os.path.join(ROOT, map_path)
            if not os.path.exists(full_path):
                continue
            if args.phase is not None and f"phase{args.phase}/" not in map_path:
                continue
            # 限制每图 seed 数 (默认与 --seeds 数相同, 或 --max-seeds-per-map)
            n_per = args.max_seeds_per_map or max(1, len(seeds))
            ms_use = ms[:n_per]
            for seed in ms_use:
                tasks.append((map_path, args.phase, seed, args.strategy,
                              args.max_cost, args.time_limit,
                              not args.no_macro))
    else:
        for map_path in maps:
            for seed in seeds:
                tasks.append((map_path, args.phase, seed, args.strategy,
                              args.max_cost, args.time_limit,
                              not args.no_macro))

    print(f"  total tasks: {len(tasks)}")

    t0 = time.perf_counter()

    if args.workers <= 1:
        results = [_worker_collect(t) for t in tasks]
    else:
        with mp.Pool(processes=args.workers) as pool:
            results = list(pool.imap_unordered(_worker_collect, tasks, chunksize=1))

    samples: List[Sample] = []
    n_ok = n_no_solve = n_label_miss = n_other = 0
    for r in results:
        if r["status"] == "no_solve":
            n_no_solve += 1
        elif r["status"].startswith("ok_miss_"):
            n_label_miss += 1
            samples.extend(r["samples"])
        elif r["status"] == "ok":
            n_ok += 1
            samples.extend(r["samples"])
        else:
            n_other += 1

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s. samples={len(samples)} "
          f"(ok {n_ok} / partial {n_label_miss} / no_solve {n_no_solve} / other {n_other})")

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT, out_path)
    save_dataset(samples, out_path)


if __name__ == "__main__":
    main()
