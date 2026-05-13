"""测量台架 — 给定 (map, seed, planner), 跑完整张图, 数每一步 engine.discrete_step.

planner 接口:
    def planner(eng: GameEngine) -> dict:
        # eng 已 reset 完毕, 实时调用 eng.discrete_step(...) 推进
        # 返回 {"won": bool, "total_steps": int, "tag_counts": {...}, ...}
        ...

测量靠 monkey-patch `eng.discrete_step`: 每次调用 +1 步, 并按 tag 分类计数
(explore / push / inspect / etc) — tag 由 planner 在调用前设置 eng._step_tag.
"""

from __future__ import annotations

import contextlib
import copy
import io
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine


@dataclass
class StepResult:
    map_path: str
    seed: int
    planner: str
    won: bool
    total_steps: int
    tag_counts: Dict[str, int] = field(default_factory=dict)
    wall_time_s: float = 0.0
    note: str = ""


def instrument(eng: GameEngine) -> Dict[str, int]:
    """给 engine 装上步数计数器, 返回外部可读的 counts dict.

    用法:
        counts = instrument(eng)
        eng._step_tag = "explore"      # planner 设置
        eng.discrete_step(...)         # 自动 counts["explore"] += 1
    """
    counts: Dict[str, int] = {"_total": 0}
    eng._step_tag = "?"
    orig = eng.discrete_step

    def wrapped(a):
        tag = getattr(eng, "_step_tag", "?")
        counts[tag] = counts.get(tag, 0) + 1
        counts["_total"] += 1
        return orig(a)

    eng.discrete_step = wrapped       # type: ignore[assignment]
    return counts


def run_planner(map_path: str, seed: int, planner_name: str,
                planner_fn: Callable[[GameEngine], None],
                *, step_limit: int = 1000) -> StepResult:
    """跑一遍 planner, 返回测量结果."""
    import random
    random.seed(seed)
    eng = GameEngine()
    eng.reset(map_path)
    counts = instrument(eng)

    # 首次 snap 锁定网格 — 否则后续第一个 action 会被吞掉当 snap.
    eng._step_tag = "init_snap"
    eng.discrete_step(6)

    t0 = time.perf_counter()
    note = ""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            planner_fn(eng)
    except Exception as e:
        note = f"exception: {type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0

    state = eng.get_state()
    return StepResult(
        map_path=map_path,
        seed=seed,
        planner=planner_name,
        won=bool(state.won),
        total_steps=counts.get("_total", 0),
        tag_counts={k: v for k, v in counts.items() if k != "_total"},
        wall_time_s=elapsed,
        note=note,
    )


def run_planners_on_map(map_path: str, seed: int,
                         planners: Dict[str, Callable[[GameEngine], None]],
                         ) -> List[StepResult]:
    """同图同 seed 上比较多个 planner."""
    results = []
    for name, fn in planners.items():
        r = run_planner(map_path, seed, name, fn)
        results.append(r)
    return results


# ── 基线 planner 实现 ─────────────────────────────────────

def planner_v1_explore_first(eng: GameEngine) -> None:
    """V1: plan_exploration_v3 + MultiBoxSolver, 当前 baseline."""
    from smartcar_sokoban.solver.explorer_v3 import plan_exploration_v3
    from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
    from smartcar_sokoban.solver.pathfinder import pos_to_grid, bfs_path
    from smartcar_sokoban.action_defs import direction_to_abs_action

    # Phase 1: explore
    eng._step_tag = "explore"
    plan_exploration_v3(eng, max_retries=15, verbose=False)

    state = eng.get_state()

    # Phase 2: solve push plan (god mode, 全 ID 已知)
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    moves = solver.solve(max_cost=300, time_limit=30.0, strategy="auto")
    if not moves:
        return

    # Phase 3: 展开每个 move 为 BFS-walk + push
    eng._step_tag = "push"
    for move in moves:
        _apply_move(eng, move)


def planner_v0_godmode_lowerbound(eng: GameEngine) -> None:
    """对照: 不探索, 直接 god-mode MultiBoxSolver. 假设 ID 已知 (作弊).
    给 push-only 下界."""
    from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
    from smartcar_sokoban.solver.pathfinder import pos_to_grid

    state = eng.get_state()
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    moves = solver.solve(max_cost=300, time_limit=30.0, strategy="auto")
    if not moves:
        return

    eng._step_tag = "push"
    for move in moves:
        _apply_move(eng, move)


# ── 低层执行辅助 (与 build_dataset_v3.apply_solver_move 等价, 但用 eng._step_tag) ──

def _apply_move(eng: GameEngine, move) -> bool:
    """把 (entity_type, eid, direction, ...) move 展开成实际 discrete_step.
    注: 不做 snap (action 6) — 它是 no-op, 但会污染步数."""
    from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
    from smartcar_sokoban.action_defs import direction_to_abs_action

    etype, eid, direction, _ = move
    state = eng.get_state()
    dx, dy = direction

    if etype == "box":
        old_pos, cid = eid
        ec, er = old_pos
    elif etype == "bomb":
        ec, er = eid
    else:
        return False
    car_target = (ec - dx, er - dy)

    obstacles = set()
    for b in state.boxes:
        obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs:
        obstacles.add(pos_to_grid(bm.x, bm.y))
    car_grid = pos_to_grid(state.car_x, state.car_y)

    if car_grid != car_target:
        path = bfs_path(car_grid, car_target, state.grid, obstacles)
        if path is None:
            return False
        for pdx, pdy in path:
            eng.discrete_step(direction_to_abs_action(pdx, pdy))
    eng.discrete_step(direction_to_abs_action(dx, dy))
    return True


# ── 主入口: 批量测一组 (map, seed) ─────────────────────────

def benchmark(maps: List[Tuple[str, int]],
              planners: Dict[str, Callable[[GameEngine], None]]
              ) -> List[StepResult]:
    out: List[StepResult] = []
    for mp, sd in maps:
        for name, fn in planners.items():
            r = run_planner(mp, sd, name, fn)
            out.append(r)
    return out


def print_table(results: List[StepResult]) -> None:
    """简洁汇总: 一行一个 (map, seed, planner) 结果."""
    print(f"{'map':<40} {'seed':>5} {'planner':<28} {'won':>4} {'steps':>6} "
          f"{'tags':<40} {'time':>6}")
    print("-" * 140)
    for r in results:
        tag_str = " ".join(f"{k}={v}" for k, v in sorted(r.tag_counts.items()))
        mp = r.map_path.split("/")[-1]
        won = "✓" if r.won else "✗"
        print(f"{mp:<40} {r.seed:>5} {r.planner:<28} {won:>4} "
              f"{r.total_steps:>6} {tag_str:<40} {r.wall_time_s:>5.1f}s")


def summary(results: List[StepResult]) -> None:
    """按 planner 平均."""
    from collections import defaultdict
    agg: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"n": 0, "won": 0, "steps_sum": 0, "time_sum": 0.0})
    for r in results:
        a = agg[r.planner]
        a["n"] += 1
        a["won"] += int(r.won)
        if r.won:
            a["steps_sum"] += r.total_steps
        a["time_sum"] += r.wall_time_s

    print()
    print(f"{'planner':<28} {'won':>10} {'avg_steps (win-only)':>22} {'avg_time':>10}")
    print("-" * 80)
    for name, a in agg.items():
        win_rate = a["won"] / max(1, a["n"]) * 100
        avg_steps = a["steps_sum"] / max(1, a["won"])
        avg_time = a["time_sum"] / max(1, a["n"])
        print(f"{name:<28} {a['won']:>4}/{a['n']:<4} ({win_rate:>5.1f}%)  "
              f"{avg_steps:>20.1f} {avg_time:>8.2f}s")
