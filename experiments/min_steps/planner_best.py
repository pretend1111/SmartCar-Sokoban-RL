"""Planner_best — 跑 v1 / v4 / v6 三个, 取胜出且步数最少的真正执行.

为了避免 monkey-patch counter 污染, measurement engine 独立从 map_path/seed 创建.
但是 planner 接口是 (eng) → None, 所以这里要 hack: 用全局 var 传 map_path/seed.
"""

from __future__ import annotations

import copy
import random
from typing import List, Tuple, Callable, Optional

from smartcar_sokoban.engine import GameEngine

from experiments.min_steps.planner_v1 import planner_v1_opportunistic
from experiments.min_steps.planner_v2 import planner_v2_walk_first
from experiments.min_steps.planner_v4 import planner_v4_detour_aware
from experiments.min_steps.planner_v6 import planner_v6_tsp_explore
from experiments.min_steps.planner_v7 import planner_v7_dp_interleave
from experiments.min_steps.harness import planner_v1_explore_first


# 全局 var (hack): _run_baseline.py 在调 planner 前设置
_BEST_MAP_PATH: str = ""
_BEST_SEED: int = 0


def set_best_context(map_path: str, seed: int) -> None:
    global _BEST_MAP_PATH, _BEST_SEED
    _BEST_MAP_PATH = map_path
    _BEST_SEED = seed


def _fresh_engine() -> GameEngine:
    """从 map_path/seed 重建一个干净 engine (没 instrumented)."""
    random.seed(_BEST_SEED)
    eng = GameEngine()
    eng.reset(_BEST_MAP_PATH)
    eng.discrete_step(6)   # init snap
    return eng


def _measure(planner: Callable) -> Tuple[int, bool]:
    eng = _fresh_engine()
    counter = {"_total": 0}
    orig = eng.discrete_step

    def wrapped(a):
        counter["_total"] += 1
        return orig(a)
    eng.discrete_step = wrapped  # type: ignore
    eng._step_tag = "?"

    try:
        planner(eng)
    except Exception:
        pass
    won = eng.get_state().won
    return counter["_total"], won


def planner_best_of_three(eng: GameEngine,
                            *, candidates: Optional[List[Callable]] = None) -> None:
    """跑 v1/v4/v6 measurement, 选最少步数, 在主 eng 上跑那个."""
    if candidates is None:
        candidates = [
            planner_v1_explore_first,
            planner_v1_opportunistic,
            planner_v2_walk_first,
            planner_v4_detour_aware,
            planner_v6_tsp_explore,
            planner_v7_dp_interleave,
        ]

    best_planner = None
    best_steps = float('inf')
    for p in candidates:
        steps, won = _measure(p)
        if won and steps < best_steps:
            best_steps = steps
            best_planner = p

    if best_planner is None:
        best_planner = candidates[0]

    eng._step_tag = "best"
    best_planner(eng)
