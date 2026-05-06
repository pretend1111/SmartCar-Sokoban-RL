from __future__ import annotations

import copy
import contextlib
import io
import os
import random
import sys
from typing import List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.rl.high_level_env import SokobanHLEnv
from smartcar_sokoban.solver.auto_player import AutoPlayer
from smartcar_sokoban.solver.high_level_teacher import (
    BOMB_DIR_DELTAS,
    BOMB_DIR_TO_INDEX,
    N_DIRS,
    N_BOMB_DIRS,
    PUSH_BOMB_START,
    PUSH_BOX_START,
    advise_exact_high_level,
)
from smartcar_sokoban.solver.pathfinder import pos_to_grid


def _make_env(map_path: str, seed: int, max_steps: int,
              include_map_layout: bool) -> SokobanHLEnv:
    env = SokobanHLEnv(
        map_file=map_path,
        base_dir=ROOT,
        max_steps=max_steps,
        include_map_layout=include_map_layout,
    )
    env.reset(seed=seed)
    return env


def _solver_action_sequence(map_path: str, seed: int, max_steps: int,
                            include_map_layout: bool,
                            max_cost: int,
                            time_limit: float,
                            strategy: str = "auto") -> Optional[List[int]]:
    env = _make_env(map_path, seed, max_steps, include_map_layout)

    actions: List[int] = []
    while True:
        state = env.engine.get_state()
        with contextlib.redirect_stdout(io.StringIO()):
            advice = advise_exact_high_level(
                state,
                max_cost=max_cost,
                time_limit=time_limit,
                strategy=strategy,
            )
        action = advice.primary_action
        if action is None:
            return None
        mask = env.action_masks()
        if not bool(mask[action]):
            return None
        actions.append(action)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    if env.engine.get_state().won:
        return actions
    return None


def _infer_push_action_from_transition(before, after) -> Optional[int]:
    before_car = pos_to_grid(before.car_x, before.car_y)
    after_car = pos_to_grid(after.car_x, after.car_y)
    delta = (after_car[0] - before_car[0], after_car[1] - before_car[1])

    moved_into = after_car

    for idx, box in enumerate(before.boxes):
        if pos_to_grid(box.x, box.y) == moved_into:
            dir_idx = BOMB_DIR_TO_INDEX.get(delta)
            if dir_idx is None or dir_idx >= N_DIRS:
                return None
            return PUSH_BOX_START + idx * N_DIRS + dir_idx

    for idx, bomb in enumerate(before.bombs):
        if pos_to_grid(bomb.x, bomb.y) == moved_into:
            dir_idx = BOMB_DIR_TO_INDEX.get(delta)
            if dir_idx is None:
                return None
            return PUSH_BOMB_START + idx * N_BOMB_DIRS + dir_idx

    return None


def _autoplayer_action_sequence(map_path: str, seed: int) -> Optional[List[int]]:
    cfg = GameConfig()
    cfg.control_mode = "discrete"

    teacher_engine = GameEngine(cfg, ROOT)
    random.seed(seed)
    teacher_engine.reset(map_path)

    with contextlib.redirect_stdout(io.StringIO()):
        low_level_actions = AutoPlayer(teacher_engine).solve()

    if not teacher_engine.get_state().won:
        return None

    replay_engine = GameEngine(cfg, ROOT)
    random.seed(seed)
    replay_engine.reset(map_path)

    high_level_actions: List[int] = []
    for low_action in low_level_actions:
        before = copy.deepcopy(replay_engine.get_state())
        before_car = pos_to_grid(before.car_x, before.car_y)
        after = replay_engine.discrete_step(low_action)
        after_car = pos_to_grid(after.car_x, after.car_y)
        delta = (after_car[0] - before_car[0], after_car[1] - before_car[1])
        if delta not in BOMB_DIR_DELTAS:
            continue

        action = _infer_push_action_from_transition(before, after)
        if action is not None:
            high_level_actions.append(action)

    if replay_engine.get_state().won:
        return high_level_actions
    return None


def collect_teacher_actions(map_path: str, seed: int, *,
                            teacher: str,
                            max_steps: int,
                            include_map_layout: bool,
                            max_cost: int,
                            time_limit: float,
                            strategy: str = "auto"
                            ) -> Tuple[Optional[List[int]], str]:
    if teacher in ("solver", "solver_ida"):
        actual_strategy = "ida" if teacher == "solver_ida" else strategy
        return (
            _solver_action_sequence(
                map_path,
                seed,
                max_steps,
                include_map_layout,
                max_cost,
                time_limit,
                strategy=actual_strategy,
            ),
            f"solver_{actual_strategy}",
        )

    if teacher == "autoplayer":
        return _autoplayer_action_sequence(map_path, seed), "autoplayer"

    if teacher == "hybrid":
        solver_actions = _solver_action_sequence(
            map_path,
            seed,
            max_steps,
            include_map_layout,
            max_cost,
            time_limit,
            strategy=strategy,
        )
        if solver_actions:
            return solver_actions, f"solver_{strategy}"
        return _autoplayer_action_sequence(map_path, seed), "autoplayer"

    raise ValueError(f"unknown teacher: {teacher}")
