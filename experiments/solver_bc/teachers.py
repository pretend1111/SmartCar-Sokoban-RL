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
from experiments.solver_bc.oracle_features import (
    DIR_TO_INDEX,
    build_solver_from_state,
    map_solver_move_to_env_action,
)
from smartcar_sokoban.rl.high_level_env import (
    DIR_DELTAS,
    N_DIRS,
    PUSH_BOMB_START,
    PUSH_BOX_START,
    SokobanHLEnv,
)
from smartcar_sokoban.solver.auto_player import AutoPlayer


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
                            time_limit: float) -> Optional[List[int]]:
    env = _make_env(map_path, seed, max_steps, include_map_layout)
    state = env.engine.get_state()
    solver = build_solver_from_state(state)
    with contextlib.redirect_stdout(io.StringIO()):
        solution = solver.solve(max_cost=max_cost, time_limit=time_limit)
    if not solution:
        return None

    actions: List[int] = []
    for move in solution:
        state = env.engine.get_state()
        action = map_solver_move_to_env_action(state, move)
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
    before_car = (int(before.car_x), int(before.car_y))
    after_car = (int(after.car_x), int(after.car_y))
    delta = (after_car[0] - before_car[0], after_car[1] - before_car[1])
    if delta not in DIR_TO_INDEX:
        return None

    moved_into = after_car

    for idx, box in enumerate(before.boxes):
        if (int(box.x), int(box.y)) == moved_into:
            return PUSH_BOX_START + idx * N_DIRS + DIR_TO_INDEX[delta]

    for idx, bomb in enumerate(before.bombs):
        if (int(bomb.x), int(bomb.y)) == moved_into:
            return PUSH_BOMB_START + idx * N_DIRS + DIR_TO_INDEX[delta]

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
        before_car = (int(before.car_x), int(before.car_y))
        after = replay_engine.discrete_step(low_action)
        after_car = (int(after.car_x), int(after.car_y))
        delta = (after_car[0] - before_car[0], after_car[1] - before_car[1])
        if delta not in DIR_DELTAS:
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
                            time_limit: float) -> Tuple[Optional[List[int]], str]:
    if teacher == "solver":
        return (
            _solver_action_sequence(
                map_path,
                seed,
                max_steps,
                include_map_layout,
                max_cost,
                time_limit,
            ),
            "solver",
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
        )
        if solver_actions:
            return solver_actions, "solver"
        return _autoplayer_action_sequence(map_path, seed), "autoplayer"

    raise ValueError(f"unknown teacher: {teacher}")
