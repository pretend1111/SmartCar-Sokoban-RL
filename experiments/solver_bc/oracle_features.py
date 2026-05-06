from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from smartcar_sokoban.rl.high_level_env import (
    DIR_DELTAS,
    BOMB_DIR_DELTAS,
    MAP_COLS,
    MAP_ROWS,
    MAX_BOMBS,
    MAX_BOXES,
    MAX_TARGETS,
    N_ACTIONS,
    N_DIRS,
    N_BOMB_DIRS,
    PUSH_BOMB_START,
    PUSH_BOX_START,
    STATE_DIM,
    STATE_DIM_WITH_MAP,
)
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid

Pos = Tuple[int, int]
SolverMove = Tuple[str, object, Tuple[int, int], int]

DIR_TO_INDEX = {delta: idx for idx, delta in enumerate(DIR_DELTAS)}
BOMB_DIR_TO_INDEX = {delta: idx for idx, delta in enumerate(BOMB_DIR_DELTAS)}


def encode_wall_layout(grid: List[List[int]]) -> List[float]:
    encoded: List[float] = []
    for row in range(MAP_ROWS):
        for col in range(MAP_COLS):
            value = 1.0 if grid[row][col] == 1 else 0.0
            encoded.append(value)
    return encoded


def build_oracle_obs(state, step_count: int, max_steps: int,
                     include_map_layout: bool = True) -> np.ndarray:
    vec: List[float] = []

    vec.extend([state.car_x / MAP_COLS, state.car_y / MAP_ROWS])

    target_by_id = {target.num_id: target for target in state.targets}

    for idx in range(MAX_BOXES):
        if idx < len(state.boxes):
            box = state.boxes[idx]
            has_match = 1.0 if box.class_id in target_by_id else 0.0
            vec.extend([
                box.x / MAP_COLS,
                box.y / MAP_ROWS,
                (box.class_id + 1) / 10.0,
                1.0,
                has_match,
            ])
        else:
            vec.extend([0.0] * 5)

    for idx in range(MAX_TARGETS):
        if idx < len(state.targets):
            target = state.targets[idx]
            vec.extend([
                target.x / MAP_COLS,
                target.y / MAP_ROWS,
                (target.num_id + 1) / 10.0,
                1.0,
            ])
        else:
            vec.extend([0.0] * 4)

    for idx in range(MAX_BOMBS):
        if idx < len(state.bombs):
            bomb = state.bombs[idx]
            vec.extend([bomb.x / MAP_COLS, bomb.y / MAP_ROWS])
        else:
            vec.extend([0.0, 0.0])

    total = max(state.total_pairs, 1)
    remaining = len(state.boxes)
    vec.extend([
        1.0 if state.total_pairs > 0 else 0.0,
        1.0 if state.total_pairs > 0 else 0.0,
        1.0 - remaining / total,
        step_count / max(max_steps, 1),
    ])

    for idx in range(MAX_BOXES):
        if idx < len(state.boxes):
            box = state.boxes[idx]
            target = target_by_id.get(box.class_id)
            if target is None:
                vec.append(-1.0)
            else:
                dist = (abs(box.x - target.x) + abs(box.y - target.y)) / 26.0
                vec.append(dist)
        else:
            vec.append(-1.0)

    if include_map_layout:
        vec.extend(encode_wall_layout(state.grid))

    expected_dim = STATE_DIM_WITH_MAP if include_map_layout else STATE_DIM
    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape != (expected_dim,):
        raise ValueError(f"unexpected obs shape: {arr.shape}, expected {(expected_dim,)}")
    return arr


def build_solver_from_state(state) -> MultiBoxSolver:
    boxes = [(pos_to_grid(box.x, box.y), box.class_id) for box in state.boxes]
    targets = {target.num_id: pos_to_grid(target.x, target.y) for target in state.targets}
    bombs = [pos_to_grid(bomb.x, bomb.y) for bomb in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    return MultiBoxSolver(state.grid, car, boxes, targets, bombs)


def map_solver_move_to_env_action(state, move: SolverMove) -> Optional[int]:
    entity_type, entity_id, direction, _ = move
    dir_idx = DIR_TO_INDEX.get(direction)
    if dir_idx is None:
        return None

    if entity_type == "box":
        old_pos, class_id = entity_id
        for idx, box in enumerate(state.boxes):
            if pos_to_grid(box.x, box.y) == old_pos and box.class_id == class_id:
                return PUSH_BOX_START + idx * N_DIRS + dir_idx
        return None

    if entity_type == "bomb":
        old_pos = entity_id
        dir_idx = BOMB_DIR_TO_INDEX.get(direction)
        if dir_idx is None:
            return None
        for idx, bomb in enumerate(state.bombs):
            if pos_to_grid(bomb.x, bomb.y) == old_pos:
                return PUSH_BOMB_START + idx * N_BOMB_DIRS + dir_idx
        return None

    return None


def checkpoint_payload(model_state: Dict[str, object], hidden_dim: int,
                       include_map_layout: bool,
                       obs_mode: str = "oracle") -> Dict[str, object]:
    return {
        "model_state": model_state,
        "hidden_dim": hidden_dim,
        "include_map_layout": include_map_layout,
        "obs_mode": obs_mode,
        "obs_dim": STATE_DIM_WITH_MAP if include_map_layout else STATE_DIM,
        "n_actions": N_ACTIONS,
    }
