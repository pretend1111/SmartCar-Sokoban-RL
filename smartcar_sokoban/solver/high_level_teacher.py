from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from smartcar_sokoban.solver.explorer import (
    compute_facing_actions,
    exploration_complete,
    find_observation_point,
    get_all_entity_positions,
    get_entity_obstacles,
    get_scan_targets,
)
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid

Pos = Tuple[int, int]
SolverMove = Tuple[str, object, Tuple[int, int], int]

MAX_BOXES = 5
MAX_TARGETS = 5
MAX_BOMBS = 3

BOX_DIR_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
BOMB_DIR_DELTAS = [
    (0, -1), (0, 1), (-1, 0), (1, 0),
    (-1, -1), (1, -1), (-1, 1), (1, 1),
]
BOX_DIR_TO_INDEX = {delta: idx for idx, delta in enumerate(BOX_DIR_DELTAS)}
BOMB_DIR_TO_INDEX = {delta: idx for idx, delta in enumerate(BOMB_DIR_DELTAS)}
N_DIRS = 4
N_BOMB_DIRS = 8

EXPLORE_BOX_START = 0
EXPLORE_TGT_START = MAX_BOXES
PUSH_BOX_START = MAX_BOXES + MAX_TARGETS
PUSH_BOMB_START = PUSH_BOX_START + MAX_BOXES * N_DIRS
N_ACTIONS = PUSH_BOMB_START + MAX_BOMBS * N_BOMB_DIRS


@dataclass
class TeacherAdvice:
    primary_action: Optional[int]
    candidate_actions: Tuple[int, ...]
    source: str
    estimated_remaining_cost: Optional[int] = None


def build_solver_from_state(state) -> MultiBoxSolver:
    boxes = [(pos_to_grid(box.x, box.y), box.class_id) for box in state.boxes]
    targets = {target.num_id: pos_to_grid(target.x, target.y) for target in state.targets}
    bombs = [pos_to_grid(bomb.x, bomb.y) for bomb in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)
    return MultiBoxSolver(state.grid, car, boxes, targets, bombs)


def map_solver_move_to_high_level_action(state, move: SolverMove) -> Optional[int]:
    etype, entity_id, direction, _ = move

    if etype == "box":
        dir_idx = BOX_DIR_TO_INDEX.get(direction)
        if dir_idx is None:
            return None
        old_pos, class_id = entity_id
        for idx, box in enumerate(state.boxes):
            if pos_to_grid(box.x, box.y) == old_pos and box.class_id == class_id:
                return PUSH_BOX_START + idx * N_DIRS + dir_idx
        return None

    if etype == "bomb":
        dir_idx = BOMB_DIR_TO_INDEX.get(direction)
        if dir_idx is None:
            return None
        old_pos = entity_id
        for idx, bomb in enumerate(state.bombs):
            if pos_to_grid(bomb.x, bomb.y) == old_pos:
                return PUSH_BOMB_START + idx * N_BOMB_DIRS + dir_idx
        return None

    return None


def advise_exact_high_level(state, *, max_cost: int = 300,
                            time_limit: float = 30.0,
                            strategy: str = "auto") -> TeacherAdvice:
    if not exploration_complete(state):
        return _advise_exploration(state)
    return _advise_solver_push(state, max_cost=max_cost,
                               time_limit=time_limit, strategy=strategy)


def _advise_exploration(state) -> TeacherAdvice:
    car_grid = pos_to_grid(state.car_x, state.car_y)
    obstacles = get_entity_obstacles(state)
    entity_positions = get_all_entity_positions(state)

    scored: List[Tuple[int, int]] = []
    for ex, ey, etype, eidx in get_scan_targets(state):
        entity_grid = pos_to_grid(ex, ey)
        result = find_observation_point(
            car_grid,
            entity_grid,
            state.grid,
            obstacles,
            entity_positions,
            current_angle=state.car_angle,
        )
        if result is None:
            continue

        obs_pos, face_angle = result
        path = bfs_path(car_grid, obs_pos, state.grid, obstacles)
        if path is None:
            continue

        face_actions = compute_facing_actions(state.car_angle, face_angle)
        cost = len(path) + len(face_actions)
        if etype == "box":
            action = EXPLORE_BOX_START + eidx
        else:
            action = EXPLORE_TGT_START + eidx
        scored.append((cost, action))

    if not scored:
        return TeacherAdvice(
            primary_action=None,
            candidate_actions=(),
            source="exact_explore_unavailable",
            estimated_remaining_cost=None,
        )

    scored.sort(key=lambda item: (item[0], item[1]))
    best_cost = scored[0][0]
    candidates = tuple(action for cost, action in scored if cost == best_cost)
    return TeacherAdvice(
        primary_action=candidates[0],
        candidate_actions=candidates,
        source="exact_explore",
        estimated_remaining_cost=best_cost,
    )


def _advise_solver_push(state, *, max_cost: int, time_limit: float,
                        strategy: str) -> TeacherAdvice:
    solver = build_solver_from_state(state)
    solution = solver.solve(
        max_cost=max_cost,
        time_limit=time_limit,
        strategy=strategy,
    )
    if not solution:
        return TeacherAdvice(
            primary_action=None,
            candidate_actions=(),
            source="exact_solver_unavailable",
            estimated_remaining_cost=None,
        )

    primary_action = map_solver_move_to_high_level_action(state, solution[0])
    if primary_action is None:
        return TeacherAdvice(
            primary_action=None,
            candidate_actions=(),
            source="exact_solver_unmapped",
            estimated_remaining_cost=sum(walk_cost + 1 for _, _, _, walk_cost in solution),
        )

    remaining_cost = sum(walk_cost + 1 for _, _, _, walk_cost in solution)
    return TeacherAdvice(
        primary_action=primary_action,
        candidate_actions=(primary_action,),
        source="exact_solver",
        estimated_remaining_cost=remaining_cost,
    )
