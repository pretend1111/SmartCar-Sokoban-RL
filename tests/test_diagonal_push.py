"""对角移动 / 斜推回归测试."""

from __future__ import annotations

import math
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.explorer import direction_to_action
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid


def _build_map(boxes=None, bombs=None, walls=None) -> str:
    rows = [["#"] + ["-"] * 14 + ["#"] for _ in range(12)]
    rows[0] = ["#"] * 16
    rows[-1] = ["#"] * 16
    for c, r in boxes or []:
        rows[r][c] = "$"
    for c, r in bombs or []:
        rows[r][c] = "*"
    for c, r in walls or []:
        rows[r][c] = "#"
    return "\n".join("".join(row) for row in rows)


BOX_DIAG_MAP = _build_map(boxes=[(6, 6)])
BOMB_DIAG_OK_MAP = _build_map(bombs=[(6, 6)], walls=[(7, 5)])
BOMB_DIAG_BLOCKED_MAP = _build_map(
    bombs=[(6, 6)],
    walls=[(7, 5), (6, 5)],
)


def _make_engine(map_string: str) -> GameEngine:
    engine = GameEngine(GameConfig(), base_dir)
    engine.reset_from_string(map_string)
    return engine


def _assert_pos(actual, expected, label: str):
    assert math.isclose(actual[0], expected[0], abs_tol=1e-6), (
        f"{label}: x={actual[0]} != {expected[0]}"
    )
    assert math.isclose(actual[1], expected[1], abs_tol=1e-6), (
        f"{label}: y={actual[1]} != {expected[1]}"
    )


def test_solver_absolute_move_ignores_heading():
    engine = _make_engine(BOX_DIAG_MAP)
    engine.state.car_x = 5.5
    engine.state.car_y = 5.5
    engine.state.car_angle = math.pi / 4

    state = engine.discrete_step(direction_to_action(0, -1))
    _assert_pos((state.car_x, state.car_y), (5.5, 4.5), "abs_move/car")
    assert math.isclose(state.car_angle, math.pi / 4, abs_tol=1e-6), (
        "abs_move: 平移不应改变朝向"
    )


def test_engine_rejects_diagonal_box_push():
    engine = _make_engine(BOX_DIAG_MAP)
    engine.state.car_x = 5.5
    engine.state.car_y = 5.5
    engine.state.car_angle = math.pi / 4

    state = engine.discrete_step(0)
    _assert_pos((state.car_x, state.car_y), (5.5, 5.5), "box_push/car")
    _assert_pos((state.boxes[0].x, state.boxes[0].y), (6.5, 6.5), "box_push/box")


def test_engine_allows_special_diagonal_bomb_detonation():
    engine = _make_engine(BOMB_DIAG_OK_MAP)
    engine.state.car_x = 5.5
    engine.state.car_y = 7.5

    state = engine.discrete_step(direction_to_action(1, -1))
    _assert_pos((state.car_x, state.car_y), (6.5, 6.5), "bomb_ok/car")
    assert len(state.bombs) == 0, "bomb_ok: 炸弹应被消耗"
    assert state.grid[5][7] == 0, "bomb_ok: 对角墙未被炸掉"


def test_engine_blocks_diagonal_bomb_when_side_blocked():
    engine = _make_engine(BOMB_DIAG_BLOCKED_MAP)
    engine.state.car_x = 5.5
    engine.state.car_y = 7.5

    state = engine.discrete_step(direction_to_action(1, -1))
    _assert_pos((state.car_x, state.car_y), (5.5, 7.5), "bomb_blocked/car")
    assert len(state.bombs) == 1, "bomb_blocked: 炸弹不应被消耗"
    assert state.grid[5][7] == 1, "bomb_blocked: 墙不应被炸掉"


def test_solver_rejects_diagonal_box_push():
    engine = _make_engine(BOX_DIAG_MAP)
    engine.state.car_x = 5.5
    engine.state.car_y = 5.5
    state = engine.get_state()

    box = state.boxes[0]
    solver = MultiBoxSolver(
        state.grid,
        pos_to_grid(state.car_x, state.car_y),
        [(pos_to_grid(box.x, box.y), box.class_id)],
        {box.class_id: (12, 9)},
        [],
    )

    pushes = list(solver._enum_pushes(solver.initial))
    assert not any(direction == (1, 1) for _, _, direction, _ in pushes), (
        "solver/box: 仍然枚举了非法斜推箱子"
    )


def test_solver_keeps_special_diagonal_bomb_detonation():
    engine = _make_engine(BOMB_DIAG_OK_MAP)
    engine.state.car_x = 5.5
    engine.state.car_y = 7.5
    state = engine.get_state()

    solver = MultiBoxSolver(
        state.grid,
        pos_to_grid(state.car_x, state.car_y),
        [],
        {},
        [pos_to_grid(state.bombs[0].x, state.bombs[0].y)],
    )

    pushes = list(solver._enum_pushes(solver.initial))
    assert any(direction == (1, -1) for _, _, direction, _ in pushes), (
        "solver/bomb: 缺少合法的对角炸弹引爆"
    )


if __name__ == "__main__":
    test_solver_absolute_move_ignores_heading()
    test_engine_rejects_diagonal_box_push()
    test_engine_allows_special_diagonal_bomb_detonation()
    test_engine_blocks_diagonal_bomb_when_side_blocked()
    test_solver_rejects_diagonal_box_push()
    test_solver_keeps_special_diagonal_bomb_detonation()
    print("diagonal push regression: ok")
