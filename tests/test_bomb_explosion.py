"""炸弹爆炸回归测试."""

from __future__ import annotations

import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine


TEST_MAP = "\n".join([
    "################",
    "#--------------#",
    "#--------------#",
    "#--------------#",
    "#-------##-----#",
    "#------*##-----#",
    "#-------##-----#",
    "#--------------#",
    "#--------------#",
    "#--------------#",
    "#--------------#",
    "################",
])


def _make_engine() -> GameEngine:
    engine = GameEngine(GameConfig(), base_dir)
    engine.reset_from_string(TEST_MAP)
    engine.state.car_x = 6.5
    engine.state.car_y = 5.5
    engine.state.car_angle = 0.0
    return engine


def _assert_exploded(state, mode: str):
    assert len(state.bombs) == 0, f"{mode}: 炸弹未被消耗"
    for r in (4, 5, 6):
        for c in (8, 9):
            assert state.grid[r][c] == 0, (
                f"{mode}: 墙体 ({c},{r}) 未被炸掉"
            )


def test_discrete_bomb_explosion():
    engine = _make_engine()
    state = engine.discrete_step(0)
    _assert_exploded(state, "discrete")


def test_continuous_bomb_explosion():
    engine = _make_engine()
    state = engine.step(forward=1.0, strafe=0.0, turn=0.0, dt=0.2)
    _assert_exploded(state, "continuous")


if __name__ == "__main__":
    test_discrete_bomb_explosion()
    test_continuous_bomb_explosion()
    print("bomb explosion regression: ok")
