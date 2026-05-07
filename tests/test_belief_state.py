"""BeliefState 单元测试 — P1.1 / P1.2."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smartcar_sokoban.symbolic.belief import (
    BeliefState, BeliefBox, BeliefTarget, BeliefBomb,
    GRID_ROWS, GRID_COLS, PLAYABLE_ROWS, PLAYABLE_COLS,
    angle_rad_to_theta8, infer_remaining_ids,
)
from smartcar_sokoban.engine import GameEngine


# ── 基础几何 ──────────────────────────────────────────────

def test_angle_rad_to_theta8():
    import math
    # engine 弧度 0 = 东 (col+)
    assert angle_rad_to_theta8(0.0) == 0
    # -π/2 = 北 (上, Y 减小) → theta 6 (N)
    assert angle_rad_to_theta8(-math.pi / 2) == 6
    # π/2 = 南 (下) → theta 2
    assert angle_rad_to_theta8(math.pi / 2) == 2
    # π / -π = 西 → theta 4
    assert angle_rad_to_theta8(math.pi) == 4
    assert angle_rad_to_theta8(-math.pi) == 4


# ── ID 排除推理 ───────────────────────────────────────────

def test_infer_remaining_ids_single_box():
    """N=2 箱: 识别 1 个, 第 2 个自动."""
    K_box = {0: 3}
    K_target = {0: 3, 1: 7}
    new_box, new_tgt, changed = infer_remaining_ids(K_box, K_target, 2, 2, {3, 7})
    assert changed
    assert new_box[1] == 7
    assert new_tgt == K_target


def test_infer_remaining_ids_5box_4identified():
    """N=5 箱: 识别 4 个, 第 5 个自动."""
    K_box = {0: 0, 1: 1, 2: 2, 3: 3}
    K_target = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    universe = {0, 1, 2, 3, 4}
    new_box, _, changed = infer_remaining_ids(K_box, K_target, 5, 5, universe)
    assert changed
    assert new_box[4] == 4


def test_infer_remaining_ids_no_change():
    """N=3 箱, 识别 1 个 → 无法推断."""
    K_box = {0: 0}
    K_target = {0: 0}
    new_box, new_tgt, changed = infer_remaining_ids(K_box, K_target, 3, 3, {0, 1, 2})
    assert not changed
    assert new_box == K_box
    assert new_tgt == K_target


def test_infer_remaining_ids_cross_dependency():
    """5 箱 5 目标: 识别 4 box + 4 target → 双方第 5 个都自动."""
    K_box = {0: 0, 1: 1, 2: 2, 3: 3}
    K_target = {0: 0, 1: 1, 2: 2, 3: 3}
    new_box, new_tgt, changed = infer_remaining_ids(K_box, K_target, 5, 5, set(range(5)))
    assert changed
    assert new_box[4] == 4
    assert new_tgt[4] == 4


# ── BeliefState reset / from_engine ──────────────────────

def _load_phase1_engine() -> GameEngine:
    eng = GameEngine()
    eng.reset("assets/maps/phase1/phase1_0001.txt")
    return eng


def test_belief_from_engine_state_partial_obs():
    eng = _load_phase1_engine()
    bs = BeliefState.from_engine_state(eng.state, fully_observed=False)

    assert bs.M.shape == (12, 16)
    assert bs.rows == 12
    assert bs.cols == 16
    assert len(bs.boxes) == len(eng.state.boxes)
    assert len(bs.targets) == len(eng.state.targets)

    # 初始 partial obs: 只有 engine FOV 看见的才有 ID
    seen_box_ids = eng.state.seen_box_ids
    for i, b in enumerate(bs.boxes):
        if i in seen_box_ids:
            assert b.class_id == eng.state.boxes[i].class_id
        else:
            # 可能因 ID 排除推理被填上 → 接受任何值
            pass


def test_belief_from_engine_state_god_mode():
    eng = _load_phase1_engine()
    bs = BeliefState.from_engine_state(eng.state, fully_observed=True)
    for i, b in enumerate(bs.boxes):
        assert b.class_id == eng.state.boxes[i].class_id
    for i, t in enumerate(bs.targets):
        assert t.num_id == eng.state.targets[i].num_id
    assert bs.visited_fov.all()


# ── BeliefState observe_box / target → ID 排除自动 ───────

def test_observe_box_triggers_id_inference():
    """5 箱场景: 识别 4 个 box → 第 5 个 ID 自动确定."""
    bs = BeliefState(
        boxes=[
            BeliefBox(2, 2),
            BeliefBox(3, 2),
            BeliefBox(4, 2),
            BeliefBox(5, 2),
            BeliefBox(6, 2),
        ],
        targets=[
            BeliefTarget(2, 5, num_id=0),
            BeliefTarget(3, 5, num_id=1),
            BeliefTarget(4, 5, num_id=2),
            BeliefTarget(5, 5, num_id=3),
            BeliefTarget(6, 5, num_id=4),
        ],
    )

    bs.observe_box(0, 4)
    bs.observe_box(1, 3)
    bs.observe_box(2, 2)
    bs.observe_box(3, 1)

    # 第 4 个未观测但应被 ID 排除推出
    assert bs.boxes[4].class_id == 0
    # 全部已识别
    assert bs.fully_identified


def test_observe_target_triggers_id_inference():
    """对称场景: 识别 4 target → 第 5 个."""
    bs = BeliefState(
        boxes=[BeliefBox(2, 2, class_id=i) for i in range(5)],
        targets=[BeliefTarget(2 + i, 5) for i in range(5)],
    )
    bs.observe_target(0, 0)
    bs.observe_target(1, 1)
    bs.observe_target(2, 2)
    bs.observe_target(3, 3)
    assert bs.targets[4].num_id == 4


# ── Pi 矩阵 ───────────────────────────────────────────────

def test_pi_matrix_all_unknown():
    """3 箱 3 目标全未知 → Pi 全 1 (兼容)."""
    bs = BeliefState(
        boxes=[BeliefBox(c, 2) for c in range(2, 5)],
        targets=[BeliefTarget(c, 5) for c in range(2, 5)],
    )
    Pi = bs.Pi
    assert Pi.shape == (3, 3)
    assert (Pi == 1.0).all()


def test_pi_matrix_fully_known():
    """全已知 → 严格对角化 (按 ID 配对)."""
    bs = BeliefState(
        boxes=[
            BeliefBox(2, 2, class_id=2),
            BeliefBox(3, 2, class_id=0),
            BeliefBox(4, 2, class_id=1),
        ],
        targets=[
            BeliefTarget(2, 5, num_id=0),
            BeliefTarget(3, 5, num_id=1),
            BeliefTarget(4, 5, num_id=2),
        ],
    )
    Pi = bs.Pi
    # box[0].class=2 ↔ target[2].num=2; box[1].class=0 ↔ target[0].num=0; box[2].class=1 ↔ target[1].num=1
    expected = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float32)
    assert np.array_equal(Pi, expected)


def test_pi_matrix_partial_known_box():
    """box 已知, target 未知 → row 由 ID 兼容性决定."""
    bs = BeliefState(
        boxes=[
            BeliefBox(2, 2, class_id=0),
            BeliefBox(3, 2, class_id=1),
        ],
        targets=[
            BeliefTarget(2, 5, num_id=0),
            BeliefTarget(3, 5),  # 未知
        ],
    )
    Pi = bs.Pi
    # box0.class=0 → target0 (num=0) 配, target1 (未知, used_tgt={0}) → 0 已用 → 不兼容
    assert Pi[0, 0] == 1.0
    assert Pi[0, 1] == 0.0
    # box1.class=1 → target0 (num=0) 不配, target1 (未知, 1 不在 used_tgt) → 兼容
    assert Pi[1, 0] == 0.0
    assert Pi[1, 1] == 1.0


# ── FOV 累积 ──────────────────────────────────────────────

def test_update_fov_accumulates():
    bs = BeliefState()
    assert not bs.visited_fov.any()

    bs.update_fov({(3, 5), (4, 5)})
    assert bs.visited_fov[5, 3]
    assert bs.visited_fov[5, 4]
    assert bs.visited_fov.sum() == 2

    bs.update_fov({(3, 5), (5, 5)})
    assert bs.visited_fov.sum() == 3  # (3,5), (4,5), (5,5)


def test_update_fov_ignores_out_of_bounds():
    bs = BeliefState()
    bs.update_fov({(-1, 0), (100, 100), (3, 5)})
    assert bs.visited_fov.sum() == 1
    assert bs.visited_fov[5, 3]


# ── sync_from_engine_state (推箱后状态变化) ──────────────

def test_sync_from_engine_after_step():
    eng = _load_phase1_engine()
    bs = BeliefState.from_engine_state(eng.state, fully_observed=True)
    init_step = bs.step_count
    bs.sync_from_engine_state(eng.state, fully_observed=True)
    assert bs.step_count == init_step + 1


# ── 投影到 playable 区域 ──────────────────────────────────

def test_to_playable_walls_shape():
    eng = _load_phase1_engine()
    bs = BeliefState.from_engine_state(eng.state)
    walls = bs.to_playable_walls()
    assert walls.shape == (PLAYABLE_ROWS, PLAYABLE_COLS)
    assert walls.dtype == np.uint8


# ── 玩家整数坐标 ──────────────────────────────────────────

def test_player_col_row():
    bs = BeliefState(p_player_col=3.5, p_player_row=5.5)
    assert bs.player_col == 3
    assert bs.player_row == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
