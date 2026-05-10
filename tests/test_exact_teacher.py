from __future__ import annotations

import os
import sys

import pytest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.rl.high_level_env import (
    N_BOMB_DIRS,
    PUSH_BOMB_START,
    PUSH_BOX_START,
    SokobanHLEnv,
)
from smartcar_sokoban.solver.explorer import exploration_complete
from smartcar_sokoban.solver.high_level_teacher import (
    advise_exact_high_level,
    map_solver_move_to_high_level_action,
)
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.offline_teacher_cache import (
    build_offline_teacher_cache,
)


def _build_map(boxes=None, targets=None, bombs=None, walls=None) -> str:
    rows = [["#"] + ["-"] * 14 + ["#"] for _ in range(12)]
    rows[0] = ["#"] * 16
    rows[-1] = ["#"] * 16
    for c, r in walls or []:
        rows[r][c] = "#"
    for c, r in boxes or []:
        rows[r][c] = "$"
    for c, r in targets or []:
        rows[r][c] = "."
    for c, r in bombs or []:
        rows[r][c] = "*"
    return "\n".join("".join(row) for row in rows)


def _make_engine(map_string: str) -> GameEngine:
    engine = GameEngine(GameConfig(), base_dir)
    engine.reset_from_string(map_string)
    return engine


def test_exact_teacher_prefers_explore_before_ids_are_known():
    map_string = _build_map(
        boxes=[(10, 2), (12, 2)],
        targets=[(10, 9), (12, 9)],
    )
    engine = _make_engine(map_string)
    state = engine.get_state()

    assert not exploration_complete(state)

    advice = advise_exact_high_level(state, max_cost=100, time_limit=1.0)
    assert advice.primary_action is not None
    assert advice.primary_action < PUSH_BOX_START
    assert advice.source == "exact_explore"


def test_exact_teacher_maps_single_box_plan_to_push_action():
    map_string = _build_map(
        boxes=[(2, 6)],
        targets=[(3, 6)],
    )
    engine = _make_engine(map_string)
    state = engine.get_state()

    assert exploration_complete(state)

    advice = advise_exact_high_level(state, max_cost=50, time_limit=1.0)
    assert advice.primary_action == PUSH_BOX_START + 3
    assert advice.source == "exact_solver"


def test_env_applies_teacher_shaping_reward(tmp_path):
    map_path = tmp_path / "teacher_map.txt"
    map_path.write_text(
        _build_map(
            boxes=[(2, 6)],
            targets=[(3, 6)],
        ),
        encoding="utf-8",
    )

    env = SokobanHLEnv(
        map_file=str(map_path),
        base_dir=str(tmp_path),
        max_steps=5,
        teacher_primary_reward=1.0,
        teacher_candidate_reward=0.0,
        teacher_mismatch_penalty=0.0,
        teacher_time_limit=1.0,
    )
    env.reset(seed=0)

    advice = advise_exact_high_level(env.engine.get_state(), max_cost=50, time_limit=1.0)
    assert advice.primary_action is not None

    _, reward, _, _, info = env.step(advice.primary_action)
    assert reward > 0.0
    assert info["teacher_action"] == advice.primary_action
    assert info["teacher_reward"] == 1.0
    assert info["teacher_source"] == "exact_solver"


def test_diagonal_bomb_move_has_unique_high_level_action():
    map_string = _build_map(bombs=[(6, 6)], walls=[(7, 5)])
    engine = _make_engine(map_string)
    state = engine.get_state()

    action = map_solver_move_to_high_level_action(state, ("bomb", (6, 6), (1, -1), 0))
    assert action == PUSH_BOMB_START + 5


def test_env_mask_allows_diagonal_bomb_push(tmp_path):
    map_path = tmp_path / "bomb_diag.txt"
    map_path.write_text(
        _build_map(bombs=[(6, 6)], walls=[(7, 5)]),
        encoding="utf-8",
    )

    env = SokobanHLEnv(
        map_file=str(map_path),
        base_dir=str(tmp_path),
        max_steps=5,
    )
    env.reset(seed=0)
    mask = env.action_masks()

    diag_action = PUSH_BOMB_START + 5
    assert mask.shape[0] == PUSH_BOMB_START + 3 * N_BOMB_DIRS
    assert bool(mask[diag_action])
    _, _, _, _, info = env.step(diag_action)
    assert info["remaining_bombs"] == 0


def test_solver_auto_respects_small_time_limit(monkeypatch):
    solver = MultiBoxSolver(
        grid=[
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        car_pos=(1, 1),
        boxes=[],
        targets={},
        bombs=[],
    )
    captured = {}

    def fake_best_first(*, max_cost, time_limit, weight=1.5):
        captured["time_limit"] = time_limit
        return None

    monkeypatch.setattr(solver, "_solve_best_first", fake_best_first)

    result = solver.solve(max_cost=10, time_limit=0.2, strategy="auto")

    assert result is None
    assert captured["time_limit"] == pytest.approx(0.2)


def test_env_uses_offline_teacher_cache_without_online_fallback(tmp_path, monkeypatch):
    map_path = tmp_path / "offline_teacher_map.txt"
    map_path.write_text(
        _build_map(
            boxes=[(2, 6)],
            targets=[(3, 6)],
        ),
        encoding="utf-8",
    )
    cache_path = tmp_path / "teacher_cache.pkl.gz"
    build_offline_teacher_cache(
        cache_path=cache_path,
        map_pool=[str(map_path)],
        seed_manifest={},
        max_steps=5,
        base_dir=str(tmp_path),
        phase_name="test",
        seeds_per_map=1,
        max_cost=50,
        time_limit=1.0,
        strategy="auto",
    )

    env = SokobanHLEnv(
        map_file=str(map_path),
        base_dir=str(tmp_path),
        max_steps=5,
        teacher_primary_reward=1.0,
        teacher_candidate_reward=0.0,
        teacher_mismatch_penalty=0.0,
        teacher_time_limit=0.0,
        teacher_offline_cache_path=str(cache_path),
        teacher_online_fallback=False,
    )
    env.reset(seed=7)

    def fail_online_teacher(*args, **kwargs):
        raise AssertionError("online teacher should not be called")

    monkeypatch.setattr(
        "smartcar_sokoban.rl.high_level_env.advise_exact_high_level",
        fail_online_teacher,
    )

    advice = env._get_teacher_advice(env.engine.get_state())
    assert advice is not None
    assert advice.source.endswith("_offline")

    _, reward, _, _, info = env.step(advice.primary_action)
    assert reward > 0.0
    assert info["teacher_source"].endswith("_offline")
    assert info["teacher_reward"] == 1.0
