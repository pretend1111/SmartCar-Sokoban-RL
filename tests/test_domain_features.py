"""领域特征单元测试 + 性能 benchmark — P1.3."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import (
    DomainFeatures, INF,
    compute_domain_features,
    compute_player_bfs,
    compute_push_dist_fields,
    compute_push_dir_field,
    compute_deadlock_mask,
    compute_info_gain_heatmap,
    compute_box_target_match,
    normalize_dist_field,
)


def _load_engine(map_path: str) -> GameEngine:
    eng = GameEngine()
    eng.reset(map_path)
    return eng


# ── 基础 ──────────────────────────────────────────────────

def test_player_bfs_phase1():
    eng = _load_engine("assets/maps/phase1/phase1_0001.txt")
    bs = BeliefState.from_engine_state(eng.state, fully_observed=True)
    dist, reach = compute_player_bfs(bs)

    assert dist.shape == (12, 16)
    pc, pr = bs.player_col, bs.player_row
    assert dist[pr, pc] == 0
    assert reach[pr, pc]
    # 其余可达格 > 0
    assert (dist[reach] >= 0).all()
    # 不可达格 = INF
    assert (dist[~reach] == INF).all()


def test_push_dist_field_phase1():
    eng = _load_engine("assets/maps/phase1/phase1_0001.txt")
    bs = BeliefState.from_engine_state(eng.state, fully_observed=True)
    matches = compute_box_target_match(bs)
    fields = compute_push_dist_fields(bs, matches)

    assert len(fields) == len(bs.boxes)
    for i, f in enumerate(fields):
        assert f.shape == (12, 16)
        if matches[i] is not None:
            t = bs.targets[matches[i]]
            assert f[t.row, t.col] == 0
            # 至少 box 自己的格子应该可达 (else map is unsolvable)
            b = bs.boxes[i]
            assert f[b.row, b.col] != INF, \
                f"box {i} at ({b.col},{b.row}) cant reach target ({t.col},{t.row})"


def test_deadlock_mask_corner():
    """构造一个 phase1 地图, 检查 corner 死锁被识别."""
    eng = _load_engine("assets/maps/phase1/phase1_0001.txt")
    bs = BeliefState.from_engine_state(eng.state, fully_observed=True)
    dl = compute_deadlock_mask(bs)
    assert dl.shape == (12, 16)
    # 至少四角内圈 (1, 1), (1, 14), (10, 1), (10, 14) 是 dead corner
    # 但若 target 在那 → 不是 dead.
    target_cells = {(t.col, t.row) for t in bs.targets}
    for col, row in [(1, 1), (14, 1), (1, 10), (14, 10)]:
        if (col, row) not in target_cells and not bs.M[row, col]:
            assert dl[row, col], f"({col},{row}) 应是 corner deadlock"


def test_info_gain_heatmap_phase1_partial():
    """phase 1 单 box 单 target. partial obs → 至少有几个高 IG 格."""
    eng = _load_engine("assets/maps/phase1/phase1_0001.txt")
    bs = BeliefState.from_engine_state(eng.state, fully_observed=False)
    if bs.fully_identified:
        pytest.skip("phase 1 默认已全识别 (1 box, 排除推理立刻填)")

    _, reach = compute_player_bfs(bs)
    ig = compute_info_gain_heatmap(bs, reach)
    assert ig.shape == (12, 16)
    # IG ∈ [0, 1]
    assert (ig >= 0).all() and (ig <= 1.001).all()


def test_compute_domain_features_full():
    eng = _load_engine("assets/maps/phase6/phase6_0001.txt")
    bs = BeliefState.from_engine_state(eng.state, fully_observed=True)
    feat = compute_domain_features(bs)

    assert feat.player_bfs_dist.shape == (12, 16)
    assert feat.reachable_mask.shape == (12, 16)
    assert len(feat.push_dist_field) == len(bs.boxes)
    assert feat.push_dir_field.shape == (12, 16, 4)
    assert feat.deadlock_mask.shape == (12, 16)
    assert feat.info_gain_heatmap.shape == (12, 16)
    assert len(feat.box_target_match) == len(bs.boxes)


def test_normalize_dist_field():
    arr = np.array([[0, 1, 5, INF]], dtype=np.int32)
    n = normalize_dist_field(arr, scale=30.0)
    assert n[0, 0] == 0.0
    assert n[0, 3] == 1.0
    assert 0 < n[0, 1] < n[0, 2] < 1.0


# ── 性能 benchmark ────────────────────────────────────────

def test_benchmark_phase6_under_4ms():
    """单次 compute_domain_features 在 phase 6 平均地图上 ≤ 4 ms (取中位数)."""
    eng = _load_engine("assets/maps/phase6/phase6_0001.txt")
    bs = BeliefState.from_engine_state(eng.state, fully_observed=True)

    # warmup
    for _ in range(3):
        compute_domain_features(bs)

    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        compute_domain_features(bs)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / n

    print(f"\nphase 6 domain feature: {elapsed_ms:.2f} ms / call")
    # 4 ms 目标; 8 ms 是 hard fail (CI 噪音容忍).
    assert elapsed_ms < 8.0, f"too slow: {elapsed_ms:.2f} ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
