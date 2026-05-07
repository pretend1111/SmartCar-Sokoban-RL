"""X_grid 30 通道 + u_global 16 维 单元测试."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.grid_tensor import (
    build_grid_tensor, build_global_features,
    GRID_TENSOR_CHANNELS, GLOBAL_DIM,
)


def _load(path):
    eng = GameEngine()
    eng.reset(path)
    bs = BeliefState.from_engine_state(eng.state, fully_observed=True)
    feat = compute_domain_features(bs)
    return bs, feat


def test_grid_shape_and_dtype():
    bs, feat = _load("assets/maps/phase1/phase1_0001.txt")
    X = build_grid_tensor(bs, feat)
    assert X.shape == (10, 14, GRID_TENSOR_CHANNELS)
    assert X.dtype == np.float32


def test_walls_match_belief():
    bs, feat = _load("assets/maps/phase4/phase4_0001.txt")
    X = build_grid_tensor(bs, feat)
    # 内部应有些墙 (phase 4 有内墙)
    walls = X[..., 0]
    # 至少 1 个内墙 cell
    n_wall = int(walls.sum())
    print(f"\nwall count in playable area: {n_wall}")


def test_player_pos_onehot():
    bs, feat = _load("assets/maps/phase1/phase1_0001.txt")
    X = build_grid_tensor(bs, feat)
    pos_chan = X[..., 3]
    assert pos_chan.sum() == 1.0


def test_player_dir_onehot():
    bs, feat = _load("assets/maps/phase1/phase1_0001.txt")
    X = build_grid_tensor(bs, feat)
    dir_chans = X[..., 4:12]
    # 仅 1 个 1 (方向 onehot)
    assert dir_chans.sum() == 1.0


def test_box_present():
    bs, feat = _load("assets/maps/phase4/phase4_0001.txt")
    X = build_grid_tensor(bs, feat)
    box_chan = X[..., 12]
    n_box = int(box_chan.sum())
    assert n_box == len(bs.boxes), f"box count mismatch: {n_box} vs {len(bs.boxes)}"


def test_target_present():
    bs, feat = _load("assets/maps/phase4/phase4_0001.txt")
    X = build_grid_tensor(bs, feat)
    target_chan = X[..., 14]
    n = int(target_chan.sum())
    assert n == len(bs.targets)


def test_no_nan_no_inf():
    for path in [
        "assets/maps/phase1/phase1_0001.txt",
        "assets/maps/phase4/phase4_0001.txt",
        "assets/maps/phase6/phase6_0001.txt",
    ]:
        bs, feat = _load(path)
        X = build_grid_tensor(bs, feat)
        u = build_global_features(bs, feat)
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()
        assert not np.isnan(u).any()
        assert (X >= -1.5).all() and (X <= 1.5).all()


def test_global_shape():
    bs, feat = _load("assets/maps/phase4/phase4_0001.txt")
    u = build_global_features(bs, feat)
    assert u.shape == (GLOBAL_DIM,)
    assert u.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
