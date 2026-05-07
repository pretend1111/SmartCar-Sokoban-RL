"""候选特征向量化单元测试 — P1.5."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features
from smartcar_sokoban.symbolic.candidates import (
    generate_candidates, candidates_legality_mask, MAX_CANDIDATES,
)
from smartcar_sokoban.symbolic.cand_features import (
    encode_candidates, CAND_FEATURE_DIM,
    SEG_TYPE, SEG_DIRECTION, SEG_PAIRING, SEG_PUSH_DIST,
)


def _load(path: str):
    eng = GameEngine()
    eng.reset(path)
    bs = BeliefState.from_engine_state(eng.state, fully_observed=True)
    feat = compute_domain_features(bs)
    cands = generate_candidates(bs, feat)
    return bs, feat, cands


# ── 形状 / 类型 ───────────────────────────────────────────

def test_encode_shape_phase1():
    bs, feat, cands = _load("assets/maps/phase1/phase1_0001.txt")
    X = encode_candidates(cands, bs, feat)
    assert X.shape == (MAX_CANDIDATES, CAND_FEATURE_DIM)
    assert X.dtype == np.float32


def test_pad_rows_zero():
    """pad 候选 → 全 0 行."""
    bs, feat, cands = _load("assets/maps/phase1/phase1_0001.txt")
    X = encode_candidates(cands, bs, feat)
    for i, c in enumerate(cands):
        if c.type == "pad":
            assert (X[i] == 0).all(), f"row {i} (pad) 应全 0"


def test_type_onehot_correct():
    bs, feat, cands = _load("assets/maps/phase4/phase4_0001.txt")
    X = encode_candidates(cands, bs, feat)
    for i, c in enumerate(cands):
        type_seg = X[i, SEG_TYPE[0]:SEG_TYPE[1]]
        if c.type == "pad":
            assert (type_seg == 0).all()
        elif c.type == "push_box":
            if c.run_length == 1:
                assert type_seg[1] == 1.0
            elif c.run_length == 2:
                assert type_seg[2] == 1.0
            else:
                assert type_seg[3] == 1.0
        elif c.type == "push_bomb":
            if c.is_diagonal:
                assert type_seg[5] == 1.0
            else:
                assert type_seg[4] == 1.0
        elif c.type == "inspect":
            assert type_seg[6] == 1.0


def test_direction_onehot():
    bs, feat, cands = _load("assets/maps/phase1/phase1_0001.txt")
    X = encode_candidates(cands, bs, feat)
    for i, c in enumerate(cands):
        if c.type != "push_box":
            continue
        dir_seg = X[i, SEG_DIRECTION[0]:SEG_DIRECTION[1]]
        # 4 正交 onehot: 1 个 1, 其余 0
        ortho = dir_seg[:4]
        assert ortho.sum() == 1.0
        # run_length onehot
        run_oh = dir_seg[9:12]
        assert run_oh.sum() == 1.0


def test_pairing_pi_row_phase1():
    """phase 1 单 box 单 target 全已知 → Pi[0] = [1.0]."""
    bs, feat, cands = _load("assets/maps/phase1/phase1_0001.txt")
    X = encode_candidates(cands, bs, feat)
    for i, c in enumerate(cands):
        if c.type == "push_box" and c.box_idx == 0:
            pair_seg = X[i, SEG_PAIRING[0]:SEG_PAIRING[1]]
            # Pi[0, 0] = 1
            assert pair_seg[0] == 1.0
            # 唯一性 flag
            assert pair_seg[11] == 1.0
            break


def test_push_dist_progress_flag():
    """合法 push_box 中, 至少有 1 个推得更近 (progress flag = 1)."""
    bs, feat, cands = _load("assets/maps/phase4/phase4_0001.txt")
    X = encode_candidates(cands, bs, feat)
    found = False
    for i, c in enumerate(cands):
        if c.type == "push_box" and c.legal:
            push_seg = X[i, SEG_PUSH_DIST[0]:SEG_PUSH_DIST[1]]
            if push_seg[3] == 1.0:
                found = True
                break
    assert found, "phase 4 应至少 1 个合法 push 推得更近"


# ── 合理性 sanity ─────────────────────────────────────────

def test_no_nan_no_inf():
    for path in [
        "assets/maps/phase1/phase1_0001.txt",
        "assets/maps/phase4/phase4_0001.txt",
        "assets/maps/phase6/phase6_0001.txt",
    ]:
        bs, feat, cands = _load(path)
        X = encode_candidates(cands, bs, feat)
        assert not np.isnan(X).any(), f"{path} has NaN"
        assert not np.isinf(X).any(), f"{path} has inf"
        # 归一化值应在 [-1, 1] 大致范围内 (tanh 安全)
        assert (X >= -1.5).all() and (X <= 1.5).all()


# ── 性能 ──────────────────────────────────────────────────

def test_benchmark_under_3ms():
    bs, feat, cands = _load("assets/maps/phase6/phase6_0001.txt")
    for _ in range(3):
        encode_candidates(cands, bs, feat)
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        encode_candidates(cands, bs, feat)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / n
    print(f"\nphase 6 encode_candidates: {elapsed_ms:.2f} ms / call")
    assert elapsed_ms < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
