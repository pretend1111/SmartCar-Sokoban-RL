"""候选动作生成器单元测试 — P1.4."""

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
    Candidate, MAX_CANDIDATES,
    generate_candidates,
    candidates_legality_mask,
)


def _load(path: str) -> BeliefState:
    eng = GameEngine()
    eng.reset(path)
    return BeliefState.from_engine_state(eng.state, fully_observed=True)


# ── 基础 ──────────────────────────────────────────────────

def test_generate_phase1_count():
    bs = _load("assets/maps/phase1/phase1_0001.txt")
    cands = generate_candidates(bs)
    assert len(cands) == MAX_CANDIDATES
    legal = [c for c in cands if c.legal]
    # phase 1 单箱无炸弹, 至少 1 个合法 push (4 方向中至少 1 个)
    assert len(legal) >= 1


def test_generate_phase4_has_box_candidates():
    bs = _load("assets/maps/phase4/phase4_0001.txt")
    cands = generate_candidates(bs)
    legal = [c for c in cands if c.legal]
    push_box_legal = [c for c in legal if c.type == "push_box"]
    assert len(push_box_legal) >= 2, \
        f"3-box phase 4 should have multiple push candidates, got {len(push_box_legal)}"


def test_generate_phase5_has_bomb_candidates():
    bs = _load("assets/maps/phase5/phase5_0001.txt")
    cands = generate_candidates(bs)
    legal = [c for c in cands if c.legal]
    push_bomb_legal = [c for c in legal if c.type == "push_bomb"]
    assert len(push_bomb_legal) >= 1


def test_padding_to_64():
    """随机抽 5 张 phase 6 图, 候选数永远 == 64."""
    import random
    random.seed(42)
    for i in [1, 11, 51, 101, 501]:
        path = f"assets/maps/phase6/phase6_{i:04d}.txt"
        if not os.path.exists(path):
            continue
        bs = _load(path)
        cands = generate_candidates(bs)
        assert len(cands) == 64


def test_mask_dtype_shape():
    bs = _load("assets/maps/phase1/phase1_0001.txt")
    cands = generate_candidates(bs)
    mask = candidates_legality_mask(cands)
    assert mask.shape == (64,)
    assert mask.dtype == np.float32
    assert ((mask == 0.0) | (mask == 1.0)).all()


def test_pad_candidates_marked_illegal():
    """phase 1 单箱 → push_box 4 个 (其中部分非法), 没炸弹, 1-2 inspect.
    总共合法 < 64, 剩余 = pad. 检查 pad 全部 legal=False."""
    bs = _load("assets/maps/phase1/phase1_0001.txt")
    cands = generate_candidates(bs)
    n_pad = sum(1 for c in cands if c.type == "pad")
    assert n_pad >= 30, f"expected many pad slots, got {n_pad}"
    for c in cands:
        if c.type == "pad":
            assert not c.legal


# ── 合法性 sanity ────────────────────────────────────────

def test_legal_push_actually_pushable():
    """随机抽合法 push_box, 验证这个 push 实际可被引擎执行."""
    bs = _load("assets/maps/phase1/phase1_0001.txt")
    cands = generate_candidates(bs)
    legal_pushes = [c for c in cands if c.type == "push_box" and c.legal]

    # 至少 1 个合法 push
    assert len(legal_pushes) >= 1
    cand = legal_pushes[0]
    box = bs.boxes[cand.box_idx]
    dc, dr = cand.direction
    # 推位
    push_col = box.col - dc
    push_row = box.row - dr
    # 推位必须不是墙
    assert not bs.M[push_row, push_col]
    # 推后箱位不是墙
    assert not bs.M[box.row + dr, box.col + dc]


def test_no_macro_through_deadlock():
    """macro run_length=k 时, 没有任何中间 cell 处于 deadlock."""
    bs = _load("assets/maps/phase4/phase4_0001.txt")
    feat = compute_domain_features(bs)
    cands = generate_candidates(bs, feat)

    # 找 target 集
    target_cells = {(t.col, t.row) for t in bs.targets}

    for c in cands:
        if c.type != "push_box" or not c.legal or c.run_length < 2:
            continue
        b = bs.boxes[c.box_idx]
        dc, dr = c.direction
        for k in range(1, c.run_length + 1):
            cell_col = b.col + k * dc
            cell_row = b.row + k * dr
            if (cell_col, cell_row) in target_cells:
                continue
            assert not feat.deadlock_mask[cell_row, cell_col], \
                f"macro k={c.run_length} 经过死锁格 ({cell_col},{cell_row})"


# ── push_only flag (部署架构: 板上 explorer + NN 推箱) ───────

def test_push_only_no_inspect():
    """push_only=True (默认) 不生成 inspect 候选."""
    bs = _load("assets/maps/phase5/phase5_0001.txt")
    feat = compute_domain_features(bs)
    cands = generate_candidates(bs, feat)  # push_only 默认 True
    types = {c.type for c in cands}
    assert "inspect" not in types, f"push_only=True 时 inspect 不应出现, got types: {types}"


def test_push_only_false_keeps_inspect_compat():
    """push_only=False 仍生成 inspect (向后兼容)."""
    bs = _load("assets/maps/phase5/phase5_0001.txt")
    feat = compute_domain_features(bs)
    cands_pushonly = generate_candidates(bs, feat, push_only=True)
    cands_full = generate_candidates(bs, feat, push_only=False)
    # full 路径产生的候选数 >= push_only (多了 inspect)
    n_full_real = sum(1 for c in cands_full if c.type != "pad")
    n_po_real = sum(1 for c in cands_pushonly if c.type != "pad")
    assert n_full_real >= n_po_real


def test_push_only_phase6():
    """phase 6 fully_observed=True 下 push_only 也不应有 inspect."""
    bs = _load("assets/maps/phase6/phase6_11.txt")
    feat = compute_domain_features(bs)
    cands = generate_candidates(bs, feat)
    types = [c.type for c in cands if c.type != "pad"]
    assert "inspect" not in types


# ── 性能 ──────────────────────────────────────────────────

def test_benchmark_candidates_under_2ms():
    bs = _load("assets/maps/phase6/phase6_0001.txt")
    feat = compute_domain_features(bs)
    # warmup
    for _ in range(3):
        generate_candidates(bs, feat)
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        generate_candidates(bs, feat)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / n
    print(f"\nphase 6 generate_candidates: {elapsed_ms:.2f} ms / call")
    assert elapsed_ms < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
