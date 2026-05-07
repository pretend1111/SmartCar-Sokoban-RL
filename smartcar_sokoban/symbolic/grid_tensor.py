"""X_grid 30 通道构造器 — SAGE-PR §4.2.

输出: [10, 14, 30] float32 (PyTorch 输入用 [B, 30, 10, 14]).

通道定义:
    0  wall_current             binary
    1  wall_init                binary (含已炸毁的墙)
    2  reachable                binary
    3  player_pos               binary
    4-11  player_dir_onehot (8) binary one-hot
    12 box_present              binary
    13 box_known_mask           binary (该格有箱且已识别 ID)
    14 target_present           binary
    15 target_known_mask        binary
    16 bomb_present              binary
    17-21 box_id_inferred (5)   float ∈ [0,1] (该 box 到匹配 target 的归一推距离)
    22 player_bfs_dist          float ∈ [0,1] tanh(d/30)
    23 push_dist_field_min      float ∈ [0,1] (所有箱的 min)
    24-27 push_dir_field_NESW   binary (one-hot 流场)
    28 deadlock_mask            binary
    29 info_gain_heatmap        float ∈ [0,1]
"""

from __future__ import annotations

import numpy as np

from smartcar_sokoban.symbolic.belief import (
    BeliefState, GRID_ROWS, GRID_COLS, PLAYABLE_ROWS, PLAYABLE_COLS,
    PLAYABLE_OFFSET,
)
from smartcar_sokoban.symbolic.features import (
    DomainFeatures, INF, normalize_dist_field,
)


GRID_TENSOR_CHANNELS = 30
GLOBAL_DIM = 16


def _crop_playable(arr: np.ndarray) -> np.ndarray:
    """[GRID_ROWS, GRID_COLS, ...] -> [PLAYABLE_ROWS, PLAYABLE_COLS, ...]."""
    return arr[
        PLAYABLE_OFFSET:PLAYABLE_OFFSET + PLAYABLE_ROWS,
        PLAYABLE_OFFSET:PLAYABLE_OFFSET + PLAYABLE_COLS,
    ]


def build_grid_tensor(bs: BeliefState, feat: DomainFeatures) -> np.ndarray:
    """生成 [10, 14, 30] 网络输入张量."""
    out = np.zeros((PLAYABLE_ROWS, PLAYABLE_COLS, GRID_TENSOR_CHANNELS),
                   dtype=np.float32)

    # 0/1: walls
    out[..., 0] = _crop_playable(bs.M.astype(np.float32))
    out[..., 1] = _crop_playable(bs.M_init.astype(np.float32))

    # 2: reachable
    out[..., 2] = _crop_playable(feat.reachable_mask.astype(np.float32))

    # 3: player_pos
    pc, pr = bs.player_col, bs.player_row
    if PLAYABLE_OFFSET <= pr < PLAYABLE_OFFSET + PLAYABLE_ROWS and \
       PLAYABLE_OFFSET <= pc < PLAYABLE_OFFSET + PLAYABLE_COLS:
        out[pr - PLAYABLE_OFFSET, pc - PLAYABLE_OFFSET, 3] = 1.0

    # 4-11: player_dir_onehot
    if 0 <= bs.theta_player < 8:
        if PLAYABLE_OFFSET <= pr < PLAYABLE_OFFSET + PLAYABLE_ROWS and \
           PLAYABLE_OFFSET <= pc < PLAYABLE_OFFSET + PLAYABLE_COLS:
            out[pr - PLAYABLE_OFFSET, pc - PLAYABLE_OFFSET, 4 + bs.theta_player] = 1.0

    # 12: box_present  /  13: box_known_mask
    for b in bs.boxes:
        if not (PLAYABLE_OFFSET <= b.row < PLAYABLE_OFFSET + PLAYABLE_ROWS and
                PLAYABLE_OFFSET <= b.col < PLAYABLE_OFFSET + PLAYABLE_COLS):
            continue
        rr, cc = b.row - PLAYABLE_OFFSET, b.col - PLAYABLE_OFFSET
        out[rr, cc, 12] = 1.0
        if b.class_id is not None:
            out[rr, cc, 13] = 1.0

    # 14: target_present  /  15: target_known_mask
    for t in bs.targets:
        if not (PLAYABLE_OFFSET <= t.row < PLAYABLE_OFFSET + PLAYABLE_ROWS and
                PLAYABLE_OFFSET <= t.col < PLAYABLE_OFFSET + PLAYABLE_COLS):
            continue
        rr, cc = t.row - PLAYABLE_OFFSET, t.col - PLAYABLE_OFFSET
        out[rr, cc, 14] = 1.0
        if t.num_id is not None:
            out[rr, cc, 15] = 1.0

    # 16: bomb_present
    for bm in bs.bombs:
        if not (PLAYABLE_OFFSET <= bm.row < PLAYABLE_OFFSET + PLAYABLE_ROWS and
                PLAYABLE_OFFSET <= bm.col < PLAYABLE_OFFSET + PLAYABLE_COLS):
            continue
        rr, cc = bm.row - PLAYABLE_OFFSET, bm.col - PLAYABLE_OFFSET
        out[rr, cc, 16] = 1.0

    # 17-21: box_id_inferred (5 通道, 每个对应 box_idx 的归一化推送距离)
    for i, b in enumerate(bs.boxes):
        if i >= 5:
            break
        if i >= len(feat.push_dist_field):
            continue
        d_field = feat.push_dist_field[i]
        # 在 box 当前格写入归一化值
        if not (PLAYABLE_OFFSET <= b.row < PLAYABLE_OFFSET + PLAYABLE_ROWS and
                PLAYABLE_OFFSET <= b.col < PLAYABLE_OFFSET + PLAYABLE_COLS):
            continue
        rr, cc = b.row - PLAYABLE_OFFSET, b.col - PLAYABLE_OFFSET
        d = d_field[b.row, b.col]
        if d != INF:
            out[rr, cc, 17 + i] = float(np.tanh(d / 20.0))
        else:
            out[rr, cc, 17 + i] = 1.0

    # 22: player_bfs_dist
    norm_pdist = normalize_dist_field(feat.player_bfs_dist, scale=30.0)
    out[..., 22] = _crop_playable(norm_pdist)

    # 23: push_dist_field_min
    if feat.push_dist_field:
        stack = np.stack(feat.push_dist_field, axis=0)
        agg = stack.min(axis=0)
        out[..., 23] = _crop_playable(normalize_dist_field(agg, scale=20.0))

    # 24-27: push_dir_field_NESW
    pdf = feat.push_dir_field   # [GRID_ROWS, GRID_COLS, 4] (E, W, S, N)
    pdf_play = _crop_playable(pdf.astype(np.float32))
    out[..., 24:28] = pdf_play

    # 28: deadlock_mask
    out[..., 28] = _crop_playable(feat.deadlock_mask.astype(np.float32))

    # 29: info_gain_heatmap
    out[..., 29] = _crop_playable(feat.info_gain_heatmap)

    return out


def build_global_features(bs: BeliefState, feat: DomainFeatures) -> np.ndarray:
    """[16] float32 全局标量."""
    out = np.zeros(GLOBAL_DIM, dtype=np.float32)
    n_box = len(bs.boxes)
    n_target = len(bs.targets)
    out[0] = float(np.tanh(bs.step_count / 50.0))
    out[1] = n_box / 5.0
    out[2] = n_target / 5.0
    out[3] = len(bs.bombs) / 3.0
    out[4] = bs.n_unidentified_boxes / 5.0
    out[5] = bs.n_unidentified_targets / 5.0
    out[6] = 1.0 if bs.fully_identified else 0.0
    # 全图剩余推送距离总和 (粗启发, 看进度)
    total_remain = 0.0
    for i, b in enumerate(bs.boxes):
        if i >= len(feat.push_dist_field):
            continue
        d = feat.push_dist_field[i][b.row, b.col]
        if d != INF:
            total_remain += float(d)
    out[7] = float(np.tanh(total_remain / 30.0))
    # reachable cells 占比
    out[8] = float(feat.reachable_mask.sum()) / max(1.0,
                                                    feat.reachable_mask.size)
    # FOV 已扫描比例
    out[9] = float(bs.visited_fov.sum()) / max(1.0, bs.visited_fov.size)
    # padding 留 6 维
    return out
