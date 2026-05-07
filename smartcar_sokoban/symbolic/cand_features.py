"""候选特征向量化 — SAGE-PR §3.4.

对每个 Candidate 生成 128 维特征向量, 输出 [64, 128] float32.

128 维拆分 (按 FINAL_ARCH_DESIGN §3.4):
    [0:8]    类型 one-hot (8): pad / push_box / push_box_macro2 / push_box_macro3 /
                              push_bomb / push_bomb_diag / inspect / return_garage
    [8:26]   对象描述 (18): box_idx 5-onehot + bomb_idx 3-onehot + 实体 (col,row) 归一化 +
                            ID 已知 flag + ID 5-onehot + last_seen_step 归一化
    [26:38]  方向 / 宏步 (12): 4 正交 onehot + 4 对角 onehot + run_length 归一化 +
                            run_length onehot 3 维
    [38:54]  配对 (16): Pi[i, :] 5 维 + 最可能 target 5-onehot + 匹配熵 + 唯一性 flag +
                       重复 padding 4
    [54:66]  路径代价 (12): 车到推位 BFS / 转向 cost / 推送步数 / 推完后可达性 / 全图归一化
                          + padding
    [66:82]  推送距离场 (16): 推前 push_dist + 推后 push_dist + 差分 (是否更近) +
                            推后到 target 距离 / 起点到 target 距离 / padding
    [82:96]  死锁 / 合法性 (14): 静态死角 / 推链阻塞 / 不可逆 / 推后是否 target /
                              是否 wall_init (可炸毁) / padding
    [96:108] 炸弹特征 (12): 是否炸弹 / 推入墙引爆 / 周围 3x3 wall 数 / wall_init 数 /
                          连通分量增益 (粗) / padding
    [108:118] 信息增益 (10): viewpoint 处 IG / 4 邻接 IG max / unidentified 数 / padding
    [118:128] 局部邻域 + 全局标量 padding (10): 当前 step / n_box / n_target / n_bombs /
                                            unidentified count / padding

注: 此模块设计为可在线推理 (numpy), 性能 < 1 ms.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from smartcar_sokoban.symbolic.belief import (
    BeliefState, GRID_ROWS, GRID_COLS,
)
from smartcar_sokoban.symbolic.features import (
    DomainFeatures, INF, DIRS_4,
)
from smartcar_sokoban.symbolic.candidates import (
    Candidate, MAX_CANDIDATES, MAX_BOXES, MAX_BOMBS,
)


CAND_FEATURE_DIM = 128

# 段索引
SEG_TYPE = (0, 8)
SEG_OBJECT = (8, 26)
SEG_DIRECTION = (26, 38)
SEG_PAIRING = (38, 54)
SEG_PATH = (54, 66)
SEG_PUSH_DIST = (66, 82)
SEG_DEADLOCK = (82, 96)
SEG_BOMB = (96, 108)
SEG_INFO_GAIN = (108, 118)
SEG_GLOBAL = (118, 128)


# ── 工具 ──────────────────────────────────────────────────

def _norm_cell(col: int, row: int) -> tuple:
    """(col, row) 归一化到 [0, 1]."""
    return (col / GRID_COLS, row / GRID_ROWS)


def _safe_dist(d: int, scale: float = 30.0) -> float:
    if d == INF:
        return 1.0
    return float(np.tanh(d / scale))


# ── 单候选 → 128 维 ──────────────────────────────────────

def encode_candidate(cand: Candidate, idx: int,
                     bs: BeliefState,
                     feat: DomainFeatures) -> np.ndarray:
    """编码单个候选."""
    v = np.zeros(CAND_FEATURE_DIM, dtype=np.float32)

    if cand.type == "pad":
        v[7] = 0.0  # 全 0 即可 (mask 里也是 0)
        return v

    n_box = len(bs.boxes)
    n_target = len(bs.targets)

    # ── [0:8] 类型 one-hot ──────────────────────────────
    # 0 pad, 1 push_box (k=1), 2 push_box_macro2, 3 push_box_macro3,
    # 4 push_bomb (orth), 5 push_bomb_diag, 6 inspect, 7 return
    if cand.type == "push_box":
        if cand.run_length == 1:
            v[1] = 1.0
        elif cand.run_length == 2:
            v[2] = 1.0
        else:
            v[3] = 1.0
    elif cand.type == "push_bomb":
        if cand.is_diagonal:
            v[5] = 1.0
        else:
            v[4] = 1.0
    elif cand.type == "inspect":
        v[6] = 1.0
    elif cand.type == "return_garage":
        v[7] = 1.0

    # ── [8:26] 对象描述 (18) ─────────────────────────────
    # box one-hot 5 + bomb one-hot 3 + (col,row) norm 2 + has_id 1 + id 5-onehot 1 +
    # last_seen norm 1 + padding ?
    off = SEG_OBJECT[0]
    if cand.box_idx is not None and cand.box_idx < MAX_BOXES:
        v[off + cand.box_idx] = 1.0
        b = bs.boxes[cand.box_idx]
        col_norm, row_norm = _norm_cell(b.col, b.row)
        v[off + 5 + 0 + 0] = 0.0  # bomb slot 0
        v[off + 8] = col_norm
        v[off + 9] = row_norm
        if b.class_id is not None:
            v[off + 10] = 1.0
            cid = max(0, min(b.class_id, 4))  # 5-bin (0-9 折叠到 5)
            v[off + 11 + cid] = 1.0
        if b.last_seen_step >= 0:
            v[off + 16] = float(np.tanh(b.last_seen_step / 30.0))
        v[off + 17] = 1.0  # 是 box (类型标志)
    elif cand.bomb_idx is not None and cand.bomb_idx < MAX_BOMBS:
        v[off + 5 + cand.bomb_idx] = 1.0
        bm = bs.bombs[cand.bomb_idx]
        col_norm, row_norm = _norm_cell(bm.col, bm.row)
        v[off + 8] = col_norm
        v[off + 9] = row_norm
        v[off + 17] = 0.0  # 不是 box
    elif cand.viewpoint_col is not None:
        col_norm, row_norm = _norm_cell(cand.viewpoint_col, cand.viewpoint_row)
        v[off + 8] = col_norm
        v[off + 9] = row_norm

    # ── [26:38] 方向 / 宏步 (12) ─────────────────────────
    off = SEG_DIRECTION[0]
    DIR_TO_IDX = {(1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3,
                  (1, 1): 4, (1, -1): 5, (-1, 1): 6, (-1, -1): 7}
    if cand.direction is not None:
        d_idx = DIR_TO_IDX.get(cand.direction)
        if d_idx is not None:
            v[off + d_idx] = 1.0
    # run_length 归一化 + onehot
    v[off + 8] = float(cand.run_length) / 3.0
    if cand.run_length == 1:
        v[off + 9] = 1.0
    elif cand.run_length == 2:
        v[off + 10] = 1.0
    elif cand.run_length >= 3:
        v[off + 11] = 1.0

    # ── [38:54] 配对 (16) ────────────────────────────────
    off = SEG_PAIRING[0]
    if cand.box_idx is not None and cand.box_idx < n_box:
        Pi_row = bs.Pi[cand.box_idx]
        for j in range(min(MAX_BOXES, len(Pi_row))):
            v[off + j] = float(Pi_row[j])
        # 最可能 target one-hot (Pi argmax, 但 Pi 全 0 → 全 0)
        if (Pi_row > 0.5).any():
            best_j = int(np.argmax(Pi_row))
            if best_j < MAX_BOXES:
                v[off + 5 + best_j] = 1.0
        # 匹配熵: row sum 大 = 不确定
        match_sum = float(Pi_row.sum())
        v[off + 10] = match_sum / max(1.0, MAX_BOXES)
        # 唯一性 flag: 该 box 配对唯一确定
        if (Pi_row > 0.5).sum() == 1:
            v[off + 11] = 1.0

    # ── [54:66] 路径代价 (12) ─────────────────────────────
    off = SEG_PATH[0]
    if cand.type == "push_box" and cand.box_idx is not None:
        b = bs.boxes[cand.box_idx]
        dc, dr = cand.direction
        push_pos_col = b.col - dc
        push_pos_row = b.row - dr
        if 0 <= push_pos_row < GRID_ROWS and 0 <= push_pos_col < GRID_COLS:
            d = feat.player_bfs_dist[push_pos_row, push_pos_col]
            v[off + 0] = _safe_dist(d, scale=30.0)
        # 推后箱位 → 玩家可达性
        end_col = b.col + cand.run_length * dc
        end_row = b.row + cand.run_length * dr
        if 0 <= end_row < GRID_ROWS and 0 <= end_col < GRID_COLS:
            v[off + 1] = 1.0 if feat.reachable_mask[end_row, end_col] else 0.0
        v[off + 2] = float(cand.run_length) / 3.0
    elif cand.type == "push_bomb" and cand.bomb_idx is not None:
        bm = bs.bombs[cand.bomb_idx]
        dc, dr = cand.direction
        push_pos_col = bm.col - dc
        push_pos_row = bm.row - dr
        if 0 <= push_pos_row < GRID_ROWS and 0 <= push_pos_col < GRID_COLS:
            d = feat.player_bfs_dist[push_pos_row, push_pos_col]
            v[off + 0] = _safe_dist(d, scale=30.0)
    elif cand.type == "inspect" and cand.viewpoint_col is not None:
        d = feat.player_bfs_dist[cand.viewpoint_row, cand.viewpoint_col]
        v[off + 0] = _safe_dist(d, scale=30.0)

    # ── [66:82] 推送距离场 (16) ───────────────────────────
    off = SEG_PUSH_DIST[0]
    if cand.type == "push_box" and cand.box_idx is not None:
        b = bs.boxes[cand.box_idx]
        dc, dr = cand.direction
        if cand.box_idx < len(feat.push_dist_field):
            field = feat.push_dist_field[cand.box_idx]
            d_before = field[b.row, b.col]
            end_col = b.col + cand.run_length * dc
            end_row = b.row + cand.run_length * dr
            if 0 <= end_row < GRID_ROWS and 0 <= end_col < GRID_COLS:
                d_after = field[end_row, end_col]
            else:
                d_after = INF
            v[off + 0] = _safe_dist(d_before, scale=20.0)
            v[off + 1] = _safe_dist(d_after, scale=20.0)
            # 差分 (是否更近)
            if d_before != INF and d_after != INF:
                diff = d_before - d_after
                v[off + 2] = float(np.tanh(diff / 5.0))
                if d_after < d_before:
                    v[off + 3] = 1.0  # progress flag
            elif d_after == INF:
                v[off + 2] = -1.0  # 推到不可达
            # 推完后是否在 target 上
            tgt_idx = feat.box_target_match[cand.box_idx]
            if tgt_idx is not None and tgt_idx < len(bs.targets):
                t = bs.targets[tgt_idx]
                if (end_col, end_row) == (t.col, t.row):
                    v[off + 4] = 1.0  # 完成 pairing

    # ── [82:96] 死锁 / 合法性 (14) ────────────────────────
    off = SEG_DEADLOCK[0]
    if cand.type == "push_box" and cand.box_idx is not None:
        b = bs.boxes[cand.box_idx]
        dc, dr = cand.direction
        end_col = b.col + cand.run_length * dc
        end_row = b.row + cand.run_length * dr
        # 推后是否进入死锁 (合法时不会, 但放进特征作为参考)
        if 0 <= end_row < GRID_ROWS and 0 <= end_col < GRID_COLS:
            v[off + 0] = 1.0 if feat.deadlock_mask[end_row, end_col] else 0.0
        # 推位是否非墙 (兜底)
        push_pos_col = b.col - dc
        push_pos_row = b.row - dr
        if 0 <= push_pos_row < GRID_ROWS and 0 <= push_pos_col < GRID_COLS:
            v[off + 1] = 0.0 if bs.M[push_pos_row, push_pos_col] else 1.0
    # 合法性自身 (即使非法也保留特征, 让网络看见 mask 之外的细节)
    v[off + 13] = 1.0 if cand.legal else 0.0

    # ── [96:108] 炸弹特征 (12) ────────────────────────────
    off = SEG_BOMB[0]
    if cand.type == "push_bomb" and cand.bomb_idx is not None:
        bm = bs.bombs[cand.bomb_idx]
        dc, dr = cand.direction
        bomb_next_col = bm.col + dc
        bomb_next_row = bm.row + dr
        # 推入墙引爆?
        if 0 <= bomb_next_row < GRID_ROWS and 0 <= bomb_next_col < GRID_COLS:
            v[off + 0] = 1.0 if bs.M[bomb_next_row, bomb_next_col] else 0.0
        # 周围 3x3 墙数 (粗略评估爆破收益)
        wall_count = 0
        wall_init_count = 0
        for ddr in [-1, 0, 1]:
            for ddc in [-1, 0, 1]:
                rr = bm.row + ddr
                cc = bm.col + ddc
                if 0 <= rr < GRID_ROWS and 0 <= cc < GRID_COLS:
                    if bs.M[rr, cc]:
                        wall_count += 1
                    if bs.M_init[rr, cc]:
                        wall_init_count += 1
        v[off + 1] = wall_count / 9.0
        v[off + 2] = wall_init_count / 9.0
        # 是否对角推
        v[off + 3] = 1.0 if cand.is_diagonal else 0.0

    # ── [108:118] 信息增益 (10) ───────────────────────────
    off = SEG_INFO_GAIN[0]
    if cand.type == "inspect" and cand.viewpoint_col is not None:
        ig = feat.info_gain_heatmap[cand.viewpoint_row, cand.viewpoint_col]
        v[off + 0] = float(ig)
        v[off + 1] = float(bs.n_unidentified_boxes) / max(1.0, MAX_BOXES)
        v[off + 2] = float(bs.n_unidentified_targets) / max(1.0, MAX_BOXES)
        # 排除推理: 若 unidentified == 1, 该次观察可能直接确定全部 ID
        if bs.n_unidentified_boxes == 1 or bs.n_unidentified_targets == 1:
            v[off + 3] = 1.0

    # ── [118:128] 全局标量 (10) ───────────────────────────
    off = SEG_GLOBAL[0]
    v[off + 0] = float(np.tanh(bs.step_count / 50.0))
    v[off + 1] = n_box / max(1.0, MAX_BOXES)
    v[off + 2] = n_target / max(1.0, MAX_BOXES)
    v[off + 3] = float(len(bs.bombs)) / max(1.0, MAX_BOMBS)
    v[off + 4] = float(bs.n_unidentified_boxes) / max(1.0, MAX_BOXES)
    v[off + 5] = float(bs.n_unidentified_targets) / max(1.0, MAX_BOXES)
    v[off + 6] = 1.0 if bs.fully_identified else 0.0
    v[off + 7] = float(idx) / MAX_CANDIDATES   # candidate slot index (位置编码)

    return v


def encode_candidates(cands: List[Candidate],
                      bs: BeliefState,
                      feat: DomainFeatures) -> np.ndarray:
    """[64, 128] float32."""
    out = np.zeros((MAX_CANDIDATES, CAND_FEATURE_DIM), dtype=np.float32)
    for i, c in enumerate(cands):
        if i >= MAX_CANDIDATES:
            break
        out[i] = encode_candidate(c, i, bs, feat)
    return out
