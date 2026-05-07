"""BeliefState — SAGE-PR 符号状态层.

参考 docs/FINAL_ARCH_DESIGN.md §2.

设计原则:
    1. 不修改 engine 物理. BeliefState 只是 engine.GameState 的可观测投影 +
       已识别 ID 集合 + Π 配对矩阵 + 累积 FOV.
    2. 全 16×12 内部存储 (跟 engine 一致). 网络输入用 to_grid_tensor() 裁
       到 14×10 playable 区域.
    3. 几何坐标 (col, row): col ∈ [0, 15], row ∈ [0, 11]. 跟 solver/pathfinder
       一致.
    4. ID 排除推理 (零神经网络) 在 belief 更新里自动跑.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ── 常量 ───────────────────────────────────────────────────

GRID_ROWS = 12
GRID_COLS = 16
PLAYABLE_ROWS = 10
PLAYABLE_COLS = 14
PLAYABLE_OFFSET = 1  # 外圈墙厚度

ANGLE_8_NAMES = ("E", "SE", "S", "SW", "W", "NW", "N", "NE")
# theta_player ∈ [0, 7], 0 = 东 (col+), 顺时针每 45°.
# 与 engine.car_angle (rad, 0=东+, -π/2=北-) 通过 angle_rad_to_theta8 转换.


def angle_rad_to_theta8(angle_rad: float) -> int:
    """engine 弧度 → belief 离散 8 朝向."""
    import math
    # engine: 0=东+, -π/2=北 (上, Y 减小)
    # 离散: 0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE
    # 注意 engine Y 轴向下，所以 SE = 角度落在 (0, π/2) 之间
    norm = (angle_rad + 2 * math.pi) % (2 * math.pi)
    bucket = int(round(norm / (math.pi / 4))) % 8
    return bucket


# ── 实体数据类 ─────────────────────────────────────────────

@dataclass
class BeliefBox:
    """箱子的 belief — 位置已知, ID 可能未知."""
    col: int
    row: int
    class_id: Optional[int] = None    # None = 未识别
    last_seen_step: int = -1


@dataclass
class BeliefTarget:
    """目标的 belief — 位置已知, ID 可能未知."""
    col: int
    row: int
    num_id: Optional[int] = None
    last_seen_step: int = -1


@dataclass
class BeliefBomb:
    """炸弹的 belief — 没 ID 概念."""
    col: int
    row: int
    last_seen_step: int = -1


# ── ID 排除推理 (零参数, 零神经网络) ────────────────────────

def infer_remaining_ids(
    K_box: Dict[int, int],
    K_target: Dict[int, int],
    n_box: int,
    n_target: int,
    id_universe: Optional[Set[int]] = None,
) -> Tuple[Dict[int, int], Dict[int, int], bool]:
    """N-1 个箱子识别后第 N 个唯一确定; 同样对 target.

    Args:
        K_box: 已识别箱子 {idx: class_id}
        K_target: 已识别目标 {idx: num_id}
        n_box: 总箱数
        n_target: 总目标数
        id_universe: 可能的 ID 集合. 默认 = {0..max(n_box, n_target)-1}.
            注: 推箱子赛题里 ID 是从 0 开始连续的, 所以 universe 默认为
            {0, ..., n-1}.

    Returns:
        (新 K_box, 新 K_target, 是否有更新).

    规则:
        - 箱子使用的 ID 集 = box ID 池 (= target ID 池, 因为它们一一配对)
        - 若 n_box-1 个箱子已识别, 第 N 个 = 池 - 已用
        - 同样对 target
        - 反复迭代到不再变化 (其实最多 1 轮就稳定了)
    """
    if id_universe is None:
        id_universe = set(range(max(n_box, n_target)))

    K_box = dict(K_box)
    K_target = dict(K_target)
    changed = False

    for _ in range(2):  # 最多 2 轮就稳定 (box → target 互锁)
        local_change = False

        # 箱子排除
        unassigned_box = [i for i in range(n_box) if i not in K_box]
        if len(unassigned_box) == 1:
            used = set(K_box.values())
            remaining = id_universe - used
            if len(remaining) == 1:
                K_box[unassigned_box[0]] = remaining.pop()
                local_change = True

        # 目标排除
        unassigned_tgt = [i for i in range(n_target) if i not in K_target]
        if len(unassigned_tgt) == 1:
            used = set(K_target.values())
            remaining = id_universe - used
            if len(remaining) == 1:
                K_target[unassigned_tgt[0]] = remaining.pop()
                local_change = True

        if not local_change:
            break
        changed = True

    return K_box, K_target, changed


# ── BeliefState ────────────────────────────────────────────

@dataclass
class BeliefState:
    """SAGE-PR 符号状态层."""

    # ── 几何 ───────────────────────────────────────────────
    rows: int = GRID_ROWS
    cols: int = GRID_COLS

    M: np.ndarray = field(default_factory=lambda: np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8))
    M_init: np.ndarray = field(default_factory=lambda: np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8))

    # ── 玩家 ───────────────────────────────────────────────
    p_player_col: float = 0.0
    p_player_row: float = 0.0
    theta_player: int = 0   # 0..7

    # ── 实体 ───────────────────────────────────────────────
    boxes: List[BeliefBox] = field(default_factory=list)
    targets: List[BeliefTarget] = field(default_factory=list)
    bombs: List[BeliefBomb] = field(default_factory=list)

    # ── 累积 FOV ───────────────────────────────────────────
    visited_fov: np.ndarray = field(default_factory=lambda: np.zeros((GRID_ROWS, GRID_COLS), dtype=bool))

    # ── 步数 ──────────────────────────────────────────────
    step_count: int = 0

    # ── 工厂 ───────────────────────────────────────────────

    @classmethod
    def from_engine_state(cls, engine_state, *, fully_observed: bool = False) -> "BeliefState":
        """从 engine.GameState 创建一个新 BeliefState.

        Args:
            engine_state: smartcar_sokoban.engine.GameState 实例
            fully_observed: True = 把所有 ID 当作已识别 (训练用 god mode);
                            False = 只把 engine.seen_box_ids / seen_target_ids
                            里的 ID 当已识别 (默认, 跟 FOV 同步).
        """
        rows = len(engine_state.grid)
        cols = len(engine_state.grid[0]) if rows else 0
        M = np.array(engine_state.grid, dtype=np.uint8)
        M_init = M.copy()

        boxes = []
        for i, b in enumerate(engine_state.boxes):
            col = int(round(b.x - 0.5))
            row = int(round(b.y - 0.5))
            cid = b.class_id if (fully_observed or i in engine_state.seen_box_ids) else None
            boxes.append(BeliefBox(col=col, row=row, class_id=cid, last_seen_step=0 if cid is not None else -1))

        targets = []
        for i, t in enumerate(engine_state.targets):
            col = int(round(t.x - 0.5))
            row = int(round(t.y - 0.5))
            nid = t.num_id if (fully_observed or i in engine_state.seen_target_ids) else None
            targets.append(BeliefTarget(col=col, row=row, num_id=nid, last_seen_step=0 if nid is not None else -1))

        bombs = []
        for b in engine_state.bombs:
            col = int(round(b.x - 0.5))
            row = int(round(b.y - 0.5))
            bombs.append(BeliefBomb(col=col, row=row, last_seen_step=0))

        bs = cls(
            rows=rows,
            cols=cols,
            M=M,
            M_init=M_init,
            p_player_col=engine_state.car_x,
            p_player_row=engine_state.car_y,
            theta_player=angle_rad_to_theta8(engine_state.car_angle),
            boxes=boxes,
            targets=targets,
            bombs=bombs,
            visited_fov=np.zeros((rows, cols), dtype=bool),
            step_count=0,
        )

        # 若 fully_observed 则把所有 visible
        if fully_observed:
            bs.visited_fov[:] = True

        # 跑一次 ID 排除推理 (init 时 partial 就排除掉)
        bs._infer_ids_inplace()
        return bs

    # ── 更新接口 ───────────────────────────────────────────

    def sync_from_engine_state(self, engine_state, *, fully_observed: bool = False) -> None:
        """每步推理后从 engine 同步 (墙、箱、玩家、炸弹).

        保留 belief 已识别 ID 和累积 FOV (它们是 belief 私有的).
        """
        # 墙
        new_M = np.array(engine_state.grid, dtype=np.uint8)
        self.M = new_M

        # 玩家
        self.p_player_col = engine_state.car_x
        self.p_player_row = engine_state.car_y
        self.theta_player = angle_rad_to_theta8(engine_state.car_angle)
        self.step_count += 1

        # 箱子: 引擎里 boxes/targets 列表配对消除会同步缩短.
        # 用 (col, row) 做 key 来匹配旧 belief (因为 ID 可能未识别)
        old_box_by_pos = {(b.col, b.row): b for b in self.boxes}
        new_boxes: List[BeliefBox] = []
        for i, b in enumerate(engine_state.boxes):
            col = int(round(b.x - 0.5))
            row = int(round(b.y - 0.5))
            old = old_box_by_pos.get((col, row))
            cid = b.class_id if (fully_observed or i in engine_state.seen_box_ids) else (
                old.class_id if old else None
            )
            last_seen = self.step_count if cid is not None else (
                old.last_seen_step if old else -1
            )
            new_boxes.append(BeliefBox(col=col, row=row, class_id=cid, last_seen_step=last_seen))
        self.boxes = new_boxes

        # 目标
        old_tgt_by_pos = {(t.col, t.row): t for t in self.targets}
        new_targets: List[BeliefTarget] = []
        for i, t in enumerate(engine_state.targets):
            col = int(round(t.x - 0.5))
            row = int(round(t.y - 0.5))
            old = old_tgt_by_pos.get((col, row))
            nid = t.num_id if (fully_observed or i in engine_state.seen_target_ids) else (
                old.num_id if old else None
            )
            last_seen = self.step_count if nid is not None else (
                old.last_seen_step if old else -1
            )
            new_targets.append(BeliefTarget(col=col, row=row, num_id=nid, last_seen_step=last_seen))
        self.targets = new_targets

        # 炸弹
        new_bombs: List[BeliefBomb] = []
        for b in engine_state.bombs:
            col = int(round(b.x - 0.5))
            row = int(round(b.y - 0.5))
            new_bombs.append(BeliefBomb(col=col, row=row, last_seen_step=self.step_count))
        self.bombs = new_bombs

        # 跑 ID 排除推理
        self._infer_ids_inplace()

    def observe_box(self, idx: int, class_id: int) -> None:
        """YOLO 识别到箱子 idx 的 class_id."""
        if 0 <= idx < len(self.boxes):
            self.boxes[idx].class_id = class_id
            self.boxes[idx].last_seen_step = self.step_count
            self._infer_ids_inplace()

    def observe_target(self, idx: int, num_id: int) -> None:
        """YOLO 识别到目标 idx 的 num_id."""
        if 0 <= idx < len(self.targets):
            self.targets[idx].num_id = num_id
            self.targets[idx].last_seen_step = self.step_count
            self._infer_ids_inplace()

    def update_fov(self, visible_cells: Set[Tuple[int, int]]) -> None:
        """累积 FOV. visible_cells: {(col, row)}."""
        for col, row in visible_cells:
            if 0 <= row < self.rows and 0 <= col < self.cols:
                self.visited_fov[row, col] = True

    # ── 派生量 ─────────────────────────────────────────────

    @property
    def K_box(self) -> Dict[int, int]:
        """{box_idx: class_id} 已识别的子集."""
        return {i: b.class_id for i, b in enumerate(self.boxes) if b.class_id is not None}

    @property
    def K_target(self) -> Dict[int, int]:
        """{target_idx: num_id} 已识别的子集."""
        return {i: t.num_id for i, t in enumerate(self.targets) if t.num_id is not None}

    @property
    def Pi(self) -> np.ndarray:
        """软配对矩阵 [N_box, N_target].

        Pi[i, j] = 1 if box i 可能配对 target j (兼容当前 K), else 0.
        - 都已识别且 ID 相等 → 1, 否则 0
        - 任一未知 → 兼容性: 用未在 K 里出现的 ID 检查
        """
        n_box = len(self.boxes)
        n_target = len(self.targets)
        Pi = np.zeros((n_box, n_target), dtype=np.float32)
        if n_box == 0 or n_target == 0:
            return Pi

        used_box_ids = set(self.K_box.values())
        used_tgt_ids = set(self.K_target.values())

        for i, b in enumerate(self.boxes):
            for j, t in enumerate(self.targets):
                bi = b.class_id
                tj = t.num_id

                if bi is not None and tj is not None:
                    Pi[i, j] = 1.0 if bi == tj else 0.0
                elif bi is not None:
                    # box 已知, target 未知 → tj 必须 == bi 且 bi 不在 used_tgt_ids
                    Pi[i, j] = 1.0 if bi not in used_tgt_ids else 0.0
                elif tj is not None:
                    Pi[i, j] = 1.0 if tj not in used_box_ids else 0.0
                else:
                    # 都未知 → 兼容
                    Pi[i, j] = 1.0
        return Pi

    @property
    def player_col(self) -> int:
        """整数列坐标 (用于 grid 计算)."""
        return int(round(self.p_player_col - 0.5))

    @property
    def player_row(self) -> int:
        """整数行坐标."""
        return int(round(self.p_player_row - 0.5))

    @property
    def n_unidentified_boxes(self) -> int:
        return sum(1 for b in self.boxes if b.class_id is None)

    @property
    def n_unidentified_targets(self) -> int:
        return sum(1 for t in self.targets if t.num_id is None)

    @property
    def fully_identified(self) -> bool:
        return self.n_unidentified_boxes == 0 and self.n_unidentified_targets == 0

    # ── 网络输入投影 ───────────────────────────────────────

    def to_playable_walls(self) -> np.ndarray:
        """裁掉外圈墙后的 [10, 14] uint8 墙图."""
        return self.M[
            PLAYABLE_OFFSET:PLAYABLE_OFFSET + PLAYABLE_ROWS,
            PLAYABLE_OFFSET:PLAYABLE_OFFSET + PLAYABLE_COLS,
        ].copy()

    def to_playable_walls_init(self) -> np.ndarray:
        return self.M_init[
            PLAYABLE_OFFSET:PLAYABLE_OFFSET + PLAYABLE_ROWS,
            PLAYABLE_OFFSET:PLAYABLE_OFFSET + PLAYABLE_COLS,
        ].copy()

    # ── 内部 ───────────────────────────────────────────────

    def _infer_ids_inplace(self) -> None:
        """跑 ID 排除推理, 把推出的 ID 写回 boxes/targets."""
        n_box = len(self.boxes)
        n_target = len(self.targets)
        if n_box == 0 and n_target == 0:
            return

        K_box_old = self.K_box
        K_target_old = self.K_target

        # ID 池 = {0..n-1} 因为 box 和 target 一一配对, 同样的 ID 池
        n = max(n_box, n_target)
        universe = set(range(n))

        K_box_new, K_target_new, changed = infer_remaining_ids(
            K_box_old, K_target_old, n_box, n_target, universe
        )

        if changed:
            for i, cid in K_box_new.items():
                if self.boxes[i].class_id is None:
                    self.boxes[i].class_id = cid
                    self.boxes[i].last_seen_step = self.step_count
            for j, nid in K_target_new.items():
                if self.targets[j].num_id is None:
                    self.targets[j].num_id = nid
                    self.targets[j].last_seen_step = self.step_count
