"""高层决策 Gymnasium 环境 v2 — 单步推箱粒度 + 效率优化奖励.

v2 改动:
    1. 动作粒度: 从"整段BFS"降为"推一格" — RL可学交错推箱
    2. 奖励函数: 以 AutoPlayer 步数为基线, 优化效率
    3. 支持动作掩码 (MaskablePPO)

动作空间 (Discrete(42)):
    0..4   EXPLORE_BOX[i]          探索第i个箱子
    5..9   EXPLORE_TGT[i]          探索第i个目标
    10..29 PUSH_BOX[i]_DIR[d]      推箱子i往方向d一格
    30..41 PUSH_BOMB[j]_DIR[d]     推炸弹j往方向d一格

状态向量 (62 维):
    见 _build_state_vector() 的注释.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.paths import PROJECT_ROOT

from smartcar_sokoban.solver.pathfinder import bfs_path, get_reachable, pos_to_grid
from smartcar_sokoban.solver.explorer import (
    find_observation_point, get_entity_obstacles, get_all_entity_positions,
    direction_to_action, compute_facing_actions, exploration_complete,
)

# ── 常量 ──────────────────────────────────────────────────

MAX_BOXES = 5
MAX_TARGETS = 5
MAX_BOMBS = 3

# 4 方向: 上下左右
DIR_UP = 0     # (0, -1)
DIR_DOWN = 1   # (0, +1)
DIR_LEFT = 2   # (-1, 0)
DIR_RIGHT = 3  # (+1, 0)
DIR_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
N_DIRS = 4

# 动作区间
EXPLORE_BOX_START = 0                                          # 0..4
EXPLORE_TGT_START = MAX_BOXES                                  # 5..9
PUSH_BOX_START = MAX_BOXES + MAX_TARGETS                       # 10..29
PUSH_BOMB_START = PUSH_BOX_START + MAX_BOXES * N_DIRS          # 30..41
N_ACTIONS = PUSH_BOMB_START + MAX_BOMBS * N_DIRS               # 42

# 地图尺寸在当前数据集中固定为 16x12.
MAP_COLS = 16
MAP_ROWS = 12
MAP_LAYOUT_DIM = MAP_COLS * MAP_ROWS

# 基础状态维度
# 车(2) + 箱子(5×5) + 目标(5×4) + 炸弹(3×2) + 进度(4) + 距离(5) = 62
STATE_DIM = 2 + MAX_BOXES * 5 + MAX_TARGETS * 4 + MAX_BOMBS * 2 + 4 + MAX_BOXES
STATE_DIM_WITH_MAP = STATE_DIM + MAP_LAYOUT_DIM

REVISIT_PENALTY = 0.75
NO_PROGRESS_PENALTY = 0.5
NO_PROGRESS_END_PENALTY = 10.0
DEADLOCK_PENALTY = 25.0
OSCILLATION_PENALTY = 4.0
OSCILLATION_REVERSE_PENALTY = 4.0
OSCILLATION_END_PENALTY = 12.0
NO_PROGRESS_LIMIT_MIN = 6
NO_PROGRESS_LIMIT_MAX = 20
OSCILLATION_LIMIT = 4
OSCILLATION_LOOKBACK = 6


class SokobanHLEnv(gym.Env):
    """高层决策环境 v2: 单步推箱粒度."""

    metadata = {"render_modes": []}

    def __init__(self,
                 map_file: Optional[str] = None,
                 map_pool: Optional[List[str]] = None,
                 base_dir: str = "",
                 max_steps: int = 60,
                 baseline_steps: Optional[int] = None,
                 seed_manifest: Optional[Dict] = None,
                 include_map_layout: bool = False):
        """
        Args:
            map_file:       单张地图路径
            map_pool:       地图池 (每次 reset 随机选)
            base_dir:       项目根目录
            max_steps:      最大高层步数 (超过则截断)
            baseline_steps: AutoPlayer 基线步数 (用于效率奖励)
            seed_manifest:  种子清单 {"地图文件名": [可用种子列表], ...}
        """
        super().__init__()

        self.base_dir = base_dir or str(PROJECT_ROOT)
        self.map_file = map_file
        self.map_pool = map_pool
        self.max_steps = max_steps
        self.baseline_steps = baseline_steps
        self.seed_manifest = seed_manifest or {}
        self.include_map_layout = include_map_layout
        self.state_dim = (STATE_DIM_WITH_MAP if include_map_layout
                          else STATE_DIM)

        # 引擎
        cfg = GameConfig()
        cfg.control_mode = "discrete"
        self.engine = GameEngine(cfg, self.base_dir)

        # Gym 空间
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.5, shape=(self.state_dim,), dtype=np.float32
        )

        # 内部状态
        self._step_count = 0
        self._total_low_steps = 0
        self._current_map = ""
        self._failed_pushes: Set[Tuple[str, Tuple[int, int], int]] = set()
        self._state_visit_counts: Dict[Tuple[Any, ...], int] = {}
        self._no_progress_streak = 0
        self._oscillation_streak = 0
        self._last_state_revisit_count = 1
        self._last_dead_boxes = 0
        self._last_progress_made = False
        self._last_truncation_reason = ""
        self._last_action: Optional[int] = None
        self._recent_state_sigs: List[Tuple[Any, ...]] = []
        self._target_reachable_cells: Dict[int, Set[Tuple[int, int]]] = {}
        self._cached_deadlock_grid_sig: Optional[Tuple[int, ...]] = None
        self._cached_deadlock_target_sig: Optional[Tuple[Any, ...]] = None

    # ══════════════════════════════════════════════════════
    #  Gym 接口
    # ══════════════════════════════════════════════════════

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 选地图 + 可用种子
        self._current_map = self._pick_map()
        map_seed = self._pick_seed(self._current_map, seed)
        random.seed(map_seed)
        self.engine.reset(self._current_map)

        # 初始 snap
        self.engine.discrete_step(6)
        state = self.engine.get_state()

        self._step_count = 0
        self._total_low_steps = 1
        self._failed_pushes.clear()
        self._state_visit_counts.clear()
        self._no_progress_streak = 0
        self._oscillation_streak = 0
        self._last_state_revisit_count = 1
        self._last_dead_boxes = self._count_dead_boxes(state)
        self._last_progress_made = False
        self._last_truncation_reason = ""
        self._last_action = None
        self._cached_deadlock_grid_sig = None
        self._cached_deadlock_target_sig = None
        self._target_reachable_cells.clear()
        state_sig = self._state_signature(state)
        self._state_visit_counts[state_sig] = 1
        self._recent_state_sigs = [state_sig]
        self._ensure_deadlock_cache(state)

        obs = self._build_state_vector()
        info = self._build_info()
        return obs, info

    def step(self, action: int):
        state_before = self.engine.get_state()
        state_before_sig = self._state_signature(state_before)
        n_boxes_before = len(state_before.boxes)
        n_seen_before = (len(state_before.seen_box_ids) +
                         len(state_before.seen_target_ids))

        # 记录推箱前每个箱子到目标的距离 (用于距离塑形)
        dists_before = self._compute_box_distances(state_before)

        # 执行动作
        steps, success = self._execute_action(action)
        self._total_low_steps += steps
        self._step_count += 1

        state_after = self.engine.get_state()
        n_boxes_after = len(state_after.boxes)
        n_seen_after = (len(state_after.seen_box_ids) +
                        len(state_after.seen_target_ids))

        # 记录推箱后距离
        dists_after = self._compute_box_distances(state_after)

        # ── 奖励计算 ──────────────────────────────────
        reward = 0.0
        distance_delta = 0.0

        # 1. 步数成本: 鼓励用更少低级步完成推箱 (导航开销)
        reward -= steps * 0.02

        # 2. 探索奖励
        newly_seen = n_seen_after - n_seen_before
        reward += newly_seen * 3.0
        if (exploration_complete(state_after) and
                not exploration_complete(state_before)):
            reward += 5.0

        # 3. 距离塑形: 推箱子靠近目标 → 正奖励, 推远 → 负奖励
        #    这是关键! 让每次推箱都有即时反馈
        if dists_before is not None and dists_after is not None:
            # 只比较仍存在的箱子 (被消除的不算)
            # 被消除的箱子: 距离从 d 变为 0, 已经在 total_after 中消失了
            # 所以这里自然包含了消除的正向收益
            # 但我们单独奖励消除, 所以只看未消除的
            common_keys = set(dists_before.keys()) & set(dists_after.keys())
            if common_keys:
                distance_delta = sum(
                    dists_before.get(k, 0) - dists_after.get(k, 0)
                    for k in common_keys
                )
                # delta > 0 → 箱子离目标更近了
                reward += distance_delta * 2.0

        # 4. 推箱配对消除: 大奖
        boxes_eliminated = n_boxes_before - n_boxes_after
        reward += boxes_eliminated * 20.0

        # 5. 通关: 效率奖励 (以 AutoPlayer 为基线)
        if state_after.won:
            base = self.baseline_steps or 100
            efficiency = (base - self._total_low_steps) / max(base, 1)
            reward += 50.0 + efficiency * 100.0

        # 6. 失败推箱惩罚 (BFS导航成功但实体没被推动 = 浪费步数)
        if not success:
            reward -= 2.0

        state_sig = self._state_signature(state_after)
        revisit_count = self._state_visit_counts.get(state_sig, 0) + 1
        self._state_visit_counts[state_sig] = revisit_count
        if revisit_count > 1:
            reward -= min(revisit_count - 1, 4) * REVISIT_PENALTY

        oscillating = self._is_recent_oscillation(state_sig)
        reverse_push = self._is_reverse_box_push(self._last_action, action)
        if oscillating:
            self._oscillation_streak += 1
            reward -= OSCILLATION_PENALTY
            if reverse_push:
                reward -= OSCILLATION_REVERSE_PENALTY
        else:
            self._oscillation_streak = 0

        novel_state = success and state_sig != state_before_sig and revisit_count == 1
        progress_made = (
            boxes_eliminated > 0 or
            newly_seen > 0 or
            (distance_delta > 0.0 and not oscillating) or
            novel_state
        )
        if progress_made:
            self._no_progress_streak = 0
        else:
            self._no_progress_streak += 1
            reward -= min(self._no_progress_streak, 10) * NO_PROGRESS_PENALTY

        dead_boxes = self._count_dead_boxes(state_after)
        if dead_boxes > 0:
            reward -= dead_boxes * DEADLOCK_PENALTY

        # ── 终止条件 ──────────────────────────────────
        terminated = state_after.won
        truncated_reason = ""
        if not terminated:
            if dead_boxes > 0:
                truncated_reason = "dead_box"
            elif self._oscillation_streak >= OSCILLATION_LIMIT:
                truncated_reason = "oscillation"
                reward -= OSCILLATION_END_PENALTY
            elif self._no_progress_streak >= self._no_progress_limit():
                truncated_reason = "no_progress"
                reward -= NO_PROGRESS_END_PENALTY
            elif self._step_count >= self.max_steps:
                truncated_reason = "max_steps"
        truncated = bool(truncated_reason)

        self._last_state_revisit_count = revisit_count
        self._last_dead_boxes = dead_boxes
        self._last_progress_made = progress_made
        self._last_truncation_reason = truncated_reason
        self._last_action = action
        self._recent_state_sigs.append(state_sig)
        if len(self._recent_state_sigs) > OSCILLATION_LOOKBACK:
            self._recent_state_sigs.pop(0)

        obs = self._build_state_vector()
        info = self._build_info()
        info['low_level_steps'] = steps
        info['subtask_success'] = success

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """返回动作掩码."""
        return self._compute_action_mask()

    # ══════════════════════════════════════════════════════
    #  状态向量 (62 维)
    # ══════════════════════════════════════════════════════

    def _build_state_vector(self) -> np.ndarray:
        state = self.engine.get_state()
        vec = []

        # ── 车位置 (2) ──
        vec.extend([state.car_x / MAP_COLS, state.car_y / MAP_ROWS])

        # ── 箱子 (5 × 5 = 25) ──
        # 每个: [x, y, class_id, is_known, has_matching_target]
        for i in range(MAX_BOXES):
            if i < len(state.boxes):
                box = state.boxes[i]
                known = 1.0 if self._box_id_known(state, i) else 0.0
                cid = (box.class_id + 1) / 10.0 if known > 0.5 else 0.0
                has_match = 0.0
                if known > 0.5:
                    tgt = self._find_matching_target(state, box.class_id)
                    has_match = 1.0 if tgt else 0.0
                vec.extend([box.x / MAP_COLS, box.y / MAP_ROWS,
                            cid, known, has_match])
            else:
                vec.extend([0.0] * 5)

        # ── 目标 (5 × 4 = 20) ──
        for i in range(MAX_TARGETS):
            if i < len(state.targets):
                tgt = state.targets[i]
                known = 1.0 if self._target_id_known(state, i) else 0.0
                nid = (tgt.num_id + 1) / 10.0 if known > 0.5 else 0.0
                vec.extend([tgt.x / MAP_COLS, tgt.y / MAP_ROWS, nid, known])
            else:
                vec.extend([0.0] * 4)

        # ── 炸弹 (3 × 2 = 6) ──
        for i in range(MAX_BOMBS):
            if i < len(state.bombs):
                b = state.bombs[i]
                vec.extend([b.x / MAP_COLS, b.y / MAP_ROWS])
            else:
                vec.extend([0.0, 0.0])

        # ── 全局进度 (4) ──
        total = max(state.total_pairs, 1)
        n_remaining = len(state.boxes)
        n_known_b = sum(1 for i in range(len(state.boxes))
                        if self._box_id_known(state, i))
        n_known_t = sum(1 for i in range(len(state.targets))
                        if self._target_id_known(state, i))
        vec.extend([
            n_known_b / total,
            n_known_t / total,
            1.0 - n_remaining / total,
            self._step_count / self.max_steps,
        ])

        # ── 每个箱子到匹配目标的曼哈顿距离 (5) ──
        for i in range(MAX_BOXES):
            if i < len(state.boxes) and self._box_id_known(state, i):
                box = state.boxes[i]
                target = self._find_matching_target(state, box.class_id)
                if target:
                    dist = (abs(box.x - target.x) +
                            abs(box.y - target.y)) / 26.0
                    vec.append(dist)
                else:
                    vec.append(-1.0)
            else:
                vec.append(-1.0)

        if self.include_map_layout:
            vec.extend(self._encode_wall_layout(state.grid))

        return np.array(vec, dtype=np.float32)

    # ══════════════════════════════════════════════════════
    #  动作掩码
    # ══════════════════════════════════════════════════════

    def _compute_action_mask(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=np.bool_)
        state = self.engine.get_state()
        fallback_actions: List[int] = []

        if state.won:
            return mask

        grid = state.grid
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        car_grid = pos_to_grid(state.car_x, state.car_y)

        # 所有实体位置 (用于通行判断)
        entity_set = set()
        for b in state.boxes:
            entity_set.add(pos_to_grid(b.x, b.y))
        for b in state.bombs:
            entity_set.add(pos_to_grid(b.x, b.y))

        # ── 探索箱子 ──
        unseen_box_count = sum(
            1 for i in range(len(state.boxes))
            if not self._box_id_known(state, i))
        if unseen_box_count >= 2:
            for i in range(len(state.boxes)):
                if not self._box_id_known(state, i):
                    mask[EXPLORE_BOX_START + i] = True

        # ── 探索目标 ──
        unseen_tgt_count = sum(
            1 for i in range(len(state.targets))
            if not self._target_id_known(state, i))
        if unseen_tgt_count >= 2:
            for i in range(len(state.targets)):
                if not self._target_id_known(state, i):
                    mask[EXPLORE_TGT_START + i] = True

        # ── 推箱子 (每个方向) ──
        for i in range(len(state.boxes)):
            bc, br = pos_to_grid(state.boxes[i].x, state.boxes[i].y)
            push_obstacles = entity_set
            reachable = get_reachable(car_grid, grid, push_obstacles)
            for d in range(N_DIRS):
                dx, dy = DIR_DELTAS[d]
                # 推的方向: 箱子移到 (bc+dx, br+dy)
                nbx, nby = bc + dx, br + dy
                # 车站位: (bc-dx, br-dy)
                car_pos = (bc - dx, br - dy)

                # 检查: 推后位置合法 + 车站位合法
                if not (0 <= nbx < cols and 0 <= nby < rows):
                    continue
                if not (0 <= car_pos[0] < cols and 0 <= car_pos[1] < rows):
                    continue
                if grid[car_pos[1]][car_pos[0]] == 1:
                    continue
                if car_pos not in reachable:
                    continue
                if self.engine._build_push_chain('box', i, dx, dy) is None:
                    continue
                action_idx = PUSH_BOX_START + i * N_DIRS + d
                fallback_actions.append(action_idx)
                if self._push_signature('box', (bc, br), d) in self._failed_pushes:
                    continue
                if self._would_deadlock_box(
                        state, state.boxes[i].class_id, nbx, nby):
                    continue
                mask[action_idx] = True

        # ── 推炸弹 (每个方向) ──
        for i in range(len(state.bombs)):
            bc, br = pos_to_grid(state.bombs[i].x, state.bombs[i].y)
            push_obstacles = entity_set
            reachable = get_reachable(car_grid, grid, push_obstacles)
            for d in range(N_DIRS):
                dx, dy = DIR_DELTAS[d]
                nbx, nby = bc + dx, br + dy
                car_pos = (bc - dx, br - dy)

                if not (0 <= nbx < cols and 0 <= nby < rows):
                    continue
                if not (0 <= car_pos[0] < cols and 0 <= car_pos[1] < rows):
                    continue
                if grid[car_pos[1]][car_pos[0]] == 1:
                    continue
                if car_pos not in reachable:
                    continue
                if self.engine._build_push_chain('bomb', i, dx, dy) is None:
                    continue
                action_idx = PUSH_BOMB_START + i * N_DIRS + d
                fallback_actions.append(action_idx)
                if self._push_signature('bomb', (bc, br), d) in self._failed_pushes:
                    continue
                mask[action_idx] = True

        # 安全: 至少保留一个动作, 但不要把整组非法推箱重新放回来.
        if not mask.any():
            if fallback_actions:
                mask[fallback_actions[0]] = True
            elif len(state.boxes) > 0:
                mask[PUSH_BOX_START] = True
            elif len(state.bombs) > 0:
                mask[PUSH_BOMB_START] = True
            else:
                mask[0] = True

        return mask

    # ══════════════════════════════════════════════════════
    #  动作执行
    # ══════════════════════════════════════════════════════

    def _execute_action(self, action: int) -> Tuple[int, bool]:
        """执行一个高层动作, 返回 (低级步数, 是否成功)."""

        if EXPLORE_BOX_START <= action < EXPLORE_TGT_START:
            idx = action - EXPLORE_BOX_START
            return self._execute_explore(idx, 'box')

        elif EXPLORE_TGT_START <= action < PUSH_BOX_START:
            idx = action - EXPLORE_TGT_START
            return self._execute_explore(idx, 'target')

        elif PUSH_BOX_START <= action < PUSH_BOMB_START:
            offset = action - PUSH_BOX_START
            box_idx = offset // N_DIRS
            dir_idx = offset % N_DIRS
            return self._execute_single_push(box_idx, dir_idx, 'box')

        elif PUSH_BOMB_START <= action < N_ACTIONS:
            offset = action - PUSH_BOMB_START
            bomb_idx = offset // N_DIRS
            dir_idx = offset % N_DIRS
            return self._execute_single_push(bomb_idx, dir_idx, 'bomb')

        return 0, False

    # ── 探索 ──────────────────────────────────────────────

    def _execute_explore(self, entity_idx: int, etype: str
                         ) -> Tuple[int, bool]:
        state = self.engine.get_state()

        if etype == 'box':
            if entity_idx >= len(state.boxes):
                return 0, False
            ex, ey = state.boxes[entity_idx].x, state.boxes[entity_idx].y
        else:
            if entity_idx >= len(state.targets):
                return 0, False
            ex, ey = state.targets[entity_idx].x, state.targets[entity_idx].y

        entity_grid = pos_to_grid(ex, ey)
        car_grid = pos_to_grid(state.car_x, state.car_y)
        obstacles = get_entity_obstacles(state)
        entity_pos = get_all_entity_positions(state)

        result = find_observation_point(
            car_grid, entity_grid, state.grid, obstacles, entity_pos,
            current_angle=state.car_angle)
        if result is None:
            return 0, False

        obs_pos, face_angle = result
        path = bfs_path(car_grid, obs_pos, state.grid, obstacles)
        if path is None:
            return 0, False

        steps = 0

        # 导航
        for dx, dy in path:
            a = direction_to_action(dx, dy)
            self.engine.discrete_step(a)
            steps += 1

        # 面向实体
        face_acts = compute_facing_actions(
            self.engine.get_state().car_angle, face_angle)
        for a in face_acts:
            self.engine.discrete_step(a)
            steps += 1

        # 检查
        new_state = self.engine.get_state()
        if etype == 'box':
            success = entity_idx in new_state.seen_box_ids
        else:
            success = entity_idx in new_state.seen_target_ids
        return steps, success

    # ── 单步推 ────────────────────────────────────────────

    def _execute_single_push(self, entity_idx: int, dir_idx: int,
                             etype: str) -> Tuple[int, bool]:
        """推一个实体一格.

        流程:
            1. BFS 导航车到实体的反方向 (站位点)
            2. 车往前走一步 → 推动实体
        """
        state = self.engine.get_state()

        # 获取实体位置
        if etype == 'box':
            if entity_idx >= len(state.boxes):
                return 0, False
            ex, ey = state.boxes[entity_idx].x, state.boxes[entity_idx].y
        else:
            if entity_idx >= len(state.bombs):
                return 0, False
            ex, ey = state.bombs[entity_idx].x, state.bombs[entity_idx].y

        ec, er = pos_to_grid(ex, ey)
        dx, dy = DIR_DELTAS[dir_idx]
        push_sig = self._push_signature(etype, (ec, er), dir_idx)

        # 车需要站在实体的反方向
        car_target = (ec - dx, er - dy)
        car_grid = pos_to_grid(state.car_x, state.car_y)

        # 障碍物: 所有实体 (排除目标实体自身, 因为车要站在它旁边)
        obstacles = set()
        for i, b in enumerate(state.boxes):
            obstacles.add(pos_to_grid(b.x, b.y))
        for i, b in enumerate(state.bombs):
            obstacles.add(pos_to_grid(b.x, b.y))

        # BFS 导航到站位点
        steps = 0

        if car_grid != car_target:
            path = bfs_path(car_grid, car_target, state.grid, obstacles)
            if path is None:
                self._failed_pushes.add(push_sig)
                return 0, False

            for pdx, pdy in path:
                a = direction_to_action(pdx, pdy)
                self.engine.discrete_step(a)
                steps += 1

        # 推: 车移动 (dx, dy) → 撞到实体 → 实体被推
        a = direction_to_action(dx, dy)

        # 记录推前实体位置
        old_entity_x, old_entity_y = ex, ey

        new_state = self.engine.discrete_step(a)
        steps += 1

        # 检查是否实际推动了实体 (不是车!)
        success = False
        if etype == 'box':
            if entity_idx < len(new_state.boxes):
                nb = new_state.boxes[entity_idx]
                success = (abs(nb.x - old_entity_x) > 0.01 or
                           abs(nb.y - old_entity_y) > 0.01)
            else:
                # 箱子被消除了 (到达目标), 也算成功
                success = True
        else:
            if entity_idx < len(new_state.bombs):
                nb = new_state.bombs[entity_idx]
                success = (abs(nb.x - old_entity_x) > 0.01 or
                           abs(nb.y - old_entity_y) > 0.01)
            else:
                # 炸弹被引爆了, 也算成功
                success = True

        if success:
            self._failed_pushes.clear()
        else:
            self._failed_pushes.add(push_sig)

        return steps, success

    # ══════════════════════════════════════════════════════
    #  辅助方法
    # ══════════════════════════════════════════════════════

    def _pick_map(self) -> str:
        if self.map_pool:
            return random.choice(self.map_pool)
        return self.map_file or "assets/maps/map1.txt"

    def _push_signature(self, etype: str, entity_pos: Tuple[int, int],
                        dir_idx: int) -> Tuple[str, Tuple[int, int], int]:
        return etype, entity_pos, dir_idx

    def _grid_signature(self, grid) -> Tuple[int, ...]:
        return tuple(cell for row in grid for cell in row)

    def _state_signature(self, state) -> Tuple[Any, ...]:
        car = pos_to_grid(state.car_x, state.car_y)

        boxes = []
        for box in state.boxes:
            bc, br = pos_to_grid(box.x, box.y)
            boxes.append((bc, br, box.class_id))

        bombs = []
        for bomb in state.bombs:
            bc, br = pos_to_grid(bomb.x, bomb.y)
            bombs.append((bc, br))

        targets = []
        for tgt in state.targets:
            tc, tr = pos_to_grid(tgt.x, tgt.y)
            targets.append((tc, tr, tgt.num_id))

        return (
            car,
            tuple(sorted(boxes)),
            tuple(sorted(bombs)),
            tuple(sorted(targets)),
            self._grid_signature(state.grid),
        )

    def _encode_wall_layout(self, grid) -> List[float]:
        feat = []
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        for row in range(MAP_ROWS):
            for col in range(MAP_COLS):
                if row < rows and col < cols:
                    feat.append(1.0 if grid[row][col] == 1 else 0.0)
                else:
                    feat.append(1.0)
        return feat

    def _pick_seed(self, map_path: str, fallback_seed) -> int:
        """为地图选择一个合法种子."""
        # 从路径中提取文件名作为 key
        basename = os.path.basename(map_path)
        if basename in self.seed_manifest:
            seeds = self.seed_manifest[basename]
            if seeds:
                return random.choice(seeds)
        # Phase 1-3 没有种子限制, 用任意种子
        return fallback_seed if fallback_seed is not None else random.randint(0, 9999)

    def _box_id_known(self, state, box_idx: int) -> bool:
        if box_idx in state.seen_box_ids:
            return True
        unseen = sum(1 for i in range(len(state.boxes))
                     if i not in state.seen_box_ids)
        return unseen <= 1

    def _target_id_known(self, state, tgt_idx: int) -> bool:
        if tgt_idx in state.seen_target_ids:
            return True
        unseen = sum(1 for i in range(len(state.targets))
                     if i not in state.seen_target_ids)
        return unseen <= 1

    def _find_matching_target(self, state, class_id: int):
        for tgt in state.targets:
            if tgt.num_id == class_id:
                return tgt
        return None

    def _target_signature(self, state) -> Tuple[Any, ...]:
        targets = []
        for tgt in state.targets:
            tc, tr = pos_to_grid(tgt.x, tgt.y)
            targets.append((tc, tr, tgt.num_id))
        return tuple(sorted(targets))

    def _is_wall(self, grid, col: int, row: int) -> bool:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return True
        return grid[row][col] == 1

    def _is_matching_target_cell(self, state, class_id: int,
                                 col: int, row: int) -> bool:
        target = self._find_matching_target(state, class_id)
        if not target:
            return False
        tc, tr = pos_to_grid(target.x, target.y)
        return (tc, tr) == (col, row)

    def _would_deadlock_box(self, state, class_id: int,
                            col: int, row: int) -> bool:
        if state.bombs:
            return False
        if self._is_matching_target_cell(state, class_id, col, row):
            return False
        self._ensure_deadlock_cache(state)
        target_reachable = self._target_reachable_cells.get(class_id)
        if target_reachable is not None and (col, row) not in target_reachable:
            return True

        wall_up = self._is_wall(state.grid, col, row - 1)
        wall_down = self._is_wall(state.grid, col, row + 1)
        wall_left = self._is_wall(state.grid, col - 1, row)
        wall_right = self._is_wall(state.grid, col + 1, row)
        return ((wall_up and wall_left) or
                (wall_up and wall_right) or
                (wall_down and wall_left) or
                (wall_down and wall_right))

    def _no_progress_limit(self) -> int:
        return max(
            NO_PROGRESS_LIMIT_MIN,
            min(NO_PROGRESS_LIMIT_MAX, self.max_steps // 3),
        )

    def _count_dead_boxes(self, state) -> int:
        if state.bombs:
            return 0

        dead_boxes = 0
        for box in state.boxes:
            bc, br = pos_to_grid(box.x, box.y)
            if self._would_deadlock_box(state, box.class_id, bc, br):
                dead_boxes += 1

        return dead_boxes

    def _compute_box_distances(self, state) -> Optional[Dict[int, float]]:
        """计算每个已知箱子到匹配目标的曼哈顿距离.

        Returns:
            {class_id: manhattan_distance} 或 None (无已知配对)
        """
        dists = {}
        for i, box in enumerate(state.boxes):
            if not self._box_id_known(state, i):
                continue
            target = self._find_matching_target(state, box.class_id)
            if target:
                d = abs(box.x - target.x) + abs(box.y - target.y)
                dists[box.class_id] = d
        return dists if dists else None

    def _ensure_deadlock_cache(self, state) -> None:
        if state.bombs:
            self._target_reachable_cells.clear()
            self._cached_deadlock_grid_sig = None
            self._cached_deadlock_target_sig = None
            return

        grid_sig = self._grid_signature(state.grid)
        target_sig = self._target_signature(state)
        if (grid_sig == self._cached_deadlock_grid_sig and
                target_sig == self._cached_deadlock_target_sig):
            return

        self._cached_deadlock_grid_sig = grid_sig
        self._cached_deadlock_target_sig = target_sig
        self._target_reachable_cells = {}

        for tgt in state.targets:
            tc, tr = pos_to_grid(tgt.x, tgt.y)
            self._target_reachable_cells[tgt.num_id] = (
                self._compute_reverse_push_reachable(state.grid, (tc, tr))
            )

    def _compute_reverse_push_reachable(
            self, grid, target_cell: Tuple[int, int]) -> Set[Tuple[int, int]]:
        reachable = {target_cell}
        frontier = [target_cell]

        while frontier:
            col, row = frontier.pop()
            for dx, dy in DIR_DELTAS:
                prev_col = col - dx
                prev_row = row - dy
                player_col = prev_col - dx
                player_row = prev_row - dy

                if self._is_wall(grid, prev_col, prev_row):
                    continue
                if self._is_wall(grid, player_col, player_row):
                    continue

                prev_cell = (prev_col, prev_row)
                if prev_cell in reachable:
                    continue
                reachable.add(prev_cell)
                frontier.append(prev_cell)

        return reachable

    def _parse_box_push_action(
            self, action: Optional[int]) -> Optional[Tuple[int, int]]:
        if action is None or not (PUSH_BOX_START <= action < PUSH_BOMB_START):
            return None
        offset = action - PUSH_BOX_START
        return offset // N_DIRS, offset % N_DIRS

    def _is_reverse_box_push(self, prev_action: Optional[int],
                             action: int) -> bool:
        prev_push = self._parse_box_push_action(prev_action)
        cur_push = self._parse_box_push_action(action)
        if prev_push is None or cur_push is None:
            return False
        if prev_push[0] != cur_push[0]:
            return False

        prev_dx, prev_dy = DIR_DELTAS[prev_push[1]]
        cur_dx, cur_dy = DIR_DELTAS[cur_push[1]]
        return prev_dx + cur_dx == 0 and prev_dy + cur_dy == 0

    def _is_recent_oscillation(self, state_sig: Tuple[Any, ...]) -> bool:
        if len(self._recent_state_sigs) <= 1:
            return False
        return state_sig in self._recent_state_sigs[:-1]

    def _build_info(self) -> Dict[str, Any]:
        state = self.engine.get_state()
        return {
            'won': state.won,
            'score': state.score,
            'total_pairs': state.total_pairs,
            'remaining_boxes': len(state.boxes),
            'remaining_bombs': len(state.bombs),
            'step_count': self._step_count,
            'total_low_steps': self._total_low_steps,
            'exploration_complete': exploration_complete(state),
            'map': self._current_map,
            'dead_boxes': self._last_dead_boxes,
            'no_progress_streak': self._no_progress_streak,
            'oscillation_streak': self._oscillation_streak,
            'progress_made': self._last_progress_made,
            'state_revisit_count': self._last_state_revisit_count,
            'truncated_reason': self._last_truncation_reason,
        }
