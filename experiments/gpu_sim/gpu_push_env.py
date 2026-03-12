from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar, Dict, List, Optional, Set, Tuple

import torch

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.rl.high_level_env import (
    DIR_DELTAS,
    EXPLORE_BOX_START,
    EXPLORE_TGT_START,
    MAP_COLS,
    MAP_ROWS,
    MAX_BOMBS,
    MAX_BOXES,
    MAX_TARGETS,
    N_ACTIONS,
    N_DIRS,
    PUSH_BOMB_START,
    PUSH_BOX_START,
    STATE_DIM,
    STATE_DIM_WITH_MAP,
)
from smartcar_sokoban.solver.explorer import (
    ANGLE_UP,
    compute_facing_actions,
    find_observation_point,
    get_all_entity_positions,
    get_entity_obstacles,
    restore_angle_actions,
)
from smartcar_sokoban.solver.pathfinder import bfs_path

GridPos = Tuple[int, int]
DIR_DELTAS_TENSOR = torch.as_tensor(DIR_DELTAS, dtype=torch.long)
BOX_ACTION_ENTITY = torch.arange(MAX_BOXES, dtype=torch.long).repeat_interleave(N_DIRS)
BOX_ACTION_DIR = torch.arange(N_DIRS, dtype=torch.long).repeat(MAX_BOXES)
BOMB_ACTION_ENTITY = torch.arange(MAX_BOMBS, dtype=torch.long).repeat_interleave(N_DIRS)
BOMB_ACTION_DIR = torch.arange(N_DIRS, dtype=torch.long).repeat(MAX_BOMBS)
PUSH_ACTION_IS_BOX = torch.cat(
    [
        torch.ones(MAX_BOXES * N_DIRS, dtype=torch.bool),
        torch.zeros(MAX_BOMBS * N_DIRS, dtype=torch.bool),
    ],
    dim=0,
)
PUSH_ACTION_ENTITY = torch.cat([BOX_ACTION_ENTITY, BOMB_ACTION_ENTITY], dim=0)
PUSH_ACTION_DIR = torch.cat([BOX_ACTION_DIR, BOMB_ACTION_DIR], dim=0)
PUSH_ACTION_IDS = torch.cat(
    [
        torch.arange(PUSH_BOX_START, PUSH_BOMB_START, dtype=torch.long),
        torch.arange(PUSH_BOMB_START, N_ACTIONS, dtype=torch.long),
    ],
    dim=0,
)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class GpuPushStepResult:
    valid: torch.Tensor
    moved: torch.Tensor
    won: torch.Tensor
    low_steps: torch.Tensor
    success: Optional[torch.Tensor] = None


@dataclass
class PushTrace:
    valid: torch.Tensor
    kind: torch.Tensor
    slot: torch.Tensor
    cell_col: torch.Tensor
    cell_row: torch.Tensor
    chain_len: torch.Tensor
    stand_col: torch.Tensor
    stand_row: torch.Tensor
    low_steps: torch.Tensor


@dataclass
class GpuStaticTables:
    dir_deltas: torch.Tensor
    push_is_box: torch.Tensor
    push_entity: torch.Tensor
    push_dir: torch.Tensor
    push_ids: torch.Tensor
    neighbor_linear: torch.Tensor
    neighbor_valid: torch.Tensor
    blast_linear: torch.Tensor
    blast_valid: torch.Tensor


class GpuPushBatchEnv:
    """GPU high-level simulation with push fast-paths and CPU-parity helpers.

    The hot path used by branch search remains vectorized around push actions.
    Explore actions and failed-push bookkeeping are supported for correctness
    parity with ``SokobanHLEnv``.
    """

    _STATIC_TABLE_CACHE: ClassVar[Dict[Tuple[str, int, int], GpuStaticTables]] = {}
    _INITIAL_STATE_CACHE: ClassVar[Dict[Tuple[str, int, str, str, int], Dict[str, torch.Tensor]]] = {}

    @staticmethod
    def _grid_coord(value: float) -> int:
        return int(round(value - 0.5))

    @classmethod
    def _build_request_state(
        cls,
        request: object,
        max_steps: int,
        base_dir: str,
        dev: torch.device,
    ) -> Dict[str, torch.Tensor]:
        import random

        cache_key = (
            str(getattr(request, "map_path")),
            int(getattr(request, "episode_seed")),
            str(base_dir),
            str(dev),
            int(max_steps),
        )
        cached = cls._INITIAL_STATE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        cfg = GameConfig()
        cfg.control_mode = "discrete"
        engine = GameEngine(cfg, base_dir)
        random.seed(int(getattr(request, "episode_seed")))
        engine.reset(str(getattr(request, "map_path")))
        engine.discrete_step(6)
        state_after_snap = engine.get_state()
        fix_actions = restore_angle_actions(state_after_snap.car_angle)
        for action in fix_actions:
            engine.discrete_step(action)
        engine_state = engine.get_state()

        seen_box = torch.zeros((MAX_BOXES,), dtype=torch.bool, device=dev)
        seen_target = torch.zeros((MAX_TARGETS,), dtype=torch.bool, device=dev)
        for idx in engine_state.seen_box_ids:
            if 0 <= idx < MAX_BOXES:
                seen_box[idx] = True
        for idx in engine_state.seen_target_ids:
            if 0 <= idx < MAX_TARGETS:
                seen_target[idx] = True

        row_state = {
            "walls": torch.as_tensor(engine_state.grid, dtype=torch.bool, device=dev),
            "car_pos": torch.tensor([
                cls._grid_coord(float(engine_state.car_x)),
                cls._grid_coord(float(engine_state.car_y)),
            ], dtype=torch.long, device=dev),
            "box_pos": torch.full((MAX_BOXES, 2), -1, dtype=torch.long, device=dev),
            "box_ids": torch.full((MAX_BOXES,), -1, dtype=torch.long, device=dev),
            "box_alive": torch.zeros((MAX_BOXES,), dtype=torch.bool, device=dev),
            "target_pos": torch.full((MAX_TARGETS, 2), -1, dtype=torch.long, device=dev),
            "target_ids": torch.full((MAX_TARGETS,), -1, dtype=torch.long, device=dev),
            "target_alive": torch.zeros((MAX_TARGETS,), dtype=torch.bool, device=dev),
            "bomb_pos": torch.full((MAX_BOMBS, 2), -1, dtype=torch.long, device=dev),
            "bomb_alive": torch.zeros((MAX_BOMBS,), dtype=torch.bool, device=dev),
            "seen_box": seen_box,
            "seen_target": seen_target,
            "failed_box_push": torch.zeros((MAP_ROWS, MAP_COLS, N_DIRS), dtype=torch.bool, device=dev),
            "failed_bomb_push": torch.zeros((MAP_ROWS, MAP_COLS, N_DIRS), dtype=torch.bool, device=dev),
            "total_pairs": torch.tensor(len(engine_state.boxes), dtype=torch.long, device=dev),
            "max_steps": torch.tensor(int(max_steps), dtype=torch.long, device=dev),
            "step_count": torch.tensor(0, dtype=torch.long, device=dev),
            "total_low_steps": torch.tensor(len(fix_actions) + 1, dtype=torch.long, device=dev),
        }

        for slot, box in enumerate(engine_state.boxes):
            row_state["box_pos"][slot, 0] = int(box.x - 0.5)
            row_state["box_pos"][slot, 1] = int(box.y - 0.5)
            row_state["box_ids"][slot] = int(box.class_id)
            row_state["box_alive"][slot] = True

        for slot, target in enumerate(engine_state.targets):
            row_state["target_pos"][slot, 0] = int(target.x - 0.5)
            row_state["target_pos"][slot, 1] = int(target.y - 0.5)
            row_state["target_ids"][slot] = int(target.num_id)
            row_state["target_alive"][slot] = True

        for slot, bomb in enumerate(engine_state.bombs):
            row_state["bomb_pos"][slot, 0] = int(bomb.x - 0.5)
            row_state["bomb_pos"][slot, 1] = int(bomb.y - 0.5)
            row_state["bomb_alive"][slot] = True

        cls._INITIAL_STATE_CACHE[cache_key] = row_state
        return row_state

    def __init__(
        self,
        walls: torch.Tensor,
        car_pos: torch.Tensor,
        box_pos: torch.Tensor,
        box_ids: torch.Tensor,
        box_alive: torch.Tensor,
        target_pos: torch.Tensor,
        target_ids: torch.Tensor,
        target_alive: torch.Tensor,
        bomb_pos: torch.Tensor,
        bomb_alive: torch.Tensor,
        total_pairs: torch.Tensor,
        seen_box: Optional[torch.Tensor] = None,
        seen_target: Optional[torch.Tensor] = None,
        failed_box_push: Optional[torch.Tensor] = None,
        failed_bomb_push: Optional[torch.Tensor] = None,
        max_steps: Optional[torch.Tensor] = None,
        step_count: Optional[torch.Tensor] = None,
        total_low_steps: Optional[torch.Tensor] = None,
    ) -> None:
        self.device = walls.device
        self.walls = walls.bool()
        self.batch_size = int(walls.shape[0])
        self.height = int(walls.shape[1])
        self.width = int(walls.shape[2])
        self.tables = self._get_static_tables()
        self._batch_idx = torch.arange(self.batch_size, device=self.device)
        self._batch_idx_boxes = self._batch_idx.unsqueeze(1).expand(self.batch_size, MAX_BOXES)
        self._batch_idx_bombs = self._batch_idx.unsqueeze(1).expand(self.batch_size, MAX_BOMBS)

        self.car_pos = car_pos.long()

        self.box_pos = box_pos.long()
        self.box_ids = box_ids.long()
        self.box_alive = box_alive.bool()

        self.target_pos = target_pos.long()
        self.target_ids = target_ids.long()
        self.target_alive = target_alive.bool()

        self.bomb_pos = bomb_pos.long()
        self.bomb_alive = bomb_alive.bool()

        self.total_pairs = total_pairs.long()
        self.seen_box = (
            seen_box.bool()
            if seen_box is not None
            else torch.zeros((self.batch_size, MAX_BOXES), dtype=torch.bool, device=self.device)
        )
        self.seen_target = (
            seen_target.bool()
            if seen_target is not None
            else torch.zeros((self.batch_size, MAX_TARGETS), dtype=torch.bool, device=self.device)
        )
        self.failed_box_push = (
            failed_box_push.bool()
            if failed_box_push is not None
            else torch.zeros((self.batch_size, self.height, self.width, N_DIRS), dtype=torch.bool, device=self.device)
        )
        self.failed_bomb_push = (
            failed_bomb_push.bool()
            if failed_bomb_push is not None
            else torch.zeros((self.batch_size, self.height, self.width, N_DIRS), dtype=torch.bool, device=self.device)
        )
        self.max_steps = (
            max_steps.long()
            if max_steps is not None
            else torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        )
        self.step_count = (
            step_count.long()
            if step_count is not None
            else torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        )
        self.total_low_steps = (
            total_low_steps.long()
            if total_low_steps is not None
            else torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        )
        self._state_version = 0
        self._analysis_version = 0
        self._occupancy_grids_version = -1
        self._box_grid_cache: Optional[torch.Tensor] = None
        self._bomb_grid_cache: Optional[torch.Tensor] = None
        self._occ_grid_cache: Optional[torch.Tensor] = None
        self._distance_map_version = -1
        self._distance_map_cache: Optional[torch.Tensor] = None
        self._distance_frontier_buf = torch.zeros_like(self.walls)
        self._distance_expanded_buf = torch.zeros_like(self.walls)
        self._distance_unvisited_buf = torch.zeros_like(self.walls)
        self._corner_mask_version = -1
        self._corner_mask_cache: Optional[torch.Tensor] = None
        self._reverse_push_reachable_version = -1
        self._reverse_push_reachable_cache: Optional[torch.Tensor] = None
        self._matching_tables_version = -1
        self._matched_exists_cache: Optional[torch.Tensor] = None
        self._matched_target_idx_cache: Optional[torch.Tensor] = None
        self._matched_target_pos_cache: Optional[torch.Tensor] = None
        self._box_distance_sums_version = -1
        self._box_distance_sums_cache: Optional[torch.Tensor] = None
        self._state_hash_version = -1
        self._state_hash_cache: Optional[torch.Tensor] = None
        self._push_trace_version = -1
        self._push_trace_cache: Optional[PushTrace] = None

    def _get_static_tables(self) -> GpuStaticTables:
        key = (str(self.device), self.height, self.width)
        cached = self._STATIC_TABLE_CACHE.get(key)
        if cached is not None:
            return cached
        tables = self._build_static_tables(device=self.device, height=self.height, width=self.width)
        self._STATIC_TABLE_CACHE[key] = tables
        return tables

    @staticmethod
    def _build_static_tables(device: torch.device, height: int, width: int) -> GpuStaticTables:
        grid_rows, grid_cols = torch.meshgrid(
            torch.arange(height, dtype=torch.long, device=device),
            torch.arange(width, dtype=torch.long, device=device),
            indexing="ij",
        )
        linear = (grid_rows * width + grid_cols).reshape(-1)

        dir_deltas = DIR_DELTAS_TENSOR.to(device=device)
        push_is_box = PUSH_ACTION_IS_BOX.to(device=device)
        push_entity = PUSH_ACTION_ENTITY.to(device=device)
        push_dir = PUSH_ACTION_DIR.to(device=device)
        push_ids = PUSH_ACTION_IDS.to(device=device)

        neighbor_offsets = torch.tensor(
            [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),           (1, 0),
                (-1, 1),  (0, 1),  (1, 1),
            ],
            dtype=torch.long,
            device=device,
        )
        blast_offsets = torch.tensor(
            [
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),  (0, 0),  (1, 0),
                (-1, 1),  (0, 1),  (1, 1),
            ],
            dtype=torch.long,
            device=device,
        )

        neighbor_cols = grid_cols.reshape(-1, 1) + neighbor_offsets[:, 0]
        neighbor_rows = grid_rows.reshape(-1, 1) + neighbor_offsets[:, 1]
        neighbor_valid = (
            (neighbor_cols >= 0)
            & (neighbor_cols < width)
            & (neighbor_rows >= 0)
            & (neighbor_rows < height)
        )
        neighbor_linear = torch.where(
            neighbor_valid,
            neighbor_rows * width + neighbor_cols,
            torch.zeros_like(neighbor_cols),
        )

        blast_cols = grid_cols.reshape(-1, 1) + blast_offsets[:, 0]
        blast_rows = grid_rows.reshape(-1, 1) + blast_offsets[:, 1]
        blast_valid = (
            (blast_cols > 0)
            & (blast_cols < width - 1)
            & (blast_rows > 0)
            & (blast_rows < height - 1)
        )
        blast_linear = torch.where(
            blast_valid,
            blast_rows * width + blast_cols,
            torch.zeros_like(blast_cols),
        )

        del linear
        return GpuStaticTables(
            dir_deltas=dir_deltas,
            push_is_box=push_is_box,
            push_entity=push_entity,
            push_dir=push_dir,
            push_ids=push_ids,
            neighbor_linear=neighbor_linear,
            neighbor_valid=neighbor_valid,
            blast_linear=blast_linear,
            blast_valid=blast_valid,
        )

    @classmethod
    def from_envs(
        cls,
        envs: List[object],
        device: str | torch.device = "auto",
    ) -> "GpuPushBatchEnv":
        dev = resolve_device(device)
        batch = len(envs)

        walls = torch.zeros((batch, MAP_ROWS, MAP_COLS), dtype=torch.bool, device=dev)
        car_pos = torch.zeros((batch, 2), dtype=torch.long, device=dev)
        box_pos = torch.full((batch, MAX_BOXES, 2), -1, dtype=torch.long, device=dev)
        box_ids = torch.full((batch, MAX_BOXES), -1, dtype=torch.long, device=dev)
        box_alive = torch.zeros((batch, MAX_BOXES), dtype=torch.bool, device=dev)
        target_pos = torch.full((batch, MAX_TARGETS, 2), -1, dtype=torch.long, device=dev)
        target_ids = torch.full((batch, MAX_TARGETS), -1, dtype=torch.long, device=dev)
        target_alive = torch.zeros((batch, MAX_TARGETS), dtype=torch.bool, device=dev)
        bomb_pos = torch.full((batch, MAX_BOMBS, 2), -1, dtype=torch.long, device=dev)
        bomb_alive = torch.zeros((batch, MAX_BOMBS), dtype=torch.bool, device=dev)
        seen_box = torch.zeros((batch, MAX_BOXES), dtype=torch.bool, device=dev)
        seen_target = torch.zeros((batch, MAX_TARGETS), dtype=torch.bool, device=dev)
        failed_box_push = torch.zeros((batch, MAP_ROWS, MAP_COLS, N_DIRS), dtype=torch.bool, device=dev)
        failed_bomb_push = torch.zeros((batch, MAP_ROWS, MAP_COLS, N_DIRS), dtype=torch.bool, device=dev)
        total_pairs = torch.zeros(batch, dtype=torch.long, device=dev)
        max_steps = torch.ones(batch, dtype=torch.long, device=dev)
        step_count = torch.zeros(batch, dtype=torch.long, device=dev)
        total_low_steps = torch.zeros(batch, dtype=torch.long, device=dev)

        for env_idx, env in enumerate(envs):
            state = env.engine.get_state()
            walls[env_idx] = torch.as_tensor(state.grid, dtype=torch.bool, device=dev)
            car_pos[env_idx, 0] = cls._grid_coord(float(state.car_x))
            car_pos[env_idx, 1] = cls._grid_coord(float(state.car_y))
            total_pairs[env_idx] = int(state.total_pairs)
            max_steps[env_idx] = int(getattr(env, "max_steps", 1))
            step_count[env_idx] = int(getattr(env, "_step_count", 0))
            total_low_steps[env_idx] = int(getattr(env, "_total_low_steps", 0))

            for slot, box in enumerate(state.boxes):
                box_pos[env_idx, slot, 0] = int(box.x - 0.5)
                box_pos[env_idx, slot, 1] = int(box.y - 0.5)
                box_ids[env_idx, slot] = int(box.class_id)
                box_alive[env_idx, slot] = True

            for slot, target in enumerate(state.targets):
                target_pos[env_idx, slot, 0] = int(target.x - 0.5)
                target_pos[env_idx, slot, 1] = int(target.y - 0.5)
                target_ids[env_idx, slot] = int(target.num_id)
                target_alive[env_idx, slot] = True

            for slot, bomb in enumerate(state.bombs):
                bomb_pos[env_idx, slot, 0] = int(bomb.x - 0.5)
                bomb_pos[env_idx, slot, 1] = int(bomb.y - 0.5)
                bomb_alive[env_idx, slot] = True

            for idx in state.seen_box_ids:
                if 0 <= idx < MAX_BOXES:
                    seen_box[env_idx, idx] = True
            for idx in state.seen_target_ids:
                if 0 <= idx < MAX_TARGETS:
                    seen_target[env_idx, idx] = True
            for etype, entity_pos, dir_idx in getattr(env, "_failed_pushes", set()):
                col, row = int(entity_pos[0]), int(entity_pos[1])
                if not (0 <= col < MAP_COLS and 0 <= row < MAP_ROWS and 0 <= dir_idx < N_DIRS):
                    continue
                if etype == "box":
                    failed_box_push[env_idx, row, col, dir_idx] = True
                elif etype == "bomb":
                    failed_bomb_push[env_idx, row, col, dir_idx] = True

        return cls(
            walls=walls,
            car_pos=car_pos,
            box_pos=box_pos,
            box_ids=box_ids,
            box_alive=box_alive,
            target_pos=target_pos,
            target_ids=target_ids,
            target_alive=target_alive,
            bomb_pos=bomb_pos,
            bomb_alive=bomb_alive,
            total_pairs=total_pairs,
            seen_box=seen_box,
            seen_target=seen_target,
            failed_box_push=failed_box_push,
            failed_bomb_push=failed_bomb_push,
            max_steps=max_steps,
            step_count=step_count,
            total_low_steps=total_low_steps,
        )

    @classmethod
    def from_map_and_seeds(
        cls,
        map_path: str,
        episode_seeds: List[int],
        max_steps: int,
        base_dir: str = "",
        device: str | torch.device = "auto",
    ) -> "GpuPushBatchEnv":
        requests = [
            SimpleNamespace(map_path=map_path, episode_seed=int(seed))
            for seed in episode_seeds
        ]
        return cls.from_requests(
            requests=requests,
            max_steps=max_steps,
            base_dir=base_dir,
            device=device,
        )

    @classmethod
    def from_requests(
        cls,
        requests: List[object],
        max_steps: int,
        base_dir: str = "",
        device: str | torch.device = "auto",
    ) -> "GpuPushBatchEnv":
        dev = resolve_device(device)
        rows = [
            cls._build_request_state(
                request=request,
                max_steps=max_steps,
                base_dir=base_dir,
                dev=dev,
            )
            for request in requests
        ]

        walls = torch.stack([row["walls"] for row in rows], dim=0)
        car_pos = torch.stack([row["car_pos"] for row in rows], dim=0)
        box_pos = torch.stack([row["box_pos"] for row in rows], dim=0)
        box_ids = torch.stack([row["box_ids"] for row in rows], dim=0)
        box_alive = torch.stack([row["box_alive"] for row in rows], dim=0)
        target_pos = torch.stack([row["target_pos"] for row in rows], dim=0)
        target_ids = torch.stack([row["target_ids"] for row in rows], dim=0)
        target_alive = torch.stack([row["target_alive"] for row in rows], dim=0)
        bomb_pos = torch.stack([row["bomb_pos"] for row in rows], dim=0)
        bomb_alive = torch.stack([row["bomb_alive"] for row in rows], dim=0)
        seen_box = torch.stack([row["seen_box"] for row in rows], dim=0)
        seen_target = torch.stack([row["seen_target"] for row in rows], dim=0)
        failed_box_push = torch.stack([row["failed_box_push"] for row in rows], dim=0)
        failed_bomb_push = torch.stack([row["failed_bomb_push"] for row in rows], dim=0)
        total_pairs = torch.stack([row["total_pairs"] for row in rows], dim=0)
        max_steps_t = torch.stack([row["max_steps"] for row in rows], dim=0)
        step_count = torch.stack([row["step_count"] for row in rows], dim=0)
        total_low_steps = torch.stack([row["total_low_steps"] for row in rows], dim=0)

        return cls(
            walls=walls,
            car_pos=car_pos,
            box_pos=box_pos,
            box_ids=box_ids,
            box_alive=box_alive,
            target_pos=target_pos,
            target_ids=target_ids,
            target_alive=target_alive,
            bomb_pos=bomb_pos,
            bomb_alive=bomb_alive,
            total_pairs=total_pairs,
            seen_box=seen_box,
            seen_target=seen_target,
            failed_box_push=failed_box_push,
            failed_bomb_push=failed_bomb_push,
            max_steps=max_steps_t,
            step_count=step_count,
            total_low_steps=total_low_steps,
        )

    def clone(self) -> "GpuPushBatchEnv":
        return GpuPushBatchEnv(
            walls=self.walls.clone(),
            car_pos=self.car_pos.clone(),
            box_pos=self.box_pos.clone(),
            box_ids=self.box_ids.clone(),
            box_alive=self.box_alive.clone(),
            target_pos=self.target_pos.clone(),
            target_ids=self.target_ids.clone(),
            target_alive=self.target_alive.clone(),
            bomb_pos=self.bomb_pos.clone(),
            bomb_alive=self.bomb_alive.clone(),
            total_pairs=self.total_pairs.clone(),
            seen_box=self.seen_box.clone(),
            seen_target=self.seen_target.clone(),
            failed_box_push=self.failed_box_push.clone(),
            failed_bomb_push=self.failed_bomb_push.clone(),
            max_steps=self.max_steps.clone(),
            step_count=self.step_count.clone(),
            total_low_steps=self.total_low_steps.clone(),
        )

    def _box_known_mask(self) -> torch.Tensor:
        unseen = self.box_alive & ~self.seen_box
        inferred = unseen.sum(dim=1) <= 1
        return self.box_alive & (self.seen_box | inferred.unsqueeze(1))

    def _target_known_mask(self) -> torch.Tensor:
        unseen = self.target_alive & ~self.seen_target
        inferred = unseen.sum(dim=1) <= 1
        return self.target_alive & (self.seen_target | inferred.unsqueeze(1))

    def exploration_complete(self) -> torch.Tensor:
        return (
            (self.box_alive & ~self.seen_box).sum(dim=1) <= 1
        ) & (
            (self.target_alive & ~self.seen_target).sum(dim=1) <= 1
        )

    @staticmethod
    def _grid_to_world(col: int, row: int) -> Tuple[float, float]:
        return col + 0.5, row + 0.5

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _apply_turn(angle: float, action: int) -> float:
        if action == 4:
            return GpuPushBatchEnv._normalize_angle(angle - math.pi / 4.0)
        if action == 5:
            return GpuPushBatchEnv._normalize_angle(angle + math.pi / 4.0)
        return angle

    @staticmethod
    def _ray_hits_wall(grid: List[List[bool]], x0: float, y0: float, x1: float, y1: float) -> bool:
        dx = x1 - x0
        dy = y1 - y0
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            return False
        steps = int(dist * 4) + 1
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        for step in range(1, steps):
            t = step / steps
            px = x0 + dx * t
            py = y0 + dy * t
            col = int(px)
            row = int(py)
            if 0 <= row < rows and 0 <= col < cols and grid[row][col]:
                return True
        return False

    @staticmethod
    def _object_blocks_ray(
        x0: float,
        y0: float,
        ray_dx: float,
        ray_dy: float,
        target_dist_sq: float,
        obj_x: float,
        obj_y: float,
    ) -> bool:
        ox = obj_x - x0
        oy = obj_y - y0
        ray_len_sq = ray_dx * ray_dx + ray_dy * ray_dy
        if ray_len_sq < 1e-3:
            return False
        t = (ox * ray_dx + oy * ray_dy) / ray_len_sq
        if t < 0.05 or t > 0.95:
            return False
        obj_dist_sq = ox * ox + oy * oy
        if obj_dist_sq >= target_dist_sq * 0.95:
            return False
        closest_x = x0 + ray_dx * t
        closest_y = y0 + ray_dy * t
        return abs(closest_x - obj_x) < 0.5 and abs(closest_y - obj_y) < 0.5

    @classmethod
    def _ray_blocked(
        cls,
        grid: List[List[bool]],
        boxes: List[Tuple[float, float]],
        bombs: List[Tuple[float, float]],
        targets: List[Tuple[float, float]],
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        exclude_x: float = -999.0,
        exclude_y: float = -999.0,
    ) -> bool:
        if cls._ray_hits_wall(grid, x0, y0, x1, y1):
            return True
        ray_dx = x1 - x0
        ray_dy = y1 - y0
        target_dist_sq = ray_dx * ray_dx + ray_dy * ray_dy
        for obj_x, obj_y in boxes + bombs + targets:
            if abs(obj_x - exclude_x) < 0.01 and abs(obj_y - exclude_y) < 0.01:
                continue
            if cls._object_blocks_ray(x0, y0, ray_dx, ray_dy, target_dist_sq, obj_x, obj_y):
                return True
        return False

    @classmethod
    def _is_in_fov(
        cls,
        grid: List[List[bool]],
        boxes: List[Tuple[float, float]],
        bombs: List[Tuple[float, float]],
        targets: List[Tuple[float, float]],
        car_x: float,
        car_y: float,
        car_angle: float,
        tx: float,
        ty: float,
        half_fov: float,
        exclude_x: float,
        exclude_y: float,
    ) -> bool:
        dx = tx - car_x
        dy = ty - car_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            return True
        angle_to_target = math.atan2(dy, dx)
        angle_diff = cls._normalize_angle(angle_to_target - car_angle)
        if abs(angle_diff) > half_fov:
            return False
        return not cls._ray_blocked(
            grid,
            boxes,
            bombs,
            targets,
            car_x,
            car_y,
            tx,
            ty,
            exclude_x,
            exclude_y,
        )

    def _row_entity_snapshot(
        self,
        row: int,
    ) -> Tuple[List[List[bool]], List[Tuple[int, int, int]], List[Tuple[int, int, int]], List[Tuple[int, int]]]:
        grid = self.walls[row].detach().cpu().tolist()
        boxes: List[Tuple[int, int, int]] = []
        targets: List[Tuple[int, int, int]] = []
        bombs: List[Tuple[int, int]] = []
        for slot in range(MAX_BOXES):
            if bool(self.box_alive[row, slot].item()):
                boxes.append((
                    int(self.box_pos[row, slot, 0].item()),
                    int(self.box_pos[row, slot, 1].item()),
                    int(self.box_ids[row, slot].item()),
                ))
        for slot in range(MAX_TARGETS):
            if bool(self.target_alive[row, slot].item()):
                targets.append((
                    int(self.target_pos[row, slot, 0].item()),
                    int(self.target_pos[row, slot, 1].item()),
                    int(self.target_ids[row, slot].item()),
                ))
        for slot in range(MAX_BOMBS):
            if bool(self.bomb_alive[row, slot].item()):
                bombs.append((
                    int(self.bomb_pos[row, slot, 0].item()),
                    int(self.bomb_pos[row, slot, 1].item()),
                ))
        return grid, boxes, targets, bombs

    def _update_visibility_sets(
        self,
        row: int,
        car_col: int,
        car_row: int,
        car_angle: float,
        seen_box: Set[int],
        seen_target: Set[int],
        grid: List[List[bool]],
        boxes: List[Tuple[int, int, int]],
        targets: List[Tuple[int, int, int]],
        bombs: List[Tuple[int, int]],
    ) -> None:
        half_fov = math.radians(GameConfig().fov) / 2.0
        car_x, car_y = self._grid_to_world(car_col, car_row)
        box_centers = [self._grid_to_world(col, row) for col, row, _ in boxes]
        target_centers = [self._grid_to_world(col, row) for col, row, _ in targets]
        bomb_centers = [self._grid_to_world(col, row) for col, row in bombs]
        offsets = ((0.0, 0.0), (0.4, 0.0), (-0.4, 0.0), (0.0, 0.4), (0.0, -0.4))

        for idx, (box_col, box_row, _) in enumerate(boxes):
            if idx in seen_box:
                continue
            box_x, box_y = self._grid_to_world(box_col, box_row)
            for ox, oy in offsets:
                if self._is_in_fov(
                    grid,
                    box_centers,
                    bomb_centers,
                    target_centers,
                    car_x,
                    car_y,
                    car_angle,
                    box_x + ox,
                    box_y + oy,
                    half_fov,
                    box_x,
                    box_y,
                ):
                    seen_box.add(idx)
                    break

        for idx, (target_col, target_row, _) in enumerate(targets):
            if idx in seen_target:
                continue
            target_x, target_y = self._grid_to_world(target_col, target_row)
            for ox, oy in offsets:
                if self._is_in_fov(
                    grid,
                    box_centers,
                    bomb_centers,
                    target_centers,
                    car_x,
                    car_y,
                    car_angle,
                    target_x + ox,
                    target_y + oy,
                    half_fov,
                    target_x,
                    target_y,
                ):
                    seen_target.add(idx)
                    break

    def _simulate_explore_row(
        self,
        row: int,
        action: int,
    ) -> Tuple[int, bool, bool, torch.Tensor, torch.Tensor, torch.Tensor]:
        grid, boxes, targets, bombs = self._row_entity_snapshot(row)
        car_col = int(self.car_pos[row, 0].item())
        car_row = int(self.car_pos[row, 1].item())
        angle = ANGLE_UP

        if EXPLORE_BOX_START <= action < EXPLORE_TGT_START:
            entity_idx = action - EXPLORE_BOX_START
            if entity_idx >= len(boxes):
                return 0, False, False, self.car_pos[row].clone(), self.seen_box[row].clone(), self.seen_target[row].clone()
            entity_grid = (boxes[entity_idx][0], boxes[entity_idx][1])
            etype = "box"
        elif EXPLORE_TGT_START <= action < PUSH_BOX_START:
            entity_idx = action - EXPLORE_TGT_START
            if entity_idx >= len(targets):
                return 0, False, False, self.car_pos[row].clone(), self.seen_box[row].clone(), self.seen_target[row].clone()
            entity_grid = (targets[entity_idx][0], targets[entity_idx][1])
            etype = "target"
        else:
            return 0, False, False, self.car_pos[row].clone(), self.seen_box[row].clone(), self.seen_target[row].clone()

        state = SimpleNamespace(
            car_x=car_col + 0.5,
            car_y=car_row + 0.5,
            grid=grid,
            boxes=[SimpleNamespace(x=col + 0.5, y=row_ + 0.5, class_id=cid) for col, row_, cid in boxes],
            targets=[SimpleNamespace(x=col + 0.5, y=row_ + 0.5, num_id=nid) for col, row_, nid in targets],
            bombs=[SimpleNamespace(x=col + 0.5, y=row_ + 0.5) for col, row_ in bombs],
        )
        obstacles = get_entity_obstacles(state)
        entity_positions = get_all_entity_positions(state)
        result = find_observation_point((car_col, car_row), entity_grid, grid, obstacles, entity_positions)
        if result is None:
            return 0, False, False, self.car_pos[row].clone(), self.seen_box[row].clone(), self.seen_target[row].clone()
        obs_pos, face_angle = result
        path = bfs_path((car_col, car_row), obs_pos, grid, obstacles)
        if path is None:
            return 0, False, False, self.car_pos[row].clone(), self.seen_box[row].clone(), self.seen_target[row].clone()

        seen_box = {
            idx for idx in range(MAX_BOXES)
            if bool(self.seen_box[row, idx].item())
        }
        seen_target = {
            idx for idx in range(MAX_TARGETS)
            if bool(self.seen_target[row, idx].item())
        }
        steps = 0
        moved = False

        for dx, dy in path:
            car_col += int(dx)
            car_row += int(dy)
            moved = True
            steps += 1
            self._update_visibility_sets(
                row,
                car_col,
                car_row,
                angle,
                seen_box,
                seen_target,
                grid,
                boxes,
                targets,
                bombs,
            )

        for turn in compute_facing_actions(ANGLE_UP, face_angle):
            angle = self._apply_turn(angle, turn)
            steps += 1
            self._update_visibility_sets(
                row,
                car_col,
                car_row,
                angle,
                seen_box,
                seen_target,
                grid,
                boxes,
                targets,
                bombs,
            )

        for turn in restore_angle_actions(face_angle):
            angle = self._apply_turn(angle, turn)
            steps += 1
            self._update_visibility_sets(
                row,
                car_col,
                car_row,
                angle,
                seen_box,
                seen_target,
                grid,
                boxes,
                targets,
                bombs,
            )

        new_car_pos = torch.tensor([car_col, car_row], dtype=torch.long, device=self.device)
        new_seen_box = torch.zeros((MAX_BOXES,), dtype=torch.bool, device=self.device)
        new_seen_target = torch.zeros((MAX_TARGETS,), dtype=torch.bool, device=self.device)
        for idx in seen_box:
            if 0 <= idx < MAX_BOXES:
                new_seen_box[idx] = True
        for idx in seen_target:
            if 0 <= idx < MAX_TARGETS:
                new_seen_target[idx] = True

        success = entity_idx in (seen_box if etype == "box" else seen_target)
        return steps, success, moved, new_car_pos, new_seen_box, new_seen_target

    def _refresh_visibility_row(self, row: int) -> None:
        grid, boxes, targets, bombs = self._row_entity_snapshot(row)
        seen_box = {
            idx for idx in range(MAX_BOXES)
            if bool(self.seen_box[row, idx].item())
        }
        seen_target = {
            idx for idx in range(MAX_TARGETS)
            if bool(self.seen_target[row, idx].item())
        }
        self._update_visibility_sets(
            row=row,
            car_col=int(self.car_pos[row, 0].item()),
            car_row=int(self.car_pos[row, 1].item()),
            car_angle=ANGLE_UP,
            seen_box=seen_box,
            seen_target=seen_target,
            grid=grid,
            boxes=boxes,
            targets=targets,
            bombs=bombs,
        )
        self.seen_box[row] = False
        self.seen_target[row] = False
        for idx in seen_box:
            if 0 <= idx < MAX_BOXES:
                self.seen_box[row, idx] = True
        for idx in seen_target:
            if 0 <= idx < MAX_TARGETS:
                self.seen_target[row, idx] = True

    def _row_failed_push_set(self, row: int) -> Set[Tuple[str, Tuple[int, int], int]]:
        failed: Set[Tuple[str, Tuple[int, int], int]] = set()
        box_rows, box_cols, box_dirs = torch.nonzero(self.failed_box_push[row], as_tuple=True)
        for grid_row, col, dir_idx in zip(box_rows.tolist(), box_cols.tolist(), box_dirs.tolist()):
            failed.add(("box", (int(col), int(grid_row)), int(dir_idx)))
        bomb_rows, bomb_cols, bomb_dirs = torch.nonzero(self.failed_bomb_push[row], as_tuple=True)
        for grid_row, col, dir_idx in zip(bomb_rows.tolist(), bomb_cols.tolist(), bomb_dirs.tolist()):
            failed.add(("bomb", (int(col), int(grid_row)), int(dir_idx)))
        return failed

    def _sync_row_from_cpu_env(self, row: int, env: object) -> None:
        state = env.engine.get_state()
        self.walls[row] = torch.as_tensor(state.grid, dtype=torch.bool, device=self.device)
        self.car_pos[row, 0] = self._grid_coord(float(state.car_x))
        self.car_pos[row, 1] = self._grid_coord(float(state.car_y))

        self.box_pos[row].fill_(-1)
        self.box_ids[row].fill_(-1)
        self.box_alive[row].fill_(False)
        for slot, box in enumerate(state.boxes):
            self.box_pos[row, slot, 0] = int(box.x - 0.5)
            self.box_pos[row, slot, 1] = int(box.y - 0.5)
            self.box_ids[row, slot] = int(box.class_id)
            self.box_alive[row, slot] = True

        self.target_pos[row].fill_(-1)
        self.target_ids[row].fill_(-1)
        self.target_alive[row].fill_(False)
        for slot, target in enumerate(state.targets):
            self.target_pos[row, slot, 0] = int(target.x - 0.5)
            self.target_pos[row, slot, 1] = int(target.y - 0.5)
            self.target_ids[row, slot] = int(target.num_id)
            self.target_alive[row, slot] = True

        self.bomb_pos[row].fill_(-1)
        self.bomb_alive[row].fill_(False)
        for slot, bomb in enumerate(state.bombs):
            self.bomb_pos[row, slot, 0] = int(bomb.x - 0.5)
            self.bomb_pos[row, slot, 1] = int(bomb.y - 0.5)
            self.bomb_alive[row, slot] = True

        self.seen_box[row].fill_(False)
        for idx in state.seen_box_ids:
            if 0 <= idx < MAX_BOXES:
                self.seen_box[row, idx] = True
        self.seen_target[row].fill_(False)
        for idx in state.seen_target_ids:
            if 0 <= idx < MAX_TARGETS:
                self.seen_target[row, idx] = True

        self.failed_box_push[row].fill_(False)
        self.failed_bomb_push[row].fill_(False)
        for etype, entity_pos, dir_idx in getattr(env, "_failed_pushes", set()):
            col, grid_row = int(entity_pos[0]), int(entity_pos[1])
            if not (0 <= col < self.width and 0 <= grid_row < self.height and 0 <= dir_idx < N_DIRS):
                continue
            if etype == "box":
                self.failed_box_push[row, grid_row, col, dir_idx] = True
            elif etype == "bomb":
                self.failed_bomb_push[row, grid_row, col, dir_idx] = True

    def _cpu_emulate_push_row(
        self,
        row: int,
        action: int,
    ) -> Tuple[int, bool]:
        from smartcar_sokoban.engine import GameState
        from smartcar_sokoban.map_loader import BombInfo, BoxInfo, TargetInfo
        from smartcar_sokoban.rl.high_level_env import SokobanHLEnv

        if action < PUSH_BOX_START:
            return 0, False

        env = SokobanHLEnv(map_file="assets/maps/map1.txt", base_dir="", max_steps=int(self.max_steps[row].item()), include_map_layout=True)
        env.engine.state = GameState(
            grid=self.walls[row].to(dtype=torch.int64).cpu().tolist(),
            car_x=float(self.car_pos[row, 0].item()) + 0.5,
            car_y=float(self.car_pos[row, 1].item()) + 0.5,
            car_angle=ANGLE_UP,
            boxes=[
                BoxInfo(
                    x=float(self.box_pos[row, slot, 0].item()) + 0.5,
                    y=float(self.box_pos[row, slot, 1].item()) + 0.5,
                    class_id=int(self.box_ids[row, slot].item()),
                )
                for slot in range(MAX_BOXES)
                if bool(self.box_alive[row, slot].item())
            ],
            targets=[
                TargetInfo(
                    x=float(self.target_pos[row, slot, 0].item()) + 0.5,
                    y=float(self.target_pos[row, slot, 1].item()) + 0.5,
                    num_id=int(self.target_ids[row, slot].item()),
                )
                for slot in range(MAX_TARGETS)
                if bool(self.target_alive[row, slot].item())
            ],
            bombs=[
                BombInfo(
                    x=float(self.bomb_pos[row, slot, 0].item()) + 0.5,
                    y=float(self.bomb_pos[row, slot, 1].item()) + 0.5,
                )
                for slot in range(MAX_BOMBS)
                if bool(self.bomb_alive[row, slot].item())
            ],
            seen_box_ids={
                idx for idx in range(MAX_BOXES)
                if bool(self.seen_box[row, idx].item())
            },
            seen_target_ids={
                idx for idx in range(MAX_TARGETS)
                if bool(self.seen_target[row, idx].item())
            },
            won=bool((self.box_alive[row].sum() == 0).item()),
            score=int(self.total_pairs[row].item() - int(self.box_alive[row].sum().item())),
            total_pairs=int(self.total_pairs[row].item()),
        )
        env._failed_pushes = self._row_failed_push_set(row)

        if action < PUSH_BOMB_START:
            offset = action - PUSH_BOX_START
            steps, success = env._execute_single_push(offset // N_DIRS, offset % N_DIRS, "box")
        else:
            offset = action - PUSH_BOMB_START
            steps, success = env._execute_single_push(offset // N_DIRS, offset % N_DIRS, "bomb")
        self._sync_row_from_cpu_env(row, env)
        return steps, success

    def _advance_state_version(self) -> None:
        self._state_version += 1
        self._occupancy_grids_version = -1
        self._distance_map_version = -1
        self._box_distance_sums_version = -1
        self._box_distance_sums_cache = None
        self._state_hash_version = -1
        self._state_hash_cache = None
        self._push_trace_version = -1
        self._push_trace_cache = None

    def _advance_analysis_version(self) -> None:
        self._analysis_version += 1
        self._corner_mask_version = -1
        self._corner_mask_cache = None
        self._reverse_push_reachable_version = -1
        self._reverse_push_reachable_cache = None
        self._matching_tables_version = -1
        self._matched_exists_cache = None
        self._matched_target_idx_cache = None
        self._matched_target_pos_cache = None

    def _build_occupancy_grids(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._occupancy_grids_version == self._state_version and self._occ_grid_cache is not None:
            return self._box_grid_cache, self._bomb_grid_cache, self._occ_grid_cache

        if self._box_grid_cache is None:
            self._box_grid_cache = torch.zeros_like(self.walls)
            self._bomb_grid_cache = torch.zeros_like(self.walls)
            self._occ_grid_cache = torch.zeros_like(self.walls)
        box_grid = self._box_grid_cache.zero_()
        bomb_grid = self._bomb_grid_cache.zero_()
        if torch.any(self.box_alive):
            alive_boxes = self.box_alive
            box_grid[
                self._batch_idx_boxes[alive_boxes],
                self.box_pos[..., 1][alive_boxes],
                self.box_pos[..., 0][alive_boxes],
            ] = True

        if torch.any(self.bomb_alive):
            alive_bombs = self.bomb_alive
            bomb_grid[
                self._batch_idx_bombs[alive_bombs],
                self.bomb_pos[..., 1][alive_bombs],
                self.bomb_pos[..., 0][alive_bombs],
            ] = True

        occ_grid = self._occ_grid_cache
        occ_grid.copy_(box_grid)
        occ_grid |= bomb_grid
        self._occupancy_grids_version = self._state_version
        return box_grid, bomb_grid, occ_grid

    def _find_entity_at(
        self,
        cols: torch.Tensor,
        rows: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = cols.shape[0]
        kind = torch.full((batch,), -1, dtype=torch.long, device=self.device)
        slot = torch.full((batch,), -1, dtype=torch.long, device=self.device)
        valid = (cols >= 0) & (cols < self.width) & (rows >= 0) & (rows < self.height)
        if not torch.any(valid):
            return kind, slot

        box_match = (
            self.box_alive
            & (self.box_pos[:, :, 0] == cols.unsqueeze(1))
            & (self.box_pos[:, :, 1] == rows.unsqueeze(1))
        )
        box_any, box_slot_idx = box_match.max(dim=1)
        has_box = box_any & valid
        if torch.any(has_box):
            kind[has_box] = 0
            slot[has_box] = box_slot_idx[has_box].long()

        bomb_match = (
            self.bomb_alive
            & (self.bomb_pos[:, :, 0] == cols.unsqueeze(1))
            & (self.bomb_pos[:, :, 1] == rows.unsqueeze(1))
        )
        bomb_any, bomb_slot_idx = bomb_match.max(dim=1)
        has_bomb = bomb_any & valid & ~has_box
        if torch.any(has_bomb):
            kind[has_bomb] = 1
            slot[has_bomb] = bomb_slot_idx[has_bomb].long()

        return kind, slot

    def _find_entity_at_many(
        self,
        cols: torch.Tensor,
        rows: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kind = torch.full_like(cols, -1, dtype=torch.long, device=self.device)
        slot = torch.full_like(cols, -1, dtype=torch.long, device=self.device)
        valid = (cols >= 0) & (cols < self.width) & (rows >= 0) & (rows < self.height)
        if not torch.any(valid):
            return kind, slot

        box_match = (
            self.box_alive.unsqueeze(1)
            & (self.box_pos[:, None, :, 0] == cols.unsqueeze(-1))
            & (self.box_pos[:, None, :, 1] == rows.unsqueeze(-1))
        )
        box_any, box_slot_idx = box_match.max(dim=-1)
        has_box = box_any & valid
        if torch.any(has_box):
            kind = torch.where(has_box, torch.zeros_like(kind), kind)
            slot = torch.where(has_box, box_slot_idx.long(), slot)

        bomb_match = (
            self.bomb_alive.unsqueeze(1)
            & (self.bomb_pos[:, None, :, 0] == cols.unsqueeze(-1))
            & (self.bomb_pos[:, None, :, 1] == rows.unsqueeze(-1))
        )
        bomb_any, bomb_slot_idx = bomb_match.max(dim=-1)
        has_bomb = bomb_any & valid & ~has_box
        if torch.any(has_bomb):
            kind = torch.where(has_bomb, torch.ones_like(kind), kind)
            slot = torch.where(has_bomb, bomb_slot_idx.long(), slot)

        return kind, slot

    def distance_map(self) -> torch.Tensor:
        if self._distance_map_version == self._state_version and self._distance_map_cache is not None:
            return self._distance_map_cache

        box_grid, bomb_grid, occ_grid = self._build_occupancy_grids()
        free = ~self.walls & ~occ_grid
        if self._distance_map_cache is None:
            self._distance_map_cache = torch.empty(
                (self.batch_size, self.height, self.width),
                dtype=torch.int16,
                device=self.device,
            )
        dist = self._distance_map_cache.fill_(-1)
        frontier = self._distance_frontier_buf.zero_()
        expanded = self._distance_expanded_buf.zero_()
        unvisited = self._distance_unvisited_buf.copy_(free)
        frontier[self._batch_idx, self.car_pos[:, 1], self.car_pos[:, 0]] = True
        dist[frontier] = 0
        unvisited[frontier] = False

        for step in range(1, self.height * self.width + 1):
            self._expand_into(frontier, expanded)
            expanded &= unvisited
            if not torch.any(expanded):
                break
            dist[expanded] = step
            unvisited[expanded] = False
            frontier, expanded = expanded, frontier

        del box_grid, bomb_grid
        self._distance_map_version = self._state_version
        return dist

    def _expand(self, frontier: torch.Tensor) -> torch.Tensor:
        expanded = torch.zeros_like(frontier)
        self._expand_into(frontier, expanded)
        return expanded

    def _expand_into(self, frontier: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        out.zero_()
        out[:, 1:, :] |= frontier[:, :-1, :]
        out[:, :-1, :] |= frontier[:, 1:, :]
        out[:, :, 1:] |= frontier[:, :, :-1]
        out[:, :, :-1] |= frontier[:, :, 1:]
        return out

    def _shift_mask(self, tensor: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        shifted = torch.zeros_like(tensor)
        self._shift_mask_into(tensor, dx, dy, shifted)
        return shifted

    def _shift_mask_into(
        self,
        tensor: torch.Tensor,
        dx: int,
        dy: int,
        out: torch.Tensor,
    ) -> torch.Tensor:
        out.zero_()
        src_row_start = max(0, -dy)
        src_row_end = self.height - max(0, dy)
        dst_row_start = max(0, dy)
        dst_row_end = self.height - max(0, -dy)

        src_col_start = max(0, -dx)
        src_col_end = self.width - max(0, dx)
        dst_col_start = max(0, dx)
        dst_col_end = self.width - max(0, -dx)

        out[..., dst_row_start:dst_row_end, dst_col_start:dst_col_end] = tensor[
            ...,
            src_row_start:src_row_end,
            src_col_start:src_col_end,
        ]
        return out

    def _compute_corner_mask(self) -> torch.Tensor:
        if self._corner_mask_version == self._analysis_version and self._corner_mask_cache is not None:
            return self._corner_mask_cache

        wall_up = self._shift_mask(self.walls, 0, 1)
        wall_down = self._shift_mask(self.walls, 0, -1)
        wall_left = self._shift_mask(self.walls, 1, 0)
        wall_right = self._shift_mask(self.walls, -1, 0)
        corner_mask = (
            (wall_up & wall_left)
            | (wall_up & wall_right)
            | (wall_down & wall_left)
            | (wall_down & wall_right)
        )
        self._corner_mask_cache = corner_mask
        self._corner_mask_version = self._analysis_version
        return corner_mask

    def _compute_reverse_push_reachable(self) -> torch.Tensor:
        if (
            self._reverse_push_reachable_version == self._analysis_version
            and self._reverse_push_reachable_cache is not None
        ):
            return self._reverse_push_reachable_cache

        reachable = torch.zeros(
            (self.batch_size, MAX_TARGETS, self.height, self.width),
            dtype=torch.bool,
            device=self.device,
        )
        frontier = torch.zeros_like(reachable)
        alive = self.target_alive
        batch_idx, target_idx = torch.nonzero(alive, as_tuple=True)
        if batch_idx.numel() == 0:
            self._reverse_push_reachable_cache = reachable
            self._reverse_push_reachable_version = self._analysis_version
            return reachable

        cols = self.target_pos[batch_idx, target_idx, 0]
        rows = self.target_pos[batch_idx, target_idx, 1]
        frontier[batch_idx, target_idx, rows, cols] = True
        reachable |= frontier
        free = ~self.walls.unsqueeze(1)
        unreached = ~reachable
        support_by_dir = [
            free & self._shift_mask(free, dx, dy)
            for dx, dy in DIR_DELTAS
        ]
        next_frontier = torch.zeros_like(frontier)
        scratch = torch.zeros_like(frontier)

        while torch.any(frontier):
            next_frontier.zero_()
            for dir_idx, (dx, dy) in enumerate(DIR_DELTAS):
                prev_cells = self._shift_mask_into(frontier, -dx, -dy, scratch)
                prev_cells &= support_by_dir[dir_idx]
                prev_cells &= unreached
                next_frontier |= prev_cells
            frontier, next_frontier = next_frontier, frontier
            reachable |= frontier
            unreached &= ~frontier

        self._reverse_push_reachable_cache = reachable
        self._reverse_push_reachable_version = self._analysis_version
        return reachable

    def _matching_target_tables(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self._matching_tables_version == self._analysis_version
            and self._matched_exists_cache is not None
            and self._matched_target_idx_cache is not None
            and self._matched_target_pos_cache is not None
        ):
            return (
                self._matched_exists_cache,
                self._matched_target_idx_cache,
                self._matched_target_pos_cache,
            )

        match = (
            self.box_alive.unsqueeze(2)
            & self.target_alive.unsqueeze(1)
            & (self.box_ids.unsqueeze(2) == self.target_ids.unsqueeze(1))
        )
        matched_exists = match.any(dim=2)
        matched_target_idx = match.float().argmax(dim=2)
        matched_target_pos = torch.gather(
            self.target_pos,
            1,
            matched_target_idx.unsqueeze(-1).expand(self.batch_size, MAX_BOXES, 2),
        )
        self._matched_exists_cache = matched_exists
        self._matched_target_idx_cache = matched_target_idx
        self._matched_target_pos_cache = matched_target_pos
        self._matching_tables_version = self._analysis_version
        return matched_exists, matched_target_idx, matched_target_pos

    def _deadlock_mask_for_pushes(
        self,
        trace: PushTrace,
        is_box: torch.Tensor,
        entity_slot: torch.Tensor,
        dir_idx: torch.Tensor,
    ) -> torch.Tensor:
        no_bombs = ~self.bomb_alive.any(dim=1)
        if not torch.any(no_bombs):
            return torch.zeros_like(trace.valid)
        candidate_mask = no_bombs.unsqueeze(1) & is_box & trace.valid
        if not torch.any(candidate_mask):
            return torch.zeros_like(trace.valid)

        corner_mask = self._compute_corner_mask()
        corner_flat = corner_mask.view(self.batch_size, -1)
        reachable = self._compute_reverse_push_reachable()
        reachable_flat = reachable.view(self.batch_size, MAX_TARGETS, -1)

        matched_exists, matched_target_idx, matched_target_pos = self._matching_target_tables()
        rows, cols = torch.nonzero(candidate_mask, as_tuple=True)
        dx, dy = self._dir_vectors_many(dir_idx[rows, cols])
        dest_col = trace.cell_col[rows, cols, 0] + dx
        dest_row = trace.cell_row[rows, cols, 0] + dy
        safe_box_slot = entity_slot[rows, cols].clamp(0, MAX_BOXES - 1)
        action_match_exists = matched_exists[rows, safe_box_slot]
        action_target_idx = matched_target_idx[rows, safe_box_slot]
        action_target_pos = matched_target_pos[rows, safe_box_slot]

        safe_dest_col = dest_col.clamp(0, self.width - 1)
        safe_dest_row = dest_row.clamp(0, self.height - 1)
        dest_linear = safe_dest_row * self.width + safe_dest_col
        corner_at_dest = corner_flat[rows, dest_linear]
        reachable_at_dest = reachable_flat[rows, action_target_idx, dest_linear]
        reachable_at_dest = torch.where(
            action_match_exists,
            reachable_at_dest,
            torch.ones_like(reachable_at_dest),
        )
        matching_target_cell = (
            action_match_exists
            & (dest_col == action_target_pos[:, 0])
            & (dest_row == action_target_pos[:, 1])
        )
        dead = ~matching_target_cell & ((~reachable_at_dest) | corner_at_dest)
        deadlock = torch.zeros_like(trace.valid)
        deadlock[rows, cols] = dead
        return deadlock

    def dead_box_counts(self) -> torch.Tensor:
        no_bombs = ~self.bomb_alive.any(dim=1)
        if not torch.any(no_bombs):
            return torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        corner_mask = self._compute_corner_mask()
        corner_flat = corner_mask.view(self.batch_size, -1)
        reachable = self._compute_reverse_push_reachable().view(self.batch_size, MAX_TARGETS, -1)
        matched_exists, matched_target_idx, matched_target_pos = self._matching_target_tables()
        cols = self.box_pos[..., 0].clamp(0, self.width - 1)
        rows = self.box_pos[..., 1].clamp(0, self.height - 1)
        linear = rows * self.width + cols
        corner = corner_flat[self._batch_idx_boxes, linear]
        reachable_here = reachable[self._batch_idx_boxes, matched_target_idx, linear]
        reachable_here = torch.where(matched_exists, reachable_here, torch.ones_like(reachable_here))
        matching_target = (
            matched_exists
            & (self.box_pos[..., 0] == matched_target_pos[..., 0])
            & (self.box_pos[..., 1] == matched_target_pos[..., 1])
        )
        dead = (
            no_bombs.unsqueeze(1)
            & self.box_alive
            & ~matching_target
            & ((~reachable_here) | corner)
        )
        return dead.sum(dim=1)

    def box_distance_sums(self) -> torch.Tensor:
        if self._box_distance_sums_version == self._state_version and self._box_distance_sums_cache is not None:
            return self._box_distance_sums_cache

        matched_exists, _, matched_target_pos = self._matching_target_tables()
        dist = (
            (self.box_pos[..., 0] - matched_target_pos[..., 0]).abs()
            + (self.box_pos[..., 1] - matched_target_pos[..., 1]).abs()
        ).float()
        dist = torch.where(self.box_alive & matched_exists, dist, torch.zeros_like(dist))
        summed = dist.sum(dim=1)
        self._box_distance_sums_cache = summed
        self._box_distance_sums_version = self._state_version
        return summed

    def _dir_vectors(self, dir_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        deltas = self.tables.dir_deltas[dir_idx]
        return deltas[:, 0], deltas[:, 1]

    def _dir_vectors_many(self, dir_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        deltas = self.tables.dir_deltas[dir_idx]
        return deltas[..., 0], deltas[..., 1]

    def _push_action_specs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_count = int(self.tables.push_ids.shape[0])
        is_box = self.tables.push_is_box.unsqueeze(0).expand(self.batch_size, action_count)
        entity = self.tables.push_entity.unsqueeze(0).expand(self.batch_size, action_count)
        dirs = self.tables.push_dir.unsqueeze(0).expand(self.batch_size, action_count)
        ids = self.tables.push_ids
        return is_box, entity, dirs, ids

    def _push_action_indices(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.where(
            actions < PUSH_BOMB_START,
            actions - PUSH_BOX_START,
            actions - PUSH_BOMB_START + MAX_BOXES * N_DIRS,
        )

    def _push_failed_mask_many(
        self,
        is_box: torch.Tensor,
        entity_slot: torch.Tensor,
        dir_idx: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, action_count = entity_slot.shape
        safe_box_slot = entity_slot.clamp(0, MAX_BOXES - 1)
        safe_bomb_slot = entity_slot.clamp(0, MAX_BOMBS - 1)
        box_pos = torch.gather(
            self.box_pos,
            1,
            safe_box_slot.unsqueeze(-1).expand(batch_size, action_count, 2),
        )
        bomb_pos = torch.gather(
            self.bomb_pos,
            1,
            safe_bomb_slot.unsqueeze(-1).expand(batch_size, action_count, 2),
        )
        pos = torch.where(is_box.unsqueeze(-1), box_pos, bomb_pos)
        cols = pos[..., 0].clamp(0, self.width - 1)
        rows = pos[..., 1].clamp(0, self.height - 1)
        batch_idx = self._batch_idx.unsqueeze(1).expand_as(cols)
        box_failed = self.failed_box_push[batch_idx, rows, cols, dir_idx]
        bomb_failed = self.failed_bomb_push[batch_idx, rows, cols, dir_idx]
        return torch.where(is_box, box_failed, bomb_failed)

    def _clear_failed_pushes(self, row_mask: torch.Tensor) -> None:
        if not torch.any(row_mask):
            return
        self.failed_box_push[row_mask] = False
        self.failed_bomb_push[row_mask] = False

    def _mark_failed_pushes(
        self,
        is_box: torch.Tensor,
        entity_slot: torch.Tensor,
        dir_idx: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> None:
        if not torch.any(row_mask):
            return
        safe_box_slot = entity_slot.clamp(0, MAX_BOXES - 1)
        safe_bomb_slot = entity_slot.clamp(0, MAX_BOMBS - 1)
        alive = torch.where(
            is_box,
            self.box_alive[self._batch_idx, safe_box_slot],
            self.bomb_alive[self._batch_idx, safe_bomb_slot],
        )
        pos = torch.where(
            is_box.unsqueeze(1),
            self.box_pos[self._batch_idx, safe_box_slot],
            self.bomb_pos[self._batch_idx, safe_bomb_slot],
        )
        in_bounds = (
            (pos[:, 0] >= 0)
            & (pos[:, 0] < self.width)
            & (pos[:, 1] >= 0)
            & (pos[:, 1] < self.height)
        )
        failed_rows = row_mask & alive & in_bounds
        if not torch.any(failed_rows):
            return
        rows = self._batch_idx[failed_rows]
        cols = pos[failed_rows, 0]
        grid_rows = pos[failed_rows, 1]
        dirs = dir_idx[failed_rows]
        box_rows = rows[is_box[failed_rows]]
        bomb_rows = rows[~is_box[failed_rows]]
        if box_rows.numel() > 0:
            box_sel = is_box[failed_rows]
            self.failed_box_push[
                box_rows,
                grid_rows[box_sel],
                cols[box_sel],
                dirs[box_sel],
            ] = True
        if bomb_rows.numel() > 0:
            bomb_sel = ~is_box[failed_rows]
            self.failed_bomb_push[
                bomb_rows,
                grid_rows[bomb_sel],
                cols[bomb_sel],
                dirs[bomb_sel],
            ] = True

    def _trace_from_many(self, trace_many: PushTrace, action_indices: torch.Tensor) -> PushTrace:
        return PushTrace(
            valid=trace_many.valid[self._batch_idx, action_indices],
            kind=trace_many.kind[self._batch_idx, action_indices],
            slot=trace_many.slot[self._batch_idx, action_indices],
            cell_col=trace_many.cell_col[self._batch_idx, action_indices],
            cell_row=trace_many.cell_row[self._batch_idx, action_indices],
            chain_len=trace_many.chain_len[self._batch_idx, action_indices],
            stand_col=trace_many.stand_col[self._batch_idx, action_indices],
            stand_row=trace_many.stand_row[self._batch_idx, action_indices],
            low_steps=trace_many.low_steps[self._batch_idx, action_indices],
        )

    def _trace_push(
        self,
        is_box: torch.Tensor,
        entity_slot: torch.Tensor,
        dir_idx: torch.Tensor,
        distance_map: Optional[torch.Tensor] = None,
    ) -> PushTrace:
        dx, dy = self._dir_vectors(dir_idx)

        head_pos = torch.where(
            is_box.unsqueeze(1),
            self.box_pos[self._batch_idx, entity_slot],
            self.bomb_pos[self._batch_idx, entity_slot],
        )
        stand_col = head_pos[:, 0] - dx
        stand_row = head_pos[:, 1] - dy

        valid = (
            (entity_slot >= 0)
            & torch.where(
                is_box,
                self.box_alive[self._batch_idx, entity_slot],
                self.bomb_alive[self._batch_idx, entity_slot],
            )
            & (stand_col >= 0)
            & (stand_col < self.width)
            & (stand_row >= 0)
            & (stand_row < self.height)
        )
        valid &= ~self.walls[self._batch_idx, stand_row.clamp(0, self.height - 1), stand_col.clamp(0, self.width - 1)]

        stand_dist = torch.zeros((self.batch_size,), dtype=torch.int16, device=self.device)
        stand_in_bounds = (
            (stand_col >= 0)
            & (stand_col < self.width)
            & (stand_row >= 0)
            & (stand_row < self.height)
        )
        valid_rows = torch.nonzero(stand_in_bounds, as_tuple=False).squeeze(1)
        if valid_rows.numel() > 0:
            if distance_map is not None:
                stand_dist[valid_rows] = distance_map[valid_rows, stand_row[valid_rows], stand_col[valid_rows]]
                valid[valid_rows] &= stand_dist[valid_rows] >= 0

        max_chain = MAX_BOXES + MAX_BOMBS
        kind = torch.full((self.batch_size, max_chain), -1, dtype=torch.long, device=self.device)
        slot = torch.full((self.batch_size, max_chain), -1, dtype=torch.long, device=self.device)
        cell_col = torch.full((self.batch_size, max_chain), -1, dtype=torch.long, device=self.device)
        cell_row = torch.full((self.batch_size, max_chain), -1, dtype=torch.long, device=self.device)
        chain_len = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)

        current_kind = torch.where(is_box, torch.zeros_like(entity_slot), torch.ones_like(entity_slot))
        current_slot = entity_slot.clone()
        current_col = head_pos[:, 0].clone()
        current_row = head_pos[:, 1].clone()
        active = valid.clone()

        for depth in range(max_chain):
            if not torch.any(active):
                break
            kind[active, depth] = current_kind[active]
            slot[active, depth] = current_slot[active]
            cell_col[active, depth] = current_col[active]
            cell_row[active, depth] = current_row[active]
            chain_len[active] = depth + 1

            next_col = current_col + dx
            next_row = current_row + dy
            in_bounds = (
                (next_col >= 0)
                & (next_col < self.width)
                & (next_row >= 0)
                & (next_row < self.height)
            )
            wall_hit = ~in_bounds
            bounded = torch.nonzero(in_bounds, as_tuple=False).squeeze(1)
            if bounded.numel() > 0:
                wall_hit[bounded] |= self.walls[bounded, next_row[bounded], next_col[bounded]]

            box_block = active & wall_hit & (current_kind == 0)
            valid[box_block] = False

            done = active & wall_hit & (current_kind == 1)
            active = active & ~wall_hit

            if not torch.any(active):
                break

            next_kind, next_slot = self._find_entity_at(next_col, next_row)
            occupied = active & (next_kind >= 0)
            empty = active & ~occupied
            done |= empty

            current_kind = next_kind
            current_slot = next_slot
            current_col = next_col
            current_row = next_row
            active = occupied

        stand_reachable = stand_in_bounds & (stand_dist >= 0)
        low_steps = (
            torch.where(stand_reachable, stand_dist + 1, torch.zeros_like(stand_dist))
            if distance_map is not None
            else torch.zeros_like(stand_dist)
        )
        return PushTrace(
            valid=valid,
            kind=kind,
            slot=slot,
            cell_col=cell_col,
            cell_row=cell_row,
            chain_len=chain_len,
            stand_col=stand_col,
            stand_row=stand_row,
            low_steps=low_steps,
        )

    def _trace_push_many(
        self,
        is_box: torch.Tensor,
        entity_slot: torch.Tensor,
        dir_idx: torch.Tensor,
        distance_map: Optional[torch.Tensor] = None,
    ) -> PushTrace:
        shape = entity_slot.shape
        batch_size, action_count = shape
        dx, dy = self._dir_vectors_many(dir_idx)
        safe_box_slot = entity_slot.clamp(0, MAX_BOXES - 1)
        safe_bomb_slot = entity_slot.clamp(0, MAX_BOMBS - 1)

        box_head = torch.gather(
            self.box_pos,
            1,
            safe_box_slot.unsqueeze(-1).expand(batch_size, action_count, 2),
        )
        bomb_head = torch.gather(
            self.bomb_pos,
            1,
            safe_bomb_slot.unsqueeze(-1).expand(batch_size, action_count, 2),
        )
        head_pos = torch.where(is_box.unsqueeze(-1), box_head, bomb_head)

        box_alive = torch.gather(self.box_alive, 1, safe_box_slot)
        bomb_alive = torch.gather(self.bomb_alive, 1, safe_bomb_slot)
        entity_alive = torch.where(is_box, box_alive, bomb_alive)

        stand_col = head_pos[..., 0] - dx
        stand_row = head_pos[..., 1] - dy
        valid = (
            entity_alive
            & (stand_col >= 0)
            & (stand_col < self.width)
            & (stand_row >= 0)
            & (stand_row < self.height)
        )

        batch_idx = self._batch_idx.unsqueeze(1).expand(batch_size, action_count)
        safe_stand_col = stand_col.clamp(0, self.width - 1)
        safe_stand_row = stand_row.clamp(0, self.height - 1)
        stand_is_wall = self.walls[batch_idx, safe_stand_row, safe_stand_col]
        valid &= ~stand_is_wall

        stand_linear = safe_stand_row * self.width + safe_stand_col
        stand_dist = torch.zeros((batch_size, action_count), dtype=torch.int16, device=self.device)
        stand_in_bounds = (
            (stand_col >= 0)
            & (stand_col < self.width)
            & (stand_row >= 0)
            & (stand_row < self.height)
        )
        valid &= stand_in_bounds
        if distance_map is not None:
            dist_flat = distance_map.view(batch_size, -1)
            stand_dist = torch.gather(dist_flat, 1, stand_linear)
            stand_dist = torch.where(stand_in_bounds, stand_dist, torch.zeros_like(stand_dist))
            valid &= stand_dist >= 0

        max_chain = MAX_BOXES + MAX_BOMBS
        chain_shape = (batch_size, action_count, max_chain)
        kind = torch.full(chain_shape, -1, dtype=torch.long, device=self.device)
        slot = torch.full(chain_shape, -1, dtype=torch.long, device=self.device)
        cell_col = torch.full(chain_shape, -1, dtype=torch.long, device=self.device)
        cell_row = torch.full(chain_shape, -1, dtype=torch.long, device=self.device)
        chain_len = torch.zeros((batch_size, action_count), dtype=torch.long, device=self.device)

        current_kind = torch.where(is_box, torch.zeros_like(entity_slot), torch.ones_like(entity_slot))
        current_slot = entity_slot.clone()
        current_col = head_pos[..., 0].clone()
        current_row = head_pos[..., 1].clone()
        active = valid.clone()

        for depth in range(max_chain):
            if not torch.any(active):
                break
            active_rows, active_cols = torch.nonzero(active, as_tuple=True)
            kind[active_rows, active_cols, depth] = current_kind[active_rows, active_cols]
            slot[active_rows, active_cols, depth] = current_slot[active_rows, active_cols]
            cell_col[active_rows, active_cols, depth] = current_col[active_rows, active_cols]
            cell_row[active_rows, active_cols, depth] = current_row[active_rows, active_cols]
            chain_len[active_rows, active_cols] = depth + 1

            next_col = current_col + dx
            next_row = current_row + dy
            in_bounds = (
                (next_col >= 0)
                & (next_col < self.width)
                & (next_row >= 0)
                & (next_row < self.height)
            )
            safe_next_col = next_col.clamp(0, self.width - 1)
            safe_next_row = next_row.clamp(0, self.height - 1)
            wall_hit = ~in_bounds | self.walls[batch_idx, safe_next_row, safe_next_col]
            wall_hit &= active

            box_block = wall_hit & (current_kind == 0)
            valid &= ~box_block
            active &= ~wall_hit
            if not torch.any(active):
                break

            next_kind, next_slot = self._find_entity_at_many(next_col, next_row)
            occupied = active & (next_kind >= 0)
            current_kind = next_kind
            current_slot = next_slot
            current_col = next_col
            current_row = next_row
            active = occupied

        low_steps = (
            torch.where(stand_dist >= 0, stand_dist + 1, torch.zeros_like(stand_dist))
            if distance_map is not None
            else torch.zeros_like(stand_dist)
        )
        return PushTrace(
            valid=valid,
            kind=kind,
            slot=slot,
            cell_col=cell_col,
            cell_row=cell_row,
            chain_len=chain_len,
            stand_col=stand_col,
            stand_row=stand_row,
            low_steps=low_steps,
        )

    def push_action_masks(self) -> torch.Tensor:
        mask = torch.zeros((self.batch_size, N_ACTIONS), dtype=torch.bool, device=self.device)
        distance_map = self.distance_map()
        is_box, entity_slot, dir_idx, action_ids = self._push_action_specs()
        trace = self._trace_push_many(
            is_box=is_box,
            entity_slot=entity_slot,
            dir_idx=dir_idx,
            distance_map=distance_map,
        )
        fallback_valid = trace.valid.clone()
        deadlock_mask = self._deadlock_mask_for_pushes(
            trace=trace,
            is_box=is_box,
            entity_slot=entity_slot,
            dir_idx=dir_idx,
        )
        failed_mask = self._push_failed_mask_many(is_box=is_box, entity_slot=entity_slot, dir_idx=dir_idx)
        trace.valid &= ~deadlock_mask & ~failed_mask
        mask[:, action_ids] = trace.valid
        no_valid = ~trace.valid.any(dim=1)
        has_fallback = fallback_valid.any(dim=1)
        fallback_rows = torch.nonzero(no_valid & has_fallback, as_tuple=False).squeeze(1)
        if fallback_rows.numel() > 0:
            fallback_idx = fallback_valid.float().argmax(dim=1)
            mask[fallback_rows, action_ids[fallback_idx[fallback_rows]]] = True
        trace.valid = fallback_valid
        self._push_trace_cache = trace
        self._push_trace_version = self._state_version
        return mask

    def action_masks(self) -> torch.Tensor:
        mask = torch.zeros((self.batch_size, N_ACTIONS), dtype=torch.bool, device=self.device)
        won = self.box_alive.sum(dim=1) == 0
        active = ~won

        known_boxes = self._box_known_mask()
        known_targets = self._target_known_mask()
        unseen_boxes = self.box_alive & ~known_boxes
        unseen_targets = self.target_alive & ~known_targets

        can_explore_boxes = active & (unseen_boxes.sum(dim=1) >= 2)
        can_explore_targets = active & (unseen_targets.sum(dim=1) >= 2)
        mask[:, EXPLORE_BOX_START:EXPLORE_TGT_START] = unseen_boxes & can_explore_boxes.unsqueeze(1)
        mask[:, EXPLORE_TGT_START:PUSH_BOX_START] = unseen_targets & can_explore_targets.unsqueeze(1)

        distance_map = self.distance_map()
        is_box, entity_slot, dir_idx, action_ids = self._push_action_specs()
        trace = self._trace_push_many(
            is_box=is_box,
            entity_slot=entity_slot,
            dir_idx=dir_idx,
            distance_map=distance_map,
        )
        fallback_valid = trace.valid.clone()
        deadlock_mask = self._deadlock_mask_for_pushes(
            trace=trace,
            is_box=is_box,
            entity_slot=entity_slot,
            dir_idx=dir_idx,
        )
        failed_mask = self._push_failed_mask_many(is_box=is_box, entity_slot=entity_slot, dir_idx=dir_idx)
        push_valid = trace.valid & ~deadlock_mask & ~failed_mask
        mask[:, action_ids] = push_valid & active.unsqueeze(1)

        no_valid = active & ~mask.any(dim=1)
        has_fallback = fallback_valid.any(dim=1)
        fallback_rows = torch.nonzero(no_valid & has_fallback, as_tuple=False).squeeze(1)
        if fallback_rows.numel() > 0:
            fallback_idx = fallback_valid.float().argmax(dim=1)
            mask[fallback_rows, action_ids[fallback_idx[fallback_rows]]] = True

        box_rows = torch.nonzero(no_valid & ~has_fallback & self.box_alive.any(dim=1), as_tuple=False).squeeze(1)
        if box_rows.numel() > 0:
            mask[box_rows, PUSH_BOX_START] = True
        bomb_rows = torch.nonzero(
            no_valid & ~has_fallback & ~self.box_alive.any(dim=1) & self.bomb_alive.any(dim=1),
            as_tuple=False,
        ).squeeze(1)
        if bomb_rows.numel() > 0:
            mask[bomb_rows, PUSH_BOMB_START] = True
        other_rows = torch.nonzero(
            no_valid & ~has_fallback & ~self.box_alive.any(dim=1) & ~self.bomb_alive.any(dim=1),
            as_tuple=False,
        ).squeeze(1)
        if other_rows.numel() > 0:
            mask[other_rows, 0] = True

        trace.valid = fallback_valid
        self._push_trace_cache = trace
        self._push_trace_version = self._state_version
        return mask

    def _compact_order(self, alive: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        slots = alive.shape[1]
        slot_order = torch.arange(slots, device=self.device).unsqueeze(0).expand_as(alive)
        keys = torch.where(alive, slot_order, slot_order + slots)
        order = torch.argsort(keys, dim=1)
        new_alive = torch.gather(alive, 1, order)
        return order, new_alive

    def _compact_slots(
        self,
        pos: torch.Tensor,
        ids: Optional[torch.Tensor],
        alive: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        order, new_alive = self._compact_order(alive)
        new_pos = torch.gather(pos, 1, order.unsqueeze(-1).expand_as(pos))
        new_pos = torch.where(new_alive.unsqueeze(-1), new_pos, torch.full_like(new_pos, -1))

        new_ids = None
        if ids is not None:
            new_ids = torch.gather(ids, 1, order)
            new_ids = torch.where(new_alive, new_ids, torch.full_like(new_ids, -1))

        return new_pos, new_ids, new_alive, order

    def _apply_bomb_explosions(self, row_mask: Optional[torch.Tensor] = None) -> bool:
        cols = self.bomb_pos[..., 0]
        rows = self.bomb_pos[..., 1]
        in_bounds = (
            (cols >= 0)
            & (cols < self.width)
            & (rows >= 0)
            & (rows < self.height)
        )
        safe_cols = cols.clamp(0, self.width - 1)
        safe_rows = rows.clamp(0, self.height - 1)
        hit_wall = self.walls[self._batch_idx_bombs, safe_rows, safe_cols]
        exploding = self.bomb_alive & (~in_bounds | hit_wall)
        if row_mask is not None:
            exploding &= row_mask.unsqueeze(1)
        if not torch.any(exploding):
            return False

        walls_flat = self.walls.view(self.batch_size, -1)
        safe_linear = (safe_rows * self.width + safe_cols).clamp(0, self.width * self.height - 1)
        neighbor_linear = self.tables.neighbor_linear[safe_linear]
        neighbor_valid = self.tables.neighbor_valid[safe_linear]
        neighbor_is_wall = torch.gather(
            walls_flat,
            1,
            neighbor_linear.view(self.batch_size, -1),
        ).view(self.batch_size, MAX_BOMBS, -1)
        valid_wall_centers = exploding.unsqueeze(-1) & neighbor_valid & neighbor_is_wall
        if torch.any(valid_wall_centers):
            clear_linear = self.tables.blast_linear[neighbor_linear]
            clear_valid = self.tables.blast_valid[neighbor_linear] & valid_wall_centers.unsqueeze(-1)
            clear_batch = self._batch_idx_bombs.unsqueeze(-1).unsqueeze(-1).expand_as(clear_linear)
            walls_flat[clear_batch[clear_valid], clear_linear[clear_valid]] = False
            self.walls = walls_flat.view(self.batch_size, self.height, self.width)

        self.bomb_alive &= ~exploding
        self.bomb_pos, _, self.bomb_alive, _ = self._compact_slots(
            self.bomb_pos,
            None,
            self.bomb_alive,
        )
        return True

    def _apply_pairings(self, row_mask: Optional[torch.Tensor] = None) -> bool:
        match = (
            self.box_alive.unsqueeze(2)
            & self.target_alive.unsqueeze(1)
            & (self.box_pos[:, :, None, 0] == self.target_pos[:, None, :, 0])
            & (self.box_pos[:, :, None, 1] == self.target_pos[:, None, :, 1])
            & (self.box_ids[:, :, None] == self.target_ids[:, None, :])
        )
        if row_mask is not None:
            match &= row_mask.view(self.batch_size, 1, 1)
        if not torch.any(match):
            return False
        box_remove = match.any(dim=2)
        target_remove = match.any(dim=1)
        self.box_alive &= ~box_remove
        self.target_alive &= ~target_remove
        self.box_pos, self.box_ids, self.box_alive, box_order = self._compact_slots(
            self.box_pos,
            self.box_ids,
            self.box_alive,
        )
        self.seen_box = torch.gather(self.seen_box, 1, box_order) & self.box_alive
        self.target_pos, self.target_ids, self.target_alive, target_order = self._compact_slots(
            self.target_pos,
            self.target_ids,
            self.target_alive,
        )
        self.seen_target = torch.gather(self.seen_target, 1, target_order) & self.target_alive
        return True

    def _step_push(
        self,
        actions: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> GpuPushStepResult:
        is_box = actions < PUSH_BOMB_START
        entity_slot = torch.where(
            is_box,
            (actions - PUSH_BOX_START) // N_DIRS,
            (actions - PUSH_BOMB_START) // N_DIRS,
        )
        dir_idx = torch.where(
            is_box,
            (actions - PUSH_BOX_START) % N_DIRS,
            (actions - PUSH_BOMB_START) % N_DIRS,
        )

        if self._push_trace_version == self._state_version and self._push_trace_cache is not None:
            trace = self._trace_from_many(self._push_trace_cache, self._push_action_indices(actions))
        else:
            trace = self._trace_push(
                is_box=is_box,
                entity_slot=entity_slot,
                dir_idx=dir_idx,
                distance_map=self.distance_map(),
            )
        safe_box_slot = entity_slot.clamp(0, MAX_BOXES - 1)
        safe_bomb_slot = entity_slot.clamp(0, MAX_BOMBS - 1)
        alive = torch.where(
            is_box,
            self.box_alive[self._batch_idx, safe_box_slot],
            self.bomb_alive[self._batch_idx, safe_bomb_slot],
        )
        failed_mask = self._push_failed_mask_many(
            is_box=is_box.unsqueeze(1),
            entity_slot=entity_slot.unsqueeze(1),
            dir_idx=dir_idx.unsqueeze(1),
        ).squeeze(1)
        valid = trace.valid & active_mask & ~failed_mask
        moved = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
        reached_stand = active_mask & (trace.low_steps > 0)
        manual_reached = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
        low_steps = trace.low_steps.clone()
        dx, dy = self._dir_vectors(dir_idx)
        max_chain = int(trace.kind.shape[1])
        depth_idx = torch.arange(max_chain, device=self.device).unsqueeze(0)
        move_mask = valid.unsqueeze(1) & (trace.chain_len.unsqueeze(1) > depth_idx)
        row_idx = self._batch_idx.unsqueeze(1).expand_as(trace.kind)
        dest_col = trace.cell_col + dx.unsqueeze(1)
        dest_row = trace.cell_row + dy.unsqueeze(1)

        box_mask = move_mask & (trace.kind == 0)
        if torch.any(box_mask):
            box_rows = row_idx[box_mask]
            box_slots = trace.slot[box_mask]
            self.box_pos[box_rows, box_slots, 0] = dest_col[box_mask]
            self.box_pos[box_rows, box_slots, 1] = dest_row[box_mask]

        bomb_mask = move_mask & (trace.kind == 1)
        if torch.any(bomb_mask):
            bomb_rows = row_idx[bomb_mask]
            bomb_slots = trace.slot[bomb_mask]
            self.bomb_pos[bomb_rows, bomb_slots, 0] = dest_col[bomb_mask]
            self.bomb_pos[bomb_rows, bomb_slots, 1] = dest_row[bomb_mask]

        moved = valid & (trace.chain_len > 0)

        self.car_pos[reached_stand, 0] = trace.stand_col[reached_stand]
        self.car_pos[reached_stand, 1] = trace.stand_row[reached_stand]
        self.car_pos[valid, 0] = trace.cell_col[valid, 0]
        self.car_pos[valid, 1] = trace.cell_row[valid, 0]
        self.step_count += active_mask.long()
        self.total_low_steps += torch.where(active_mask, trace.low_steps, torch.zeros_like(trace.low_steps))

        manual_rows = torch.nonzero(active_mask & alive & ~moved, as_tuple=False).squeeze(1)
        for row_tensor in manual_rows:
            row = int(row_tensor.item())
            extra_steps, cpu_success = self._cpu_emulate_push_row(row=row, action=int(actions[row].item()))
            low_steps[row] = int(extra_steps)
            self.total_low_steps[row] += int(extra_steps)
            valid[row] = bool(cpu_success)
            moved[row] = bool(cpu_success)
            manual_reached[row] = bool(extra_steps > 0 or cpu_success)

        exploded = self._apply_bomb_explosions(row_mask=active_mask)
        paired = self._apply_pairings(row_mask=active_mask)
        if torch.any(valid | manual_reached):
            self._advance_state_version()
            if exploded or paired or manual_rows.numel() > 0:
                self._advance_analysis_version()
        fast_rows = active_mask & ~manual_reached
        self._clear_failed_pushes(moved & fast_rows)
        self._mark_failed_pushes(
            is_box=is_box,
            entity_slot=entity_slot,
            dir_idx=dir_idx,
            row_mask=fast_rows & alive & ~moved,
        )
        visibility_rows = torch.nonzero((reached_stand | manual_reached) & fast_rows, as_tuple=False).squeeze(1).detach().cpu().tolist()
        for row in visibility_rows:
            self._refresh_visibility_row(int(row))
        won = self.box_alive.sum(dim=1) == 0
        return GpuPushStepResult(
            valid=valid,
            moved=moved,
            won=won,
            low_steps=low_steps,
            success=moved,
        )

    def _step_explore(
        self,
        actions: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> GpuPushStepResult:
        valid = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
        moved = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
        low_steps = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)
        any_state_change = False

        active_rows = torch.nonzero(active_mask, as_tuple=False).squeeze(1).detach().cpu().tolist()
        for row in active_rows:
            steps, success, car_moved, new_car_pos, new_seen_box, new_seen_target = self._simulate_explore_row(
                row=row,
                action=int(actions[row].item()),
            )
            self.step_count[row] += 1
            self.total_low_steps[row] += int(steps)
            low_steps[row] = int(steps)
            valid[row] = bool(success)
            moved[row] = bool(car_moved)
            if car_moved:
                self.car_pos[row] = new_car_pos
                any_state_change = True
            if not torch.equal(self.seen_box[row], new_seen_box):
                self.seen_box[row] = new_seen_box
            if not torch.equal(self.seen_target[row], new_seen_target):
                self.seen_target[row] = new_seen_target

        if any_state_change:
            self._advance_state_version()
        won = self.box_alive.sum(dim=1) == 0
        return GpuPushStepResult(
            valid=valid,
            moved=moved,
            won=won,
            low_steps=low_steps,
            success=valid,
        )

    def step(
        self,
        actions: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> GpuPushStepResult:
        actions = actions.to(device=self.device, dtype=torch.long)
        if active_mask is None:
            active_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        else:
            active_mask = active_mask.to(device=self.device, dtype=torch.bool)

        valid = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
        moved = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
        low_steps = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)
        success = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)

        push_rows = active_mask & (actions >= PUSH_BOX_START)
        if torch.any(push_rows):
            push_result = self._step_push(actions=actions, active_mask=push_rows)
            valid |= push_result.valid
            moved |= push_result.moved
            low_steps += push_result.low_steps
            if push_result.success is not None:
                success |= push_result.success

        explore_rows = active_mask & (actions < PUSH_BOX_START)
        if torch.any(explore_rows):
            explore_result = self._step_explore(actions=actions, active_mask=explore_rows)
            valid |= explore_result.valid
            moved |= explore_result.moved
            low_steps += explore_result.low_steps
            if explore_result.success is not None:
                success |= explore_result.success

        won = self.box_alive.sum(dim=1) == 0
        return GpuPushStepResult(
            valid=valid,
            moved=moved,
            won=won,
            low_steps=low_steps,
            success=success,
        )

    def build_state_obs(self, include_map_layout: bool = True) -> torch.Tensor:
        state_dim = STATE_DIM_WITH_MAP if include_map_layout else STATE_DIM
        obs = torch.zeros(
            (self.batch_size, state_dim),
            dtype=torch.float32,
            device=self.device,
        )
        obs[:, 0] = (self.car_pos[:, 0].float() + 0.5) / MAP_COLS
        obs[:, 1] = (self.car_pos[:, 1].float() + 0.5) / MAP_ROWS

        known_boxes = self._box_known_mask()
        known_targets = self._target_known_mask()
        box_target_match = (
            self.box_alive.unsqueeze(2)
            & self.target_alive.unsqueeze(1)
            & (self.box_ids.unsqueeze(2) == self.target_ids.unsqueeze(1))
        )

        box_feat = torch.zeros((self.batch_size, MAX_BOXES, 5), dtype=torch.float32, device=self.device)
        box_feat[..., 0] = torch.where(
            self.box_alive,
            (self.box_pos[..., 0].float() + 0.5) / MAP_COLS,
            torch.zeros_like(self.box_pos[..., 0], dtype=torch.float32),
        )
        box_feat[..., 1] = torch.where(
            self.box_alive,
            (self.box_pos[..., 1].float() + 0.5) / MAP_ROWS,
            torch.zeros_like(self.box_pos[..., 1], dtype=torch.float32),
        )
        box_feat[..., 2] = torch.where(
            known_boxes,
            (self.box_ids.float() + 1.0) / 10.0,
            torch.zeros_like(self.box_ids, dtype=torch.float32),
        )
        box_feat[..., 3] = known_boxes.float()
        box_feat[..., 4] = torch.where(known_boxes, box_target_match.any(dim=2).float(), torch.zeros_like(box_feat[..., 3]))
        obs[:, 2:2 + MAX_BOXES * 5] = box_feat.view(self.batch_size, -1)

        target_offset = 2 + MAX_BOXES * 5
        target_feat = torch.zeros((self.batch_size, MAX_TARGETS, 4), dtype=torch.float32, device=self.device)
        target_feat[..., 0] = torch.where(
            self.target_alive,
            (self.target_pos[..., 0].float() + 0.5) / MAP_COLS,
            torch.zeros_like(self.target_pos[..., 0], dtype=torch.float32),
        )
        target_feat[..., 1] = torch.where(
            self.target_alive,
            (self.target_pos[..., 1].float() + 0.5) / MAP_ROWS,
            torch.zeros_like(self.target_pos[..., 1], dtype=torch.float32),
        )
        target_feat[..., 2] = torch.where(
            known_targets,
            (self.target_ids.float() + 1.0) / 10.0,
            torch.zeros_like(self.target_ids, dtype=torch.float32),
        )
        target_feat[..., 3] = known_targets.float()
        obs[:, target_offset:target_offset + MAX_TARGETS * 4] = target_feat.view(self.batch_size, -1)

        bomb_offset = target_offset + MAX_TARGETS * 4
        bomb_feat = torch.zeros((self.batch_size, MAX_BOMBS, 2), dtype=torch.float32, device=self.device)
        bomb_feat[..., 0] = torch.where(
            self.bomb_alive,
            (self.bomb_pos[..., 0].float() + 0.5) / MAP_COLS,
            torch.zeros_like(self.bomb_pos[..., 0], dtype=torch.float32),
        )
        bomb_feat[..., 1] = torch.where(
            self.bomb_alive,
            (self.bomb_pos[..., 1].float() + 0.5) / MAP_ROWS,
            torch.zeros_like(self.bomb_pos[..., 1], dtype=torch.float32),
        )
        obs[:, bomb_offset:bomb_offset + MAX_BOMBS * 2] = bomb_feat.view(self.batch_size, -1)

        total = self.total_pairs.clamp(min=1).float()
        progress_offset = bomb_offset + MAX_BOMBS * 2
        obs[:, progress_offset + 0] = known_boxes.sum(dim=1).float() / total
        obs[:, progress_offset + 1] = known_targets.sum(dim=1).float() / total
        obs[:, progress_offset + 2] = 1.0 - self.box_alive.sum(dim=1).float() / total
        obs[:, progress_offset + 3] = self.step_count.float() / self.max_steps.clamp(min=1).float()

        matched_exists, _, matched_target_pos = self._matching_target_tables()
        dist_offset = progress_offset + 4
        dist = (
            (self.box_pos[..., 0] - matched_target_pos[..., 0]).abs()
            + (self.box_pos[..., 1] - matched_target_pos[..., 1]).abs()
        ).float() / 26.0
        obs[:, dist_offset:dist_offset + MAX_BOXES] = torch.where(
            known_boxes & matched_exists,
            dist,
            torch.full_like(dist, -1.0),
        )

        if include_map_layout:
            obs[:, dist_offset + MAX_BOXES:] = self.walls.reshape(self.batch_size, -1).float()
        return obs

    def build_oracle_obs(self) -> torch.Tensor:
        obs = torch.zeros(
            (self.batch_size, STATE_DIM_WITH_MAP),
            dtype=torch.float32,
            device=self.device,
        )
        obs[:, 0] = (self.car_pos[:, 0].float() + 0.5) / MAP_COLS
        obs[:, 1] = (self.car_pos[:, 1].float() + 0.5) / MAP_ROWS

        box_target_match = (
            self.box_alive.unsqueeze(2)
            & self.target_alive.unsqueeze(1)
            & (self.box_ids.unsqueeze(2) == self.target_ids.unsqueeze(1))
        )

        box_feat = torch.zeros((self.batch_size, MAX_BOXES, 5), dtype=torch.float32, device=self.device)
        box_feat[..., 0] = torch.where(
            self.box_alive,
            (self.box_pos[..., 0].float() + 0.5) / MAP_COLS,
            torch.zeros_like(self.box_pos[..., 0], dtype=torch.float32),
        )
        box_feat[..., 1] = torch.where(
            self.box_alive,
            (self.box_pos[..., 1].float() + 0.5) / MAP_ROWS,
            torch.zeros_like(self.box_pos[..., 1], dtype=torch.float32),
        )
        box_feat[..., 2] = torch.where(
            self.box_alive,
            (self.box_ids.float() + 1.0) / 10.0,
            torch.zeros_like(self.box_ids, dtype=torch.float32),
        )
        box_feat[..., 3] = self.box_alive.float()
        box_feat[..., 4] = box_target_match.any(dim=2).float()
        obs[:, 2:2 + MAX_BOXES * 5] = box_feat.view(self.batch_size, -1)

        target_offset = 2 + MAX_BOXES * 5
        target_feat = torch.zeros((self.batch_size, MAX_TARGETS, 4), dtype=torch.float32, device=self.device)
        target_feat[..., 0] = torch.where(
            self.target_alive,
            (self.target_pos[..., 0].float() + 0.5) / MAP_COLS,
            torch.zeros_like(self.target_pos[..., 0], dtype=torch.float32),
        )
        target_feat[..., 1] = torch.where(
            self.target_alive,
            (self.target_pos[..., 1].float() + 0.5) / MAP_ROWS,
            torch.zeros_like(self.target_pos[..., 1], dtype=torch.float32),
        )
        target_feat[..., 2] = torch.where(
            self.target_alive,
            (self.target_ids.float() + 1.0) / 10.0,
            torch.zeros_like(self.target_ids, dtype=torch.float32),
        )
        target_feat[..., 3] = self.target_alive.float()
        obs[:, target_offset:target_offset + MAX_TARGETS * 4] = target_feat.view(self.batch_size, -1)

        bomb_offset = target_offset + MAX_TARGETS * 4
        bomb_feat = torch.zeros((self.batch_size, MAX_BOMBS, 2), dtype=torch.float32, device=self.device)
        bomb_feat[..., 0] = torch.where(
            self.bomb_alive,
            (self.bomb_pos[..., 0].float() + 0.5) / MAP_COLS,
            torch.zeros_like(self.bomb_pos[..., 0], dtype=torch.float32),
        )
        bomb_feat[..., 1] = torch.where(
            self.bomb_alive,
            (self.bomb_pos[..., 1].float() + 0.5) / MAP_ROWS,
            torch.zeros_like(self.bomb_pos[..., 1], dtype=torch.float32),
        )
        obs[:, bomb_offset:bomb_offset + MAX_BOMBS * 2] = bomb_feat.view(self.batch_size, -1)

        remaining = self.box_alive.sum(dim=1).float()
        total = self.total_pairs.clamp(min=1).float()
        offset = bomb_offset + MAX_BOMBS * 2
        obs[:, offset + 0] = (self.total_pairs > 0).float()
        obs[:, offset + 1] = (self.total_pairs > 0).float()
        obs[:, offset + 2] = 1.0 - remaining / total
        obs[:, offset + 3] = self.step_count.float() / self.max_steps.clamp(min=1).float()
        offset += 4

        target_index = box_target_match.float().argmax(dim=2)
        gather_index = target_index.unsqueeze(-1)
        matched_target_pos = torch.gather(
            self.target_pos,
            1,
            gather_index.expand(self.batch_size, MAX_BOXES, 2),
        )
        matched = box_target_match.any(dim=2)
        dist = (
            (self.box_pos[..., 0] - matched_target_pos[..., 0]).abs()
            + (self.box_pos[..., 1] - matched_target_pos[..., 1]).abs()
        ).float() / 26.0
        obs[:, offset:offset + MAX_BOXES] = torch.where(
            self.box_alive & matched,
            dist,
            torch.full_like(dist, -1.0),
        )
        offset += MAX_BOXES

        obs[:, offset:] = self.walls.reshape(self.batch_size, -1).float()
        return obs

    def state_hash(self) -> torch.Tensor:
        if self._state_hash_version == self._state_version and self._state_hash_cache is not None:
            return self._state_hash_cache

        car_linear = (self.car_pos[:, 1] * self.width + self.car_pos[:, 0] + 1).long()
        box_slots = torch.arange(MAX_BOXES, device=self.device, dtype=torch.long).unsqueeze(0)
        target_slots = torch.arange(MAX_TARGETS, device=self.device, dtype=torch.long).unsqueeze(0)
        bomb_slots = torch.arange(MAX_BOMBS, device=self.device, dtype=torch.long).unsqueeze(0)

        box_linear = (self.box_pos[..., 1].clamp(min=0) * self.width + self.box_pos[..., 0].clamp(min=0) + 1).long()
        target_linear = (self.target_pos[..., 1].clamp(min=0) * self.width + self.target_pos[..., 0].clamp(min=0) + 1).long()
        bomb_linear = (self.bomb_pos[..., 1].clamp(min=0) * self.width + self.bomb_pos[..., 0].clamp(min=0) + 1).long()
        wall_weights = torch.arange(1, self.width * self.height + 1, device=self.device, dtype=torch.long)
        wall_term = (self.walls.view(self.batch_size, -1).long() * wall_weights.unsqueeze(0)).sum(dim=1)

        box_term = (
            self.box_alive.long()
            * (box_linear * 1315423911 + (self.box_ids.clamp(min=0) + 1) * 2654435761 + box_slots * 97531)
        ).sum(dim=1)
        target_term = (
            self.target_alive.long()
            * (target_linear * 1140071481 + (self.target_ids.clamp(min=0) + 1) * 40503 + target_slots * 8191)
        ).sum(dim=1)
        bomb_term = (
            self.bomb_alive.long()
            * (bomb_linear * 780291637 + bomb_slots * 15485863)
        ).sum(dim=1)

        state_hash = (
            car_linear
            ^ (box_term << 1)
            ^ (target_term << 7)
            ^ (bomb_term << 13)
            ^ (wall_term << 17)
        )
        self._state_hash_cache = state_hash
        self._state_hash_version = self._state_version
        return state_hash

    def debug_state(self, index: int) -> Dict[str, object]:
        def collect(pos: torch.Tensor, ids: Optional[torch.Tensor], alive: torch.Tensor) -> List[Dict[str, int]]:
            rows: List[Dict[str, int]] = []
            for slot in range(alive.shape[0]):
                if not bool(alive[slot].item()):
                    continue
                item = {
                    "col": int(pos[slot, 0].item()),
                    "row": int(pos[slot, 1].item()),
                }
                if ids is not None:
                    item["id"] = int(ids[slot].item())
                rows.append(item)
            return rows

        return {
            "car": tuple(int(v) for v in self.car_pos[index].tolist()),
            "boxes": collect(self.box_pos[index], self.box_ids[index], self.box_alive[index]),
            "targets": collect(self.target_pos[index], self.target_ids[index], self.target_alive[index]),
            "bombs": collect(self.bomb_pos[index], None, self.bomb_alive[index]),
            "seen_box": [idx for idx in range(MAX_BOXES) if bool(self.seen_box[index, idx].item())],
            "seen_target": [idx for idx in range(MAX_TARGETS) if bool(self.seen_target[index, idx].item())],
            "won": bool((self.box_alive[index].sum() == 0).item()),
        }
