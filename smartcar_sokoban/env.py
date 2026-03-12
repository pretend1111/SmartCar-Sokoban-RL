"""Gymnasium 环境接口."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.renderer import Renderer
from smartcar_sokoban.paths import PROJECT_ROOT


class SokobanEnv(gym.Env):
    """推箱子 Gymnasium 环境.

    观测模式 (obs_mode):
        "matrix"  — 结构化状态 dict
        "pixel"   — 渲染后的像素数组
        "both"    — 两者合并

    动作空间:
        Box(low=-1, high=1, shape=(3,)) — [forward, strafe, turn]
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self,
                 config: Optional[GameConfig] = None,
                 map_path: str = "assets/maps/map1.txt",
                 render_mode: Optional[str] = None,
                 base_dir: str = "",
                 dt: float = 1.0 / 60.0):
        super().__init__()

        self.cfg = config or GameConfig()
        self.base_dir = base_dir or str(PROJECT_ROOT)
        self.map_path = map_path
        self._dt = dt
        self._render_mode = render_mode

        self.engine = GameEngine(self.cfg, self.base_dir)
        self._renderer: Optional[Renderer] = None

        # 动作空间: 连续控制 [forward, strafe, turn]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # 观测空间（根据模式）
        if self.cfg.obs_mode == "pixel":
            h = self.cfg.view_height * (2 if self.cfg.render_mode == "full" else 1)
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(h, self.cfg.view_width, 3),
                dtype=np.uint8
            )
        elif self.cfg.obs_mode == "matrix":
            self.observation_space = spaces.Dict({
                "grid": spaces.Box(0, 1,
                                   shape=(self.cfg.map_rows, self.cfg.map_cols),
                                   dtype=np.int8),
                "car_pos": spaces.Box(-50, 50, shape=(2,), dtype=np.float32),
                "car_angle": spaces.Box(-math.pi, math.pi,
                                        shape=(1,), dtype=np.float32),
                "boxes": spaces.Box(-50, 50, shape=(10, 3), dtype=np.float32),
                "targets": spaces.Box(-50, 50, shape=(10, 3), dtype=np.float32),
            })
        else:  # "both"
            h = self.cfg.view_height * (2 if self.cfg.render_mode == "full" else 1)
            self.observation_space = spaces.Dict({
                "pixel": spaces.Box(0, 255,
                                    shape=(h, self.cfg.view_width, 3),
                                    dtype=np.uint8),
                "grid": spaces.Box(0, 1,
                                   shape=(self.cfg.map_rows, self.cfg.map_cols),
                                   dtype=np.int8),
                "car_pos": spaces.Box(-50, 50, shape=(2,), dtype=np.float32),
                "car_angle": spaces.Box(-math.pi, math.pi,
                                        shape=(1,), dtype=np.float32),
            })

    def reset(self, *, seed=None, options=None
              ) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)
        if options and "map_path" in options:
            self.map_path = options["map_path"]

        self.engine.reset(self.map_path)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray
             ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        forward = float(action[0])
        strafe = float(action[1])
        turn = float(action[2])

        old_score = self.engine.state.score
        state = self.engine.step(forward, strafe, turn, self._dt)

        # Reward: +1 per successful pairing
        reward = float(state.score - old_score)

        terminated = state.won
        truncated = False

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self._renderer is None:
            self._renderer = Renderer(self.cfg, self.base_dir)

        state = self.engine.get_state()
        fov_rays = self.engine.get_fov_rays() \
            if self.cfg.render_mode == "simple" else None

        pixels = self._renderer.render(state, fov_rays)

        if self._render_mode == "rgb_array":
            return pixels
        return None

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None

    # ── 内部 ──────────────────────────────────────────────

    def _get_obs(self) -> Any:
        state = self.engine.get_state()

        if self.cfg.obs_mode == "pixel":
            return self.render() if self._renderer else \
                np.zeros((self.cfg.view_height, self.cfg.view_width, 3),
                         dtype=np.uint8)

        matrix_obs = self._build_matrix_obs(state)

        if self.cfg.obs_mode == "matrix":
            return matrix_obs
        else:  # "both"
            pixel = self.render() if self._renderer else \
                np.zeros((self.cfg.view_height, self.cfg.view_width, 3),
                         dtype=np.uint8)
            return {**matrix_obs, "pixel": pixel}

    def _build_matrix_obs(self, state) -> Dict[str, np.ndarray]:
        grid = np.array(state.grid, dtype=np.int8)
        car_pos = np.array([state.car_x, state.car_y], dtype=np.float32)
        car_angle = np.array([state.car_angle], dtype=np.float32)

        # Boxes: [x, y, class_id] padded to 10
        boxes = np.zeros((10, 3), dtype=np.float32)
        for i, box in enumerate(state.boxes):
            if i >= 10:
                break
            boxes[i] = [box.x, box.y, box.class_id]

        targets = np.zeros((10, 3), dtype=np.float32)
        for i, tgt in enumerate(state.targets):
            if i >= 10:
                break
            targets[i] = [tgt.x, tgt.y, tgt.num_id]

        return {
            "grid": grid,
            "car_pos": car_pos,
            "car_angle": car_angle,
            "boxes": boxes,
            "targets": targets,
        }

    def _get_info(self) -> Dict[str, Any]:
        state = self.engine.get_state()
        return {
            "score": state.score,
            "total_pairs": state.total_pairs,
            "won": state.won,
            "remaining_boxes": len(state.boxes),
            "remaining_bombs": len(state.bombs),
        }
