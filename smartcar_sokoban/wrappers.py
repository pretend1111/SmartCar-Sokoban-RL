"""动作空间包装器."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscreteWrapper(gym.ActionWrapper):
    """将连续动作空间包装为离散动作空间.

    离散动作:
        0 = 前进
        1 = 后退
        2 = 左移
        3 = 右移
        4 = 左转
        5 = 右转
        6 = 不动
    """

    # 每个离散动作对应的 [forward, strafe, turn]
    ACTION_MAP = {
        0: np.array([1.0, 0.0, 0.0], dtype=np.float32),   # 前进
        1: np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # 后退
        2: np.array([0.0, -1.0, 0.0], dtype=np.float32),  # 左移
        3: np.array([0.0, 1.0, 0.0], dtype=np.float32),   # 右移
        4: np.array([0.0, 0.0, -1.0], dtype=np.float32),  # 左转
        5: np.array([0.0, 0.0, 1.0], dtype=np.float32),   # 右转
        6: np.array([0.0, 0.0, 0.0], dtype=np.float32),   # 不动
    }

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(7)

    def action(self, action: int) -> np.ndarray:
        return self.ACTION_MAP.get(action,
                                   np.zeros(3, dtype=np.float32))
