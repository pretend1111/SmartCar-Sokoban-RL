"""地图解析与配对分配."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.paths import PROJECT_ROOT


# ── 地图符号常量 ──────────────────────────────────────────
WALL = '#'
AIR = '-'
TARGET = '.'
BOX = '$'
BOMB = '*'

SYMBOL_TO_CELL = {
    WALL: 1,
    AIR: 0,
    TARGET: 0,   # 目的地不影响通行
    BOX: 0,      # 箱子位置底下是空地
    BOMB: 0,     # 炸弹位置底下是空地
}


@dataclass
class BoxInfo:
    """箱子信息."""
    x: float                  # 中心 x（列）
    y: float                  # 中心 y（行）
    class_id: int             # 类别编号 0-9
    image_path: str = ""      # 选定的贴图路径


@dataclass
class TargetInfo:
    """目的地箱子信息."""
    x: float                  # 中心 x（列）
    y: float                  # 中心 y（行）
    num_id: int               # 数字编号 0-9
    image_path: str = ""      # 对应的数字贴图路径


@dataclass
class BombInfo:
    """炸弹信息."""
    x: float                  # 中心 x
    y: float                  # 中心 y


@dataclass
class MapData:
    """解析后的完整地图数据."""
    grid: List[List[int]] = field(default_factory=list)   # 12×16, 1=墙 0=空
    boxes: List[BoxInfo] = field(default_factory=list)
    targets: List[TargetInfo] = field(default_factory=list)
    bombs: List[BombInfo] = field(default_factory=list)
    car_x: float = 0.0
    car_y: float = 0.0


class MapLoader:
    """加载并解析地图 txt 文件."""

    def __init__(self, config: Optional[GameConfig] = None, base_dir: str = ""):
        self.cfg = config or GameConfig()
        self.base_dir = base_dir or str(PROJECT_ROOT)

    # ── 公共接口 ──────────────────────────────────────────

    def load(self, map_path: str) -> MapData:
        """加载地图文件并返回 MapData."""
        full_path = os.path.join(self.base_dir, map_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n').rstrip('\r') for line in f.readlines()]

        # 验证尺寸
        assert len(lines) == self.cfg.map_rows, \
            f"地图行数应为 {self.cfg.map_rows}，实际 {len(lines)}"
        for i, line in enumerate(lines):
            assert len(line) == self.cfg.map_cols, \
                f"第 {i} 行列数应为 {self.cfg.map_cols}，实际 {len(line)}"

        # 解析网格与元素位置
        grid: List[List[int]] = []
        box_positions: List[Tuple[int, int]] = []
        target_positions: List[Tuple[int, int]] = []
        bomb_positions: List[Tuple[int, int]] = []

        for row_idx, line in enumerate(lines):
            grid_row: List[int] = []
            for col_idx, ch in enumerate(line):
                grid_row.append(SYMBOL_TO_CELL.get(ch, 0))
                if ch == BOX:
                    box_positions.append((col_idx, row_idx))
                elif ch == TARGET:
                    target_positions.append((col_idx, row_idx))
                elif ch == BOMB:
                    bomb_positions.append((col_idx, row_idx))
            grid.append(grid_row)

        # 验证箱子和目的地数量匹配
        assert len(box_positions) == len(target_positions), \
            f"箱子数 ({len(box_positions)}) != 目的地数 ({len(target_positions)})"

        n = len(box_positions)

        # 随机分配编号
        ids = list(range(n))
        random.shuffle(ids)
        box_ids = ids[:]

        random.shuffle(ids)
        target_ids = ids[:]

        # 构建 BoxInfo 列表
        boxes: List[BoxInfo] = []
        for i, (cx, cy) in enumerate(box_positions):
            cid = box_ids[i]
            img_path = self._pick_class_image(cid)
            boxes.append(BoxInfo(
                x=cx + 0.5, y=cy + 0.5,
                class_id=cid, image_path=img_path
            ))

        # 构建 TargetInfo 列表
        targets: List[TargetInfo] = []
        for i, (cx, cy) in enumerate(target_positions):
            nid = target_ids[i]
            img_path = self._get_num_image(nid)
            targets.append(TargetInfo(
                x=cx + 0.5, y=cy + 0.5,
                num_id=nid, image_path=img_path
            ))

        # 构建 BombInfo 列表
        bombs = [BombInfo(x=cx + 0.5, y=cy + 0.5)
                 for cx, cy in bomb_positions]

        # 计算车的初始位置
        car_x, car_y = self._calc_car_spawn(grid)

        return MapData(
            grid=grid, boxes=boxes, targets=targets, bombs=bombs,
            car_x=car_x, car_y=car_y
        )

    def load_from_string(self, map_string: str) -> MapData:
        """从字符串直接解析地图（无文件I/O，RL训练用）."""
        lines = [l for l in map_string.split('\n') if l.strip()]

        grid = []
        box_positions = []
        target_positions = []
        bomb_positions = []

        for row_idx, line in enumerate(lines):
            grid_row = []
            for col_idx, ch in enumerate(line):
                grid_row.append(SYMBOL_TO_CELL.get(ch, 0))
                if ch == BOX:
                    box_positions.append((col_idx, row_idx))
                elif ch == TARGET:
                    target_positions.append((col_idx, row_idx))
                elif ch == BOMB:
                    bomb_positions.append((col_idx, row_idx))
            grid.append(grid_row)

        n = len(box_positions)
        ids = list(range(n))
        random.shuffle(ids)
        box_ids = ids[:]
        random.shuffle(ids)
        target_ids = ids[:]

        boxes = [BoxInfo(x=cx + 0.5, y=cy + 0.5, class_id=box_ids[i])
                 for i, (cx, cy) in enumerate(box_positions)]
        targets = [TargetInfo(x=cx + 0.5, y=cy + 0.5, num_id=target_ids[i])
                   for i, (cx, cy) in enumerate(target_positions)]
        bombs = [BombInfo(x=cx + 0.5, y=cy + 0.5)
                 for cx, cy in bomb_positions]

        car_x, car_y = self._calc_car_spawn(grid)
        return MapData(grid=grid, boxes=boxes, targets=targets, bombs=bombs,
                       car_x=car_x, car_y=car_y)

    # ── 内部方法 ──────────────────────────────────────────

    def _calc_car_spawn(self, grid: List[List[int]]) -> Tuple[float, float]:
        """计算车的初始位置：第2列或倒数第2列的中间行."""
        mid_y = self.cfg.map_rows / 2.0   # 6.0 → 第6/7行交界

        # 优先尝试第2列（索引1），如果被墙挡就用倒数第2列（索引14）
        col_candidates = [1, self.cfg.map_cols - 2]
        for col in col_candidates:
            row_check = int(mid_y)
            if 0 <= row_check < self.cfg.map_rows and grid[row_check][col] == 0:
                return col + 0.5, mid_y
        # fallback
        return col_candidates[0] + 0.5, mid_y

    def _pick_class_image(self, class_id: int) -> str:
        """从 image_class 的对应类别文件夹中随机选取一张图."""
        class_dir = os.path.join(self.base_dir, self.cfg.image_class_dir)
        if not os.path.isdir(class_dir):
            return ""

        # 找到编号匹配的子目录（如 "00mickey_mouse"）
        prefix = f"{class_id:02d}"
        for dirname in sorted(os.listdir(class_dir)):
            if dirname.startswith(prefix):
                folder = os.path.join(class_dir, dirname)
                images = [f for f in os.listdir(folder)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))]
                if images:
                    return os.path.join(folder, random.choice(images))
        return ""

    def _get_num_image(self, num_id: int) -> str:
        """获取 image_num 中对应编号的数字图路径."""
        num_dir = os.path.join(self.base_dir, self.cfg.image_num_dir)
        filename = f"{num_id:02d}.jpg"
        path = os.path.join(num_dir, filename)
        return path if os.path.isfile(path) else ""
