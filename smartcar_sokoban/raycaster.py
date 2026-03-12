"""3D Raycasting 引擎 — Wolfenstein 3D 风格第一人称视角.

所有可见物体（墙壁、箱子、目的地箱子、炸弹）均通过射线投射渲染为 3D 方块。
"""

from __future__ import annotations

import math
import os
import random as _rng
from typing import List, Optional, Tuple

import pygame

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.map_loader import BoxInfo, TargetInfo, BombInfo
from smartcar_sokoban.paths import PROJECT_ROOT


class Raycaster:
    """Raycasting 渲染器, 输出 640×480 的 Pygame Surface."""

    def __init__(self, config: Optional[GameConfig] = None, base_dir: str = ""):
        self.cfg = config or GameConfig()
        self.base_dir = base_dir or str(PROJECT_ROOT)
        self.width = self.cfg.view_width    # 640
        self.height = self.cfg.view_height  # 480

        # 贴图缓存
        self._wall_texture: Optional[pygame.Surface] = None
        self._texture_cache: dict[str, pygame.Surface] = {}

        # 预计算
        self.fov_rad = math.radians(self.cfg.fov)
        self.half_fov = self.fov_rad / 2.0
        # 投影平面距离 (使得 1 单位宽 = 1 单位高 的方块看起来是正方形)
        self.proj_dist = (self.width / 2.0) / math.tan(self.half_fov)

    # ── 公共接口 ──────────────────────────────────────────

    def render(self, surface: pygame.Surface,
               car_x: float, car_y: float, car_angle: float,
               grid: List[List[int]],
               boxes: List[BoxInfo],
               targets: List[TargetInfo],
               bombs: List[BombInfo]):
        """渲染 3D 第一人称视角到给定 surface."""
        # 天空 + 地面
        surface.fill((200, 200, 200))  # 天空灰白
        pygame.draw.rect(surface, self.cfg.color_floor,
                         (0, self.height // 2, self.width, self.height // 2))

        wall_tex = self._get_wall_texture()

        # 收集所有可射线检测的方块物体 (AABB)
        # 每个元素: (min_x, min_y, max_x, max_y, texture, fallback_color)
        aabb_objects: List[Tuple[float, float, float, float,
                                 Optional[pygame.Surface], tuple]] = []

        for box in boxes:
            tex = self._get_texture(box.image_path) if box.image_path else None
            aabb_objects.append((
                box.x - 0.5, box.y - 0.5, box.x + 0.5, box.y + 0.5,
                tex, self.cfg.color_box
            ))

        for target in targets:
            tex = self._get_texture(target.image_path) \
                if target.image_path else None
            aabb_objects.append((
                target.x - 0.5, target.y - 0.5, target.x + 0.5, target.y + 0.5,
                tex, self.cfg.color_target
            ))

        for bomb in bombs:
            aabb_objects.append((
                bomb.x - 0.5, bomb.y - 0.5, bomb.x + 0.5, bomb.y + 0.5,
                None, self.cfg.color_bomb
            ))

        # 逐列投射射线
        for x in range(self.width):
            # 计算射线角度
            ray_screen_x = (x / self.width - 0.5) * 2.0  # -1..1
            ray_angle = car_angle + math.atan(
                ray_screen_x * math.tan(self.half_fov))

            ray_dx = math.cos(ray_angle)
            ray_dy = math.sin(ray_angle)
            cos_diff = math.cos(ray_angle - car_angle)  # 鱼眼校正

            # --- 1. 墙壁射线 (DDA) ---
            wall_dist, wall_side, wall_tex_x = self._cast_wall_ray(
                car_x, car_y, ray_dx, ray_dy, grid
            )
            wall_perp = wall_dist * cos_diff if wall_dist < 1e9 else 1e9

            # --- 2. 物体射线 (ray-AABB) ---
            best_obj_dist = 1e9
            best_obj_side = 0
            best_obj_tex_x = 0.0
            best_obj_tex: Optional[pygame.Surface] = None
            best_obj_color: tuple = (0, 0, 0)

            for (ax0, ay0, ax1, ay1, obj_tex, obj_color) in aabb_objects:
                t, side, tex_ratio = self._ray_aabb_intersect(
                    car_x, car_y, ray_dx, ray_dy, ax0, ay0, ax1, ay1
                )
                if 0 < t < best_obj_dist:
                    best_obj_dist = t
                    best_obj_side = side
                    best_obj_tex_x = tex_ratio
                    best_obj_tex = obj_tex
                    best_obj_color = obj_color

            obj_perp = best_obj_dist * cos_diff if best_obj_dist < 1e9 else 1e9

            # --- 3. 取最近的碰撞  ---
            if wall_perp <= 0.01 and obj_perp <= 0.01:
                continue

            if wall_perp <= obj_perp and wall_perp > 0.01:
                # 渲染墙壁
                self._draw_column(surface, x, wall_perp, wall_side,
                                  wall_tex_x, wall_tex, None)
            elif obj_perp < wall_perp and obj_perp > 0.01:
                # 渲染物体
                self._draw_column(surface, x, obj_perp, best_obj_side,
                                  best_obj_tex_x, best_obj_tex,
                                  best_obj_color)

    # ── 绘制单列 ──────────────────────────────────────────

    def _draw_column(self, surface: pygame.Surface, x: int,
                     perp_dist: float, side: int, tex_x_ratio: float,
                     texture: Optional[pygame.Surface],
                     fallback_color: Optional[tuple]):
        """绘制一列墙壁/物体."""
        line_height = int(self.proj_dist / perp_dist)
        draw_start = max(0, self.height // 2 - line_height // 2)
        draw_end = min(self.height, self.height // 2 + line_height // 2)
        actual_height = draw_end - draw_start

        if actual_height <= 0:
            return

        if texture is not None:
            tex_w = texture.get_width()
            tex_h = texture.get_height()
            tex_x = int(tex_x_ratio * tex_w) % tex_w
            # 从贴图取竖条
            column = texture.subsurface((tex_x, 0, 1, tex_h))
            scaled = pygame.transform.scale(column, (1, actual_height))
            # 暗面处理
            if side == 1:
                dark = pygame.Surface((1, actual_height))
                dark.fill((0, 0, 0))
                dark.set_alpha(60)
                scaled.blit(dark, (0, 0))
            surface.blit(scaled, (x, draw_start))
        elif fallback_color is not None:
            # 纯色 fallback
            color = fallback_color
            if side == 1:
                color = tuple(max(0, c - 40) for c in color)
            pygame.draw.line(surface, color, (x, draw_start), (x, draw_end))
        else:
            # 无贴图无颜色 (不应发生)
            color = (100, 100, 100) if side == 0 else (80, 80, 80)
            pygame.draw.line(surface, color, (x, draw_start), (x, draw_end))

    # ── DDA 墙壁射线投射 ──────────────────────────────────

    def _cast_wall_ray(self, ox: float, oy: float,
                       dx: float, dy: float,
                       grid: List[List[int]]
                       ) -> Tuple[float, int, float]:
        """DDA 射线投射 (仅检测墙壁). 返回 (距离, 碰撞面, 贴图x比例)."""
        map_x = int(ox)
        map_y = int(oy)

        # 避免除零
        if abs(dx) < 1e-10:
            dx = 1e-10
        if abs(dy) < 1e-10:
            dy = 1e-10

        delta_dist_x = abs(1.0 / dx)
        delta_dist_y = abs(1.0 / dy)

        if dx < 0:
            step_x = -1
            side_dist_x = (ox - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - ox) * delta_dist_x

        if dy < 0:
            step_y = -1
            side_dist_y = (oy - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - oy) * delta_dist_y

        side = 0
        max_steps = 64

        for _ in range(max_steps):
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1

            if map_y < 0 or map_y >= len(grid) or \
               map_x < 0 or map_x >= len(grid[0]):
                return (1e9, 0, 0.0)

            if grid[map_y][map_x] == 1:
                break
        else:
            return (1e9, 0, 0.0)

        # 计算精确距离
        if side == 0:
            perp_dist = (map_x - ox + (1 - step_x) / 2.0) / dx
        else:
            perp_dist = (map_y - oy + (1 - step_y) / 2.0) / dy

        # 贴图 X 坐标
        if side == 0:
            wall_x = oy + perp_dist * dy
        else:
            wall_x = ox + perp_dist * dx
        wall_x -= math.floor(wall_x)

        return (abs(perp_dist), side, wall_x)

    # ── Ray-AABB 交叉检测 ─────────────────────────────────

    def _ray_aabb_intersect(self, ox: float, oy: float,
                            dx: float, dy: float,
                            ax0: float, ay0: float,
                            ax1: float, ay1: float
                            ) -> Tuple[float, int, float]:
        """射线与 AABB 交叉检测.

        返回: (距离t, 碰撞面side, 贴图x比例)
        side: 0=X面(左/右), 1=Y面(上/下)
        距离 < 0 或 inf 表示未碰撞.
        """
        # 避免除零
        if abs(dx) < 1e-10:
            dx = 1e-10 if dx >= 0 else -1e-10
        if abs(dy) < 1e-10:
            dy = 1e-10 if dy >= 0 else -1e-10

        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        # X 方向 slab
        if inv_dx >= 0:
            tx_min = (ax0 - ox) * inv_dx
            tx_max = (ax1 - ox) * inv_dx
        else:
            tx_min = (ax1 - ox) * inv_dx
            tx_max = (ax0 - ox) * inv_dx

        # Y 方向 slab
        if inv_dy >= 0:
            ty_min = (ay0 - oy) * inv_dy
            ty_max = (ay1 - oy) * inv_dy
        else:
            ty_min = (ay1 - oy) * inv_dy
            ty_max = (ay0 - oy) * inv_dy

        # 区间交集
        t_enter = max(tx_min, ty_min)
        t_exit = min(tx_max, ty_max)

        if t_enter > t_exit or t_exit < 0:
            return (1e9, 0, 0.0)  # 未命中

        t = t_enter if t_enter > 0.01 else t_exit
        if t < 0.01:
            return (1e9, 0, 0.0)  # 在物体内部或太近

        # 确定碰撞面和贴图坐标
        hit_x = ox + dx * t
        hit_y = oy + dy * t

        if t_enter == tx_min:
            # 碰到 X 面 (左或右)
            side = 0
            tex_ratio = hit_y - ay0
        else:
            # 碰到 Y 面 (上或下)
            side = 1
            tex_ratio = hit_x - ax0

        # 归一化贴图坐标到 [0, 1)
        box_w = ax1 - ax0
        if box_w > 0:
            tex_ratio = tex_ratio / box_w
        tex_ratio = tex_ratio - math.floor(tex_ratio)

        return (t, side, tex_ratio)

    # ── 贴图管理 ──────────────────────────────────────────

    def _get_wall_texture(self) -> Optional[pygame.Surface]:
        """获取墙壁石砖贴图."""
        if self._wall_texture is not None:
            return self._wall_texture

        # 尝试加载 textures/wall.png
        tex_path = os.path.join(self.base_dir, self.cfg.textures_dir, "wall.png")
        if os.path.isfile(tex_path):
            self._wall_texture = pygame.image.load(tex_path).convert()
            return self._wall_texture

        # 生成程序化石砖贴图
        self._wall_texture = self._generate_stone_texture(64, 64)
        return self._wall_texture

    def _generate_stone_texture(self, w: int, h: int) -> pygame.Surface:
        """程序化生成深色石砖贴图."""
        surf = pygame.Surface((w, h))
        for y in range(h):
            for x in range(w):
                base = 70
                noise = _rng.randint(-15, 15)
                brick_h = h // 4
                brick_w = w // 2
                row = y // brick_h
                offset = (brick_w // 2) if row % 2 else 0
                bx = (x + offset) % brick_w
                by = y % brick_h

                if by < 1 or bx < 1:
                    val = 30  # 深色砖缝
                else:
                    val = max(20, min(120, base + noise))
                surf.set_at((x, y), (val, val, val))
        return surf

    def _get_texture(self, path: str) -> Optional[pygame.Surface]:
        """加载并缓存贴图."""
        if not path or not os.path.isfile(path):
            return None
        if path in self._texture_cache:
            return self._texture_cache[path]
        try:
            tex = pygame.image.load(path).convert_alpha()
            self._texture_cache[path] = tex
            return tex
        except Exception:
            return None
