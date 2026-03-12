"""渲染主控 — 组合 2D 俯视图 + 3D Raycasting."""

from __future__ import annotations

import math
import os
from typing import Optional, List, Tuple

import numpy as np
import pygame

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameState
from smartcar_sokoban.raycaster import Raycaster
from smartcar_sokoban.paths import PROJECT_ROOT


class Renderer:
    """Pygame 渲染器, 支持 full (3D+2D) 和 simple (仅增强2D) 两种模式."""

    def __init__(self, config: Optional[GameConfig] = None, base_dir: str = ""):
        self.cfg = config or GameConfig()
        self.base_dir = base_dir or str(PROJECT_ROOT)
        self.raycaster = Raycaster(self.cfg, self.base_dir)

        self.w = self.cfg.view_width     # 640
        self.h = self.cfg.view_height    # 480

        self._screen: Optional[pygame.Surface] = None
        self._3d_surface: Optional[pygame.Surface] = None
        self._2d_surface: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._initialized = False

    # ── 初始化 ────────────────────────────────────────────

    def init(self):
        """初始化 Pygame 窗口."""
        if self._initialized:
            return
        pygame.init()
        pygame.font.init()

        if self.cfg.render_mode == "full":
            self._screen = pygame.display.set_mode((self.w, self.h * 2))
            pygame.display.set_caption("推箱子 — Full Mode (3D + 2D)")
        else:
            self._screen = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("推箱子 — Simple Mode (2D)")

        self._3d_surface = pygame.Surface((self.w, self.h))
        self._2d_surface = pygame.Surface((self.w, self.h))
        self._font = pygame.font.SysFont("Arial", 16, bold=True)
        self._big_font = pygame.font.SysFont("Arial", 24, bold=True)
        self._initialized = True

    def switch_mode(self, mode: str):
        """切换渲染模式."""
        self.cfg.render_mode = mode
        self._initialized = False
        self.init()

    # ── 主渲染 ────────────────────────────────────────────

    def render(self, state: GameState,
               fov_rays: Optional[Tuple] = None) -> np.ndarray:
        """渲染一帧, 返回像素数组."""
        self.init()

        if self.cfg.render_mode == "full":
            # 3D 视角
            self._3d_surface.fill((0, 0, 0))
            self.raycaster.render(
                self._3d_surface,
                state.car_x, state.car_y, state.car_angle,
                state.grid, state.boxes, state.targets, state.bombs
            )
            self._screen.blit(self._3d_surface, (0, 0))

            # 2D 俯视图
            self._render_2d(self._2d_surface, state, fov_rays,
                            show_labels=False)
            self._screen.blit(self._2d_surface, (0, self.h))
        else:
            # Simple 模式: 仅增强 2D
            self._render_2d(self._2d_surface, state, fov_rays,
                            show_labels=True)
            self._screen.blit(self._2d_surface, (0, 0))

        # 通关提示
        if state.won:
            overlay = pygame.Surface(self._screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self._screen.blit(overlay, (0, 0))
            text = self._big_font.render("LEVEL COMPLETE!", True,
                                         (0, 255, 0))
            rect = text.get_rect(center=(self.w // 2,
                                         self._screen.get_height() // 2))
            self._screen.blit(text, rect)

        pygame.display.flip()

        # 返回像素数组
        return pygame.surfarray.array3d(self._screen).transpose(1, 0, 2)

    # ── 2D 俯视图渲染 ────────────────────────────────────

    def _render_2d(self, surface: pygame.Surface, state: GameState,
                   fov_rays: Optional[Tuple],
                   show_labels: bool):
        """渲染 2D 俯视图."""
        surface.fill(self.cfg.color_bg)

        cols = self.cfg.map_cols  # 16
        rows = self.cfg.map_rows  # 12

        # 计算格子大小 (保持宽高比)
        cell_w = self.w / cols    # 640/16 = 40
        cell_h = self.h / rows    # 480/12 = 40

        # 背景网格纹
        for r in range(rows):
            for c in range(cols):
                x = c * cell_w
                y = r * cell_h

                if state.grid[r][c] == 1:
                    color = self.cfg.color_wall
                else:
                    color = self.cfg.color_floor

                pygame.draw.rect(surface, color,
                                 (x, y, cell_w, cell_h))
                # 网格线
                pygame.draw.rect(surface, (40, 40, 40),
                                 (x, y, cell_w, cell_h), 1)

        # 目的地箱子（紫色）
        for i, target in enumerate(state.targets):
            tx = (target.x - 0.5) * cell_w
            ty = (target.y - 0.5) * cell_h
            pygame.draw.rect(surface, self.cfg.color_target,
                             (tx, ty, cell_w, cell_h))
            # 标注编号（如果已被看到 或 show_labels 且 full 模式显示全部）
            if show_labels and i in state.seen_target_ids:
                label = self._font.render(str(target.num_id), True,
                                          (255, 255, 255))
                label_rect = label.get_rect(
                    center=(tx + cell_w / 2, ty + cell_h / 2))
                surface.blit(label, label_rect)

        # 可移动箱子（黄色）
        for i, box in enumerate(state.boxes):
            bx = (box.x - 0.5) * cell_w
            by = (box.y - 0.5) * cell_h
            pygame.draw.rect(surface, self.cfg.color_box,
                             (bx, by, cell_w, cell_h))
            if show_labels and i in state.seen_box_ids:
                label = self._font.render(str(box.class_id), True,
                                          (0, 0, 0))
                label_rect = label.get_rect(
                    center=(bx + cell_w / 2, by + cell_h / 2))
                surface.blit(label, label_rect)

        # 炸弹（红色）
        for bomb in state.bombs:
            bx = (bomb.x - 0.5) * cell_w
            by = (bomb.y - 0.5) * cell_h
            pygame.draw.rect(surface, self.cfg.color_bomb,
                             (bx, by, cell_w, cell_h))

        # 车 (青色车头 + 绿色车尾)
        self._draw_car(surface, state, cell_w, cell_h)

        # FOV 射线（simple 模式）
        if show_labels and fov_rays is not None:
            self._draw_fov_lines(surface, state, fov_rays, cell_w, cell_h)

    def _draw_car(self, surface: pygame.Surface, state: GameState,
                  cell_w: float, cell_h: float):
        """绘制车（青色车头 + 绿色车尾）."""
        cx = state.car_x * cell_w
        cy = state.car_y * cell_h
        half_w = cell_w / 2
        half_h = cell_h / 2
        angle = state.car_angle

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        def rotate(lx, ly):
            return (lx * cos_a - ly * sin_a + cx,
                    lx * sin_a + ly * cos_a + cy)

        # 车的四个角（本地坐标，+X = 车头方向）
        #   tl = (-half, -half)   tr = (+half, -half)
        #   bl = (-half, +half)   br = (+half, +half)
        tl = rotate(-half_w, -half_h)
        tr = rotate(half_w, -half_h)
        br = rotate(half_w, half_h)
        bl = rotate(-half_w, half_h)

        # 中线分割点（x=0 的两个端点）
        mid_top = rotate(0, -half_h)
        mid_bot = rotate(0, half_h)

        # 车头（+X 半边）= 青色
        front_poly = [mid_top, tr, br, mid_bot]
        pygame.draw.polygon(surface, self.cfg.color_car_front, front_poly)

        # 车尾（-X 半边）= 绿色
        back_poly = [tl, mid_top, mid_bot, bl]
        pygame.draw.polygon(surface, self.cfg.color_car_back, back_poly)

        # 边框
        pygame.draw.polygon(surface, (0, 0, 0), [tl, tr, br, bl], 2)

    def _draw_fov_lines(self, surface: pygame.Surface, state: GameState,
                        fov_rays: Tuple, cell_w: float, cell_h: float):
        """绘制 FOV 视野锥射线."""
        left_end, right_end = fov_rays
        car_px = state.car_x * cell_w
        car_py = state.car_y * cell_h

        left_px = left_end[0] * cell_w
        left_py = left_end[1] * cell_h
        right_px = right_end[0] * cell_w
        right_py = right_end[1] * cell_h

        # 半透明线
        fov_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        pygame.draw.line(fov_surface, (255, 255, 255, 100),
                         (car_px, car_py), (left_px, left_py), 2)
        pygame.draw.line(fov_surface, (255, 255, 255, 100),
                         (car_px, car_py), (right_px, right_py), 2)

        # 可选：填充扇形区域
        # 用三角形简化
        pygame.draw.polygon(fov_surface, (255, 255, 255, 30),
                            [(car_px, car_py),
                             (left_px, left_py),
                             (right_px, right_py)])

        surface.blit(fov_surface, (0, 0))

    # ── 工具 ──────────────────────────────────────────────

    def close(self):
        """关闭 Pygame."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
