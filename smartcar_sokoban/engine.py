"""游戏核心引擎 — 纯逻辑，零渲染依赖."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from smartcar_sokoban.action_defs import ACTION_TO_ABS_WORLD_MOVE
from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.map_loader import MapLoader, MapData, BoxInfo, TargetInfo, BombInfo


@dataclass
class GameState:
    """可序列化的完整游戏状态快照."""
    grid: List[List[int]] = field(default_factory=list)
    car_x: float = 0.0
    car_y: float = 0.0
    car_angle: float = 0.0          # 弧度, 0 = 朝上 (Y-)
    boxes: List[BoxInfo] = field(default_factory=list)
    targets: List[TargetInfo] = field(default_factory=list)
    bombs: List[BombInfo] = field(default_factory=list)
    seen_box_ids: Set[int] = field(default_factory=set)      # 已被 FOV 看到的 box 索引
    seen_target_ids: Set[int] = field(default_factory=set)   # 已被 FOV 看到的 target 索引
    won: bool = False
    score: int = 0
    total_pairs: int = 0


class GameEngine:
    """推箱子游戏核心引擎.

    所有逻辑在此实现，不依赖任何渲染库。
    """

    def __init__(self, config: Optional[GameConfig] = None, base_dir: str = ""):
        self.cfg = config or GameConfig()
        self.loader = MapLoader(self.cfg, base_dir)
        self.state = GameState()

    # ── 公共接口 ──────────────────────────────────────────

    def reset(self, map_path: str) -> GameState:
        """加载地图并重置游戏状态."""
        map_data = self.loader.load(map_path)

        self.state = GameState(
            grid=map_data.grid,
            car_x=map_data.car_x,
            car_y=map_data.car_y,
            car_angle=-math.pi / 2,   # 朝上 (Y 减小方向)
            boxes=list(map_data.boxes),
            targets=list(map_data.targets),
            bombs=list(map_data.bombs),
            total_pairs=len(map_data.boxes),
        )
        # 初始 FOV 扫描
        self._update_fov_visibility()
        return self.get_state()

    def reset_from_string(self, map_string: str) -> GameState:
        """从字符串加载地图并重置 (无文件 I/O, RL 训练用)."""
        map_data = self.loader.load_from_string(map_string)

        self.state = GameState(
            grid=map_data.grid,
            car_x=map_data.car_x,
            car_y=map_data.car_y,
            car_angle=-math.pi / 2,
            boxes=list(map_data.boxes),
            targets=list(map_data.targets),
            bombs=list(map_data.bombs),
            total_pairs=len(map_data.boxes),
        )
        self._update_fov_visibility()
        return self.get_state()

    def step(self, forward: float, strafe: float, turn: float,
             dt: float) -> GameState:
        """执行一帧更新.

        Args:
            forward: -1.0 ~ 1.0 (后退 ~ 前进)
            strafe:  -1.0 ~ 1.0 (左移 ~ 右移)
            turn:    -1.0 ~ 1.0 (左转 ~ 右转)
            dt:      时间步长（秒）
        """
        if self.state.won:
            return self.get_state()

        s = self.state

        # 1) 转向
        s.car_angle += turn * self.cfg.car_turn_speed * dt
        # 归一化到 [-π, π]
        s.car_angle = math.atan2(math.sin(s.car_angle), math.cos(s.car_angle))

        # 2) 计算期望位移（世界坐标）
        cos_a = math.cos(s.car_angle)
        sin_a = math.sin(s.car_angle)

        # 车头方向: angle=0 → 朝上 → dx=0, dy=-1
        # 但我们用标准数学角: angle 从 +X 轴逆时针
        # 车头方向向量: (cos(angle), sin(angle))
        # 注意: 在屏幕坐标系中 Y 轴向下, 所以 "朝上" = angle = -π/2
        # forward: 沿车头方向
        # strafe:  沿车头方向的右侧 (顺时针 90°)

        fwd_dx = cos_a * forward * self.cfg.car_move_speed * dt
        fwd_dy = sin_a * forward * self.cfg.car_move_speed * dt

        # 右侧方向 = 车头方向顺时针 90°（屏幕坐标 Y 向下）
        # forward = (cos(a), sin(a)), right = (-sin(a), cos(a))
        # strafe > 0 = 右移
        str_dx = -sin_a * strafe * self.cfg.car_strafe_speed * dt
        str_dy = cos_a * strafe * self.cfg.car_strafe_speed * dt

        dx = fwd_dx + str_dx
        dy = fwd_dy + str_dy

        # 3) 移动车并处理碰撞 (分轴)
        if dx != 0:
            new_x = s.car_x + dx
            if not self._car_collides_wall(new_x, s.car_y):
                # 检查是否推动箱子/炸弹
                push_ok = self._try_push(new_x, s.car_y, dx, 0)
                if push_ok:
                    s.car_x = new_x
            # 碰墙: X 方向速度分量归零 (不更新 x)

        if dy != 0:
            new_y = s.car_y + dy
            if not self._car_collides_wall(s.car_x, new_y):
                push_ok = self._try_push(s.car_x, new_y, 0, dy)
                if push_ok:
                    s.car_y = new_y

        # 4) 检查配对
        self._check_pairings()

        # 5) 更新 FOV 可见性
        self._update_fov_visibility()

        # 6) 检查胜利
        if len(s.boxes) == 0 and s.total_pairs > 0:
            s.won = True

        return self.get_state()

    def get_state(self) -> GameState:
        """返回当前状态的浅拷贝."""
        return copy.copy(self.state)

    def discrete_step(self, action: int) -> GameState:
        """离散模式: 移动整1格 / 旋转45°.

        Actions:
            0 = 前进1格
            1 = 后退1格
            2 = 左移1格
            3 = 右移1格
            4 = 左转45°
            5 = 右转45°
            6 = 不动
            7..14 = 求解器专用绝对方向平移（上下左右 + 4 个对角）
        """
        if self.state.won:
            return self.get_state()

        s = self.state

        # ── 吸附检测：如果不在网格上，先吸附 ──────────────
        snap_x = round(s.car_x - 0.5) + 0.5
        snap_y = round(s.car_y - 0.5) + 0.5
        # 角度吸附到最近的 45° 倍数
        snap_angle = round(s.car_angle / (math.pi / 4)) * (math.pi / 4)
        snap_angle = math.atan2(math.sin(snap_angle), math.cos(snap_angle))

        needs_snap = (abs(s.car_x - snap_x) > 0.01 or
                      abs(s.car_y - snap_y) > 0.01 or
                      abs(s.car_angle - snap_angle) > 0.01)

        if needs_snap:
            # 吸附到最近的合法位置（不碰墙就吸附）
            if not self._car_collides_wall(snap_x, snap_y):
                s.car_x = snap_x
                s.car_y = snap_y
            s.car_angle = snap_angle
            # 吸附本身就是这一步的动画，不执行实际动作
            self._update_fov_visibility()
            return self.get_state()

        # ── 正常离散动作 ──────────────────────────────────
        if action == 4:  # 左转 45°
            s.car_angle -= math.pi / 4
            s.car_angle = math.atan2(math.sin(s.car_angle),
                                     math.cos(s.car_angle))
        elif action == 5:  # 右转 45°
            s.car_angle += math.pi / 4
            s.car_angle = math.atan2(math.sin(s.car_angle),
                                     math.cos(s.car_angle))
        elif action in ACTION_TO_ABS_WORLD_MOVE:
            dx, dy = ACTION_TO_ABS_WORLD_MOVE[action]
            self._try_discrete_move(float(dx), float(dy))
        elif action != 6:  # 移动
            cos_a = math.cos(s.car_angle)
            sin_a = math.sin(s.car_angle)

            if action == 0:    # 前进
                dx = round(cos_a)
                dy = round(sin_a)
            elif action == 1:  # 后退
                dx = -round(cos_a)
                dy = -round(sin_a)
            elif action == 2:  # 左移
                dx = round(sin_a)
                dy = round(-cos_a)
            elif action == 3:  # 右移
                dx = round(-sin_a)
                dy = round(cos_a)
            else:
                dx, dy = 0, 0

            if dx != 0 or dy != 0:
                diagonal_push_attempt = (
                    dx != 0 and dy != 0 and
                    bool(self._get_pushables_at(s.car_x + dx, s.car_y + dy))
                )
                move_dx = float(dx)
                move_dy = float(dy)
                moved = self._try_discrete_move(move_dx, move_dy)

                # 对角离散移动先尝试整段对角；若被墙角/推链阻挡，
                # 再按连续模式同样的 X→Y 顺序回退为滑移。
                if (not moved and dx != 0 and dy != 0 and
                        not diagonal_push_attempt):
                    self._try_discrete_move(move_dx, 0.0)
                    self._try_discrete_move(0.0, move_dy)

        # 检查配对、FOV、胜利
        self._check_pairings()
        self._update_fov_visibility()
        if len(s.boxes) == 0 and s.total_pairs > 0:
            s.won = True

        return self.get_state()

    # ── 碰撞检测 ──────────────────────────────────────────

    def _car_collides_wall(self, cx: float, cy: float) -> bool:
        """检测车在 (cx, cy) 时是否与任何墙壁碰撞 (使用较小的碰撞体)."""
        half = self.cfg.car_wall_size / 2.0
        return self._rect_collides_wall(cx - half, cy - half, cx + half, cy + half)

    def _is_diagonal_move(self, dx: float, dy: float) -> bool:
        return abs(dx) > 1e-6 and abs(dy) > 1e-6

    def _diagonal_sweep_blocked(self, cx: float, cy: float,
                                dx: float, dy: float,
                                half: float) -> bool:
        """检测对角移动是否会擦过墙角.

        对角移动时，若 X-only 或 Y-only 中间位置任一与墙重叠，
        则说明该移动会从墙角“穿过去”，应视为被阻挡。
        """
        if not self._is_diagonal_move(dx, dy):
            return False

        if self._rect_collides_wall(cx + dx - half, cy - half,
                                    cx + dx + half, cy + half):
            return True
        if self._rect_collides_wall(cx - half, cy + dy - half,
                                    cx + half, cy + dy + half):
            return True
        return False

    def _try_discrete_move(self, dx: float, dy: float) -> bool:
        """离散模式下尝试移动一步，返回是否成功."""
        if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
            return False

        s = self.state
        if self._diagonal_sweep_blocked(
                s.car_x, s.car_y, dx, dy, self.cfg.car_wall_size / 2.0):
            return False

        target_x = s.car_x + dx
        target_y = s.car_y + dy
        if self._car_collides_wall(target_x, target_y):
            return False

        push_ok = self._try_push(target_x, target_y, dx, dy)
        if not push_ok:
            return False

        s.car_x = target_x
        s.car_y = target_y
        return True

    def _rect_collides_wall(self, x1: float, y1: float,
                            x2: float, y2: float) -> bool:
        """检测矩形区域是否与墙壁重叠."""
        return bool(self._get_overlapping_wall_cells(x1, y1, x2, y2))

    def _get_overlapping_wall_cells(self, x1: float, y1: float,
                                    x2: float, y2: float
                                    ) -> List[Tuple[int, int]]:
        """返回与矩形重叠的所有墙格坐标."""
        grid = self.state.grid
        min_col = max(0, int(math.floor(x1)))
        max_col = min(self.cfg.map_cols - 1, int(math.floor(x2)))
        min_row = max(0, int(math.floor(y1)))
        max_row = min(self.cfg.map_rows - 1, int(math.floor(y2)))

        overlaps = []
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                if grid[r][c] == 1:
                    # 墙壁格子占据 [c, c+1) × [r, r+1)
                    if x2 > c and x1 < c + 1 and y2 > r and y1 < r + 1:
                        overlaps.append((c, r))
        return overlaps

    def _rect_collides_rect(self, ax: float, ay: float, asize: float,
                            bx: float, by: float, bsize: float) -> bool:
        """两个正方形碰撞体是否重叠 (中心+边长)."""
        ah = asize / 2.0
        bh = bsize / 2.0
        return (abs(ax - bx) < ah + bh) and (abs(ay - by) < ah + bh)

    # ── 推动系统 ──────────────────────────────────────────

    def _try_push(self, new_car_x: float, new_car_y: float,
                  dx: float, dy: float) -> bool:
        """尝试推动车与之碰撞的箱子/炸弹链.

        返回 True 表示车可以移动到 new 位置，False 表示被阻挡。
        """
        s = self.state
        # 收集车碰到的所有可推物体
        pushables = self._get_pushables_at(new_car_x, new_car_y)

        if not pushables:
            return True  # 没碰到可推物体，车可以自由移动

        # 对每个被碰到的物体，沿推力方向尝试推动
        for ptype, pidx in pushables:
            chain = self._build_push_chain(ptype, pidx, dx, dy)
            if chain is None:
                return False  # 链条被墙阻挡，车无法移动

        # 所有链条都能推 → 执行推动
        already_moved = set()
        for ptype, pidx in pushables:
            chain = self._build_push_chain(ptype, pidx, dx, dy)
            if chain:
                for ctype, cidx, move_dx, move_dy in chain:
                    key = (ctype, cidx)
                    if key in already_moved:
                        continue
                    already_moved.add(key)
                    if ctype == 'box':
                        s.boxes[cidx].x += move_dx
                        s.boxes[cidx].y += move_dy
                    elif ctype == 'bomb':
                        s.bombs[cidx].x += move_dx
                        s.bombs[cidx].y += move_dy

        # 检查炸弹是否碰墙 → 爆炸（传入推动方向）
        self._check_bomb_explosions(dx, dy)

        return True

    def _grid_cell_has_entity(self, col: int, row: int,
                              exclude: Optional[Tuple[str, int]] = None) -> bool:
        """检测某个网格是否被箱子/炸弹占据."""
        s = self.state
        for i, box in enumerate(s.boxes):
            if exclude == ('box', i):
                continue
            if (int(box.x), int(box.y)) == (col, row):
                return True
        for i, bomb in enumerate(s.bombs):
            if exclude == ('bomb', i):
                continue
            if (int(bomb.x), int(bomb.y)) == (col, row):
                return True
        return False

    def _is_special_diagonal_bomb_push(self, ptype: str, pidx: int,
                                       dx: float, dy: float) -> bool:
        """仅允许炸弹斜推入对角墙，且夹角两侧必须都是空气."""
        if ptype != 'bomb' or not self._is_diagonal_move(dx, dy):
            return False

        bomb = self.state.bombs[pidx]
        bc, br = int(bomb.x), int(bomb.y)
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1
        wall_col = bc + step_x
        wall_row = br + step_y

        if not (0 <= wall_col < self.cfg.map_cols and
                0 <= wall_row < self.cfg.map_rows):
            return False
        if self.state.grid[wall_row][wall_col] != 1:
            return False

        side_a = (bc + step_x, br)
        side_b = (bc, br + step_y)
        for col, row in (side_a, side_b):
            if not (0 <= col < self.cfg.map_cols and
                    0 <= row < self.cfg.map_rows):
                return False
            if self.state.grid[row][col] == 1:
                return False
            if self._grid_cell_has_entity(col, row, exclude=('bomb', pidx)):
                return False

        return True

    def _get_pushables_at(self, cx: float, cy: float
                          ) -> List[Tuple[str, int]]:
        """获取车在 (cx,cy) 时碰到的所有可推物体."""
        s = self.state
        result = []
        for i, box in enumerate(s.boxes):
            if self._rect_collides_rect(cx, cy, self.cfg.car_size,
                                        box.x, box.y, 1.0):
                result.append(('box', i))
        for i, bomb in enumerate(s.bombs):
            if self._rect_collides_rect(cx, cy, self.cfg.car_size,
                                        bomb.x, bomb.y, 1.0):
                result.append(('bomb', i))
        return result

    def _build_push_chain(self, ptype: str, pidx: int,
                          dx: float, dy: float
                          ) -> Optional[List[Tuple[str, int, float, float]]]:
        """构建推动链条. 返回 None 表示被墙阻挡."""
        s = self.state
        chain = []

        if self._is_diagonal_move(dx, dy):
            if self._is_special_diagonal_bomb_push(ptype, pidx, dx, dy):
                chain.append((ptype, pidx, dx, dy))
                return chain
            return None

        # 获取当前物体位置
        if ptype == 'box':
            obj = s.boxes[pidx]
        else:
            obj = s.bombs[pidx]

        new_x = obj.x + dx
        new_y = obj.y + dy

        # 检查新位置是否碰墙（使用较小的碰撞体，使箱子能通过1格走廊）
        half = self.cfg.box_wall_size / 2.0
        if self._rect_collides_wall(new_x - half, new_y - half,
                                    new_x + half, new_y + half):
            if ptype == 'bomb':
                # 炸弹碰墙 → 允许推动（会在后续爆炸处理中消耗）
                chain.append((ptype, pidx, dx, dy))
                return chain
            else:
                return None  # 箱子碰墙 → 阻挡

        chain.append((ptype, pidx, dx, dy))

        # 检查新位置是否碰到其他可推物体 → 连锁
        # 使用 box_wall_size 避免箱子之间"太厚"
        bws = self.cfg.box_wall_size
        for i, box in enumerate(s.boxes):
            if ptype == 'box' and i == pidx:
                continue
            if self._rect_collides_rect(new_x, new_y, bws, box.x, box.y, bws):
                sub = self._build_push_chain('box', i, dx, dy)
                if sub is None:
                    return None
                chain.extend(sub)

        for i, bomb in enumerate(s.bombs):
            if ptype == 'bomb' and i == pidx:
                continue
            if self._rect_collides_rect(new_x, new_y, bws, bomb.x, bomb.y, bws):
                sub = self._build_push_chain('bomb', i, dx, dy)
                if sub is None:
                    return None
                chain.extend(sub)

        return chain

    # ── 爆炸系统 ──────────────────────────────────────────

    def _check_bomb_explosions(self, push_dx: float, push_dy: float):
        """检查是否有炸弹碰墙，执行爆炸.

        爆炸规则: 炸弹被推向的那面墙为中心 3×3 爆炸（单次）。
        push_dx/push_dy 是推动方向。
        """
        s = self.state
        bombs_to_remove = []
        step_x = 1 if push_dx > 1e-6 else -1 if push_dx < -1e-6 else 0
        step_y = 1 if push_dy > 1e-6 else -1 if push_dy < -1e-6 else 0

        for i, bomb in enumerate(s.bombs):
            half = 0.5
            wall_cells = self._get_overlapping_wall_cells(
                bomb.x - half, bomb.y - half,
                bomb.x + half, bomb.y + half)
            if not wall_cells:
                continue

            # 连续模式下 push_dx/push_dy 可能是小数，因此不能再用“炸弹格+方向”
            # 反推墙位置；直接取炸弹当前实际重叠到的墙格。
            if step_x > 0:
                wall_col, wall_row = max(wall_cells, key=lambda pos: pos[0])
            elif step_x < 0:
                wall_col, wall_row = min(wall_cells, key=lambda pos: pos[0])
            elif step_y > 0:
                wall_col, wall_row = max(wall_cells, key=lambda pos: pos[1])
            elif step_y < 0:
                wall_col, wall_row = min(wall_cells, key=lambda pos: pos[1])
            else:
                wall_col, wall_row = wall_cells[0]

            # 以实际接触到的墙格为中心执行 3×3 爆炸
            self._explode(wall_col, wall_row)

            bombs_to_remove.append(i)

        # 从后往前删除
        for i in sorted(bombs_to_remove, reverse=True):
            s.bombs.pop(i)

    def _explode(self, wall_col: int, wall_row: int):
        """以墙壁格子 (wall_col, wall_row) 为中心执行 3×3 爆炸."""
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                r = wall_row + dy
                c = wall_col + dx
                # 不炸最外圈
                if r <= 0 or r >= self.cfg.map_rows - 1:
                    continue
                if c <= 0 or c >= self.cfg.map_cols - 1:
                    continue
                if self.state.grid[r][c] == 1:
                    self.state.grid[r][c] = 0

    # ── 配对系统 ──────────────────────────────────────────

    def _check_pairings(self):
        """检查可移动箱子是否到达对应的目的地."""
        s = self.state
        boxes_to_remove = []
        targets_to_remove = []

        for bi, box in enumerate(s.boxes):
            for ti, target in enumerate(s.targets):
                # 检查箱子中心是否在目的地所在格子内
                tgx = int(target.x - 0.5)  # 目的地所在列
                tgy = int(target.y - 0.5)  # 目的地所在行
                if (tgx < box.x < tgx + 1) and (tgy < box.y < tgy + 1):
                    if box.class_id == target.num_id:
                        # 编号匹配 → 得分，标记移除
                        boxes_to_remove.append(bi)
                        targets_to_remove.append(ti)
                        s.score += 1
                        break
                    # 编号不匹配 → 穿过（目的地无碰撞体，不做任何事）

        # 从后往前删除（避免索引偏移）
        for i in sorted(set(boxes_to_remove), reverse=True):
            s.boxes.pop(i)
        for i in sorted(set(targets_to_remove), reverse=True):
            s.targets.pop(i)

        # 更新 seen 集合中的索引（删除后索引会变，需要重建）
        # 简化处理：删除后清空 seen 集合中被移除的索引
        # 其实 seen 集合存的是 box/target 的 id，删除后就无效了
        # 由于我们用列表索引，删除后需要调整
        # 更好的方案：用 id() 或额外 uid，但目前用索引够用
        # 重建 seen 集合
        new_seen_boxes = set()
        for old_idx in s.seen_box_ids:
            # 计算 old_idx 在删除后的新索引
            removed_before = sum(1 for r in boxes_to_remove if r < old_idx)
            if old_idx not in boxes_to_remove:
                new_seen_boxes.add(old_idx - removed_before)
        s.seen_box_ids = new_seen_boxes

        new_seen_targets = set()
        for old_idx in s.seen_target_ids:
            removed_before = sum(1 for r in targets_to_remove if r < old_idx)
            if old_idx not in targets_to_remove:
                new_seen_targets.add(old_idx - removed_before)
        s.seen_target_ids = new_seen_targets

    # ── FOV 可见性 (简易模式用) ───────────────────────────

    # 严格识别参数: 必须近距离 (≤ √2 + ε) + 朝向 entity (±22.5°)
    IDENT_MAX_DIST = 1.5         # 4 邻 = 1.0, 8 邻 = √2 ≈ 1.414
    IDENT_HALF_ANGLE = math.pi / 8   # 22.5° (8-向系统的 1 个 tick)

    def _update_fov_visibility(self):
        """更新 FOV 可见性: 只有 "怼到 entity 旁边 + 朝向 entity" 才算识别.

        新规则 (实车摄像头近距离对准):
          1. 车与 entity 中心距离 ≤ 1.5 (4 邻或 8 邻紧贴)
          2. 车头朝向跟 entity 方向夹角 ≤ ±22.5°
          3. 射线无遮挡

        三个条件全部满足才标记为已识别.
        """
        s = self.state

        for i, box in enumerate(s.boxes):
            if i not in s.seen_box_ids:
                if self._can_identify_entity(box.x, box.y):
                    s.seen_box_ids.add(i)

        for i, target in enumerate(s.targets):
            if i not in s.seen_target_ids:
                if self._can_identify_entity(target.x, target.y):
                    s.seen_target_ids.add(i)

    def _can_identify_entity(self, tx: float, ty: float) -> bool:
        """严格识别检查: 距离 + 朝向 + 视线不阻挡."""
        s = self.state
        dx = tx - s.car_x
        dy = ty - s.car_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > self.IDENT_MAX_DIST:
            return False
        if dist < 0.01:
            return True  # 重叠 (理论上不会发生, 兜底)

        angle_to = math.atan2(dy, dx)
        diff = angle_to - s.car_angle
        diff = math.atan2(math.sin(diff), math.cos(diff))   # → [-π, π]
        if abs(diff) > self.IDENT_HALF_ANGLE:
            return False

        # 视线遮挡 (1.5 距离内基本不可能, 兜底)
        return not self._ray_blocked(s.car_x, s.car_y, tx, ty, tx, ty)

    def _is_in_fov(self, tx: float, ty: float, half_fov: float,
                   exclude_x: float = -999, exclude_y: float = -999) -> bool:
        """判断目标 (tx,ty) 是否在车的 FOV 锥内且未被墙/箱子/炸弹遮挡."""
        s = self.state
        dx = tx - s.car_x
        dy = ty - s.car_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            return True  # 重叠

        # 目标相对车头的角度
        angle_to_target = math.atan2(dy, dx)
        angle_diff = angle_to_target - s.car_angle
        # 归一化到 [-π, π]
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        if abs(angle_diff) > half_fov:
            return False  # 不在 FOV 内

        # 射线检测：是否被墙或其他物体遮挡
        return not self._ray_blocked(s.car_x, s.car_y, tx, ty,
                                     exclude_x, exclude_y)

    def _ray_hits_wall(self, x0: float, y0: float,
                       x1: float, y1: float) -> bool:
        """简单射线检测：从 (x0,y0) 到 (x1,y1) 之间是否有墙.

        使用步进采样法。
        """
        dx = x1 - x0
        dy = y1 - y0
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            return False

        steps = int(dist * 4) + 1  # 每格 4 个采样点
        for i in range(1, steps):
            t = i / steps
            px = x0 + dx * t
            py = y0 + dy * t
            col = int(px)
            row = int(py)
            if 0 <= row < self.cfg.map_rows and 0 <= col < self.cfg.map_cols:
                if self.state.grid[row][col] == 1:
                    return True
        return False

    def _ray_blocked(self, x0: float, y0: float,
                     x1: float, y1: float,
                     exclude_x: float = -999,
                     exclude_y: float = -999) -> bool:
        """射线检测：从 (x0,y0) 到 (x1,y1) 之间是否被墙壁或物体遮挡.

        exclude_x/y 用于排除目标自身（避免自己挡住自己）。
        """
        # 先检查墙壁
        if self._ray_hits_wall(x0, y0, x1, y1):
            return True

        # 检查是否被其他箱子/炸弹遮挡
        s = self.state
        ray_dx = x1 - x0
        ray_dy = y1 - y0
        target_dist_sq = ray_dx * ray_dx + ray_dy * ray_dy

        for box in s.boxes:
            if abs(box.x - exclude_x) < 0.01 and abs(box.y - exclude_y) < 0.01:
                continue  # 排除目标自身
            if self._object_blocks_ray(x0, y0, ray_dx, ray_dy,
                                       target_dist_sq, box.x, box.y):
                return True

        for bomb in s.bombs:
            if abs(bomb.x - exclude_x) < 0.01 and abs(bomb.y - exclude_y) < 0.01:
                continue
            if self._object_blocks_ray(x0, y0, ray_dx, ray_dy,
                                       target_dist_sq, bomb.x, bomb.y):
                return True

        # 目的地箱子也会遮挡视线
        for target in s.targets:
            if abs(target.x - exclude_x) < 0.01 and abs(target.y - exclude_y) < 0.01:
                continue
            if self._object_blocks_ray(x0, y0, ray_dx, ray_dy,
                                       target_dist_sq, target.x, target.y):
                return True
        return False

    def _object_blocks_ray(self, x0: float, y0: float,
                           ray_dx: float, ray_dy: float,
                           target_dist_sq: float,
                           obj_x: float, obj_y: float) -> bool:
        """检查一个物体是否挡在射线路径上（且比目标更近）."""
        half = 0.5
        # 简单 AABB 检测：物体中心到射线的最近点
        # 将物体位置投影到射线上
        ox = obj_x - x0
        oy = obj_y - y0
        ray_len_sq = ray_dx * ray_dx + ray_dy * ray_dy
        if ray_len_sq < 0.001:
            return False
        t = (ox * ray_dx + oy * ray_dy) / ray_len_sq
        if t < 0.05 or t > 0.95:  # 只关心射线中间段
            return False
        # 物体必须比目标近
        obj_dist_sq = ox * ox + oy * oy
        if obj_dist_sq >= target_dist_sq * 0.95:
            return False
        # 射线上最近点到物体中心的距离
        closest_x = x0 + ray_dx * t
        closest_y = y0 + ray_dy * t
        if abs(closest_x - obj_x) < half and abs(closest_y - obj_y) < half:
            return True
        return False

    # ── FOV 射线端点计算 (渲染用) ─────────────────────────

    def get_fov_rays(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """计算 FOV 锥的两条射线端点（从车位置出发）.

        返回两条射线的终点坐标 (用于渲染 FOV 锥线).
        """
        s = self.state
        half_fov = math.radians(self.cfg.fov) / 2.0
        max_dist = max(self.cfg.map_cols, self.cfg.map_rows) * 1.5

        # 左射线
        left_angle = s.car_angle - half_fov
        left_end = self._cast_ray_to_wall(
            s.car_x, s.car_y,
            math.cos(left_angle), math.sin(left_angle),
            max_dist
        )

        # 右射线
        right_angle = s.car_angle + half_fov
        right_end = self._cast_ray_to_wall(
            s.car_x, s.car_y,
            math.cos(right_angle), math.sin(right_angle),
            max_dist
        )

        return left_end, right_end

    def _cast_ray_to_wall(self, ox: float, oy: float,
                          dx: float, dy: float,
                          max_dist: float) -> Tuple[float, float]:
        """从 (ox,oy) 沿 (dx,dy) 方向投射射线，返回碰墙点."""
        step_size = 0.1
        dist = 0.0
        while dist < max_dist:
            dist += step_size
            px = ox + dx * dist
            py = oy + dy * dist
            col = int(px)
            row = int(py)
            if row < 0 or row >= self.cfg.map_rows or \
               col < 0 or col >= self.cfg.map_cols:
                return (px, py)
            if self.state.grid[row][col] == 1:
                return (px, py)
        return (ox + dx * max_dist, oy + dy * max_dist)
