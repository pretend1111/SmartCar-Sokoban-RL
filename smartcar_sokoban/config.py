"""全局配置 — 所有可调参数集中管理."""

from dataclasses import dataclass


@dataclass
class GameConfig:
    """游戏核心配置."""

    # ── 地图 ──────────────────────────────────────────────
    map_cols: int = 16
    map_rows: int = 12

    # ── 车辆 ──────────────────────────────────────────────
    car_move_speed: float = 3.0       # 格/秒
    car_strafe_speed: float = 3.0     # 格/秒
    car_turn_speed: float = 2.5       # 弧度/秒
    car_size: float = 1.0             # 推箱子碰撞体边长（格）
    car_wall_size: float = 0.9        # 与墙壁碰撞的判定边长（略小，使车可穿过1格走廊）
    box_wall_size: float = 0.9        # 箱子/炸弹与墙壁碰撞的判定边长
    control_mode: str = "continuous"  # "continuous" | "discrete"
    discrete_anim_speed: float = 6.0  # 离散模式动画速度（越大越快）

    # ── 渲染 ──────────────────────────────────────────────
    view_width: int = 640
    view_height: int = 480
    fov: float = 90.0                 # 度
    render_mode: str = "full"         # "full" | "simple"
    fps: int = 60

    # ── 观测 ──────────────────────────────────────────────
    obs_mode: str = "matrix"          # "matrix" | "pixel" | "both"

    # ── 颜色 (2D 俯视图) ─────────────────────────────────
    color_wall: tuple = (64, 64, 64)
    color_floor: tuple = (0, 0, 255)
    color_car_front: tuple = (0, 255, 255)   # 青色 = 车头
    color_car_back: tuple = (0, 255, 0)      # 绿色 = 车尾
    color_box: tuple = (255, 255, 0)         # 黄色 = 可移动箱子
    color_target: tuple = (255, 0, 255)      # 紫色 = 目的地箱子
    color_bomb: tuple = (255, 0, 0)          # 红色 = 炸弹
    color_bg: tuple = (192, 192, 192)        # 灰色 = 地图外
    color_fov_line: tuple = (255, 255, 255, 100)  # 半透明白 = FOV 射线

    # ── 资源路径 ──────────────────────────────────────────
    maps_dir: str = "assets/maps"
    image_num_dir: str = "assets/images/num"
    image_class_dir: str = "assets/images/class"
    textures_dir: str = "assets/textures"
