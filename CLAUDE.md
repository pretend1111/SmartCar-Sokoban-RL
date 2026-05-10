# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

SmartCar-Sokoban-RL 是面向第 21 届全国大学生智能车竞赛"智能视觉组推箱子赛题"的仿真、搜索与强化学习平台。代码库整合了：

- 网格物理引擎 + 第一人称 3D / 俯视双渲染
- 多层 BFS 求解器栈（`AutoPlayer` 启发式、`MultiBoxSolver` 精确解）
- 基于 `MaskablePPO` 的高层 RL 智能体（离散推箱动作空间）
- Phase 1–6 课程学习地图及生成 / 验证工具

`README.md` 与 `AGENTS.md` 都对项目有所介绍。`AGENTS.md` 更详细，但部分内容已过时（详见末尾"注意事项"）。

## 常用命令

所有入口都从仓库根目录运行。项目没有 `setup.py` / `pyproject.toml`，代码通过 `smartcar_sokoban` 包导入。

```bash
# 安装依赖
pip install -r requirements.txt        # 基础依赖 + RL 依赖一并安装

# 人工游玩（pygame UI）
python -m smartcar_sokoban.play [--phase N] [--map phaseN_xxx.txt] [--god]

# 求解器回放
python -m smartcar_sokoban.auto_solve
python -m smartcar_sokoban.preview_failed --phase 6 --map phase6_11.txt --solver auto
python -m smartcar_sokoban.preview_failed --phase 6 --map phase6_11.txt --solver exact

# RL 训练 / 评估（输出到 runs/rl/）
python -m smartcar_sokoban.rl.train                       # 从 phase 1 开始
python -m smartcar_sokoban.rl.train --phase 4
python -m smartcar_sokoban.rl.train --resume runs/rl/models/phase4_best.zip
python -m smartcar_sokoban.rl.train --eval   runs/rl/models/phase6_best.zip

# RL 策略可视化
python -m smartcar_sokoban.preview_policy --model runs/rl/models/phase6_best.zip

# 求解器评测（多进程并行跑全部 Phase）
python -m smartcar_sokoban.benchmark                       # 两种求解器都跑
python -m smartcar_sokoban.benchmark --solver exact --save # 仅 MultiBoxSolver
python -m smartcar_sokoban.benchmark --phase 4 -j 1        # 单进程串行 + 单 Phase

# 地图工具
python scripts/maps/verify_maps.py
python scripts/maps/gen_quality_maps.py --phase 4
python scripts/maps/rebuild_manifest.py
```

### 测试

测试位于 `tests/`。部分基于 pytest（`test_exact_teacher.py`、`test_diagonal_push.py`），其余可作为脚本直接运行。需在仓库根目录运行，否则各文件中的 `sys.path.insert(...)` 不会指向正确位置。

```bash
python -m pytest tests/                       # 全量
python -m pytest tests/test_exact_teacher.py  # 单文件
python -m pytest tests/test_exact_teacher.py::test_name  # 单测
python tests/test_solver.py map1.txt          # 脚本风格、跑单图
```

### 实验目录

`experiments/solver_bc/` 与 `experiments/gpu_sim/` 是隔离的原型，各自带 README。它们把输出写到 `.agent/solver_bc/`，并依赖主包。其 CLI 参数风格不代表主代码的约定，不要外推。

## 架构

依赖自上而下：RL → solver → engine → renderer。每层都可独立使用。

```
smartcar_sokoban.rl.high_level_env  (Gymnasium 环境，54 维离散动作 + ActionMasker)
        │ 每一步选择一个推箱 / 探索动作
        ▼
smartcar_sokoban.solver             (BFS pathfinder、push_solver、multi_box_solver、explorer、bomb_planner、auto_player)
        │ 输出底层离散动作 (前后 / 平移 / 旋转 / 吸附)
        ▼
smartcar_sokoban.engine             (16×12 网格物理：碰撞、推箱链、3×3 爆炸、ID 配对)
        │
        ▼
smartcar_sokoban.renderer + raycaster   (俯视图 + DDA 第一人称)
```

### 各层职责

- **engine.py** — 16×12 网格，格中心位于 `(col+0.5, row+0.5)`。同时支持连续 / 离散控制（由 `GameConfig.control_mode` 切换）。推箱链递归构建（`_build_push_chain`），墙体碰撞用浮点 AABB（`_rect_collides_wall`），炸弹推入墙时引爆。箱-目标按类别 / 数字 ID 在同一格上消除。
- **solver/** — 各 BFS 模块互相独立：`pathfinder`（可达性）、`push_solver`（单箱计划）、`multi_box_solver`（联合精确解，配合匈牙利算法分配，用作 "exact" 基线）、`explorer`（FOV 扫描 + 同类只剩一个未知时的排除法推断）、`bomb_planner`、`auto_player`（总控：探索 → 配对 → 推箱）。`high_level_teacher` 与 `offline_teacher_cache` 输出与 RL 动作空间对齐的监督信号。
- **rl/high_level_env.py** — 一次 RL 动作 = 一格推箱 *或* 一次探索目标选择；环境内部调用 BFS 把车导航到推位、旋转后，再发出 `engine.discrete_step(...)` 序列。动作掩码会过滤不可行推箱（无法到达推位、推链受阻、近期失败过、死角等）。默认观测维度为 254（`STATE_DIM_WITH_MAP` = 62 基础 + 192 地图布局）；旧的 62 维模型与当前 `--resume` 不自动兼容。
- **rl/train.py** — 6 阶段课程（1 箱空旷 → 1 箱有墙 → 2 箱 → 3 箱 → +TNT → 混合）。每个 Phase 开训前会逐图算 `AutoPlayer` 基线步数，用于通关效率奖励。Phase 4–6 优先使用 `phase456_seed_manifest.json` 中验证过的种子。默认配置：`MaskablePPO`、`MlpPolicy` `[128,128,64]`、`lr=3e-4`、`n_steps=1024`、`batch_size=512`、`n_epochs=8`、CPU `SubprocVecEnv`（BFS 和动作掩码都是 CPU 密集型）。

### 动作空间细节（以代码为准，不是 AGENTS.md）

当前 `smartcar_sokoban/rl/high_level_env.py` 实际暴露 **54** 个离散动作，而非 `AGENTS.md` 写的 42 个：

```
0..4    EXPLORE_BOX[i]              i ∈ 0..MAX_BOXES-1     (MAX_BOXES=5)
5..9    EXPLORE_TGT[i]              i ∈ 0..MAX_TARGETS-1   (MAX_TARGETS=5)
10..29  PUSH_BOX[i]_DIR[d]          5 箱 × 4 方向 (上下左右)
30..53  PUSH_BOMB[j]_DIR[d]         3 炸弹 × 8 方向 (上下左右 + 4 个对角)
```

炸弹推送多出的 4 个对角方向用于支持"对角推炸弹入墙"这一被允许的特例（项目中唯一保留的对角推规则，见 README"运行约定"）。

## 路径与约定

`smartcar_sokoban/paths.py` 是文件系统布局的唯一真源 — 优先 `from smartcar_sokoban.paths import MAPS_ROOT, IMAGES_ROOT, PROJECT_ROOT, RUNS_ROOT`，不要硬编码路径字符串。关键布局：

- `assets/maps/phaseN/` — 课程地图；`phase456_seed_manifest.json` 列出已验证种子；`batch_manifest.json`（不入库，按需重建）列出生成地图
- `assets/images/class/` 与 `assets/images/num/` — 箱子类别贴图与编号贴图
- `runs/rl/{logs,models}/` — RL 输出（gitignore）
- `runs/benchmark/` — benchmark JSON（gitignore）
- `scripts/maps/` 默认以仓库根目录为工作根；`scripts/debug/` 是临时脚本，文件名以 `_` 开头

## 注意事项

- `AGENTS.md` 的文件路径写于代码迁入 `smartcar_sokoban/` 包之前。文中的 `engine.py` / `rl/train.py` 现在都在 `smartcar_sokoban/engine.py` / `smartcar_sokoban/rl/train.py`；`tools/` 也已拆成 `scripts/maps/` 与 `scripts/debug/`。
- `AGENTS.md` 写动作空间是 42 维，实际代码是 54 维（后来增加了炸弹对角方向）。
- AutoPlayer 的 `solve()` 会向 stdout 打印进度。训练 / 评测里都用 `redirect_stdout(devnull)` 包了一层 — 改这些代码路径时不要把它去掉。
- 用 `--resume` 加载未启用 `include_map_layout` 训练出的旧 checkpoint 会因为观测维度不匹配而失败。要么重训，要么在 `train.py` 里把 map layout 关掉。
- `runs/`、`tensorboard/`、`assets/maps/**/_tmp*.txt` 已 gitignore — 在新克隆里默认不存在。
