# 推箱子智能车仿真平台

> 第 21 届全国大学生智能车竞赛 · 智能视觉组 · 推箱子赛题

本项目是一个**全栈推箱子仿真平台**，包含物理引擎、3D 渲染器、BFS 求解器、以及基于强化学习的高层决策系统。
平台以 Python 实现，使用 Gymnasium 接口对接 RL 训练框架 (`MaskablePPO` + ActionMasking)。

---

## 项目结构

```
.
├── AGENTS.md                  # 本文件：项目介绍
├── requirements.txt           # 基础依赖 (pygame, gymnasium, numpy, pillow)
│
├── ── 核心引擎 ──────────────────
├── config.py                  # 全局配置 (分辨率、物理参数、FOV 等)
├── engine.py                  # 游戏物理引擎 (碰撞检测、推箱链、爆炸、配对)
├── map_loader.py              # 地图加载器 (解析 .txt 地图文件)
├── raycaster.py               # 3D 光线投射器 (DDA 算法)
├── renderer.py                # 渲染器 (俯视图 + 3D 第一人称)
│
├── ── Gymnasium 环境 ──────────
├── env.py                     # 低层 Gym 环境 (连续动作空间)
├── wrappers.py                # 动作包装器 (连续→离散)
│
├── ── 求解器 (solver/) ────────
├── solver/
│   ├── pathfinder.py          # BFS 寻路 + 可达性分析
│   ├── push_solver.py         # 单箱推箱 BFS 求解
│   ├── multi_box_solver.py    # 多箱配对求解 (匈牙利匹配)
│   ├── explorer.py            # 视野探索 (FOV 扫描、排除法推导)
│   ├── bomb_planner.py        # 炸弹路径规划 (爆炸清障)
│   └── auto_player.py         # 自动求解总控 (探索→配对→推箱)
│
├── ── 强化学习 (rl/) ──────────
├── rl/
│   ├── high_level_env.py      # 高层 RL 环境 v2 (单步推箱粒度 + 掩码)
│   ├── train.py               # 训练脚本 (课程学习 6 阶段 + checkpoint/评估)
│   └── map_generator.py       # 程序化地图生成器 (连通性/可推性约束)
│
├── ── 可视化入口 ──────────────
├── play.py                    # 人工游玩入口 (键盘操控)
├── auto_solve.py              # BFS 自动求解 + 动画回放
├── preview_policy.py          # RL 策略可视化预览
├── preview_failed.py          # 失败地图排查工具
│
├── ── 数据与资源 ──────────────
├── maps/
│   ├── phase1/ ~ phase6/      # 课程学习 6 阶段地图 (共 62 张)
│   ├── phase456_seed_manifest.json  # Phase 4-6 可用种子清单
│   └── map1~3.txt             # 初始测试地图
├── image_class/               # 箱子类别图片 (10 种卡通角色)
├── image_num/                 # 数字标签图片
├── docx/                      # 设计文档与竞赛规则
│
└── ── 工具与测试 (tools/) ─────
    └── tools/
        ├── create_maps.py         # 地图批量创建
        ├── gen_quality_maps.py    # 高质量地图生成
        ├── gen_verified_maps.py   # 带验证的地图生成
        ├── generate_phase1_maps.py # Phase 1 地图生成
        ├── regen_phase456.py      # Phase 4-6 重新生成
        ├── verify_maps.py         # 地图可解性验证
        ├── test_solver.py         # 求解器测试
        ├── test_bomb_fix.py       # 炸弹逻辑测试
        ├── play_failed.py         # 失败地图调试
        └── _debug_*.py / _test_*.py / _diag.py  # 临时调试脚本
```

---

## 架构概览

```
┌─────────────────────────────────────────────────────┐
│                   RL 训练层                          │
│  MaskablePPO ← SokobanHLEnv (42 个离散动作)          │
│  课程学习: Phase 1→6, 从 1 箱到 3 箱+TNT            │
└───────────────────────┬─────────────────────────────┘
                        │ 高层决策
                        ▼
┌─────────────────────────────────────────────────────┐
│                   BFS 求解层                         │
│  pathfinder → push_solver → multi_box_solver         │
│  explorer (FOV扫描) → bomb_planner (爆炸清障)        │
│  auto_player (总控: 探索→配对→推箱)                  │
└───────────────────────┬─────────────────────────────┘
                        │ 低级动作 (discrete_step)
                        ▼
┌─────────────────────────────────────────────────────┐
│                   物理引擎层                         │
│  engine.py: 碰撞检测 + 推箱链 + 爆炸 + 配对          │
│  map_loader.py: 地图解析                            │
│  config.py: 全局参数                                │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                   渲染层                             │
│  raycaster.py: DDA 光线投射                          │
│  renderer.py: 俯视图 + 3D 第一人称视角               │
└─────────────────────────────────────────────────────┘
```

---

## 核心模块说明

### 1. 物理引擎 (`engine.py`)

- **坐标系**: 16×12 网格，每格 1.0 单位，中心为 `(col+0.5, row+0.5)`
- **离散动作**: 前进/后退/左移/右移/左转45°/右转45°/吸附 (共 7 种)
- **推箱物理**: 
  - 支持链式推 (A 推 B 推 C)，`_build_push_chain` 递归构建
  - 碰撞用 `_rect_collides_wall` 浮点矩形检测 (非整格)
  - 炸弹推入墙壁 → 3×3 爆炸清除障碍
- **配对消除**: 箱子网格位置与同 ID 目标重合时自动消除

### 2. 求解器 (`solver/`)

| 模块 | 功能 |
|------|------|
| `pathfinder.py` | BFS 寻路 + `get_reachable` 可达性分析 |
| `push_solver.py` | BFS 求单箱子推到目标的最短路径 |
| `multi_box_solver.py` | 匈牙利算法匹配全部箱-目标配对 |
| `explorer.py` | 视野探索 (先排除法推导，再 BFS 导航扫描) |
| `bomb_planner.py` | 分析障碍 → 规划炸弹引爆路径 → 清障 |
| `auto_player.py` | 总控: Phase 1 探索 → Phase 2 配对 → Phase 3 推箱 |

### 3. RL 环境 (`rl/high_level_env.py`)

**环境定位**:
- RL 不直接学习底层行走控制；策略只选择“探索哪个实体 / 推哪个实体往哪个方向一格”
- 每个高层动作都由环境内部调用 BFS 完成导航、站位和朝向调整，再落到 `engine.discrete_step()` 执行

**动作空间**: 42 个离散动作，配合 `MaskablePPO`
- `0..4` — 探索第 i 个箱子 (BFS 导航到观测点)
- `5..9` — 探索第 i 个目标
- `10..29` — 推箱子 i 往方向 d 一格 (5 箱 × 4 方向)
- `30..41` — 推炸弹 j 往方向 d 一格 (3 炸弹 × 4 方向)

**部分可观测设定**:
- 箱子类别 ID / 目标数字 ID 需要先进入视野，才会写入 `seen_box_ids` / `seen_target_ids`
- 当同类实体只剩 1 个未知时，环境会用排除法将其视为已知，因此探索动作只在“至少 2 个未知实体”时开放

**状态向量**: 基础 62 维 + 可选 192 维地图布局
- 基础 62 维 = 车位置 (2) + 箱子信息 (5×5) + 目标信息 (5×4) + 炸弹 (3×2) + 进度 (4) + 距离 (5)
- 当前 `rl/train.py` 默认开启 `include_map_layout=True`，因此训练默认使用 **254 维观测**
- `--eval` 会根据模型观测维度自动判断是否附带地图布局；`--resume` 若加载旧 62 维模型，会判定与当前默认训练配置不兼容

**奖励设计**:
1. 步数成本: `-steps × 0.02`
2. 探索奖励: `+3.0` 每发现新实体
3. **距离塑形**: `±2.0` 根据推箱方向 (靠近/远离目标)
4. 配对消除: `+20.0`
5. 通关: `+50 + 效率 × 100` (以 AutoPlayer 步数为基线)
6. 失败推箱: `-2.0` (BFS 导航到位但实体没被推动)

**动作掩码 (`action_masks`)**:
- 探索动作: 只对仍需辨识的箱子/目标开放
- 推箱/推炸弹动作: 同时检查车的站位格是否可达、推后位置是否合法、`_build_push_chain()` 是否可行
- 推箱还会过滤近期失败过的推送和明显死锁位置；若所有动作都被过滤，会保底放出 1 个 fallback 动作避免全零掩码

**防死锁机制**:
- 状态重访惩罚、振荡检测、来回反推惩罚、无进展累积惩罚
- 死角检测 (墙角箱子)、反向推箱可达性分析、死锁箱计数
- 失败推送记忆 (`_failed_pushes`)
- 除通关外，环境还会在 `dead_box` / `oscillation` / `no_progress` / `max_steps` 条件下截断回合

### 4. 训练系统 (`rl/train.py`)

**6 阶段课程学习**:

| 阶段 | 内容 | 训练步数 | max_steps |
|------|------|---------|-----------|
| 1 | 1 箱, 空旷 | 1.5M | 30 |
| 2 | 1 箱, 有墙 | 2.5M | 40 |
| 3 | 2 箱 | 50M | 60 |
| 4 | 3 箱 | 20M | 80 |
| 5 | 3 箱 + TNT | 30M | 100 |
| 6 | 混合 | 35M | 100 |

**训练特性**:
- AutoPlayer 基线: 训练前按地图预计算多 seed 平均步数，用于通关效率奖励
- 种子清单: Phase 4-6 优先使用 `phase456_seed_manifest.json` 中验证过的 seed 保证可解
- 并行环境: 优先使用 `SubprocVecEnv`；由于 BFS、动作掩码和死锁分析都是 CPU 密集型，默认使用 CPU 训练
- 默认网络/超参: `MlpPolicy`, `net_arch=[128,128,64]`, `ReLU`, `lr=3e-4`, `n_steps=1024`, `batch_size=512`, `n_epochs=8`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.005`
- 周期性评估: 使用 deterministic policy 逐图统计 `win_rate` / `avg_steps`，并按“胜率 → 解出地图数 → 平均步数”保存 best checkpoint
- 进度条: 单行实时显示 FPS / Win% / Steps / ETA
- 弱图重训接口: `WeightedMapEnv` 预留了“70% 随机选图 + 30% 弱图重训”的能力；当前默认 `train()` 流程未传入 `weak_maps`，因此现代码路径下实际仍以随机选图为主
- 输出目录: 日志与模型默认保存到 `~/rl_sokoban/logs` 和 `~/rl_sokoban/models`

**评估流程**:
- `python -m rl.train --eval model.zip` 会遍历 Phase 1~6 全部地图
- 评估时优先使用验证过的合法 seed，并以 deterministic 策略逐图运行，输出每张图的通关率、平均步数和相对 AutoPlayer 基线差值

---

## 快速开始

### 安装依赖

```bash
# 基础功能 (游玩、求解)
pip install pygame>=2.5 gymnasium numpy pillow

# RL 训练 (推荐使用独立 conda 环境)
conda create -n rl python=3.12
conda activate rl
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install sb3-contrib tensorboard gymnasium numpy
```

### 人工游玩

```bash
python play.py
# W/A/S/D 移动, ←/→ 转向, 1-9 切换关卡
```

### BFS 自动求解

```bash
python auto_solve.py [地图编号]
# 空格播放, ←/→ 调速
```

### RL 训练

```bash
conda activate rl
python -m rl.train                    # 从 Phase 1 开始
python -m rl.train --phase 4          # 从 Phase 4 开始
python -m rl.train --resume model.zip # 继续训练
python -m rl.train --eval model.zip   # 评估模型
# 默认输出到 ~/rl_sokoban/logs 和 ~/rl_sokoban/models
```

### RL 策略可视化

```bash
python preview_policy.py --model ~/rl_sokoban/models/phase6_best.zip
```

---

> **`tools/` 目录**包含地图生成器、验证工具、测试脚本和开发调试文件。
> 这些文件均为独立运行的脚本，不被核心代码 import，可按需使用。

---

## 关键设计决策

1. **分层架构**: RL 做高层决策 (推哪个箱子往哪个方向), BFS 做低层执行 (导航到站位点+推一格)
2. **单步推箱粒度 (v2)**: 从 "整段 BFS 路径" 改为 "推一格"，使 RL 可以学交错推箱等高级策略
3. **动作掩码**: 物理不可行的推箱方向在掩码中禁用，结合 BFS 可达性检测和死角检测
4. **距离塑形奖励**: 每次推箱都有即时反馈 (靠近目标 +2, 远离 -2)，解决稀疏奖励问题
5. **地图种子管理**: Phase 4-6 复杂地图需要特定种子才可解，通过 `seed_manifest.json` 管理
6. **部分可观测 + 探索先行**: 箱子/目标 ID 需要先观测才能参与稳定配对，RL 需要决定“先探测谁，再推动谁”
7. **训练默认使用 254 维观测**: 在 62 维状态基础上附加完整墙体布局，使策略能直接感知地图结构
8. **效率导向而非只求通关**: 通关奖励以 AutoPlayer 为基线按低层执行步数折算，鼓励学到更短、更稳的策略
