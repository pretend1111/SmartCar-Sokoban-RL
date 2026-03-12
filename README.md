# SmartCar-Sokoban-RL

智能车推箱子仿真、搜索求解与强化学习训练平台。

项目围绕全国大学生智能车竞赛“智能视觉组推箱子赛题”构建，目标不是单纯复刻一个推箱子游戏，而是把比赛里的核心问题放进同一个可复现实验平台里：受限感知、网格物理、编号探索、炸弹清障、全局规划，以及基于 RL 的高层决策。

## 项目定位

- 智能车推箱子赛题的可视化仿真平台
- 结合规则建模、搜索求解与强化学习的 AI 工程项目
- 适合做课程设计、竞赛展示、算法验证和 RL 实验基座

## 核心能力

- 2D 俯视图 + 3D 第一人称渲染
- 连续 / 离散双控制模式
- 链式推箱、TNT 爆炸、编号配对消除
- `AutoPlayer` 完整探索-配对-推箱流程
- `MultiBoxSolver` 多箱全局搜索
- Gymnasium 高层环境 + `MaskablePPO`
- Phase 1~6 课程学习地图与评测工具

## 架构概览

```text
RL Policy
    |
    v
smartcar_sokoban.rl.high_level_env
    |
    v
smartcar_sokoban.solver
    |
    v
smartcar_sokoban.engine
    |
    v
smartcar_sokoban.renderer / raycaster
```

## 目录结构

```text
.
├── smartcar_sokoban/         # 主包：引擎、渲染、求解器、RL、入口模块
│   ├── engine.py
│   ├── renderer.py
│   ├── play.py
│   ├── auto_solve.py
│   ├── preview_failed.py
│   ├── preview_policy.py
│   ├── benchmark.py
│   ├── solver/
│   └── rl/
├── assets/
│   ├── maps/                 # 训练 / 测试地图与 seed manifest
│   └── images/               # 编号贴图与类别贴图
├── scripts/
│   ├── maps/                 # 地图生成、验证、重建脚本
│   ├── debug/                # 调试脚本
│   └── play_failed.py        # 手动排查关卡
├── tests/                    # 回归测试脚本
├── experiments/              # 额外实验：solver BC、GPU 仿真
├── docs/                     # 设计文档、规则整理
├── runs/                     # 本地输出：模型、日志、benchmark 结果
├── requirements.txt
└── AGENTS.md
```

## 安装

基础依赖：

```bash
pip install -r requirements.txt
```

如果要训练 RL，建议单独环境安装 `torch`、`sb3-contrib` 等训练依赖。

## 快速开始

人工游玩：

```bash
python -m smartcar_sokoban.play
```

自动求解：

```bash
python -m smartcar_sokoban.auto_solve
```

按阶段可视化回放：

```bash
python -m smartcar_sokoban.preview_failed --phase 6 --map phase6_11.txt --solver auto
python -m smartcar_sokoban.preview_failed --phase 6 --map phase6_11.txt --solver exact
```

RL 训练 / 评估：

```bash
python -m smartcar_sokoban.rl.train
python -m smartcar_sokoban.rl.train --phase 4
python -m smartcar_sokoban.rl.train --resume runs/rl/models/phase4_best.zip
python -m smartcar_sokoban.rl.train --eval runs/rl/models/phase6_best.zip
```

Benchmark：

```bash
python -m smartcar_sokoban.benchmark
python -m smartcar_sokoban.benchmark --solver exact --save
```

地图工具：

```bash
python scripts/maps/verify_maps.py
python scripts/maps/gen_quality_maps.py --phase 4
```

调试脚本：

```bash
python scripts/debug/debug_phase6_11.py
python scripts/play_failed.py --phase 6 --map phase6_11.txt
```

## 运行约定

- 地图资源统一放在 `assets/maps/`
- 图片资源统一放在 `assets/images/`
- 训练输出默认写入 `runs/rl/`
- benchmark 输出默认写入 `runs/benchmark/`
- `scripts/` 下脚本默认按仓库根目录为工作根运行

## 规则说明

- 小车朝向仅影响观察和旋转，不限制 8 向移动
- 默认禁止普通斜推
- 唯一保留的对角推特例：
  炸弹与墙对角相邻，且夹角两侧正交格无阻挡时，可斜推炸弹入墙并引爆

## 推荐仓库名

`SmartCar-Sokoban-RL`

这个名字直观，GitHub 搜索友好，也同时覆盖“智能车 / 推箱子 / RL”三个关键词。
