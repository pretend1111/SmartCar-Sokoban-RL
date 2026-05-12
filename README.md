# SmartCar-Sokoban-RL

面向第 21 届全国大学生智能车竞赛"智能视觉组推箱子赛题"的仿真、搜索与学习平台。
最新版核心是 **SAGE-PR** (Symbolic-Aided Grid-Equivariant Policy Ranker) — 神经-符号混合候选评分器, 用于在 MCU 端做实时推箱决策。

---

## 1. 项目目标

赛题: 智能车在 16×12 网格上识别箱子 ID 配对, 推到对应数字目标格, 必要时用炸弹清障。
通关率目标: phase 1-6 全部 ≥ 95% deterministic 通关率。
部署目标: OpenART mini MCU (TFLM, INT8 ≤ 500 KB, 推理 ≤ 50 ms)。

---

## 2. 架构

```
[感知/识别]
    ↓
[Belief State]  ─ 14×10 墙图 + 车位姿 + 箱/目/炸弹 + 已识别 ID + FOV 记忆 + Π 兼容矩阵
    ↓
[领域特征]      ─ BFS 距离场 / 推送距离场 / 流场 / 死角 mask / 信息增益热度图
    ↓
[Candidate Generator]  ─ ≤ 64 候选 (push_box × 4 dir × 3 macro + push_bomb × 4 dir + inspect)
                         合法性 mask 由经典规则给出
    ↓
[Neural Candidate Ranker (SAGE-PR)]  ─ ~105K params, Conv 主干 + Shared MLP 候选评分
                                       Score + Aux heads (value, deadlock, progress, info_gain)
    ↓
[BFS 执行器] → engine
```

### 物理规则 (与 engine 严格匹配)

- 网格 16×12, 可玩区 14×10 (外圈固定墙)
- **车移动: 仅 4 方向 (上下左右 + ±90° 旋转)** — 不允许 45° 转弯或对角平移
- **识别 FOV**: 必须 `dist ≤ 1.0 + 朝向精确正对 + 无遮挡` — "怼着 entity 4 邻紧贴"
- 推箱: 链式推 (box→box, box→bomb 都支持)
- 炸弹: 推到墙引爆, 3×3 范围消除墙体 + 周围实体
- 配对: box.class_id == target.num_id 时落到 target 上自动消除

---

## 3. 当前状态

| 组件 | 状态 |
|------|------|
| 符号层 (Belief, candidates, features) | ✅ 完成 |
| 4-方向 engine + 严格 FOV | ✅ 完成 |
| `plan_exploration_v3` (推开障碍 + 拓扑预配对) | ✅ 完成 |
| 翻译器 (move ↔ candidate) 100% 验证 | ✅ 完成 (0 diverge) |
| V1 数据集 (~89k samples) | ✅ 落盘 `runs/sage_pr/full_v5_v3/` (gitignored) |
| SAGE-PR 模型 (PyTorch) | ✅ 完成 |
| BC 训练 + L_info 辅助监督 | ✅ 完成 |
| DAgger 迭代 | ✅ 通用框架 |
| Rollout search 推理 | ✅ 完成 |
| OpenART 部署 (TFLM + INT8 + QAT) | 🔄 进行中 |

最新 eval (`v3_large9`, **全量 verified maps**, β=8 ℓ=25 rollout, mode v1):
```
phase 4: 98.91%  (1000/1011)   失败 11 张
phase 5: 97.82%  (988/1010)    失败 22 张
phase 6: 98.81%  (999/1011)    失败 12 张
phase 1-3: 全量 eval 跑中 (subset eval: p1=100%/p2=99%/p3=98% on first 100)
```

⚠️ **重要说明:此 95%+ 通关率有信息泄漏前提**,详见 §10 [遗留问题](#10-遗留问题待合作者解决).

---

## 4. 目录结构

```
.
├── smartcar_sokoban/             # 核心包
│   ├── engine.py                  # 4-方向网格引擎, FOV 识别
│   ├── renderer.py                # 2D 俯视 + 3D 第一人称
│   ├── raycaster.py               # DDA 3D 视觉
│   ├── action_defs.py             # 离散动作定义
│   ├── play.py / auto_solve.py    # 手动游玩 / 自动求解
│   ├── preview_*.py               # 多种可视化 (老师/失败图/策略)
│   ├── solver/
│   │   ├── pathfinder.py          # BFS 路径
│   │   ├── push_solver.py         # 单箱推送
│   │   ├── multi_box_solver.py    # 多箱全局搜索
│   │   ├── explorer.py            # 探索器 V1 (含 4-dir 严格 FOV)
│   │   ├── explorer_v2.py         # +推开障碍补丁
│   │   ├── explorer_v3.py         # +拓扑预配对 (当前正式版)
│   │   ├── bomb_planner.py
│   │   └── auto_player.py
│   ├── symbolic/
│   │   ├── belief.py              # BeliefState + 4-向 theta
│   │   ├── features.py            # 领域特征 (deadlock_mask 等)
│   │   ├── candidates.py          # 候选生成 + 合法性
│   │   ├── cand_features.py       # 128 维候选特征
│   │   └── grid_tensor.py         # 30 通道空间张量
│   └── rl/                        # 旧版高层 RL (已被 SAGE-PR 替代, 保留对照)
│
├── experiments/sage_pr/           # SAGE-PR 主战场
│   ├── model.py                   # ~105K param 网络
│   ├── train_sage_pr.py           # BC 训练 (+L_info, GPU-resident)
│   ├── build_dataset_v5.py        # 数据集生成 (V1 路线, 含翻译器验证)
│   ├── dagger_v1.py               # V1 DAgger (并行)
│   ├── evaluate_sage_pr.py        # Greedy/top-k eval (+老师对比)
│   ├── rollout_search_eval.py     # Rollout search 推理
│   ├── preview_trajectory.py      # 通用 trajectory 预览 (见 docs/PREVIEW_TRAJECTORY.md)
│   ├── extract_fails.py           # 提取数据集失败地图清单
│   └── verify_translator.py       # 翻译器 100% 验证脚本
│
├── assets/
│   ├── maps/                      # phase 1-6 各 ~1010 张 + verified seed manifest
│   │   ├── phase{1..6}/*.txt
│   │   └── phase456_seed_manifest.json
│   └── images/                    # 编号 / 类别贴图
│
├── docs/
│   ├── FINAL_ARCH_DESIGN.md       # SAGE-PR 完整设计 (loss / 训练范式 / 部署)
│   ├── RFC_neural_arch_design.md  # 任务规则 / 硬件 / 监督来源 RFC
│   ├── PREVIEW_TRAJECTORY.md      # 通用预览脚本使用手册
│   └── 第二十一届智能车竞赛智能视觉组推箱子规则详解.md
│
├── scripts/
│   ├── maps/                      # 地图生成 / 验证 / 重建
│   ├── monitor_resources.py       # 训练资源监控
│   └── debug/                     # 临时调试脚本
│
├── tests/                         # pytest 回归测试
│
├── 专家分析/                       # 设计阶段专家分析 (D₄ 群论, 信息瓶颈等)
│
├── TODO.md                        # 当前迭代进度 + 完成条件
├── CLAUDE.md                      # Claude Code 协作约定
└── requirements.txt
```

---

## 5. 快速开始

### 5.1 环境

```bash
pip install -r requirements.txt
```

测试通过的环境: Python 3.12 + PyTorch 2.x + pygame 2.6 + numpy + RTX-class GPU (CUDA)。

### 5.2 手动游玩 / 看求解器跑

```bash
# 玩
python -m smartcar_sokoban.play --phase 4 --god

# 看求解器自动通关
python -m smartcar_sokoban.preview_failed --phase 6 --map phase6_11.txt --solver exact
```

### 5.3 SAGE-PR 完整训练流程

**a. 生成数据集** (V1 路线 = `plan_exploration_v3` + `MultiBoxSolver`, ~10 min):
```bash
mkdir -p runs/sage_pr/full_v5_v3
for p in 1 2 3; do
  python experiments/sage_pr/build_dataset_v5.py --phase $p --no-jepp --verify --workers 6 \
    --out runs/sage_pr/full_v5_v3/phase${p}_exact.npz
done
for p in 4 5 6; do
  python experiments/sage_pr/build_dataset_v5.py --phase $p --use-verified-seeds \
    --max-seeds-per-map 5 --no-jepp --verify --workers 6 \
    --out runs/sage_pr/full_v5_v3/phase${p}_exact.npz
done
```

输出: ~89k samples 跨 phase 1-6, **0 label_miss / 0 diverge** (翻译器 100% 验证).

**b. BC 训练** (80 epoch, GPU-resident, ~5 min on RTX 5060 Ti):
```bash
python experiments/sage_pr/train_sage_pr.py \
  --data runs/sage_pr/full_v5_v3/phase{1,2,3,4,5,6}_exact.npz \
  --tag v3_bc1 --phase-dist hard --epochs 80 --batch-size 1024 --lr 3e-4 --gpu-resident
```

**c. Eval (含老师对比)**:
```bash
# 快速 greedy eval
python experiments/sage_pr/evaluate_sage_pr.py --ckpt .agent/sage_pr/runs/v3_bc1/best.pt \
  --phases 1 2 3 4 5 6 --use-verified-seeds --max-maps 100 --top-k 4 \
  --mode v1 --external-explorer --record-teacher

# Rollout search (更准, 慢 50x)
python experiments/sage_pr/rollout_search_eval.py --ckpt .agent/sage_pr/runs/v3_bc1/best.pt \
  --phases 1 2 3 4 5 6 --use-verified-seeds --max-maps 100 \
  --beam 8 --lookahead 25 --step-limit 150 --mode v1
```

**d. DAgger 迭代** (并行 8 worker, ~2-5 min/round):
```bash
python experiments/sage_pr/dagger_v1.py --ckpt .agent/sage_pr/runs/v3_bc1/best.pt \
  --phases 3 4 5 6 --use-verified-seeds --max-maps 400 --max-seeds-per-map 2 \
  --workers 8 --out runs/sage_pr/dagger_v1_r1.npz

python experiments/sage_pr/train_sage_pr.py \
  --data runs/sage_pr/full_v5_v3/phase{1..6}_exact.npz runs/sage_pr/dagger_v1_r1.npz \
  --tag v3_dag1 --init-ckpt .agent/sage_pr/runs/v3_bc1/best.pt \
  --phase-dist hard --epochs 40 --batch-size 1024 --lr 1e-4 --gpu-resident
```

### 5.4 可视化

通用 trajectory 预览 — 见 `docs/PREVIEW_TRAJECTORY.md`:

```bash
# 看当前 V1+v3 老师在某图上的解法
python experiments/sage_pr/preview_trajectory.py \
  --recorders v1_v3 --map assets/maps/phase5/phase5_0001.txt

# 同图对比 V1 原始 vs V1+v3 拓扑配对效果
python experiments/sage_pr/preview_trajectory.py \
  --recorders v1_orig,v1_v3 --map assets/maps/phase5/phase5_0001.txt

# 批量看仍失败的图
python experiments/sage_pr/preview_trajectory.py \
  --recorders v1_v3 --fails-list runs/sage_pr/v5_v3_fails.json --phase-only 5
```

### 5.5 测试

```bash
python -m pytest tests/                  # 全量
python tests/test_solver.py phase1_001.txt   # 单图脚本风格
```

---

## 6. 关键设计决策

### 6.1 神经-符号混合

模型 **不学规则**: 合法性检查、死角检测、ID 排除推理、BFS 路径都由经典算法保证。
模型 **只学价值排序**: 在 ≤64 个合法候选里挑最优的那个。
这让 ~105K 参数足够支撑 95%+ 通关率 (小到可以 INT8 量化塞进 MCU)。

### 6.2 4-方向 + 严格 FOV

车只能 ±90° 旋转, 不允许对角推/平移 (除了文档里炸弹对角入墙的赛题特例)。
FOV 识别要求"紧贴 4-邻 + 精确正对" — 强迫探索器走"怼一下"的路径, 跟实车摄像头近距对准对齐。

### 6.3 V1 路线 ("先探索, 再推箱")

- **探索阶段**: `plan_exploration_v3` 跑一段固定 low-level 序列, 完成全部 ID 识别
- **推箱阶段**: 模型在 fully-observed 状态下每步选一个 push 候选 (rollout search 可选)
- 这条路线比 V2 (god-mode + 抑制场嵌入 inspect) 路径更高效, 模型只需学好 push 决策

### 6.4 翻译器 100% 准确

`(model candidate) ↔ (engine low-level actions)` 双向无损翻译。
脚本: `experiments/sage_pr/verify_translator.py` (跨 phase 全量 0 diverge 验证).

---

## 7. 已知问题

- Phase 5 (含炸弹) 通关率仍卡 ~72%, 需要更多 DAgger 加固
- 仍有 ~49 张 / 6197 张 verified 地图 (0.8%) 老师本身解不出 (拓扑死锁), 训练中跳过
- INT8 量化导出未完成 (P6 阶段)
- OpenART 实板 profiling 未完成 (P7 阶段)

详情见 `TODO.md`。

---

## 8. 参考文档

| 文档 | 内容 |
|------|------|
| `docs/RFC_neural_arch_design.md` | 任务规则 / 硬件约束 / 监督来源 (中性 RFC) |
| `docs/FINAL_ARCH_DESIGN.md` | SAGE-PR 完整架构 / 数学论证 / 训练范式 |
| `docs/PREVIEW_TRAJECTORY.md` | 通用预览脚本使用手册 |
| `专家分析/{1..6}.md` | 设计阶段专家分析 |
| `TODO.md` | 当前迭代进度 / 完成判定 |
| `CLAUDE.md` | Claude Code 协作约定 |

---

## 9. 许可证

待补充 (项目仍在迭代中)。

---

## 10. 遗留问题待合作者解决

### 10.1 ⚠️ Fully-observed BeliefState 信息泄漏 — 模型不会"探索"

**症状**:可视化模型决策(`preview_trajectory.py --recorders model_search --map phase6_11.txt`)能看到:
- 小车**一上来直接推箱**,**从来不转头看 entity**,从来不选 inspect 候选
- 看似"模型自己解出来",其实是因为模型从 t=0 就拿到了全部 box class_id / target num_id

**根因**:训练数据生成时强制 `fully_observed=True`:
```python
# build_dataset_v3.py / build_dataset_v5.py:
bs = BeliefState.from_engine_state(s, fully_observed=True)
# → Π 兼容矩阵是 ground-truth 恒等映射
# → X_cand[:, :, 108:118] (info_gain 段) 对 inspect 候选恒为 0
# → 模型学到的策略: 永远不选 inspect, 直接推
```
Eval 时(`rollout_search_eval.py --mode v1`)也用同样的 `fully_observed=True`,所以 eval 跟训练分布完美一致,得到 ~98% 通关率。但这个数字**不代表纯端到端神经网络的部署能力**。

**部署场景的两个分支**:
- ✅ **场景 A — 视觉模块负责识别**:OpenART 板上电后由 vision (YOLO/AprilTag) 360° 扫描识别全部 entity → 喂给模型一个全识别的 belief → 当前模型可以直接用。这是当前数据集训练目标实际匹配的场景。
- ❌ **场景 B — 模型自主识别**:模型需要选 inspect 候选(走到 viewpoint + 转向)主动识别 entity → **当前模型完全做不到**,需要重做数据集 + 重训。

**修复路径(给合作者)**:
1. 数据生成切到 `fully_observed=False`(不预设 Π,模型必须主动 inspect)
   - 修改 `build_dataset_v3.py:_worker_collect` 中 `BeliefState.from_engine_state(s, fully_observed=False)`
   - 老师老师除了 push moves 还要展开 `plan_exploration_v3` 的 inspect 步,产出 inspect 标签
2. 训练目标补充 inspect 决策(已有 candidate 类型,但 info_gain 实际非零需要 partial-obs)
3. Eval 改 `--mode v2` (`fully_observed=False + enforce_sigma_lock=True`),才能反映真实部署能力

**工作量估计**:重做数据集 + 重训 5-8 hr GPU + DAgger 迭代,约 1-2 天。

### 10.2 ⚠️ 训练样本严重不均衡 — "单 verified seed" 图欠拟合

**症状**:全量 eval 失败的 11-22 张图,绝大多数 `verified_seeds: [0]`(单 trajectory 训练)。例:phase5_0003 / phase6_0025 等。

**根因**:数据生成命令 `--use-verified-seeds --max-seeds-per-map 5`:
- 多 verified seed 的图(典型 ≥5 seeds): 5 trajectories 训练 → 模型鲁棒
- 单 verified seed 的图(verified_seeds=[0] only): 1 trajectory 训练 → ~15 个样本/图 → rollout 累积误差后状态 OOD → 模型崩
- 旧 2 位命名手作图(`phase6_11.txt` 等)大多单 seed,被前 100 张字母序 eval 漏过,误以为没问题

**修复路径**:
1. 给单 seed 图额外生成多个非 verified seeds(放宽老师超时 60→120s 提高成功率)
2. 或在 DAgger 阶段针对单 seed 图做 targeted oversample(我之前做了 5 轮 `dagger_targeted_p5_v[1-5]`,只对 phase 5 的 10-13 张 hard 图,phase 4/6 还没补)

### 10.3 ⚠️ Eval `--max-maps=100` 默认值掩盖了真实通关率

**症状**:之前报告 phase 6 = 97%,实际是字母序前 100 张(全是新生成 4 位命名图)。旧 2 位命名图(phase6_NN.txt)排第 1001-1011 位,**从未被 eval 测过**。

**根因**:`rollout_search_eval.py --max-maps` 默认 100,字符串排序 `"phase6_0001.txt" < "phase6_01.txt"`,导致只测前 100 张新图。

**修复路径**:
- ✅ 已修复:本次 README 数字是 `--max-maps 1010/1011` 全量
- 建议:把默认值改为 None (跑全部)或加显式 warning

### 10.4 推理路径不一致 — greedy vs beam search

- `evaluate_sage_pr.py` 默认 `top_k=1` greedy:phase6_11.txt seed=1 **fail** (52 macros)
- `rollout_search_eval.py beam=8 lookahead=25`:同图 **win** (34 macros / 130 low)
- 两者都是 "deterministic 纯 NN" 推理但能力差 ~10pp,**部署目标需要明确选用哪个**

OpenART 板上 beam=8 × lookahead=25 = 每决策 ~200 次 NN 前向 + engine 模拟,可能超 50ms 推理预算。需要测:
- 单次 NN 前向在 TFLM 上多久
- 是否可行 beam=4 / lookahead=10 之类的折中
- 或干脆只用 greedy 接受较低通关率

### 10.5 数据生成 / 老师 time_limit 不一致

- `build_dataset_v5.py` 默认 `time_limit=60s`
- `preview_trajectory.py:_record_solver_phase` 之前默认 30s (已修)
- 单图诊断时容易报"老师无解",其实是 time_limit 给少了

**已修复**:`preview_trajectory.py` 默认 60s 跟数据生成对齐。

---

## 11. 当前模型在哪 (给合作者)

```
.agent/sage_pr/runs/v3_large9/best.pt
```

`build_model_from_ckpt()` 会自动从 ckpt 中识别 large 还是 default 尺寸(194K vs 105K params)。
`v3_large9` 是 large 模型,基于 V1 + plan_exploration_v3 数据集 + 16 轮 DAgger + 5 轮 targeted DAgger 训练。

```bash
# 看模型在某张图上的可视化
python experiments/sage_pr/preview_trajectory.py \
    --recorders v1_v3,model_search \
    --map assets/maps/phase6/phase6_11.txt

# 全量 eval (~50 min/phase)
python experiments/sage_pr/rollout_search_eval.py \
    --ckpt .agent/sage_pr/runs/v3_large9/best.pt \
    --phases 5 --max-maps 1010 --use-verified-seeds \
    --beam 8 --lookahead 25 --mode v1
```
