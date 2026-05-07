# 当前训练攻关状态

> 时间点：2026-05-08 通宵到此为止的快照。
> 用途：留存现有进度、模型、数据，方便之后接手。
> **不**写"下一步要做什么"——单独看 TODO 或专家分析。

---

## 1. 当前架构

文件：`experiments/solver_bc/policy_conv.py` 的 `MaskedConvBCPolicy`。

```
输入: 254 维 flat obs
  ├─ 0..61    实体特征 (车位 + 5 箱 + 5 目标 + 3 炸弹 + 进度 + 距离)
  └─ 62..253  墙体 16×12=192 维 (row-major)

forward:
  ent  = obs[:, :62]
  walls = obs[:, 62:].view(B, 1, 12, 16)
  x = ReLU(Conv2d(1→16, 3x3, pad=1)(walls))
  x = ReLU(Conv2d(16→32, 3x3, pad=1, stride=2)(x))   # → (B, 32, 6, 8)
  wall_emb = ReLU(Linear(1536→wall_emb_dim)(x.flatten(1)))
  h = cat([ent, wall_emb], dim=1)
  h = ReLU(Linear(126→hidden)(h))
  h = ReLU(Linear(hidden→hidden)(h))
  logits = Linear(hidden→54)(h)
  # mask 时把非法动作位置 -∞ 后 softmax
```

| 配置 | hidden | wall_emb | 总参数 | INT8 大小 | 推理 (cuda bs=512) |
|---|---|---|---|---|---|
| 当前最佳 | **1024** | 64 | ~1.2 M | ~1.2 MB | <1 ms |
| 备选 | 512 | 64 | ~215 K | ~215 KB | 0.32 ms |

**TFLite Micro 友好 op 限定**：仅 Conv2D / FC / ReLU / Reshape / Concat，无 BN/LayerNorm/Dropout。

**当前最佳 checkpoint**：`.agent/runs/combined_v4final_h1024/best.pt`（hidden=1024，183K samples 训出）

---

## 2. 训练数据

全部用 `MultiBoxSolver(strategy='auto')`（BestFirst weighted A\* 1.5×OPT）当老师，per-step 调用。

| Phase | verified maps | 训练 episodes | 训练 samples | 备注 |
|---|---|---|---|---|
| 1 | 1010 / 1010 (100%) | 3030 | 35,520 | 单 seed 老 build |
| 2 | 962 / 1010 (95%) | 2886 | 31,209 | 单 seed 老 build |
| 3 | 990 / 1010 (98%) | 2880 | 53,948 | 单 seed 老 build |
| 4 v4 BF | 1011 / 1011 (100%) | 973 | 22,119 | 1 verified seed/map |
| 5 v4 all-seeds | 1006 / 1010 (99.6%) | 3580 | 47,548 | **5 seeds/map** |
| 6 v4 all-seeds | 990 / 1011 (97.9%) | 3418 | 58,942 | **5 seeds/map** |

**最大合并数据集**：~250K samples（phase 4 单 seed + phase 5/6 多 seeds，phase 1-3 老 build）

---

## 3. 评估指标

`evaluate_bc.py` 在每 phase 1010-1011 张图上 deterministic rollout，每张图用 `phase456_seed_manifest.json` 里的 verified seed（1 个/图）。

### Greedy（直接 argmax）

最近一次完整 eval：`combined_v4final_h1024`（phase 4 v4 + phase 5/6 v4 BF, ~150K samples, 80 epoch）

| Phase | win_rate | 距目标 |
|---|---|---|
| 1 | **99.90%** | 95% ✓ |
| 2 | **95.45%** | 95% ✓ |
| 3 | 60.69% | 95% (-34pp) |
| 4 | 34.72% | 95% (-60pp) |
| 5 | 39.70% | 95% (-55pp) |
| 6 | 41.35% | 90% (-49pp) |

### Branch search inference

测试时不贪心，对低置信度步骤分叉 top-k 试，命中即赢。**不可部署**（OpenART 上 budget≤8）。

| Phase | greedy | b256 top_k=3 | b512 top_k=4 | verify cap |
|---|---|---|---|---|
| 2 | 95.7% | **99.60%** | — | 95.2% |
| 3 | 63.7% | **95.25%** | 94.75% | 98.0% |
| 4 | 34.7% | **81.11%** | — | 100% |
| 5 | 41.8% | 75.15% | **76.53%** | 99.6% |
| 6 | 44.0% | 80.81% | **81.60%** | 97.9% |

Branch budget=256 大致 = PC CPU 5-15 min/eval，完全无法部署到 OpenART。

### Per-phase train_acc 诊断（v3 model 上）

| Phase | train_acc | greedy rollout | 0.93^步数 |
|---|---|---|---|
| 1 | 99.65% | 100% | 0.9965^16=0.95 |
| 2 | 98.64% | 95.25% | 0.9864^12=0.85 |
| 3 | 93.47% | 63.17% | 0.9347^16=0.34 |
| 4 | 93.25% | 17.90% | 0.9325^18=0.28 |
| 5 | 92.16% | 33.80% | 0.9216^16=0.27 |
| 6 | 92.89% | 26.90% | 0.9289^19=0.25 |

per-step 93% × 16 步 trajectory ≈ 30% 通关率，**这是 BC 数学上限**——不靠搜索类算法兜底打不破。

---

## 4. 主要做了什么（按时间）

### P0 基线审计
- P0.1：跑当前 BC prototype phase 6 baseline → 9.33% (MLP h=256, 50 maps × 3 seed, 2K samples)
- P0.2：IDA\* 严格最优解 phase 6 前 50 张 → 仅 25/50 解出 (50%, 60s 限制)
- P0.3：跳过（无 phase6_best.zip RL ckpt）

### P1 地图生成器诊断 + 重 verify
- P1.1：诊断 → 生成器无 IDA\* 闸、用步数当推数过滤、phase 6 出现 4 推 trivial 图
- P1.2：写 `scripts/maps/verify_optimal.py`，用 IDA\*+BestFirst 重新验证
  - 第一轮 (3 seeds: 7/42/137)：phase 4-6 仅 56-66% 通过
  - **关键发现**：同一张图 seed 不同 → box/target ID 配对不同 → 解题难度不同
  - V4 (10 seeds): phase 1-6 全部 100/95/98/100/99.6/97.9% 通过
- 写 `scripts/maps/find_all_seeds.py` → 每张图找出所有可解 seed (中位 5-10 个/图)

### P2 架构升级
- 写 `experiments/solver_bc/policy_conv.py` MaskedConvBCPolicy
- 容量扫：MLP h=256 (9%) → Conv h=256 (11%) → Conv h=512 (18%) → Conv h=1024 (94% phase 1)
- 改 `train_bc.py` / `evaluate_bc.py` / `branch_search.py` / `self_improve_loop.py` 全部支持 `--policy conv`

### P3 数据集构建
- 写 `experiments/solver_bc/build_dataset_v2.py` 消费 verified.json
- 写 `experiments/solver_bc/build_dataset_v3.py` 消费 all-seeds.json (多 seed/map)
- 改 `teachers.py` 加 `strategy` 参数 (auto/ida/best_first)，加 `solver_ida` teacher 类型

### P4 联合训练
- 写 `scripts/p4_combined_train.py` 合并多 phase npz 训单一模型
- 跨 phase 迁移：单 phase 训 → 合训提升 phase 6 +5-10pp
- 容量扫：h=512 → h=1024 phase 1/2 +3.7pp / +1.8pp，phase 6 几乎不动（数据瓶颈）
- Epoch 扫：80 → 200 完全收敛 (best_val_loss 0.297 不变)

### P5 评估 + Branch search
- Branch search inference (`branch_search.py`)：对低置信度步骤分叉 top-k
  - 关键转折：phase 3 直接 63% → **95%** ✓ (b256, top_k=3)
  - phase 4-6 也涨 +30-46pp，但只到 verify cap

### 副产品
- `scripts/monitor_resources.py` CPU/GPU/VRAM 5s 监控
- `scripts/p3_p4_p5_pipeline.py` 一键 build+train+eval
- `scripts/maps/verify_all_phases.py` 链式 phase 1-5 verify
- `scripts/maps/recover_low_push_maps.py` 救回 push_too_low 图
- `scripts/maps/merge_v4_to_manifest.py` 把 v4 verified seeds 合进 phase456_seed_manifest
- 把 `MultiBoxSolver._car_bfs_*` 从 8 向改回 4 向（用户要求）
- 修 `branch_search.py` 默认 backend = cpu（绕过 gpu_push_env 旧 bomb 表 bug）

---

## 5. 已写未走的死路

- **GPU rollout backend** (`experiments/gpu_sim/gpu_push_env.py`): bomb 表按 N_DIRS=4 建，跟当前 54 维动作空间 (N_BOMB_DIRS=8) 不兼容。改默认 cpu rollout 绕开。
- **self_improve_loop**: 没真正跑通，理论上应能修 phase 3-6 distribution shift，但每次 branch search 加重训成本太高，没在本轮跑。
- **量化 / TFLite 导出**: 没做。手头最佳 fp32 权重在 `.agent/runs/combined_v4final_h1024/best.pt`。

---

## 6. 现在卡在哪

**不是模型问题，是架构归纳偏置错配**：

1. 当前 obs 把 192 维墙体 flat → conv，但箱子位置作 62 维向量进 MLP head → **失掉了对箱子位置的平移等变性**。每张多箱图 BC 都要重学 (位置1, 位置2, 位置3) 组合 → distribution shift 严重。

2. 输出是 54 维扁平 softmax (5 box × 4 dir + 5 target + 5 explore + 3 bomb × 8 dir)，**箱子动作 slot 跟 box 在数组里的序号绑死**。同一张图 ID 排序换 → 学到的 slot 映射全错。

3. BFS 距离场没作为输入通道 → 网络要靠 conv 自己学全图最短路，深度不够。

合 phase 1-3 (1-2 箱) 不出问题，phase 4-6 (3 箱 + 炸弹) 直接被这三件事联合卡死。

per-step 93% 上 16 步 → 0.93^16 ≈ 30%，**branch search 是补救手段不是根治**——budget=256 才到 80%，部署 budget=4-8 大概只能 50-60%。

---

## 7. 文件清单

### 训练数据 (`.agent/runs/` gitignored)
- `p1_conv_h512/phase1_v2.npz` — 35K
- `p2_conv_h512/phase2_v2.npz` — 31K
- `p3_conv_h512/phase3_v2.npz` — 54K
- `p4_v4_bf/phase4_v2.npz` — 22K (v4 BestFirst, 1 seed)
- `p5_v4_all/phase5.npz` — 47K (v4 all-seeds, 5/map)
- `p6_v4_all/phase6.npz` — 59K (v4 all-seeds, 5/map)

### 模型 checkpoint (`.agent/runs/`)
- `combined_v4final_h1024/best.pt` — 当前 greedy 最佳
- `combined_all_v2_h1024/best.pt` — 老版 v2 数据训练
- `combined_v3_h1024/best.pt` — v3 数据 + h=1024
- `p3_conv_h512/phase3_train/best.pt` — 单 phase pipeline 产物

### 验证 manifest (`assets/maps/`)
- `phase{1..6}_verified.json` — 第一轮 (3 seeds, 严格 IDA\*)
- `phase{4..6}_verified_v2.json` — 救回 push_too_low
- `phase{4..6}_verified_v3.json` — BestFirst strategy
- `phase{4..6}_verified_v4.json` — **10 seeds，pass rate 100/99.6/97.9%**
- `phase{4..6}_all_seeds.json` — 每图所有可解 seed
- `phase456_seed_manifest.json` — eval 用，已合并 v4 verified seeds (.bak 是旧版)

### 评估日志 (`.agent/eval/`)
所有 `combined_*_phase{N}.log` 和 `p{N}_branch*.json`。最有意义的是：
- `combined_v4final_h1024_phase{1..6}.log` — 最新 greedy 数字
- `p{4,5,6}_v4final_branch.json` — 最新 branch search 数字 (b256)
- `p{5,6}_v4_branch_b512.json` — 大 budget 上限测试

### 脚本 (`scripts/`)
全部新写的脚本，仓库根 commit 历史里能找到。
