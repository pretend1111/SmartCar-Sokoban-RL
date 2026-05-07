# 推箱子赛题专用神经架构 — 终极整合方案

> 本文档基于 6 位 AI 研究者的独立提案（`专家分析/1.md` ~ `6.md`）综合整理。  
> 任务：把分歧点收敛、共识点强化、各家优势创新整合，给出可直接实施的最终架构。  
> 项目代号：**SAGE-PR（Symbolic-Aided Grid-Equivariant Policy Ranker）**

---

## 0. 摘要

**最终架构**：`显式 Belief 状态层 + 合法候选生成器 + 候选集合等变评分器 + 浅层引导搜索 + BFS 低层执行器`

```
                    ┌──────────────────────────────────────┐
[YOLO/CV/AprilTag] →│   Belief State Manager (符号)         │
                    │   • 14×10 当前墙图 (内墙可炸)         │
                    │   • 实体位置 + 已知/未知 ID           │
                    │   • Π 矩阵 (软配对信念)               │
                    │   • FOV 累积可见 / 上次可见时刻       │
                    │   • ID 排除推理 (确定性)              │
                    └────────────┬─────────────────────────┘
                                 ▼
                    ┌──────────────────────────────────────┐
                    │   领域特征预计算（≤2 ms）            │
                    │   • BFS 可达性 + 到任意格距离        │
                    │   • Reverse-push 距离场             │
                    │   • Push 推荐方向场（流场）          │
                    │   • 死角 mask（角 + 边）             │
                    │   • 信息增益热度图（FOV scan）        │
                    └────────────┬─────────────────────────┘
                                 ▼
                    ┌──────────────────────────────────────┐
                    │   候选动作生成器（符号，≤1 ms）       │
                    │   枚举 ≤64 个合法 macro action：      │
                    │   • push(box_i, dir, max_run)        │
                    │   • push(bomb_k, dir, special_diag)  │
                    │   • inspect(viewpoint, heading)      │
                    │   • return_garage（终局可选）         │
                    │   非法 / 死锁动作 → mask=−∞           │
                    └────────────┬─────────────────────────┘
                                 ▼
       ┌──────────────────────────────────────────────────┐
       │   SAGE-PR 神经候选评分器（INT8, ~150K params）    │
       │                                                  │
       │   ┌─ Grid Encoder ─────────────────────────────┐ │
       │   │  X_grid ∈ R^{10×14×30}                     │ │
       │   │  → 6 × DepthwiseSeparable Block (32 ch)   │ │
       │   │  → GAP → z_grid ∈ R^{96}                   │ │
       │   └────────────────────────────────────────────┘ │
       │                          │                       │
       │   ┌─ Candidate Set Encoder ───────────────────┐ │
       │   │  X_cand ∈ R^{64×128}（每候选 128 维特征） │ │
       │   │  → Shared Conv1×1 MLP (128→96→96)        │ │
       │   │  → e_i ∈ R^{96}, z_set = mean_i(e_i)      │ │
       │   └────────────────────────────────────────────┘ │
       │                          │                       │
       │   ┌─ Context Fusion + Score Head ─────────────┐ │
       │   │  c = MLP([z_grid, z_set, u_global])        │ │
       │   │  s_i = w_score · ReLU(e_i + c) + mask_i    │ │
       │   │  → π(a_i|s) = softmax({s_i})              │ │
       │   │                                             │ │
       │   │  Aux heads：                                │ │
       │   │    V(s) ∈ R: 局面价值                       │ │
       │   │    p_dead(s, a_i) ∈ [0,1]: 动作死锁概率   │ │
       │   │    p_progress(s, a_i) ∈ R: 估计剩余推数   │ │
       │   └────────────────────────────────────────────┘ │
       └──────────────────────┬───────────────────────────┘
                              ▼
            ┌──────────────────────────────────────────┐
            │   Neural-Guided Beam Search (B=3, D=3)   │
            │   J(a₀:D) = -Σ α·logπ(aₜ) + λ_v·V(s_D)   │
            │             + λ_d·1{死锁} - λ_i·IG(aₜ)   │
            │   选最优首动作                            │
            └──────────────────────┬───────────────────┘
                                   ▼
                    ┌──────────────────────────────────────┐
                    │   BFS 低层执行器                      │
                    │   • 把 macro 翻译成低层动作序列      │
                    │   • 实时碰撞与重规划                  │
                    └──────────────────────────────────────┘
```

**关键参数**：

| 项 | 值 | 备注 |
|---|---|---|
| 输入 grid | `10 × 14 × 30` | 已裁掉外圈墙 |
| 输入候选集 | `64 × 128` | padding 到 64 |
| 主干通道 | `32 → 32 → 32 → 48 → 48` | DepthwiseSeparable |
| 总参数量 (fp32) | **~140 K** | |
| INT8 权重 | **~140 KB** | < RFC 500 KB |
| 推理 MAC | ~5.7 M | |
| OpenART mini 时延（保守） | **15–35 ms** | < RFC 50 ms |
| 推理 RAM | **150–400 KB** | < RFC 2 MB |
| 训练范式 | BC + DAgger + QAT 三段 | + soft Q + hard negative |
| 部署 | 单 TFLite Micro，必要时手写 CMSIS-NN | |

**预期性能（综合多家估计的上下界）**：

| Phase | 目标 | 推断下界 | 推断上界 |
|---|---|---|---|
| 1 | ≥95% | 99% | 100% |
| 2 | ≥95% | 98% | 100% |
| 3 | ≥95% | 95% | 99% |
| 4 | ≥95% | 90% | 97% |
| 5 | ≥95% | 88% | 95% |
| 6 | ≥90% | 86% | 93% |

phase 4-6 是核心攻坚区，也是最不确定区。

---

## 1. 6 份提案对比与判断

### 1.1 各家路线总览

| # | 名称 | 核心思想 | 模型骨架 | 参数 | 推理时延 | 关键创新 |
|---|---|---|---|---|---|---|
| 1 | LAMP-Net | 神经-符号双头评分 | 8×DSConv + Spatial+Global head | 273K | 12-30ms | belief matrix 渲染到 grid 通道；soft Q label |
| 2 | SpatialPolicyNet | 全卷积 ResNet + BFS距离场作输入 | 6-block ResNet | 265K | 31-52ms | D₄ 数学论证最严谨；策略头 1×1 conv 全卷积 |
| 3 | GridNet | 极致瘦身 FCN + 显式领域特征 | 5×DSConv | 13K | 20ms | 参数量最小；推送方向流场作输入 |
| 4 | ConvGRU 微型空间循环 | DSConv + ConvGRU | 60K | 不明 | **使用 RFC 明确禁用的 GRU**，机器翻译式语言 |
| 5 | CandidateRanker | 神经-符号变长候选评分 | DSConv + Shared MLP | 35-65K | 15-35ms | 变长候选集；事件触发推理；项目管理最完整 |
| 6 | SAGE-Push (PushRank-μ) | 候选集合等变排序 + 浅层搜索 | DSConv + DeepSets head | 140K | 8-25ms | belief matrix Π；macro action（多步推送）；浅层 beam search |

### 1.2 我的水平评估（与用户暗示一致）

**强（核心参考）**：

- **方案 2** — **数学论证最深**：D₄ 群等变性、信息瓶颈、Rademacher 边界、感受野计算逐层推导。是论证质量最高的一份。但实现细节（GroupNorm-fold）有点冒险。
- **方案 5** — **工程实施最完整**：从 RFC 解析、研究综述、风险表、人员/预算/时间线/里程碑。把"如何做"写得最具体。
- **方案 6** — **神经-符号融合最彻底**：候选集合等变、belief matrix 把 N! 配对压缩成 N²、macro action 多步推送、轻量 beam search。

**中（值得参考具体技巧）**：

- **方案 1** — 跟方案 2 大方向相似但更工程化，一些 head 设计（如 spatial+global 双头融合）有趣。
- **方案 3** — 思路与 2 高度雷同但参数量极致瘦身。13K 太小，phase 6 可能欠拟合。

**弱（基本不取）**：

- **方案 4** — 用了 RFC 明确禁止的 ConvGRU；表达机器翻译化（"系统级降级预警"等）；理论错置（说 LSTM 不擅长空间任务但又用 GRU）。可忽略。

### 1.3 五家共识（这是真理）

下列 5 件事，方案 1, 2, 3, 5, 6 不约而同提出，可以直接当作铁律：

1. **必须裁掉外圈墙，输入 14×10 而非 16×12**（节省 27% 算力，零信息损失）
2. **BFS 距离场 / 推送距离场必须作为输入通道**（不是后处理）
3. **死角 mask 作为输入通道 + 辅助头双重监督**（phase 4-6 的关键）
4. **训练范式必须是 BC + DAgger + QAT 三段式**，单纯 BC 在 phase 4-6 上必然 distribution shift
5. **D₄ 数据增强（水平翻转 + 垂直翻转 + 180° 旋转）+ ID 重命名增强**

### 1.4 三家以上同意（强建议）

6. **belief matrix Π(i,j) 显式维护**（方案 1, 5, 6）— 避免让网络从数据里"学排除推理"
7. **候选集合等变 / 变长动作空间**（方案 5, 6）— 比固定 54 类 softmax 更强的归纳偏置
8. **Macro action 多步推送**（方案 1, 5, 6）— 把"推到拐点 / 进入死角风险点为止"作为单个动作
9. **浅层 beam search 推理**（方案 1, 2, 5, 6）— top-k + value lookahead，简单高效
10. **soft Q label 而非 one-hot**（方案 1, 6）— 多个近似最优动作不冲突

### 1.5 主要分歧点（我做出选择）

#### 分歧 A：动作输出方式

- **空间策略头 H×W×K**（方案 1, 2, 3）：保持平移等变；每格 5-9 logits
- **候选集合排序 64×1**（方案 5, 6）：变长动作空间，对实体数变化稳健

**选择 → 候选集合排序**

理由：
- 推箱子动作不只是"在某格子推某方向"，还包括 inspect (探索观察)、return_garage (入库)、对角炸弹推送等异构动作；空间头难以统一表达
- 箱子 / 炸弹数从 1 → 5 变化时，空间头依然要把每格的 5 个 channel 都 mask 掉非箱子格（浪费）；候选排序天然只看"实际存在的箱子"
- 集合等变性（Deep Sets / Pointer-style）对 Sokoban 的"实体可交换"先验匹配最好

但**保留空间编码作为骨干** — Grid Encoder 仍用 CNN，只是 head 切到候选集合上。

#### 分歧 B：模型规模

- 方案 3：13K（5 blocks × 32 ch）
- 方案 5：35-65K
- 方案 6：140K
- 方案 1, 2：265-273K

**选择 → ~140K（参数预算 30 万以内的中位数，留余量）**

理由：
- 13K 在 Sokoban 这种长程组合任务上几乎肯定欠拟合（Brock et al. NF-ResNet 经验：MCU 类问题 50K+ 才稳）
- 270K 在 OpenART mini 上推理 30+ ms 接近预算上限，没空间给 lookahead
- 140K 在 INT8 后约 140-180KB，远低于 500KB；同时为可能的 D₄ 测试时增强 / beam search 留出 30+ ms 余量

#### 分歧 C：是否在网络内做规划（VIN-style）

- 方案 5 提到 VIN 但建议 不采用
- 方案 2 没正式讨论但暗示"用 BFS 外置等价"

**选择 → 不在网络内规划，BFS 距离场作为输入**

理由：
- VIN 需要可微的迭代（Bellman update 嵌入），算子超出 TFLite Micro
- 经典 BFS 在 140 节点上 < 0.5 ms，比让 16 层 CNN 重建 BFS 等价物便宜 ~2 个数量级
- 这是方案 2 和 5 都引用的"信息瓶颈论证"：BFS 距离场已经是任务相关的近似充分统计量

#### 分歧 D：BatchNorm / GroupNorm 替代

- 方案 1：用 fixed scale + bias
- 方案 2：GroupNorm 训后 fold 进 conv 权重（冒险）
- 方案 3：纯 He init + 残差，不做 normalization
- 方案 5：Fixup Init
- 方案 6：He init + 残差

**选择 → Fixup Initialization (方案 5 备选 B)**

理由：
- Fixup 专为无 BN 设计，残差路径上加可学的 scalar，可 fold 进 conv weights
- 比 GroupNorm fold 风险小（GN 严格说不能精确 fold，需要训练集均值近似）
- 比"什么都不加"训练稳定性更好

如训练发散，fallback 到方案 3 的"纯 He init"。

#### 分歧 E：搜索策略

- 方案 1, 2, 6：浅层 beam search (B=3-4, D=2-3)
- 方案 3：1-step lookahead (top-K + value)
- 方案 5：top-2 lookahead 仅在低置信度时触发（事件触发）

**选择 → 事件触发的浅层 beam search (B=3, D=3)**

理由：
- 默认 greedy（直接 argmax）以保 5 Hz 主循环
- 当 top-1 logit 与 top-2 差 < 阈值（典型 0.5）时启用 beam（多耗 ~30 ms）
- 当 deadlock head 输出 > 0.3 时启用 beam（防灾难）
- 命中率 < 30%，平均开销可控

---

## 2. 状态层（Symbolic Belief State）

### 2.1 完全沿用方案 6 的 belief state 结构

```python
class BeliefState:
    M: np.ndarray          # 当前墙图 [10, 14] (内墙可能被炸过)
    M_init: np.ndarray     # 初始墙图 [10, 14]（提示哪些墙是"还可炸的"）
    p_player: tuple        # 车位置 (x, y)
    theta_player: int      # 车朝向 0-7
    boxes: List[Entity]    # 剩余箱子
    targets: List[Entity]  # 剩余目的地
    bombs: List[Position]  # 剩余炸弹
    K: dict                # 已识别 ID 集合 {box_idx: class_id, target_idx: num_id}
    Pi: np.ndarray         # 软配对矩阵 [N_box, N_target]，0/1 二值或 概率
    visited_fov: np.ndarray # 累积已扫描格 [10, 14]
    last_seen_step: dict   # {entity: 上次可见的步数}
```

### 2.2 belief 更新规则

每次 YOLO 给出新识别结果或 FOV 改变时：

1. **写入新 ID**：`K[box_i] = class_id`（首次识别）
2. **belief 收缩**：`Π[i, j] = 1[K[box_i] == K[target_j] OR (未识别但仍兼容)]`
3. **ID 排除推理**（关键，零参数零神经网络）：

```python
def infer_remaining_ids(K_box, K_target, N):
    """N-1 个箱子已识别 → 第 N 个箱子的 ID 唯一确定"""
    while True:
        unassigned_boxes = [i for i in range(N) if i not in K_box]
        if len(unassigned_boxes) == 1:
            used_ids = set(K_box.values())
            all_ids = set(range(10)) - {fixed_ids}  # 排除明确不在的
            remaining = list(all_ids - used_ids)
            if len(remaining) == 1:
                K_box[unassigned_boxes[0]] = remaining[0]
                continue
        # 同样对 target 做
        ...
        break
```

这是 RFC §2.4 直接允许的逻辑，**不让神经网络学**。

### 2.3 领域特征预计算（≤2 ms）

每次 belief 更新或玩家位置改变时重算（事件触发）：

| 特征 | 算法 | 复杂度 | 估时 |
|---|---|---|---|
| `player_bfs_dist` | 4 邻接 BFS from player | O(140) | 0.3 ms |
| `reachable_mask` | 同上副产物 | O(140) | 0 (复用) |
| `push_dist_field[i]` | reverse-push BFS for box_i 到 target_match[i] | O(N_box × 140) | 1.5 ms |
| `push_dir_field` | argmin over push_dist_field 邻居 | O(140 × 4) | 0.5 ms |
| `deadlock_mask` | 静态 corner + edge-line 检测 | O(140) | 0.3 ms |
| `info_gain_heatmap` | raycasting from grid cells, 模拟新观测的 belief 收缩 | O(viewpoint × ray) | 1.0 ms |
| **总计** | | | **~3.6 ms** |

实测可能稍高，但远低于神经网络推理时延（15-35 ms）。

---

## 3. 候选动作生成器（Symbolic）

### 3.1 候选类型与上界

| 类型 | 形式 | 数量上界 |
|---|---|---|
| 推箱（正交） | `push_box(i, d)` | 5 boxes × 4 dirs = **20** |
| 推箱（多步 macro） | `push_box(i, d, run_length=k)` | 与上同（每个 (i,d) 选一个 k）= **20** |
| 推炸弹（正交 + 对角） | `push_bomb(k, d)`，d ∈ 8 方向 | 3 bombs × 8 = **24** |
| 探索观察 | `inspect(viewpoint, heading)` | ≤ **8**（保留 top-8 信息增益视点） |
| 入库 | `return_garage` | 终局唯一可选 |
| **合计** | | **≤ 64** |

padding 到 64 即可（方案 6 的设定）。

### 3.2 合法性 mask 流程

对每个候选：

```
push(i, d) 合法 iff:
   - box[i] 还存在
   - box[i] 朝 d 的邻格在地图内
   - 把 box[i] 沿 d 推一格不会撞墙（除非是炸弹的对角推墙特例）
   - 推完不会进入静态死锁角（corner 死角 + edge 边线死角）
   - 车能到达推位（player_bfs_dist[push_pos] != ∞）
   - 推链（如果推到其他实体）能完整完成

inspect(viewpoint, heading) 合法 iff:
   - viewpoint 可达
   - 在该位姿下至少有一个未识别实体进入 FOV 且无遮挡
   - 信息增益 > 0
```

非法候选直接 mask=−∞。

### 3.3 macro 推送的"停止条件"

`push_box(i, d, k)` 表示从当前位置沿 d 推 k 格。**不是盲推**，而是直到下列任一发生：

- `k = max_run`（如 3 格上限）
- 推到目标格（配对完成）
- 下一格会进入死锁
- 推到拐点（有岔路）
- 距离场出现"该转方向"的拐点（push_dist_field 沿 d 不再下降）

这把"15-40 步推送"压缩到典型 5-10 个 macro action，决策频率降低。

### 3.4 候选特征向量（128 维）

```
| 维度 | 含义 |
|---|---|
| 0-7 | 类型 one-hot：push_box / push_bomb / push_diag / inspect / return / padding (8) |
| 8-25 | 对象描述：类型、(x,y)、是否已知 ID、ID embedding（0-9 one-hot 浓缩为 10 维 +）、是否在 FOV、是否遮挡 (18) |
| 26-37 | 方向 / 宏步：4 正交 + 4 对角 + 步长归一化 + 推链深度 (12) |
| 38-53 | 配对：候选箱对各目标的 Π(i,j)（5 维）、最可能 target、匹配熵、是否唯一确定（16） |
| 54-65 | 路径代价：车到推位 BFS、转向 cost、推送步数、推完后可达性 (12) |
| 66-81 | 推送距离场：推前 / 推后到目标的 reverse-push distance、差分（是否更近）(16) |
| 82-95 | 死锁 / 合法性：静态死角、动态冻结、推链阻塞、不可逆推送风险 (14) |
| 96-107 | 炸弹特征：可毁墙数、连通分量增益、爆破后路径增益 (12) |
| 108-117 | 信息增益：预期可见实体数、ID 熵下降、能否排除最后未知 ID (10) |
| 118-127 | 局部 3×3 / 5×5 邻域 + 全局标量 padding (10) |
```

这 128 维全部由经典算法计算，**网络只学这些特征的非线性组合**。

---

## 4. 神经评分器 SAGE-PR（详细层表）

### 4.1 输入

```
X_grid:   [10, 14, 30]   空间张量，详见 §4.2
X_cand:   [64, 1, 128]   候选集合（padding 到 64）
u_global: [16]            全局标量
mask:     [64]            外部送入，softmax 前置 −∞
```

### 4.2 X_grid 30 通道

继承方案 1, 2, 6 的共识：

| # | 名称 | 类型 |
|---|---|---|
| 0 | wall_current | binary |
| 1 | wall_init | binary（提示哪些墙原本存在） |
| 2 | reachable | binary |
| 3 | player_pos | binary |
| 4-11 | player_dir_onehot | 8 维 one-hot |
| 12 | box_present | binary |
| 13 | box_known_mask | binary（只在已知 ID 的箱格为 1） |
| 14 | target_present | binary |
| 15 | target_known_mask | binary |
| 16 | bomb_present | binary |
| 17-21 | box_id_inferred | 5 通道（仅画在已知 ID 的箱格上，箱 → target 的 push_dist 编码） |
| 22 | player_bfs_dist | float ∈ [0, 1]（tanh 归一化） |
| 23 | push_dist_field_min | float ∈ [0, 1]（所有已配对箱-目标对的 min） |
| 24-27 | push_dir_field_NESW | 4 通道（流场，每格"该往哪推"的 logit） |
| 28 | deadlock_mask | binary |
| 29 | info_gain_heatmap | float ∈ [0, 1] |

通道数 30。

### 4.3 Grid Encoder

Fixup Init，无 BatchNorm：

```
Input X_grid: [10, 14, 30]
  ↓
Conv 3×3, 30 → 32, padding=1, ReLU       [params 8704]
  ↓
ResBlock(DepthwiseSeparable, 32 → 32) ×2  [params ~2.7K]
  ↓
ResBlock(DSConv, 32 → 48, dilation=2) ×1   [params ~3.4K, 扩感受野]
  ↓
ResBlock(DSConv, 48 → 48) ×2              [params ~5.6K]
  ↓
GlobalAvgPool                              → [48]
  ↓
FC 48 → 96, ReLU                           [params 4704]
  ↓
z_grid: [96]
```

每个 ResBlock：

```python
def ds_resblock(x, in_ch, out_ch, dilation=1):
    y = DepthwiseConv2D(3×3, dilation=dilation, padding=dilation)(x)
    y = ReLU(y)
    y = Conv2D(1×1, in_ch → out_ch)(y)
    if in_ch == out_ch:
        return ReLU(x + y)  # Fixup-style: 学习 scalar α 调节 y
    else:
        return ReLU(y)  # transition block
```

Fixup-style：在残差路径上加 `α` （初始化为 0），训练时学习。导出前 fold 进 conv weights。

**主干参数估计**：~25 K
**主干 MAC**：~3.2 M

### 4.4 Candidate Set Encoder（Deep Sets）

```
Input X_cand: [64, 1, 128]
  ↓
Conv 1×1, 128 → 96, ReLU                   [params 12384]
  ↓
Conv 1×1, 96 → 96, ReLU                    [params 9312]
  ↓
e_i: [64, 1, 96]
  ↓
GlobalAvgPool over 64 dim                  → z_set: [96]
```

**候选编码器参数**：~22 K
**候选 MAC**：~1.4 M

### 4.5 Context Fusion

```
h = concat([z_grid, z_set, u_global])  # [96 + 96 + 16 = 208]
  ↓
FC 208 → 128, ReLU                          [params 26880]
  ↓
FC 128 → 96, ReLU                           [params 12384]
  ↓
c: [96]
```

**Fusion 参数**：~39 K

### 4.6 Score & Aux Heads

将 c 广播 add 到每个候选 e_i：

```
e_tilde_i = e_i + c                         # [64, 1, 96]
  ↓
Conv 1×1, 96 → 96, ReLU                     [params 9312]
  ↓
Conv 1×1, 96 → 1                            [params 97]
  ↓
score_logits: [64]
  ↓
mask + softmax → π(a_i | s)
```

**辅助头**（共享 96 维 e_tilde_i 输入）：

```
deadlock_head:    Conv 1×1, 96 → 1, sigmoid        # 每候选死锁概率
progress_head:    Conv 1×1, 96 → 1                  # 估计剩余推数
info_gain_head:   Conv 1×1, 96 → 1                  # 学经典 IG 估计的回归

value_head:       FC 96 → 32 → 1                    # 全局价值 V(s)
```

辅助头参数：~3 K

### 4.7 总参数与 MAC

| 模块 | 参数（fp32） | INT8 大小 | MAC |
|---|---:|---:|---:|
| Grid encoder | ~25 K | 25 KB | 3.2 M |
| Candidate encoder | ~22 K | 22 KB | 1.4 M |
| Context fusion | ~39 K | 39 KB | 0.04 M |
| Score head | ~10 K | 10 KB | 0.6 M |
| Aux heads | ~3 K | 3 KB | 0.02 M |
| **合计** | **~99 K** | **~99 KB** | **~5.3 M** |

加 per-channel scales、tflite metadata、INT8 校准统计 ≈ **130-150 KB**，远低于 RFC 的 500 KB 上限。

---

## 5. 训练范式（三段式 + 增强）

### 5.1 数据生成

| 数据来源 | 用途 | 比例 |
|---|---|---|
| IDA* | phase 1-3 全部 + phase 4 verified-seed | 30% |
| BestFirst (1.5×OPT) | phase 4-6 主体 | 50% |
| AutoPlayer | 探索 / 入库 / 紧急回退 | 10% |
| **合成 hard cases** | 4-5 箱 OOD、对角炸弹、ID 高熵态 | 10% |

每张地图 × 3 种 ID 配对 seed = 18000 episodes。每个 episode 平均 25 (s, a) 对 → ~450K 训练样本。

#### 关键加分项（多家共识）

- **方案 5 提到的 "4-5 箱子 OOD 集" 必须合成**（规则允许但 phase 数据缺失）
- **方案 5 提到的 "识别噪声集"**：模拟 YOLO 漏检 / 错检
- **方案 5 提到的 "炸弹边缘规则集"**：对角推送等罕见状态

### 5.2 损失函数

```
L = L_policy + λ_r·L_ranking + λ_v·L_value + λ_d·L_deadlock + λ_p·L_progress + λ_i·L_info
```

**Policy distillation (soft Q label)**（方案 1, 6）：

```
y_i = exp(-Q*(s, a_i) / τ) / Σ_j exp(-Q*(s, a_j) / τ)
L_policy = -Σ_i:legal y_i · log π_θ(a_i | s)
```

τ = 0.5（温度）。多个近似最优动作不会被强制对立。

**Pairwise ranking (hard negative)**（方案 1, 6）：

对每个 (s, a+) 采样 hard negative a−（推入死角 / 错 ID / 绕远）：

```
L_ranking = max(0, γ - score(s, a+) + score(s, a−))
```

γ = 1.0。这强迫专家动作的分数明显高于显式坏动作。

**Value loss (Huber + BCE)**：

```
L_value = Huber(V_θ(s), -C*(s), δ=4)
        + BCE(p_win(s), 1{episode wins})
        + BCE(p_dead_at_state(s), 1{state is dead-end})
```

**辅助头 BCE 与 Huber**：

```
L_deadlock = BCE(p_dead(s, a_i), 1{action a_i 推入 deadlock})
L_progress = Huber(p_progress(s, a_i), n_remaining_pushes_after_a_i, δ=2)
L_info     = Huber(p_ig(s, a_i), IG_classical(s, a_i), δ=2)
```

权重建议（方案 6 数值，经验起点）：

| | λ |
|---|---:|
| L_policy | 1.0 |
| L_ranking | 0.5 |
| L_value | 0.3 |
| L_deadlock | 0.2 |
| L_progress | 0.2 |
| L_info | 0.1 |

### 5.3 数据增强

#### A. 几何对称（方案 2, 3, 5, 6 共识）

由于 14×10 不是正方形，**只做** D₂（4 元素）：

- 单位变换
- 水平翻转 (沿 vertical axis)
- 垂直翻转 (沿 horizontal axis)
- 180° 旋转

**不做 90° 旋转**（会变成 10×14）。

每个 batch 随机抽 g ∈ D₂ 应用到样本。变换时需同步：
- 输入张量空间维度旋转 / 翻转
- 玩家朝向 cos/sin 通道相应变换
- 候选动作的 (x, y, dir) 同步变换
- 推送方向流场通道（24-27）相应循环 / 翻转

#### B. ID 重命名增强（方案 2, 3, 5, 6 共识）

每个 batch 随机置换 σ ∈ S_10：把所有 box class_id 与 target num_id 一致地 σ 映射。任务等价。

这避免学到"3 比 7 更特殊"的伪规律。

#### C. 部分可观测性增强（方案 1, 5, 6）

随机 mask 一部分已识别实体回到 unknown 状态：

```
p_mask ∈ [0.1, 0.4] 随机：
   把已识别 box / target 的 K[i] 设回 unknown
   重置 Π 矩阵相应行 / 列为均匀
```

模拟"识别延迟 / 漏识别"。

#### D. 反事实负例（方案 1, 5, 6）

对每个专家 (s, a*)，采样 K=4 个非专家合法动作：

- 推入死角的动作
- 把箱子推离目标的动作  
- 未识别 ID 时贸然推送的动作
- 错误爆破方向的动作

这些标为 hard negative，用于 ranking loss。

### 5.4 三阶段流水线

```
Stage A：监督预训练 (BC, 80 epoch, ~12 GPU·h)
   • 数据：IDA* + BestFirst trajectories with soft Q labels
   • 增强：D₂ + ID permutation + 偶尔 part-occlusion
   • 损失：L_policy + 0.3·L_value + 0.2·L_deadlock
   • 优化：AdamW, lr=3e-4 cosine, batch 256, weight decay 1e-4
   ↓
Stage B：DAgger 在线纠偏 (3-5 轮, ~6 GPU·h)
   • 当前模型在 200 张 verify-seed 图上 deterministic rollout
   • 收集失败前 5-20 步状态 + low-confidence 状态 (max π < 0.4) + deadlock 前状态
   • 用 BestFirst 给标签（极端难图夜间用 IDA*）
   • 加入 replay buffer，重训 3 epoch / 轮
   ↓
Stage C：QAT 量化感知微调 (10 epoch, ~2 GPU·h)
   • 插入 fake quant ops
   • 校准 representative dataset：覆盖各 phase 的 hard states
   • 学习率衰减到 3e-5
   • 验证 INT8 win_rate 与 fp32 差距 ≤ 2pp
```

### 5.5 课程策略（phase-stratified sampling）

不顺序训（会灾难性遗忘 phase 1-3），而是按比例混合：

| Phase | 采样权重 | 原因 |
|---|---:|---|
| 1 | 5% | 容易，少量保持记忆 |
| 2 | 10% | |
| 3 | 15% | |
| 4 | 25% | 多箱组合是核心 |
| 5 | 20% | 炸弹关键场景 |
| 6 | 25% | 最终目标 |

DAgger 阶段进一步把 hard-fail 状态权重 ×3。

### 5.6 多老师质量分派（方案 1, 6 共识）

不同来源的样本 loss 权重不同：

| 老师 | 样本 loss 权重 | 原因 |
|---|---:|---|
| IDA* | 1.0 | 严格最优 |
| BestFirst | 0.8 | 1.5×OPT 仍可信 |
| AutoPlayer | 0.4 | 仅用作"覆盖广度" |

DAgger 修正样本（teacher-on-failed-states）：1.5（更重要）。

---

## 6. 推理时浅层搜索

### 6.1 默认：贪心 argmax（保 5 Hz）

```python
score_logits = SAGE_PR.forward(state)  # 64 维
score_logits[~mask] = -inf
a_star = argmax(score_logits)
```

无 beam，时延 ~20 ms。

### 6.2 触发条件：beam search (B=3, D=3)

只在以下情况启用（约 30% 决策点）：

```python
top1, top2 = top2(score_logits)
if top1 - top2 < 0.5 \
   or any(p_dead_head[a_top1] > 0.3) \
   or value_head < 0:
    use_beam_search()
```

beam search 路径评分：

```
J(a_0:D) = -Σ_t [α·log π(a_t|s_t)
                + λ_v·V(s_t)
                + λ_d·1{deadlock}
                - λ_i·IG(a_t)]
         + λ_terminal·V(s_D)
         - λ_w·log p_win(s_D)
```

α=1.0, λ_v=0.3, λ_d=2.0, λ_i=0.2。

每步用符号模拟器算 `s_t+1 = T_symbolic(s_t, a_t)`，所以 beam 不需要 NN 预测下一个状态。NN 只评估：每个候选动作的 score（policy）+ 每个新状态的 V（value）。

3×3=27 个 NN 调用，每个 20 ms → ~540 ms（仅在触发时）。OpenART mini 主循环可短暂掉到 1.5-2 Hz，可接受。

---

## 7. 部署细节

### 7.1 主循环（每 200 ms 一帧 = 5 Hz）

```
t=0     抓帧 (5 ms)
t=5     YOLO 推理（仅在新 ID 进入 FOV 时） (60 ms)
t=65    传统 CV: AprilTag, 颜色 (25 ms)
t=90    Belief 更新 + 领域特征预计算 (5 ms)
t=95    [事件触发] 候选生成 + SAGE-PR 推理 (25 ms)
t=120   [可选] beam search (60 ms)
t=180   BFS 路径规划 + 电机指令 (5 ms)
t=185   缓冲（其他系统服务）(15 ms)
t=200   下一帧
```

**关键事件触发原则**：
- YOLO 仅当未识别实体进入 FOV 才跑
- SAGE-PR 仅当车到达"格中心" / "完成推送" / "新 ID 识别" / "炸弹爆炸" 时跑
- 中间帧只跑 BFS 低层执行

平均决策频率 ≈ 1.5 Hz，实际 NN 推理负担降到 25%。

### 7.2 部署路径优先级

#### Option 1 (首选)：YOLO + SAGE-PR 都用 TFLite Micro

- 两个独立 Interpreter，各自独立 arena
- YOLO arena ~150 KB + SAGE-PR arena ~150 KB = 300 KB ≪ 2 MB
- 在 OpenART Plus 上更稳定（dual-core，M4 跑控制）

#### Option 2 (mini 救场)：YOLO 用 TFLM，SAGE-PR 手写 CMSIS-NN

如果 OpenART mini 上两个 TFLite 模型 driver 冲突（参考 K230 教训）：
- SAGE-PR 结构非常简单（DSConv + FC + 1×1 Conv）
- 手写 INT8 推理 200 行 C，调 CMSIS-NN kernel
- 完全绕开 TFLite arena，跟 YOLO 零冲突

### 7.3 量化策略

```
1. 训练 fp32 收敛（Stage A）
2. DAgger 完后（Stage B）插 QAT fake quant
3. 校准集：~500 张 representative states，覆盖各 phase + hard cases
4. Per-channel symmetric INT8 for weights
5. Per-tensor asymmetric INT8 for activations
6. Policy logits & Value 输出保 INT16（敏感）
7. 验证：fp32 vs INT8 win_rate gap ≤ 2pp
   - 若超过，加深 QAT 5 epoch
   - 仍不行，部署用 top-3 beam（容错）
```

### 7.4 关键算子兼容性

仅使用 RFC §4.4 允许的 op：

| 用到的 op | 用途 |
|---|---|
| `Conv2D` | 3×3 主干 + 1×1 Score head |
| `DepthwiseConv2D` | DSConv block |
| `FullyConnected` | Fusion + Value head |
| `ReLU` | 激活 |
| `Add` | Residual + broadcast c to e_i |
| `Reshape` | candidate set tensor 整形 |
| `Concat` | z_grid + z_set + u_global |
| `AvgPool2D` | Global pool |
| `Softmax` | 输出 π（可选；部署可只 argmax） |

**无 BatchNorm / GroupNorm / LayerNorm / LSTM / GRU / Attention**（Fixup init fold 进 weights）。

---

## 8. 实施路线图

借用方案 5 的格式但缩成 5 周（不到方案 5 的 8 周，因为 SAGE-PR 比候选 ranker 简单，且很多基建已存在）：

| 周 | 任务 |
|---|---|
| W1 (5d) | belief state + 领域特征预计算 + 候选生成器（Python 实现 + 单测） |
| W2 (5d) | SAGE-PR PyTorch 实现 + Fixup init + 单元前向测试；BC 数据生成 |
| W3 (7d) | Stage A BC 训练（80 epoch）+ 第一次 PC 评测；Stage B DAgger 3 轮 |
| W4 (5d) | Stage C QAT；TFLite Micro 导出；OpenART mini 实测时延 |
| W5 (5d) | 集成 YOLO + CV，HIL 测试，Bug 修复，比赛准备 |

如时间允许，预留 1 周给 Plan B（手写 CMSIS-NN fallback）。

---

## 9. 风险点 & 备选

整合各家提出的风险并归并去重。

### 风险 1：phase 6 炸弹时序学不会

**症状**：phase 6 win_rate 卡 80-85%，phase 1-5 正常。

**根因**：炸弹动作"立即收益为 0、长期收益高"，BC 难学。

**主缓解**：
- 训练时上采样 phase 6 含炸弹 trajectory ×3
- 候选特征中加 `blast_gain` = 爆破后连通分量增益
- 触发 beam search 时 D=4

**Plan B**：单独训一个 BombValueHead（小型 MLP，输入炸弹位置 + 内墙拓扑，输出爆破收益）。

### 风险 2：BC distribution shift on phase 4-6

**症状**：phase 4-6 鲁棒性差。

**主缓解**：DAgger 5 轮（默认）。

**Plan B**：DAgger 后加一轮 IQL/AWR 风格的离线 RL fine-tune（方案 5 推荐）。

### 风险 3：模型在 OpenART mini 上推理超时

**症状**：实测 SAGE-PR 推理 > 50 ms。

**主缓解**：
- 通道宽度 32 → 24，参数减半
- 减一个 ResBlock
- AvgPool 之前加一次 stride=2 下采样到 5×7

**Plan B**：手写 CMSIS-NN（绕开 TFLite Micro overhead）。

### 风险 4：YOLO + SAGE-PR arena 冲突（K230 教训）

**主缓解**：用两个独立 Interpreter + 两块 arena（Option 1）。

**Plan B**：SAGE-PR 改手写 CMSIS-NN（Option 2）。

### 风险 5：未识别 ID 末期探索过激进 / 保守

**症状**：phase 6 反复横跳扫描或贸然推错。

**主缓解**：
- `info_gain_heatmap` 输入通道精确编码"还有多少 ID 不确定"
- 候选特征中 `match_entropy` 引导：低熵时 inspect 优先级降低
- belief state 在 N-1 已识别时立即排除推理填入

**Plan B**：硬规则 fallback：若 `max_j Π(i,j) < 0.8` 则禁止把 box_i 推入"高代价不可逆"区域。

### 风险 6：对角推送学不会（数据稀少）

**主缓解**：候选生成器枚举对角推送时给 `+5 bonus` logit；网络只学"什么时候不该这么做"。

### 风险 7：量化损失 > 2pp

**主缓解**：QAT 加深到 15 epoch + ranking margin loss。

**Plan B**：部署用 top-3 beam search 抵消（容错性强）。

---

## 10. 跟当前 baseline 的对比

| 维度 | 当前 baseline | SAGE-PR | 估计提升 |
|---|---|---|---|
| 状态表示 | 254-D 扁平向量 + 部分 2D | 14×10×30 全空间 + belief state | +10-20pp on 多箱 |
| 动作输出 | 54 类 softmax 扁平 | 64 候选集合 ranking | +5-10pp |
| BFS 距离场使用 | 仅作 path planning 后处理 | **作为输入通道** | +5-10pp（信息瓶颈论证） |
| 死角处理 | 无显式机制 | 输入通道 + 辅助头 | -10-15pp 死锁失败率 |
| ID 排除推理 | 模型自学 | 显式预处理 | +1-2pp |
| 部分可观测性 | 无显式 | belief matrix Π | +3-5pp |
| 训练范式 | BC | BC + DAgger + QAT + soft Q + ranking | +3-8pp |
| 量化损失 | 不详 | QAT + ranking margin | ≤ 2pp |
| 推理 | 浅层 branch search | 事件触发 beam search + value | +1-3pp |
| D₂ 对称利用 | 无 | 数据增强 | +1-3pp |

**整体估计**：phase 4-6 从 baseline 的 50-80% → SAGE-PR 的 88-93%。

---

## 11. 核心一句话

> **不要训练一个"小脑袋"去凭感觉输出动作类别；要训练一个"神经启发式函数"去给符号生成的合法候选动作排序。**
> 
> （此句来自方案 6，是 6 份提案最精炼的总结。）

让 BFS、合法性、死角、ID 排除、FOV 信息增益这些**确定逻辑**留给经典算法；让神经网络专注于经典算法做不好的事：**多任务调度的全局优先级、长程后果的价值预测、不确定下的探索时机**。

这才是这道题的归纳偏置。

---

## 附录 A：训练超参表

| 项 | 值 |
|---|---|
| Optimizer | AdamW |
| 学习率 (Stage A) | 3e-4 → 3e-5 cosine |
| 学习率 (Stage B / DAgger) | 1e-4 → 1e-5 cosine |
| 学习率 (Stage C / QAT) | 5e-5 fixed |
| Batch size | 256 |
| Weight decay | 1e-4 |
| Gradient clip | 1.0 (L2) |
| Stage A epochs | 80 |
| DAgger 轮数 | 3-5 |
| QAT epochs | 10 |
| 数据增强：D₂ | 100% 触发 |
| 数据增强：ID 重命名 | 100% 触发 |
| 数据增强：occlusion | p ∈ [0.1, 0.4] |
| 数据增强：counterfactual neg | K=4 per sample |
| 总训练时长（RTX 5060 Ti 16GB） | ~24 GPU·h |

## 附录 B：参数与 MAC 详细推导

| 模块 | 算子 | 参数（fp32） | INT8 | MAC |
|---|---|---:|---:|---:|
| stem Conv 3×3 30→32 | Conv2D | 8704 | 9KB | 1.2M |
| ResBlock 1 (DSConv 32→32) | DW + PW | 1376 | 1.4KB | 192K |
| ResBlock 2 (DSConv 32→32) | DW + PW | 1376 | 1.4KB | 192K |
| ResBlock 3 (DSConv 32→48 dilate=2) | DW + PW | 1856 | 1.9KB | 260K |
| ResBlock 4 (DSConv 48→48) | DW + PW | 2784 | 2.8KB | 390K |
| ResBlock 5 (DSConv 48→48) | DW + PW | 2784 | 2.8KB | 390K |
| GAP + FC 48→96 | FC | 4704 | 4.8KB | 5K |
| Cand encoder Conv 1×1 128→96 | Conv2D | 12384 | 12.4KB | 786K |
| Cand encoder Conv 1×1 96→96 | Conv2D | 9312 | 9.4KB | 590K |
| Cand pool | AvgPool | 0 | 0 | 0 |
| Fusion FC 208→128 | FC | 26880 | 27KB | 27K |
| Fusion FC 128→96 | FC | 12384 | 12.4KB | 12K |
| Score head Conv 1×1 96→96 | Conv2D | 9312 | 9.4KB | 590K |
| Score head Conv 1×1 96→1 | Conv2D | 97 | 0.1KB | 6K |
| Aux deadlock head | Conv 96→1 | 97 | 0.1KB | 6K |
| Aux progress head | Conv 96→1 | 97 | 0.1KB | 6K |
| Aux info gain head | Conv 96→1 | 97 | 0.1KB | 6K |
| Value head FC 96→32→1 | FC | 3137 | 3.2KB | 3K |
| **总计** | | **~98 K** | **~98 KB** | **~5.5 M** |

加 INT8 校准 metadata、bias、padding：~120-150 KB。

## 附录 C：完整 PyTorch skeleton

（实际实现时按下表层结构 1:1 写出 SAGE_PR forward 即可，约 200 行；省略以保持文档简洁。具体实现在 W2 完成。）

---

## 附录 D：致谢

本方案整合自 6 位匿名评审者（专家分析/1-6.md）的独立提案。对方案 2 的数学论证、方案 5 的工程化路线、方案 6 的神经-符号哲学和候选集合等变设计，给予了最大权重。

特别的，方案 6 提出的核心洞察 — "**让神经网络给规划排序，而不是替代规划**" — 是本设计的灵魂。

*整合人：ralph-loop · 2026-05*
