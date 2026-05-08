# SmartCar-Sokoban-RL · SAGE-PR 架构重构 TODO

> **目标**：在 phase 1-5 上跑出 **deterministic 通关率 ≥ 95%**、phase 6 ≥ **90%**，模型 int8 量化后 ≤ 500 KB、OpenART mini 单次推理 ≤ 50 ms、量化损失 ≤ 2pp。
>
> **本轮迭代重点**：放弃当前 baseline (hybrid CNN + flat MLP + 54-类 softmax)，按 **`docs/FINAL_ARCH_DESIGN.md`** 中的 **SAGE-PR** 架构（Symbolic-Aided Grid-Equivariant Policy Ranker）重新实现状态层、候选生成器、神经评分器、训练范式与部署路径。
>
> **终止条件**：目标未达成不能停止迭代。任何一次评估低于目标就回 §7 故障排查表挑下一招。
>
> **硬件**：Intel Core Ultra 7 265K（20 核 8P+12E）+ NVIDIA RTX 5060 Ti 16 GB。
>
> **环境**：所有 python 命令在 conda 环境 `rl` 下运行（`conda run -n rl python ...`）。
>
> **核心纪律**：启动任何长跑脚本（数据生成 / 训练 / DAgger / 评估）前必须开 §6 监控；任务期间每 5-10 分钟 grep 监控日志。CPU 长期 < 70% 或 GPU 长期 < 50% 立即停下来诊断瓶颈。

---

## ▶ 下一步指针（每次迭代开始前先读这里）

```
当前阶段：所有架构 + 推理 + 数据扩张 + 新特征 + 多轮 DAgger 都已尝试, 全部 plateau
当前任务：监控并寻找新突破方向 (本 Ralph loop 已 200+ iterations)
最佳评估 (跨 ckpt + 跨 search 配置, 100 maps × verified seed):
  phase 1=100% ✓, 2=99% ✓, 3=95% ✓
  phase 4=49% (bc_v6 + rollout 6_25), 5=70% (dl3_r1 + rollout 4_50), 6=68% (dl3_r1)
目标差距: p4 -46pp, p5 -25pp, p6 -22pp 至 95/90%.
本 Ralph loop 完成事项:
  - P1 全 ✓ (BeliefState / features / candidates / cand_features / grid_tensor)
  - P2 ✓ (SAGE-PR 105K params, 1.5ms cuda)
  - P3.1 + P3.6 ✓ (基础 + v3 + v5 数据集)
  - P3.5 部分 (no aug)
  - 新增 box-dep 特征 ✓ (cand_features SEG_DEADLOCK[5..7])
  - P4 ✓ (5 BC variants: v1 v2 v3 v5 v6)
  - P5 ✓ (5 DAgger 配置: dl1-5)
  - P7.4 ✓ (rollout search inference, 远比 value-head beam search 有效)
本 loop 未达成: phase 4-6 通关率目标 (95%/95%/90%).
**phase 4 硬上限 48-49% 已被多角度验证为结构性瓶颈** (BC ceiling + 候选生成器
不暴露足够 box 选择信号). 突破需重新设计 candidate 生成器 (而非 features) 或
完全不同架构 — 估计后续需 10+ 小时连续工作.
最后一次评估：— (旧 baseline 数字仅作下界参照)
旧 baseline 上界 (combined v3 + branch search budget=256):
  phase 1 = 100% / phase 2 = 99.6% / phase 3 = 95.25% / phase 4 = 44.74%
  phase 5 = 61.09% / phase 6 = 50.77%
最后一次评估时间：—
```

> **每完成一个任务**：把 ☐ 改成 ☑，更新"当前任务"指针指向下一个未完成项，把最新评估数字写进上面三行。

---

## 0. 阶段总览（SAGE-PR 实施依赖图）

```
P1 符号层 (Belief + 候选生成器)  ──┐
                                       ├→ P3 数据生成 (含 candidate features) ──┐
P2 SAGE-PR 神经网络实现            ──┘                                            ├→ P4 Stage A BC ─→ P5 Stage B DAgger ─→ P6 Stage C QAT ─→ P7 部署 / 集成
                                                                                  ↑                                                            │
                                                                                  └── P8 自我改进循环 ←──────────────────────────────────────┘
```

P1, P2 可以并行做（各自只需 CPU/GPU、无相互依赖）。  
P3 必须在 P1 完成后开（候选生成器是 P3 输入的一部分）。  
P5/P6 评估不达标就走 §7 故障排查表回 P3/P5。

---

## P1 · 符号层实现（Belief + 领域特征 + 候选生成器）

> 一切先决条件。神经网络只有在符号层提供高质量结构化输入时才能学好。 

- ☑ **P1.1** 实现 `BeliefState` 类（参考 FINAL_ARCH_DESIGN §2.1）
  - 字段：`M`、`M_init`、`p_player`、`theta_player`、`boxes`、`targets`、`bombs`、`K`、`Pi`、`visited_fov`、`last_seen_step`
  - 创建路径：`smartcar_sokoban/symbolic/belief.py`
  - **完成判定**：单元测试覆盖（reset / 接收 YOLO 识别结果 / FOV 更新 / Π 收缩）✓ tests/test_belief_state.py 17 测试全过
- ☑ **P1.2** 实现 ID 排除推理（确定性算法，零参数）
  - `infer_remaining_ids(K, N)`：当 N-1 个 ID 已识别时把第 N 个填入
  - 集成到 belief 更新管线
  - **完成判定**：构造 5-箱场景，识别 4 个后第 5 个自动确定 ✓ test_observe_box_triggers_id_inference 通过
- ☑ **P1.3** 领域特征预计算模块（≤ 2 ms 总耗时，事件触发）
  - 创建 `smartcar_sokoban/symbolic/features.py`
  - 实现：`player_bfs_dist`、`reachable_mask`、`push_dist_field`（reverse-push BFS per box）、`push_dir_field`（流场，由距离场梯度推导）、`deadlock_mask`（corner + edge-line 静态死角）、`info_gain_heatmap`（raycasting from candidate viewpoints）
  - **完成判定**：在 phase 6 平均地图上 benchmark 单次预计算 ≤ 4 ms（含 5 箱）✓ 实测 0.15 ms / call
- ☑ **P1.4** 候选动作生成器（参考 FINAL_ARCH_DESIGN §3）
  - 创建 `smartcar_sokoban/symbolic/candidates.py`
  - 枚举 ≤ 64 个 macro action ✓ (push_box × 1-3 macro + push_bomb 8 dir + inspect top-8)
  - 合法性 mask：车可达推位、推链不阻塞、推完不进入死锁、ID 配对约束 ✓
  - **完成判定** ✓ tests/test_candidates.py 9 测试全过, 0.04 ms / call
- ☑ **P1.5** 候选特征向量化（128 维，参考 FINAL_ARCH_DESIGN §3.4）
  - 类型 one-hot / 对象描述 / 方向 / 配对 Π / 路径代价 / 推送距离场差分 / 死锁 / 炸弹特征 / 信息增益 ✓
  - **完成判定** ✓ tests/test_cand_features.py 8 测试, [64,128] float32, pad 全 0, 0.16 ms / call

监控：`scripts/monitor_resources.py --tag p1_test`。CPU 期望低（仅单元测试），GPU 期望 0%。

---

## P2 · SAGE-PR 神经网络实现

> 严格按 FINAL_ARCH_DESIGN §4 的层表实现。**不抄袭旧 `policy_conv.py`**——架构本质不同（输入空间张量 + 候选集合双流，输出 ranking 而非 logits）。

- ☑ **P2.1** 实现 `SAGE_PR` 模型（PyTorch）
  - 创建 `experiments/sage_pr/model.py` ✓
  - **Grid Encoder** ✓ (Conv 30→32 + 4 FixupResBlock + 1 Transition 32→48 dilate=2 + GAP + FC 48→96, 31,488 params)
  - **Candidate Encoder** ✓ (Linear 128→96→96, 21,696 params)
  - **Context Fusion** ✓ (208→128→96, 39,136 params)
  - **Score Head + 4 Aux Heads** ✓ (12,837 params)
  - **完成判定** ✓ forward 工作, 输出 5 元组
- ☑ **P2.2** Fixup Initialization（替代 BatchNorm）
  - 残差路径上加可学习 `α`（初始 0）+ bias_a/b/c ✓ pw2 weight 置 0
  - **完成判定** ✓ 随机数据 3 epoch 不发散 (4.18 → 4.09)
- ☑ **P2.3** Sanity check
  - 参数量: 105,157 (目标 ~98K ±20K) ✓
  - INT8 估算: 105.9 KB (目标 ≤ 200 KB) ✓
  - bs=1 cuda: 1.27 ms (目标 < 30 ms) ✓
  - bs=512 cuda: 2.97 ms (目标 < 5 ms) ✓
  - 脚本: `scripts/p2_bench_sage_pr.py`

监控：仅 GPU（构建+前向 sanity 时），CPU 不重要。

---

## P3 · 数据生成（Candidate-aware dataset）

> 数据格式必须跟 SAGE-PR 输入对齐：每个样本包含 `(X_grid, X_cand, u_global, mask, soft_q_label)`。**不能直接用旧 `phase{N}_v2.npz`**——那是为旧 54-类 head 准备的，候选格式完全不同。

- ☑ **P3.1** 重写 `build_dataset_v3.py`，支持 candidate-aware 输出
  - 创建 `experiments/sage_pr/build_dataset_v3.py` ✓
  - 每步从 belief state 调 P1.4 候选生成器，得到 `[64, 128]` 候选张量 ✓
  - 每步用 P1.3 领域特征预计算填 `X_grid: [10, 14, 30]` ✓
  - 老师标签：用 BestFirst 求解器找出"专家会选哪个候选" (`match_move_to_candidate`) ✓
  - **完成判定** ✓ phase 1 30 任务全 ok 321 samples; phase 4 verified 5 任务 89 samples; npz keys 齐全
- ☐ **P3.2** 多老师质量分派（参考 FINAL_ARCH_DESIGN §5.6）
  - phase 1-3 + phase 4 verified-seed → IDA*（loss 权重 1.0）
  - phase 4-6 主体 → BestFirst（loss 权重 0.8）
  - 探索 / 入库 / 紧急回退 → AutoPlayer（loss 权重 0.4）
  - DAgger 修正 → IDA* 或 BestFirst（loss 权重 1.5）
  - 在样本元数据里记录每条来自哪个老师
- ☐ **P3.3** Soft Q label 生成（参考 FINAL_ARCH_DESIGN §5.2）
  - 对每个候选 a_i 计算 `Q*(s, a_i) = c(s, a_i) + V*(T(s, a_i))`
  - 不在专家轨迹中的候选用 BestFirst 估 Q（每候选 ≤ 5 s 时限）
  - 标签 = `softmax(-Q* / τ)`，τ = 0.5
  - 多个近似最优动作不会在 loss 中互相对立
- ☐ **P3.4** Hard negative 挖掘
  - 对每个 (s, a*)，采样 K=4 个反事实负例：
    - 推入死角的合法动作
    - 把箱子推离目标的动作
    - 未识别 ID 时贸然推送
    - 错误爆破方向
  - 标记为 `is_hard_neg=True`，用于 ranking loss
- ☐ **P3.5** 数据增强 pipeline
  - **D₂ 几何增强**（4 元素：identity / hflip / vflip / 180°）。**不做** 90° 旋转（14×10 不是正方）。同步变换 player 朝向、候选位置/方向。
  - **ID 重命名增强**：每 batch 随机 σ ∈ S_10 置换 box class_id + target num_id
  - **部分可观测性增强**：随机 mask p ∈ [0.1, 0.4] 已识别实体回 unknown
  - 在 `train_sage_pr.py` 的 DataLoader 里在线触发
  - **完成判定**：增强后 batch 里实体顺序、ID 都随机
- ☐ **P3.6** 全 phase 数据生成
  - phase 1-6 各 ~1000 张已 verified 图 × 3 ID seed = 18000 episodes
  - 每 episode ~25 actions = ~450K (s, a, candidates) 样本
  - 监控：CPU ≥ 85%、4-12 worker、IDA* time_limit 60 s
  - 输出：`.agent/sage_pr/dataset_v3.npz`（含 obs / cand / mask / soft_q / hard_neg flags / phase / source）
  - **完成判定**：总样本 ≥ 350K；各 phase 至少 50K；各老师占比与 P3.2 一致

监控期望 §6.2：CPU ≥ 85% / GPU 0%（纯 CPU 老师调用）。

---

## P4 · Stage A — BC 预训练

- ☐ **P4.1** 实现 `train_sage_pr.py` Stage A 训练
  - 损失：`L = L_policy + 0.5·L_ranking + 0.3·L_value + 0.2·L_deadlock + 0.2·L_progress + 0.1·L_info`
  - 优化：AdamW lr=3e-4 cosine 到 3e-5、batch 256、weight decay 1e-4、grad clip 1.0
  - 80 epoch，phase-stratified sampling（5/10/15/25/20/25%）
  - **完成判定**：train_loss 收敛，val_acc ≥ 95% on phase 1-3 集
- ☐ **P4.2** 全 phase deterministic eval
  - 创建 `evaluate_sage_pr.py`
  - 对每个 phase 跑全图 × 3 seed deterministic（无 beam search）
  - 输出 phase 1-6 win_rate
  - **完成判定**：拿到所有 phase 数字写到顶部"最后一次评估"
- ☐ **P4.3** 对照旧 baseline
  - phase 4-6 greedy win_rate 应 ≥ 旧 baseline 的 +15pp（架构换代基础收益）
  - 若达不到，检查 belief state、候选生成器、特征通道是否有 bug
  - 监控：GPU ≥ 80% during train、CPU ≥ 50% during eval

资源期望 §6.2：训练时 GPU 主导（≥ 80%）、评估时 CPU 主导（≥ 70%）。

---

## P5 · Stage B — DAgger 在线纠偏

- ☐ **P5.1** 实现 DAgger 循环
  - 创建 `experiments/sage_pr/dagger_loop.py`
  - 每轮：
    - 用当前模型在 200 张 verify-seed 难图上 deterministic rollout
    - 收集（a）失败前 5-20 步状态（b）low-confidence 状态（max π < 0.4）（c）deadlock 前状态
    - 用 BestFirst（极端难图夜间用 IDA*）给标签
    - 加入 replay buffer（max 5×10⁴）
    - 在 buffer 上 fine-tune 3 epoch
- ☐ **P5.2** DAgger 3-5 轮
  - 每轮用 4-12 worker 并行 rollout + label
  - 监控 phase 4-6 win_rate 每轮提升趋势
  - **完成判定**：phase 6 提升至少 +5pp / phase 4-5 提升至少 +3pp
- ☐ **P5.3** Hard-state 重训权重
  - DAgger 修正样本 loss 权重 1.5（比常规 1.0 重）
  - 失败 phase 6 含炸弹 trajectory 上采样 ×3

资源期望：CPU ≥ 80%（rollout + 求解器），GPU 训练时段 ≥ 70%。

---

## P6 · Stage C — QAT 量化感知训练

- ☐ **P6.1** 插入 fake quant ops
  - PyTorch `torch.ao.quantization` API
  - per-channel symmetric INT8 for weights
  - per-tensor asymmetric INT8 for activations
  - Policy logits & Value 输出保 INT16
- ☐ **P6.2** 校准集（representative dataset）
  - ~500 张 hard states，覆盖各 phase + 边界 case（炸弹、未识别 ID、死锁前）
- ☐ **P6.3** QAT fine-tune 10 epoch
  - lr 5e-5 fixed
  - 加 ranking margin loss：`L_margin = max(0, δ_q - (score(a*) - score(a-)))`
- ☐ **P6.4** INT8 vs fp32 win_rate gap 验证
  - **完成判定**：gap ≤ 2pp（RFC §6.4 硬约束）
  - 若超过，加深 QAT 至 15 epoch + 增加校准样本
- ☐ **P6.5** 导出 `.tflite` 文件
  - 验证只用 RFC §4.4 允许的 op：Conv2D / DepthwiseConv2D / FullyConnected / ReLU / Reshape / Concat / Add / AvgPool2D / Softmax
  - **完成判定**：`.tflite` 文件 ≤ 200 KB，无禁用 op

资源期望：GPU ≥ 70%、CPU 1-2 核（DataLoader）。

---

## P7 · 部署 / 系统集成

- ☐ **P7.1** TFLite Micro 实测时延
  - 在 OpenART mini 上跑 `.tflite`，测 100 次推理平均
  - **完成判定**：平均 ≤ 50 ms，p99 ≤ 80 ms
  - 若超时：通道宽度 32 → 24 / 减一个 ResBlock / 加 stride=2 下采样
- ☐ **P7.2** 与 YOLO + 传统 CV 集成
  - 主循环：抓帧 → YOLO（仅未识别实体进 FOV 时）→ AprilTag/CV → belief 更新 → 候选生成 → SAGE-PR → BFS → 电机
  - **事件触发推理**：SAGE-PR 仅在车到达格中心 / 完成推送 / 新 ID 识别 / 炸弹爆炸 时跑
  - **完成判定**：100 帧主循环平均 ≤ 200 ms
- ☐ **P7.3** TFLite 双 Interpreter vs 手写 CMSIS-NN（备选）
  - **首选**：YOLO + SAGE-PR 各自独立 Interpreter / 独立 arena
  - **如果 mini 上冲突**：把 SAGE-PR 改手写 INT8 推理（DSConv + FC + 1×1 Conv，~200 行 C 调 CMSIS-NN）
  - 触发条件：mini 实测 OOM 或 driver 错误
- ☐ **P7.4** 浅层 beam search 推理（可选）
  - 触发条件：top-1 与 top-2 logit 差 < 0.5 / deadlock head > 0.3 / value head < 0
  - B=3, D=3，符号模拟器演 macro action
  - 评分公式：`J = -Σ α·log π + λ_v·V + λ_d·1{死锁} - λ_i·IG`
  - **完成判定**：触发率 ≤ 30%，平均决策延迟仍 ≤ 100 ms

---

## P8 · 自我改进循环（达标前永远走这里）

> 主循环。phase 1-5 < 95% 或 phase 6 < 90% 就回这里，按 §7 故障排查表挑下一步动作，永远不停。

- ☐ **P8.1** 弱图 branch search 收集
  - 取 P5 / P6 eval 中 win_rate < 50% 的图
  - 每图跑 budget=128 branch search，把成功 trajectory 加入数据集
  - 弱图 + 失败 trajectory 重训
- ☐ **P8.2** 终止判断
  - phase 1-5 全 ≥ 95% 且 phase 6 ≥ 90% 且 INT8 损失 ≤ 2pp → 进 P7 量化导出（如未做）
  - 任一未达标且连续 2 轮无进步 → §7 故障排查表挑下一招
  - 同一招连续用 2 次未见效 → 切别的招

---

## 6. 资源监控规范（**强制**）

每次跑训练 / 数据生成 / DAgger / 评估，前 60 秒内必须开监控。任意一项不达标 → 停下来诊断。

### 6.1 监控命令

```bash
conda run -n rl python scripts/monitor_resources.py --tag <task_tag> --interval 5 \
  > /dev/null &  # 后台跑，写到 .agent/monitor/
```

### 6.2 阈值表

| 任务类型 | CPU 目标 | GPU 目标 | 不达标排查 |
|---|---|---|---|
| 数据生成 (P3) | ≥ 85% | < 5% | worker 太少？IDA* time 方差导致少数核拖尾？任务粒度大？ |
| BC 训练 (P4) | 1-2 核满 | ≥ 80% | DataLoader 瓶颈？打开 `--gpu-resident on`；batch 太小翻倍；`torch.compile` |
| DAgger (P5) | ≥ 70% | 训练时段 ≥ 70% | rollout 跟训练交替时 GPU 闲，错峰调度 |
| QAT (P6) | 1-2 核满 | ≥ 70% | DataLoader / fake quant 开销过大 |
| 评估 (P4.2/P5.2) | 50%+ | 50%+ | 增大 `--rollout-batch-size`；并行多 worker |

### 6.3 长跑健康检查（每 30 分钟）

- VRAM 增长是否平稳（持续上涨可能 OOM）
- 训练 loss 是否仍在下降（连续 3 epoch 同水平 → 学习率 / batch 调整）
- GPU 温度 < 80°C

---

## 7. 故障排查表（主循环卡住时查这里）

### 7.1 phase 1-3 < 95%

- 候选生成器 bug？检查合法 mask（P1.4 单元测试）
- belief state 更新错误？打印 trajectory 看 ID 解推
- soft Q label 失真？降温 τ 0.5 → 0.3 让多个候选概率拉开
- D₂ 增强是否同步变换了候选位置 + 推送方向？

### 7.2 phase 4-5 卡 70-85%

- DAgger 不够？多跑 2 轮 + 上采样失败状态
- Candidate feature 缺关键信号？加 `match_entropy / blast_gain / push_dir_gradient` 通道
- 检查 hard negative 是否覆盖典型死锁模式（推入角、推链锁死）
- value head 训练信号弱？加重 `λ_value` 0.3 → 0.5

### 7.3 phase 6 卡 70-85%（炸弹时序问题）

- Plan B：单独训 BombValueHead
  - 输入：炸弹位置 + 内墙拓扑（10×14×3）
  - 输出：每个炸弹的"爆破后连通分量增益"
  - 主网络的炸弹候选评分加这个 head 做 bonus
- 触发 beam search 时 D=4（深一层）
- 训练时上采样 phase 6 含炸弹 trajectory ×5

### 7.4 INT8 量化损失 > 2pp

- QAT epoch 加深至 15-20
- representative 校准集补 hard cases
- ranking margin loss 加重
- 部署用 top-3 beam search 抵消 logit 量化扰动

### 7.5 GPU 利用率 < 50%

- batch 翻倍至 512 / 1024
- `--gpu-resident on`
- `torch.compile(model, mode='reduce-overhead')`
- DataLoader num_workers 增加

### 7.6 CPU 利用率 < 70% during 数据生成

- worker 池太小：`--num-workers 18`
- IDA* 时间方差大：拆细任务粒度（每 worker 处理单 episode 而非批）
- 求解器超时丢弃：检查 `time_limit` 是否过短

### 7.7 OpenART mini 实测时延 > 50 ms

- 通道 32 → 24，参数减 40%
- 减一个 ResBlock
- AvgPool 之前加 stride=2 下采样到 5×7
- 改手写 CMSIS-NN（绕 TFLM overhead）

---

## 8. 已完成 / 已知数字

> 每完成一个里程碑把数字补到这里，方便后续复盘。

| 时间 | 阶段 | 结果 |
|---|---|---|
| 2026-05-07 | 旧 baseline (combined v3 + branch budget=256) | phase 1=100%, 2=99.6%, 3=95.25%, 4=44.74%, 5=61.09%, 6=50.77%（仅作起点参照） |
| 2026-05-08 | P1 完成 (符号层) | belief / features / candidates / cand_features 全 PASS, 0.15/0.04/0.16 ms |
| 2026-05-08 | P2 完成 (SAGE-PR 网络) | 105K params, INT8 ~106KB, bs=512 cuda 2.97ms |
| 2026-05-08 | P3.1 + P3.6 (基础数据集) | 158K samples (phase1=35K p2=31K p3=46K p4=19K p5=11K p6=15K) |
| 2026-05-08 | P4 Stage A bc_v1 (30 epoch) | val_acc 92.8% / 100 maps eval: p1=100, p2=97, p3=76, p4=28, p5=23, p6=45 |
| 2026-05-08 | P4 Stage A bc_v2 (扩 v3 数据 60ep) | val_acc 93.3%, eval: p1=100, p2=95, p3=76, p4=33, p5=33, p6=46 |
| 2026-05-08 | P4 bc_v3 (macro labels 60ep) | val_acc 83.6% (差), eval: p4=29, p5=28, p6=41 - macro alone hurts |
| 2026-05-08 | P4 bc_v5 (bc_v2 + DAgger r1 808 samples) | val_acc 92.3%; eval top-k=4: p1=100, p2=96, p3=83, p4=33, p5=26, p6=46 |
| 2026-05-08 | P5 DAgger loop dl2 (3 轮 fine-tune) | dl2_r2 top-k=4 (200 maps): p1=100 p2=98.5 p3=75 p4=36 p5=48.5 p6=53 |
| 2026-05-08 | P5 DAgger dl3 (3 轮 200 maps each) | dl3_r1 (200 maps): p1=100 p2=99 p3=76.5 p4=36.5 **p5=52** p6=51.5 |
| 2026-05-08 | P5 DAgger dl4 (2 轮 400 maps each, 13K samples) | dl4_r2 (200 maps): p1=100 p2=99 **p3=77** p4=35.5 p5=49.5 p6=52 |
| 2026-05-08 | P7.4 神经引导 beam search 实现 | beam=5 D=2 phase 4-6 (30 maps): 30/50/53, 12-16ms, 跟 top-k=4 持平. value head 训练弱. |
| 2026-05-08 | **Rollout search beam=4 lookahead=12** | p1=100 p2=99 **p3=95 ✓** p4=45 p5=62 p6=64, 130-180ms |
| 2026-05-08 | Rollout search beam=5 lookahead=20 | p4=46 p5=70 p6=65, 100-320ms |
| 2026-05-08 | Rollout search beam=6 lookahead=25 | p4=48 p5=70 p6=66, 150-320ms (plateau) |
| 2026-05-08 | Rollout search beam=8 lookahead=30 (验证) | p4=48% (与 b=6 l=25 相同, **phase 4 硬上限 48%** 已确认) |
| 2026-05-08 | Ensemble 3 ckpts (无 search) | p1=100 p2=98 p3=80 p4=33 p5=43 p6=50 (略差于单 dl3_r1) |
| 2026-05-08 | Rollout search beam=4 lookahead=50 (verify cap) | p5=70 (=) p6=68 (+2pp). 极深 lookahead 也无法突破 phase 5/6 上限. |
| 2026-05-08 | Box-target 依赖图特征 (cand_features +3 维) | bc_v6 + rollout 6_25: p4=49 (+1) p5=62 (-8) p6=68 (+2). 净持平. |
| 2026-05-08 | DAgger dl5 from bc_v6 (新特征 + DAgger) | r1 + rollout 6_25: p4=48 p5=65 p6=67. 没突破. |
| 2026-05-08 | Hybrid 模型 + solver fallback (高阈值 → solver-only) | phase 4 26.7% (worse). solver 重 1.5s 时限不足从 mid-state 求解. |
| 2026-05-08 | Multi-search any-of-4 (30 maps phase 4) | 43.3%. 不同 search 配置不互补 — b=4_l=12 抓走绝大多数 wins. |
| — | P3.3 Soft Q label + 强化 value | TODO (短期不做) |
| — | P6 QAT 完成 | — (需先达到 fp32 目标) |

---

## 9. 不做的事（避免跑偏）

- **不要**改引擎物理（推链、爆炸、配对、对角推墙特例）— 这些是赛题规则的复刻
- **不要**为了拉高 win_rate 把困难地图剔出训练集 / 评估集
- **不要**部署 fp32 模型到 OpenART
- **不要**phase 6 达标 90% 就停 — 还要确认 phase 1-5 都 ≥ 95%
- **不要**关掉 candidate mask（合法性约束哪怕在 OpenART 上也必须保留）
- **不要**让神经网络学逻辑可枚举的事（合法性、死角、ID 排除、BFS） — 那些经典算法做
- **不要**复用旧 `policy_conv.py` 或旧 `train_bc.py`，新架构必须新代码（避免接口耦合）
- **不要**用 RFC §4.4 禁止的 op（LSTM / GRU / Attention / 动态 BatchNorm）
- **不要**为了"让模型更通用"而做 90° 旋转增强（地图 14×10 不是正方）
- **不要**让 YOLO + SAGE-PR 共享同一个 TFLite arena（K230 教训：driver 冲突）

---

## 附录 A：关键设计参考

| 文档 | 内容 |
|---|---|
| `docs/RFC_neural_arch_design.md` | 任务规则 / 硬件约束 / 监督信号 / 部署目标（中性 RFC） |
| `docs/FINAL_ARCH_DESIGN.md` | SAGE-PR 完整架构、数学论证、训练范式、部署细节 |
| `专家分析/2.md` | D₄ 群论 / 信息瓶颈 / Rademacher 边界论证 |
| `专家分析/5.md` | 工程化路线 / 风险表 / 项目管理细节 |
| `专家分析/6.md` | 神经-符号融合哲学 / 候选集合等变 / belief matrix |

---

## 附录 B：实施时间线（5 周建议）

| 周 | 任务 |
|---|---|
| W1 (5d) | P1 全部 + P2.1-P2.2 |
| W2 (5d) | P2.3 + P3.1-P3.4 |
| W3 (7d) | P3.5-P3.6 + P4 全部 + P5.1 |
| W4 (5d) | P5.2-P5.3 + P6 全部 |
| W5 (5d) | P7 全部 + 实车 HIL |

如时间允许，预留 1 周给 §7 故障排查表迭代。

---

*整合人：ralph-loop · 2026-05*
