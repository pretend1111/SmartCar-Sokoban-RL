# SmartCar-Sokoban-RL · SAGE-PR 架构重构 TODO

> **🎯 当前主任务 (2026-05-13 起): 推箱专一架构 (Explore-by-Algorithm + NN-for-Push)**
>
> 部署架构定型: OpenART 板上 **传统 BFS 算法做识别探索, NN 专心做推箱**. NN 模型 / 数据 / 候选生成 全部**砍掉跟探索相关的部分**, 重训一个干净、专注、高效的推箱模型. 整个模型架构对 inspect / info_gain / partial-obs 的预留 inductive bias 不再需要 — 全删, 让模型容量集中在推箱.
>
> **完成判定**: 在 `--external-explorer` 模式下 (跑 plan_exploration_v3 → NN 推箱) 全量 1010+ maps eval, **phase 1-6 全部 ≥ 95%**, explorer-fail 图回退到初始状态 NN 也能搞定. 同时模型架构清单确认没有 inspect / info_gain 相关代码路径残留. 输出 `<promise>DONE</promise>` 仅当全部条件 unequivocally TRUE.
>
> **硬件**: Intel Core Ultra 7 265K + RTX 5060 Ti 16 GB. 环境: `conda run -n rl python ...`.

---

## 🚀 新架构任务清单 (按顺序执行)

### Step 1. 候选生成器砍掉 inspect 类型 (`smartcar_sokoban/symbolic/candidates.py`) ✅
- [x] `generate_candidates(bs, feat, push_only=True)` 加 push_only 默认 True 时**不生成 inspect 候选**
- [x] 旧 inspect 相关分支保留 (push_only=False) 但不再走默认路径, 兼容旧 npz 加载
- [x] 单测: 任一 phase 5 / 6 图 fully_observed=True 下生成的 cands `[c.type for c in cands]` 全 push_box / push_bomb, 无 inspect
- [x] V2 调用方 (build_dataset_v6, belief_ida_solver, dagger_v2) 显式 push_only=False 保兼容
- **完成判定**: tests/test_candidates.py 加 3 个 push_only test, 12 项全过 (0.13s)

### Step 2. 候选特征瘦身 (`smartcar_sokoban/symbolic/cand_features.py`) ✅
- [x] `encode_candidate` 保持 128 输出 (npz 向后兼容), 加 `slice_push_only_cand()` 切到 118 维
- [x] 切掉 SEG_INFO_GAIN [108:118] 10 维 (探索信号, NN 不需要)
- [x] 加常量 `CAND_FEATURE_DIM_PUSH = 118`
- **完成判定**: tests/test_cand_features.py 加 2 个 push_only 切片 test, 10 项全过 (0.18s)

### Step 3. 网格特征瘦身 (`smartcar_sokoban/symbolic/grid_tensor.py`) ✅
- [x] 加 `slice_push_only_grid()` 切掉 ch13 (box_known_mask) / ch15 (target_known_mask) / ch29 (info_gain_heatmap)
- [x] 加 `slice_push_only_global()` 切掉 u_global [4,5,6,9] (unidentified / fully_obs / FOV ratio)
- [x] ch17-21 box_id_inferred 保留 (push 距离归一化, 推箱必需)
- [x] `GRID_TENSOR_CHANNELS_PUSH = 27`, `GLOBAL_DIM_PUSH = 12`
- **完成判定**: tests/test_grid_tensor.py 新增 test_push_only_slice, 9 项全过

### Step 4. 模型架构瘦身 (`experiments/sage_pr/model.py`) ✅
- [x] 新 `SAGEPushOnlyRanker` 类 + `ScoreValueHeads` 只含 score + value (无 deadlock/progress/info_gain)
- [x] `forward()` 返回 `(score, value)` 二元组
- [x] `build_push_only_model()` 工厂: 102K params (跟 default 105K 接近 — 因为去掉 3 个 1x96 Linear 只省 ~3K params, 主要是输入维度切片省的)
- [x] `build_push_only_large()` 工厂: 190K params
- [x] `build_model_from_ckpt()` 加 `detect_model_arch()` 自动识别 push_only / full
- **完成判定**: build_push_only_model().num_parameters() = 102,530; 旧 v3_large9 ckpt 仍能加载 (向后兼容)

### Step 5. 训练 loss 瘦身 (`experiments/sage_pr/train_sage_pr.py`)
- [ ] 删除 L_info / L_progress 计算和加权项 (loss 只剩 L_policy + L_value)
- [ ] 数据加载时把旧 npz 的 X_grid / X_cand 切片到新维度 (兼容老数据格式不重新生成)
- [ ] `--arch push_only` 默认: 用 `build_push_only_model()`
- **完成判定**: 训 1 epoch 跑通 no error, val_acc 出数字

### Step 6. 数据集复用 (不重生)
- [ ] 用现有 `runs/sage_pr/full_v5_v3/phase{1..6}_exact.npz` (已经 post-explorer)
- [ ] 用所有 dagger_v1_r* 和 dagger_targeted_p5_* (也是 post-explorer)
- [ ] 训练加载时 slice 到新维度 (新 X_grid_ch < 30, X_cand_dim < 128)
- **完成判定**: 训练 log 显示 `train n=` 跟之前数字一致 (例如 347248)

### Step 7. 训练 v5_push_only
- [ ] `python experiments/sage_pr/train_sage_pr.py --arch push_only --tag v5_push_only --batch-size 256 --lr 3e-4 --epochs 80 --phase-dist hard --num-workers 0` 用全部数据
- [ ] 大约 25-40 min 在 RTX 5060 Ti
- **完成判定**: train.log 显示 val_acc ≥ 0.95

### Step 8. 全量 eval 加 --external-explorer
- [ ] `rollout_search_eval.py --ckpt v5_push_only/best.pt --phases 1-6 --use-verified-seeds --max-maps 1011 --beam 8 --lookahead 25 --mode v1 --external-explorer`
- [ ] 串行跑 6 phase (每个 ~50 min, 共 ~5 hr)
- **完成判定**: 6 个 phase 全部 ≥ 95%

### Step 9. 如有 phase < 95% → targeted DAgger
- [ ] 用 `experiments/sage_pr/dagger_targeted.py` 对 fail 图集中 oversample
- [ ] 重训 v5_push_only_dag1 加新 DAgger 数据
- [ ] 重 eval (Step 8)
- **完成判定**: 全 phase ≥ 95%

### Step 10. 写 README + push GitHub
- [ ] 把新架构写到 README §3 当前状态
- [ ] §10 遗留问题里把旧 (V1 fully-observed leakage) 的 §10.1 改成"已解决, 当前架构 = 传统 explorer + NN push"
- [ ] commit + push

---

> **历史目标 (已废弃 - 自主探索路线)**: 用 V2 数据集 (god-mode A + 抑制场 + 嵌入 inspect) 训练模型让自己学探索. 已确认效率低 / inductive bias 错配 / 不必要. **改走 explorer 算法 + NN push 双系统**, 模型干净专注.
>
> **当前模型 (历史 baseline)**: `v3_large9/best.pt` (large 194K params 含 info_gain_head 等冗余). 不再使用, 但留作对比 baseline.
>
> **硬件**: Intel Core Ultra 7 265K + RTX 5060 Ti 16 GB. **环境**: `conda run -n rl python ...`.
>
> **核心纪律**: 启动长跑脚本前开 §6 监控; 每 5-10 min grep 监控日志; CPU < 70% 或 GPU < 50% 立即诊断瓶颈.

---

## ▶ 下一步指针（每次迭代开始前先读这里）

```
当前阶段：P4 重训 (V2 数据 + L_info / L_ranking 监督)
当前任务：训 V2 模型, 评估全 phase, 不达 95% 不停

数据集状态:
  V1 = pure exact + plan_exploration  (runs/sage_pr/full_v5/, 84k samples, 翻译 100% 验证, 保留备用)
  V2 = god-mode A + 抑制场 + 嵌入 inspect (runs/sage_pr/full_v6/, 96k samples, 翻译 100% 验证)
  覆盖: V2 救回 V1 失去的 471 张 explore_incomplete 图, 仅余 4 张 god 真无解.

V2 数据组成:
  phase 1-2: 0% inspect (单类自锁)
  phase 3:   8.3% inspect, 50% partial-obs samples
  phase 4: 16.5% inspect, 51% partial-obs
  phase 5: 19.3% inspect, 47% partial-obs
  phase 6: 18.1% inspect, 51% partial-obs

架构与训练对齐情况 (FINAL_ARCH_DESIGN.md §5.2 vs train_sage_pr.py 现状):
  L_policy   : ✅ 已实现 (硬 CE — 文档要求 soft Q label τ=0.5, P3.3 待做)
  L_value    : ⚠️  弱监督 (全 1.0, 文档要求 Huber + BCE)
  L_deadlock : ⚠️  弱监督 (老师选的非死锁 = 0)
  L_progress : ⚠️  弱监督 (全 0.5)
  L_info     : ❌ 未实现 — GT 已在 X_cand[:, :, 108], 加 ~10 行
  L_ranking  : ❌ 未实现 — 需 P3.4 hard negative 数据

V2 数据让架构里早已预留的探索 inductive bias 真正生效:
  X_grid ch13/15/17-21/29   (V1 全退化, V2 50% 样本带信号)
  u_global [4][5][6][9]      (V1 全退化, V2 带不确定性)
  X_cand 类型 inspect / 段 [108:118] 信息增益 (V1 全 mask, V2 11k 样本)
  score_head 选 inspect 分支 / info_gain_head (V1 零梯度, V2 真训)

历史 baseline (V1 + plan_exploration 外挂, 100 maps × verified seed):
  hybrid_v2 (rollout + solver fallback): 1=100 2=100 3=99 4=100 5=96 6=99
  这是 "外挂老师 + 模型 + 求解器兜底" 的合作结果, **不是纯神经网络**.

**本轮目标 (纯神经网络模型, 不挂任何 solver / 外挂 plan_exploration)**:
  phase 1=≥95 / 2=≥95 / 3=≥95 / 4=≥95 / 5=≥95 / 6=≥95
  任一 < 95% 不停.

最后一次评估 (v12_bc1, 50 maps × verified seed, top-k=4 step-limit=100):
  v2 模式: p1=100 p2=98 p3=58 p4=56 p5=22 p6=52
  v1 模式: p1=100 p2=98 p3=84 p4=92 p5=28 p6=86
  进展: 比 v2_bc1 phase 5 提升 8→22 (V2 mode, +14pp).
  问题: phase 3/4/5/6 < 95%, phase 5 最弱.
  下一招: DAgger (§7.0 通用诊断 + §7.2/7.3) — 收集 v12_bc1 失败状态, 老师重新打 label.
最后一次评估时间：2026-05-11
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

> 数据格式必须跟 SAGE-PR 输入对齐：每个样本包含 `(X_grid, X_cand, u_global, mask, label, phase, source)`。
>
> 当前已有两套数据集:
> - **V1** (`runs/sage_pr/full_v5/phase{1..6}_exact.npz`, 84k samples): pure exact (god-mode + plan_exploration), 全 fully_obs, 无 inspect 标签 — 模型只学 push, 推理需外挂 plan_exploration
> - **V2** (`runs/sage_pr/full_v6/phase{1..6}_v2.npz`, 96k samples): god-mode A + 抑制场 + 嵌入 inspect, 50% partial-obs + 11k inspect 标签 — 模型自主探索

- ☑ **P3.1** 重写 `build_dataset_v3.py`，支持 candidate-aware 输出
  - 创建 `experiments/sage_pr/build_dataset_v3.py` ✓
  - 每步从 belief state 调 P1.4 候选生成器，得到 `[64, 128]` 候选张量 ✓
  - 每步用 P1.3 领域特征预计算填 `X_grid: [10, 14, 30]` ✓
  - 老师标签：用 BestFirst 求解器找出"专家会选哪个候选" (`match_move_to_candidate`) ✓
  - **完成判定** ✓ phase 1 30 任务全 ok 321 samples; phase 4 verified 5 任务 89 samples; npz keys 齐全

- ☑ **P3.1.1** 翻译器 100% 验证 (build_dataset_v5 + verify_translator.py)
  - 修复 candidate generator 三个 bug: bomb-aware deadlock_mask + box→bomb 链推 + cycle-check
  - 90 maps × 1349 push 步 0 diverge, 0 label_miss
  - 全量 V1 (84k samples) inline verify 0 diverge

- ☑ **P3.1.2** V2 数据集 + 抑制场 (build_dataset_v6.py)
  - god-mode A 路径 + partial-obs 重放 + suppression on push-onto-target-with-unlocked-σ + insert inspect
  - 全量 V2 (96k samples) inline verify 0 diverge
  - 救回 471 张 V1 失去的 explore_incomplete 图 (phase 5 救回 279 张)

- ☑ **P3.1.3** 抑制场实现
  - candidates.py: `_gen_push_box_candidates(enforce_sigma_lock=True)` — push 落到 target cell 必须 σ 锁定 (Π 单射), 否则标 illegal
  - macro 候选 (run_length=2,3) 同步检查
  - V1 不开 (默认 False), V2 开
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

## P4 · Stage A — BC 预训练 (V2 数据 + 完整辅助监督)

> 当前 train_sage_pr.py 只用 P4 stage-A 简化损失 (L_policy + 0.3·L_value + 0.2·L_deadlock + 0.2·L_progress, 全部弱监督). FINAL_ARCH_DESIGN.md §5.2 设计的 **L_info / L_ranking** 还没实现 — 这是本轮要补的事。

- ☐ **P4.0** 补充辅助监督损失 (architecture-already-supports, 训练代码补)
  - **L_info (Huber, λ=0.1)**: GT 已经写在 `X_cand[:, :, 108]` (cand_features.py 段 [108:118] 的"信息增益: viewpoint IG, n_unidentified, exclusion-1-flag")
    ```python
    target_ig = X_cand[..., 108].detach()
    loss_info = F.smooth_l1_loss(info_gain, target_ig, reduction='none')
    loss_info = (loss_info * mask).sum() / mask.sum().clamp_min(1.0)
    ```
  - **L_value 改强监督 (Huber + BCE, λ=0.3)**: 用 episode 是否赢 + n_remaining_pushes 作 GT (要在 build_dataset_v6 加 episode 元数据)
  - **L_deadlock 改 per-cand 强监督 (BCE, λ=0.2)**: 模拟 push 后看 deadlock_mask, 给所有 push 候选打 0/1 (而非只监督选中)
  - **L_progress 改 per-cand 强监督 (Huber, λ=0.2)**: 用 push_dist_field 推完后比推前的差作 GT
  - **L_ranking (margin, λ=0.5)**: 需要 P3.4 hard negative — 短期 P4 可先跳过, P5 DAgger 时补
  - 总损失: `L = L_policy + 0.5·L_ranking + 0.3·L_value + 0.2·L_deadlock + 0.2·L_progress + 0.1·L_info`
  - **完成判定**: train_sage_pr.py 跑 1 epoch, 5 个 loss 项都有非零梯度

- ☐ **P4.1** 训练 V2 模型 (主 baseline, 不带 ranking)
  - 数据: `runs/sage_pr/full_v6/phase{1..6}_v2.npz` (96k samples)
  - 优化：AdamW lr=3e-4 cosine 到 3e-5、batch 256、weight decay 1e-4、grad clip 1.0
  - 80 epoch，phase-stratified sampling（按文档权重 5/10/15/25/20/25%）
  - 数据增强: D₂ 几何 (4 元素) + ID 重命名 (S_10) + 部分可观测 mask (p ∈ [0.1, 0.4])
  - **完成判定**：train_loss 收敛，val_acc ≥ 95% on phase 1-3 集 (pure-神经, 无外挂)

- ☐ **P4.2** 全 phase deterministic eval (纯神经, 无 plan_exploration)
  - 创建 / 改 `evaluate_sage_pr.py`: 模型从 t=0 partial-obs 开始, 自主决策 inspect / push, 直到 won 或 step_limit
  - 对每个 phase 跑全图 × 3 seed deterministic（greedy, 无 beam search）
  - 输出 phase 1-6 win_rate
  - **完成判定 (硬目标)**：phase 1=≥95% / 2=≥95% / 3=≥95% / 4=≥95% / 5=≥95% / 6=≥95%
    - 任一不达标 → P4.3 / P5 / §7

- ☐ **P4.3** 对照诊断
  - 比较 V1 model (含 plan_exploration 外挂) vs V2 model (纯神经)
  - 比较 V2 model 的 push 准确率 vs inspect 准确率
  - 失败 phase 抽样可视化 trajectory (preview_sage_pr.py), 看模型何时该 inspect 没 inspect / 何时该 push 没 push
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

> 主循环。**任一 phase < 95%** 就回这里，按 §7 故障排查表挑下一步动作，永远不停。

- ☐ **P8.1** 弱图 branch search 收集
  - 取 P5 / P6 eval 中 win_rate < 50% 的图
  - 每图跑 budget=128 branch search，把成功 trajectory 加入数据集
  - 弱图 + 失败 trajectory 重训
- ☐ **P8.2** 终止判断 (硬目标)
  - **phase 1-6 全部 ≥ 95% 且 INT8 损失 ≤ 2pp → 进 P7 量化导出 (如未做)**
  - 任一未达标且连续 2 轮无进步 → §7 故障排查表挑下一招
  - 同一招连续用 2 次未见效 → 切别的招
  - **不允许"phase X 太难, 接受 < 95%"的妥协**: 改 eval seed 集 / 改训练数据 / 改架构 / 加 DAgger / 加 ranking, 都要试到达标为止
- ☐ **P8.3** V2 失败救图机制
  - 若 V2 模型 + 纯神经推理在某 phase 卡死 < 95%, 尝试:
    1. P4.0 完整辅助监督 (L_info / 强 L_value 等) 是否打开
    2. DAgger 收集 V2 模型自己 rollout 的失败 state, 老师 (god-mode + 抑制场) 重新打 label
    3. 加 hard negative (P3.4) 训 ranking
    4. 推理时加 short beam search (B=3, D=3)
    5. 若 4 招都用过仍不达 → 检查数据集里那批失败图的 partial-obs 标签是否一致 (god-mode A 在不同 seed 下指方向不同 = 训练歧义)

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

### 7.0 V2 训练 / 推理通用诊断 (任意 phase < 95% 都先查)

- **L_info 是否开**: 打 1 个 batch 看 `info_gain_head` 输出方差; 全 0 = 没监督
- **抑制场是否生效**: dataset gen 时打 `args.enforce_sigma_lock`, eval 时同步开 (推理候选生成必须跟训练一致, 否则模型见到训练没见过的 legal mask)
- **inspect 标签比例**: phase 4-6 应 15-20%; 如果远低 → suppression 没触发 / pick_inspect_for_unlock 选错
- **partial-obs samples 占比**: 应 ~50% in phase 4-6; 太低 → 数据生成 bug
- **inference rollout 死锁**: 模型 partial-obs 状态下选了非法 push? 检查推理时是否同步 enforce_sigma_lock=True 在候选生成
- **label_miss 步数高**: V2 dataset gen 里某些步数 trajectory 没录 sample, 模型推理遇到这种状态盲 → DAgger 用这些 (map, seed, step) 重新收集

### 7.1 phase 1-3 < 95%

- 候选生成器 bug？检查合法 mask（P1.4 单元测试）
- belief state 更新错误？打印 trajectory 看 ID 解推
- soft Q label 失真？降温 τ 0.5 → 0.3 让多个候选概率拉开
- D₂ 增强是否同步变换了候选位置 + 推送方向？
- V2 phase 1-2 应该跟 V1 完全一致 (单类无 inspect), 如果差 → 检查 partial-obs 模拟是否引错噪声

### 7.2 phase 4-5 卡 70-85%

- DAgger 不够？多跑 2 轮 + 上采样失败状态
- Candidate feature 缺关键信号？加 `match_entropy / blast_gain / push_dir_gradient` 通道
- 检查 hard negative 是否覆盖典型死锁模式（推入角、推链锁死）
- value head 训练信号弱？加重 `λ_value` 0.3 → 0.5
- inspect 决策错误率高 (模型该 inspect 时 push)? → 加重 L_info 0.1 → 0.3, 或 inspect 样本上采样 ×2

### 7.3 phase 6 卡 70-85%（炸弹时序问题）

- Plan B：单独训 BombValueHead
  - 输入：炸弹位置 + 内墙拓扑（10×14×3）
  - 输出：每个炸弹的"爆破后连通分量增益"
  - 主网络的炸弹候选评分加这个 head 做 bonus
- 触发 beam search 时 D=4（深一层）
- 训练时上采样 phase 6 含炸弹 trajectory ×5
- V2 数据 force_apply_unsupp ~146 (phase 5) / 69 (phase 6) — 这些 step 没录 sample, 模型推理遇到会盲, 加 DAgger 修

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
| 2026-05-08 | **Pure solver baseline (BestFirst 30s)** | p1=100 p2=100 p3=100 p4=46 p5=100 **p6=71** |
| 2026-05-08 | Pure solver auto+60s | p4=47 (+1) p6=71 (=) |
| 2026-05-08 | Pure solver IDA* 25s | p4=13 p6=44 (IDA* 太慢, 弱于 BestFirst) |
| 2026-05-10 | 翻译器 100% 准 + 全量验证 V1 | 84k samples 0 diverge, 0 label_miss; 修 bomb-aware deadlock + box→bomb 链 + cycle-check |
| 2026-05-10 | V2 数据集生成 + 验证 | 96k samples 0 diverge; 救回 471 张图; 11k inspect 样本 + 50% partial-obs |
| 2026-05-10 | candidates.py 加抑制场 (enforce_sigma_lock) | 推到 target 必须 σ 锁定; macro 同步检查 |
| — | **P4.0 加 L_info / L_value / L_deadlock 强监督** | **下一步, 不达 95% 不停** |
| — | P4.1 V2 训 + P4.2 全 phase 纯神经 eval | **下一步** |
| — | P3.3 Soft Q label + 强化 value | TODO (P5 时再考虑) |
| — | P6 QAT 完成 | — (需先达到 fp32 目标) |

---

## 9. 不做的事（避免跑偏）

- **不要**改引擎物理（推链、爆炸、配对、对角推墙特例）— 这些是赛题规则的复刻
- **不要**为了拉高 win_rate 把困难地图剔出训练集 / 评估集
- **不要**部署 fp32 模型到 OpenART
- **不要**phase 6 达标 90% 就停 — 本轮目标统一 ≥ 95%, **任一 phase < 95% 都不能停**
- **不要**用外挂 plan_exploration / solver fallback 的 win_rate 充当本轮目标 — 那是 V1 路线已达成的合作结果, 本轮目标是**纯神经网络模型自主完成探索 + 推箱**
- **不要**为了让 phase 4 verified seeds "够 95%" 而砍掉难图 / 换 seed 集 — 95% 必须在原 verified seed 集上达成 (改了 eval 协议等于自欺)
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
