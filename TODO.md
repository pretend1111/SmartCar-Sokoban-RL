# SmartCar-Sokoban-RL · 训练攻关 TODO

> **目标**：在 phase 6 全图集上跑出 **deterministic policy 通关率 ≥ 90%** 的小模型（int8 量化后 < 500 KB，可部署 OpenART mini）。
>
> **终止条件**：上一条不达成，**不能停止迭代**。任何一次评估指标低于 90% 就回到下面"故障排查表"找下一步动作，永远有事可做。
>
> **硬件**：Intel Core Ultra 7 265K（20 核 8P+12E）+ NVIDIA RTX 5060 Ti 16 GB。
>
> **环境**：所有 python 命令必须在 conda 环境 **`rl`** 下运行。Bash 里用 `conda run -n rl python ...` 或先 `conda activate rl`。
>
> **核心纪律**：每次启动训练 / 数据生成脚本，**必须并行起一个资源监控**（见 §6）。CPU 利用率 <70% 或 GPU 利用率 <50% 时立即停下来诊断瓶颈，不要让脚本空跑浪费时间。

---

## ▶ 下一步指针（每次迭代开始前先读这里）

```
当前阶段：P0 — 基线审计
当前任务：P0.2 (P0.1 已完成)
最后一次评估：phase6 win=9.33% (14/150, 50图×3seed) | val_acc=64.7% | dataset=2043样本
最后一次评估时间：2026-05-07 02:55
```

> **每完成一个任务**：把 ☐ 改成 ☑，更新"当前任务"指针指向下一个未完成项，把评估数字写进上面三行。

---

## 0. 阶段总览（依赖图）

```
P0 基线审计           ─┐
P1 地图生成器修复      ─┴→ P3 数据集构建 ─┐
P2 模型架构升级        ──────────────────┴→ P4 BC 训练 ─→ P5 评估 ─→ P6 自我改进迭代 ─→ P7 量化与导出
                                                          ↑                  │
                                                          └──── 反馈到 P3 ───┘
```

P0 / P1 / P2 是前置工作，可以并行。**P5 通关率 < 90% 时永远走 P6 的循环回到 P3**，直到达标才能进 P7。

---

## P0 · 基线审计（建立可比指标）

> 不知道现在多差，就不知道每一步改进了多少。先把当前底盘的数字钉死。

- ☑ **P0.1** 跑一次当前 BC prototype 在 phase 6 上的 baseline (2026-05-07)
  - 结果：50 图×3 seed，dataset 2043 样本（hybrid 老师：84 solver + 17 autoplayer，49 ep 失败）
  - 训练 20 epoch，hidden=256，val_acc=64.7%
  - **rollout win_rate=9.33% (14/150)**，离 90% 目标差 80+ 个百分点
  - 资源：CPU build 期 87–96%（达标 ≥85%），train 期 GPU 利用率被 5060Ti 上 ~1ms/step 的 FC 跑得起飞，瞬时 50% 但平均较低（数据集太小、20 epoch 仅 30 秒就完）
  - 命令（重新 build 当前 prototype 的数据 + 训练）：
    ```bash
    conda run -n rl python -m experiments.solver_bc.build_dataset \
      --phase 6 --output .agent/baseline/p6.npz \
      --seeds-per-map 3 --time-limit 30 --num-workers 16
    conda run -n rl python -m experiments.solver_bc.train_bc \
      --dataset .agent/baseline/p6.npz \
      --output-dir .agent/baseline/p6_run \
      --device auto --epochs 20 --batch-size 256 --hidden-dim 256
    conda run -n rl python -m experiments.solver_bc.evaluate_bc \
      --checkpoint .agent/baseline/p6_run/best.pt --phase 6 --device auto \
      --rollout-batch-size 64
    ```
  - **完成判定**：拿到 phase 6 的 `win_rate / avg_steps` 数字写到本文顶部"最后一次评估"。
  - 监控：训练时 GPU 利用率应 > 70%（FC-only 模型小，DataLoader 不应是瓶颈，用 `--gpu-resident on`）；数据生成时 CPU 应 90%+。
- ☐ **P0.2** 跑一次 phase 6 上 IDA\* 严格最优解的"老师上限"
  - 用刚改完的 4 向 `MultiBoxSolver`，对每张 phase 6 图算 `(strategy='ida', time_limit=300)` 的最优推数，写到 `.agent/baseline/phase6_optimal.json`。
  - 这是 BC 学得再好也不可能超过的天花板。
  - **完成判定**：所有 phase 6 图都有最优解（或被标记为 IDA\* 5 分钟内无解，需要回 P1 改图）。
- ☐ **P0.3** 跑当前 RL `MaskablePPO` 模型（如果 `runs/rl/models/phase6_best.zip` 存在）的 deterministic 评估，作为 RL 路线的对照。
  - 命令：`conda run -n rl python -m smartcar_sokoban.rl.train --eval runs/rl/models/phase6_best.zip`
  - 没有就跳过，本任务标 ☑（跳过原因写到顶部）。

---

## P1 · 地图生成器修复（输入数据质量）

> "垃圾输入垃圾输出"。这一阶段不达标就别开始 P3。

- ☐ **P1.1** 诊断当前生成器
  - 读 `scripts/maps/gen_quality_maps.py` / `gen_1000_maps.py` / `regen_phase456.py`
  - 输出诊断报告 `.agent/diag/map_gen.md`，至少回答：
    - 现有"质量"过滤标准是什么？是否真的在过滤还是只是采样？
    - 生成器有没有验证图能被 IDA\* 解？
    - 推数分布、箱子数分布、死锁出现率？
    - 哪些图是 trivially 解（推 ≤ 5 步）或 unsolvable？
  - 监控：诊断本身 CPU 用不到，OK。
- ☐ **P1.2** 写一个 `verify_optimal.py`，对每张图：
  - 跑 `MultiBoxSolver(strategy='ida', time_limit=60)`，要求**找到最优解**才算通过
  - 跑 AutoPlayer 必须也能解（不能解的图过难）
  - 推数 ≥ 阈值（phase 1: ≥ 3, phase 6: ≥ 15），过滤 trivial 图
  - 用 `ProcessPoolExecutor(max_workers=18)`
  - 监控：CPU 必须 90%+。低于 70% 检查 worker 数 / 是否被 GIL 锁。
- ☐ **P1.3** 重新生成 phase 1–6 各 **1000 张验证过的图**（共 6000 张），每张图带 ≥ 5 个验证可解的 seed
  - 推数分布要均匀：phase 6 应该有 15/20/25/30+ 推的图，不能扎堆
  - 写到 `assets/maps_v2/phase{N}/` 下，**不要覆盖现有 `assets/maps/`**，先 A/B 对比
  - 监控：长时间（预计数小时）的批量任务，确保 CPU > 85% 且没有少数 worker 拖后腿
  - **完成判定**：6000 张图全部通过 P1.2 验证；推数分布直方图（写入 `.agent/diag/maps_v2_dist.png` 或 .json）合理。

---

## P2 · 模型架构升级

> 当前 145 K 纯 FC 不够。把容量花在墙体的卷积上。

- ☐ **P2.1** 在 `experiments/solver_bc/` 下新建 `policy_conv.py`，实现：
  ```
  墙体 16×12 → Conv(1→16, 3×3) → ReLU → Conv(16→32, 3×3) → ReLU
            → Flatten → Linear(...→64)        ← 空间嵌入
  实体 62 维 ──────────────────────────────────┐
                                                 concat → Linear(126→256) → ReLU
                                                       → Linear(256→256) → ReLU
                                                       → Linear(256→54)
  ```
  - 严格只用 TFLite Micro 支持的 op：`Conv2D / FullyConnected / ReLU / Reshape`，不要 LayerNorm / Dropout / Attention
  - 输入要从扁平 254 维拆出 `(walls_16x12, entities_62)`，加个 `split_obs(obs)` 工具
- ☐ **P2.2** 让 `train_bc.py` 支持新策略选择：`--policy {mlp, conv}`
- ☐ **P2.3** 单步前向 sanity check：在 GPU 上跑一个 batch=512 的前向，验证显存占用 < 1 GB、单步 < 5 ms。
  - **完成判定**：模型能跑通、参数数量 ≈ 110–500 K（拍照写到 `.agent/diag/policy_conv_summary.txt`）

---

## P3 · 数据集构建（IDA\* 严格最优老师）

- ☐ **P3.1** 改 `experiments/solver_bc/teachers.py`：
  - `_solver_action_sequence` 把 `advise_exact_high_level` 的 `time_limit` 加大（比如 120 s），并改成 `strategy='ida'`（如果不暴露就在 `high_level_teacher.py` 加一层）
  - 让 hybrid 老师**不再回退到 AutoPlayer 拿 BC 数据**——AutoPlayer 不是最优解，会污染监督信号
  - 改成：IDA\* 失败的 episode 直接丢弃，记到日志
- ☐ **P3.2** 重建数据集（用 P1.3 的 `maps_v2`）
  - 每个 phase: 1000 图 × 5 seed = 5000 episode
  - 命令模板：
    ```bash
    conda run -n rl python -m experiments.solver_bc.build_dataset \
      --phase 6 --output .agent/data/p6_v2.npz \
      --seeds-per-map 5 --time-limit 120 --max-cost 200 \
      --teacher solver \
      --num-workers 18
    ```
  - 监控：CPU 必须 90%+，否则诊断（worker 之间 IDA\* 时间方差大可能让某些核空转，可以加任务粒度切分）
  - **完成判定**：每个 phase 至少 100 K (state, action) 样本；老师成功率 ≥ 95%。生成 `.agent/data/p{N}_v2.json` 摘要。
- ☐ **P3.3** 数据集合并 + 分层（保留 phase 标签作为辅助分析维度）：
  - 输出 `.agent/data/all_v2.npz`（≈ 600 K 样本）
  - 数据集本身预计 < 500 MB，可以全量 GPU resident（5060 Ti 16 GB）

---

## P4 · BC 训练（吃满 GPU）

- ☐ **P4.1** 训练 baseline conv 策略
  - 命令：
    ```bash
    conda run -n rl python -m experiments.solver_bc.train_bc \
      --dataset .agent/data/all_v2.npz \
      --output-dir .agent/runs/conv_h256 \
      --policy conv --hidden-dim 256 \
      --epochs 80 --batch-size 1024 --lr 5e-4 \
      --device auto --gpu-resident on
    ```
  - **资源目标**：
    - GPU util **>= 80%**（nvidia-smi 监控）
    - VRAM 占用 < 14 GB
    - 单 epoch < 60 s（600 K 样本 × bs 1024 = ~600 step，~30 ms/step）
  - 如果 GPU < 50%：先增大 batch_size 到 2048 / 4096，再开 `torch.compile(model)`，再排查 DataLoader（应该已经 GPU resident）。
- ☐ **P4.2** 跑一组超参 sweep
  - 网格：`hidden ∈ {256, 512}`，`epochs ∈ {80, 200}`，`lr ∈ {1e-3, 5e-4, 1e-4}`
  - **同时只跑 1 组**（GPU 是单卡），跑完用 P5 评估，记录到 `.agent/runs/sweep.json`
  - 每组训练前先 `nvidia-smi --query-gpu=memory.used --format=csv` 看 VRAM 是不是干净
- ☐ **P4.3** 选当前最佳 checkpoint 进入 P5

---

## P5 · 评估

- ☐ **P5.1** 全 phase deterministic 评估
  - 命令：
    ```bash
    conda run -n rl python -m experiments.solver_bc.evaluate_bc \
      --checkpoint .agent/runs/<best>/best.pt \
      --phase 6 --device auto --rollout-batch-size 256
    ```
  - 同样跑 phase 1/2/3/4/5 / 6 各一次，结果写入 `.agent/eval/<run>.json`
  - 把 phase 6 win_rate 写到本文顶部"最后一次评估"
- ☐ **P5.2** 失败地图归类
  - 列出 phase 6 中 win_rate < 100% 的图集
  - 写入 `.agent/eval/<run>_weak.json`
- ☐ **P5.3** **达标判断**
  - **phase 6 win_rate ≥ 90%** 且 **phase 1–5 win_rate ≥ 95%**：进 P7
  - 否则：进 P6

---

## P6 · 自我改进循环（直到达标）

> 这是主循环。每跑一遍 P6 就触发一次 P3→P4→P5。**永远不会跳过这一节直到达标**。

每轮迭代必须做：

- ☐ **P6.1** 弱图 branch search
  - 取 P5.2 的 weak maps，对每张图 × 多 seed 跑 `branch_search.py`：
    ```bash
    conda run -n rl python -m experiments.solver_bc.branch_search \
      --checkpoint <best.pt> --phase 6 \
      --map-filter <weak_map> --seeds-per-map 8 \
      --branch-budget 256 --branches-per-rollout 16 \
      --rollout-batch-size 64 \
      --output-npz .agent/data/improve_iter{N}.npz \
      --output-json .agent/data/improve_iter{N}.json \
      --device auto
    ```
  - 监控：GPU rollout 时利用率应 > 60%（注意 branch_search 有 CPU 串行调度，不会 100%）
- ☐ **P6.2** 把改进的 trajectory 拼回主数据集
  - `np.savez_compressed(.agent/data/all_v3.npz, ...)`
  - 检查样本数有显著增加（< 1% 说明 branch search 没找到改进，要换策略：调大 budget 或先跑下面 P6.3）
- ☐ **P6.3** 重训
  - 用同 P4 的命令，`--dataset .agent/data/all_v{N+1}.npz`
  - 输出到 `.agent/runs/conv_iter{N}`
- ☐ **P6.4** 重评估（同 P5）→ 更新顶部"最后一次评估"
- ☐ **P6.5** 终止判断
  - 达标：跳出循环 → P7
  - 未达标但有进步（比上轮高 ≥ 1%）：迭代次数 +1，回 P6.1
  - **未达标且连续 2 轮没进步**：触发"故障排查表"§7

---

## P7 · 量化与导出

> 仅在 P5/P6 达标后做。

- ☐ **P7.1** PyTorch → ONNX
  - `torch.onnx.export(...)`，opset 13+
- ☐ **P7.2** ONNX → TFLite int8（用 onnx2tf 或 TFLiteConverter + representative dataset）
  - representative dataset 用训练集的 1024 个样本
- ☐ **P7.3** 量化精度回归
  - 在 PC 上用 tflite_runtime 跑 P5 同样的评估
  - **要求**：phase 6 win_rate 下降 ≤ 2%
  - 否则回 P7.2 调（试 per-channel quantization、提高校准样本数）
- ☐ **P7.4** 部署文件打包到 `runs/deploy/<date>/`：`policy.tflite + labels + obs_layout.md`

---

## 6. 资源监控规范（**强制**）

每次跑长时间任务（数据生成 / 训练 / branch search），**前 60 秒内必须开监控**。任意一项不达标 → 停下来诊断。

### 6.1 监控命令模板

新开一个 shell（或同一脚本背景），每 5 秒打印一次：

```bash
# 综合（长跑用）
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
           --format=csv -l 5 > .agent/monitor/gpu_$(date +%Y%m%d_%H%M).log &

# CPU（PowerShell 也行）
typeperf "\Processor(_Total)\% Processor Time" -si 5 -sc 720 \
  > .agent/monitor/cpu_$(date +%Y%m%d_%H%M).log &
```

或者一行总结脚本（推荐写到 `scripts/monitor_resources.py`）：每 5 s 打印一行 `t=... cpu=__% gpu=__% vram=__/16GB ram=__GB`，方便 loop agent grep。

### 6.2 阈值与对策

| 任务类型 | CPU 目标 | GPU 目标 | 不达标时排查 |
|---|---|---|---|
| 地图生成 / 验证 / 数据集构建 | **≥ 85%** | < 5% | worker 数太少？IDA\* 时间方差导致少数核拖尾？任务粒度太大？尝试把每个 worker 的 batch 拆细 |
| BC 训练 | 1–2 核满 | **≥ 80%** | DataLoader 瓶颈？打开 `--gpu-resident on`；batch_size 太小？翻倍；模型太小？开 `torch.compile`；CPU 预处理在 hot path 上？挪到 GPU |
| Branch search rollout | 50%+ | 50%+ | 增大 `--rollout-batch-size`；切到 `--rollout-backend gpu`（见 `experiments/gpu_sim/`） |
| 评估 (evaluate_bc) | 50%+ | 50%+ | 增大 `--rollout-batch-size`；评估并行化 |

### 6.3 长跑健康检查（每 30 分钟一次）

- VRAM 增长是否正常（应该平稳，持续上涨可能 OOM）
- 训练 loss 是否在下降（连续 3 epoch 同水平 → 学习率太低 / batch 太小）
- GPU 温度 < 80°C（5060 Ti 没问题，但记录一下）

---

## 7. 故障排查表（**主循环卡住时查这里**）

> 进入 P6 第二轮之后没有进步、或某个具体指标卡住时，按下面顺序排查。

### 7.1 win_rate 在 60–75% 区间卡住

- 检查老师质量：随机抽 10 张 phase 6 弱图，手动用 `preview_failed.py --solver exact` 看 IDA\* 解法是否合理
- 检查数据集多样性：每张图的 seed 是否真的产生不同初始状态？（`seed` 影响箱子初始位置）
- 加大模型容量：`--hidden-dim 512`，加一层 ResBlock
- 加 augmentation：训练时随机左右镜像（同时镜像 obs 的墙体维度和动作的左右）

### 7.2 win_rate 在 80–88% 区间卡住

- 用 `branch_search` 配 `--branch-budget 1024 --branches-per-rollout 32` 加大搜索深度
- 部署时考虑 inference-time top-k 兜底（在 OpenART 端选 top-2 动作分别 rollout 一步看哪个不死锁）—— 这是 P7 后的部署优化，但训练阶段可以先在 PC 上验证有效性
- 检查 distribution shift：把 BC 推理时遇到的"专家从未见过的状态"采集回来 → 这正是 self_improve_loop 该做的

### 7.3 训练 loss 不降

- 学习率太大（NaN）/ 太小（卡 plateau）
- 数据集严重不平衡（某动作占 80%）：检查 `np.bincount(actions)`
- mask 用错（loss 在算非法动作上的损失）

### 7.4 GPU 利用率永远 < 50%

- 模型太小：当前 conv 网络确实小，但开 `torch.compile` 应该能拉到 70%+
- batch_size 太小：先翻倍到 2048
- 数据每 step 都从 CPU 拷贝：`--gpu-resident on`
- 真的瓶颈在 host：用 `torch.profiler` 抓一段看

### 7.5 数据生成阶段，少数 worker 跑得慢

- IDA\* 在难图上会把 60–120 s 全用满，简单图几秒就完
- 把生成任务按"每图独立"投递，而不是每 worker 一批图
- 极难图（IDA\* 5 分钟还没解）直接丢弃，回 P1.3 重生成

---

## 8. 已完成 / 已知数字

> 每次完成一个任务把数字补到这里，方便后续复盘。

| 时间 | 阶段 | 结果 |
|---|---|---|
| — | — | — |

---

## 9. 不做的事（避免跑偏）

- **不要**改引擎物理（推链、爆炸、配对规则）—— 这是赛题规则的复刻
- **不要**为了拉高 win_rate 而把困难地图剔出训练集
- **不要**部署没量化的 fp32 模型到 OpenART
- **不要**在 phase 6 上达标 90% 就停下，要先验证 phase 1–5 也都 ≥ 95%
- **不要**关掉 action mask（哪怕在 OpenART 上也要保留 mask 推理）
