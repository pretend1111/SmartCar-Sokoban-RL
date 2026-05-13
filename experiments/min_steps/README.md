# min_steps — Belief-Aware Sokoban Teacher (v18)

迭代 18 代演化得到的 belief-aware oracle,用于 SAGE-PR 神经网络的监督信号生成。
对车载推箱子赛题的"部分可观测 + N-1 排除律 + 几何 forced pair"做端到端规划。

## TL;DR

```python
from experiments.min_steps.teacher import teach

eng = GameEngine(); eng.reset(map_path)
eng.discrete_step(6)   # init snap
teach(eng)
# eng.won = True, trajectory ~ 49 step / 13% gambling avg
```

**30-map verified-seed benchmark:**

| Planner | 步数 | Gambling | 时间 |
|---|---|---|---|
| v1 explore-first (老 baseline) | 69.50 | 0% | — |
| v6 (no penalty, oracle) | 48.87 (物理下界) | 25.6% | ~30s |
| **v18 K-best (current)** | **49.40** | **12.8%** | **3 min** |

距物理下界 +0.53 步 (1.1%),gambling 砍半,30 张图全部 won。

---

## 核心问题与设计

这是个 **POMDP** 规划问题:
- **观测限制**:车头 FOV 极窄(dist≈1 紧贴 + 朝向精确对齐 + LOS)
- **N-1 排除律**:`{box.class_id}` 与 `{target.num_id}` 集合双射,识别 N-1 个就推出最后一个
- **Forced pair 几何**:某些 box/target 的几何关系强制配对(只有这一种合法分配),agent 通过拓扑推理无需识别就知道配对
- **盲推风险**:推到 target 但 class ≠ num → 不消除,卡死

目标:**最小化总步数,同时不无脑赌博**。Multi-objective: step count vs gambling rate。

### 评分函数 (multiplicative expected utility)

```
S = log(1 + 1/steps) × p_per_gamble^k
```
- `k` = consumption-side gambling 次数 (推到 target 时 box.class 或 target.num 任一未知)
- `p_per_gamble` ∈ (0,1):单次赌博的假设"成功率",默认 0.85
- 非线性容忍:短路径下 1 次赌损失大,长路径下损失小

线性 fallback: `S = -(steps + λ × k)`

---

## 架构层级

```
planner_oracle_v18 (顶层)
    │ K-best god plans × multi-alpha v14 DP, multiplicative score, sim-based gambling count
    ▼
oracle_v14_min_steps (核心 DP)
    │ Dijkstra over (push_idx, scan_mask, car_pos, q4); relaxed-goal
    │ trust_walk_reveal=True (动态 trigger map per step)
    ▼
MultiBoxSolver.solve_kbest (god plan 多样化)
    │ base solve + first-push 约束变体 + push-swap 交换变体
    ▼
_walk_path_executor (执行器)
    │ 跟随 planner 录制的 path; 失败 fallback BFS
```

### 关键模块

| 文件 | 职责 |
|---|---|
| `teacher.py` | 生产入口 `teach(eng)`,默认 v18 + p=0.85 + k=5 |
| `planner_oracle_v18.py` | 顶层 wrapper,枚举 (plan, alpha) 全部组合,sim-based 评分 |
| `planner_oracle_v14.py` | 核心 belief-aware DP + 执行器 `_execute_actions_v14` |
| `planner_oracle_v4.py` | `_bfs_path_cells_optimal` 等长最短路径中按 (max walk-reveal, end-orient match) 字典序优选 |
| `planner_oracle_v6.py` | 老 baseline (无 penalty,大量盲推) |
| `planner_oracle.py` | `_god_plan(s)` 入口 + `_simulate_god_and_record` 等共享工具 |
| `smartcar_sokoban/solver/multi_box_solver.py` | `solve_kbest` K-best 求解器(本仓库已修改) |

---

## 18 代演化简史

| 版本 | 关键创新 | 30-map step | gambling |
|---|---|---|---|
| v0 | god mode + 0 scan (cheat reference) | 48.00 | 50% |
| v1 | explore-first (先扫完再推) | 69.50 | 0% |
| v6 | base oracle, no penalty | 48.87 | 25.6% |
| v8 | + α × (n-1) soft penalty per gambling first-push | 49.03 | 10.4% |
| v9 | + pair-implicit scan | 49.00 | 类似 |
| v10 | + log entropy penalty | 48.97 | 14.3% |
| v12 | + bijective universe propagation | 49.00 | 11.7% |
| **v14** | **+ relaxed goal (DP 不强制 scan_set 满)** | **48.73** | **14.3%** |
| v18 (旧, single plan) | + per-map best α (multi-α + multiplicative) | 52.33 | 9.0% |
| v18 K-best | + K-best god plans (first-push 多样化) | 54.57 | 2.6% |
| v18 K-best fast | + heuristic memo + top-1 dir + slim alpha | 53.47 | 5.1% |
| **v18 当前** | **+ trust_walk_reveal=True + sim-based gambling count** | **49.40** | **12.8%** |

---

## 关键架构修复(2026-05-13 第八轮深度优化)

破除"+2 步壁垒"的双管修复:

### 1. `trust_walk_reveal=True` (v14 DP 默认)

DP 信任 walk-reveal 估计,生成更短 plan(用 walk-reveal 代替 explicit scan)。
配合 **dynamic trigger map per step** — 每步重建 `(cell, q4) -> scan_idx` 映射,
反映当前 entity 位置(推完 box 移位后 trigger 配置随之更新)。

```python
# planner_oracle_v14.py
trigger_maps_per_step = [_build_trigger_map_at_step(k) for k in range(len(snapshots))]
def reveals_along_walk(start_pos, path, step_k):
    tmap = trigger_maps_per_step[step_k]
    # 沿 path 累计 (cell, last_step_q4) 命中 trigger 的 entity bits
```

### 2. `count_gambling_via_sim` (v18 wrapper)

旧版 wrapper 用 explicit-scan-only 的悲观 count,会把"靠 walk-reveal 识别 entity"
的 plan 误判为 gambling-heavy → 拒绝。修复:在 fresh engine 上**实际 simulate** plan,
用 engine 的 `seen_*_ids` 判 gambling。

```python
# planner_oracle_v18.py
def count_gambling_via_sim(res, plan):
    eng_sim = _fresh_engine_from_eng(eng)
    # monkey-patch discrete_step 在 consumption 时检查 belief
    _execute_actions_v14(eng_sim, res.interleaving, plan, vps_per_scan, scans=scans)
    return gamble_count[0]
```

效果(相对修复前):
- 步数 51.20 → **49.40** (-1.80)
- gambling 不变 12.8%
- 时间 6 min → **3 min** (2× 加速,因为 DP 更快收敛)

---

## 8 项核心创新

1. **Soft penalty** (v8): `α × (n_possible − 1)` per gambling — 替代 hard belief gate,允许理性赌博。
2. **Walk-reveal** (v3b): 车走路时触发 FOV 顺路识别 entity。
3. **Multi-viewpoint** (v4): 每 entity 4 个 (vp, q4) 候选,DP 选最便宜。
4. **Proactive rotation** (v4): 静止旋转 1 步换识别。
5. **Dynamic forced pair** (v6): 推进中拓扑变化检测新 forced。
6. **Pair-implicit scan** (v9): box pair-consumed 时双方 ID 隐式揭示。
7. **Bijective propagation** (v12): N-1 box class 已知 ⇒ N target num 全已知。
8. **Relaxed goal** (v14): DP 不强制 scan_set 全满,跳过 redundant scan。最大单点改进。

---

## K-best god plans (Plan A)

`MultiBoxSolver.solve_kbest(k=5)` 枚举多样化 god plan:

1. **Base solve**: 标准 A* 找最优 plan
2. **First-push 变体**: 对每个能合法 first-push 的箱子取最便宜方向,从 state-after-first-push 继续 A*
3. **Push-swap 变体**: 对每个 plan 找相邻 commutative 推对,交换生成新 plan
4. **去重 + cost 排序**: top-K 返回

v18 wrapper 在所有 (plan_i, α_j) 组合上跑 v14 DP,multiplicative score 选最优。

加速优化:
- `_heuristic` / `_is_deadlock` memoization (跨 sub-A* 共享)
- 每箱只试 1 个 first-push 方向 (vs 4)
- α 集瘦身 (11 → 6)

---

## Consumption-side gambling 定义(关键)

旧版只检查 box class 是否已识别 — 错误。正确定义:

**Consumption (推到 target 消除) 时刻,若 (box.class_id, target.num_id) 中任一在 agent belief 未知 → gambling。**

Agent belief 包含:
- `seen_box_ids` / `seen_target_ids` (FOV 实测)
- `forced_pair_classes` / `forced_pair_nums` (几何推理)
- `consumed` (已 consume 的 class/num 通过排除已知)
- Bijective propagation (N-1 已知 → 全已知)

---

## 可视化

```bash
# 单 map static panel (含 FP 黄边框 + GAMBLE 红边框)
python -m experiments.min_steps.visualize --planner oracle_v18 \
  --out-dir /tmp/viz --maps phase6:phase6_11.txt:1

# 交互式 HTML (滑块 + 播放 + v6 vs v18 并排)
python -m experiments.min_steps.viz_compare_html --out /tmp/compare.html \
  --maps phase4:phase4_02.txt:999 phase6:phase6_11.txt:1

# GIF
python -m experiments.min_steps.viz_gif --planner oracle_v18 --out-dir /tmp/gif
```

**Viz 颜色约定:**
- 🟢 **绿框**: FOV 实测识别 (具体 class id 已知)
- 🟡 **黄框 "FP"**: 几何 forced-pair 配对已知,具体 id 未知
- 🔴 **红虚线 "B?"** / 🔵 **蓝虚线 "T?"**: 真未知
- **整框红边**: consumption 是 gambling
- **整框绿边**: consumption 是 informed

---

## 用法 (CLI)

```bash
# 30-map 基准
python -m experiments.min_steps._bench_new_v18

# v1 vs v6 vs v18 三方对比
python -m experiments.min_steps._bench_v1_v18

# 走线浪费 audit (revisit ratio + 180° 回头对)
python -m experiments.min_steps._route_audit
```

详细数据见 [`RESULTS.md`](./RESULTS.md)。

---

## 参数调优旋钮

| 参数 | 默认 | 含义 |
|---|---|---|
| `p_per_gamble` | 0.85 | 单次盲推的假设成功率;**0.5** 严控 gambling,**0.95** 优先步数 |
| `k_plans` | 5 | K-best god plans 数量 |
| `alphas` | (0, 0.7, 2, 5, 25, 100) | v14 DP 内 gambling penalty 系数集 |
| `score_mode` | 'multiplicative' | 也支持 'additive' (legacy) |
| `gambling_weight` | 20.0 | additive 模式下的 λ |
| `trust_walk_reveal` | True | v14 DP 是否信任 walk-reveal 模型 |

例:
```python
# 严控 gambling (产生数据集时用)
teach(eng, p_per_gamble=0.5)

# 步数优先 (评测时用)
teach(eng, p_per_gamble=0.95)
```

---

## 已知局限

- **+0.53 步壁垒**:距 v6 物理下界(god view 无 scan)还有约半步,是 belief-aware 的内在税。继续突破需联合 search (push + scan 同空间 DP),状态空间 ~10^7。
- **NN 接入断层**:目前 `experiments/sage_pr/build_dataset_v3-v6.py` 用独立 god-mode 流程,**没用 v18 teacher**。要落地到学生模型需改 build_dataset 调用 `teacher.teach`。
- **运行时长**:K-best 30-map 3 min,适合离线 dataset 生成;实时部署需进一步压缩。

---

## 文件清单

```
teacher.py                  生产入口
planner_oracle_v18.py       顶层 K-best wrapper
planner_oracle_v14.py       belief-aware DP 核心
planner_oracle_v6.py        baseline (no penalty)
planner_oracle.py           共享工具: _god_plan/s, _simulate_god_and_record, _walk_to_executor
planner_oracle_v[2-13,15-17].py  历史迭代版本
planner_v[1-7].py           更早期 prototype
planner_best.py             set_best_context 工具
harness.py                  step-counting wrapper
visualize.py                单 map static panel renderer
viz_compare_html.py         v6 vs v18 并排交互 HTML
viz_interactive.py          单 planner 交互 HTML
viz_gif.py                  GIF 输出
_bench_*.py                 benchmarks
_route_audit.py             走线浪费分析
_audit_gambling.py          gambling 计数审计
_analyze_dynamic_forced.py  dynamic forced pair 探测
RESULTS.md                  全量基准数据 + 迭代历史
```
