# Oracle 进化结果

## 重要修复 (2026-05-13): consumption-side gambling + multiplicative scoring

旧 v18 的 gambling 定义只检查 box.class 是否已识别 — 把"扫一眼箱子就推到未知 target"也算 informed。
实际"非赌博"要求 (box.class_id, target.num_id) **双侧**都已识别。修复后:

- `planner_oracle_v14.py`: penalty 触发时机从 first_push → last_push (consumption); K_box/K_tgt 分开计算 (不再 merge); walk-reveal 从静态 q4 改为动态 (per-step direction).
- `planner_oracle_v18.py`: gambling_count 用 consumption-side 双侧检查; 评分从加法 (steps + λ·k) 改为乘法 (log(1+1/steps) × p^k), 单次赌博"成功率" p_per_gamble=0.5.

### 30-map (consumption-side metric)
| Planner | step | gambling | box-blind | tgt-blind |
|---|---|---|---|---|
| v6 (no penalty) | 48.87 | 20/78 (25.6%) | 5 | 18 |
| v18 (mult, broken vp) | 51.43 | 9/78 (11.5%) | 5 | 9 |
| **v18 (mult + walk-aware)** | **49.27** | **17/78 (21.8%)** | 12 | 16 |

注 (2026-05-13 二轮): 第一版 multiplicative 那个 11.5% 是 misleading — 由于 `_bfs_path_cells_optimal`
错误地允许 path 终点在 obstacle 上, 让 DP 把"走进 box 同格"算成 scan vp。执行时这变成 chain-push,
plan 失败但 fallback BFS 把车走到附近, walk-reveal 顺路把 box 识别了。这种"幸运的失败"夸大了识别率。

修了 bug 后 (`_bfs_path_cells_optimal` end 不再允许 obstacle), 21.8% 是真实数字 — 比 v6 25.6% 降 15%。

## 三方对比: v1 (explore-first) vs v6 (no penalty) vs v18 K-best (current)

30 verified-seed map:

| Planner | Won | 步数 | Gambling | 备注 |
|---|---|---|---|---|
| **v1_explore** (先观察再推) | 28/30 | **69.50** | 0% | 早期 baseline. 探索完才动手, 2 张图卡住. |
| **v6** (no penalty, oracle) | 30/30 | **48.87** | 25.6% | 大量盲推. |
| v18 single-plan (p=0.5) | 30/30 | 52.33 | 9.0% | 单 god plan + multi-α |
| **v18 K-best (p=0.5)** | 30/30 | 54.57 | **2.6%** | **当前默认**: K=5 god plans |
| v18 K-best (p=0.7) | 30/30 | 53.67 | 3.8% | 平衡 |
| v18 K-best (p=0.85) | 30/30 | **51.07** | 11.5% | 接近 v6 步数 |

v18 K-best p=0.5: 相对 v1 节省 22% step, 相对 v6 gambling 砍 90%.
p_per_gamble 是 step↔gambling 旋钮 — 用户可调.

## 第三轮深度优化 (2026-05-13): walk-aware paths + 解耦 DP 信任

修复了四处:

1. **`_bfs_path_cells_optimal` (新, `planner_oracle_v4.py`)** — 分层 DP, 在等长最短路径中
   按 (max walk-reveal bits, end-orient match) 字典序优选. scan_set / prefer_end_q4 状态依赖.
   把"走到 obstacle"当 end 的旧 bug 修了 (`nc != end` 例外删除).

2. **v14 DP 调用 walk_path_optimal** — push/scan 都走最优路径. Scan walk 还偏好末步方向 == vp_q4 省 rotate.

3. **`_execute_actions_v14` 记录具体 path** — `parent_act` 存 `("push", k, path)` /
   `("scan", j, vp_idx, path)`. Executor 按 planner 选的路径走 (`_walk_path_executor`),
   不再让 executor 用自己的 BFS 而与 planner 模型脱节. 失败自动 fallback BFS.

4. **`trust_walk_reveal=False` (默认)** — 关键修复. DP 的 `trigger_map` 是基于 STATE0 静态构建,
   推完 box 后位置错位, 后续步骤 walk-reveal 模型失真 → DP 高估识别 → 少做 scan → 实际 gambling 多.
   关掉 DP 信任 walk-reveal 后, DP 完全靠 explicit scan 决策, 但 walker 仍走 reveal-rich 路径,
   engine 执行时**仍然顺路免费识别** — 两全。

### 30-map (consumption-side metric, multiplicative p=0.5)

| 版本 | 步数 | Gambling | Box盲 | Tgt盲 | 备注 |
|---|---|---|---|---|---|
| v6 (no penalty) | 48.87 | 25.6% | 5 | 18 | baseline |
| v18 第二轮 (broken vp) | 51.43 | 11.5% | 5 | 9 | misleading — vp 走 obstacle 但 fallback 救活 |
| v18 第三轮 (trust=True) | 49.27 | 21.8% | 12 | 16 | walk-aware path 选优, 但 DP 高估 reveal |
| **v18 第三轮 (trust=False)** | **52.47** | **9.0%** | **3** | **7** | **当前最佳** |

最终结果: **gambling 从 25.6% 降到 9.0% (相对 65% reduction), 步数 +7.4%**.

phase6_04 仍 2/3 gambling — topology 限制,scan vp 物理上不可达,这种 map 必赌。

## 第四轮深度优化 (2026-05-13 cont): runtime scan-skip + dynamic trigger maps

新增:

1. **`_simulate_god_and_record` 扩展为 6-tuple** — 现在每个 snapshot 还存
   `box_pos_by_idx` 和 `tgt_pos_by_idx` (per-step entity 位置). 用于动态 trigger 计算.

2. **`trigger_maps_per_step[k]`** — 每步 k 按当前 entity 位置重建 trigger map.
   只在 walk-reveal 时使用 (即 reveals_along_walk 用 step_k=k 的 dynamic map).
   单元格 reveal (scan vp / init / after-push) 仍用 static (因为 vps_per_scan 也是 static, 一致).

3. **Runtime 冗余 scan 跳过** — `_execute_actions_v14` 在执行 scan 前检查
   `eng.state.seen_box_ids` / `seen_target_ids`. 已识别的实体, scan + 它的 walk 整段跳过.
   关键: scan-skip 不会破坏后续 push 的 path-following — 当前位置与录制不符时,
   `_walk_path_executor` 自动 fallback 到 BFS, 仍能到达 push_pos.

收益 (30 map): 52.47 → 52.33 步 (-0.14), gambling 持平 9.0%.
单点改进有限因为 multiplicative 评分本就避免冗余 scan-heavy 方案了, 但作为"安全网"防止运行时 mismatch.

## 已尝试但未带来稳定收益

- **完全 dynamic walk-reveal (trust=True, dynamic map)** — DP 信任 walk-reveal 估计.
  理论上应能减步, 实际 gambling 上升到 26.9% (高于 v6) — 执行端 engine FOV 跟模型对不齐.
  保留代码 (`trust_walk_reveal` flag), 默认 False.

## 第五轮深度优化 (2026-05-13 cont.): K-best god plans

最关键的优化. v18 之前只用 MultiBoxSolver.solve() 返回的单个 god plan, 然后跑 multi-α
v14 DP 搜 belief-aware 最优. 现在改成枚举多个 first-push 多样化的 god plans, 在
(plan_i, α_j) 全部组合上选 multiplicative 最优.

实现:
- `MultiBoxSolver.solve_kbest(k=5, ...)` (`smartcar_sokoban/solver/multi_box_solver.py`):
  base solve + 对每个能合法 first-push 的箱子/方向, 强制 first move 后再 A*. 去重 + cost 排序.
- `_god_plans` helper in `planner_oracle.py`: 包装 solve_kbest.
- `planner_oracle_v18` 主循环改 `for plan in plans: for alpha in alphas`.

30-map 收益 (相对 single-plan v18 at p=0.5):
- gambling 9.0% → **2.6%** (砍掉 71%)
- step 52.33 → 54.57 (+4%)
- 全部 30 张图都赢

Pareto 改进的: 同 step 水平 gambling 都更低. p=0.85 下 step 反而比 single-plan 还少 (51.07).
默认 `p_per_gamble=0.5`, `k_plans=5`.

### 速度优化 (2026-05-13 cont.)

慢版 K-best 14 min / 30 map. 优化后 **3 min 11 s** (4.4× 提速):

1. **`_heuristic` / `_is_deadlock` memoization** (per-MultiBoxSolver instance):
   `_heuristic_cache` / `_deadlock_cache` 跨 solve_kbest 的子 A* 共享.
2. **每箱只试 1 个 first-push 方向** (最便宜 walk_cost): N 变体 vs N×4 变体, 多样性近乎无损.
3. **Alpha 集瘦身**: 11 个 → 6 个 (`{0, 0.7, 2, 5, 25, 100}`). DP 调用 55 → 30 次/图.

代价: 53.47 步 / 5.1% gambling (vs 慢版 54.57 / 2.6%). 接受。

## 第六轮: p_per_gamble Pareto 扫描 (Fast K-best)

p=0.5 → 0.7 → 0.85 → 0.95 sweep:

| p | 步数 | gambling | 评价 |
|---|---|---|---|
| 0.5 | 53.47 | 5.1% | gambling 严控 |
| 0.7 | 53.47 | 5.1% | 与 p=0.5 相同 (multiplicative 评分在低 p 区都强偏好 0-gambling) |
| **0.85** | **50.87** | **12.8%** | **Pareto 甜点 (新默认): 接近 v6 步数, gambling 减半** |
| 0.95 | 48.87 | 23.1% | 等于 v6 步数, gambling 只比 v6 略好 |
| v6 ref | 48.87 | 25.6% | baseline |

**默认改为 p=0.85**: 相对 v6 同步数水平 (+2 step, 4%) 把 gambling 砍半 (25.6 → 12.8%).
适合 step 优先场景 (如 RL 训练数据). gambling 严控场景仍可用 p=0.5.

## 第七轮: Plan A 攻坚 — belief-aware god plan (2026-05-13 cont.)

试图突破"v18 比 v6 多 2 步"的 gap. 用户问"决策是否最优", 答案是各层 DP 都 provably optimal under cost model,
但 god plan 是 class-blind 的, 不优化 belief-aware 执行 cost.

**尝试 A.1: belief-aware bonus 注入 MultiBoxSolver cost function**

在 `_enum_pushes` 给每个 push 加 bonus = walk 路径上 unique trigger 配置数, augmented walk_cost = max(0, walk - bw × bonus).
让 solver 主动选 walk-reveal 友好的推法。

30-map sweep bw ∈ {0, 2, 5, 10}: **全部 51.00 step / 12.8% gambling, 完全相同**.
原因: K-best 通过 first-push 约束枚举 N 个变体, bw 只改变 solver 内部 ranking, 不改变 v18 候选集合.
v18 通过 multiplicative score 已经从 K 候选中选了最优 plan, bonus 没贡献新候选.

代码保留 `trigger_map`/`belief_weight` 接口和 `_compute_belief_bonus` 助手, 默认 bw=0.

**尝试 A.2: 每箱 top-2 方向 first-push 枚举**

恢复每箱 2 个方向 (vs top-1) → K-best 候选 2× 多样性. 30-map: 51.20 step / 11.5% gambling.
相对 top-1 (51.00 / 12.8%): +0.20 step, -1.3pp gambling, +50% wall time.
multiplicative 评分 (p=0.85) 略优于 top-1, **保留为默认**.

**结论 — 当前架构的本质瓶颈**

- v14 DP 在 cost model 下 provably optimal
- K-best 多 plan + multi-alpha + multiplicative 已穷尽给定 god plan 候选
- +2 step gap vs v6 ≈ 避免 gambling 的内在成本 (每 ~5pp gambling 减少 = ~1 step)

要进一步突破需要其中之一:
1. **联合 search (push + scan + belief)** — 把 MultiBoxSolver 和 v14 合并成单一状态空间 DP. 状态空间 ~10^7, 需启发式剪枝. 大改造.
2. **执行端 walk-reveal 完美对齐 DP 预测** — dynamic trigger map 此前尝试 gambling 升到 26%, 怀疑 engine FOV 精度问题. 需细致 debug.
3. **god plan 的更彻底多样化** — 比如 push-swap permutation (相邻 commutative push 交换), 或 box-target assignment permutation. 复杂度高, 收益未知.

## 第八轮: 突破 +2 步壁垒 (2026-05-13 cont.)

之前误以为 +2 step gap 是"belief-aware 的内在税". 实际上, 问题在于:
- v14 DP 必须保守 (`trust_walk_reveal=False`), 否则 v18 wrapper 的 `count_gambling_consumption`
  (仅看 explicit scan) 会把"靠 walk-reveal 识别 entity"的 plan 误判为 gambling-heavy → 拒绝.
- 结果 DP 过度插入 explicit scan, 步数偏多.

**双管齐下修复:**

1. **`trust_walk_reveal=True`** (v14 DP 默认改回 True) — DP 信任 dynamic trigger map 的 walk-reveal,
   生成更短的 plan (用 walk-reveal 代替 explicit scan).
2. **`count_gambling_via_sim` (v18)** — wrapper 不再用 interleaving 的 explicit-scan-only 估算,
   而是在 fresh engine 上 *实际 simulate* plan, 用 engine 的 `seen_*_ids` 判定 gambling.
   实测 1 次 sim ≈ 50 engine step ≈ 几十 ms, 总开销可接受.

**实现:** `planner_oracle_v18.py:count_gambling_via_sim` — clone engine, monkey-patch discrete_step
counting consumption events, 运行 `_execute_actions_v14`.

**30-map 收益 (相对前最佳 50.87/12.8%):**

| 维度 | 之前最佳 | 突破后 |
|---|---|---|
| 步数 | 50.87 | **49.40** (-1.47) |
| 距 v6 物理下界 | +2.00 | **+0.53 (-77% gap)** |
| Gambling | 12.8% | 12.8% (持平) |
| 30-map wall time | 9 min | **3 min (3× 加速)** |

时间反而 3× 加速因为 trust=True 让 v14 DP 早点收敛 (用 walk-reveal 替代 explicit scan, 状态空间窄).

至此 v18 步数距物理下界 < 1 step, 已贴近理论极限.

## 30-map verified sample (seed=0)

Physical lower bound (v0 god + 0 scan, CHEATS): 48.00 step, 43.2% gambling.

| Variant | step | gambling | gap-to-lb | 评价 |
|---|---|---|---|---|
| v0_lb cheat | 48.00 | 50.6% | 0 | cheating reference |
| v6 (no penalty) | 48.87 | 31.2% | +0.87 | base oracle, no soft penalty |
| v8 α=0.4 (linear) | 49.03 | 10.4% | +1.03 | soft penalty + full_mask |
| v9 (pair info) | 49.00 | similar | +1.00 | + implicit pair scan |
| v10 α=0.4 (log) | 48.97 | 14.3% | +0.97 | log entropy penalty |
| v12 α=0.4 (bij) | 49.00 | 11.7% | +1.00 | + bijective propagation |
| v14 α=0.4 (relaxed) | 48.13 | 29.9% | +0.13 | **relaxed goal** — biggest jump |
| **v14 α=0.7** | **48.73** | **14.3%** | **+0.73** | sweet spot for v14 |
| **v18 λ=5** | **48.67** | **13.0%** | **+0.67** | **per-map α — current champion** |

## 90-map verified sample

| Variant | step | gambling | gap-to-lb |
|---|---|---|---|
| v0_lb cheat | 49.06 | 43.2% | 0 |
| v8 α=0.4 | 50.01 | 21.8% | +0.95 |
| v12 α=0.4 | 50.00 | 21.8% | +0.94 |
| v14 α=0.4 | 49.26 | 30.6% | +0.20 |
| **v14 α=0.7** | **49.68** | **20.5%** | **+0.62** |
| v14 α=1.0 | 49.76 | 20.1% | +0.70 |
| v14 α=2.0 | 50.01 | 17.5% | +0.95 |
| **v18 λ=2** | **49.47** | 22.3% | **+0.41** ← lowest step legit |
| **v18 λ=5** | **49.78** | **17.5%** | +0.72 ← lowest gambling at moderate step |
| v18 λ=10 | 49.78 | 17.5% | +0.72 (saturated) |

**Best legit-teacher gap to physical lb on 90 map: +0.41 step (v18 λ=2)** — within 0.83% of cheating reference.

## 关键架构创新

1. **soft penalty (v8)**: 替代 hard belief gate。penalty = α × (n-1) per gambling first-push。
2. **walk-reveal (v3b)**: 车走路时 FOV 顺路免费识别 entity。
3. **multi-viewpoint (v4)**: 每 entity 有 4 个 trigger 配置, DP 选最便宜。
4. **proactive rotation (v4)**: 静止旋转可解锁 reveal, cost 1 step。
5. **dynamic forced pair (v6)**: mid-game 拓扑变化时检测新 forced。
6. **pair-implicit scan (v9)**: box pair-consumed 时双方 ID 隐式揭示。
7. **bijective propagation (v12)**: 已知 N-1 box class → N target num 全已知。
8. **relaxed goal (v14)**: DP 不强制 scan_set 全满, 跳过 redundant scan。最大单点改善。
9. **per-map α (v18)**: 每图试多个 α, 选最低 joint score 的 plan。

## 当前最佳 teacher

`teacher.teach(eng, variant='v18', gambling_weight=5.0)` — 30-map: 48.67/13%, 90-map: TBD。

Step gap to physical lb: ~0.6-0.7 step/map。
Gambling rate: 13-17%, 全部"rational gambles"(α 抑制纯赌博)。
Win rate: 100% on tested samples。

## 不能再优化的部分 (理论 belief-aware lb)

剩余 ~0.6 step 主要是:
- 必要的 explicit scan (无 walk-reveal/forced-pair 覆盖)
- scan 的 vp 不在 god 推箱 walk 路径上 → 必绕路 ~2-3 step
- 这是 belief-aware 的"信息税"

进一步压缩需要:
- 完整 belief-MDP A* (push 顺序也由 DP 选, 极大状态空间)
- 多 god plan 候选 (v5/v17 大多 infeasible)
- 改 engine FOV 规则 (违反物理)
