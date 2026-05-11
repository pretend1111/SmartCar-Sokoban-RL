# preview_trajectory.py 使用手册

通用 trajectory 预览工具, 把 "录制 trajectory" 跟 "渲染回放" 解耦。
新增老师 / 模型只需要在 `RECORDERS` dict 里加一行函数, 不需要改 UI 代码。

文件: `experiments/sage_pr/preview_trajectory.py`

---

## 1. 快速开始

```bash
# 看单个老师 (默认 v1_v3) 在某张图上的解法
D:/anaconda3/envs/rl/python.exe experiments/sage_pr/preview_trajectory.py \
  --recorders v1_v3 --map assets/maps/phase5/phase5_0001.txt
```

打开 pygame 窗口, 自动播放 trajectory, 标题栏显示当前 step。

---

## 2. CLI 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--recorders` | `v1_v3` | 逗号分隔的 recorder 名 (见 §4). 多个 = 并排对比. e.g. `v1_orig,v1_v3` |
| `--map` | None | 单图路径. 跟 `--fails-list` 二选一. 同 phase 文件夹的其他图可用 N/P 翻 |
| `--fails-list` | None | JSON 清单批量看, 每项需含 `map` (+ `seed`). e.g. `runs/sage_pr/v5_v3_fails.json` |
| `--phase-only` | None | 配合 `--fails-list`, 只看某 phase |
| `--seed` | None | 强制 seed (覆盖 verified_seeds_manifest 的默认) |
| `--time-limit` | 30.0 | MultiBoxSolver 单次时限 (秒) |
| `--start` | 0 | 起始 idx (从清单/文件夹中) |

---

## 3. 快捷键

| 键 | 动作 |
|----|------|
| `SPACE` | 播放 / 暂停 |
| `R` | 重置当前图 (从 step 0 开始) |
| `N` / `PageDown` | 下一张图 |
| `P` / `PageUp` | 上一张图 |
| `←` | 慢一点 (step_delay +0.02s) |
| `→` | 快一点 (step_delay -0.02s) |
| `ESC` | 退出 |

窗口下方 info 行显示:
```
step k/N | explore=X solve=Y won=YES/no | forced=[(i,j), ...]
```
- `explore` = 探索阶段 low-level 步数
- `solve` = 推箱阶段 low-level 步数
- `won` = `YES` (通关) / `no` (没通) / `explore_incomplete` 等错误
- `forced` (仅 v1_v3 显示) = 拓扑预配对的 (box_idx, target_idx) 列表, 空 = 此图无 wall-locked 自动配对

---

## 4. Recorder 当前清单

每个 recorder 是一个函数 `(map_path, seed) → (action_log, info)`:
- `action_log: List[int]` — 低层离散动作序列 (engine action 0-14)
- `info: dict` — 元信息 (won, n_explore, n_solve, 任意补充字段)

### 4.1 内置 recorders

| 名字 | 说明 | 用途 |
|------|------|------|
| `v1_orig` | 原始 V1: `plan_exploration` + `MultiBoxSolver` (无补丁) | 基线对照 |
| `v1_v2` | V1 + 推开障碍补丁 (`plan_exploration_v2`) | 中间版本 |
| **`v1_v3`** | **当前正式版**: V1 + `plan_exploration_v3` (推开障碍 + 拓扑预配对) | 数据集生成实际用的老师 |

### 4.2 增加新 recorder (维护说明)

在 `preview_trajectory.py` 加:

```python
def _recorder_<name>(map_path: str, seed: int):
    random.seed(seed)
    eng = GameEngine(); eng.reset(map_path)
    log: List[int] = []
    info = {"won": False}
    # ... 录制逻辑, 每个 engine.discrete_step(a) 调用要 log.append(a) ...
    info["won"] = eng.get_state().won
    return log, info

RECORDERS["<name>"] = _recorder_<name>
```

UI 部分 (rendering + 切图 + 控制) 不用动, 自动支持。

---

## 5. 常用场景

### 5.1 人工筛查全 phase 5 失败图
```bash
D:/anaconda3/envs/rl/python.exe experiments/sage_pr/preview_trajectory.py \
  --recorders v1_v3 --fails-list runs/sage_pr/v5_v3_fails.json --phase-only 5
```
按 N/P 翻 33 张图, 看为啥老师解不出。

### 5.2 对比 v3 vs v2 (看拓扑配对补丁效果)
```bash
D:/anaconda3/envs/rl/python.exe experiments/sage_pr/preview_trajectory.py \
  --recorders v1_v2,v1_v3 --map assets/maps/phase5/phase5_0001.txt
```
左 v2 (无拓扑配对), 右 v3 (有)。看 forced=[...] 是否非空。

### 5.3 测试某具体地图 / seed
```bash
D:/anaconda3/envs/rl/python.exe experiments/sage_pr/preview_trajectory.py \
  --recorders v1_v3 --map assets/maps/phase4/phase4_11.txt --seed 137
```

---

## 6. 输出 / 副作用

- 不写文件, 不修改 dataset
- console 打印每张图加载时的 recorder summary
- 仅 pygame 窗口显示

---

## 7. 已知限制

- 推理录制需要等到 trajectory 跑完才能播放 (不流式)
- 单 GPU/CPU 串行运行多个 recorder (双 panel 慢约 2x)
- v3 recorder 的 forced pair 信息需要在 record 时单独调 `find_forced_pairs`, 跟 trajectory 录制独立
- 切图时不保存上一张的播放进度, 切回来要从头

---

## 8. 维护历史

| 日期 | 改动 |
|------|------|
| 2026-05-11 | 初版, 含 `v1_orig` / `v1_v2` / `v1_v3` 三个 recorder |
