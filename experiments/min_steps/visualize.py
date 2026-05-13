"""推理可视化 — 给定 (map, seed, planner), 生成 step-by-step 轨迹图.

用 matplotlib 画:
  - 网格 (墙=深灰, 可走=白)
  - 箱子 (彩色方框, 标 class_id)
  - 目标 (彩色圆环, 标 num_id)
  - 炸弹 (橙色 X)
  - 车 (蓝色三角, 朝向)
  - 已识别 entity (绿色边框)
  - 当前动作 tag 显示在标题
  - 累积轨迹 (淡红线)

输出: 单图多子图 panel 或 PNG 序列 → 拼成 PDF.
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import random
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import pos_to_grid


# ── 单帧绘制 ────────────────────────────────────────────

def _draw_frame(ax, state, walls, trail: List[Tuple[float, float]],
                title: str, init_seen_box: set = None, init_seen_tgt: set = None,
                fp_box_ids: set = None, fp_target_ids: set = None):
    """渲染单帧.
       fp_box_ids / fp_target_ids: 几何 forced-pair 已配对的 entity index 集合.
       这些 entity 用黄边框 + "FP" 表示"配对已知但 class id 未知" (vs 绿框 = FOV 识别).
    """
    fp_box_ids = fp_box_ids or set()
    fp_target_ids = fp_target_ids or set()
    rows = len(walls); cols = len(walls[0])
    # 墙背景
    grid = np.array(walls)
    ax.imshow(grid, cmap='Greys', vmin=0, vmax=1.6, extent=(0, cols, rows, 0),
              aspect='equal', interpolation='nearest')
    # 网格线
    for r in range(rows+1):
        ax.axhline(r, color='lightgray', lw=0.3)
    for c in range(cols+1):
        ax.axvline(c, color='lightgray', lw=0.3)

    # 目标 (圆环 — 绿色实线=FOV识别; 黄色实线=forced-pair配对; 蓝色虚线=未知)
    for i, t in enumerate(state.targets):
        col, row = pos_to_grid(t.x, t.y)
        cx, cy = col + 0.5, row + 0.5
        seen_fov = i in state.seen_target_ids
        is_fp = (not seen_fov) and (i in fp_target_ids)
        if seen_fov:
            edge = 'limegreen'; ls = '-'; label = str(t.num_id); lbl_col = 'royalblue'
        elif is_fp:
            edge = 'gold'; ls = '-'; label = 'FP'; lbl_col = 'darkorange'
        else:
            edge = 'royalblue'; ls = '--'; label = 'T?'; lbl_col = 'royalblue'
        circ = mpatches.Circle((cx, cy), 0.42, fill=True,
                                facecolor='lightyellow', alpha=0.75,
                                edgecolor=edge, linewidth=2.5, linestyle=ls)
        ax.add_patch(circ)
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=11, color=lbl_col, fontweight='bold')

    # 箱子 (方框 — 绿=FOV识别; 黄=forced-pair配对; 红虚线=未知)
    for i, b in enumerate(state.boxes):
        col, row = pos_to_grid(b.x, b.y)
        seen_fov = i in state.seen_box_ids
        is_fp = (not seen_fov) and (i in fp_box_ids)
        color_map = {0: '#ff9999', 1: '#99ff99', 2: '#9999ff',
                     3: '#ffff99', 4: '#ff99ff', 5: '#99ffff'}
        if seen_fov:
            edge = 'limegreen'; ls = '-'
            face = color_map.get(b.class_id, '#cccccc'); label = str(b.class_id); lbl_col = 'crimson'
        elif is_fp:
            edge = 'gold'; ls = '-'
            face = '#fff3cd'; label = 'FP'; lbl_col = 'darkorange'
        else:
            edge = 'crimson'; ls = '--'
            face = '#ffe0e0'; label = 'B?'; lbl_col = 'crimson'
        rect = mpatches.Rectangle((col+0.1, row+0.1), 0.8, 0.8,
                                    facecolor=face, edgecolor=edge,
                                    linewidth=2.5, linestyle=ls)
        ax.add_patch(rect)
        ax.text(col+0.5, row+0.5, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color=lbl_col)

    # 炸弹
    for bm in state.bombs:
        col, row = pos_to_grid(bm.x, bm.y)
        ax.text(col+0.5, row+0.5, '×', ha='center', va='center',
                fontsize=18, color='orange', fontweight='bold')

    # 轨迹
    if len(trail) >= 2:
        xs = [p[0] for p in trail]
        ys = [p[1] for p in trail]
        ax.plot(xs, ys, color='red', alpha=0.4, linewidth=1.5, zorder=2)

    # 车 (三角朝当前 angle)
    cx, cy = state.car_x, state.car_y
    ca = state.car_angle
    dx, dy = math.cos(ca), math.sin(ca)
    pts = np.array([
        [cx + 0.32*dx, cy + 0.32*dy],
        [cx - 0.2*dx + 0.18*dy, cy - 0.2*dy - 0.18*dx],
        [cx - 0.2*dx - 0.18*dy, cy - 0.2*dy + 0.18*dx],
    ])
    tri = mpatches.Polygon(pts, facecolor='steelblue', edgecolor='navy',
                            linewidth=1.5, zorder=3)
    ax.add_patch(tri)

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


# ── 单 map 跑 planner 收集帧 ──────────────────────────

def run_and_collect_frames(map_path: str, seed: int, planner_fn, planner_name: str):
    """跑 planner, 每步收集 state snapshot + tag + 标记 consumption-push gambling.

    赌博定义 (consumption 时刻):
      推 box 落到 target 并消除时, 若 (box.class_id, target.num_id) 中任一在 agent
      belief 里未知 -> gambling。两端都已识别才算 informed push。
    """
    from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs
    random.seed(seed)
    eng = GameEngine()
    eng.reset(map_path)
    eng._step_tag = "init"

    state0 = eng.get_state()
    N = len(state0.boxes)
    N_t = len(state0.targets)
    box_universe = {b.class_id for b in state0.boxes}
    tgt_universe = {t.num_id for t in state0.targets}
    # running forced pair tracking — 推进过程中新出现的几何 forced 也累计进 agent 信念
    running_fp_box_ids = set()
    running_fp_tgt_ids = set()
    running_fc_box = set()  # 已 forced 的 class 集合
    running_fc_tgt = set()
    # 初始一次
    init_forced = find_forced_pairs(state0)
    for bi, ti in init_forced:
        if bi < N:
            running_fp_box_ids.add(bi)
            running_fc_box.add(state0.boxes[bi].class_id)
        if ti < N_t:
            running_fp_tgt_ids.add(ti)
            running_fc_tgt.add(state0.targets[ti].num_id)

    gamble_events = []
    frames = []
    trail = []
    step_no = [0]
    orig = eng.discrete_step

    def update_running_forced(s):
        """重算当前 state 的 forced pair, 把新出现的加入 running set."""
        try:
            new_forced = find_forced_pairs(s)
        except Exception:
            return
        for bi, ti in new_forced:
            if bi < len(s.boxes) and bi not in running_fp_box_ids:
                running_fp_box_ids.add(bi)
                running_fc_box.add(s.boxes[bi].class_id)
            if ti < len(s.targets) and ti not in running_fp_tgt_ids:
                running_fp_tgt_ids.add(ti)
                running_fc_tgt.add(s.targets[ti].num_id)

    def wrapped(a):
        step_no[0] += 1
        tag = getattr(eng, '_step_tag', '?')
        s_b = eng.get_state()
        # 推进过程中更新 forced (topology 可能变化)
        update_running_forced(s_b)
        # box 侧
        seen_box = {s_b.boxes[i].class_id for i in s_b.seen_box_ids}
        consumed_box = box_universe - {b.class_id for b in s_b.boxes}
        explicit_box = seen_box | set(running_fc_box) | consumed_box
        eff_box = box_universe if len(explicit_box) >= N - 1 else explicit_box
        # target 侧
        seen_tgt = {s_b.targets[i].num_id for i in s_b.seen_target_ids}
        consumed_tgt = tgt_universe - {t.num_id for t in s_b.targets}
        explicit_tgt = seen_tgt | set(running_fc_tgt) | consumed_tgt
        eff_tgt = tgt_universe if len(explicit_tgt) >= N_t - 1 else explicit_tgt
        classes_before = {b.class_id for b in s_b.boxes}

        result = orig(a)

        s_a = eng.get_state()
        classes_after = {b.class_id for b in s_a.boxes}
        just_consumed = classes_before - classes_after
        if tag == 'push' and just_consumed:
            for cnum in just_consumed:
                box_known = cnum in eff_box
                tgt_known = cnum in eff_tgt
                is_gamble = not (box_known and tgt_known)
                gamble_events.append((step_no[0], cnum, is_gamble, box_known, tgt_known))

        s = eng.get_state()
        import copy as _cp
        snap_state = _cp.copy(s)
        snap_state.boxes = [_cp.copy(b) for b in s.boxes]
        snap_state.targets = [_cp.copy(t) for t in s.targets]
        snap_state.bombs = [_cp.copy(b) for b in s.bombs]
        snap_state.seen_box_ids = set(s.seen_box_ids)
        snap_state.seen_target_ids = set(s.seen_target_ids)
        snap_state.car_x = s.car_x; snap_state.car_y = s.car_y
        snap_state.car_angle = s.car_angle
        snap_state.grid = [row[:] for row in s.grid]
        # snapshot 当前 fp 集合 (新增字段)
        snap_state._fp_box_ids = set(running_fp_box_ids) - snap_state.seen_box_ids
        snap_state._fp_target_ids = set(running_fp_tgt_ids) - snap_state.seen_target_ids
        frames.append((snap_state, tag, step_no[0]))
        trail.append((s.car_x, s.car_y))
        return result

    eng.discrete_step = wrapped
    eng._step_tag = "init_snap"
    eng.discrete_step(6)
    try:
        planner_fn(eng)
    except Exception as e:
        print(f"  Exception during planner: {e}")

    # 把 gamble_events 挂到 frames list 上, 后续 render 用
    return frames, trail, eng.get_state().won, step_no[0], gamble_events


# ── 主输出 ────────────────────────────────────────────

def render_map(map_path: str, seed: int, planner_fn, planner_name: str,
                 out_path: str, show_every: Optional[int] = None,
                 max_frames: int = 12):
    print(f'  Rendering {map_path} seed={seed} via {planner_name}...')
    ret = run_and_collect_frames(map_path, seed, planner_fn, planner_name)
    if len(ret) == 5:
        frames, trail, won, total, gambles = ret
    else:
        frames, trail, won, total = ret; gambles = []
    n = len(frames)
    n_gamble = sum(1 for ev in gambles if ev[2])
    n_consume = len(gambles)
    print(f'    total {n} steps, won={won}, gambling: {n_gamble}/{n_consume} consumption-pushes')

    # 选 max_frames 个: 优先包含 consumption 事件 + tag 变化
    important = set()
    for ev in gambles:
        ev_step = ev[0]
        for i, (_, _, sn) in enumerate(frames):
            if sn == ev_step:
                important.add(i); break

    if n <= max_frames:
        indices = list(range(n))
    else:
        tag_changes = [0]
        prev_tag = frames[0][1]
        for i in range(1, n):
            if frames[i][1] != prev_tag:
                tag_changes.append(i)
                prev_tag = frames[i][1]
        if n-1 not in tag_changes:
            tag_changes.append(n-1)
        important_list = sorted(important | set(tag_changes))
        if len(important_list) > max_frames:
            step = len(important_list) // max_frames
            indices = important_list[::step][:max_frames-1] + [important_list[-1]]
        else:
            indices = important_list
    indices = sorted(set(indices))[:max_frames]
    sel = [frames[i] for i in indices]

    # 把 gamble event 按 step 索引: step -> (cnum, is_gamble, box_known, tgt_known)
    gamble_at_step = {ev[0]: ev[1:] for ev in gambles}

    cols = 3
    rows = (len(sel) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5.0, rows*4.0))
    axes = np.atleast_2d(axes).flatten()

    map_name = os.path.basename(map_path)
    gamble_str = f'{n_gamble}/{n_consume} gambling consumes' if n_consume else ''
    fig.suptitle(f'{map_name} seed={seed} | {planner_name} | {total} steps | won={won} | {gamble_str}',
                  fontsize=11, y=0.99)

    for ax_i, (state, tag, step_no) in enumerate(sel):
        idx_in_frames = indices[ax_i]
        partial_trail = trail[:idx_in_frames+1]
        ev = gamble_at_step.get(step_no)
        if ev is not None:
            cnum, is_gamble, box_known, tgt_known = ev
            box_tag = f'{cnum}' if box_known else '?'
            tgt_tag = f'{cnum}' if tgt_known else '?'
            kind = 'GAMBLE' if is_gamble else 'KNOWN'
            marker = f' [{kind} cls={box_tag} -> num={tgt_tag}]'
            title = f'step {step_no}/{total}  tag={tag}{marker}'
        else:
            title = f'step {step_no}/{total}  tag={tag}'
        fp_box = getattr(state, '_fp_box_ids', None)
        fp_tgt = getattr(state, '_fp_target_ids', None)
        _draw_frame(axes[ax_i], state, state.grid, partial_trail, title,
                     fp_box_ids=fp_box, fp_target_ids=fp_tgt)
        if ev is not None:
            color = 'red' if ev[1] else 'green'
            for spine in axes[ax_i].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

    # 空余 axes 隐藏
    for ax in axes[len(sel):]:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'    → {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--planner', default='oracle_v18',
                    choices=['v1_explore', 'oracle_v1', 'oracle_v3b',
                              'oracle_v4', 'oracle_v6', 'oracle_v14',
                              'oracle_v18'])
    p.add_argument('--out-dir', default='/tmp/viz')
    p.add_argument('--maps', nargs='+', default=None,
                    help='list of "phase:name:seed" or omit for default 4 maps')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    planner_map = {
        'v1_explore': lambda: __import__('experiments.min_steps.harness',
            fromlist=['planner_v1_explore_first']).planner_v1_explore_first,
        'oracle_v1': lambda: __import__('experiments.min_steps.planner_oracle',
            fromlist=['planner_oracle']).planner_oracle,
        'oracle_v3b': lambda: __import__('experiments.min_steps.planner_oracle_v3b',
            fromlist=['planner_oracle_v3b']).planner_oracle_v3b,
        'oracle_v4': lambda: __import__('experiments.min_steps.planner_oracle_v4',
            fromlist=['planner_oracle_v4']).planner_oracle_v4,
        'oracle_v6': lambda: __import__('experiments.min_steps.planner_oracle_v6',
            fromlist=['planner_oracle_v6']).planner_oracle_v6,
        'oracle_v14': lambda: __import__('experiments.min_steps.planner_oracle_v14',
            fromlist=['planner_oracle_v14']).planner_oracle_v14,
        'oracle_v18': lambda: __import__('experiments.min_steps.planner_oracle_v18',
            fromlist=['planner_oracle_v18']).planner_oracle_v18,
    }
    from experiments.min_steps.planner_best import set_best_context
    fn = planner_map[args.planner]()

    if args.maps:
        targets = []
        for spec in args.maps:
            ph, name, sd = spec.split(':')
            targets.append((f'assets/maps/{ph}/{name}', int(sd)))
    else:
        targets = [
            ('assets/maps/phase4/phase4_05.txt', 0),
            ('assets/maps/phase5/phase5_01.txt', 0),
            ('assets/maps/phase6/phase6_02.txt', 0),
            ('assets/maps/phase6/phase6_04.txt', 137),
        ]

    for mp, sd in targets:
        if not os.path.exists(mp):
            print(f'  SKIP {mp}: not found')
            continue
        set_best_context(mp, sd)
        out = os.path.join(args.out_dir,
                            f'{os.path.basename(mp).replace(".txt", "")}_seed{sd}_{args.planner}.png')
        render_map(mp, sd, fn, args.planner, out)


if __name__ == '__main__':
    main()
