"""可视化 explore_incomplete 的图.

读取 runs/sage_pr/explore_failed_maps.json (build_dataset_v5 跑完留下的清单),
按顺序循环展示每张图. 针对每张图:
  1. 显示初始状态
  2. 跑 plan_exploration, 把 explore 走过的步骤回放
  3. 标记: 哪些 box / target 仍未识别 (红色叠加)
  4. 用户按 K=keep / D=delete 决策, 退出时把 keep 列表写到 JSON

快捷键:
  N / →  下一张        P / ←  上一张
  SPACE  播放探索        R    重跑探索
  K      标记 keep      D     标记 delete
  S      保存决策 + 退出 ESC  不保存退出
  Tab    切换 simple/full 渲染
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import random
import sys
import time
from contextlib import redirect_stdout

import pygame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine, GameState
from smartcar_sokoban.paths import PROJECT_ROOT
from smartcar_sokoban.renderer import Renderer
from smartcar_sokoban.solver.explorer import plan_exploration, exploration_complete
from smartcar_sokoban.solver.pathfinder import pos_to_grid


def lerp_state(prev: GameState, cur: GameState, t: float) -> GameState:
    s = copy.deepcopy(cur)
    s.car_x = prev.car_x + (cur.car_x - prev.car_x) * t
    s.car_y = prev.car_y + (cur.car_y - prev.car_y) * t
    ad = cur.car_angle - prev.car_angle
    ad = math.atan2(math.sin(ad), math.cos(ad))
    s.car_angle = prev.car_angle + ad * t
    return s


def diagnose_unidentified(state):
    """返回 (未识别 box 索引列表, 未识别 target 索引列表)."""
    unid_box = [i for i in range(len(state.boxes)) if i not in state.seen_box_ids]
    unid_tgt = [i for i in range(len(state.targets)) if i not in state.seen_target_ids]
    return unid_box, unid_tgt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str,
                        default='runs/sage_pr/explore_failed_maps.json',
                        help='失败图清单 JSON')
    parser.add_argument('--out', type=str,
                        default='runs/sage_pr/explore_decisions.json',
                        help='保存 keep/delete 决策的 JSON')
    parser.add_argument('--start', type=int, default=0,
                        help='起始 idx')
    parser.add_argument('--phase-only', type=int, default=None,
                        help='只看某个 phase')
    args = parser.parse_args()

    list_path = args.list if os.path.isabs(args.list) else os.path.join(ROOT, args.list)
    with open(list_path) as f:
        all_items = json.load(f)
    if args.phase_only is not None:
        all_items = [x for x in all_items if x['phase'] == args.phase_only]
    if not all_items:
        print('no items')
        return

    decisions: dict = {}
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    if os.path.exists(out_path):
        with open(out_path) as f:
            decisions = json.load(f)

    cfg = GameConfig()
    cfg.render_mode = "simple"
    cfg.control_mode = "discrete"
    engine = GameEngine(cfg, str(PROJECT_ROOT))
    renderer = Renderer(cfg, str(PROJECT_ROOT))
    renderer.init()
    clock = pygame.time.Clock()

    idx = max(0, min(args.start, len(all_items) - 1))
    actions: list = []
    act_idx = 0
    playing = False
    animating = False
    anim_progress = 0.0
    prev_state = None
    state = None
    explore_complete_flag = False
    unid_box: list = []
    unid_tgt: list = []
    step_delay = 0.05

    def load_current():
        nonlocal actions, act_idx, prev_state, state, playing, animating
        nonlocal anim_progress, explore_complete_flag, unid_box, unid_tgt
        item = all_items[idx]
        random.seed(item['seed'])
        engine.reset(item['map'])
        with redirect_stdout(io.StringIO()):
            actions = list(plan_exploration(engine))
        # 重置用于回放 (plan_exploration 会真的走车, 我们要 visualize 过程)
        random.seed(item['seed'])
        engine.reset(item['map'])
        state = engine.get_state()
        prev_state = None
        act_idx = 0
        playing = True   # 自动播放
        animating = False
        anim_progress = 0.0
        explore_complete_flag = False
        unid_box, unid_tgt = diagnose_unidentified(state)
        key = f"{item['map']}|{item['seed']}"
        prev_dec = decisions.get(key, '?')
        print(f"\n[{idx+1}/{len(all_items)}] phase{item['phase']} {os.path.basename(item['map'])} "
              f"seed={item['seed']} | explore_actions={len(actions)} | dec={prev_dec}")

    load_current()
    running = True
    while running:
        dt = clock.tick(cfg.fps) / 1000.0
        dt = min(dt, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_n, pygame.K_RIGHT):
                    idx = (idx + 1) % len(all_items); load_current()
                elif event.key in (pygame.K_p, pygame.K_LEFT):
                    idx = (idx - 1) % len(all_items); load_current()
                elif event.key == pygame.K_r:
                    load_current()
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_k:
                    item = all_items[idx]
                    decisions[f"{item['map']}|{item['seed']}"] = 'keep'
                    print(f"  → KEEP")
                    idx = (idx + 1) % len(all_items); load_current()
                elif event.key == pygame.K_d:
                    item = all_items[idx]
                    decisions[f"{item['map']}|{item['seed']}"] = 'delete'
                    print(f"  → DELETE")
                    idx = (idx + 1) % len(all_items); load_current()
                elif event.key == pygame.K_s:
                    with open(out_path, 'w') as f:
                        json.dump(decisions, f, indent=2)
                    keeps = sum(1 for v in decisions.values() if v == 'keep')
                    dels = sum(1 for v in decisions.values() if v == 'delete')
                    print(f"\n📝 saved {len(decisions)} decisions ({keeps} keep, {dels} delete) → {out_path}")
                    running = False
                elif event.key == pygame.K_TAB:
                    m = "simple" if cfg.render_mode == "full" else "full"
                    renderer.switch_mode(m)

        # 回放 explore actions
        now = time.perf_counter()
        if playing and not animating and act_idx < len(actions):
            if now - getattr(load_current, '_last_step', [0])[0] >= step_delay:
                prev_state = copy.deepcopy(state)
                state = engine.discrete_step(actions[act_idx])
                act_idx += 1
                animating = True
                anim_progress = 0.0
                load_current._last_step = [now]

        if act_idx >= len(actions) and not animating and playing:
            playing = False
            unid_box, unid_tgt = diagnose_unidentified(state)
            explore_complete_flag = exploration_complete(state)
            print(f"  ↳ explore done. complete={explore_complete_flag} "
                  f"unid_box={unid_box} unid_target={unid_tgt}")

        # 动画插值
        if animating and prev_state is not None:
            anim_progress += cfg.discrete_anim_speed * dt
            if anim_progress >= 1.0:
                anim_progress = 1.0; animating = False
            render_state = lerp_state(prev_state, state, anim_progress)
        else:
            render_state = state

        if render_state:
            renderer.render(render_state, engine.get_fov_rays())

        item = all_items[idx]
        key = f"{item['map']}|{item['seed']}"
        dec = decisions.get(key, '?')
        pygame.display.set_caption(
            f"explore-failed | [{idx+1}/{len(all_items)}] "
            f"phase{item['phase']} {os.path.basename(item['map'])} "
            f"seed={item['seed']} | step {act_idx}/{len(actions)} "
            f"unid_box={len(unid_box)} unid_tgt={len(unid_tgt)} | dec={dec}"
        )

    if decisions:
        with open(out_path, 'w') as f:
            json.dump(decisions, f, indent=2)
        print(f"\n📝 final save: {len(decisions)} decisions → {out_path}")

    renderer.close()


if __name__ == "__main__":
    main()
