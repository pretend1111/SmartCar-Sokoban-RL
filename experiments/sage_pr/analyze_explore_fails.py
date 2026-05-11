"""分类 475 个 explore_incomplete 失败的根本原因.

对每张失败图:
  1. 重跑 plan_exploration
  2. 找出仍未识别的 entity
  3. 对每个 unid entity, 分析为什么 find_observation_point 返回 None
     - A: 8-邻全是墙 (entity 在死胡同/墙角)
     - B: 8-邻有合法但被其他实体堵 (obstacle on 8-neighbor cell)
     - C: 8-邻合法但 LOS 被遮挡 (墙缝中)
     - D: 8-邻合法可达但车走不到 (BFS 路径全堵)
     - E: viewpoint 存在但车被某个 entity 锁死 (车自己被堵)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from smartcar_sokoban.solver.explorer import (
    plan_exploration, exploration_complete, has_line_of_sight,
)

DIRS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIRS_8 = [(1, 0), (-1, 0), (0, 1), (0, -1),
          (1, 1), (1, -1), (-1, 1), (-1, -1)]


def classify_entity(ec: int, er: int, state, grid, obstacles, entity_positions
                     ) -> Tuple[str, dict]:
    """对未识别 entity 在 (ec, er), 返回 (reason_code, detail)."""
    rows = len(grid); cols = len(grid[0])
    # 1. 8-邻分类
    all_walls = True
    obstacle_neighbors = []
    los_blocked = []
    valid_vp = []
    for dc, dr in DIRS_8:
        nc, nr = ec + dc, er + dr
        if not (0 <= nc < cols and 0 <= nr < rows):
            continue
        if grid[nr][nc] == 1:
            continue
        all_walls = False
        if (nc, nr) in obstacles:
            obstacle_neighbors.append((nc, nr))
            continue
        if not has_line_of_sight(nc, nr, ec, er, grid, entity_positions):
            los_blocked.append((nc, nr))
            continue
        valid_vp.append((nc, nr))

    if all_walls:
        return "A_all_wall_neighbors", {"valid_vp": []}
    if not valid_vp and not obstacle_neighbors and los_blocked:
        return "C_los_blocked", {"los_blocked": los_blocked}
    if not valid_vp and obstacle_neighbors:
        return "B_obstacle_on_neighbor", {
            "obstacle": obstacle_neighbors, "los_blocked": los_blocked}
    if not valid_vp:
        return "C_los_blocked", {"los_blocked": los_blocked}

    # 2. 有合法 viewpoint, 检查车能否走到 (任意一个)
    car_grid = pos_to_grid(state.car_x, state.car_y)
    visited = {car_grid}
    q = [car_grid]
    while q:
        c, r = q.pop(0)
        if (c, r) in valid_vp:
            return "E_unknown", {"reachable_vp": (c, r), "valid_vp": valid_vp}
        for dx, dy in DIRS_4:
            nc, nr = c + dx, r + dy
            if (nc, nr) in visited:
                continue
            if not (0 <= nc < cols and 0 <= nr < rows):
                continue
            if grid[nr][nc] == 1 or (nc, nr) in obstacles:
                continue
            visited.add((nc, nr))
            q.append((nc, nr))
    # valid_vp 存在但车走不到 → entity 锁住或车自己被锁
    return "D_vp_unreachable", {
        "valid_vp": valid_vp, "car_at": car_grid,
        "n_reachable_cells": len(visited),
    }


def analyze(item: dict) -> Dict:
    map_path = item["map"]
    seed = item["seed"]
    random.seed(seed)
    eng = GameEngine()
    try:
        eng.reset(map_path)
        with contextlib.redirect_stdout(io.StringIO()):
            plan_exploration(eng)
    except Exception as e:
        return {"map": map_path, "seed": seed, "err": str(e), "fails": []}
    state = eng.get_state()
    grid = state.grid
    entity_positions = set()
    for b in state.boxes: entity_positions.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs: entity_positions.add(pos_to_grid(bm.x, bm.y))
    obstacles = set()
    for b in state.boxes: obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs: obstacles.add(pos_to_grid(bm.x, bm.y))

    fails = []
    # Unidentified box
    for i, b in enumerate(state.boxes):
        if i in state.seen_box_ids: continue
        ec, er = pos_to_grid(b.x, b.y)
        # 排除 entity 自身 from obstacles 做 find_observation_point
        obs_no_self = obstacles - {(ec, er)}
        reason, detail = classify_entity(ec, er, state, grid,
                                          obs_no_self, entity_positions)
        fails.append({"type": "box", "idx": i, "pos": (ec, er),
                      "reason": reason, "detail": detail})
    for i, t in enumerate(state.targets):
        if i in state.seen_target_ids: continue
        ec, er = pos_to_grid(t.x, t.y)
        # target 不在 obstacles, 但 entity_positions 包含其他箱炸弹
        reason, detail = classify_entity(ec, er, state, grid,
                                          obstacles, entity_positions)
        fails.append({"type": "target", "idx": i, "pos": (ec, er),
                      "reason": reason, "detail": detail})
    return {"map": map_path, "seed": seed, "fails": fails}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default="runs/sage_pr/explore_failed_maps.json")
    parser.add_argument("--out", default="runs/sage_pr/explore_fails_analysis.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    list_path = args.list if os.path.isabs(args.list) else os.path.join(ROOT, args.list)
    with open(list_path) as f:
        items = json.load(f)
    if args.limit:
        items = items[:args.limit]
    print(f"analyzing {len(items)} maps")

    reasons_counter = Counter()
    reasons_by_phase = {p: Counter() for p in [3, 4, 5, 6]}
    results = []
    for i, item in enumerate(items):
        r = analyze(item)
        results.append(r)
        for f in r.get("fails", []):
            reasons_counter[f["reason"]] += 1
            reasons_by_phase[item["phase"]][f["reason"]] += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)}...")

    print("\n=== 整体 reason 分布 ===")
    for r, c in sorted(reasons_counter.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")
    print("\n=== Per-phase 分布 ===")
    for p in sorted(reasons_by_phase):
        if reasons_by_phase[p]:
            print(f"  phase {p}: {dict(reasons_by_phase[p])}")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    with open(out_path, "w") as f:
        json.dump({"reasons": dict(reasons_counter),
                    "by_phase": {p: dict(c) for p, c in reasons_by_phase.items()},
                    "details": results}, f, indent=2)
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
