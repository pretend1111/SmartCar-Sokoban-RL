"""统计 mid-game 新 forced pair 频次, 看 D3-2 收益空间."""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs, _box_axis_lock, _push_reachable_along_axis
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from experiments.min_steps.planner_oracle import _god_plan, _fresh_engine_from_eng
from experiments.min_steps.planner_best import set_best_context
from experiments.sage_pr.build_dataset_v3 import apply_solver_move


def _forced_pairs_from_snapshot(walls, box_data, target_data, bomb_positions):
    """Re-impl find_forced_pairs from raw data (no engine state object)."""
    other_obstacles = set()
    box_pos_list = []
    for i, (bp, _cid) in enumerate(box_data):
        box_pos_list.append((i, bp))
        other_obstacles.add(bp)
    for bp in bomb_positions:
        other_obstacles.add(bp)
    target_pos = {i: tp for i, (tp, _num) in enumerate(target_data)}
    pairs = []
    used = set()
    for i, bp in box_pos_list:
        axis = _box_axis_lock(walls, *bp)
        if axis not in ("horizontal", "vertical"):
            continue
        obs_no_self = other_obstacles - {bp}
        reach = _push_reachable_along_axis(walls, bp, axis, obs_no_self)
        reachable_tgts = [j for j, tp in target_pos.items()
                          if tp in reach and j not in used]
        if len(reachable_tgts) == 1:
            j = reachable_tgts[0]
            pairs.append((i, j))
            used.add(j)
    return pairs


manifest = json.load(open('assets/maps/phase456_seed_manifest.json'))
N_PER_PHASE = 20

trials = []
for phase in ['phase4', 'phase5', 'phase6']:
    items = list(manifest['phases'][phase].items())[:N_PER_PHASE]
    for k, v in items:
        seeds = v.get('verified_seeds', [])[:2]
        path = f'assets/maps/{phase}/{k}'
        if not os.path.exists(path): continue
        for sd in seeds:
            trials.append((path, sd))

print(f'Total trials: {len(trials)}')

total_static = 0
total_dynamic_extra = 0
n_trials = 0

for path, sd in trials:
    set_best_context(path, sd)
    random.seed(sd)
    e = GameEngine(); e.reset(path); e.discrete_step(6)
    plan = _god_plan(e)
    if not plan: continue
    state = e.get_state()
    static_pairs = find_forced_pairs(state)
    max_pairs = len(static_pairs)
    # 模拟整局, 每一步检查 forced pairs
    sim = _fresh_engine_from_eng(e)
    sim_state = sim.get_state()
    pairs_0 = find_forced_pairs(sim_state)
    max_pairs = max(max_pairs, len(pairs_0))
    for move in plan:
        if not apply_solver_move(sim, move):
            break
        ss = sim.get_state()
        pairs_k = find_forced_pairs(ss)
        if len(pairs_k) > max_pairs:
            max_pairs = len(pairs_k)
    extra = max_pairs - len(static_pairs)
    total_static += len(static_pairs)
    total_dynamic_extra += extra
    n_trials += 1

print(f'\nTrials analyzed: {n_trials}')
print(f'Avg static forced pairs at t=0: {total_static/n_trials:.2f}')
print(f'Avg additional forced pairs gained mid-game: {total_dynamic_extra/n_trials:.2f}')
print(f'-> Potential dynamic-detection savings: ~{total_dynamic_extra/n_trials*2:.1f} step/map (1 scan ≈ 2 step)')
