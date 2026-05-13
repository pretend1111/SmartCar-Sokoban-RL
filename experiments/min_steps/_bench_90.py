"""90-map bench: 30 maps per phase × 3 verified seeds each."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.min_steps.harness import (
    run_planner, planner_v0_godmode_lowerbound, planner_v1_explore_first,
)
from experiments.min_steps.planner_best import set_best_context
from experiments.min_steps.planner_oracle import planner_oracle
from experiments.min_steps.planner_oracle_v3b import planner_oracle_v3b

# 可选: 跑完 v4 后加进来
try:
    from experiments.min_steps.planner_oracle_v4 import planner_oracle_v4
    HAS_V4 = True
except ImportError:
    HAS_V4 = False
try:
    from experiments.min_steps.planner_oracle_v6 import planner_oracle_v6
    HAS_V6 = True
except ImportError:
    HAS_V6 = False

manifest = json.load(open('assets/maps/phase456_seed_manifest.json'))
N_MAPS_PER_PHASE = 30
N_SEEDS_PER_MAP = 3

maps = []
for phase in ['phase4', 'phase5', 'phase6']:
    items = list(manifest['phases'][phase].items())[:N_MAPS_PER_PHASE]
    for k, v in items:
        seeds = v.get('verified_seeds', [])[:N_SEEDS_PER_MAP]
        path = f'assets/maps/{phase}/{k}'
        if not os.path.exists(path):
            continue
        for sd in seeds:
            maps.append((path, sd))

print(f'Total trials: {len(maps)}', flush=True)

planners = {
    "v0_lower": planner_v0_godmode_lowerbound,
    "oracle_v1": planner_oracle,
    "oracle_v3b": planner_oracle_v3b,
}
if HAS_V4:
    planners["oracle_v4"] = planner_oracle_v4
if HAS_V6:
    planners["oracle_v6"] = planner_oracle_v6

results = []
for mp, sd in maps:
    set_best_context(mp, sd)
    for name, fn in planners.items():
        r = run_planner(mp, sd, name, fn)
        results.append(r)
        print(f"  {mp.split('/')[-1]:<18} seed={sd:<4} {name:<13} "
              f"{'✓' if r.won else '✗'} {r.total_steps:>4}", flush=True)

from collections import defaultdict
agg = defaultdict(lambda: {"n": 0, "won": 0, "steps_sum": 0})
for r in results:
    a = agg[r.planner]
    a["n"] += 1
    a["won"] += int(r.won)
    if r.won: a["steps_sum"] += r.total_steps

print('\n=== SUMMARY ===', flush=True)
for name in planners:
    a = agg[name]
    avg = a["steps_sum"] / max(1, a["won"])
    print(f"  {name:<13} {a['won']}/{a['n']}, avg-on-win={avg:.2f}", flush=True)

# All-win 子集
by_trial = defaultdict(dict)
for r in results:
    by_trial[(r.map_path, r.seed)][r.planner] = (r.won, r.total_steps)
keys_to_check = list(planners.keys())
sums = {p: 0 for p in keys_to_check}
n = 0
for trial, pd in by_trial.items():
    if all(pd[p][0] for p in keys_to_check):
        n += 1
        for p in keys_to_check:
            sums[p] += pd[p][1]
print(f'\nAll-{len(keys_to_check)} winners: {n} trials', flush=True)
if n > 0:
    print(f'  Apples-to-apples avg:', flush=True)
    for p in keys_to_check:
        print(f'    {p:<13} {sums[p]/n:.2f}', flush=True)
    print(f'  v3b gap to v0: +{(sums["oracle_v3b"]-sums["v0_lower"])/n:.2f}', flush=True)
    if HAS_V4:
        print(f'  v4 gap to v0:  +{(sums["oracle_v4"]-sums["v0_lower"])/n:.2f}', flush=True)
        print(f'  v4 vs v3b:    {(sums["oracle_v4"]-sums["oracle_v3b"])/n:+.2f}', flush=True)
    if HAS_V6:
        print(f'  v6 gap to v0:  +{(sums["oracle_v6"]-sums["v0_lower"])/n:.2f}', flush=True)
        print(f'  v6 vs v4:     {(sums["oracle_v6"]-sums["oracle_v4"])/n:+.2f}', flush=True)
