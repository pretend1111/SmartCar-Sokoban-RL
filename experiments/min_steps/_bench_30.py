"""30 张 verified 图广覆盖测试. 每张图用其 verified seed[0]."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.min_steps.harness import (
    run_planner, planner_v0_godmode_lowerbound, planner_v1_explore_first,
    print_table, summary,
)
from experiments.min_steps.planner_best import planner_best_of_three, set_best_context
from experiments.min_steps.planner_oracle import planner_oracle
from experiments.min_steps.planner_oracle_v2 import planner_oracle_v2
from experiments.min_steps.planner_oracle_v3b import planner_oracle_v3b

manifest = json.load(open('assets/maps/phase456_seed_manifest.json'))
N_PER_PHASE = 10

maps = []
for phase in ['phase4', 'phase5', 'phase6']:
    items = list(manifest['phases'][phase].items())
    # 取前 N_PER_PHASE 张, 每张用首个 verified seed
    for k, v in items[:N_PER_PHASE]:
        seeds = v.get('verified_seeds', [])
        if not seeds:
            continue
        seed = seeds[0]
        path = f'assets/maps/{phase}/{k}'
        if os.path.exists(path):
            maps.append((path, seed))

print(f"Total maps: {len(maps)}\n")

planners = {
    "v0_lower": planner_v0_godmode_lowerbound,
    "v1_explore": planner_v1_explore_first,
    "oracle": planner_oracle,
    "oracle_v3b": planner_oracle_v3b,
}

results = []
for mp, sd in maps:
    set_best_context(mp, sd)
    for name, fn in planners.items():
        r = run_planner(mp, sd, name, fn)
        results.append(r)
        # 实时打 progress
        print(f"  {mp.split('/')[-1]:<18} seed={sd:<4} {name:<12} "
              f"{'✓' if r.won else '✗'} {r.total_steps:>4} ({r.wall_time_s:.1f}s)",
              flush=True)

print()
summary(results)

# 详细 per-map 对比
print("\nPer-map breakdown (only maps where all planners won):")
from collections import defaultdict
by_map = defaultdict(dict)
for r in results:
    by_map[(r.map_path, r.seed)][r.planner] = (r.won, r.total_steps)

print(f"{'map':<22} {'seed':>4} {'v0':>5} {'v1ex':>6} {'oracle':>7} {'orcl3b':>7} {'3b-v0':>6}")
n_won_all = 0
sum_v0=sum_v1=sum_o=sum_o3=0
for (mp, sd), planners_res in by_map.items():
    keys = ['v0_lower','v1_explore','oracle','oracle_v3b']
    if all(planners_res[p][0] for p in keys):
        v0=planners_res['v0_lower'][1]
        v1=planners_res['v1_explore'][1]
        o=planners_res['oracle'][1]
        o3=planners_res['oracle_v3b'][1]
        gap = o3 - v0
        n_won_all += 1
        sum_v0+=v0; sum_v1+=v1; sum_o+=o; sum_o3+=o3
        mp_short = mp.split('/')[-1]
        print(f"{mp_short:<22} {sd:>4} {v0:>5} {v1:>6} {o:>7} {o3:>7} {gap:>+6}")
print(f"\n{n_won_all} maps all win.")
if n_won_all > 0:
    print(f"avg: v0={sum_v0/n_won_all:.1f}  v1={sum_v1/n_won_all:.1f}  "
          f"oracle={sum_o/n_won_all:.1f}  oracle_v3b={sum_o3/n_won_all:.1f}")
    print(f"v3b vs oracle_v1: {(sum_o3-sum_o)/n_won_all:+.2f} step  ({(sum_o3-sum_o)/sum_o*100:+.1f}%)")
