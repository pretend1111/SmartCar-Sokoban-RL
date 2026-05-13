"""benchmark v6 vs v18 (new consumption-side penalty + multiplicative).

每张图各跑一遍, 用新 gambling 定义 (consumption-side, box+target 双侧) 统计:
  steps, gambling consumptions, box-blind, tgt-blind
对比 v6 (no penalty) 和 v18 (new model).
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.min_steps.planner_oracle_v6 import planner_oracle_v6
from experiments.min_steps.planner_oracle_v18 import planner_oracle_v18
from experiments.min_steps.planner_best import set_best_context
from experiments.min_steps.visualize import run_and_collect_frames


def bench(map_path, seed, fn, name):
    set_best_context(map_path, seed)
    ret = run_and_collect_frames(map_path, seed, fn, name)
    frames, trail, won, total, gambles = ret
    n_g = sum(1 for ev in gambles if ev[2])
    n_consume = len(gambles)
    n_bb = sum(1 for ev in gambles if not ev[3])
    n_tb = sum(1 for ev in gambles if not ev[4])
    return dict(planner=name, won=won, steps=total,
                gambling=n_g, consume=n_consume,
                box_blind=n_bb, tgt_blind=n_tb)


manifest = json.load(open('assets/maps/phase456_seed_manifest.json'))
N_PER_PHASE = 10

maps = []
for phase in ['phase4', 'phase5', 'phase6']:
    items = list(manifest['phases'][phase].items())
    for k, v in items[:N_PER_PHASE]:
        seeds = v.get('verified_seeds', [])
        if not seeds: continue
        seed = seeds[0]
        path = f'assets/maps/{phase}/{k}'
        if os.path.exists(path):
            maps.append((path, seed))

print(f"Total {len(maps)} maps × 2 planners\n")

planners = [
    ('v6', planner_oracle_v6),
    ('v18-mult', planner_oracle_v18),
]

results = []
for mp, sd in maps:
    row = {'map': os.path.basename(mp), 'seed': sd}
    for name, fn in planners:
        r = bench(mp, sd, fn, name)
        row[name] = r
        print(f"  {row['map']:<22} sd={sd:<4} {name:<10} "
              f"{'won' if r['won'] else 'LOST'} steps={r['steps']:>3} "
              f"g={r['gambling']}/{r['consume']} "
              f"bb={r['box_blind']} tb={r['tgt_blind']}", flush=True)
    results.append(row)

print("\n=== Aggregate ===")
for name, _ in planners:
    rs = [r[name] for r in results if r[name]['won']]
    if not rs: continue
    avg_s = sum(r['steps'] for r in rs) / len(rs)
    sum_g = sum(r['gambling'] for r in rs)
    sum_c = sum(r['consume'] for r in rs)
    sum_bb = sum(r['box_blind'] for r in rs)
    sum_tb = sum(r['tgt_blind'] for r in rs)
    print(f"{name:<10}  won={len(rs)}/{len(results)}  "
          f"avg_step={avg_s:.2f}  gambling={sum_g}/{sum_c} ({sum_g*100/sum_c:.1f}%)  "
          f"box-blind={sum_bb} tgt-blind={sum_tb}")
