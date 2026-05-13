"""走线审计 — 检查 plan 是否有可疑的浪费模式.

度量:
- revisit_ratio: 走过的相同 cell 平均访问次数 (理想 1.0, > 1.5 = 大量回头)
- scan_detour: 每个 scan 离最近 push 路径的额外步数 (越小越好)
- back_track: 推完一箱后立即原路返回的步数 (浪费)
- key_decisions: 找"以小博大"的关键节点 (短期+N 步 → 后续 -M 步, M >> N 的最大例子)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collections import Counter
from experiments.min_steps.planner_oracle_v6 import planner_oracle_v6
from experiments.min_steps.planner_oracle_v18 import planner_oracle_v18
from experiments.min_steps.planner_best import set_best_context
from experiments.min_steps.visualize import run_and_collect_frames
from smartcar_sokoban.solver.pathfinder import pos_to_grid


def audit(map_path, seed, planner_fn, planner_name):
    set_best_context(map_path, seed)
    ret = run_and_collect_frames(map_path, seed, planner_fn, planner_name)
    frames, trail, won, total, gambles = ret
    # 把 trail (浮点坐标) 转 grid cells
    trail_cells = [pos_to_grid(x, y) for x, y in trail]
    if not trail_cells: return None

    # revisit_ratio
    n_visits = Counter(trail_cells)
    unique_cells = len(n_visits)
    revisit_ratio = len(trail_cells) / unique_cells if unique_cells else 1.0

    # 找连续段最长 stretch of cells 都 revisited (回头走)
    # 简单版: count adjacent pairs where car_t and car_{t+2} same (perfectly 180度回头)
    rev_pairs = sum(1 for i in range(len(trail_cells)-2)
                    if trail_cells[i] == trail_cells[i+2])

    # 标记 scan vp 的 cells (从 tags 找 scan_walk / scan_rot)
    scan_segments = []
    cur_scan_start = None
    for i, (s, tag, sn) in enumerate(frames):
        if tag in ('scan_walk', 'scan_rot') and cur_scan_start is None:
            cur_scan_start = i
        elif tag not in ('scan_walk', 'scan_rot') and cur_scan_start is not None:
            scan_segments.append((cur_scan_start, i-1))
            cur_scan_start = None
    if cur_scan_start is not None:
        scan_segments.append((cur_scan_start, len(frames)-1))

    return dict(
        planner=planner_name, won=won, steps=total,
        n_gamble=sum(1 for ev in gambles if ev[2]),
        n_consume=len(gambles),
        unique_cells=unique_cells,
        revisit_ratio=revisit_ratio,
        n_rev_pairs=rev_pairs,
        n_scan_segments=len(scan_segments),
        trail_cells_len=len(trail_cells),
    )


if __name__ == '__main__':
    manifest = json.load(open('assets/maps/phase456_seed_manifest.json'))
    maps = []
    for phase in ['phase4', 'phase5', 'phase6']:
        for k, v in list(manifest['phases'][phase].items())[:10]:
            seeds = v.get('verified_seeds', [])
            if not seeds: continue
            p = f'assets/maps/{phase}/{k}'
            if os.path.exists(p):
                maps.append((p, seeds[0]))

    print(f'{"map":<22} {"sd":>4} | v6 step|rev|180  |  v18 step|rev|180  |  delta')
    print('-' * 90)
    sums = {'v6': [0,0,0], 'v18': [0,0,0]}
    for mp, sd in maps:
        r6 = audit(mp, sd, planner_oracle_v6, 'v6')
        r18 = audit(mp, sd, planner_oracle_v18, 'v18')
        if r6 is None or r18 is None: continue
        sums['v6'][0] += r6['steps']; sums['v6'][1] += r6['revisit_ratio']; sums['v6'][2] += r6['n_rev_pairs']
        sums['v18'][0] += r18['steps']; sums['v18'][1] += r18['revisit_ratio']; sums['v18'][2] += r18['n_rev_pairs']
        d_step = r18['steps'] - r6['steps']
        print(f'{os.path.basename(mp):<22} {sd:>4} | '
              f'{r6["steps"]:>3} {r6["revisit_ratio"]:>4.2f} {r6["n_rev_pairs"]:>3}  | '
              f'{r18["steps"]:>3} {r18["revisit_ratio"]:>4.2f} {r18["n_rev_pairs"]:>3}  | '
              f'{d_step:+4d}')

    n = len(maps)
    print(f'\nAvg: v6  step={sums["v6"][0]/n:.1f}  revisit_ratio={sums["v6"][1]/n:.2f}  '
          f'180_pairs={sums["v6"][2]/n:.1f}')
    print(f'Avg: v18 step={sums["v18"][0]/n:.1f}  revisit_ratio={sums["v18"][1]/n:.2f}  '
          f'180_pairs={sums["v18"][2]/n:.1f}')
