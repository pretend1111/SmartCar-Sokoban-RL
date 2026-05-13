"""Final validation: 90-map sample, test best teacher candidates."""
import os, sys, json, random, functools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs
from experiments.min_steps.harness import (
    run_planner, planner_v0_godmode_lowerbound,
)
from experiments.min_steps.planner_best import set_best_context
from experiments.min_steps.planner_oracle_v6 import planner_oracle_v6
from experiments.min_steps.planner_oracle_v8 import planner_oracle_v8
from experiments.min_steps.planner_oracle_v10 import planner_oracle_v10


def audit(mp, sd, planner):
    set_best_context(mp, sd); random.seed(sd)
    eng = GameEngine(); eng.reset(mp); eng.discrete_step(6)
    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    fc = {state0.boxes[bi].class_id for bi,_ in forced if bi < len(state0.boxes)}
    N = len(state0.boxes); universe = {b.class_id for b in state0.boxes}
    first = set(); g = 0; total = 0
    orig = eng.discrete_step
    def wr(a):
        nonlocal g, total
        tag = getattr(eng, '_step_tag', '')
        sb = eng.get_state()
        bb = [(pos_to_grid(b.x, b.y), b.class_id) for b in sb.boxes]
        seen = {sb.boxes[i].class_id for i in sb.seen_box_ids}
        consumed = universe - {b.class_id for b in sb.boxes}
        explicit = seen | fc | consumed
        eff = universe if len(explicit) >= N - 1 else explicit
        r = orig(a)
        if tag == 'push':
            sa = eng.get_state()
            for bp, bc in bb:
                if not any(pos_to_grid(b.x, b.y) == bp and b.class_id == bc for b in sa.boxes):
                    if bc in first: break
                    first.add(bc); total += 1
                    if bc not in eff: g += 1
                    break
        return r
    eng.discrete_step = wr
    try: planner(eng)
    except: pass
    return total, g


manifest = json.load(open('assets/maps/phase456_seed_manifest.json'))
N_PER_PHASE = 30
trials = []
for ph in ['phase4', 'phase5', 'phase6']:
    for k, v in list(manifest['phases'][ph].items())[:N_PER_PHASE]:
        for sd in v.get('verified_seeds', [])[:1]:
            trials.append((f'assets/maps/{ph}/{k}', sd))
print(f'trials: {len(trials)}', flush=True)

cfgs = [
    ('v0_lb (cheat)', planner_v0_godmode_lowerbound),
    ('v6 (no penalty)', planner_oracle_v6),
    ('v8 α=0.4 (linear)', functools.partial(planner_oracle_v8, alpha=0.4)),
    ('v10 α=0.4 (log)', functools.partial(planner_oracle_v10, alpha=0.4)),
    ('v8 α=1.0', functools.partial(planner_oracle_v8, alpha=1.0)),
]
for label, pl in cfgs:
    tot=0; wins=0; tf=0; tg=0
    for i, (mp, sd) in enumerate(trials):
        set_best_context(mp, sd)
        r = run_planner(mp, sd, label, pl)
        if r.won: wins += 1; tot += r.total_steps
        f, g = audit(mp, sd, pl)
        tf += f; tg += g
    avg = tot/max(1,wins); rate = tg/max(1,tf)*100
    print(f'  {label:<24}: wins {wins:>3}/{len(trials)}  avg={avg:.2f}  '
          f'gambling={tg}/{tf}={rate:.1f}%', flush=True)
