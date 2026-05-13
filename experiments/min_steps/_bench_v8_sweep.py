"""30-map sample × alpha sweep, compare gambling rate vs avg steps."""
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


def audit(mp, sd, planner):
    set_best_context(mp, sd); random.seed(sd)
    eng = GameEngine(); eng.reset(mp); eng.discrete_step(6)
    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    forced_classes = {state0.boxes[bi].class_id for bi, _ in forced if bi < len(state0.boxes)}
    N = len(state0.boxes); universe = {b.class_id for b in state0.boxes}
    first_pushed = set(); g = 0; total = 0
    orig = eng.discrete_step
    def wrapped(a):
        nonlocal g, total
        tag = getattr(eng, '_step_tag', '')
        s_b = eng.get_state()
        bb = [(pos_to_grid(b.x, b.y), b.class_id) for b in s_b.boxes]
        seen = {s_b.boxes[i].class_id for i in s_b.seen_box_ids}
        explicit = seen | forced_classes
        eff = universe if len(explicit) >= N - 1 else explicit
        r = orig(a)
        if tag == 'push':
            s_a = eng.get_state()
            for bp, bc in bb:
                if not any(pos_to_grid(b.x, b.y) == bp and b.class_id == bc for b in s_a.boxes):
                    if bc in first_pushed: break
                    first_pushed.add(bc); total += 1
                    if bc not in eff: g += 1
                    break
        return r
    eng.discrete_step = wrapped
    try: planner(eng)
    except: pass
    return total, g, eng.get_state().won


manifest = json.load(open('assets/maps/phase456_seed_manifest.json'))
N_PER_PHASE = 10
trials = []
for phase in ['phase4', 'phase5', 'phase6']:
    for k, v in list(manifest['phases'][phase].items())[:N_PER_PHASE]:
        sds = v.get('verified_seeds', [])[:1]   # 1 seed per map
        for sd in sds:
            trials.append((f'assets/maps/{phase}/{k}', sd))
print(f'trials: {len(trials)}')

results = {}
for alpha in [0.1, 0.2, 0.3, 0.4, 0.7, 1.5, 3.0]:
    pl = functools.partial(planner_oracle_v8, alpha=alpha)
    tot_step = 0; wins = 0; total_first = 0; total_gamb = 0
    for mp, sd in trials:
        set_best_context(mp, sd)
        r = run_planner(mp, sd, f'v8_a{alpha}', pl)
        if r.won: wins += 1; tot_step += r.total_steps
        first, gamb, won = audit(mp, sd, pl)
        total_first += first; total_gamb += gamb
    avg = tot_step / max(1, wins)
    rate = total_gamb / max(1, total_first) * 100
    print(f'  α={alpha:>5}: wins {wins}/{len(trials)}  avg={avg:.2f}  '
          f'gambling-rate (1st push) = {total_gamb}/{total_first} = {rate:.1f}%')
    results[alpha] = (wins, avg, rate)

# baseline 对照
print('\n--- baseline ---')
for name, pl in [('v0_lower', planner_v0_godmode_lowerbound),
                  ('v6_oracle', planner_oracle_v6)]:
    tot_step = 0; wins = 0; total_first = 0; total_gamb = 0
    for mp, sd in trials:
        set_best_context(mp, sd)
        r = run_planner(mp, sd, name, pl)
        if r.won: wins += 1; tot_step += r.total_steps
        first, gamb, won = audit(mp, sd, pl)
        total_first += first; total_gamb += gamb
    avg = tot_step / max(1, wins)
    rate = total_gamb / max(1, total_first) * 100
    print(f'  {name:<12}: wins {wins}/{len(trials)}  avg={avg:.2f}  '
          f'gambling = {total_gamb}/{total_first} = {rate:.1f}%')
