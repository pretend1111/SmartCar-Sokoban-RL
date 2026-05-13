"""审计 oracle 计划里 push 是否赌博 — 看 push 时 box.class_id 是否已识别."""

import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from experiments.min_steps.planner_best import set_best_context
from experiments.min_steps.harness import run_planner
import sys as _sys
_PLANNER_NAME = _sys.argv[1] if len(_sys.argv) > 1 else 'v6'
if _PLANNER_NAME == 'v7':
    from experiments.min_steps.planner_oracle_v7 import planner_oracle_v7 as planner_oracle_v6
else:
    from experiments.min_steps.planner_oracle_v6 import planner_oracle_v6
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs


def audit_one(map_path, seed):
    """跑 v6, 在每个 push 之前检查 box 的 class_id 是否已被 FOV 识别 (或属于 forced pair)."""
    set_best_context(map_path, seed)
    random.seed(seed)
    eng = GameEngine()
    eng.reset(map_path)
    eng.discrete_step(6)

    # forced pair = 拓扑可推 (不需要 scan)
    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    forced_box_classes = set()
    for bi, ti in forced:
        if bi < len(state0.boxes):
            forced_box_classes.add(state0.boxes[bi].class_id)

    # Hook 在 push 类 step 检查 belief
    push_audit = []   # list of dict {step, box_class, was_known, was_forced}

    orig = eng.discrete_step
    def wrapped(a):
        tag = getattr(eng, '_step_tag', '')
        s_before = eng.get_state()
        seen_classes = {s_before.boxes[i].class_id for i in s_before.seen_box_ids}
        # 推之前要找哪个 box 会被推
        # 这里简化: 任何 push tag 的 step 都看作 push action
        boxes_before = [(pos_to_grid(b.x, b.y), b.class_id, b in [s_before.boxes[i] for i in s_before.seen_box_ids]) for b in s_before.boxes]
        car_before = pos_to_grid(s_before.car_x, s_before.car_y)
        result = orig(a)
        if tag == 'push':
            # 找哪个 box 移动了
            s_after = eng.get_state()
            box_moved = None
            for b_a in s_after.boxes:
                pos_a = pos_to_grid(b_a.x, b_a.y)
                # 看有没有 box 在 before 时不在这位置 (新出现的)
                found_match = False
                for bp, bc, _ in boxes_before:
                    if bp == pos_a and bc == b_a.class_id:
                        found_match = True; break
                if not found_match:
                    box_moved = b_a; break
            # 或者反查: 消失的位置
            if box_moved is None:
                for bp, bc, _ in boxes_before:
                    still_there = any(pos_to_grid(b.x, b.y) == bp and b.class_id == bc
                                      for b in s_after.boxes)
                    if not still_there:
                        # 应用排除律: 如果 known set 已含 N-1 个 → universe 完成
                        n_box = len(s_before.boxes)
                        universe = {b.class_id for b in s_before.boxes}
                        explicit_known = seen_classes | forced_box_classes
                        if len(explicit_known) >= n_box - 1:
                            effective_known = universe
                        else:
                            effective_known = explicit_known
                        is_forced = bc in forced_box_classes
                        was_scanned = bc in seen_classes
                        was_known = bc in effective_known
                        push_audit.append({
                            'class_id': bc,
                            'was_scanned': was_scanned,
                            'is_forced': is_forced,
                            'was_known': was_known,
                        })
                        break
        return result
    eng.discrete_step = wrapped   # type: ignore
    try:
        planner_oracle_v6(eng)
    except Exception:
        pass
    return push_audit, len(forced)


# Sample 30 maps × 2 seeds
manifest = json.load(open('assets/maps/phase456_seed_manifest.json'))
trials = []
for phase in ['phase4', 'phase5', 'phase6']:
    for k, v in list(manifest['phases'][phase].items())[:10]:
        for sd in v.get('verified_seeds', [])[:2]:
            trials.append((f'assets/maps/{phase}/{k}', sd))

total_pushes = 0
total_known = 0
total_scanned = 0
total_forced = 0
total_gambling = 0
fail_count = 0

for mp, sd in trials:
    if not os.path.exists(mp): continue
    audit, n_forced = audit_one(mp, sd)
    if not audit:
        fail_count += 1; continue
    for p in audit:
        total_pushes += 1
        if p['was_known']: total_known += 1
        if p['was_scanned']: total_scanned += 1
        if p['is_forced']: total_forced += 1
        if not p['was_known']: total_gambling += 1

print(f'Trials: {len(trials)}, fail: {fail_count}')
print(f'Total pushes: {total_pushes}')
print(f'  known at push time: {total_known} ({total_known/total_pushes*100:.1f}%)')
print(f'    of which scanned: {total_scanned}')
print(f'    of which forced:  {total_forced}')
print(f'  GAMBLING (unknown): {total_gambling} ({total_gambling/total_pushes*100:.1f}%)')
