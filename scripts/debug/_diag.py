"""诊断v4: 检查修复后的env - 无效推是否被正确标记."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smartcar_sokoban.rl.high_level_env import SokobanHLEnv, PUSH_BOX_START, N_DIRS, DIR_DELTAS

env = SokobanHLEnv(map_file='assets/maps/phase1/phase1_01.txt', max_steps=50, baseline_steps=30)
obs, info = env.reset(seed=5)

dir_names = ['UP','DN','LF','RT']
total_r = 0

for step in range(50):
    state = env.engine.get_state()
    if not state.boxes:
        break
    
    box = state.boxes[0]
    tgt = env.engine.get_state().targets[0] if env.engine.get_state().targets else None
    if not tgt:
        break
    
    mask = env.action_masks()
    valid = [i for i, v in enumerate(mask) if v]
    
    # 贪心: pick direction that reduces distance most
    best_a = valid[0]
    best_reduction = -999
    for a in valid:
        if a >= PUSH_BOX_START:
            d = (a - PUSH_BOX_START) % N_DIRS
            dx, dy = DIR_DELTAS[d]
            dist_before = abs(box.x - tgt.x) + abs(box.y - tgt.y)
            dist_after = abs(box.x + dx - tgt.x) + abs(box.y + dy - tgt.y)
            reduction = dist_before - dist_after
            if reduction > best_reduction:
                best_reduction = reduction
                best_a = a
    
    obs, r, term, trunc, info = env.step(best_a)
    total_r += r
    
    dn = dir_names[(best_a - PUSH_BOX_START) % 4] if best_a >= PUSH_BOX_START else f'A{best_a}'
    
    print(f"  step{step:2d}: {dn:3s} r={r:+6.2f} low={info['low_level_steps']:2d} "
          f"ok={info['subtask_success']} rem={info['remaining_boxes']} won={info['won']}")
    
    if term:
        print(f"\n✅ WIN! total_r={total_r:.1f}, low_steps={info['total_low_steps']}")
        break
    if trunc:
        print(f"\n❌ TRUNCATED total_r={total_r:.1f}")
        break

print("Done!")
