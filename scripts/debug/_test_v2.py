"""测试 v2 环境: 单步推箱."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smartcar_sokoban.rl.high_level_env import SokobanHLEnv, N_ACTIONS, PUSH_BOX_START, PUSH_BOMB_START
import numpy as np

ACTION_NAMES = {}
for i in range(5): ACTION_NAMES[i] = f'EXP_B{i}'
for i in range(5): ACTION_NAMES[5+i] = f'EXP_T{i}'
dirs = ['UP','DN','LF','RT']
for i in range(5):
    for d in range(4):
        ACTION_NAMES[10+i*4+d] = f'PUSH_B{i}_{dirs[d]}'
for i in range(3):
    for d in range(4):
        ACTION_NAMES[30+i*4+d] = f'PUSH_TNT{i}_{dirs[d]}'

for map_name in ['assets/maps/map1.txt']:
    print(f"{'='*50}")
    print(f"  {map_name} | N_ACTIONS={N_ACTIONS}")
    print(f"{'='*50}")

    env = SokobanHLEnv(map_file=map_name, max_steps=60, baseline_steps=80)
    obs, info = env.reset(seed=42)
    print(f"  Obs shape: {obs.shape}")
    print(f"  Info: {info}")

    mask = env.action_masks()
    valid = [ACTION_NAMES.get(i, f'A{i}') for i,v in enumerate(mask) if v]
    print(f"  Valid actions ({len(valid)}): {valid}")

    np.random.seed(42)
    total_r = 0
    for step in range(60):
        mask = env.action_masks()
        valid_ids = [i for i,v in enumerate(mask) if v]
        if not valid_ids:
            print(f"  Step {step}: NO VALID")
            break
        action = np.random.choice(valid_ids)
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        aname = ACTION_NAMES.get(action, f'A{action}')
        if r > 1 or r < -1 or step < 5 or step % 10 == 0:
            print(f"  Step {step:3d}: {aname:14s} r={r:+6.1f} "
                  f"low={info['low_level_steps']:2d} "
                  f"rem={info['remaining_boxes']} "
                  f"expl={'Y' if info['exploration_complete'] else 'N'}")
        if term:
            print(f"\n  WIN! steps={info['total_low_steps']} reward={total_r:.1f}")
            break
        if trunc:
            print(f"\n  TRUNCATED steps={info['total_low_steps']} reward={total_r:.1f}")
            break

print("\nDone!")
