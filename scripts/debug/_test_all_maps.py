"""测试 map3 上的高层环境 (含 TNT)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smartcar_sokoban.rl.high_level_env import SokobanHLEnv
import numpy as np

ACTION_NAMES = {}
for i in range(5):
    ACTION_NAMES[i] = f'EXP_BOX{i}'
for i in range(5):
    ACTION_NAMES[5 + i] = f'EXP_TGT{i}'
for i in range(5):
    ACTION_NAMES[10 + i] = f'PUSH_{i}'
for i in range(3):
    ACTION_NAMES[15 + i] = f'DET_{i}'

for map_name in ['assets/maps/map1.txt', 'assets/maps/map2.txt', 'assets/maps/map3.txt']:
    print(f"\n{'='*50}")
    print(f"  {map_name}")
    print(f"{'='*50}")

    env = SokobanHLEnv(map_file=map_name, max_steps=25)
    obs, info = env.reset(seed=42)
    print(f"  Pairs={info['total_pairs']}, Bombs={info['remaining_bombs']}")

    np.random.seed(42)
    total_r = 0

    for step in range(25):
        mask = env.action_masks()
        valid = [i for i, v in enumerate(mask) if v]
        if not valid:
            print(f"  Step {step}: NO VALID ACTIONS")
            break

        action = np.random.choice(valid)
        obs, r, term, trunc, info = env.step(action)
        total_r += r

        aname = ACTION_NAMES.get(action, f'A{action}')
        print(f"  Step {step:2d}: {aname:10s} "
              f"r={r:+7.1f}  low={info['low_level_steps']:3d}  "
              f"ok={str(info['subtask_success']):5s}  "
              f"rem_box={info['remaining_boxes']}  "
              f"expl={'Y' if info['exploration_complete'] else 'N'}")

        if term:
            print(f"\n  WIN! total_low={info['total_low_steps']}, "
                  f"reward={total_r:.1f}")
            break
        if trunc:
            print(f"\n  TRUNCATED. total_low={info['total_low_steps']}, "
                  f"reward={total_r:.1f}")
            break

print("\nDone!")
