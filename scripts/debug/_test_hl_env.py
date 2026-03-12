"""快速验证高层环境: 随机动作 + 完整 episode."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smartcar_sokoban.rl.high_level_env import SokobanHLEnv
import numpy as np

env = SokobanHLEnv(map_file='assets/maps/map1.txt', max_steps=20)
obs, info = env.reset(seed=42)

print(f"初始状态: {info}")
print(f"Obs shape: {obs.shape}")

total_reward = 0
for step in range(20):
    mask = env.action_masks()
    valid = [i for i, v in enumerate(mask) if v]
    if not valid:
        print(f"  Step {step}: 无有效动作, 结束")
        break

    action = np.random.choice(valid)
    action_names = {
        0: 'EXPLORE_BOX0', 1: 'EXPLORE_BOX1',
        5: 'EXPLORE_TGT0', 6: 'EXPLORE_TGT1',
        10: 'PUSH_BOX0', 11: 'PUSH_BOX1',
        15: 'DET_TNT0',
    }
    aname = action_names.get(action, f'ACTION_{action}')

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    print(f"  Step {step}: {aname:15s} | "
          f"reward={reward:+.1f} | "
          f"low_steps={info.get('low_level_steps', 0):3d} | "
          f"success={info.get('subtask_success', False)} | "
          f"remaining={info['remaining_boxes']}")

    if terminated:
        print(f"\n  通关! 总奖励={total_reward:.1f}, "
              f"总步数={info['total_low_steps']}")
        break
    if truncated:
        print(f"\n  截断! 总奖励={total_reward:.1f}, "
              f"总步数={info['total_low_steps']}")
        break

print("\n环境验证完成")
