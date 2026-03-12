"""测试炸弹/墙壁障碍修复 — 不依赖渲染."""
import os
import sys
import random
import time

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.auto_player import AutoPlayer


def test_seed(seed, map_name="map3.txt"):
    """测试指定 seed 能否通关."""
    cfg = GameConfig()
    cfg.control_mode = "discrete"
    engine = GameEngine(cfg, base_dir)

    random.seed(seed)
    map_path = os.path.join("assets", "maps", map_name)
    state = engine.reset(map_path)

    print(f"\n{'='*60}")
    print(f"测试: {map_name}, seed={seed}")
    print(f"车: ({state.car_x:.1f}, {state.car_y:.1f})")
    for i, b in enumerate(state.boxes):
        print(f"  箱子{i}: ({b.x:.1f},{b.y:.1f}) id={b.class_id}")
    for i, t in enumerate(state.targets):
        print(f"  目标{i}: ({t.x:.1f},{t.y:.1f}) id={t.num_id}")
    for i, b in enumerate(state.bombs):
        print(f"  炸弹{i}: ({b.x:.1f},{b.y:.1f})")
    print()

    player = AutoPlayer(engine)
    t0 = time.perf_counter()
    actions = player.solve()
    dt = time.perf_counter() - t0

    state = engine.get_state()
    result = '✅ 通关!' if state.won else '❌ 未通关'
    print(f"\n结果: {result}, 步数: {len(actions)}, 耗时: {dt*1000:.0f}ms")
    return state.won


def batch_test(map_name="map3.txt", num_seeds=50):
    """批量测试多个 seed."""
    pass_count = 0
    fail_seeds = []

    for seed in range(num_seeds):
        won = test_seed(seed, map_name)
        if won:
            pass_count += 1
        else:
            fail_seeds.append(seed)

    print(f"\n{'='*60}")
    print(f"批量测试: {map_name}, {num_seeds} 个种子")
    print(f"通过: {pass_count}/{num_seeds} "
          f"({pass_count/num_seeds*100:.0f}%)")
    if fail_seeds:
        print(f"失败种子: {fail_seeds}")
    else:
        print("🎉 全部通过!")


def batch_all_maps(num_seeds=20):
    """批量测试所有地图."""
    import glob
    map_files = sorted(glob.glob(os.path.join(base_dir, "assets", "maps", "*.txt")))
    map_files = [f for f in map_files
                 if not os.path.basename(f).startswith('_')]

    for mf in map_files:
        map_name = os.path.basename(mf)
        print(f"\n{'#'*60}")
        print(f"# 测试地图: {map_name}")
        print(f"{'#'*60}")
        batch_test(map_name, num_seeds)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        batch_test(num_seeds=n)
    elif len(sys.argv) > 1 and sys.argv[1] == "all":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        batch_all_maps(n)
    else:
        # 默认测试关键种子
        seeds = [42, 3, 0, 1, 10]
        if len(sys.argv) > 1:
            seeds = [int(s) for s in sys.argv[1:]]

        results = {}
        for s in seeds:
            results[s] = test_seed(s)

        print(f"\n{'='*60}")
        print("总结:")
        for s, ok in results.items():
            print(f"  seed={s}: {'✅' if ok else '❌'}")
