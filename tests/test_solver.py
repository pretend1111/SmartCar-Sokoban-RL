"""测试 BFS 求解器 — 不依赖渲染."""
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, base_dir)

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.auto_player import AutoPlayer

def test_map(map_name):
    cfg = GameConfig()
    cfg.control_mode = "discrete"
    engine = GameEngine(cfg, base_dir)
    
    map_path = os.path.join("assets", "maps", map_name)
    state = engine.reset(map_path)
    
    print(f"\n{'='*50}")
    print(f"测试地图: {map_name}")
    print(f"车位置: ({state.car_x:.1f}, {state.car_y:.1f})")
    print(f"箱子数: {len(state.boxes)}")
    for i, b in enumerate(state.boxes):
        print(f"  箱子{i}: pos=({b.x:.1f},{b.y:.1f}) id={b.class_id}")
    print(f"目标数: {len(state.targets)}")
    for i, t in enumerate(state.targets):
        print(f"  目标{i}: pos=({t.x:.1f},{t.y:.1f}) id={t.num_id}")
    print(f"炸弹数: {len(state.bombs)}")
    for i, b in enumerate(state.bombs):
        print(f"  炸弹{i}: pos=({b.x:.1f},{b.y:.1f})")
    print()
    
    player = AutoPlayer(engine)
    actions = player.solve()
    
    state = engine.get_state()
    print(f"\n结果: {'通关!' if state.won else '未通关'}")
    print(f"总步数: {len(actions)}")
    return state.won

if __name__ == "__main__":
    maps = ["map1.txt", "map2.txt", "map3.txt"]
    if len(sys.argv) > 1:
        maps = [sys.argv[1]]
    
    results = {}
    for m in maps:
        try:
            results[m] = test_map(m)
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            results[m] = False
    
    print(f"\n{'='*50}")
    print("总结:")
    for m, ok in results.items():
        print(f"  {m}: {'✅' if ok else '❌'}")
