"""调试: MultiBoxSolver 跑 phase6_11.txt"""
import os, sys, random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid

cfg = GameConfig()
engine = GameEngine(cfg, ROOT)

random.seed(42)
engine.reset('assets/maps/phase6/phase6_11.txt')
state = engine.get_state()

# 打印地图信息
print("=== phase6_11.txt ===")
print(f"车位置: ({state.car_x}, {state.car_y}) -> grid {pos_to_grid(state.car_x, state.car_y)}")
print(f"\n箱子 ({len(state.boxes)}):")
for i, b in enumerate(state.boxes):
    print(f"  [{i}] pos=({b.x}, {b.y}) grid=({int(b.x)}, {int(b.y)}) class_id={b.class_id}")
print(f"\n目标 ({len(state.targets)}):")
for i, t in enumerate(state.targets):
    print(f"  [{i}] pos=({t.x}, {t.y}) grid=({int(t.x)}, {int(t.y)}) num_id={t.num_id}")
print(f"\n炸弹 ({len(state.bombs)}):")
for i, b in enumerate(state.bombs):
    print(f"  [{i}] pos=({b.x}, {b.y}) grid=({int(b.x)}, {int(b.y)})")

# 准备求解器输入
car_pos = pos_to_grid(state.car_x, state.car_y)
boxes = [((int(b.x), int(b.y)), b.class_id) for b in state.boxes]
targets = {t.num_id: (int(t.x), int(t.y)) for t in state.targets}
bombs = [(int(b.x), int(b.y)) for b in state.bombs]

print(f"\n--- 求解器输入 ---")
print(f"car_pos = {car_pos}")
print(f"boxes   = {boxes}")
print(f"targets = {targets}")
print(f"bombs   = {bombs}")

# 打印地图
print(f"\n--- 地图网格 ---")
for r, row in enumerate(state.grid):
    line = ""
    for c, cell in enumerate(row):
        pos = (c, r)
        if pos == car_pos:
            line += "C"
        elif any(pos == (int(b.x), int(b.y)) for b in state.boxes):
            line += "$"
        elif any(pos == (int(t.x), int(t.y)) for t in state.targets):
            line += "."
        elif any(pos == (int(b.x), int(b.y)) for b in state.bombs):
            line += "*"
        elif cell == 1:
            line += "#"
        else:
            line += "-"
    print(f"  {r:2d}: {line}")

# 求解
print(f"\n--- MultiBoxSolver 求解中 (time_limit=1800s) ... ---")
solver = MultiBoxSolver(
    grid=state.grid,
    car_pos=car_pos,
    boxes=boxes,
    targets=targets,
    bombs=bombs,
)
solution = solver.solve(max_cost=1000, time_limit=1800.0)

if solution:
    print(f"\n✅ 找到解!")
    print(f"推操作数: {len(solution)}")
    walk_steps = sum(wc + 1 for _, _, _, wc in solution)
    print(f"总步数: {walk_steps}")
    for i, (etype, eid, direction, wc) in enumerate(solution):
        print(f"  [{i}] {etype} {eid} dir={direction} walk={wc}")
else:
    print(f"\n❌ 无解或超时")
