"""临时测试: phase 6 ida_timeout maps 在 8-向 + 长 IDA* time 下能否解.

看是质量低 (能解但慢) 还是真无解 (生成器 bug).
不修改原 multi_box_solver, 而是 monkey-patch DIRS 临时变成 8 向.
"""
import contextlib
import io
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.paths import PROJECT_ROOT
from smartcar_sokoban.solver import multi_box_solver as mbs
from smartcar_sokoban.solver.explorer import plan_exploration
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import pos_to_grid


def test_one(map_path: str, time_limit: float, dirs_8: bool, seed: int = 42):
    cfg = GameConfig()
    cfg.control_mode = "discrete"
    engine = GameEngine(cfg, str(PROJECT_ROOT))
    random.seed(seed)
    engine.reset(map_path)

    devnull = io.StringIO()
    with contextlib.redirect_stdout(devnull):
        plan_exploration(engine)

    state = engine.get_state()
    boxes = [(pos_to_grid(b.x, b.y), b.class_id) for b in state.boxes]
    targets = {t.num_id: pos_to_grid(t.x, t.y) for t in state.targets}
    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)

    # 8-向 monkey-patch (临时)
    if dirs_8:
        old_dirs = mbs.DIRS
        mbs.DIRS = mbs.DIRS_8

    solver = MultiBoxSolver(state.grid, car, boxes, targets, bombs)
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(devnull):
            sol = solver.solve(max_cost=300, time_limit=time_limit, strategy="auto")
    finally:
        if dirs_8:
            mbs.DIRS = old_dirs

    elapsed = time.perf_counter() - t0
    if sol is None:
        return {"map": os.path.basename(map_path), "dirs": 8 if dirs_8 else 4,
                "time_s": time_limit, "status": "fail", "elapsed_s": round(elapsed, 1)}
    pushes = len(sol)
    return {"map": os.path.basename(map_path), "dirs": 8 if dirs_8 else 4,
            "time_s": time_limit, "status": "ok", "pushes": pushes,
            "elapsed_s": round(elapsed, 1)}


if __name__ == "__main__":
    maps = ["phase6_0001.txt", "phase6_0003.txt", "phase6_0006.txt",
            "phase6_0008.txt", "phase6_0013.txt"]
    for m in maps:
        for dirs8 in [False, True]:
            for tl in [60, 180]:
                r = test_one(f"assets/maps/phase6/{m}", tl, dirs8)
                print(r)
                if r["status"] == "ok":
                    break
            print("---")
