"""Quick test: verify no infinite bomb loop"""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ['SDL_VIDEODRIVER'] = 'dummy'

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.auto_player import AutoPlayer

cfg = GameConfig()
cfg.control_mode = 'discrete'
engine = GameEngine(cfg, ROOT)

for seed in [42, 789, 0, 1, 2, 3]:
    random.seed(seed)
    state = engine.reset('assets/maps/map3.txt')
    player = AutoPlayer(engine)
    actions = player.solve()
    state = engine.get_state()
    status = "WON" if state.won else f"FAIL(rem={len(state.boxes)},bombs={len(state.bombs)})"
    print(f"seed={seed:3d}: {status} ({len(actions)} steps)\n")
