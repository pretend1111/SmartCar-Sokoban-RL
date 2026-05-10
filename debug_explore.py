"""Debug: compare exploration results for phase6_11."""
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.explorer import plan_exploration
from smartcar_sokoban.solver.pathfinder import pos_to_grid
import random

random.seed(42)
engine = GameEngine()
engine.reset('assets/maps/phase6/phase6_11.txt')
actions = plan_exploration(engine)
state = engine.get_state()
print(f'Explore steps: {len(actions)}')
print(f'Car pos: {pos_to_grid(state.car_x, state.car_y)}')
print(f'Seen boxes: {state.seen_box_ids}')
print(f'Seen targets: {state.seen_target_ids}')
