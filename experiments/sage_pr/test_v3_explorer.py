import contextlib
import io
import json
import multiprocessing as mp
import os
import random
import sys
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _check(args):
    map_path, seed, phase = args
    from smartcar_sokoban.engine import GameEngine
    from smartcar_sokoban.solver.explorer import exploration_complete
    from smartcar_sokoban.solver.explorer_v3 import plan_exploration_v3
    random.seed(seed)
    eng = GameEngine()
    try:
        eng.reset(map_path)
        with contextlib.redirect_stdout(io.StringIO()):
            plan_exploration_v3(eng, max_retries=15)
        if exploration_complete(eng.get_state()):
            return None
        return {"phase": phase, "map": map_path, "seed": seed}
    except Exception as e:
        return {"phase": phase, "map": map_path, "seed": seed, "err": str(e)}


if __name__ == "__main__":
    with open(os.path.join(ROOT, "runs/sage_pr/v5_4dir_fails.json")) as f:
        fails = json.load(f)
    tasks = [(x["map"], x["seed"], x["phase"]) for x in fails]
    with mp.Pool(8) as pool:
        results = list(pool.imap_unordered(_check, tasks, chunksize=2))
    still = [r for r in results if r is not None]
    print(f"v3 救回 {len(fails) - len(still)} / {len(fails)} (still fail: {len(still)})")
    print(f"still by phase: {dict(collections.Counter(x['phase'] for x in still))}")
    with open(os.path.join(ROOT, "runs/sage_pr/v5_v3_still_fails.json"), "w") as f:
        json.dump(still, f, indent=2)
