import contextlib
import io
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.rl.map_generator import generate_map
from smartcar_sokoban.solver.auto_player import AutoPlayer


TOTAL_MAPS = 1000
SEED_START = 10001
TMP_NAME = "_tmp_phase1.txt"


def clear_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in out_dir.iterdir():
        if entry.is_file():
            entry.unlink()


def is_valid_map_text(map_str: str) -> bool:
    lines = map_str.splitlines()
    if len(lines) != 12:
        return False
    allowed = {"#", "-", "$", ".", "*"}
    for line in lines:
        if len(line) != 16:
            return False
        if any(ch not in allowed for ch in line):
            return False
    return True


def main() -> None:
    cfg = GameConfig()
    cfg.render_mode = "simple"
    cfg.control_mode = "discrete"

    out_dir = ROOT / "assets" / "maps" / "phase1"
    clear_output_dir(out_dir)

    tmp_path = out_dir / TMP_NAME
    tmp_rel = tmp_path.relative_to(ROOT).as_posix()

    engine = GameEngine(cfg, str(ROOT))

    saved = 0
    attempts = 0
    seed = SEED_START

    while saved < TOTAL_MAPS:
        attempts += 1
        map_str = generate_map(n_boxes=1, n_bombs=0, wall_density=0.0, seed=seed)
        seed += 1

        if not map_str or not is_valid_map_text(map_str):
            continue

        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(map_str)

        random.seed(42)
        engine.reset(tmp_rel)
        player = AutoPlayer(engine)

        with contextlib.redirect_stdout(io.StringIO()):
            actions = player.solve()

        if not actions:
            tmp_path.unlink(missing_ok=True)
            continue

        random.seed(42)
        engine.reset(tmp_rel)
        for action in actions:
            engine.discrete_step(action)

        if engine.get_state().won:
            saved += 1
            dest = out_dir / f"phase1_{saved:04d}.txt"
            os.replace(tmp_path, dest)
            if saved % 100 == 0:
                print(f"已生成 {saved}/{TOTAL_MAPS} (尝试 {attempts})")
        else:
            tmp_path.unlink(missing_ok=True)

    print(f"完成! 共生成 {saved} 张可解地图，累计尝试 {attempts} 次")


if __name__ == "__main__":
    main()
