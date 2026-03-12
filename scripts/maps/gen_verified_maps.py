"""批量生成可解地图 — 生成 + AutoPlayer 验证闭环.

每个阶段生成 1000 张保证可解的地图。
用法:
    python gen_verified_maps.py              # 生成全部 6 个阶段
    python gen_verified_maps.py --phase 3    # 只生成 Phase 3
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time
import io
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.rl.map_generator import generate_map
from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.auto_player import AutoPlayer

# 各阶段参数
PHASE_CONFIGS = {
    1: {'n_boxes': 1, 'n_bombs': 0, 'wd_min': 0.0,  'wd_max': 0.0,  'seed_base': 10001},
    2: {'n_boxes': 1, 'n_bombs': 0, 'wd_min': 0.05, 'wd_max': 0.08, 'seed_base': 20001},
    3: {'n_boxes': 2, 'n_bombs': 0, 'wd_min': 0.05, 'wd_max': 0.08, 'seed_base': 30001},
    4: {'n_boxes': 3, 'n_bombs': 0, 'wd_min': 0.08, 'wd_max': 0.10, 'seed_base': 40001},
    5: {'n_boxes': 3, 'n_bombs': 1, 'wd_min': 0.10, 'wd_max': 0.12, 'seed_base': 50001},
    6: {'n_boxes': -1, 'n_bombs': -1, 'wd_min': 0.0, 'wd_max': 0.12, 'seed_base': 60001},  # 随机
}

TARGET_COUNT = 1000


def gen_phase(phase: int):
    pcfg = PHASE_CONFIGS[phase]
    out_dir = os.path.join(ROOT, 'assets', 'maps', f'phase{phase}')
    os.makedirs(out_dir, exist_ok=True)

    # 清空旧文件
    old_files = glob.glob(os.path.join(out_dir, '*.txt'))
    for f in old_files:
        os.remove(f)
    print(f"  已清空 {len(old_files)} 个旧文件")

    cfg = GameConfig()
    cfg.render_mode = "simple"
    cfg.control_mode = "discrete"

    tmp_path = os.path.join(out_dir, '_tmp_gen.txt')
    tmp_rel = os.path.relpath(tmp_path, ROOT).replace('\\', '/')

    saved = 0
    attempts = 0
    seed = pcfg['seed_base']
    rng = random.Random(seed)
    t0 = time.perf_counter()

    while saved < TARGET_COUNT:
        attempts += 1
        cur_seed = seed
        seed += 1

        # Phase 6 随机参数
        if pcfg['n_boxes'] == -1:
            n_boxes = rng.randint(1, 3)
            wd = rng.uniform(pcfg['wd_min'], pcfg['wd_max'])
            n_bombs = rng.randint(0, 1) if wd > 0.03 else 0
        else:
            n_boxes = pcfg['n_boxes']
            n_bombs = pcfg['n_bombs']
            if pcfg['wd_min'] == pcfg['wd_max']:
                wd = pcfg['wd_min']
            else:
                wd = rng.uniform(pcfg['wd_min'], pcfg['wd_max'])

        # 1. 生成地图
        map_str = generate_map(n_boxes=n_boxes, n_bombs=n_bombs,
                               wall_density=wd, seed=cur_seed)
        if map_str is None:
            continue

        # 2. 写临时文件
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(map_str)

        # 3. 验证可解性 (抑制 solver 输出)
        try:
            engine = GameEngine(cfg, ROOT)
            random.seed(42)
            engine.reset(tmp_rel)

            devnull = io.StringIO()
            with redirect_stdout(devnull):
                player = AutoPlayer(engine)
                actions = player.solve()

            if not actions:
                continue

            # 回放验证
            random.seed(42)
            engine.reset(tmp_rel)
            for a in actions:
                engine.discrete_step(a)

            if not engine.get_state().won:
                continue

        except Exception:
            continue

        # 4. 通关！保存
        saved += 1
        dest = os.path.join(out_dir, f'phase{phase}_{saved:04d}.txt')
        if os.path.exists(dest):
            os.remove(dest)
        os.rename(tmp_path, dest)

        # 进度
        if saved % 100 == 0 or saved == 1:
            elapsed = time.perf_counter() - t0
            rate = saved / max(elapsed, 0.01)
            pct = attempts > 0 and f"{saved/attempts*100:.0f}%" or "?"
            print(f"  [{saved:4d}/{TARGET_COUNT}] "
                  f"尝试={attempts} 通过率={pct} "
                  f"速度={rate:.1f}/s 耗时={elapsed:.0f}s")

    # 清理临时文件
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    elapsed = time.perf_counter() - t0
    pass_rate = saved / max(attempts, 1) * 100
    print(f"  ✅ Phase {phase} 完成: {saved} 张可解地图, "
          f"尝试 {attempts} 次 (通过率 {pass_rate:.1f}%), "
          f"耗时 {elapsed:.1f}s")
    return saved, attempts, elapsed


def main():
    parser = argparse.ArgumentParser(description='批量生成可解推箱子地图')
    parser.add_argument('--phase', type=int, default=None,
                        help='只生成指定阶段 (1-6)')
    args = parser.parse_args()

    phases = [args.phase] if args.phase else list(range(1, 7))

    print(f"🔧 批量生成可解地图")
    print(f"   阶段: {phases}")
    print(f"   每阶段: {TARGET_COUNT} 张\n")

    total_saved = 0
    total_attempts = 0
    total_time = 0

    for p in phases:
        print(f"\n{'='*50}")
        print(f"  Phase {p}")
        print(f"{'='*50}")
        saved, attempts, elapsed = gen_phase(p)
        total_saved += saved
        total_attempts += attempts
        total_time += elapsed

    print(f"\n{'='*50}")
    print(f"  全部完成!")
    print(f"  总计: {total_saved} 张可解地图")
    print(f"  尝试: {total_attempts} 次")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
