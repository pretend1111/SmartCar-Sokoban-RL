"""单 map GIF 输出 — 用 PIL 拼接 frames."""

from __future__ import annotations

import argparse
import io
import os
import sys
import math
import random
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.min_steps.visualize import _draw_frame, run_and_collect_frames


def map_to_gif(map_path: str, seed: int, planner_fn, planner_name: str,
                 out_path: str, dpi: int = 80, fps: int = 4):
    from experiments.min_steps.planner_best import set_best_context
    set_best_context(map_path, seed)

    map_name = os.path.basename(map_path)
    ret = run_and_collect_frames(map_path, seed, planner_fn, planner_name)
    if len(ret) == 5:
        frames_data, trail, won, total, _ = ret
    else:
        frames_data, trail, won, total = ret
    print(f'  {map_name} seed={seed}: {total} steps, won={won}')

    images = []
    for i, (s, tag, step_no) in enumerate(frames_data):
        fig, ax = plt.subplots(figsize=(7, 5))
        title = f'{map_name} | {planner_name} | step {step_no}/{total}  [{tag}]'
        _draw_frame(ax, s, s.grid, trail[:i+1], title)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        images.append(Image.open(buf).convert('RGB'))

    duration_ms = int(1000 / fps)
    # 最后帧多停 2 秒
    durations = [duration_ms] * len(images)
    if durations: durations[-1] = 2000

    images[0].save(out_path, save_all=True, append_images=images[1:],
                    duration=durations, loop=0, optimize=False)
    sz_kb = os.path.getsize(out_path) / 1024
    print(f'    → {out_path} ({sz_kb:.0f} KB)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--planner', default='oracle_v6')
    ap.add_argument('--out-dir', default='/tmp/viz_v6')
    ap.add_argument('--fps', type=int, default=4)
    ap.add_argument('--maps', nargs='+', default=None)
    args = ap.parse_args()

    from experiments.min_steps.planner_oracle_v6 import planner_oracle_v6
    fn = planner_oracle_v6

    if args.maps:
        targets = []
        for spec in args.maps:
            ph, name, sd = spec.split(':')
            targets.append((f'assets/maps/{ph}/{name}', int(sd)))
    else:
        targets = [
            ('assets/maps/phase6/phase6_02.txt', 0),
            ('assets/maps/phase6/phase6_04.txt', 137),
        ]

    os.makedirs(args.out_dir, exist_ok=True)
    for mp, sd in targets:
        if not os.path.exists(mp):
            print(f'  SKIP {mp}'); continue
        name = os.path.basename(mp).replace('.txt', '')
        out = os.path.join(args.out_dir, f'{name}_seed{sd}_oracle_v6.gif')
        map_to_gif(mp, sd, fn, args.planner, out, fps=args.fps)


if __name__ == '__main__':
    main()
