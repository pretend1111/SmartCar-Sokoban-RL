"""推理可视化 — 输出可控播放 HTML.

每张地图跑 planner, 每步生成 PNG (base64), 嵌入单文件 HTML.
浏览器打开即可:
  - 滑块 scrub
  - Play/Pause
  - ±1 单步, ±10 跳步
  - 速度可调
  - 多地图切换
  - 显示当前 step / tag / 累计步数
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import math
import random
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.pathfinder import pos_to_grid
from experiments.min_steps.visualize import _draw_frame, run_and_collect_frames


def render_frame_to_b64(state, trail, step_no: int, total: int, tag: str,
                          map_name: str, planner_name: str,
                          figsize=(7, 5), dpi=80) -> str:
    """单帧渲染为 PNG base64."""
    fig, ax = plt.subplots(figsize=figsize)
    title = f'{map_name} | {planner_name} | step {step_no}/{total}  [{tag}]'
    _draw_frame(ax, state, state.grid, trail, title)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('ascii')


def collect_map_frames(map_path: str, seed: int, planner_fn, planner_name: str):
    """跑 planner, 返回 frames base64 list + 元数据."""
    map_name = os.path.basename(map_path)
    ret = run_and_collect_frames(map_path, seed, planner_fn, planner_name)
    if len(ret) == 5:
        frames, trail, won, total, gambles = ret
    else:
        frames, trail, won, total = ret; gambles = []
    print(f'    {map_name} seed={seed}: {total} steps, won={won}', flush=True)

    b64_frames = []
    tags = []
    for i, (s, tag, step_no) in enumerate(frames):
        partial_trail = trail[:i+1]
        b64 = render_frame_to_b64(s, partial_trail, step_no, total, tag,
                                     map_name, planner_name)
        b64_frames.append(b64)
        tags.append(tag)
    return {
        'map_name': map_name,
        'seed': seed,
        'total_steps': total,
        'won': won,
        'frames': b64_frames,
        'tags': tags,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SmartCar Oracle Visualization</title>
<style>
body { font-family: -apple-system, sans-serif; margin: 0; padding: 20px;
       background: #f5f5f5; }
.app { max-width: 1100px; margin: 0 auto; background: white; padding: 20px;
       border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
h1 { margin-top: 0; font-size: 20px; }
.map-selector { margin: 12px 0; }
.map-selector select { padding: 6px 12px; font-size: 14px; }
.viewer img { max-width: 100%; display: block; margin: 0 auto;
              border: 1px solid #ddd; border-radius: 4px; }
.info { margin: 10px 0; font-family: monospace; font-size: 14px;
        text-align: center; }
.tag { background: #ffe; padding: 2px 8px; border-radius: 3px;
       border: 1px solid #cc9; margin-left: 8px; }
.tag.scan-walk, .tag.scan-rot { background: #def; border-color: #9bc; }
.tag.push, .tag.push-walk { background: #efd; border-color: #9c9; }
.tag.proactive-rot { background: #fde; border-color: #c9b; }
.controls { display: flex; gap: 8px; align-items: center; justify-content: center;
            margin: 12px 0; flex-wrap: wrap; }
.controls button { padding: 8px 14px; font-size: 14px; cursor: pointer;
                   border: 1px solid #888; background: white; border-radius: 4px; }
.controls button:hover { background: #f0f0f0; }
.controls button:active { background: #ddd; }
.slider-row { display: flex; gap: 12px; align-items: center; margin: 12px 0; }
.slider-row input[type=range] { flex: 1; }
.speed { font-size: 13px; color: #666; }
.legend { margin-top: 16px; padding: 12px; background: #f9f9f9;
          border-radius: 4px; font-size: 12px; line-height: 1.6; }
.legend-item { display: inline-block; margin-right: 16px; }
.swatch { display: inline-block; width: 16px; height: 16px; margin-right: 4px;
          vertical-align: middle; border: 1px solid #888; }
</style>
</head>
<body>
<div class="app">
<h1>SmartCar 推理轨迹可视化 (oracle_v6)</h1>

<div class="map-selector">
  <label>地图: </label>
  <select id="map-select"></select>
  <span id="meta" style="margin-left:14px; font-size:13px; color:#666;"></span>
</div>

<div class="viewer">
  <img id="frame" src="" alt="frame">
</div>

<div class="info">
  <span id="step-info">step ?/?</span>
  <span id="tag-info" class="tag">?</span>
</div>

<div class="slider-row">
  <span>0</span>
  <input type="range" id="slider" min="0" max="0" value="0">
  <span id="slider-max">0</span>
</div>

<div class="controls">
  <button onclick="jump(-10)">⏪ -10</button>
  <button onclick="jump(-1)">◀ -1</button>
  <button id="play-btn" onclick="togglePlay()">▶ Play</button>
  <button onclick="jump(1)">▶ +1</button>
  <button onclick="jump(10)">⏩ +10</button>
  <span class="speed">速度:</span>
  <select id="speed-select" onchange="updateSpeed()">
    <option value="500">慢 (2 fps)</option>
    <option value="200" selected>中 (5 fps)</option>
    <option value="100">快 (10 fps)</option>
    <option value="50">极快 (20 fps)</option>
  </select>
</div>

<div class="legend">
  <div class="legend-item"><span class="swatch" style="background:#ffe0e0;border-color:crimson;border-style:dashed;"></span>未识别 box (B?)</div>
  <div class="legend-item"><span class="swatch" style="background:#ff9999;border-color:limegreen;"></span>已识别 box (class_id)</div>
  <div class="legend-item"><span class="swatch" style="background:lightyellow;border-radius:50%;border-color:royalblue;border-style:dashed;"></span>未识别 target (T?)</div>
  <div class="legend-item"><span class="swatch" style="background:lightyellow;border-radius:50%;border-color:limegreen;"></span>已识别 target (num_id)</div>
  <div class="legend-item"><span style="color:steelblue;font-weight:bold;">▲</span> 车 (朝向)</div>
  <div class="legend-item"><span style="color:red;">━</span> 累积轨迹</div>
  <div class="legend-item"><span style="color:orange;font-weight:bold;">×</span> 炸弹</div>
</div>
</div>

<script>
const DATA = __DATA__;
let curMap = 0;
let curFrame = 0;
let playing = false;
let playTimer = null;
let playInterval = 200;

function init() {
  const sel = document.getElementById('map-select');
  DATA.forEach((m, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `${m.map_name} (seed=${m.seed}, ${m.total_steps} steps, won=${m.won})`;
    sel.appendChild(opt);
  });
  sel.onchange = e => { curMap = parseInt(e.target.value); curFrame = 0; refresh(); };
  document.getElementById('slider').oninput = e => {
    curFrame = parseInt(e.target.value);
    refresh(false);
  };
  document.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft') jump(-1);
    if (e.key === 'ArrowRight') jump(1);
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  });
  refresh();
}

function refresh(updateSlider = true) {
  const m = DATA[curMap];
  const n = m.frames.length;
  if (curFrame >= n) curFrame = n - 1;
  if (curFrame < 0) curFrame = 0;
  document.getElementById('frame').src = 'data:image/png;base64,' + m.frames[curFrame];
  document.getElementById('step-info').textContent =
    `step ${curFrame + 1}/${n}`;
  const tag = m.tags[curFrame];
  const tagEl = document.getElementById('tag-info');
  tagEl.textContent = tag;
  tagEl.className = 'tag ' + tag.replace(/_/g, '-');
  document.getElementById('meta').textContent =
    `${m.total_steps} steps · won=${m.won}`;
  if (updateSlider) {
    const sl = document.getElementById('slider');
    sl.max = n - 1;
    sl.value = curFrame;
    document.getElementById('slider-max').textContent = n - 1;
  } else {
    document.getElementById('slider').value = curFrame;
  }
}

function jump(d) {
  curFrame += d;
  refresh(false);
}

function togglePlay() {
  if (playing) stopPlay();
  else startPlay();
}

function startPlay() {
  playing = true;
  document.getElementById('play-btn').textContent = '⏸ Pause';
  playTimer = setInterval(() => {
    const m = DATA[curMap];
    if (curFrame >= m.frames.length - 1) {
      stopPlay();
      return;
    }
    curFrame++;
    refresh(false);
  }, playInterval);
}

function stopPlay() {
  playing = false;
  document.getElementById('play-btn').textContent = '▶ Play';
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
}

function updateSpeed() {
  playInterval = parseInt(document.getElementById('speed-select').value);
  if (playing) { stopPlay(); startPlay(); }
}

init();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--planner', default='oracle_v18',
                     choices=['v1_explore', 'oracle_v1', 'oracle_v3b',
                              'oracle_v4', 'oracle_v6', 'oracle_v14',
                              'oracle_v18'])
    ap.add_argument('--out', default='/tmp/viz_v6/interactive.html')
    ap.add_argument('--maps', nargs='+', default=None,
                     help='list of "phase:name:seed"')
    args = ap.parse_args()

    planner_map = {
        'v1_explore': lambda: __import__('experiments.min_steps.harness',
            fromlist=['planner_v1_explore_first']).planner_v1_explore_first,
        'oracle_v1': lambda: __import__('experiments.min_steps.planner_oracle',
            fromlist=['planner_oracle']).planner_oracle,
        'oracle_v3b': lambda: __import__('experiments.min_steps.planner_oracle_v3b',
            fromlist=['planner_oracle_v3b']).planner_oracle_v3b,
        'oracle_v4': lambda: __import__('experiments.min_steps.planner_oracle_v4',
            fromlist=['planner_oracle_v4']).planner_oracle_v4,
        'oracle_v6': lambda: __import__('experiments.min_steps.planner_oracle_v6',
            fromlist=['planner_oracle_v6']).planner_oracle_v6,
        'oracle_v14': lambda: __import__('experiments.min_steps.planner_oracle_v14',
            fromlist=['planner_oracle_v14']).planner_oracle_v14,
        'oracle_v18': lambda: __import__('experiments.min_steps.planner_oracle_v18',
            fromlist=['planner_oracle_v18']).planner_oracle_v18,
    }
    from experiments.min_steps.planner_best import set_best_context
    fn = planner_map[args.planner]()

    if args.maps:
        targets = []
        for spec in args.maps:
            ph, name, sd = spec.split(':')
            targets.append((f'assets/maps/{ph}/{name}', int(sd)))
    else:
        targets = [
            ('assets/maps/phase4/phase4_05.txt', 0),
            ('assets/maps/phase5/phase5_01.txt', 0),
            ('assets/maps/phase6/phase6_02.txt', 0),
            ('assets/maps/phase6/phase6_04.txt', 137),
        ]

    print(f'Rendering {len(targets)} maps via {args.planner}...')
    data = []
    for mp, sd in targets:
        if not os.path.exists(mp):
            print(f'  SKIP {mp}'); continue
        set_best_context(mp, sd)
        d = collect_map_frames(mp, sd, fn, args.planner)
        data.append(d)

    print(f'Rendered {len(data)} maps, total '
          f'{sum(len(d["frames"]) for d in data)} frames')
    html = HTML_TEMPLATE.replace('__DATA__', json.dumps(data))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(html)
    sz_kb = os.path.getsize(args.out) / 1024
    print(f'  → {args.out} ({sz_kb:.0f} KB)')


if __name__ == '__main__':
    main()
