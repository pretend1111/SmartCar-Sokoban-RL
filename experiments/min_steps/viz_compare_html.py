"""v6 vs v18 并排对比 HTML — 同图同 seed, 滑块同步, 红/绿边框标 gambling."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.min_steps.visualize import _draw_frame, run_and_collect_frames


def _render_b64(state, trail, step_no, total, tag, map_name, planner_name,
                gamble_ev=None, figsize=(6.2, 4.6), dpi=80):
    fig, ax = plt.subplots(figsize=figsize)
    if gamble_ev is not None:
        cnum, is_gamble, box_known, tgt_known = gamble_ev
        box_tag = f'{cnum}' if box_known else '?'
        tgt_tag = f'{cnum}' if tgt_known else '?'
        kind = 'GAMBLE' if is_gamble else 'KNOWN'
        marker = f'  [{kind} cls={box_tag} -> num={tgt_tag}]'
        title = f'{planner_name} step {step_no}/{total} [{tag}]{marker}'
    else:
        title = f'{planner_name} step {step_no}/{total} [{tag}]'
    fp_box = getattr(state, '_fp_box_ids', None)
    fp_tgt = getattr(state, '_fp_target_ids', None)
    _draw_frame(ax, state, state.grid, trail, title,
                 fp_box_ids=fp_box, fp_target_ids=fp_tgt)
    if gamble_ev is not None:
        color = 'red' if gamble_ev[1] else 'green'
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
        # 整张图染色边框
        fig.patch.set_edgecolor(color)
        fig.patch.set_linewidth(4)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor=fig.patch.get_edgecolor())
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _collect(map_path, seed, planner_fn, planner_name):
    from experiments.min_steps.planner_best import set_best_context
    set_best_context(map_path, seed)
    map_name = os.path.basename(map_path)
    ret = run_and_collect_frames(map_path, seed, planner_fn, planner_name)
    if len(ret) == 5:
        frames, trail, won, total, gambles = ret
    else:
        frames, trail, won, total = ret; gambles = []
    gamble_at = {ev[0]: ev[1:] for ev in gambles}  # cnum, is_gamble, box_known, tgt_known
    n_g = sum(1 for ev in gambles if ev[2])
    n_consume = len(gambles)
    n_box_blind = sum(1 for ev in gambles if not ev[3])
    n_tgt_blind = sum(1 for ev in gambles if not ev[4])
    print(f'  [{planner_name}] {map_name} sd={seed}: {total} steps, '
          f'{n_g}/{n_consume} gambling (box-blind={n_box_blind}, tgt-blind={n_tgt_blind})',
          flush=True)

    b64_frames, tags, step_nos, gamble_flags, flag_labels = [], [], [], [], []
    for i, (s, tag, step_no) in enumerate(frames):
        partial_trail = trail[:i+1]
        ev = gamble_at.get(step_no)
        b64 = _render_b64(s, partial_trail, step_no, total, tag,
                          map_name, planner_name, gamble_ev=ev)
        b64_frames.append(b64)
        tags.append(tag)
        step_nos.append(step_no)
        if ev is None:
            gamble_flags.append('none')
            flag_labels.append('')
        else:
            cnum, is_gamble, box_known, tgt_known = ev
            gamble_flags.append('gamble' if is_gamble else 'known')
            box_tag = f'{cnum}' if box_known else '?'
            tgt_tag = f'{cnum}' if tgt_known else '?'
            flag_labels.append(f'cls={box_tag}->num={tgt_tag}')
    return dict(planner=planner_name, map_name=map_name, seed=seed,
                total_steps=total, won=won, n_gamble=n_g, n_consume=n_consume,
                n_box_blind=n_box_blind, n_tgt_blind=n_tgt_blind,
                frames=b64_frames, tags=tags, step_nos=step_nos,
                gamble_flags=gamble_flags, flag_labels=flag_labels)


HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>v6 vs v18 对比 — gambling 可视化</title>
<style>
body{font-family:-apple-system,sans-serif;margin:0;padding:18px;background:#f3f4f6}
.app{max-width:1500px;margin:0 auto;background:white;padding:18px;border-radius:8px;
     box-shadow:0 2px 10px rgba(0,0,0,.08)}
h1{margin:0 0 6px;font-size:20px}
.sub{color:#666;font-size:13px;margin-bottom:14px}
.map-sel{margin-bottom:12px}
.map-sel select{padding:7px 12px;font-size:14px;min-width:380px}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.panel{border:1px solid #ddd;border-radius:6px;padding:10px;background:#fafafa}
.panel h2{margin:0 0 6px;font-size:15px}
.panel .stats{font-family:monospace;font-size:12px;color:#555;margin-bottom:6px}
.panel img{width:100%;display:block;border-radius:4px}
.info{font-family:monospace;font-size:13px;margin-top:6px}
.tag{background:#ffe;padding:1px 6px;border-radius:3px;border:1px solid #cc9;
     margin-left:6px;font-size:11px}
.flag-gamble{background:#fee;border:1px solid #f33;color:#c00;padding:1px 6px;
             border-radius:3px;font-weight:bold;margin-left:6px}
.flag-known{background:#efe;border:1px solid #3a3;color:#060;padding:1px 6px;
            border-radius:3px;font-weight:bold;margin-left:6px}
.slider-row{display:flex;gap:10px;align-items:center;margin:14px 0 8px}
.slider-row input{flex:1}
.controls{display:flex;gap:8px;justify-content:center;align-items:center;
          flex-wrap:wrap;margin-top:6px}
.controls button{padding:7px 12px;cursor:pointer;border:1px solid #888;
                 background:white;border-radius:4px;font-size:13px}
.controls button:hover{background:#eee}
.legend{margin-top:12px;padding:10px;background:#f7f7f7;border-radius:4px;
        font-size:12px;line-height:1.7}
.legend-item{display:inline-block;margin-right:14px}
.summary{margin-top:10px;padding:8px 12px;background:#eef5ff;border-radius:4px;
         font-size:13px;border-left:3px solid #468}
</style>
</head>
<body>
<div class="app">
<h1>SmartCar Oracle: v6 (no penalty) vs v18 (best-of-α + λ=5) 对比</h1>
<div class="sub">同图同 seed。<b>红边框 = consumption-push 是赌博</b>: 推箱落到 target 消除瞬间, (box.class_id, target.num_id)
  中任一在 agent belief 里未知。要做到"非赌博",必须 box 侧 + target 侧都识别 (推之前)。
  <code>cls=X→num=Y</code> 显示推之前 agent 对 box class / target num 的知识 (<code>?</code> 表示未知)。</div>

<div class="map-sel">
  <label>地图: </label>
  <select id="mapsel"></select>
</div>

<div id="summary" class="summary"></div>

<div class="slider-row">
  <span>0</span>
  <input type="range" id="slider" min="0" max="0" value="0">
  <span id="slmax">0</span>
</div>

<div class="controls">
  <button onclick="jump(-10)">⏪ -10</button>
  <button onclick="jump(-1)">◀ -1</button>
  <button id="playbtn" onclick="togglePlay()">▶ Play</button>
  <button onclick="jump(1)">+1 ▶</button>
  <button onclick="jump(10)">+10 ⏩</button>
  <span style="font-size:12px;color:#666">速度:</span>
  <select id="speed" onchange="updSpeed()">
    <option value="500">慢 2fps</option>
    <option value="250" selected>中 4fps</option>
    <option value="120">快 8fps</option>
    <option value="60">极快 16fps</option>
  </select>
</div>

<div class="panels">
  <div class="panel">
    <h2>v6 (baseline, no penalty)</h2>
    <div class="stats" id="stats-l"></div>
    <img id="img-l" src="">
    <div class="info"><span id="info-l"></span></div>
  </div>
  <div class="panel">
    <h2>v18 (penalty λ=5 + per-map α)</h2>
    <div class="stats" id="stats-r"></div>
    <img id="img-r" src="">
    <div class="info"><span id="info-r"></span></div>
  </div>
</div>

<div class="legend">
  <div class="legend-item"><b>红边框</b>: 首次推这个 class 时, 该 class 在求解器的 belief 里属于 "未识别 & 非排除法可推断"。属于真·赌博。</div>
  <div class="legend-item"><b>绿边框</b>: 首次推之前已通过 scan / walk-reveal / forced pair / 排除法 拿到 class_id, 是信息驱动的 push。</div>
  <div class="legend-item">每图最多 3 个 first-push 事件 (按箱子数)。slider 在 v6 / v18 长度不同时按比例对齐。</div>
</div>
</div>

<script>
const DATA = __DATA__;
let cur = 0, frame = 0, playing = false, timer = null, interval = 250;

function init(){
  const sel = document.getElementById('mapsel');
  DATA.forEach((d,i)=>{
    const o = document.createElement('option');
    o.value = i;
    const lg = d.l.n_gamble, lf = d.l.n_consume, rg = d.r.n_gamble, rf = d.r.n_consume;
    o.textContent = `${d.map_name} sd=${d.seed}  |  v6: ${d.l.total_steps}步 ${lg}/${lf}赌  →  v18: ${d.r.total_steps}步 ${rg}/${rf}赌`;
    sel.appendChild(o);
  });
  sel.onchange = e=>{cur=+e.target.value; frame=0; refresh();};
  document.getElementById('slider').oninput = e=>{frame=+e.target.value; refresh(false);};
  document.addEventListener('keydown', e=>{
    if(e.key==='ArrowLeft') jump(-1);
    if(e.key==='ArrowRight') jump(1);
    if(e.key===' '){e.preventDefault(); togglePlay();}
  });
  refresh();
}

function maxN(){
  const d = DATA[cur];
  return Math.max(d.l.frames.length, d.r.frames.length);
}

function refresh(updSlider=true){
  const d = DATA[cur];
  const N = maxN();
  if(frame >= N) frame = N-1;
  if(frame < 0) frame = 0;
  // 按比例映射到各 planner 的 frame
  const fL = Math.min(Math.round(frame * (d.l.frames.length-1) / Math.max(1,N-1)), d.l.frames.length-1);
  const fR = Math.min(Math.round(frame * (d.r.frames.length-1) / Math.max(1,N-1)), d.r.frames.length-1);

  document.getElementById('img-l').src = 'data:image/png;base64,'+d.l.frames[fL];
  document.getElementById('img-r').src = 'data:image/png;base64,'+d.r.frames[fR];

  const flagSpan = (f, lbl)=>{
    if(f==='gamble') return ' <span class="flag-gamble">GAMBLE '+(lbl||'')+'</span>';
    if(f==='known') return ' <span class="flag-known">KNOWN '+(lbl||'')+'</span>';
    return '';
  };
  document.getElementById('info-l').innerHTML =
    `step ${d.l.step_nos[fL]}/${d.l.total_steps}<span class="tag">${d.l.tags[fL]}</span>${flagSpan(d.l.gamble_flags[fL], d.l.flag_labels[fL])}`;
  document.getElementById('info-r').innerHTML =
    `step ${d.r.step_nos[fR]}/${d.r.total_steps}<span class="tag">${d.r.tags[fR]}</span>${flagSpan(d.r.gamble_flags[fR], d.r.flag_labels[fR])}`;

  document.getElementById('stats-l').textContent =
    `${d.l.total_steps} steps · won=${d.l.won} · gambling=${d.l.n_gamble}/${d.l.n_consume} (box-blind=${d.l.n_box_blind}, tgt-blind=${d.l.n_tgt_blind})`;
  document.getElementById('stats-r').textContent =
    `${d.r.total_steps} steps · won=${d.r.won} · gambling=${d.r.n_gamble}/${d.r.n_consume} (box-blind=${d.r.n_box_blind}, tgt-blind=${d.r.n_tgt_blind})`;

  const dS = d.l.total_steps - d.r.total_steps;
  const dG = d.l.n_gamble - d.r.n_gamble;
  document.getElementById('summary').innerHTML =
    `<b>${d.map_name} sd=${d.seed}</b>: v18 比 v6 ${dS>0?'省 '+dS+' 步':dS<0?'多 '+(-dS)+' 步':'步数相同'}, 赌博次数 ${dG>0?'减少 '+dG:dG<0?'增加 '+(-dG):'相同'}。`;

  if(updSlider){
    const sl = document.getElementById('slider');
    sl.max = N-1; sl.value = frame;
    document.getElementById('slmax').textContent = N-1;
  } else {
    document.getElementById('slider').value = frame;
  }
}

function jump(d){frame += d; refresh(false);}
function togglePlay(){playing ? stop() : start();}
function start(){
  playing = true;
  document.getElementById('playbtn').textContent = '⏸ Pause';
  timer = setInterval(()=>{
    if(frame >= maxN()-1){stop(); return;}
    frame++; refresh(false);
  }, interval);
}
function stop(){
  playing = false;
  document.getElementById('playbtn').textContent = '▶ Play';
  if(timer){clearInterval(timer); timer=null;}
}
function updSpeed(){
  interval = +document.getElementById('speed').value;
  if(playing){stop(); start();}
}
init();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/viz_compare/compare_v6_v18.html')
    ap.add_argument('--maps', nargs='+', default=None,
                    help='list of "phase:name:seed"')
    args = ap.parse_args()

    from experiments.min_steps.planner_oracle_v6 import planner_oracle_v6
    from experiments.min_steps.planner_oracle_v18 import planner_oracle_v18

    if args.maps:
        targets = []
        for spec in args.maps:
            ph, name, sd = spec.split(':')
            targets.append((f'assets/maps/{ph}/{name}', int(sd)))
    else:
        # 混合 phase4/5/6, 包含炸弹图 (phase5 系列) 和无炸弹图
        targets = [
            ('assets/maps/phase4/phase4_02.txt', 999),     # 3 box, 0 bomb
            ('assets/maps/phase5/phase5_02.txt', 0),       # 2 box, 1 bomb
            ('assets/maps/phase5/phase5_05.txt', 0),       # 2 box, 1 bomb
            ('assets/maps/phase6/phase6_03.txt', 200),     # 3 box, 0 bomb
            ('assets/maps/phase6/phase6_04.txt', 137),     # 3 box, 0 bomb (still gambles)
        ]

    print(f'Rendering {len(targets)} maps × 2 planners...')
    data = []
    for mp, sd in targets:
        if not os.path.exists(mp):
            print(f'  SKIP {mp}'); continue
        l = _collect(mp, sd, planner_oracle_v6, 'v6')
        r = _collect(mp, sd, planner_oracle_v18, 'v18')
        data.append(dict(map_name=os.path.basename(mp), seed=sd, l=l, r=r))

    print(f'Done. Writing HTML...')
    html = HTML.replace('__DATA__', json.dumps(data))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(html)
    sz_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f'  → {args.out} ({sz_mb:.1f} MB)')


if __name__ == '__main__':
    main()
