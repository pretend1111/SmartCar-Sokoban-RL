"""从已有地图文件重建 batch_manifest.json (不重新生成地图)."""
import hashlib
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.maps import gen_quality_maps as g

MANIFEST_PATH = os.path.join(ROOT, 'assets', 'maps', 'batch_manifest.json')


def rebuild():
    manifest = {}
    
    for phase in range(1, 7):
        map_dir = os.path.join(ROOT, 'assets', 'maps', f'phase{phase}')
        if not os.path.isdir(map_dir):
            print(f"Phase {phase}: no directory, skipping")
            continue
        
        files = sorted([f for f in os.listdir(map_dir) if f.endswith('.txt')])
        print(f"Phase {phase}: found {len(files)} files, scanning...")
        
        maps = []
        for fname in files:
            fpath = os.path.join(map_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as fh:
                map_str = fh.read()
            
            # 从文件名提取编号
            m = re.match(r'phase\d+_(\d+)\.txt', fname)
            idx = int(m.group(1)) if m else 0
            
            # 尝试解析地图获取基本 metrics
            try:
                grid, boxes, targets, bombs = g.parse_map_string(map_str.strip())
                steps = 0  # 不重新求解
                metrics = {
                    'phase': phase,
                    'boxes': len(boxes),
                    'targets': len(targets),
                    'bombs': len(bombs),
                    'wall_ratio': round(g.interior_wall_ratio(grid), 3),
                }
            except Exception:
                metrics = {'phase': phase}
            
            maps.append({
                'filename': fname,
                'seed': idx,  # 用编号作为 seed 占位
                'steps': 0,
                'score': 0,
                'metrics': metrics,
            })
        
        # 统计
        phase_key = f'phase{phase}'
        manifest[phase_key] = {
            'count': len(maps),
            'maps': maps,
        }
        print(f"  -> {len(maps)} maps recorded in manifest")
    
    # 保存
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    
    print(f"\nManifest saved to: {MANIFEST_PATH}")
    print("Summary:")
    for phase in range(1, 7):
        key = f'phase{phase}'
        if key in manifest:
            print(f"  {key}: {manifest[key]['count']} maps")


if __name__ == '__main__':
    rebuild()
