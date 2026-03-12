"""批量验证所有训练地图是否可解.

解析地图文件 → 构造 grid/boxes/targets/bombs → 调用 MultiBoxSolver 求解.
车初始位置固定在 (1, 6).
"""
import os
import sys
import time
import glob

# 确保能导入 solver
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver

# 地图字符含义
CHAR_WALL = '#'
CHAR_EMPTY = '-'
CHAR_BOX = '$'
CHAR_TARGET = '.'
CHAR_BOMB = '*'

CAR_START = (1, 6)  # (col, row)


def parse_map(filepath):
    """解析地图文件, 返回 (grid, boxes, targets, bombs)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    lines = content.split('\n')
    rows = len(lines)
    cols = len(lines[0]) if lines else 0

    grid = []
    boxes = []
    targets = []
    bombs = []
    box_id = 0
    target_id = 0

    for r, line in enumerate(lines):
        row = []
        for c, ch in enumerate(line):
            if ch == CHAR_WALL:
                row.append(1)
            elif ch == CHAR_EMPTY:
                row.append(0)
            elif ch == CHAR_BOX:
                row.append(0)
                boxes.append(((c, r), box_id))
                box_id += 1
            elif ch == CHAR_TARGET:
                row.append(0)
                targets.append((target_id, (c, r)))
                target_id += 1
            elif ch == CHAR_BOMB:
                row.append(0)
                bombs.append((c, r))
            else:
                row.append(0)
        grid.append(row)

    # 匹配箱子和目标: 按顺序一一配对
    # 将 box_id 与 target_id 匹配
    matched_boxes = []
    matched_targets = {}
    for i in range(min(len(boxes), len(targets))):
        pos, _ = boxes[i]
        tid, tpos = targets[i]
        matched_boxes.append((pos, i))
        matched_targets[i] = tpos

    return grid, matched_boxes, matched_targets, bombs


def verify_map(filepath, time_limit=60.0):
    """验证单张地图是否可解."""
    name = os.path.basename(filepath)
    grid, boxes, targets, bombs = parse_map(filepath)

    if not boxes:
        return name, False, "没有箱子"

    if len(boxes) != len(targets):
        return name, False, f"箱={len(boxes)} 目标={len(targets)} 不匹配"

    solver = MultiBoxSolver(
        grid=grid,
        car_pos=CAR_START,
        boxes=boxes,
        targets=targets,
        bombs=bombs,
    )

    t0 = time.perf_counter()
    solution = solver.solve(max_cost=500, time_limit=time_limit)
    elapsed = time.perf_counter() - t0

    if solution is not None:
        total_steps = sum(wc + 1 for _, _, _, wc in solution)
        return name, True, f"✅ {len(solution)}推 {total_steps}步 {elapsed:.1f}s"
    else:
        return name, False, f"❌ 无解 ({elapsed:.1f}s)"


def main():
    phases = [
        path for path in sorted(glob.glob(os.path.join(ROOT, 'assets', 'maps', 'phase*')))
        if os.path.isdir(path)
    ]

    total_ok = 0
    total_fail = 0
    failed_maps = []

    for phase_dir in phases:
        phase_name = os.path.basename(phase_dir)
        map_files = sorted(glob.glob(os.path.join(phase_dir, '*.txt')))
        print(f"\n{'='*60}")
        print(f"  {phase_name}: {len(map_files)} 张地图")
        print(f"{'='*60}")

        for mf in map_files:
            name, ok, msg = verify_map(mf, time_limit=60.0)
            print(f"  {msg}  ← {name}")
            if ok:
                total_ok += 1
            else:
                total_fail += 1
                failed_maps.append(name)

    print(f"\n{'='*60}")
    print(f"  结果: ✅ {total_ok} 张可解, ❌ {total_fail} 张不可解")
    if failed_maps:
        print(f"  不可解地图: {', '.join(failed_maps)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
