"""
批量生成高质量推箱子地图
=========================
用法:
    python gen_1000_maps.py                    # 所有 phase 各 1000 张
    python gen_1000_maps.py --phase 1          # 仅 phase 1
    python gen_1000_maps.py --phase 3 --count 500
    python gen_1000_maps.py --resume           # 从上次中断处继续

输出:
    assets/maps/phase{1-6}/phase{N}_{NNNN}.txt   — 地图文件
    assets/maps/batch_manifest.json               — 种子 & 指标清单
"""
from __future__ import annotations

import functools

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

# ── 路径设置 ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import gen_quality_maps as g

# ── 常量 ──────────────────────────────────────────────────
ROWS, COLS = 12, 16
MANIFEST_PATH = os.path.join(ROOT, 'assets', 'maps', 'batch_manifest.json')

# 每个 phase 的相似度阈值 (字符差异数, 必须 >= threshold 才认为不同)
#   P1 无墙, 差异只在 box+target 位置 (2 个实体 = 至少 2 字符变化)
#   P2 有迷宫 + 单箱, 天然差异较大
#   P3+ 多箱 + 墙, 差异更大
SIMILARITY_THRESHOLDS = {
    1: 2,    # 任何实体位置变化都算不同
    2: 6,    # 迷宫 + 单箱
    3: 8,    # 迷宫 + 双箱 
    4: 10,   # BSP/紧凑 + 三箱
    5: 8,    # 紧凑 + 炸弹
    6: 8,    # 混合
}


@dataclass
class MapRecord:
    filename: str
    seed: int
    phase: int
    steps: int
    score: float
    metrics: Dict
    map_str: str = field(repr=False)


# ── 相似度检测 ────────────────────────────────────────────

def map_diff(a: str, b: str) -> int:
    """逐字符比较两张地图, 返回不同字符数量."""
    diff = 0
    for left, right in zip(a.splitlines(), b.splitlines()):
        diff += sum(1 for lc, rc in zip(left, right) if lc != rc)
    return diff


def entity_fingerprint(map_str: str) -> str:
    """提取地图中实体位置的指纹 (忽略空地/墙差异).
    
    返回: 排序后的实体位置字符串, 例如 'B3,5|B7,2|T10,8|T4,3'
    """
    parts = []
    for r, line in enumerate(map_str.splitlines()):
        for c, ch in enumerate(line):
            if ch == '$':  # box
                parts.append(f'B{c},{r}')
            elif ch == '.':  # target
                parts.append(f'T{c},{r}')
            elif ch == '*':  # bomb
                parts.append(f'X{c},{r}')
    return '|'.join(sorted(parts))


def is_too_similar(map_str: str, existing: Sequence[MapRecord],
                   threshold: int, hash_set: Set[str],
                   entity_set: Set[str]) -> bool:
    """检查新地图是否与已有地图过于相似或完全相同.

    三级过滤:
    1. MD5 完全相同检查 (O(1))
    2. 实体指纹相同检查 (O(1)) — 同样的实体位置一定太相似
    3. 逐字符差异抽样比较
    """
    # 快速完全相同检查
    h = hashlib.md5(map_str.encode()).hexdigest()
    if h in hash_set:
        return True

    # 实体指纹相同 = 实体位置完全一致 (墙可能不同但太相似)
    fp = entity_fingerprint(map_str)
    if fp in entity_set:
        return True

    n = len(existing)
    if n == 0:
        return False

    # 最近 200 张 — 连续生成的地图最容易相似
    window = min(200, n)
    for i in range(n - 1, max(-1, n - window - 1), -1):
        if map_diff(map_str, existing[i].map_str) < threshold:
            return True

    # 如果总数 > 200, 再随机抽 100 张做快速比对
    if n > 200:
        rng = random.Random(hash(map_str) & 0xFFFFFFFF)
        sample_size = min(100, n - 200)
        sample_indices = rng.sample(range(n - 200), sample_size)
        for idx in sample_indices:
            if map_diff(map_str, existing[idx].map_str) < threshold:
                return True

    return False


# ── Phase 1 扩展生成器 ────────────────────────────────────
# Phase 1 原始生成器使用空白网格, 实体位置多样性有限 (~300 张)
# 扩展版: 在空白网格和低密度墙网格之间交替, 扩大地图多样性

def make_phase1_extended(rng: random.Random):
    """Phase 1 扩展: 交替使用空白和低密度墙网格."""
    if rng.random() < 0.5:
        # 50% 原始空白网格 (和原 Phase 1 一致)
        return g.PHASE_GENERATORS[1](rng)
    else:
        # 50% 低密度墙网格 (1-3 个内墙, 增加路径多样性)
        grid = g.create_maze_grid(rng, (0.01, 0.05))
        if grid is None:
            return g.PHASE_GENERATORS[1](rng)  # 回退
        
        cells = [p for p in g.open_cells(grid, non_corner_only=True)
                 if p != g.CAR_SPAWN and g.manhattan(p, g.CAR_SPAWN) >= 5]
        if len(cells) < 4:
            return None
        target = rng.choice(cells)

        def score_fn(idx, boxes, targets, pull, actual):
            new_box = pull['new_box']
            away = g.manhattan(new_box, targets[idx]) * 10
            center = g.manhattan(new_box, (7, 5))
            edge_penalty = 20 if new_box[0] in {1, 14} or new_box[1] in {1, 10} else 0
            return away + center - edge_penalty

        pulled = g.run_reverse_pull(grid, [target], rng, 15, 36, score_fn, bonus_walks=1)
        if pulled is None:
            return None
        box = pulled['boxes'][0]
        solved = g.solve_single_box(grid, box, target)
        if solved is None or g.manhattan(box, target) < 8:
            return None
        if not (12 <= solved['steps'] <= 50) or solved['turns'] < 1:
            return None
        metrics = {
            'phase': 1, 'steps': solved['steps'], 'turns': solved['turns'],
            'manhattan': g.manhattan(box, target), 'pulls': pulled['pull_count'],
            'wall_ratio': round(g.interior_wall_ratio(grid), 3),
        }
        score = solved['steps'] + 2.5 * solved['turns'] + 1.5 * metrics['manhattan']
        return g.Candidate(g.grid_to_string(grid, [box], [target], []), score, metrics)


def make_phase3_fast(rng: random.Random):
    """Phase 3 快速版: 降低 solver 时限 (12s -> 3s), 大幅提速.
    
    大多数可解地图在 1-2s 内找到解, 不可解地图浪费整个时限.
    3s 足以找到绝大多数 2-box 解.
    """
    grid = g.create_maze_grid(rng, (0.20, 0.30))
    if grid is None:
        return None
    cells = [p for p in g.open_cells(grid, non_corner_only=True) if g.manhattan(p, g.CAR_SPAWN) >= 4]
    if len(cells) < 8:
        return None
    rng.shuffle(cells)
    target1 = target2 = None
    for a in cells:
        close = [b for b in cells if b != a and 3 <= g.manhattan(a, b) <= 6]
        if close:
            target1 = a
            target2 = rng.choice(close)
            break
    if target1 is None:
        return None
    anchors = sorted([p for p in cells if p not in {target1, target2}],
                     key=lambda p: g.manhattan(p, target1) + g.manhattan(p, target2), reverse=True)
    if len(anchors) < 2:
        return None
    anchor_map = {0: anchors[0], 1: anchors[1]}

    def score_fn(idx, boxes, targets, pull, actual):
        new_box = pull['new_box']
        return g.manhattan(new_box, targets[idx]) * 6 + max(0, 16 - g.manhattan(new_box, anchor_map[idx])) * 2

    pulled = g.run_reverse_pull(grid, [target1, target2], rng, 34, 56, score_fn, bonus_walks=2)
    if pulled is None:
        return None
    boxes = pulled['boxes']
    if len(set(boxes)) < 2 or set(boxes) & {target1, target2} or any(not g.is_non_corner(grid, box) for box in boxes):
        return None
    map_str = g.grid_to_string(grid, boxes, [target1, target2], [])
    _, parsed_boxes, parsed_targets, _ = g.parse_map_string(map_str)
    solver, solution = g.solve_multi_box(grid, parsed_boxes, parsed_targets, [],
                                         time_limit=3.0, max_cost=500)
    if solution is None or solver is None:
        return None
    steps = sum(wc + 1 for _, _, _, wc in solution)
    paths = g.trace_box_paths(parsed_boxes, solution)
    overlap = g.overlap_cells(paths)
    box_switches = g.count_box_changes(solution)
    if not (30 <= steps <= 80) or len(overlap) < 1 or box_switches < 1:
        return None
    score, parts = g.evaluate_solution(grid, solver, solution)
    metrics = {
        'phase': 3, 'steps': steps,
        'turns': g.count_turns(solver.solution_to_actions(solution)),
        'wall_ratio': round(g.interior_wall_ratio(grid), 3),
        'target_distance': g.manhattan(target1, target2),
        'path_overlap': len(overlap), 'box_changes': box_switches,
        'pulls': pulled['pull_count'],
    }
    return g.Candidate(map_str, score + len(overlap) * 8, metrics)


def make_phase4_fast(rng: random.Random):
    """Phase 4 快速版: 降低 solver 时限 (16s -> 4s)."""
    grid = g.create_maze_grid(rng, (0.25, 0.28))
    if grid is None:
        return None
    cells = [p for p in g.open_cells(grid, non_corner_only=True)
             if g.manhattan(p, g.CAR_SPAWN) >= 4]
    if len(cells) < 10:
        return None
    targets = g.pick_spaced_cells(cells, 3, rng, min_dist=3)
    if targets is None:
        return None

    def score_fn(idx, boxes, targets_l, pull, actual):
        new_box = pull['new_box']
        return g.manhattan(new_box, targets_l[idx]) * 6 + max(0, 12 - g.manhattan(new_box, (7, 5))) * 2

    pulled = g.run_reverse_pull(grid, targets, rng, 18, 34, score_fn, bonus_walks=2)
    if pulled is None:
        return None
    boxes = pulled['boxes']
    if len(set(boxes)) < 3 or set(boxes) & set(targets) or any(not g.is_non_corner(grid, box) for box in boxes):
        return None
    map_str = g.grid_to_string(grid, boxes, targets, [])
    _, parsed_boxes, parsed_targets, _ = g.parse_map_string(map_str)
    solver, solution = g.solve_multi_box(grid, parsed_boxes, parsed_targets, [],
                                         time_limit=4.0, max_cost=560)
    if solution is None or solver is None:
        return None
    steps = sum(wc + 1 for _, _, _, wc in solution)
    if not (25 <= steps <= 70):
        return None
    score, _ = g.evaluate_solution(grid, solver, solution)
    metrics = {
        'phase': 4, 'steps': steps,
        'turns': g.count_turns(solver.solution_to_actions(solution)),
        'wall_ratio': round(g.interior_wall_ratio(grid), 3),
        'pulls': pulled['pull_count'],
    }
    return g.Candidate(map_str, score, metrics)


def make_phase5_compact_nobomb(rng: random.Random):
    """Phase 5 紧凑双箱 (无炸弹) — 快速路线."""
    grid = g.create_compact_grid(rng, (38, 52))
    if grid is None:
        return None
    cells = [p for p in g.open_cells(grid, non_corner_only=True)
             if p != g.CAR_SPAWN and g.manhattan(p, g.CAR_SPAWN) >= 3]
    if len(cells) < 8:
        return None
    targets = g.pick_spaced_cells(cells, 2, rng, min_dist=2)
    if targets is None:
        return None

    def score_fn(idx, boxes, targets_l, pull, actual):
        new_box = pull['new_box']
        return g.manhattan(new_box, targets_l[idx]) * 5 + max(0, 10 - g.manhattan(new_box, (7, 5))) * 2

    pulled = g.run_reverse_pull(grid, targets, rng, 18, 34, score_fn, bonus_walks=2)
    if pulled is None:
        return None
    boxes = pulled['boxes']
    if len(set(boxes)) < 2 or set(boxes) & set(targets) or any(not g.is_non_corner(grid, box) for box in boxes):
        return None
    map_str = g.grid_to_string(grid, boxes, targets, [])
    _, parsed_boxes, parsed_targets, _ = g.parse_map_string(map_str)
    solver, solution = g.solve_multi_box(grid, parsed_boxes, parsed_targets, [],
                                         time_limit=4.0, max_cost=500)
    if solution is None or solver is None:
        return None
    steps = sum(wc + 1 for _, _, _, wc in solution)
    if not (25 <= steps <= 65):
        return None
    wr = round(g.interior_wall_ratio(grid), 3)
    if not (0.50 <= wr <= 0.72):
        return None
    score, _ = g.evaluate_solution(grid, solver, solution)
    metrics = {
        'phase': 5, 'steps': steps,
        'turns': g.count_turns(solver.solution_to_actions(solution)),
        'wall_ratio': wr, 'bomb_used': False,
    }
    return g.Candidate(map_str, score, metrics)


def make_phase5_bomb(rng: random.Random):
    """Phase 5 炸弹版 (较慢, 需要验证炸弹必要性)."""
    grid = g.create_compact_grid(rng, (38, 50))
    if grid is None:
        return None
    cells = [p for p in g.open_cells(grid, non_corner_only=True) if p != g.CAR_SPAWN]
    if len(cells) < 8:
        return None
    boxes = g.pick_spaced_cells(cells, 2, rng, min_dist=2)
    if boxes is None:
        return None
    rem = [p for p in cells if p not in boxes]
    targets = g.pick_spaced_cells(rem, 2, rng, min_dist=2)
    if targets is None:
        return None
    rem = [p for p in rem if p not in targets]
    if not rem:
        return None
    bomb = rng.choice(rem)
    map_str = g.grid_to_string(grid, boxes, targets, [bomb])
    _, parsed_boxes, parsed_targets, parsed_bombs = g.parse_map_string(map_str)
    solver, solution = g.solve_multi_box(grid, parsed_boxes, parsed_targets, parsed_bombs,
                                         time_limit=2.0, max_cost=460)
    if solution is None:
        return None
    bomb_used = any(etype == 'bomb' for etype, _, _, _ in solution)
    if not bomb_used:
        return None
    # 快速检查无炸弹不可解
    _, no_bomb = g.solve_multi_box(grid, parsed_boxes, parsed_targets, [],
                                    time_limit=1.0, max_cost=460)
    if no_bomb is not None:
        return None
    steps = sum(wc + 1 for _, _, _, wc in solution)
    if not (30 <= steps <= 65):
        return None
    wr = round(g.interior_wall_ratio(grid), 3)
    if not (0.55 <= wr <= 0.72):
        return None
    metrics = {
        'phase': 5, 'steps': steps,
        'wall_ratio': wr, 'bomb_used': True,
    }
    score = steps + 45.0 * wr
    return g.Candidate(map_str, score, metrics)


def make_phase5_fast(rng: random.Random):
    """Phase 5 混合: 70% 紧凑双箱 (快) + 30% 炸弹版 (慢但多样)."""
    if rng.random() < 0.7:
        return make_phase5_compact_nobomb(rng)
    else:
        return make_phase5_bomb(rng)


def make_phase6_fast(rng: random.Random):
    """Phase 6 混合: 50% open 三箱 + 50% 紧凑双箱."""
    if rng.random() < 0.5:
        return make_phase4_fast(rng)
    else:
        return make_phase5_compact_nobomb(rng)


# 扩展后的 Phase 生成器映射
EXTENDED_GENERATORS = {
    1: make_phase1_extended,
    3: make_phase3_fast,
    4: make_phase4_fast,
    5: make_phase5_fast,
    6: make_phase6_fast,
}


# ── Phase 生成器调度 ──────────────────────────────────────

def generate_one(phase: int, seed: int) -> Optional[MapRecord]:
    """用给定种子尝试生成一张 phase 地图, 返回 MapRecord 或 None."""
    rng = random.Random(seed)
    
    # 优先使用扩展生成器 (更多样化)
    generator = EXTENDED_GENERATORS.get(phase, g.PHASE_GENERATORS.get(phase))
    if generator is None:
        return None

    candidate = generator(rng)
    if candidate is None:
        return None

    return MapRecord(
        filename='',
        seed=seed,
        phase=phase,
        steps=candidate.metrics.get('steps', 0),
        score=candidate.score,
        metrics=candidate.metrics,
        map_str=candidate.map_str,
    )


# ── 批量生成 ──────────────────────────────────────────────

def load_manifest() -> Dict:
    """加载已有的 manifest (用于断点续生)."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    return {}


def save_manifest(manifest: Dict) -> None:
    """保存 manifest JSON."""
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def load_existing_records(phase: int, manifest: Dict) -> Tuple[List[MapRecord], int, Set[str], Set[str]]:
    """从 manifest 加载已有地图记录, 返回 (records, next_seed, hash_set, entity_set)."""
    phase_key = f'phase{phase}'
    records = []
    hash_set: Set[str] = set()
    entity_set: Set[str] = set()
    next_seed = 0

    if phase_key not in manifest:
        return records, 0, hash_set, entity_set

    phase_data = manifest[phase_key]
    map_dir = os.path.join(ROOT, 'assets', 'maps', f'phase{phase}')

    for entry in phase_data.get('maps', []):
        fname = entry['filename']
        map_path = os.path.join(map_dir, fname)
        if os.path.exists(map_path):
            with open(map_path, 'r', encoding='utf-8') as fh:
                map_str = fh.read()
            records.append(MapRecord(
                filename=fname,
                seed=entry['seed'],
                phase=phase,
                steps=entry.get('steps', 0),
                score=entry.get('score', 0),
                metrics=entry.get('metrics', {}),
                map_str=map_str,
            ))
            hash_set.add(hashlib.md5(map_str.encode()).hexdigest())
            entity_set.add(entity_fingerprint(map_str))
            next_seed = max(next_seed, entry['seed'] + 1)

    return records, next_seed, hash_set, entity_set


def generate_phase_batch(phase: int, count: int, resume: bool = False):
    """为指定 phase 生成 count 张地图."""

    manifest = load_manifest() if resume else {}
    if resume:
        records, start_seed, hash_set, entity_set = load_existing_records(phase, manifest)
    else:
        records, start_seed, hash_set, entity_set = [], 0, set(), set()

    map_dir = os.path.join(ROOT, 'assets', 'maps', f'phase{phase}')
    os.makedirs(map_dir, exist_ok=True)

    threshold = SIMILARITY_THRESHOLDS[phase]
    already = len(records)
    needed = count - already

    if needed <= 0:
        print(f"  Phase {phase}: 已有 {already} 张, 不需要再生成")
        return records

    print(f"  Phase {phase}: 已有 {already} 张, 还需 {needed} 张")
    print(f"  相似度阈值: ≥{threshold} 字符差异")

    # 最大尝试次数: 给足够空间, 但不无限
    # P1/P2 速度快但相似度高、P5/P6 耗时高但天然多样
    max_total_attempts = max(count * 500, 200_000)

    seed = start_seed
    attempts = 0
    gen_failures = 0
    sim_rejects = 0
    phase_start = time.perf_counter()
    last_print_time = phase_start

    while len(records) < count and attempts < max_total_attempts:
        attempts += 1
        seed += 1

        record = generate_one(phase, seed)
        if record is None:
            gen_failures += 1
            continue

        # 相似度 + 去重检查
        if is_too_similar(record.map_str, records, threshold, hash_set, entity_set):
            sim_rejects += 1
            continue

        # 通过! 保存
        h = hashlib.md5(record.map_str.encode()).hexdigest()
        hash_set.add(h)
        entity_set.add(entity_fingerprint(record.map_str))

        idx = len(records) + 1
        fname = f'phase{phase}_{idx:04d}.txt'
        record.filename = fname

        map_path = os.path.join(map_dir, fname)
        with open(map_path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(record.map_str)

        records.append(record)

        elapsed = time.perf_counter() - phase_start
        rate = idx / elapsed if elapsed > 0 else 0
        eta = (count - idx) / rate if rate > 0 else 0

        now = time.perf_counter()
        should_print = (
            idx <= 5 or
            idx % 10 == 0 or
            idx == count or
            now - last_print_time >= 30.0
        )
        if should_print:
            last_print_time = now
            print(
                f"  [{idx:4d}/{count}] seed={seed:>8d}  "
                f"steps={record.steps:>3d}  score={record.score:5.1f}  "
                f"att={attempts:>7d}  "
                f"fail/sim={gen_failures}/{sim_rejects}  "
                f"{rate:.1f}/s  ETA {format_eta(eta)}"
            )

        # 每 50 张自动保存 manifest (防中断丢失)
        if idx % 50 == 0:
            _save_phase_manifest(phase, records, manifest)
            save_manifest(manifest)

    # 最终保存
    _save_phase_manifest(phase, records, manifest)
    save_manifest(manifest)

    elapsed = time.perf_counter() - phase_start
    print(
        f"  ✅ Phase {phase} 完成: {len(records)}/{count}  "
        f"总尝试 {attempts}  生成失败 {gen_failures}  相似拒绝 {sim_rejects}  "
        f"耗时 {format_eta(elapsed)}"
    )

    return records


def format_eta(seconds: float) -> str:
    """格式化时间为 hh:mm:ss 或 mm:ss."""
    seconds = int(seconds)
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}h{m:02d}m{s:02d}s"
    elif seconds >= 60:
        m = seconds // 60
        s = seconds % 60
        return f"{m}m{s:02d}s"
    else:
        return f"{seconds}s"


def _save_phase_manifest(phase: int, records: List[MapRecord], manifest: Dict):
    """将 phase 的记录写入 manifest 字典."""
    phase_key = f'phase{phase}'

    # 统计步数分布
    step_values = [r.steps for r in records]
    step_stats = {}
    if step_values:
        step_stats = {
            'min': min(step_values),
            'max': max(step_values),
            'avg': round(sum(step_values) / len(step_values), 1),
        }

    manifest[phase_key] = {
        'count': len(records),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'similarity_threshold': SIMILARITY_THRESHOLDS[phase],
        'step_stats': step_stats,
        'maps': [
            {
                'filename': r.filename,
                'seed': r.seed,
                'steps': r.steps,
                'score': round(r.score, 2),
                'metrics': {k: v for k, v in r.metrics.items()
                           if k not in ('score_parts',)},
            }
            for r in records
        ],
        'seed_summary': {
            'min_seed': min(r.seed for r in records) if records else 0,
            'max_seed': max(r.seed for r in records) if records else 0,
            'total_unique_seeds': len(records),
            'all_seeds': [r.seed for r in records],
        },
    }


# ── 主入口 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='批量生成高质量推箱子地图 (每 phase 1000 张)')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4, 5, 6],
                        help='仅生成指定 phase')
    parser.add_argument('--count', type=int, default=1000,
                        help='每 phase 生成数量 (默认 1000)')
    parser.add_argument('--resume', action='store_true',
                        help='从上次中断处继续生成')
    args = parser.parse_args()

    phases = [args.phase] if args.phase else [1, 2, 3, 4, 5, 6]

    print('=' * 60)
    print('  推箱子地图批量生成器')
    print(f'  Phases: {phases}')
    print(f'  每 phase: {args.count} 张')
    print(f'  续传: {"是" if args.resume else "否"}')
    print(f'  输出: assets/maps/phase{{N}}/')
    print(f'  清单: {MANIFEST_PATH}')
    print('=' * 60)

    total_start = time.perf_counter()
    summary = {}

    for phase in phases:
        print(f"\n{'─' * 60}")
        print(f"  Phase {phase}")
        print(f"{'─' * 60}")

        records = generate_phase_batch(phase, args.count, args.resume)
        summary[f'phase{phase}'] = len(records)

    total_elapsed = time.perf_counter() - total_start

    print(f"\n{'=' * 60}")
    print(f"  全部完成!  总耗时 {format_eta(total_elapsed)}")
    for k, v in summary.items():
        print(f"    {k}: {v} 张")
    print(f"  清单: {MANIFEST_PATH}")
    print('=' * 60)


if __name__ == '__main__':
    # 强制 print 立即刷新 (解决非交互终端缓冲问题)
    import builtins
    builtins.print = functools.partial(builtins.print, flush=True)
    main()
