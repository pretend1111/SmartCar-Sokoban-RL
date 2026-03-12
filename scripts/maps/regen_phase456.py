from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import gen_quality_maps as g

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED_SCAN_LIMIT = 256


@dataclass
class Candidate:
    map_str: str
    phase: int
    style: str
    family: str
    steps: int
    wall_ratio: float
    open_count: int
    boxes: int
    bombs: int
    bomb_used: bool
    score: float


def add_segment(grid, x1: int, y1: int, x2: int, y2: int) -> None:
    if x1 == x2:
        step = 1 if y2 >= y1 else -1
        for y in range(y1, y2 + step, step):
            grid[y][x1] = 1
        return
    if y1 == y2:
        step = 1 if x2 >= x1 else -1
        for x in range(x1, x2 + step, step):
            grid[y1][x] = 1


SPARSE_FAMILIES: Dict[str, List[Tuple[int, int, int, int]]] = {
    'offset': [
        (2, 1, 2, 3),
        (5, 2, 5, 4),
        (8, 1, 8, 3),
        (11, 3, 11, 5),
        (3, 7, 5, 7),
        (8, 8, 10, 8),
        (12, 6, 12, 8),
        (4, 10, 6, 10),
    ],
    'spine': [
        (7, 1, 7, 3),
        (7, 5, 7, 8),
        (4, 2, 5, 2),
        (9, 3, 11, 3),
        (3, 8, 5, 8),
        (9, 9, 12, 9),
    ],
    'islands': [
        (3, 2, 4, 2),
        (3, 2, 3, 4),
        (10, 2, 12, 2),
        (12, 2, 12, 4),
        (5, 7, 6, 7),
        (6, 7, 6, 9),
        (10, 8, 12, 8),
        (10, 8, 10, 10),
    ],
    'zigzag': [
        (2, 2, 4, 2),
        (4, 2, 4, 4),
        (6, 4, 8, 4),
        (8, 4, 8, 6),
        (9, 7, 11, 7),
        (11, 7, 11, 9),
        (4, 8, 6, 8),
        (2, 9, 2, 10),
    ],
}


PHASE5_GADGETS = [
    (
        'gadget_a',
        '\n'.join(
            [
                '################',
                '#--------------#',
                '#--------------#',
                '#-------##-----#',
                '#-------##-----#',
                '#------#-$-----#',
                '#------###-----#',
                '#------*.-#----#',
                '#------#-------#',
                '#--------------#',
                '#--------------#',
                '################',
            ]
        ),
    ),
    (
        'gadget_b',
        '\n'.join(
            [
                '################',
                '#--------------#',
                '#--------------#',
                '#------###-----#',
                '#---------#----#',
                '#-----#-*#-----#',
                '#------#$##----#',
                '#------.-#-----#',
                '#-----#-#------#',
                '#--------------#',
                '#--------------#',
                '################',
            ]
        ),
    ),
    (
        'gadget_c',
        '\n'.join(
            [
                '################',
                '#--------------#',
                '#--------------#',
                '#-----#-#------#',
                '#------#-$-----#',
                '#------####----#',
                '#------.*-#----#',
                '#-----##-------#',
                '#-------#-#----#',
                '#--------------#',
                '#--------------#',
                '################',
            ]
        ),
    ),
    (
        'gadget_d',
        '\n'.join(
            [
                '################',
                '#--------------#',
                '#--------------#',
                '#-------#-#----#',
                '#------$-#-----#',
                '#------##------#',
                '#-------*.-----#',
                '#-----##--#----#',
                '#-----#####----#',
                '#--------------#',
                '#--------------#',
                '################',
            ]
        ),
    ),
]


EXTRA_PAIRS = [
    ((2, 10), (4, 10)),
    ((2, 9), (4, 9)),
    ((3, 9), (5, 9)),
    ((2, 2), (4, 2)),
    ((3, 2), (5, 2)),
    ((2, 1), (4, 1)),
]


def create_sparse_scaffold(rng: random.Random, extra_range: Tuple[int, int]) -> Tuple[List[List[int]], str]:
    grid = g.blank_grid(0)
    family = rng.choice(sorted(SPARSE_FAMILIES))
    for seg in SPARSE_FAMILIES[family]:
        add_segment(grid, *seg)
    for _ in range(rng.randint(*extra_range)):
        if rng.random() < 0.55:
            x = rng.randint(2, 12)
            y = rng.randint(2, 9)
            horizontal = rng.random() < 0.5
            cells = [(x + i, y) for i in range(2)] if horizontal else [(x, y + i) for i in range(2)]
        else:
            cells = [(rng.randint(2, 13), rng.randint(1, 10))]
        old = []
        valid = True
        for c, r in cells:
            if not (1 <= c < g.COLS - 1 and 1 <= r < g.ROWS - 1):
                valid = False
                break
            if (c, r) == g.CAR_SPAWN:
                valid = False
                break
            old.append((c, r, grid[r][c]))
        if not valid:
            continue
        for c, r, _ in old:
            grid[r][c] = 1
        if not g.is_all_connected(grid):
            for c, r, value in old:
                grid[r][c] = value
    grid[g.CAR_SPAWN[1]][g.CAR_SPAWN[0]] = 0
    grid[g.CAR_SPAWN[1]][g.CAR_SPAWN[0] + 1] = 0
    return grid, family


def identity_seed_lists(limit: int = SEED_SCAN_LIMIT) -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = {}
    for n in (2, 3):
        seeds = []
        for seed in range(limit):
            ids = list(range(n))
            rr = random.Random(seed)
            rr.shuffle(ids)
            box_ids = ids[:]
            rr.shuffle(ids)
            target_ids = ids[:]
            perm = tuple(target_ids.index(box_ids[i]) for i in range(n))
            if perm == tuple(range(n)):
                seeds.append(seed)
        result[n] = seeds
    return result


IDENTITY_SEEDS = identity_seed_lists()


def map_diff(a: str, b: str) -> int:
    diff = 0
    for left, right in zip(a.splitlines(), b.splitlines()):
        diff += sum(1 for lc, rc in zip(left, right) if lc != rc)
    return diff


def too_similar(map_str: str, existing: Sequence[Candidate], threshold: int) -> bool:
    return any(map_diff(map_str, item.map_str) < threshold for item in existing)


def quick_metrics(
    map_str: str,
    *,
    require_bomb: bool,
    with_bomb_time: float,
    with_bomb_cost: int,
    no_bomb_time: float = 1.0,
) -> Optional[Dict[str, object]]:
    grid, boxes, targets, bombs = g.parse_map_string(map_str)
    if len(boxes) != len(targets):
        return None
    solver, solution = g.solve_multi_box(
        grid,
        boxes,
        targets,
        bombs,
        time_limit=with_bomb_time,
        max_cost=with_bomb_cost,
    )
    if solution is None:
        return None
    bomb_used = any(etype == 'bomb' for etype, _, _, _ in solution)
    if require_bomb and (not bombs or not bomb_used):
        return None
    if require_bomb:
        _, no_bomb = g.solve_multi_box(
            grid,
            boxes,
            targets,
            [],
            time_limit=no_bomb_time,
            max_cost=with_bomb_cost,
        )
        if no_bomb is not None:
            return None
    steps = sum(wc + 1 for _, _, _, wc in solution)
    return {
        'grid': grid,
        'boxes': boxes,
        'targets': targets,
        'bombs': bombs,
        'steps': steps,
        'bomb_used': bomb_used,
        'wall_ratio': round(g.interior_wall_ratio(grid), 3),
        'open_count': g.interior_open_count(grid),
    }


def make_phase4_open_candidate(rng: random.Random) -> Optional[Candidate]:
    for _ in range(120):
        grid, family = create_sparse_scaffold(rng, (2, 4))
        cells = [
            p
            for p in g.open_cells(grid, non_corner_only=True)
            if p != g.CAR_SPAWN and g.manhattan(p, g.CAR_SPAWN) >= 4
        ]
        if len(cells) < 8:
            continue
        targets = g.pick_spaced_cells(cells, 3, rng, min_dist=3)
        if targets is None:
            continue

        def score_fn(idx, boxes, targets_local, pull, actual):
            new_box = pull['new_box']
            return g.manhattan(new_box, targets_local[idx]) * 6 + max(0, 12 - g.manhattan(new_box, (7, 5))) * 2

        pulled = g.run_reverse_pull(grid, targets, rng, 18, 32, score_fn, bonus_walks=2)
        if pulled is None:
            continue
        boxes = pulled['boxes']
        if len(set(boxes)) < 3 or set(boxes) & set(targets) or any(not g.is_non_corner(grid, box) for box in boxes):
            continue
        map_str = g.grid_to_string(grid, boxes, targets, [])
        metrics = quick_metrics(map_str, require_bomb=False, with_bomb_time=4.5, with_bomb_cost=560)
        if metrics is None:
            continue
        if not (28 <= metrics['steps'] <= 60):
            continue
        if not (0.14 <= metrics['wall_ratio'] <= 0.30):
            continue
        return Candidate(
            map_str=map_str,
            phase=4,
            style='open',
            family=family,
            steps=metrics['steps'],
            wall_ratio=metrics['wall_ratio'],
            open_count=metrics['open_count'],
            boxes=len(metrics['boxes']),
            bombs=len(metrics['bombs']),
            bomb_used=bool(metrics['bomb_used']),
            score=metrics['steps'] + metrics['open_count'] * 0.1,
        )
    return None


def make_phase4_compact_candidate(rng: random.Random) -> Optional[Candidate]:
    for _ in range(220):
        grid = g.create_compact_grid(rng, (42, 58))
        if grid is None:
            continue
        cells = [
            p
            for p in g.open_cells(grid, non_corner_only=True)
            if p != g.CAR_SPAWN and g.manhattan(p, g.CAR_SPAWN) >= 3
        ]
        if len(cells) < 8:
            continue
        targets = g.pick_spaced_cells(cells, 3, rng, min_dist=2)
        if targets is None:
            continue

        def score_fn(idx, boxes, targets_local, pull, actual):
            new_box = pull['new_box']
            return g.manhattan(new_box, targets_local[idx]) * 5 + max(0, 10 - g.manhattan(new_box, (7, 5))) * 2

        pulled = g.run_reverse_pull(grid, targets, rng, 18, 30, score_fn, bonus_walks=2)
        if pulled is None:
            continue
        boxes = pulled['boxes']
        if len(set(boxes)) < 3 or set(boxes) & set(targets) or any(not g.is_non_corner(grid, box) for box in boxes):
            continue
        map_str = g.grid_to_string(grid, boxes, targets, [])
        metrics = quick_metrics(map_str, require_bomb=False, with_bomb_time=6.0, with_bomb_cost=560)
        if metrics is None:
            continue
        if not (36 <= metrics['steps'] <= 60):
            continue
        if not (0.58 <= metrics['wall_ratio'] <= 0.72):
            continue
        return Candidate(
            map_str=map_str,
            phase=4,
            style='compact',
            family='compact_reverse_pull',
            steps=metrics['steps'],
            wall_ratio=metrics['wall_ratio'],
            open_count=metrics['open_count'],
            boxes=len(metrics['boxes']),
            bombs=len(metrics['bombs']),
            bomb_used=bool(metrics['bomb_used']),
            score=metrics['steps'] + 40.0 * metrics['wall_ratio'],
        )
    return None


def mutate_phase5_open_grid(
    rng: random.Random,
    base_grid: List[List[int]],
    protected: Sequence[Tuple[int, int]],
) -> List[List[int]]:
    grid = g.copy_grid(base_grid)
    protected_set = set(protected) | {g.CAR_SPAWN, (g.CAR_SPAWN[0] + 1, g.CAR_SPAWN[1])}
    for _ in range(rng.randint(2, 6)):
        if rng.random() < 0.6:
            x = rng.randint(2, 13)
            y = rng.randint(1, 10)
            horizontal = rng.random() < 0.5
            cells = [(x + i, y) for i in range(2)] if horizontal else [(x, y + i) for i in range(2)]
        else:
            cells = [(rng.randint(2, 13), rng.randint(1, 10))]
        old = []
        valid = True
        for c, r in cells:
            if not (1 <= c < g.COLS - 1 and 1 <= r < g.ROWS - 1):
                valid = False
                break
            if (c, r) in protected_set:
                valid = False
                break
            old.append((c, r, grid[r][c]))
        if not valid:
            continue
        for c, r, _ in old:
            grid[r][c] = 1
        if not g.is_all_connected(grid):
            for c, r, value in old:
                grid[r][c] = value
    return grid


def make_phase5_open_candidate(rng: random.Random) -> Optional[Candidate]:
    for _ in range(140):
        family, base_map = rng.choice(PHASE5_GADGETS)
        base_grid, base_boxes, base_targets, bombs = g.parse_map_string(base_map)
        protected = list(base_boxes) + list(base_targets) + list(bombs)
        grid = mutate_phase5_open_grid(rng, base_grid, protected)
        pair_options = list(EXTRA_PAIRS)
        rng.shuffle(pair_options)
        for extra_box, extra_target in pair_options:
            if not g.is_open(grid, extra_box) or not g.is_open(grid, extra_target):
                continue
            boxes = [extra_box] + base_boxes
            targets = [extra_target] + base_targets
            if set(boxes) & set(targets):
                continue
            map_str = g.grid_to_string(grid, boxes, targets, bombs)
            metrics = quick_metrics(
                map_str,
                require_bomb=True,
                with_bomb_time=4.0,
                with_bomb_cost=800,
                no_bomb_time=1.1,
            )
            if metrics is None:
                continue
            if not (30 <= metrics['steps'] <= 60):
                continue
            if not (0.06 <= metrics['wall_ratio'] <= 0.16):
                continue
            return Candidate(
                map_str=map_str,
                phase=5,
                style='open',
                family=family,
                steps=metrics['steps'],
                wall_ratio=metrics['wall_ratio'],
                open_count=metrics['open_count'],
                boxes=len(metrics['boxes']),
                bombs=len(metrics['bombs']),
                bomb_used=bool(metrics['bomb_used']),
                score=metrics['steps'] + metrics['open_count'] * 0.08,
            )
    return None


def make_phase5_compact_candidate(rng: random.Random) -> Optional[Candidate]:
    for _ in range(320):
        grid = g.create_compact_grid(rng, (38, 50))
        if grid is None:
            continue
        cells = [p for p in g.open_cells(grid, non_corner_only=True) if p != g.CAR_SPAWN]
        if len(cells) < 8:
            continue
        boxes = g.pick_spaced_cells(cells, 2, rng, min_dist=2)
        if boxes is None:
            continue
        rem = [p for p in cells if p not in boxes]
        targets = g.pick_spaced_cells(rem, 2, rng, min_dist=2)
        if targets is None:
            continue
        rem = [p for p in rem if p not in targets]
        if not rem:
            continue
        bomb = rng.choice(rem)
        map_str = g.grid_to_string(grid, boxes, targets, [bomb])
        metrics = quick_metrics(
            map_str,
            require_bomb=True,
            with_bomb_time=3.2,
            with_bomb_cost=460,
            no_bomb_time=1.5,
        )
        if metrics is None:
            continue
        if not (40 <= metrics['steps'] <= 65):
            continue
        if not (0.62 <= metrics['wall_ratio'] <= 0.72):
            continue
        return Candidate(
            map_str=map_str,
            phase=5,
            style='compact',
            family='compact_bomb',
            steps=metrics['steps'],
            wall_ratio=metrics['wall_ratio'],
            open_count=metrics['open_count'],
            boxes=len(metrics['boxes']),
            bombs=len(metrics['bombs']),
            bomb_used=bool(metrics['bomb_used']),
            score=metrics['steps'] + 45.0 * metrics['wall_ratio'],
        )
    return None


def make_phase6_open_candidate(rng: random.Random) -> Optional[Candidate]:
    for _ in range(180):
        grid, family = create_sparse_scaffold(rng, (4, 7))
        cells = [
            p
            for p in g.open_cells(grid, non_corner_only=True)
            if p != g.CAR_SPAWN and g.manhattan(p, g.CAR_SPAWN) >= 4
        ]
        if len(cells) < 8:
            continue
        targets = g.pick_spaced_cells(cells, 3, rng, min_dist=3)
        if targets is None:
            continue

        def score_fn(idx, boxes, targets_local, pull, actual):
            new_box = pull['new_box']
            return g.manhattan(new_box, targets_local[idx]) * 6 + max(0, 10 - g.manhattan(new_box, (7, 5))) * 2

        pulled = g.run_reverse_pull(grid, targets, rng, 20, 36, score_fn, bonus_walks=2)
        if pulled is None:
            continue
        boxes = pulled['boxes']
        if len(set(boxes)) < 3 or set(boxes) & set(targets) or any(not g.is_non_corner(grid, box) for box in boxes):
            continue
        map_str = g.grid_to_string(grid, boxes, targets, [])
        metrics = quick_metrics(map_str, require_bomb=False, with_bomb_time=5.2, with_bomb_cost=600)
        if metrics is None:
            continue
        if not (42 <= metrics['steps'] <= 70):
            continue
        if not (0.16 <= metrics['wall_ratio'] <= 0.26):
            continue
        return Candidate(
            map_str=map_str,
            phase=6,
            style='open',
            family=family,
            steps=metrics['steps'],
            wall_ratio=metrics['wall_ratio'],
            open_count=metrics['open_count'],
            boxes=len(metrics['boxes']),
            bombs=len(metrics['bombs']),
            bomb_used=bool(metrics['bomb_used']),
            score=metrics['steps'] + metrics['open_count'] * 0.08,
        )
    return None


def make_phase6_compact_plain_candidate(rng: random.Random) -> Optional[Candidate]:
    for _ in range(260):
        grid = g.create_compact_grid(rng, (42, 58))
        if grid is None:
            continue
        cells = [
            p
            for p in g.open_cells(grid, non_corner_only=True)
            if p != g.CAR_SPAWN and g.manhattan(p, g.CAR_SPAWN) >= 3
        ]
        if len(cells) < 8:
            continue
        targets = g.pick_spaced_cells(cells, 3, rng, min_dist=2)
        if targets is None:
            continue

        def score_fn(idx, boxes, targets_local, pull, actual):
            new_box = pull['new_box']
            return g.manhattan(new_box, targets_local[idx]) * 5 + max(0, 10 - g.manhattan(new_box, (7, 5))) * 2

        pulled = g.run_reverse_pull(grid, targets, rng, 18, 30, score_fn, bonus_walks=2)
        if pulled is None:
            continue
        boxes = pulled['boxes']
        if len(set(boxes)) < 3 or set(boxes) & set(targets) or any(not g.is_non_corner(grid, box) for box in boxes):
            continue
        map_str = g.grid_to_string(grid, boxes, targets, [])
        metrics = quick_metrics(map_str, require_bomb=False, with_bomb_time=6.0, with_bomb_cost=600)
        if metrics is None:
            continue
        if not (36 <= metrics['steps'] <= 70):
            continue
        if not (0.58 <= metrics['wall_ratio'] <= 0.72):
            continue
        return Candidate(
            map_str=map_str,
            phase=6,
            style='compact',
            family='compact_plain',
            steps=metrics['steps'],
            wall_ratio=metrics['wall_ratio'],
            open_count=metrics['open_count'],
            boxes=len(metrics['boxes']),
            bombs=len(metrics['bombs']),
            bomb_used=bool(metrics['bomb_used']),
            score=metrics['steps'] + 42.0 * metrics['wall_ratio'],
        )
    return None


def make_phase6_compact_bomb_candidate(rng: random.Random) -> Optional[Candidate]:
    item = make_phase5_compact_candidate(rng)
    if item is None:
        return None
    return Candidate(
        map_str=item.map_str,
        phase=6,
        style='compact',
        family='compact_bomb',
        steps=item.steps,
        wall_ratio=item.wall_ratio,
        open_count=item.open_count,
        boxes=item.boxes,
        bombs=item.bombs,
        bomb_used=item.bomb_used,
        score=item.score + 6.0,
    )


def collect_candidates(
    label: str,
    count: int,
    maker: Callable[[random.Random], Optional[Candidate]],
    rng: random.Random,
    existing: List[Candidate],
    threshold: int,
) -> List[Candidate]:
    picked: List[Candidate] = []
    attempts = 0
    limit = count * 80
    while len(picked) < count and attempts < limit:
        attempts += 1
        candidate = maker(rng)
        if candidate is None:
            continue
        if too_similar(candidate.map_str, existing + picked, threshold):
            continue
        picked.append(candidate)
        print(
            f'  {label} [{len(picked):02d}/{count}] '
            f'attempt={attempts} steps={candidate.steps} '
            f'wall={candidate.wall_ratio:.3f} family={candidate.family}'
        )
    if len(picked) < count:
        raise RuntimeError(f'{label}: only collected {len(picked)}/{count} maps after {attempts} attempts')
    return picked


def generate_phase(phase: int, rng: random.Random) -> List[Candidate]:
    if phase == 4:
        open_maps = collect_candidates('phase4-open', 5, make_phase4_open_candidate, rng, [], threshold=22)
        compact_maps = collect_candidates('phase4-compact', 5, make_phase4_compact_candidate, rng, open_maps, threshold=18)
        return open_maps + compact_maps
    if phase == 5:
        open_maps = collect_candidates('phase5-open', 5, make_phase5_open_candidate, rng, [], threshold=20)
        compact_maps = collect_candidates('phase5-compact', 5, make_phase5_compact_candidate, rng, open_maps, threshold=16)
        return open_maps + compact_maps
    if phase == 6:
        open_maps = collect_candidates('phase6-open', 5, make_phase6_open_candidate, rng, [], threshold=22)
        compact_plain = collect_candidates('phase6-compact-plain', 3, make_phase6_compact_plain_candidate, rng, open_maps, threshold=18)
        compact_bomb = collect_candidates('phase6-compact-bomb', 2, make_phase6_compact_bomb_candidate, rng, open_maps + compact_plain, threshold=16)
        return open_maps + compact_plain + compact_bomb
    raise ValueError(phase)


def save_phase_maps(phase: int, items: Sequence[Candidate]) -> Dict[str, Dict[str, object]]:
    out_dir = os.path.join(ROOT, 'assets', 'maps', f'phase{phase}')
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if name.lower().endswith('.txt'):
            os.remove(os.path.join(out_dir, name))
    report: Dict[str, Dict[str, object]] = {}
    for idx, item in enumerate(items, 1):
        fname = f'phase{phase}_{idx:02d}.txt'
        path = os.path.join(out_dir, fname)
        with open(path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(item.map_str)
        report[fname] = {
            'style': item.style,
            'family': item.family,
            'boxes': item.boxes,
            'bombs': item.bombs,
            'open_count': item.open_count,
            'wall_ratio': item.wall_ratio,
            'steps': item.steps,
            'bomb_used': item.bomb_used,
        }
    return report


def build_seed_manifest(phase: int, items: Sequence[Candidate]) -> Dict[str, Dict[str, object]]:
    manifest: Dict[str, Dict[str, object]] = {}
    for idx, item in enumerate(items, 1):
        fname = f'phase{phase}_{idx:02d}.txt'
        manifest[fname] = {
            'verified_seed_range': [0, SEED_SCAN_LIMIT - 1],
            'pairing_rule': 'scan-order box i -> scan-order target i',
            'verified_permutation': list(range(item.boxes)),
            'verified_seeds': IDENTITY_SEEDS[item.boxes],
            'is_exhaustive': False,
            'box_count': item.boxes,
            'style': item.style,
            'family': item.family,
            'bomb_required': bool(item.bombs),
        }
    return manifest


def update_json(path: str, phase_key: str, payload: Dict[str, object], wrapper_key: Optional[str] = None) -> None:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    else:
        data = {}
    if wrapper_key is not None:
        data.setdefault(wrapper_key, {})
        data[wrapper_key][phase_key] = payload
        data['generated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        data[phase_key] = payload
        data['generated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def update_quality_report(phase: int, phase_report: Dict[str, Dict[str, object]]) -> None:
    report_path = os.path.join(ROOT, 'assets', 'maps', 'quality_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    else:
        data = {}
    data.setdefault('phases', {})
    data['phases'][f'phase{phase}'] = phase_report
    data['generated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    data['seed_note'] = 'phase4-6 seed validity is documented in phase456_seed_manifest.json'
    with open(report_path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def update_seed_report(phase: int, phase_manifest: Dict[str, Dict[str, object]]) -> None:
    report_path = os.path.join(ROOT, 'assets', 'maps', 'phase456_seed_manifest.json')
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    else:
        data = {}
    data.setdefault('phases', {})
    data['phases'][f'phase{phase}'] = phase_manifest
    data['generated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    data['note'] = (
        'Set random.seed(seed) immediately before MapLoader.load / engine.reset. '
        'verified_seeds are guaranteed solvable for the documented pairing, but are not an exhaustive list of all solvable seeds.'
    )
    with open(report_path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description='Regenerate balanced phase4-6 maps with verified seeds.')
    parser.add_argument('--phase', type=int, choices=[4, 5, 6], required=True)
    parser.add_argument('--seed', type=int, default=20260306)
    args = parser.parse_args()

    phase = args.phase
    phase_rng = random.Random(args.seed + phase * 1000)
    print(f'phase={phase} seed={args.seed}')
    items = generate_phase(phase, phase_rng)
    phase_report = save_phase_maps(phase, items)
    phase_manifest = build_seed_manifest(phase, items)
    update_quality_report(phase, phase_report)
    update_seed_report(phase, phase_manifest)
    print(f'saved phase{phase}: {len(items)} maps')


if __name__ == '__main__':
    main()
