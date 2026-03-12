from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
from collections import deque
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.push_solver import bfs_push

ROWS = 12
COLS = 16
WALL = '#'
AIR = '-'
BOX = '$'
TARGET = '.'
BOMB = '*'
CAR_SPAWN = (1, 6)
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


@dataclass(frozen=True)
class RoomDef:
    room_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, pos: Tuple[int, int]) -> bool:
        c, r = pos
        return self.x1 <= c <= self.x2 and self.y1 <= r <= self.y2


@dataclass(frozen=True)
class DoorDef:
    cells: Tuple[Tuple[int, int], ...]
    rooms: Tuple[int, int]
    orientation: str


@dataclass
class Layout:
    grid: List[List[int]]
    rooms: List[RoomDef]
    doors: List[DoorDef]


@dataclass
class Candidate:
    map_str: str
    score: float
    metrics: Dict[str, object]


def blank_grid(fill: int = 0) -> List[List[int]]:
    grid = [[fill] * COLS for _ in range(ROWS)]
    for c in range(COLS):
        grid[0][c] = 1
        grid[ROWS - 1][c] = 1
    for r in range(ROWS):
        grid[r][0] = 1
        grid[r][COLS - 1] = 1
    return grid


def copy_grid(grid: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in grid]


def in_bounds(c: int, r: int) -> bool:
    return 0 <= c < COLS and 0 <= r < ROWS


def is_open(grid: List[List[int]], pos: Tuple[int, int]) -> bool:
    c, r = pos
    return in_bounds(c, r) and grid[r][c] == 0


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def count_turns(actions: Sequence[Tuple[int, int]]) -> int:
    return sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])


def interior_wall_ratio(grid: List[List[int]]) -> float:
    walls = sum(grid[r][c] for r in range(1, ROWS - 1) for c in range(1, COLS - 1))
    return walls / ((ROWS - 2) * (COLS - 2))


def interior_open_count(grid: List[List[int]]) -> int:
    return sum(1 for r in range(1, ROWS - 1) for c in range(1, COLS - 1) if grid[r][c] == 0)


def neighbors4(pos: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
    c, r = pos
    for dc, dr in DIRS:
        yield c + dc, r + dr


def flood_fill(grid: List[List[int]], start: Tuple[int, int], blocked: Optional[Set[Tuple[int, int]]] = None) -> Set[Tuple[int, int]]:
    blocked = blocked or set()
    if not is_open(grid, start) or start in blocked:
        return set()
    seen = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in neighbors4(cur):
            if nxt in seen or nxt in blocked or not is_open(grid, nxt):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen


def is_all_connected(grid: List[List[int]]) -> bool:
    opens = [(c, r) for r in range(1, ROWS - 1) for c in range(1, COLS - 1) if grid[r][c] == 0]
    if not opens:
        return False
    return len(flood_fill(grid, opens[0])) == len(opens)


def is_non_corner(grid: List[List[int]], pos: Tuple[int, int]) -> bool:
    c, r = pos
    if not is_open(grid, pos):
        return False
    up = grid[r - 1][c] == 1
    down = grid[r + 1][c] == 1
    left = grid[r][c - 1] == 1
    right = grid[r][c + 1] == 1
    return not ((up and left) or (up and right) or (down and left) or (down and right))


def open_cells(grid: List[List[int]], room: Optional[RoomDef] = None, non_corner_only: bool = False) -> List[Tuple[int, int]]:
    cells = []
    for r in range(1, ROWS - 1):
        for c in range(1, COLS - 1):
            pos = (c, r)
            if grid[r][c] != 0:
                continue
            if room is not None and not room.contains(pos):
                continue
            if non_corner_only and not is_non_corner(grid, pos):
                continue
            cells.append(pos)
    return cells


def pick_spaced_cells(cells: Sequence[Tuple[int, int]], count: int, rng: random.Random, min_dist: int = 2) -> Optional[List[Tuple[int, int]]]:
    if len(cells) < count:
        return None
    for _ in range(200):
        chosen: List[Tuple[int, int]] = []
        pool = list(cells)
        rng.shuffle(pool)
        for cell in pool:
            if all(manhattan(cell, prev) >= min_dist for prev in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def grid_to_string(grid: List[List[int]], boxes: Sequence[Tuple[int, int]], targets: Sequence[Tuple[int, int]], bombs: Sequence[Tuple[int, int]]) -> str:
    box_set = set(boxes)
    target_set = set(targets)
    bomb_set = set(bombs)
    lines = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            pos = (c, r)
            if pos in box_set:
                row.append(BOX)
            elif pos in target_set:
                row.append(TARGET)
            elif pos in bomb_set:
                row.append(BOMB)
            elif grid[r][c] == 1:
                row.append(WALL)
            else:
                row.append(AIR)
        lines.append(''.join(row))
    return '\n'.join(lines)


def parse_map_string(map_str: str):
    grid, boxes, targets, bombs = [], [], [], []
    for r, line in enumerate(map_str.splitlines()):
        row = []
        for c, ch in enumerate(line):
            if ch == WALL:
                row.append(1)
            else:
                row.append(0)
                if ch == BOX:
                    boxes.append((c, r))
                elif ch == TARGET:
                    targets.append((c, r))
                elif ch == BOMB:
                    bombs.append((c, r))
        grid.append(row)
    return grid, boxes, targets, bombs


def room_of(rooms: Sequence[RoomDef], pos: Tuple[int, int]) -> Optional[int]:
    for room in rooms:
        if room.contains(pos):
            return room.room_id
    return None


def room_area(room: RoomDef) -> int:
    return (room.x2 - room.x1 + 1) * (room.y2 - room.y1 + 1)


def solve_single_box(grid: List[List[int]], box: Tuple[int, int], target: Tuple[int, int]):
    actions = bfs_push(CAR_SPAWN, box, target, grid)
    if actions is None:
        return None
    return {'actions': actions, 'steps': len(actions), 'turns': count_turns(actions)}


def solve_multi_box(grid, boxes, targets, bombs, time_limit: float, max_cost: int = 800):
    solver = MultiBoxSolver(
        grid=grid,
        car_pos=CAR_SPAWN,
        boxes=[(pos, idx) for idx, pos in enumerate(boxes)],
        targets={idx: pos for idx, pos in enumerate(targets)},
        bombs=list(bombs),
    )
    with redirect_stdout(io.StringIO()):
        solution = solver.solve(max_cost=max_cost, time_limit=time_limit)
    return solver, solution


def trace_box_paths(boxes, solution):
    paths = {idx: [pos] for idx, pos in enumerate(boxes)}
    for etype, eid, direction, _ in solution:
        if etype != 'box':
            continue
        old_pos, cid = eid
        paths[cid].append((old_pos[0] + direction[0], old_pos[1] + direction[1]))
    return paths


def overlap_cells(paths):
    seen: Dict[Tuple[int, int], int] = {}
    for points in paths.values():
        for cell in set(points):
            seen[cell] = seen.get(cell, 0) + 1
    return {cell for cell, count in seen.items() if count >= 2}


def count_box_changes(solution):
    changes = 0
    prev = None
    for etype, eid, _, _ in solution:
        if etype != 'box':
            continue
        cid = eid[1]
        if prev is not None and cid != prev:
            changes += 1
        prev = cid
    return changes


def door_crossings(paths, door_cells):
    used = set()
    for points in paths.values():
        for cell in points:
            if cell in door_cells:
                used.add(cell)
    return len(used)


def evaluate_solution(grid, solver, solution):
    steps = sum(wc + 1 for _, _, _, wc in solution)
    actions = solver.solution_to_actions(list(solution))
    turns = count_turns(actions)
    walls = interior_wall_ratio(grid)
    box_switches = count_box_changes(solution)
    compact = 1.0 - (interior_open_count(grid) / 140.0)
    parts = {
        'steps': min(steps / 80.0, 1.0) * 25.0,
        'turns': min(turns / 8.0, 1.0) * 20.0,
        'walls': min(walls / 0.3, 1.0) * 15.0,
        'box_changes': min(box_switches / 4.0, 1.0) * 20.0,
        'compact': compact * 20.0,
    }
    return sum(parts.values()), parts


class ReversePullEngine:
    def __init__(self, grid: List[List[int]], rng: random.Random):
        self.grid = grid
        self.rng = rng

    def _car_reachable(self, start, goal, blocked):
        if goal in blocked or not is_open(self.grid, goal):
            return False
        if start == goal:
            return True
        seen = {start}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nxt in neighbors4(cur):
                if nxt in seen or nxt in blocked or not is_open(self.grid, nxt):
                    continue
                if nxt == goal:
                    return True
                seen.add(nxt)
                queue.append(nxt)
        return False

    def random_adjacent(self, box, boxes):
        occupied = set(boxes)
        options = [p for p in neighbors4(box) if is_open(self.grid, p) and p not in occupied]
        self.rng.shuffle(options)
        for option in options:
            if self._car_reachable(CAR_SPAWN, option, occupied):
                return option
        return options[0] if options else None

    def enum_pulls(self, car, box_idx, boxes):
        box = boxes[box_idx]
        others = set(pos for i, pos in enumerate(boxes) if i != box_idx)
        results = []
        for dx, dy in DIRS:
            car_must_be = (box[0] + dx, box[1] + dy)
            new_car = (box[0] + 2 * dx, box[1] + 2 * dy)
            new_box = car_must_be
            if not is_open(self.grid, car_must_be) or not is_open(self.grid, new_car):
                continue
            if car_must_be in others or new_car in others:
                continue
            if not self._car_reachable(car, car_must_be, others | {box}):
                continue
            results.append({'new_car': new_car, 'new_box': new_box, 'direction': (dx, dy)})
        return results

    def walk_to_random_box_side(self, car, boxes):
        occupied = set(boxes)
        idxs = list(range(len(boxes)))
        self.rng.shuffle(idxs)
        for idx in idxs:
            sides = [p for p in neighbors4(boxes[idx]) if is_open(self.grid, p) and p not in occupied]
            self.rng.shuffle(sides)
            for side in sides:
                if self._car_reachable(car, side, occupied):
                    return side
        return car


def run_reverse_pull(grid, targets, rng, min_pulls, max_pulls, score_fn, bonus_walks=0):
    boxes = list(targets)
    engine = ReversePullEngine(grid, rng)
    car = engine.random_adjacent(boxes[0], boxes)
    if car is None:
        return None
    desired = rng.randint(min_pulls, max_pulls)
    actual = 0
    seen = {tuple(boxes)}
    for _ in range(desired * 6 + 40):
        if actual >= desired:
            break
        candidates = []
        for idx in range(len(boxes)):
            for pull in engine.enum_pulls(car, idx, boxes):
                new_boxes = boxes[:]
                new_boxes[idx] = pull['new_box']
                score = float(score_fn(idx, boxes, targets, pull, actual))
                if tuple(new_boxes) in seen:
                    score -= 30.0
                candidates.append((score, pull, new_boxes))
        if not candidates:
            new_car = engine.walk_to_random_box_side(car, boxes)
            if new_car == car:
                break
            car = new_car
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        top_k = min(4, len(candidates))
        score, pull, new_boxes = rng.choice(candidates[:top_k]) if rng.random() < 0.18 else candidates[0]
        boxes = new_boxes
        car = pull['new_car']
        seen.add(tuple(boxes))
        actual += 1
    for _ in range(bonus_walks):
        car = engine.walk_to_random_box_side(car, boxes)
    return {'boxes': boxes, 'car': car, 'pull_count': actual}


def pull_boxes_to_rooms(grid, targets, rooms, goal_room_id, assignment, rng, extra_steps=18):
    boxes = list(targets)
    engine = ReversePullEngine(grid, rng)
    car = engine.random_adjacent(boxes[0], boxes)
    if car is None:
        return None
    door_cells = {cell for door in create_virtual_doors_from_rooms(rooms) for cell in door}

    def pull_score(idx, current_boxes, pull):
        new_box = pull['new_box']
        cur_room = room_of(rooms, current_boxes[idx])
        new_room = room_of(rooms, new_box)
        score = manhattan(new_box, targets[idx]) * 5
        if cur_room == goal_room_id and new_room != goal_room_id:
            score += 100
        if new_room == assignment[idx]:
            score += 180
        if new_box in door_cells:
            score += 30
        return score

    for idx in range(len(boxes)):
        for _ in range(180):
            if room_of(rooms, boxes[idx]) == assignment[idx] and boxes[idx] != targets[idx]:
                break
            pulls = engine.enum_pulls(car, idx, boxes)
            if not pulls:
                car = engine.walk_to_random_box_side(car, boxes)
                continue
            pulls.sort(key=lambda item: pull_score(idx, boxes, item), reverse=True)
            top = pulls[: min(3, len(pulls))]
            choice = rng.choice(top) if rng.random() < 0.2 else top[0]
            boxes[idx] = choice['new_box']
            car = choice['new_car']
        else:
            return None

    for _ in range(extra_steps):
        candidates = []
        for idx in range(len(boxes)):
            for pull in engine.enum_pulls(car, idx, boxes):
                score = manhattan(pull['new_box'], targets[idx]) * 3
                if room_of(rooms, pull['new_box']) == assignment[idx]:
                    score += 40
                candidates.append((score, idx, pull))
        if not candidates:
            car = engine.walk_to_random_box_side(car, boxes)
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, idx, pull = rng.choice(candidates[: min(4, len(candidates))])
        boxes[idx] = pull['new_box']
        car = pull['new_car']
    return {'boxes': boxes, 'car': car}


def create_virtual_doors_from_rooms(rooms):
    doors = []
    for room_a in rooms:
        for room_b in rooms:
            if room_a.room_id >= room_b.room_id:
                continue
            if room_a.x2 + 2 == room_b.x1 or room_b.x2 + 2 == room_a.x1:
                wall_x = room_a.x2 + 1 if room_a.x2 < room_b.x1 else room_b.x2 + 1
                overlap_top = max(room_a.y1, room_b.y1)
                overlap_bottom = min(room_a.y2, room_b.y2)
                if overlap_top <= overlap_bottom:
                    doors.append(tuple((wall_x, r) for r in range(overlap_top, overlap_bottom + 1)))
            if room_a.y2 + 2 == room_b.y1 or room_b.y2 + 2 == room_a.y1:
                wall_y = room_a.y2 + 1 if room_a.y2 < room_b.y1 else room_b.y2 + 1
                overlap_left = max(room_a.x1, room_b.x1)
                overlap_right = min(room_a.x2, room_b.x2)
                if overlap_left <= overlap_right:
                    doors.append(tuple((c, wall_y) for c in range(overlap_left, overlap_right + 1)))
    return doors

def create_maze_grid(rng: random.Random, wall_ratio_range: Tuple[float, float]):
    target_ratio = rng.uniform(*wall_ratio_range)
    target_walls = int(round(target_ratio * 140))
    grid = [[1] * COLS for _ in range(ROWS)]
    for c in range(COLS):
        grid[0][c] = 1
        grid[ROWS - 1][c] = 1
    for r in range(ROWS):
        grid[r][0] = 1
        grid[r][COLS - 1] = 1

    start = (2, 2)
    grid[start[1]][start[0]] = 0
    stack = [start]
    visited = {start}
    while stack:
        cx, cy = stack[-1]
        neighbors = []
        for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
            nx, ny = cx + dx, cy + dy
            if 1 <= nx <= COLS - 2 and 1 <= ny <= ROWS - 2 and (nx, ny) not in visited:
                neighbors.append((nx, ny, cx + dx // 2, cy + dy // 2))
        if neighbors:
            nx, ny, mx, my = rng.choice(neighbors)
            grid[ny][nx] = 0
            grid[my][mx] = 0
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()

    grid[CAR_SPAWN[1]][CAR_SPAWN[0]] = 0
    grid[CAR_SPAWN[1]][CAR_SPAWN[0] + 1] = 0
    walls = [(c, r) for r in range(1, ROWS - 1) for c in range(1, COLS - 1) if grid[r][c] == 1]
    rng.shuffle(walls)
    current_walls = len(walls)
    for c, r in walls:
        if current_walls <= target_walls:
            break
        grid[r][c] = 0
        current_walls -= 1
    if not is_all_connected(grid):
        return None
    ratio = interior_wall_ratio(grid)
    if not (wall_ratio_range[0] <= ratio <= wall_ratio_range[1] + 0.02):
        return None
    return grid


def create_bsp_rooms(rng: random.Random):
    grid = blank_grid(0)
    rooms: List[RoomDef] = []
    doors: List[DoorDef] = []

    if rng.random() < 0.5:
        split_x = rng.randint(6, 9)
        for r in range(1, ROWS - 1):
            grid[r][split_x] = 1
        main_door_rows = [rng.randint(3, 8)]
        if rng.random() < 0.5:
            extra = max(2, min(9, main_door_rows[0] + rng.choice([-1, 1])))
            if extra not in main_door_rows:
                main_door_rows.append(extra)
        for dr in main_door_rows:
            grid[dr][split_x] = 0

        split_left = rng.random() < 0.5
        if split_left:
            split_y = rng.randint(4, 7)
            for c in range(1, split_x):
                grid[split_y][c] = 1
            side_door_cols = [rng.randint(2, split_x - 2)]
            if rng.random() < 0.4:
                extra = max(2, min(split_x - 2, side_door_cols[0] + rng.choice([-1, 1])))
                if extra not in side_door_cols:
                    side_door_cols.append(extra)
            for dc in side_door_cols:
                grid[split_y][dc] = 0
            rooms = [
                RoomDef(0, 1, 1, split_x - 1, split_y - 1),
                RoomDef(1, 1, split_y + 1, split_x - 1, ROWS - 2),
                RoomDef(2, split_x + 1, 1, COLS - 2, ROWS - 2),
            ]
            main_cells = tuple((split_x, dr) for dr in sorted(main_door_rows))
            doors = [
                DoorDef(main_cells, (0, 2), 'vertical'),
                DoorDef(main_cells, (1, 2), 'vertical'),
                DoorDef(tuple((dc, split_y) for dc in sorted(side_door_cols)), (0, 1), 'horizontal'),
            ]
        else:
            split_y = rng.randint(4, 7)
            for c in range(split_x + 1, COLS - 1):
                grid[split_y][c] = 1
            side_door_cols = [rng.randint(split_x + 2, COLS - 3)]
            if rng.random() < 0.4:
                extra = max(split_x + 2, min(COLS - 3, side_door_cols[0] + rng.choice([-1, 1])))
                if extra not in side_door_cols:
                    side_door_cols.append(extra)
            for dc in side_door_cols:
                grid[split_y][dc] = 0
            rooms = [
                RoomDef(0, 1, 1, split_x - 1, ROWS - 2),
                RoomDef(1, split_x + 1, 1, COLS - 2, split_y - 1),
                RoomDef(2, split_x + 1, split_y + 1, COLS - 2, ROWS - 2),
            ]
            main_cells = tuple((split_x, dr) for dr in sorted(main_door_rows))
            doors = [
                DoorDef(main_cells, (0, 1), 'vertical'),
                DoorDef(main_cells, (0, 2), 'vertical'),
                DoorDef(tuple((dc, split_y) for dc in sorted(side_door_cols)), (1, 2), 'horizontal'),
            ]
    else:
        split_y = rng.randint(4, 7)
        for c in range(1, COLS - 1):
            grid[split_y][c] = 1
        main_door_cols = [rng.randint(3, 12)]
        if rng.random() < 0.5:
            extra = max(2, min(13, main_door_cols[0] + rng.choice([-1, 1])))
            if extra not in main_door_cols:
                main_door_cols.append(extra)
        for dc in main_door_cols:
            grid[split_y][dc] = 0

        split_top = rng.random() < 0.5
        if split_top:
            split_x = rng.randint(6, 9)
            for r in range(1, split_y):
                grid[r][split_x] = 1
            side_door_rows = [rng.randint(2, split_y - 2)]
            if rng.random() < 0.4:
                extra = max(2, min(split_y - 2, side_door_rows[0] + rng.choice([-1, 1])))
                if extra not in side_door_rows:
                    side_door_rows.append(extra)
            for dr in side_door_rows:
                grid[dr][split_x] = 0
            rooms = [
                RoomDef(0, 1, 1, split_x - 1, split_y - 1),
                RoomDef(1, split_x + 1, 1, COLS - 2, split_y - 1),
                RoomDef(2, 1, split_y + 1, COLS - 2, ROWS - 2),
            ]
            main_cells = tuple((dc, split_y) for dc in sorted(main_door_cols))
            doors = [
                DoorDef(main_cells, (0, 2), 'horizontal'),
                DoorDef(main_cells, (1, 2), 'horizontal'),
                DoorDef(tuple((split_x, dr) for dr in sorted(side_door_rows)), (0, 1), 'vertical'),
            ]
        else:
            split_x = rng.randint(6, 9)
            for r in range(split_y + 1, ROWS - 1):
                grid[r][split_x] = 1
            side_door_rows = [rng.randint(split_y + 2, ROWS - 3)]
            if rng.random() < 0.4:
                extra = max(split_y + 2, min(ROWS - 3, side_door_rows[0] + rng.choice([-1, 1])))
                if extra not in side_door_rows:
                    side_door_rows.append(extra)
            for dr in side_door_rows:
                grid[dr][split_x] = 0
            rooms = [
                RoomDef(0, 1, 1, COLS - 2, split_y - 1),
                RoomDef(1, 1, split_y + 1, split_x - 1, ROWS - 2),
                RoomDef(2, split_x + 1, split_y + 1, COLS - 2, ROWS - 2),
            ]
            main_cells = tuple((dc, split_y) for dc in sorted(main_door_cols))
            doors = [
                DoorDef(main_cells, (0, 1), 'horizontal'),
                DoorDef(main_cells, (0, 2), 'horizontal'),
                DoorDef(tuple((split_x, dr) for dr in sorted(side_door_rows)), (1, 2), 'vertical'),
            ]

    grid[CAR_SPAWN[1]][CAR_SPAWN[0]] = 0
    grid[CAR_SPAWN[1]][CAR_SPAWN[0] + 1] = 0
    protected = {CAR_SPAWN}
    for door in doors:
        protected.update(door.cells)
        for cell in door.cells:
            for adj in neighbors4(cell):
                if in_bounds(*adj):
                    protected.add(adj)
    for room in rooms:
        for _ in range(rng.randint(1, 3)):
            orientation = rng.choice(['h', 'v', 'dot'])
            if orientation == 'dot':
                c = rng.randint(room.x1, room.x2)
                r = rng.randint(room.y1, room.y2)
                if (c, r) in protected:
                    continue
                grid[r][c] = 1
                if not is_all_connected(grid):
                    grid[r][c] = 0
                continue
            if orientation == 'h':
                if room.x2 - room.x1 + 1 < 2:
                    continue
                c0 = rng.randint(room.x1, room.x2 - 1)
                r0 = rng.randint(room.y1, room.y2)
                cells = [(c0, r0), (c0 + 1, r0)]
            else:
                if room.y2 - room.y1 + 1 < 2:
                    continue
                c0 = rng.randint(room.x1, room.x2)
                r0 = rng.randint(room.y1, room.y2 - 1)
                cells = [(c0, r0), (c0, r0 + 1)]
            if any(cell in protected for cell in cells):
                continue
            for c, r in cells:
                grid[r][c] = 1
            if not is_all_connected(grid):
                for c, r in cells:
                    grid[r][c] = 0
    if not is_all_connected(grid):
        return None
    return Layout(grid=grid, rooms=rooms, doors=doors)


def create_compact_grid(rng: random.Random, open_target_range: Tuple[int, int]):
    target_open = rng.randint(*open_target_range)
    grid = blank_grid(1)
    grid[CAR_SPAWN[1]][CAR_SPAWN[0]] = 0
    start = (2, 6)
    grid[start[1]][start[0]] = 0
    opened = {CAR_SPAWN, start}
    current = start
    steps = 0
    while len(opened) < target_open and steps < 800:
        steps += 1
        options = []
        for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
            nxt = (current[0] + dx, current[1] + dy)
            between = (current[0] + dx // 2, current[1] + dy // 2)
            if 1 < nxt[0] < COLS - 1 and 1 < nxt[1] < ROWS - 1:
                options.append((nxt, between))
        if not options:
            current = rng.choice(list(opened))
            continue
        nxt, between = rng.choice(options)
        for cell in (between, nxt):
            grid[cell[1]][cell[0]] = 0
            opened.add(cell)
        current = nxt
        if rng.random() < 0.22:
            extra = (current[0] + rng.choice([-1, 1]), current[1])
            if 1 <= extra[0] < COLS - 1 and 1 <= extra[1] < ROWS - 1:
                grid[extra[1]][extra[0]] = 0
                opened.add(extra)
    grid[CAR_SPAWN[1]][CAR_SPAWN[0] + 1] = 0
    opened.add((CAR_SPAWN[0] + 1, CAR_SPAWN[1]))
    if not is_all_connected(grid):
        return None
    count = interior_open_count(grid)
    if not (open_target_range[0] <= count <= open_target_range[1]):
        return None
    return grid


def create_phase4_lane_layout(rng: random.Random):
    rows = rng.choice([(2, 6, 9), (2, 5, 9), (2, 6, 8)])
    box_x = rng.choice([2, 3])
    target_x = rng.choice([12, 13])
    left_door = rng.choice([4, 5])
    right_door = rng.choice([9, 10])
    grid = blank_grid(1)
    for y in range(rows[0], rows[-1] + 1):
        grid[y][1] = 0
    for y in rows:
        for x in range(1, left_door):
            grid[y][x] = 0
        for x in range(left_door + 1, right_door):
            grid[y][x] = 0
        for x in range(right_door + 1, target_x + 1):
            grid[y][x] = 0
        grid[y][left_door] = 0
        grid[y][right_door] = 0
    if rng.random() < 0.5:
        for y in range(rows[0], rows[1] + 1):
            grid[y][right_door - 1] = 0
    if rng.random() < 0.5:
        for y in range(rows[1], rows[2] + 1):
            grid[y][left_door + 1] = 0
    boxes = [(box_x, y) for y in rows]
    targets = [(target_x, y) for y in rows]
    door_cells = {(left_door, y) for y in rows} | {(right_door, y) for y in rows}
    return grid, boxes, targets, door_cells


def add_easy_lane(grid, boxes, targets):
    extra_grid = copy_grid(grid)
    for y in range(1, CAR_SPAWN[1] + 1):
        extra_grid[y][1] = 0
    for x in range(1, 5):
        extra_grid[1][x] = 0
    extra_box = (2, 1)
    extra_target = (4, 1)
    if extra_box in boxes or extra_target in targets:
        return None
    return extra_grid, [extra_box] + list(boxes), [extra_target] + list(targets)

def make_phase1_candidate(rng: random.Random):
    grid = blank_grid(0)
    target = (rng.randint(3, 12), rng.randint(3, 8))

    def score_fn(idx, boxes, targets, pull, actual):
        new_box = pull['new_box']
        away = manhattan(new_box, targets[idx]) * 10
        center = manhattan(new_box, (7, 5))
        edge_penalty = 20 if new_box[0] in {1, 14} or new_box[1] in {1, 10} else 0
        return away + center - edge_penalty

    pulled = run_reverse_pull(grid, [target], rng, 18, 36, score_fn, bonus_walks=1)
    if pulled is None:
        return None
    box = pulled['boxes'][0]
    solved = solve_single_box(grid, box, target)
    if solved is None or manhattan(box, target) < 12:
        return None
    if not (15 <= solved['steps'] <= 40) or solved['turns'] < 2:
        return None
    metrics = {'phase': 1, 'steps': solved['steps'], 'turns': solved['turns'], 'manhattan': manhattan(box, target), 'pulls': pulled['pull_count'], 'wall_ratio': 0.0}
    score = solved['steps'] + 2.5 * solved['turns'] + 1.5 * metrics['manhattan']
    return Candidate(grid_to_string(grid, [box], [target], []), score, metrics)


def make_phase2_candidate(rng: random.Random):
    grid = create_maze_grid(rng, (0.15, 0.25))
    if grid is None:
        return None
    cells = [p for p in open_cells(grid, non_corner_only=True) if p != CAR_SPAWN and manhattan(p, CAR_SPAWN) >= 5]
    if len(cells) < 4:
        return None
    target = rng.choice(cells)

    def score_fn(idx, boxes, targets, pull, actual):
        return manhattan(pull['new_box'], targets[idx]) * 9 + 3 * actual

    pulled = run_reverse_pull(grid, [target], rng, 28, 52, score_fn, bonus_walks=2)
    if pulled is None:
        return None
    box = pulled['boxes'][0]
    solved = solve_single_box(grid, box, target)
    wall_ratio = interior_wall_ratio(grid)
    if solved is None or box == target or not is_non_corner(grid, box):
        return None
    if not (20 <= solved['steps'] <= 60) or solved['turns'] < 4:
        return None
    if not (0.15 <= wall_ratio <= 0.25):
        return None
    metrics = {'phase': 2, 'steps': solved['steps'], 'turns': solved['turns'], 'manhattan': manhattan(box, target), 'pulls': pulled['pull_count'], 'wall_ratio': round(wall_ratio, 3)}
    score = solved['steps'] + 3.0 * solved['turns'] + 80.0 * wall_ratio
    return Candidate(grid_to_string(grid, [box], [target], []), score, metrics)


def make_phase3_candidate(rng: random.Random):
    grid = create_maze_grid(rng, (0.20, 0.30))
    if grid is None:
        return None
    cells = [p for p in open_cells(grid, non_corner_only=True) if manhattan(p, CAR_SPAWN) >= 4]
    if len(cells) < 8:
        return None
    rng.shuffle(cells)
    target1 = target2 = None
    for a in cells:
        close = [b for b in cells if b != a and 3 <= manhattan(a, b) <= 6]
        if close:
            target1 = a
            target2 = rng.choice(close)
            break
    if target1 is None:
        return None
    anchors = sorted([p for p in cells if p not in {target1, target2}], key=lambda p: manhattan(p, target1) + manhattan(p, target2), reverse=True)
    if len(anchors) < 2:
        return None
    anchor_map = {0: anchors[0], 1: anchors[1]}

    def score_fn(idx, boxes, targets, pull, actual):
        new_box = pull['new_box']
        return manhattan(new_box, targets[idx]) * 6 + max(0, 16 - manhattan(new_box, anchor_map[idx])) * 2 + max(0, 10 - manhattan(new_box, (7, 5)))

    pulled = run_reverse_pull(grid, [target1, target2], rng, 34, 56, score_fn, bonus_walks=2)
    if pulled is None:
        return None
    boxes = pulled['boxes']
    if len(set(boxes)) < 2 or set(boxes) & {target1, target2} or any(not is_non_corner(grid, box) for box in boxes):
        return None
    map_str = grid_to_string(grid, boxes, [target1, target2], [])
    _, parsed_boxes, parsed_targets, _ = parse_map_string(map_str)
    solver, solution = solve_multi_box(grid, parsed_boxes, parsed_targets, [], time_limit=12.0, max_cost=500)
    if solution is None or solver is None:
        return None
    steps = sum(wc + 1 for _, _, _, wc in solution)
    paths = trace_box_paths(parsed_boxes, solution)
    overlap = overlap_cells(paths)
    box_switches = count_box_changes(solution)
    if not (30 <= steps <= 80) or len(overlap) < 1 or box_switches < 1:
        return None
    score, parts = evaluate_solution(grid, solver, solution)
    metrics = {'phase': 3, 'steps': steps, 'turns': count_turns(solver.solution_to_actions(solution)), 'wall_ratio': round(interior_wall_ratio(grid), 3), 'target_distance': manhattan(target1, target2), 'path_overlap': len(overlap), 'box_changes': box_switches, 'score_parts': parts, 'pulls': pulled['pull_count']}
    return Candidate(map_str, score + len(overlap) * 8, metrics)


def choose_goal_room(layout: Layout) -> RoomDef:
    return max(layout.rooms, key=room_area)


def make_phase4_candidate(rng: random.Random):
    grid, boxes, targets, door_cells = create_phase4_lane_layout(rng)
    solver, solution = solve_multi_box(grid, boxes, targets, [], time_limit=16.0, max_cost=500)
    if solution is None or solver is None:
        return None
    steps = sum(wc + 1 for _, _, _, wc in solution)
    paths = trace_box_paths(boxes, solution)
    crossings = door_crossings(paths, door_cells)
    if not (40 <= steps <= 120) or crossings < 2:
        return None
    score, parts = evaluate_solution(grid, solver, solution)
    metrics = {'phase': 4, 'steps': steps, 'turns': count_turns(solver.solution_to_actions(solution)), 'rooms': 3, 'door_crossings': crossings, 'wall_ratio': round(interior_wall_ratio(grid), 3), 'score_parts': parts}
    return Candidate(grid_to_string(grid, boxes, targets, []), score + crossings * 6, metrics)


def make_phase5_candidate(rng: random.Random):
    for _ in range(120):
        grid = create_compact_grid(rng, (38, 48))
        if grid is None:
            continue
        cells = [p for p in open_cells(grid, non_corner_only=True) if p != CAR_SPAWN]
        if len(cells) < 8:
            continue
        boxes = pick_spaced_cells(cells, 2, rng, min_dist=2)
        if boxes is None:
            continue
        rem = [p for p in cells if p not in boxes]
        targets = pick_spaced_cells(rem, 2, rng, min_dist=2)
        if targets is None:
            continue
        rem = [p for p in rem if p not in targets]
        if not rem:
            continue
        bomb = rng.choice(rem)
        _, no_bomb_solution = solve_multi_box(grid, boxes, targets, [], time_limit=5.0, max_cost=500)
        solver, solution = solve_multi_box(grid, boxes, targets, [bomb], time_limit=8.0, max_cost=600)
        if solution is None or solver is None:
            continue
        if no_bomb_solution is not None or not any(etype == 'bomb' for etype, _, _, _ in solution):
            continue
        augmented = add_easy_lane(grid, boxes, targets)
        if augmented is None:
            continue
        aug_grid, aug_boxes, aug_targets = augmented
        _, no_bomb_aug = solve_multi_box(aug_grid, aug_boxes, aug_targets, [], time_limit=6.0, max_cost=700)
        solver2, sol2 = solve_multi_box(aug_grid, aug_boxes, aug_targets, [bomb], time_limit=14.0, max_cost=800)
        if sol2 is None or solver2 is None:
            continue
        if no_bomb_aug is not None or not any(etype == 'bomb' for etype, _, _, _ in sol2):
            continue
        steps = sum(wc + 1 for _, _, _, wc in sol2)
        if not (50 <= steps <= 150):
            continue
        score, parts = evaluate_solution(aug_grid, solver2, sol2)
        metrics = {'phase': 5, 'steps': steps, 'turns': count_turns(solver2.solution_to_actions(sol2)), 'wall_ratio': round(interior_wall_ratio(aug_grid), 3), 'bomb_required': True, 'bomb_used': True, 'open_count': interior_open_count(aug_grid), 'score_parts': parts}
        return Candidate(grid_to_string(aug_grid, aug_boxes, aug_targets, [bomb]), score + 18, metrics)
    return None


def make_phase6_compact_candidate(rng: random.Random, n_boxes: int):
    grid = create_compact_grid(rng, (40, 50))
    if grid is None:
        return None
    cells = [p for p in open_cells(grid, non_corner_only=True) if p != CAR_SPAWN]
    if len(cells) < n_boxes * 2 + 2:
        return None
    targets = pick_spaced_cells(cells, n_boxes, rng, min_dist=2)
    if targets is None:
        return None

    def score_fn(idx, boxes, targets_local, pull, actual):
        new_box = pull['new_box']
        return manhattan(new_box, targets_local[idx]) * 5 + max(0, 9 - manhattan(new_box, (7, 5))) * 2

    pulled = run_reverse_pull(grid, targets, rng, 24, 48, score_fn, bonus_walks=2)
    if pulled is None:
        return None
    boxes = pulled['boxes']
    if len(set(boxes)) < n_boxes or set(boxes) & set(targets):
        return None
    if n_boxes == 1:
        solved = solve_single_box(grid, boxes[0], targets[0])
        if solved is None or not (30 <= solved['steps'] <= 150):
            return None
        metrics = {'phase': 6, 'steps': solved['steps'], 'turns': solved['turns'], 'open_count': interior_open_count(grid), 'bomb_used': False, 'wall_ratio': round(interior_wall_ratio(grid), 3)}
        score = solved['steps'] + 3.0 * solved['turns'] + (70 - metrics['open_count'])
        return Candidate(grid_to_string(grid, boxes, targets, []), score, metrics)
    solver, solution = solve_multi_box(grid, boxes, targets, [], time_limit=18.0, max_cost=700)
    if solution is None or solver is None:
        return None
    steps = sum(wc + 1 for _, _, _, wc in solution)
    if not (30 <= steps <= 150):
        return None
    score, parts = evaluate_solution(grid, solver, solution)
    metrics = {'phase': 6, 'steps': steps, 'turns': count_turns(solver.solution_to_actions(solution)), 'open_count': interior_open_count(grid), 'bomb_used': False, 'wall_ratio': round(interior_wall_ratio(grid), 3), 'score_parts': parts}
    return Candidate(grid_to_string(grid, boxes, targets, []), score, metrics)


def make_phase6_bomb_candidate(rng: random.Random):
    grid = create_compact_grid(rng, (38, 48))
    if grid is None:
        return None
    cells = open_cells(grid, non_corner_only=True)
    pivot_candidates = [c for c in range(5, 11) if any(cell[0] == c for cell in cells)]
    if not pivot_candidates:
        return None
    pivot = rng.choice(pivot_candidates)
    door_row = rng.randint(3, 8)
    if not is_open(grid, (pivot - 1, door_row)) or not is_open(grid, (pivot + 1, door_row)):
        return None
    for r in range(1, ROWS - 1):
        if r == door_row:
            continue
        grid[r][pivot] = 1
    blocked_grid = copy_grid(grid)
    blocked_grid[door_row][pivot] = 1
    left_cells = [p for p in open_cells(blocked_grid, non_corner_only=True) if p[0] < pivot]
    right_cells = [p for p in open_cells(blocked_grid, non_corner_only=True) if p[0] > pivot]
    boxes = pick_spaced_cells(left_cells, 2, rng, min_dist=3)
    targets = pick_spaced_cells(right_cells, 2, rng, min_dist=2)
    if boxes is None or targets is None:
        return None
    bomb_pos = (pivot - 1, door_row)
    if bomb_pos in boxes or bomb_pos in targets or not is_open(blocked_grid, bomb_pos):
        return None
    _, no_bomb_solution = solve_multi_box(blocked_grid, boxes, targets, [], time_limit=6.0, max_cost=400)
    if no_bomb_solution is not None:
        return None
    solver, solution = solve_multi_box(blocked_grid, boxes, targets, [bomb_pos], time_limit=18.0, max_cost=600)
    if solution is None or solver is None or not any(etype == 'bomb' for etype, _, _, _ in solution):
        return None
    steps = sum(wc + 1 for _, _, _, wc in solution)
    if not (30 <= steps <= 150):
        return None
    metrics = {'phase': 6, 'steps': steps, 'turns': count_turns(solver.solution_to_actions(solution)), 'open_count': interior_open_count(blocked_grid), 'bomb_used': True, 'wall_ratio': round(interior_wall_ratio(blocked_grid), 3)}
    score, _ = evaluate_solution(blocked_grid, solver, solution)
    return Candidate(grid_to_string(blocked_grid, boxes, targets, [bomb_pos]), score + 12, metrics)


def make_phase6_candidate(rng: random.Random):
    if rng.random() < 0.45:
        return make_phase6_bomb_candidate(rng)
    return make_phase6_compact_candidate(rng, rng.choice([1, 2, 3]))


PHASE_GENERATORS = {1: make_phase1_candidate, 2: make_phase2_candidate, 3: make_phase3_candidate, 4: make_phase4_candidate, 5: make_phase5_candidate, 6: make_phase6_candidate}


def generate_phase_maps(phase: int, count: int, rng: random.Random):
    generator = PHASE_GENERATORS[phase]
    candidates, seen = [], set()
    attempts = 0
    limit = count * 500
    started = time.perf_counter()
    while len(candidates) < count and attempts < limit:
        attempts += 1
        candidate = generator(rng)
        if candidate is None or candidate.map_str in seen:
            continue
        seen.add(candidate.map_str)
        candidates.append(candidate)
        elapsed = time.perf_counter() - started
        print(f"  [{len(candidates):02d}/{count}] attempt={attempts} score={candidate.score:.1f} elapsed={elapsed:.1f}s")
    if len(candidates) < count:
        raise RuntimeError(f"phase {phase} only generated {len(candidates)}/{count} maps after {attempts} attempts")
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:count]


def save_phase_maps(phase: int, candidates: Sequence[Candidate]):
    out_dir = os.path.join(ROOT, 'assets', 'maps', f'phase{phase}')
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if name.lower().endswith('.txt'):
            os.remove(os.path.join(out_dir, name))
    report = {}
    for idx, candidate in enumerate(candidates, 1):
        fname = f'phase{phase}_{idx:02d}.txt'
        with open(os.path.join(out_dir, fname), 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(candidate.map_str)
        report[fname] = candidate.metrics
    return report


def verify_phase_outputs(phase: int, report):
    out_dir = os.path.join(ROOT, 'assets', 'maps', f'phase{phase}')
    for fname in sorted(report):
        map_str = open(os.path.join(out_dir, fname), 'r', encoding='utf-8').read()
        grid, boxes, targets, bombs = parse_map_string(map_str)
        if len(boxes) != len(targets):
            raise RuntimeError(f'{fname}: boxes != targets')
        if phase in {1, 2}:
            if solve_single_box(grid, boxes[0], targets[0]) is None:
                raise RuntimeError(f'{fname}: not solvable')
        else:
            _, solution = solve_multi_box(grid, boxes, targets, bombs, time_limit=25.0, max_cost=900)
            if solution is None:
                raise RuntimeError(f'{fname}: not solvable')
            if bombs and not any(etype == 'bomb' for etype, _, _, _ in solution):
                raise RuntimeError(f'{fname}: bomb exists but was not used')


def main():
    parser = argparse.ArgumentParser(description='Generate high-quality Sokoban maps from scratch.')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4, 5, 6], help='Generate only one phase')
    parser.add_argument('--count', type=int, default=10, help='Maps per phase')
    parser.add_argument('--seed', type=int, default=20260306, help='Base random seed')
    args = parser.parse_args()
    phases = [args.phase] if args.phase else [1, 2, 3, 4, 5, 6]
    full_report = {}
    print('High-quality map generation')
    print(f'  phases={phases}')
    print(f'  count={args.count}')
    print(f'  seed={args.seed}')
    for phase in phases:
        print(f"\n{'=' * 60}\nPhase {phase}\n{'=' * 60}")
        phase_rng = random.Random(args.seed + phase * 1000)
        started = time.perf_counter()
        candidates = generate_phase_maps(phase, args.count, phase_rng)
        phase_report = save_phase_maps(phase, candidates)
        verify_phase_outputs(phase, phase_report)
        print(f"  saved={len(candidates)} verified in {time.perf_counter() - started:.1f}s")
        full_report[f'phase{phase}'] = phase_report
    report_path = os.path.join(ROOT, 'assets', 'maps', 'quality_report.json')
    with open(report_path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump({'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'seed': args.seed, 'count_per_phase': args.count, 'phases': full_report}, fh, ensure_ascii=False, indent=2)
    print(f'\nreport={report_path}')


if __name__ == '__main__':
    main()
