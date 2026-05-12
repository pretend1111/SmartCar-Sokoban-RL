"""Level B JEPP 老师 — Belief-space IDA* + LCP.

形式化:
    σ_pair: box_idx → target_idx 双射. 一开始 |Σ| = N!
    每个 σ_pair 用规范化标签 (c_i = i, n_{σp(i)} = i) 跑 IDA* 拿严格最优 plan.

    每步循环:
        1. 当前 K_box, K_target 过滤 Σ → 留下兼容的 σ_pair
        2. 找 LCP across plans[σp][pointer:] for σp ∈ Σ
        3. lcp > 0: 执行第 1 步, 推进所有 pointer
        4. lcp = 0: 找最 cheapest inspect (走 + 旋转), 应用并把对应 entity 的真实 ID 填入 K, 重新过滤 Σ
        5. |Σ| = 1 后剩余直接执行
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
from smartcar_sokoban.solver.pathfinder import bfs_path, pos_to_grid
from smartcar_sokoban.solver.explorer import compute_facing_actions
from smartcar_sokoban.action_defs import direction_to_abs_action
from smartcar_sokoban.symbolic.belief import BeliefState
from smartcar_sokoban.symbolic.features import compute_domain_features, INF
from smartcar_sokoban.symbolic.candidates import (
    Candidate, generate_candidates,
)


# ── σ_pair 枚举 + 过滤 ─────────────────────────────────────

SigmaPair = Tuple[int, ...]    # σ[i] = target_idx for box_i


def enum_sigma_pairs(n: int) -> List[SigmaPair]:
    return list(itertools.permutations(range(n)))


def filter_compatible(sigma_pairs: List[SigmaPair],
                      K_box: Dict[int, int],
                      K_target: Dict[int, int]) -> List[SigmaPair]:
    """∀i ∈ K_box, j ∈ K_target: σp(i) == j iff K_box[i] == K_target[j]."""
    out: List[SigmaPair] = []
    for σp in sigma_pairs:
        ok = True
        for i, c_i in K_box.items():
            for j, n_j in K_target.items():
                same_label = (c_i == n_j)
                same_pair = (σp[i] == j)
                if same_label != same_pair:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append(σp)
    return out


# ── per-σ_pair IDA* plan ──────────────────────────────────

Move = Tuple[str, object, Tuple[int, int], int]


def solve_for_sigma_pair(state, σ_pair: SigmaPair,
                          time_limit: float = 60.0,
                          strategy: str = "auto") -> Optional[List[Move]]:
    """规范化标签: 把 box_i 当 class=i, target_{σp[i]} 当 num=i.

    让 solver 计算 box_i → target_{σp[i]} 的最优 plan. 自带 BestFirst → IDA* fallback (auto).
    """
    n = len(state.boxes)
    if len(σ_pair) != n:
        return None

    boxes_with_canon = [(pos_to_grid(b.x, b.y), i) for i, b in enumerate(state.boxes)]
    target_positions = [pos_to_grid(t.x, t.y) for t in state.targets]

    targets_dict: Dict[int, Tuple[int, int]] = {}
    for i in range(n):
        if σ_pair[i] >= len(target_positions):
            return None
        targets_dict[i] = target_positions[σ_pair[i]]

    bombs = [pos_to_grid(b.x, b.y) for b in state.bombs]
    car = pos_to_grid(state.car_x, state.car_y)

    solver = MultiBoxSolver(state.grid, car, boxes_with_canon, targets_dict, bombs)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            sol = solver.solve(max_cost=300, time_limit=time_limit, strategy=strategy)
            if sol is not None:
                return sol
            # IDA* / auto 失败 → 强制 BestFirst 兜底 (这个 σ_pair 可能就是难解的, 用 BF 1.5×OPT)
            return solver.solve(max_cost=300, time_limit=time_limit, strategy="best_first")
        except Exception:
            return None


# ── LCP ────────────────────────────────────────────────────

def _normalize_move(move: Move) -> Tuple[str, Tuple[int, int], Tuple[int, int]]:
    """跨 σ_pair 比较: (etype, position, direction). 忽略 class_id (用规范化时不同 σ 给的 cid 不同)."""
    etype, eid, direction, _ = move
    if etype == "box":
        old_pos, _ = eid
        return ("box", old_pos, direction)
    if etype == "bomb":
        return ("bomb", eid, direction)
    return (etype, (0, 0), direction)


def find_lcp_length(plans: List[List[Move]], pointers: List[int]) -> int:
    if not plans:
        return 0
    min_remaining = min(len(p) - ptr for p, ptr in zip(plans, pointers))
    if min_remaining <= 0:
        return 0
    lcp = 0
    while lcp < min_remaining:
        norms = set()
        for p, ptr in zip(plans, pointers):
            norms.add(_normalize_move(p[ptr + lcp]))
            if len(norms) > 1:
                break
        if len(norms) > 1:
            break
        lcp += 1
    return lcp


# ── 翻译 move → candidate index (忽略 class_id 因为是规范化的) ──

def match_move_to_candidate_index(move: Move, cands: List[Candidate],
                                   bs: BeliefState) -> Optional[int]:
    etype, eid, direction, _ = move
    if etype == "box":
        old_pos, _ = eid
        for i, b in enumerate(bs.boxes):
            if (b.col, b.row) != old_pos:
                continue
            for k, c in enumerate(cands):
                if (c.type == "push_box" and c.legal
                        and c.box_idx == i
                        and c.direction == direction
                        and c.run_length == 1):
                    return k
            return None
    elif etype == "bomb":
        old_pos = eid
        for i, bm in enumerate(bs.bombs):
            if (bm.col, bm.row) != old_pos:
                continue
            for k, c in enumerate(cands):
                if (c.type == "push_bomb" and c.legal
                        and c.bomb_idx == i
                        and c.direction == direction):
                    return k
            return None
    return None


# ── inspect 选择 ──────────────────────────────────────────

def pick_best_inspect_candidate(bs: BeliefState,
                                 candidates: List[Candidate],
                                 sigma_pairs: List[SigmaPair],
                                 K_box: Dict[int, int],
                                 K_target: Dict[int, int]
                                 ) -> Optional[Candidate]:
    """挑 inspect: 信息增益高 (能切断更多 σ) 且 cost 低."""
    walls = bs.M.astype(bool)
    obstacles = {(b.col, b.row) for b in bs.boxes}
    obstacles.update({(bm.col, bm.row) for bm in bs.bombs})
    car = (bs.player_col, bs.player_row)

    def _walk_cost(target):
        if car == target:
            return 0
        rows, cols = walls.shape
        visited = {car}
        q = deque()
        q.append((car, 0))
        while q:
            (c, r), d = q.popleft()
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc, nr = c + dc, r + dr
                if (nc, nr) in visited:
                    continue
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if walls[nr, nc]:
                    continue
                if (nc, nr) in obstacles:
                    continue
                visited.add((nc, nr))
                if (nc, nr) == target:
                    return d + 1
                q.append(((nc, nr), d + 1))
        return INF

    best = None
    best_score = -INF
    for c in candidates:
        if c.type != "inspect" or not c.legal:
            continue
        wc = _walk_cost((c.viewpoint_col, c.viewpoint_row))
        if wc == INF:
            continue
        rot = (c.inspect_heading - bs.theta_player) % 8
        rot = min(rot, 8 - rot)
        cost = wc + rot

        # 信息增益: 假设各种揭示结果, 平均 |Σ_after| 减少
        # 简化: |Σ_distinct_outcomes| 越多, 信息越大
        if c.inspect_target_type == "box":
            # 如果观察 box_i, 不同 σ_pair 给不同 c_i? 但 c_i 是 ground truth 决定的, 跟 σ_pair 无关.
            # 实际上观察 box 不会缩减 |Σ_pair| 单独, 但和已知 K_target 一起会.
            # 信息增益 = 假设 c_i 揭示后, 跟 K_target 联动后过滤的 σ_pair 数变化.
            n_outcomes = len(set(σ[c.inspect_target_idx] for σ in sigma_pairs))   # 不同 σp(i) 数
            # σp(i) ∈ {0..N-1}, 每个值对应一个未知的 n_{σp(i)}
            ig = float(n_outcomes)
        elif c.inspect_target_type == "target":
            # 观察 target_j 揭示 n_j. 跟已知 K_box 联动.
            # σp 中 σp^{-1}(j) 不同的话, 信息有.
            n_outcomes = len(set(σ.index(c.inspect_target_idx) for σ in sigma_pairs))
            ig = float(n_outcomes)
        else:
            ig = 1.0

        # 总分: 先比信息增益 (大优先), 再比 cost (小优先)
        score = ig * 100 - cost
        if score > best_score:
            best_score = score
            best = c
    return best


# ── 执行 ───────────────────────────────────────────────────

def _heading_to_angle(heading: int) -> float:
    angles = {0: 0.0, 1: math.pi / 4, 2: math.pi / 2, 3: 3 * math.pi / 4,
              4: math.pi, 5: -3 * math.pi / 4, 6: -math.pi / 2, 7: -math.pi / 4}
    return angles.get(heading, 0.0)


def apply_move(eng: GameEngine, move: Move) -> bool:
    eng.discrete_step(6)
    state = eng.get_state()
    etype, eid, direction, _ = move
    dx, dy = direction
    if etype == "box":
        old_pos, _ = eid
        ec, er = old_pos
    elif etype == "bomb":
        ec, er = eid
    else:
        return False
    car_target = (ec - dx, er - dy)
    obstacles = set()
    for b in state.boxes:
        obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs:
        obstacles.add(pos_to_grid(bm.x, bm.y))
    car_grid = pos_to_grid(state.car_x, state.car_y)
    if car_grid != car_target:
        path = bfs_path(car_grid, car_target, state.grid, obstacles)
        if path is None:
            return False
        for pdx, pdy in path:
            eng.discrete_step(direction_to_abs_action(pdx, pdy))
    eng.discrete_step(direction_to_abs_action(dx, dy))
    return True


def apply_inspect(eng: GameEngine, cand: Candidate) -> bool:
    eng.discrete_step(6)
    state = eng.get_state()
    obstacles = set()
    for b in state.boxes:
        obstacles.add(pos_to_grid(b.x, b.y))
    for bm in state.bombs:
        obstacles.add(pos_to_grid(bm.x, bm.y))
    car_grid = pos_to_grid(state.car_x, state.car_y)
    target = (cand.viewpoint_col, cand.viewpoint_row)
    if car_grid != target:
        path = bfs_path(car_grid, target, state.grid, obstacles)
        if path is None:
            return False
        for pdx, pdy in path:
            eng.discrete_step(direction_to_abs_action(pdx, pdy))
    state = eng.get_state()
    rot_acts = compute_facing_actions(state.car_angle,
                                       _heading_to_angle(cand.inspect_heading or 0))
    for a in rot_acts:
        eng.discrete_step(a)
    return True


# ── 主求解 ────────────────────────────────────────────────

def belief_ida_solve(map_path: str, seed: int,
                      *, ida_time_limit: float = 15.0,
                      step_limit: int = 80,
                      strategy: str = "auto"
                      ) -> Tuple[Optional[List[Dict]], str]:
    """跑 belief-space IDA* + LCP. 简化版: 每步 (或拓扑变化时) 重新 enum + IDA*.

    避免索引重映射麻烦. 代价是每步多调几次 IDA*.

    K 用 (class_id_set, num_id_set) 跟踪已揭示的 *标签* (跟 box index 解耦).
    σ 是基于当前 box→target 的双射.
    """
    random.seed(seed)
    eng = GameEngine()
    state = eng.reset(map_path)
    n_box_init = len(state.boxes)
    if n_box_init == 0:
        return None, "size_mismatch"

    # 用"标签集"而非 idx 存观察: 已揭示的 class_id, 已揭示的 num_id
    # 揭示某 box → 把 box.class_id 加入 known_classes, 同理 target
    known_classes: Set[int] = set()
    known_nums: Set[int] = set()
    # 同时用 idx-based K 帮 σ_pair 过滤 (idx 在每次 topology 变化时 reset)
    K_box: Dict[int, int] = {}
    K_target: Dict[int, int] = {}

    # 缓存: (n_box_current) → (sigma_pairs_list, plans_dict, pointers_dict)
    cache_key: Optional[int] = None
    sigma_pairs: List[SigmaPair] = []
    plans: Dict[SigmaPair, List[Move]] = {}
    pointers: Dict[SigmaPair, int] = {}

    samples_meta: List[Dict] = []
    n_steps = 0
    last_n_box = n_box_init

    while not eng.get_state().won and n_steps < step_limit:
        n_steps += 1
        cur_state = eng.get_state()
        cur_n_box = len(cur_state.boxes)

        # 拓扑变化 → 清空 cache + K (idx 失效)
        if cur_n_box != last_n_box:
            cache_key = None
            K_box = {}
            K_target = {}
            # 但 known_classes / known_nums 保留 (label-based)
            last_n_box = cur_n_box

        # 重新 enum + IDA* 如果 cache miss
        if cache_key != cur_n_box:
            sigma_pairs = enum_sigma_pairs(cur_n_box)
            plans = {}
            for σp in sigma_pairs:
                plan = solve_for_sigma_pair(cur_state, σp,
                                             time_limit=ida_time_limit, strategy=strategy)
                if plan is not None:
                    plans[σp] = plan
            sigma_pairs = list(plans.keys())
            pointers = {σ: 0 for σ in sigma_pairs}
            cache_key = cur_n_box

            if not sigma_pairs:
                return None, "no_perm_solvable"

        bs = BeliefState.from_engine_state(cur_state, fully_observed=False)
        feat = compute_domain_features(bs)
        cands = generate_candidates(bs, feat, push_only=False)  # JEPP 需要 inspect

        # 过滤 σ_pairs by current K
        sigma_pairs = filter_compatible(sigma_pairs, K_box, K_target)
        if not sigma_pairs:
            # 拓扑刚变化, K 已清空; 这里仍 empty 说明硬冲突
            return None, "sigmas_empty"

        # LCP
        lcp_len = find_lcp_length([plans[σ] for σ in sigma_pairs],
                                    [pointers[σ] for σ in sigma_pairs])

        if lcp_len > 0:
            ref_σ = sigma_pairs[0]
            move = plans[ref_σ][pointers[ref_σ]]
            label = match_move_to_candidate_index(move, cands, bs)
            if label is None:
                return None, "lcp_match_fail"
            samples_meta.append({
                "bs": bs, "feat": feat, "cands": cands,
                "label": label, "type": "push",
            })
            if not apply_move(eng, move):
                return None, "apply_fail"
            for σ in sigma_pairs:
                pointers[σ] += 1
            continue

        # 必须 inspect
        if len(sigma_pairs) == 1 and pointers[sigma_pairs[0]] >= len(plans[sigma_pairs[0]]):
            return None, "plan_exhausted"

        inspect_cand = pick_best_inspect_candidate(bs, cands, sigma_pairs,
                                                     K_box, K_target)
        if inspect_cand is None:
            # 兜底: inspect 不可用 (entity 全被堵) → 跑 majority vote push 解开僵局.
            # 取所有 σ_pair 第一动作出现频率最高的, 试推一下让物理状态变化.
            from collections import Counter
            first_moves = []
            for σ in sigma_pairs:
                if pointers[σ] < len(plans[σ]):
                    first_moves.append(_normalize_move(plans[σ][pointers[σ]]))
            if not first_moves:
                return None, "no_inspect"
            most_common, _ = Counter(first_moves).most_common(1)[0]
            # 找对应 σ
            picked_σ = None
            for σ in sigma_pairs:
                if (pointers[σ] < len(plans[σ])
                        and _normalize_move(plans[σ][pointers[σ]]) == most_common):
                    picked_σ = σ
                    break
            if picked_σ is None:
                return None, "no_inspect"
            move = plans[picked_σ][pointers[picked_σ]]
            label = match_move_to_candidate_index(move, cands, bs)
            if label is None:
                return None, "majority_vote_match_fail"
            samples_meta.append({
                "bs": bs, "feat": feat, "cands": cands,
                "label": label, "type": "push",
            })
            if not apply_move(eng, move):
                return None, "majority_vote_apply_fail"
            # 跳出 LCP 走野了, 强制下一轮重新 enum + IDA* (cache miss)
            cache_key = None
            continue

        label = -1
        for k, c in enumerate(cands):
            if c is inspect_cand:
                label = k
                break
        if label < 0:
            return None, "inspect_match_fail"

        samples_meta.append({
            "bs": bs, "feat": feat, "cands": cands,
            "label": label, "type": "inspect",
        })

        if not apply_inspect(eng, inspect_cand):
            return None, "inspect_apply_fail"

        state_after = eng.get_state()
        et = inspect_cand.inspect_target_type
        ei = inspect_cand.inspect_target_idx
        if et == "box":
            if ei in state_after.seen_box_ids and ei < len(state_after.boxes):
                cls = state_after.boxes[ei].class_id
                K_box[ei] = cls
                known_classes.add(cls)
        elif et == "target":
            if ei in state_after.seen_target_ids and ei < len(state_after.targets):
                num = state_after.targets[ei].num_id
                K_target[ei] = num
                known_nums.add(num)

    won = eng.get_state().won
    return (samples_meta if won else None), ("ok" if won else "step_limit")
