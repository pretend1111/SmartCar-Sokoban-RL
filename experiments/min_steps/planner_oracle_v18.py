"""Planner oracle v18 — per-map best α via expected-utility score.

每张图试 v14 with α ∈ alphas, 用乘法 (期望效用) 选最优:
  S = log(1 + 1/steps) × p_per_gamble^k   (multiplicative, 默认)
其中 k = consumption-side gambling 次数, p_per_gamble ∈ (0,1) 是单次赌博的"成功率".
路径长时 1 次额外赌博的相对损失 ≈ log p_per_gamble × log(steps); 短路径下损失更大 — 非线性容忍。

也支持 additive: S = -(steps + λ × k), 选最大 (等价 step + λk 选最小).
"""

import math
from typing import List
from smartcar_sokoban.engine import GameEngine
from smartcar_sokoban.action_defs import direction_to_abs_action

from experiments.min_steps.planner_oracle import (
    _god_plan, _god_plans, _list_scans, _get_push_pos, _fresh_engine_from_eng,
    _walk_to_executor, _rotate_executor,
)
from experiments.min_steps.planner_oracle_v4 import (
    OracleResult4, _build_entity_vps, Q4_RAD,
)
from experiments.min_steps.planner_oracle_v14 import oracle_v14_min_steps
from smartcar_sokoban.solver.explorer_v3 import find_forced_pairs


def planner_oracle_v18(eng: GameEngine,
                        *, alphas=(0.0, 0.7, 2.0, 5.0, 25.0, 100.0),
                        score_mode: str = 'multiplicative',
                        p_per_gamble: float = 0.85,
                        gambling_weight: float = 20.0,
                        k_plans: int = 5,
                        kplan_time_limit: float = 20.0) -> None:
    """v18 enumeration over (god_plan_i, alpha_j) → 选 multiplicative max.

    K-best god plans 来源于 MultiBoxSolver.solve_kbest, first-push 多样化.
    每个 plan 跑 v14 DP at multiple alphas, 取 multiplicative best across all.
    """
    plans = _god_plans(eng, k=k_plans, time_limit=kplan_time_limit)
    if not plans:
        single = _god_plan(eng, time_limit=kplan_time_limit)
        plans = [single] if single else []
    if not plans: return

    state0 = eng.get_state()
    forced = find_forced_pairs(state0)
    fb = {i for i,_ in forced}; ft = {j for _,j in forced}
    # 不污染 engine state — 见 v14 同位说明.
    scans = _list_scans(state0, fb, ft)

    forced_classes = {state0.boxes[bi].class_id for bi,_ in forced if bi < len(state0.boxes)}
    forced_nums = {state0.targets[ti].num_id for _,ti in forced if ti < len(state0.targets)}
    N = len(state0.boxes)
    N_t = len(state0.targets)
    universe = {b.class_id for b in state0.boxes}
    universe_t = {t.num_id for t in state0.targets}

    scan_types = []; scan_labels = []
    for vp, ang, et, ei in scans:
        scan_types.append(et)
        scan_labels.append(state0.boxes[ei].class_id if et == 'box' else state0.targets[ei].num_id)

    def make_push_meta(plan):
        push_class_required = []
        for k in range(len(plan)):
            m = plan[k]; et, eid, _, _ = m
            push_class_required.append(eid[1] if et == 'box' else None)
        last_push_step = {}
        for k, c in enumerate(push_class_required):
            if c is not None: last_push_step[c] = k
        return push_class_required, last_push_step

    def count_gambling_consumption(res, push_class_required, last_push_step):
        """旧版: 只看 interleaving 的 explicit scan, 不考虑 walk-reveal.
        保留作为快速估算; 与 v14 DP 的 trust_walk_reveal=False 模型一致.
        """
        scan_set = 0; consumed = set(); gambling = 0
        for action in res.interleaving:
            if action[0] == 'scan':
                scan_set |= (1 << action[1])
            elif action[0] == 'push':
                k = action[1]
                pc = push_class_required[k]
                if pc is None: continue
                if last_push_step.get(pc) == k:
                    sc_c = set(); sc_n = set()
                    for j in range(len(scans)):
                        if scan_set & (1 << j):
                            if scan_types[j] == 'box': sc_c.add(scan_labels[j])
                            else: sc_n.add(scan_labels[j])
                    K_box = sc_c | forced_classes | consumed
                    K_tgt = sc_n | forced_nums | consumed
                    if len(K_box) >= N - 1: K_box = set(universe)
                    if len(K_tgt) >= N_t - 1: K_tgt = set(universe_t)
                    if (pc not in K_box) or (pc not in K_tgt):
                        gambling += 1
                    consumed.add(pc)
        return gambling

    def count_gambling_via_sim(res, plan):
        """精确版: 在 fresh engine 上 simulate plan, 用 engine 的 seen_*_ids 算 gambling.
        与 v14 DP trust_walk_reveal=True (信任 walk-reveal) 配合: 让 wrapper 看到
        plan 实际执行时 walk-reveal 触发的 entity 识别, 不误判过度.
        """
        from experiments.min_steps.planner_oracle_v14 import _execute_actions_v14
        from smartcar_sokoban.solver.pathfinder import pos_to_grid

        eng_sim = _fresh_engine_from_eng(eng)
        eng_sim._step_tag = 'init_snap'
        eng_sim.discrete_step(6)

        s0 = eng_sim.get_state()
        forced_local = find_forced_pairs(s0)
        fc_b = {s0.boxes[bi].class_id for bi, _ in forced_local if bi < len(s0.boxes)}
        fc_t = {s0.targets[ti].num_id for _, ti in forced_local if ti < len(s0.targets)}
        N_b = len(s0.boxes); N_t_local = len(s0.targets)
        box_universe = {b.class_id for b in s0.boxes}
        tgt_universe = {t.num_id for t in s0.targets}

        gamble_count = [0]
        orig_step = eng_sim.discrete_step

        def wrapped(a):
            s_b = eng_sim.get_state()
            seen_box = {s_b.boxes[i].class_id for i in s_b.seen_box_ids}
            consumed_box = box_universe - {b.class_id for b in s_b.boxes}
            eff_box = seen_box | fc_b | consumed_box
            if len(eff_box) >= N_b - 1: eff_box = set(box_universe)
            seen_tgt = {s_b.targets[i].num_id for i in s_b.seen_target_ids}
            consumed_tgt = tgt_universe - {t.num_id for t in s_b.targets}
            eff_tgt = seen_tgt | fc_t | consumed_tgt
            if len(eff_tgt) >= N_t_local - 1: eff_tgt = set(tgt_universe)
            classes_before = {b.class_id for b in s_b.boxes}
            result = orig_step(a)
            s_a = eng_sim.get_state()
            classes_after = {b.class_id for b in s_a.boxes}
            consumed_now = classes_before - classes_after
            for cnum in consumed_now:
                if cnum not in eff_box or cnum not in eff_tgt:
                    gamble_count[0] += 1
            return result

        eng_sim.discrete_step = wrapped
        try:
            _execute_actions_v14(eng_sim, res.interleaving, plan, vps_per_scan, scans=scans)
        except Exception:
            return 99   # 失败 plan, 给高 gambling penalty
        return gamble_count[0]

    vps_per_scan = _build_entity_vps(scans, state0)

    def joint_score(real_steps, gambling_count):
        if score_mode == 'additive':
            return -(real_steps + gambling_weight * gambling_count)
        step_reward = math.log(1.0 + 1.0 / max(real_steps, 1))
        win_score = p_per_gamble ** gambling_count
        return step_reward * win_score

    best_res = None
    best_plan = None
    best_score = float('-inf')
    for plan in plans:
        pcr, lps = make_push_meta(plan)
        for alpha in alphas:
            clone = _fresh_engine_from_eng(eng)
            res = oracle_v14_min_steps(clone, plan=plan, alpha=alpha,
                                          trust_walk_reveal=True)
            if res is None: continue
            # 用 sim-based count: 在 fresh engine 上跑 plan, engine seen_*_ids 算真实 gambling.
            # 与 trust_walk_reveal=True 配合 — DP 信任 walk-reveal 出短 plan, sim 验证实际 gambling.
            g = count_gambling_via_sim(res, plan)
            score = joint_score(res.cost, g)
            if score > best_score:
                best_score = score; best_res = res; best_plan = plan

    if best_res is None or best_plan is None:
        from experiments.min_steps.planner_oracle_v14 import planner_oracle_v14
        planner_oracle_v14(eng, alpha=0.7); return

    from experiments.min_steps.planner_oracle_v14 import _execute_actions_v14
    _execute_actions_v14(eng, best_res.interleaving, best_plan, vps_per_scan, scans=scans)
