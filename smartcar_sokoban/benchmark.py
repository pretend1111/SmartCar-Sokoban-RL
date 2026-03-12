"""求解器评测脚本 — 多进程并行跑完所有 Phase 1~6 地图，输出统计报告.

用法:
    python -m smartcar_sokoban.benchmark                # 跑全部 Phase (多进程并行)
    python -m smartcar_sokoban.benchmark --solver exact # 只用 MultiBoxSolver
    python -m smartcar_sokoban.benchmark --phase 4      # 只跑某个 Phase
    python -m smartcar_sokoban.benchmark --save         # 结果保存到 runs/benchmark/
    python -m smartcar_sokoban.benchmark -j 1           # 单进程串行 (调试用)
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime

from smartcar_sokoban.paths import MAPS_ROOT, PROJECT_ROOT, RUNS_ROOT


# ── 单图求解 (子进程中执行) ────────────────────────────────

def _solve_one(task: dict) -> dict:
    """在子进程中求解单张地图. task 包含所有必要信息."""
    # 子进程需要重新 import
    sys.path.insert(0, task["root"])
    from smartcar_sokoban.config import GameConfig
    from smartcar_sokoban.engine import GameEngine
    from smartcar_sokoban.solver.auto_player import AutoPlayer
    from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
    from smartcar_sokoban.solver.pathfinder import pos_to_grid

    cfg = GameConfig()
    engine = GameEngine(cfg, task["root"])

    rel = task["rel"]
    seed = task["seed"]
    solver_mode = task["solver_mode"]

    random.seed(seed)
    engine.reset(rel)

    devnull = io.StringIO()

    if solver_mode == "exact":
        result = _do_exact(engine, devnull)
    elif solver_mode == "auto":
        result = _do_auto(engine, devnull)
    else:  # fallback
        result = _do_auto(engine, devnull)
        if not result["won"]:
            random.seed(seed)
            engine.reset(rel)
            result = _do_exact(engine, devnull)
            result["solver_used"] = "auto->exact"

    result["seed"] = seed
    result["map_name"] = task["map_name"]
    result["phase"] = task["phase"]
    return result


def _do_auto(engine, devnull):
    from smartcar_sokoban.solver.auto_player import AutoPlayer
    t0 = time.perf_counter()
    with redirect_stdout(devnull):
        player = AutoPlayer(engine)
        actions = player.solve()
    elapsed = time.perf_counter() - t0
    state = engine.get_state()
    return {
        "won": state.won,
        "steps": len(actions) if actions else 0,
        "time_ms": round(elapsed * 1000, 1),
        "solver_used": "auto",
    }


def _do_exact(engine, devnull):
    from smartcar_sokoban.solver.multi_box_solver import MultiBoxSolver
    from smartcar_sokoban.solver.pathfinder import pos_to_grid

    state = engine.get_state()
    boxes = [((int(b.x), int(b.y)), b.class_id) for b in state.boxes]
    targets = {t.num_id: (int(t.x), int(t.y)) for t in state.targets}
    bombs = [(int(b.x), int(b.y)) for b in state.bombs]

    t0 = time.perf_counter()
    solver = MultiBoxSolver(
        grid=state.grid,
        car_pos=pos_to_grid(state.car_x, state.car_y),
        boxes=boxes,
        targets=targets,
        bombs=bombs,
    )
    with redirect_stdout(devnull):
        solution = solver.solve(max_cost=1000, time_limit=30.0)
    elapsed = time.perf_counter() - t0

    if solution is None:
        return {"won": False, "steps": 0, "time_ms": round(elapsed * 1000, 1),
                "solver_used": "exact"}

    walk_steps = sum(wc + 1 for _, _, _, wc in solution)
    return {"won": True, "steps": walk_steps, "pushes": len(solution),
            "time_ms": round(elapsed * 1000, 1), "solver_used": "exact"}


# ── 种子管理 ───────────────────────────────────────────────

def load_seed_manifest():
    path = MAPS_ROOT / "phase456_seed_manifest.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get('phases', {})
    except Exception:
        return {}


def get_seed(manifest, phase, map_name, default_seed):
    phase_key = f'phase{phase}'
    if phase_key in manifest:
        item = manifest[phase_key].get(map_name)
        if item:
            seeds = item.get('verified_seeds') or []
            if seeds:
                return int(seeds[0])
    return default_seed


# ── 主流程 ─────────────────────────────────────────────────

def build_tasks(phases, solver_mode, default_seed):
    """构建所有待求解任务列表."""
    manifest = load_seed_manifest()
    tasks = []
    for phase in phases:
        phase_dir = MAPS_ROOT / f"phase{phase}"
        maps = sorted(glob.glob(str(phase_dir / "*.txt")))
        maps = [f for f in maps if 'verify_' not in os.path.basename(f)]
        for fpath in maps:
            name = os.path.basename(fpath)
            rel = os.path.relpath(fpath, PROJECT_ROOT).replace('\\', '/')
            seed = get_seed(manifest, phase, name, default_seed)
            tasks.append({
                "root": str(PROJECT_ROOT),
                "rel": rel,
                "seed": seed,
                "solver_mode": solver_mode,
                "map_name": name,
                "phase": phase,
            })
    return tasks


def run_benchmark(phases, solver_mode, default_seed, workers):
    tasks = build_tasks(phases, solver_mode, default_seed)
    if not tasks:
        print("  ⚠️  无地图文件")
        return {}

    total = len(tasks)
    done = 0

    # 按 phase 收集结果
    all_results = {}

    print(f"  共 {total} 张地图, {workers} 进程并行\n", flush=True)

    if workers == 1:
        # 串行模式 (调试/兼容)
        for task in tasks:
            result = _solve_one(task)
            done += 1
            mark = "." if result["won"] else "✗"
            print(f"\r  进度: {done}/{total} {mark}", end="", flush=True)
            phase_key = f"phase{result['phase']}"
            all_results.setdefault(phase_key, {})[result["map_name"]] = result
    else:
        # 多进程并行
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_solve_one, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                done += 1
                mark = "." if result["won"] else "✗"
                print(f"\r  进度: {done}/{total} {mark}", end="", flush=True)
                phase_key = f"phase{result['phase']}"
                all_results.setdefault(phase_key, {})[result["map_name"]] = result

    print()  # 换行
    return all_results


# ── 报告生成 ───────────────────────────────────────────────

def print_report(all_results, solver_mode):
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    DIM = "\033[2m"

    print(f"\n{'='*64}")
    print(f"  {BOLD}📊 求解器评测报告{RESET}")
    print(f"  {DIM}时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  模式: {solver_mode}{RESET}")
    print(f"{'='*64}")

    total_maps = 0
    total_wins = 0
    phase_summaries = []
    all_failed = []
    all_outliers = []

    for phase_key in sorted(all_results.keys()):
        results = all_results[phase_key]
        phase_num = phase_key.replace('phase', '')

        wins = sum(1 for r in results.values() if r["won"])
        total = len(results)
        total_maps += total
        total_wins += wins

        won_steps = [r["steps"] for r in results.values() if r["won"]]

        if won_steps:
            avg = statistics.mean(won_steps)
            med = statistics.median(won_steps)
            mn = min(won_steps)
            mx = max(won_steps)
            std = statistics.stdev(won_steps) if len(won_steps) > 1 else 0
        else:
            avg = med = mn = mx = std = 0

        rate = wins / total if total > 0 else 0
        if rate == 1.0:
            rate_color = GREEN
        elif rate >= 0.7:
            rate_color = YELLOW
        else:
            rate_color = RED

        times = [r["time_ms"] for r in results.values()]
        avg_time = statistics.mean(times) if times else 0

        phase_summaries.append({
            "phase": phase_num, "total": total, "wins": wins,
            "avg_steps": avg, "med_steps": med,
            "min_steps": mn, "max_steps": mx, "std_steps": std,
            "avg_time": avg_time,
        })

        print(f"\n  {BOLD}Phase {phase_num}{RESET}  "
              f"{rate_color}{wins}/{total} 通关 ({rate*100:.0f}%){RESET}")
        if won_steps:
            print(f"    步数: 平均 {CYAN}{avg:.1f}{RESET}  "
                  f"中位 {med:.0f}  范围 [{mn}, {mx}]  σ={std:.1f}")
        print(f"    耗时: 平均 {avg_time:.0f} ms")

        for name, r in sorted(results.items()):
            if not r["won"]:
                all_failed.append((phase_num, name, r))

        if won_steps and std > 0:
            threshold = max(avg + 1.5 * std, avg * 1.3)
            for name, r in sorted(results.items()):
                if r["won"] and r["steps"] > threshold:
                    all_outliers.append((phase_num, name, r, avg))

    overall_rate = total_wins / total_maps if total_maps > 0 else 0
    overall_color = GREEN if overall_rate == 1.0 else (YELLOW if overall_rate >= 0.8 else RED)

    print(f"\n{'─'*64}")
    print(f"  {BOLD}总计: {overall_color}{total_wins}/{total_maps} 通关 "
          f"({overall_rate*100:.1f}%){RESET}")

    if all_failed:
        print(f"\n  {RED}{BOLD}❌ 未通关地图 ({len(all_failed)} 张):{RESET}")
        for phase, name, r in all_failed:
            solver_info = f"  solver={r.get('solver_used', '?')}"
            print(f"    Phase {phase} / {name}  "
                  f"{DIM}({r['time_ms']:.0f}ms{solver_info}){RESET}")

    if all_outliers:
        print(f"\n  {YELLOW}{BOLD}⚠️  步数异常偏高 ({len(all_outliers)} 张):{RESET}")
        for phase, name, r, avg in all_outliers:
            ratio = r['steps'] / avg if avg > 0 else 0
            print(f"    Phase {phase} / {name}: "
                  f"{r['steps']} 步 (平均 {avg:.0f}, {YELLOW}{ratio:.1f}x{RESET})"
                  f"  {DIM}solver={r.get('solver_used', '?')}{RESET}")

    if not all_failed and not all_outliers:
        print(f"\n  {GREEN}🎉 全部通关且无异常！{RESET}")

    print(f"\n{'='*64}\n")

    return {
        "phase_summaries": phase_summaries,
        "failed": [(p, n, r) for p, n, r in all_failed],
        "outliers": [(p, n, r, a) for p, n, r, a in all_outliers],
        "overall": {"wins": total_wins, "total": total_maps},
    }


def save_results(all_results, report_data, solver_mode):
    out_dir = RUNS_ROOT / "benchmark"
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"benchmark_{solver_mode}_{timestamp}.json"
    filepath = out_dir / filename

    output = {
        "timestamp": datetime.now().isoformat(),
        "solver_mode": solver_mode,
        "overall": report_data["overall"],
        "phase_summaries": report_data["phase_summaries"],
        "failed_maps": [
            {"phase": p, "map": n, "time_ms": r["time_ms"]}
            for p, n, r in report_data["failed"]
        ],
        "outlier_maps": [
            {"phase": p, "map": n, "steps": r["steps"],
             "phase_avg": round(a, 1),
             "ratio": round(r["steps"] / a, 2) if a > 0 else 0}
            for p, n, r, a in report_data["outliers"]
        ],
        "details": all_results,
    }

    with open(filepath, 'w', encoding='utf-8') as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"  📁 结果已保存: {os.path.relpath(filepath, PROJECT_ROOT)}")
    return str(filepath)


# ── CLI 入口 ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='求解器评测 — 多进程并行跑 Phase 1~6 所有地图')
    parser.add_argument('--phase', type=int, default=None,
                        help='只跑指定 Phase (1-6)')
    parser.add_argument('--solver', choices=['auto', 'exact', 'fallback'],
                        default='auto',
                        help='求解模式: auto=仅AutoPlayer (默认), '
                             'exact=仅MultiBoxSolver, '
                             'fallback=auto失败回退exact')
    parser.add_argument('--seed', type=int, default=42,
                        help='默认随机种子 (默认 42)')
    parser.add_argument('--save', action='store_true',
                        help='保存结果到 runs/benchmark/ 目录')
    parser.add_argument('-j', '--workers', type=int, default=0,
                        help='并行进程数, 0=自动 (CPU核数), 1=串行')
    args = parser.parse_args()

    phases = [args.phase] if args.phase else list(range(1, 7))
    workers = args.workers if args.workers > 0 else os.cpu_count()

    print(f"\n  🚀 开始评测  模式={args.solver}  seed={args.seed}")
    print(f"  目标: Phase {', '.join(str(p) for p in phases)}")

    t_start = time.perf_counter()
    all_results = run_benchmark(phases, args.solver, args.seed, workers)
    total_time = time.perf_counter() - t_start

    report_data = print_report(all_results, args.solver)
    print(f"  ⏱  总耗时: {total_time:.1f}s")

    if args.save:
        save_results(all_results, report_data, args.solver)


if __name__ == "__main__":
    main()
