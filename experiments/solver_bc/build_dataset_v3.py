"""P3 v3 — 数据集构建 v3: 消费 phase{N}_all_seeds.json, 每张图用所有可解 seeds.

vs v2:
  - v2: 每张图 1 个 verified_seed
  - v3: 每张图所有 all_seeds 列表里的 seed (typically 4-10 per map)
  - 直接 3-10x 数据扩张, 但每个 (map, seed) 都是可解的

用法:
  python -m experiments.solver_bc.build_dataset_v3 \
    --all-seeds assets/maps/phase5_all_seeds.json \
    --output .agent/runs/p5_v4_all/phase5.npz \
    --teacher solver --num-workers 6 --max-seeds-per-map 5
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.solver_bc.oracle_features import build_oracle_obs
from experiments.solver_bc.teachers import collect_teacher_actions
from smartcar_sokoban.paths import MAPS_ROOT
from smartcar_sokoban.rl.high_level_env import N_ACTIONS, SokobanHLEnv


@dataclass(frozen=True)
class Job:
    map_path: str
    episode_seed: int
    teacher: str
    strategy: str
    max_steps: int
    include_map_layout: bool
    obs_mode: str
    max_cost: int
    time_limit: float


def _collect_episode(job: Job):
    random.seed(job.episode_seed)
    teacher_actions, teacher_used = collect_teacher_actions(
        job.map_path, job.episode_seed,
        teacher=job.teacher, max_steps=job.max_steps,
        include_map_layout=job.include_map_layout,
        max_cost=job.max_cost, time_limit=job.time_limit,
        strategy=job.strategy,
    )
    map_name = os.path.basename(job.map_path)
    if not teacher_actions:
        return {"map_name": map_name, "seed": job.episode_seed, "kept": False, "reason": "no_actions"}

    env = SokobanHLEnv(
        map_file=job.map_path, base_dir=ROOT,
        max_steps=job.max_steps, include_map_layout=job.include_map_layout,
    )
    env.reset(seed=job.episode_seed)

    obs_l, mask_l, act_l = [], [], []
    for action in teacher_actions:
        state = env.engine.get_state()
        mask = env.action_masks().astype(np.bool_)
        if action < 0 or action >= N_ACTIONS or not bool(mask[action]):
            return {"map_name": map_name, "seed": job.episode_seed, "kept": False, "reason": "invalid_action"}
        if job.obs_mode == "oracle":
            obs = build_oracle_obs(state, step_count=env._step_count,
                                   max_steps=env.max_steps,
                                   include_map_layout=job.include_map_layout)
        else:
            obs = env._build_state_vector()
        obs_l.append(obs); mask_l.append(mask); act_l.append(action)
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break

    if not act_l or not env.engine.get_state().won:
        return {"map_name": map_name, "seed": job.episode_seed, "kept": False, "reason": "not_won"}

    return {
        "map_name": map_name, "seed": job.episode_seed,
        "kept": True, "obs": np.stack(obs_l).astype(np.float32),
        "masks": np.stack(mask_l).astype(np.bool_),
        "actions": np.asarray(act_l, dtype=np.int64),
    }


def _resolve_map_path(map_name: str, default_phase_dir: Optional[str]) -> str:
    if default_phase_dir:
        p = os.path.join(default_phase_dir, map_name)
        if os.path.exists(p):
            return p
    for sub in os.listdir(MAPS_ROOT):
        candidate = MAPS_ROOT / sub / map_name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(map_name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all-seeds", required=True,
                   help="phase{N}_all_seeds.json (find_all_seeds.py 输出)")
    p.add_argument("--output", required=True)
    p.add_argument("--teacher", choices=["solver", "solver_ida", "hybrid"], default="solver")
    p.add_argument("--strategy", default="auto")
    p.add_argument("--time-limit", type=float, default=30.0)
    p.add_argument("--max-cost", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--include-map-layout", action="store_true", default=True)
    p.add_argument("--obs-mode", choices=["oracle", "state"], default="oracle")
    p.add_argument("--max-seeds-per-map", type=int, default=5,
                   help="每张图最多用前 N 个最快解的 seed")
    p.add_argument("--max-maps", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    with open(args.all_seeds, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    phase = manifest.get("phase")
    phase_dir = (str(MAPS_ROOT / f"phase{phase}") if phase else None)

    entries = manifest["results"]
    if args.max_maps > 0:
        entries = entries[:args.max_maps]
    print(f"[v3] phase={phase} maps={len(entries)} max_seeds_per_map={args.max_seeds_per_map}",
          flush=True)

    jobs: List[Job] = []
    for r in entries:
        try:
            map_path = _resolve_map_path(r["map"], phase_dir)
        except FileNotFoundError:
            continue
        seeds = [s["seed"] for s in r.get("all_seeds", [])][:args.max_seeds_per_map]
        for seed in seeds:
            jobs.append(Job(
                map_path=map_path, episode_seed=seed,
                teacher=args.teacher, strategy=args.strategy,
                max_steps=args.max_steps,
                include_map_layout=args.include_map_layout,
                obs_mode=args.obs_mode,
                max_cost=args.max_cost, time_limit=args.time_limit,
            ))

    print(f"[v3] jobs={len(jobs)}", flush=True)

    workers = args.num_workers if args.num_workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    obs_list, mask_list, action_list = [], [], []
    map_names, seeds_list, step_indices, episode_indices = [], [], [], []
    kept = 0
    reason_counts: Counter[str] = Counter()
    t0 = time.time()
    done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_collect_episode, j) for j in jobs]
        for ep_idx, fut in enumerate(concurrent.futures.as_completed(futs)):
            try:
                r = fut.result()
            except Exception as e:
                r = {"kept": False, "reason": f"crash:{e}"}
            done += 1
            reason_counts[r.get("reason", "ok") if not r.get("kept") else "ok"] += 1
            if r.get("kept"):
                obs_list.extend(r["obs"])
                mask_list.extend(r["masks"])
                action_list.extend(r["actions"].tolist())
                n_act = len(r["actions"])
                map_names.extend([r["map_name"]] * n_act)
                seeds_list.extend([r["seed"]] * n_act)
                step_indices.extend(list(range(n_act)))
                episode_indices.extend([ep_idx] * n_act)
                kept += 1
            if done % 100 == 0 or done == len(jobs):
                ela = time.time() - t0
                eta = ela / done * (len(jobs) - done) if done else 0
                print(f"  [{done}/{len(jobs)}] kept={kept} samples={len(action_list)} "
                      f"elapsed={int(ela)}s ETA={int(eta)}s", flush=True)

    if not obs_list:
        raise SystemExit("no expert samples collected")

    obs_arr = np.stack(obs_list).astype(np.float32)
    mask_arr = np.stack(mask_list).astype(np.bool_)
    action_arr = np.asarray(action_list, dtype=np.int64)
    map_arr = np.asarray(map_names)
    seed_arr = np.asarray(seeds_list, dtype=np.int64)
    step_arr = np.asarray(step_indices, dtype=np.int64)
    np.savez_compressed(
        args.output,
        obs=obs_arr, masks=mask_arr, actions=action_arr,
        map_names=map_arr, seeds=seed_arr, step_indices=step_arr,
        episode_indices=np.asarray(episode_indices, dtype=np.int64),
        obs_mode=np.asarray(args.obs_mode),
    )

    summary = {
        "phase": phase,
        "all_seeds_input": os.path.abspath(args.all_seeds),
        "n_maps": len(entries),
        "max_seeds_per_map": args.max_seeds_per_map,
        "n_jobs": len(jobs),
        "kept_episodes": kept,
        "samples": int(len(action_arr)),
        "teacher": args.teacher,
        "strategy": args.strategy,
        "reason_counts": dict(reason_counts),
        "obs_mode": args.obs_mode,
        "num_workers": workers,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(os.path.splitext(args.output)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
