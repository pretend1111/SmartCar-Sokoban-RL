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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.solver_bc.oracle_features import (
    build_oracle_obs,
)
from experiments.solver_bc.teachers import collect_teacher_actions
from smartcar_sokoban.rl.high_level_env import N_ACTIONS, SokobanHLEnv
from smartcar_sokoban.rl.train import CURRICULUM, get_map_pool, load_seed_manifest


def default_seeds_for_map(map_name: str, seed_manifest: Dict[str, List[int]],
                          seeds_per_map: int, base_seed: int) -> List[int]:
    verified = list(seed_manifest.get(map_name, []))
    if verified:
        return verified[:seeds_per_map]
    return [base_seed + 9973 * i for i in range(seeds_per_map)]


@dataclass(frozen=True)
class EpisodeJob:
    map_path: str
    episode_seed: int
    teacher: str
    max_steps: int
    include_map_layout: bool
    obs_mode: str
    max_cost: int
    time_limit: float


@dataclass
class EpisodeResult:
    map_name: str
    episode_seed: int
    teacher_used: str
    obs: Optional[np.ndarray]
    masks: Optional[np.ndarray]
    actions: Optional[np.ndarray]
    step_indices: Optional[np.ndarray]
    kept: bool
    won: bool
    rejected_moves: int


def _collect_episode(job: EpisodeJob) -> EpisodeResult:
    random.seed(job.episode_seed)
    teacher_actions, teacher_used = collect_teacher_actions(
        job.map_path,
        job.episode_seed,
        teacher=job.teacher,
        max_steps=job.max_steps,
        include_map_layout=job.include_map_layout,
        max_cost=job.max_cost,
        time_limit=job.time_limit,
    )
    map_name = os.path.basename(job.map_path)
    if not teacher_actions:
        return EpisodeResult(
            map_name=map_name,
            episode_seed=job.episode_seed,
            teacher_used=teacher_used,
            obs=None,
            masks=None,
            actions=None,
            step_indices=None,
            kept=False,
            won=False,
            rejected_moves=0,
        )

    env = SokobanHLEnv(
        map_file=job.map_path,
        base_dir=ROOT,
        max_steps=job.max_steps,
        include_map_layout=job.include_map_layout,
    )
    env.reset(seed=job.episode_seed)

    episode_obs: List[np.ndarray] = []
    episode_masks: List[np.ndarray] = []
    episode_actions: List[int] = []
    rejected_moves = 0

    for action in teacher_actions:
        state = env.engine.get_state()
        mask = env.action_masks().astype(np.bool_)
        if action < 0 or action >= N_ACTIONS or not bool(mask[action]):
            rejected_moves += 1
            return EpisodeResult(
                map_name=map_name,
                episode_seed=job.episode_seed,
                teacher_used=teacher_used,
                obs=None,
                masks=None,
                actions=None,
                step_indices=None,
                kept=False,
                won=False,
                rejected_moves=rejected_moves,
            )

        if job.obs_mode == "oracle":
            obs = build_oracle_obs(
                state,
                step_count=env._step_count,
                max_steps=env.max_steps,
                include_map_layout=job.include_map_layout,
            )
        elif job.obs_mode == "state":
            obs = env._build_state_vector()
        else:
            raise ValueError(f"unknown obs_mode: {job.obs_mode}")
        episode_obs.append(obs)
        episode_masks.append(mask)
        episode_actions.append(action)

        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    final_state = env.engine.get_state()
    won = bool(final_state.won)
    if not episode_actions or not won:
        return EpisodeResult(
            map_name=map_name,
            episode_seed=job.episode_seed,
            teacher_used=teacher_used,
            obs=None,
            masks=None,
            actions=None,
            step_indices=None,
            kept=False,
            won=won,
            rejected_moves=rejected_moves,
        )

    return EpisodeResult(
        map_name=map_name,
        episode_seed=job.episode_seed,
        teacher_used=teacher_used,
        obs=np.stack(episode_obs).astype(np.float32),
        masks=np.stack(episode_masks).astype(np.bool_),
        actions=np.asarray(episode_actions, dtype=np.int64),
        step_indices=np.arange(len(episode_actions), dtype=np.int64),
        kept=True,
        won=won,
        rejected_moves=rejected_moves,
    )


def _run_jobs_serial(jobs: Sequence[EpisodeJob]) -> List[EpisodeResult]:
    return [_collect_episode(job) for job in jobs]


def _run_jobs_parallel(jobs: Sequence[EpisodeJob], num_workers: int) -> List[EpisodeResult]:
    ordered_results: List[Optional[EpisodeResult]] = [None] * len(jobs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(_collect_episode, job): idx
            for idx, job in enumerate(jobs)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            ordered_results[idx] = future.result()
    return [result for result in ordered_results if result is not None]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-maps", type=int, default=0)
    parser.add_argument("--seeds-per-map", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--max-cost", type=int, default=300)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument(
        "--teacher",
        choices=["solver", "autoplayer", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--include-map-layout", action="store_true", default=True)
    parser.add_argument("--obs-mode", choices=["oracle", "state"], default="oracle")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="0=auto, 1=serial, >1=process workers")
    args = parser.parse_args()

    if args.phase not in CURRICULUM:
        raise SystemExit(f"unknown phase: {args.phase}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    phase_cfg = CURRICULUM[args.phase]
    map_pool = get_map_pool(phase_cfg)
    if args.max_maps > 0:
        map_pool = map_pool[:args.max_maps]
    if not map_pool:
        raise SystemExit("no maps found")

    seed_manifest = load_seed_manifest()

    obs_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    action_list: List[int] = []
    map_names: List[str] = []
    seeds: List[int] = []
    step_indices: List[int] = []
    rejected_moves = 0
    mapping_failures = 0
    solved_episodes = 0
    kept_episodes = 0
    teacher_counts: Counter[str] = Counter()

    t0 = time.perf_counter()

    jobs: List[EpisodeJob] = []
    for map_path in map_pool:
        map_name = os.path.basename(map_path)
        for episode_seed in default_seeds_for_map(
            map_name,
            seed_manifest,
            args.seeds_per_map,
            args.base_seed,
        ):
            jobs.append(
                EpisodeJob(
                    map_path=map_path,
                    episode_seed=episode_seed,
                    teacher=args.teacher,
                    max_steps=phase_cfg["max_steps"],
                    include_map_layout=args.include_map_layout,
                    obs_mode=args.obs_mode,
                    max_cost=args.max_cost,
                    time_limit=args.time_limit,
                )
            )

    if args.num_workers == 1:
        episode_results = _run_jobs_serial(jobs)
        num_workers = 1
    else:
        auto_workers = max(1, (os.cpu_count() or 1) - 1)
        num_workers = args.num_workers if args.num_workers > 1 else auto_workers
        episode_results = _run_jobs_parallel(jobs, num_workers)

    episode_indices: List[int] = []
    for episode_idx, result in enumerate(episode_results):
        rejected_moves += result.rejected_moves
        if not result.kept or result.obs is None or result.masks is None or result.actions is None:
            continue

        obs_list.extend(result.obs)
        mask_list.extend(result.masks)
        action_list.extend(result.actions.tolist())
        map_names.extend([result.map_name] * len(result.actions))
        seeds.extend([result.episode_seed] * len(result.actions))
        step_indices.extend(result.step_indices.tolist())
        episode_indices.extend([episode_idx] * len(result.actions))
        kept_episodes += 1
        if result.won:
            solved_episodes += 1
        teacher_counts[result.teacher_used] += 1

    if not obs_list:
        raise SystemExit("no expert samples collected")

    obs_arr = np.stack(obs_list).astype(np.float32)
    mask_arr = np.stack(mask_list).astype(np.bool_)
    action_arr = np.asarray(action_list, dtype=np.int64)
    map_arr = np.asarray(map_names)
    seed_arr = np.asarray(seeds, dtype=np.int64)
    step_arr = np.asarray(step_indices, dtype=np.int64)

    np.savez_compressed(
        args.output,
        obs=obs_arr,
        masks=mask_arr,
        actions=action_arr,
        map_names=map_arr,
        seeds=seed_arr,
        step_indices=step_arr,
        episode_indices=np.asarray(episode_indices, dtype=np.int64),
        obs_mode=np.asarray(args.obs_mode),
    )

    summary = {
        "phase": args.phase,
        "output": os.path.abspath(args.output),
        "n_maps": len(map_pool),
        "seeds_per_map": args.seeds_per_map,
        "samples": int(len(action_arr)),
        "kept_episodes": kept_episodes,
        "solved_episodes": solved_episodes,
        "mapping_failures": mapping_failures,
        "rejected_moves": rejected_moves,
        "teacher_counts": dict(teacher_counts),
        "obs_mode": args.obs_mode,
        "num_workers": num_workers,
        "jobs": len(jobs),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    }
    summary_path = os.path.splitext(args.output)[0] + ".json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
