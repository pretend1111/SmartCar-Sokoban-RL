"""P3 — 数据集构建 v2：以 P1.2 verify_optimal.py 输出的 verified.json 为输入.

特性:
  * 每张图只用 verified_seed (而不是默认 42 / manifest)
  * teacher 默认 "solver_ida" (strategy='ida' strict optimal)
  * 不退回 AutoPlayer (避免污染监督信号), 失败 episode 直接丢
  * 多 worker 并行, ProcessPoolExecutor
  * 输出格式跟 build_dataset.py 一致, 直接复用 train_bc.py

用法:
  python -m experiments.solver_bc.build_dataset_v2 \
    --verified assets/maps/phase6_verified.json \
    --output .agent/data/p6_v2.npz \
    --teacher solver_ida --time-limit 60 --max-cost 200 \
    --num-workers 18 \
    --extra-seeds 1 \  # 0 = 仅 verified_seed; >0 = 多加几个 seed (用 base+i*9973)
    --max-maps 0
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.solver_bc.oracle_features import build_oracle_obs
from experiments.solver_bc.teachers import collect_teacher_actions
from smartcar_sokoban.paths import MAPS_ROOT
from smartcar_sokoban.rl.high_level_env import N_ACTIONS, SokobanHLEnv


@dataclass(frozen=True)
class EpisodeJob:
    map_path: str
    episode_seed: int
    teacher: str
    strategy: str
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
    reason: str


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
        strategy=job.strategy,
    )
    map_name = os.path.basename(job.map_path)
    if not teacher_actions:
        return EpisodeResult(
            map_name=map_name, episode_seed=job.episode_seed,
            teacher_used=teacher_used,
            obs=None, masks=None, actions=None, step_indices=None,
            kept=False, won=False, reason="teacher_no_actions",
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

    for action in teacher_actions:
        state = env.engine.get_state()
        mask = env.action_masks().astype(np.bool_)
        if action < 0 or action >= N_ACTIONS or not bool(mask[action]):
            return EpisodeResult(
                map_name=map_name, episode_seed=job.episode_seed,
                teacher_used=teacher_used,
                obs=None, masks=None, actions=None, step_indices=None,
                kept=False, won=False, reason="invalid_teacher_action",
            )

        if job.obs_mode == "oracle":
            obs = build_oracle_obs(
                state, step_count=env._step_count,
                max_steps=env.max_steps,
                include_map_layout=job.include_map_layout,
            )
        elif job.obs_mode == "state":
            obs = env._build_state_vector()
        else:
            return EpisodeResult(
                map_name=map_name, episode_seed=job.episode_seed,
                teacher_used=teacher_used,
                obs=None, masks=None, actions=None, step_indices=None,
                kept=False, won=False, reason=f"bad_obs_mode:{job.obs_mode}",
            )

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
            map_name=map_name, episode_seed=job.episode_seed,
            teacher_used=teacher_used,
            obs=None, masks=None, actions=None, step_indices=None,
            kept=False, won=won,
            reason="not_won" if not won else "no_actions",
        )

    return EpisodeResult(
        map_name=map_name, episode_seed=job.episode_seed,
        teacher_used=teacher_used,
        obs=np.stack(episode_obs).astype(np.float32),
        masks=np.stack(episode_masks).astype(np.bool_),
        actions=np.asarray(episode_actions, dtype=np.int64),
        step_indices=np.arange(len(episode_actions), dtype=np.int64),
        kept=True, won=True, reason="ok",
    )


def _resolve_map_path(map_name: str, default_phase_dir: Optional[str]) -> str:
    """先尝试 default_phase_dir/map_name, 不存在再扫所有 MAPS_ROOT/phase*."""
    if default_phase_dir:
        p = os.path.join(default_phase_dir, map_name)
        if os.path.exists(p):
            return p
    for sub in os.listdir(MAPS_ROOT):
        candidate = MAPS_ROOT / sub / map_name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(map_name)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verified", required=True,
                   help="verify_optimal.py 输出的 JSON")
    p.add_argument("--output", required=True)
    p.add_argument("--teacher", choices=["solver", "solver_ida", "hybrid"],
                   default="solver_ida")
    p.add_argument("--strategy", default="auto",
                   help="MultiBoxSolver strategy (auto / ida / best_first); "
                        "若 teacher=solver_ida 此参数被覆盖为 ida")
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--max-cost", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=120,
                   help="env max_steps (高层动作上限); phase 6 推荐 100-150")
    p.add_argument("--include-map-layout", action="store_true", default=True)
    p.add_argument("--obs-mode", choices=["oracle", "state"], default="oracle")
    p.add_argument("--extra-seeds", type=int, default=0,
                   help="除 verified_seed 外, 再用 base+i*9973 试 i=1..N 个 seed")
    p.add_argument("--base-seed", type=int, default=7)
    p.add_argument("--max-maps", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    with open(args.verified, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    phase = manifest.get("phase")
    phase_dir = (str(MAPS_ROOT / f"phase{phase}") if phase else None)

    ok_entries = [r for r in manifest["results"] if r.get("status") == "ok"]
    if args.max_maps > 0:
        ok_entries = ok_entries[:args.max_maps]

    if not ok_entries:
        raise SystemExit(f"no verified maps in {args.verified}")

    print(f"[build_dataset_v2] verified={len(ok_entries)} from {args.verified}",
          flush=True)

    jobs: List[EpisodeJob] = []
    for entry in ok_entries:
        try:
            map_path = _resolve_map_path(entry["map"], phase_dir)
        except FileNotFoundError:
            print(f"  ⚠️ map missing: {entry['map']}", flush=True)
            continue
        seeds = [int(entry["verified_seed"])]
        for i in range(1, args.extra_seeds + 1):
            seeds.append(args.base_seed + i * 9973)
        for seed in seeds:
            jobs.append(EpisodeJob(
                map_path=map_path,
                episode_seed=seed,
                teacher=args.teacher,
                strategy=args.strategy,
                max_steps=args.max_steps,
                include_map_layout=args.include_map_layout,
                obs_mode=args.obs_mode,
                max_cost=args.max_cost,
                time_limit=args.time_limit,
            ))

    print(f"[build_dataset_v2] jobs={len(jobs)} (verified+{args.extra_seeds} extra seeds each)",
          flush=True)

    workers = (args.num_workers if args.num_workers > 0
               else max(1, (os.cpu_count() or 4) - 2))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    obs_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    action_list: List[int] = []
    map_names: List[str] = []
    seeds_list: List[int] = []
    step_indices: List[int] = []
    episode_indices: List[int] = []
    teacher_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    kept = 0
    solved = 0

    t0 = time.time()
    done = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_collect_episode, j) for j in jobs]
        for ep_idx, fut in enumerate(concurrent.futures.as_completed(futs)):
            try:
                r = fut.result()
            except Exception as e:
                r = EpisodeResult(
                    map_name="?", episode_seed=-1, teacher_used="?",
                    obs=None, masks=None, actions=None, step_indices=None,
                    kept=False, won=False, reason=f"crash:{e}",
                )
            done += 1
            reason_counts[r.reason] += 1
            if r.kept and r.obs is not None and r.actions is not None:
                obs_list.extend(r.obs)
                mask_list.extend(r.masks)
                action_list.extend(r.actions.tolist())
                map_names.extend([r.map_name] * len(r.actions))
                seeds_list.extend([r.episode_seed] * len(r.actions))
                step_indices.extend(r.step_indices.tolist())
                episode_indices.extend([ep_idx] * len(r.actions))
                kept += 1
                if r.won:
                    solved += 1
                teacher_counts[r.teacher_used] += 1
            if done % 25 == 0 or done == len(jobs):
                ela = time.time() - t0
                eta = (ela / done * (len(jobs) - done)) if done else 0
                print(f"  [{done}/{len(jobs)}] kept={kept} solved={solved} "
                      f"samples={len(action_list)} elapsed={int(ela)}s ETA={int(eta)}s "
                      f"reason_top={reason_counts.most_common(3)}",
                      flush=True)

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
        "verified": os.path.abspath(args.verified),
        "output": os.path.abspath(args.output),
        "phase": phase,
        "n_verified_maps": len(ok_entries),
        "extra_seeds": args.extra_seeds,
        "n_jobs": len(jobs),
        "kept_episodes": kept,
        "solved_episodes": solved,
        "samples": int(len(action_arr)),
        "teacher": args.teacher,
        "strategy": args.strategy,
        "time_limit": args.time_limit,
        "teacher_counts": dict(teacher_counts),
        "reason_counts": dict(reason_counts),
        "obs_mode": args.obs_mode,
        "num_workers": workers,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    summary_path = os.path.splitext(args.output)[0] + ".json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
