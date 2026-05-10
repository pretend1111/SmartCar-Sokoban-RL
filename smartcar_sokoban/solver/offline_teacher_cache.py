from __future__ import annotations

import contextlib
import gzip
import io
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from smartcar_sokoban.paths import PROJECT_ROOT
from smartcar_sokoban.solver.high_level_teacher import (
    TeacherAdvice,
    advise_exact_high_level,
)

TeacherSig = Tuple[Any, ...]
TeacherTable = Dict[TeacherSig, TeacherAdvice]
TeacherCacheBundle = Dict[str, Any]

CACHE_VERSION = 1


def normalize_map_key(map_path: str | os.PathLike[str]) -> str:
    path = Path(map_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()

    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def empty_cache_bundle() -> TeacherCacheBundle:
    return {
        "version": CACHE_VERSION,
        "maps": {},
        "stats": {},
        "config": {},
    }


def load_offline_teacher_cache_bundle(
        cache_path: str | os.PathLike[str]) -> TeacherCacheBundle:
    path = Path(cache_path)
    if not path.exists():
        return empty_cache_bundle()

    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "maps" in payload:
        bundle = empty_cache_bundle()
        bundle.update(payload)
        return bundle

    # Backward-compatible fallback if the file only stores the map table.
    bundle = empty_cache_bundle()
    bundle["maps"] = payload if isinstance(payload, dict) else {}
    return bundle


def save_offline_teacher_cache_bundle(
        cache_path: str | os.PathLike[str],
        bundle: TeacherCacheBundle) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


def select_teacher_cache_seeds(map_path: str,
                               seed_manifest: Dict[str, List[int]],
                               seeds_per_map: int) -> List[int]:
    basename = os.path.basename(map_path)
    verified = list(seed_manifest.get(basename, []))
    if verified:
        return verified[:min(len(verified), seeds_per_map)]
    return [i * 31 + 7 for i in range(seeds_per_map)]


def build_offline_teacher_cache(
        *,
        cache_path: str | os.PathLike[str],
        map_pool: Sequence[str],
        seed_manifest: Optional[Dict[str, List[int]]] = None,
        max_steps: int | Dict[str, int] = 60,
        base_dir: str = "",
        phase_name: str = "",
        seeds_per_map: int = 16,
        max_cost: int = 300,
        time_limit: float = 0.5,
        strategy: str = "auto") -> TeacherCacheBundle:
    from smartcar_sokoban.rl.high_level_env import SokobanHLEnv

    seed_manifest = seed_manifest or {}
    bundle = load_offline_teacher_cache_bundle(cache_path)
    bundle["version"] = CACHE_VERSION
    bundle["config"] = {
        "phase_name": phase_name,
        "seeds_per_map": seeds_per_map,
        "max_cost": max_cost,
        "time_limit": time_limit,
        "strategy": strategy,
    }

    total_maps = len(map_pool)
    for idx, map_path in enumerate(map_pool, start=1):
        map_key = normalize_map_key(map_path)
        table: TeacherTable = bundle["maps"].setdefault(map_key, {})
        map_stats = bundle["stats"].setdefault(
            map_key,
            {"completed_seeds": [], "episodes": 0, "states": 0},
        )
        completed_seeds = set(map_stats.get("completed_seeds", []))
        seeds = select_teacher_cache_seeds(map_path, seed_manifest, seeds_per_map)
        pending_seeds = [seed for seed in seeds if seed not in completed_seeds]

        if not pending_seeds:
            print(f"    Teacher cache [{idx}/{total_maps}] "
                  f"{os.path.basename(map_path)}: reuse {len(table)} states")
            continue

        env_max_steps = (max_steps.get(map_path, 60)
                         if isinstance(max_steps, dict) else max_steps)
        before_states = len(table)

        for seed in pending_seeds:
            env = SokobanHLEnv(
                map_file=map_path,
                base_dir=base_dir,
                max_steps=env_max_steps,
                include_map_layout=False,
                teacher_primary_reward=0.0,
                teacher_candidate_reward=0.0,
                teacher_mismatch_penalty=0.0,
                teacher_time_limit=0.0,
            )
            try:
                env.reset(seed=seed)
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    state = env.engine.get_state()
                    sig = env._teacher_signature(state)
                    advice = table.get(sig)
                    if advice is None:
                        with contextlib.redirect_stdout(io.StringIO()):
                            advice = advise_exact_high_level(
                                state,
                                max_cost=max_cost,
                                time_limit=time_limit,
                                strategy=strategy,
                            )
                        table[sig] = advice

                    if advice.primary_action is None:
                        break

                    _, _, terminated, truncated, _ = env.step(
                        advice.primary_action
                    )
            finally:
                env.close()

            map_stats["episodes"] = int(map_stats.get("episodes", 0)) + 1
            completed_seeds.add(seed)

        map_stats["completed_seeds"] = sorted(completed_seeds)
        map_stats["states"] = len(table)
        added_states = len(table) - before_states
        print(f"    Teacher cache [{idx}/{total_maps}] "
              f"{os.path.basename(map_path)}: "
              f"+{added_states} states, total={len(table)}, "
              f"seeds={len(map_stats['completed_seeds'])}")
        save_offline_teacher_cache_bundle(cache_path, bundle)

    return bundle
