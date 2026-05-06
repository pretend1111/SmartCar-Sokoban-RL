from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.solver_bc.branch_search import (
    RolloutRequest,
    RolloutResult,
    default_seeds_for_map,
    resolve_device,
    rollout_with_prefixes,
    save_improved_dataset,
    search_requests,
)
from experiments.solver_bc.train_bc import MaskedBCPolicy
from smartcar_sokoban.rl.train import CURRICULUM, get_map_pool, load_seed_manifest


def load_policy(checkpoint: str, device: torch.device) -> Tuple[Dict[str, object], MaskedBCPolicy]:
    payload = torch.load(checkpoint, map_location="cpu")
    policy_kind = str(payload.get("policy", "mlp")).lower()
    if policy_kind == "conv":
        from experiments.solver_bc.policy_conv import MaskedConvBCPolicy
        model = MaskedConvBCPolicy(
            n_actions=int(payload["n_actions"]),
            hidden_dim=int(payload["hidden_dim"]),
            wall_emb_dim=int(payload.get("wall_emb_dim", 64)),
        )
    else:
        model = MaskedBCPolicy(payload["obs_dim"], payload["n_actions"], payload["hidden_dim"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return payload, model


def evaluate_policy(model: MaskedBCPolicy, payload: Dict[str, object], phase: int,
                    seeds_per_map: int, base_seed: int, device: torch.device,
                    rollout_batch_size: int,
                    rollout_backend: str) -> Dict[str, object]:
    phase_cfg = CURRICULUM[phase]
    map_pool = get_map_pool(phase_cfg)
    seed_manifest = load_seed_manifest()

    episode_rows: List[Dict[str, object]] = []
    map_rows: List[Dict[str, object]] = []
    total_wins = 0
    total_episodes = 0
    requests: List[RolloutRequest] = []
    request_meta: List[Tuple[str, int]] = []

    for map_path in map_pool:
        map_name = os.path.basename(map_path)
        seeds_to_try = default_seeds_for_map(
            map_name,
            seed_manifest,
            seeds_per_map,
            base_seed,
        )
        for episode_seed in seeds_to_try:
            requests.append(
                RolloutRequest(
                    map_path=map_path,
                    episode_seed=episode_seed,
                    prefix=(),
                )
            )
            request_meta.append((map_name, episode_seed))

    results = rollout_with_prefixes(
        model,
        payload,
        phase,
        requests,
        device=device,
        branch_top_k=1,
        inference_batch_size=rollout_batch_size,
        rollout_backend=rollout_backend,
    )

    per_map_rows: Dict[str, List[Tuple[int, RolloutResult]]] = defaultdict(list)
    for (map_name, episode_seed), result in zip(request_meta, results):
        per_map_rows[map_name].append((episode_seed, result))

    for map_path in map_pool:
        map_name = os.path.basename(map_path)
        rows = per_map_rows.get(map_name, [])
        wins = 0
        low_steps_if_win: List[int] = []
        high_steps_if_win: List[int] = []

        for episode_seed, result in rows:
            row = {
                "map": map_name,
                "seed": episode_seed,
                "won": result.won,
                "low_steps": (result.low_steps if result.won else None),
                "high_steps": result.high_steps,
                "truncated_reason": result.truncated_reason,
            }
            episode_rows.append(row)
            total_episodes += 1
            if result.won:
                wins += 1
                total_wins += 1
                low_steps_if_win.append(result.low_steps)
                high_steps_if_win.append(result.high_steps)

        map_rows.append(
            {
                "map": map_name,
                "episodes": len(rows),
                "wins": wins,
                "win_rate": wins / max(len(rows), 1),
                "avg_low_steps_if_win": (
                    round(sum(low_steps_if_win) / len(low_steps_if_win), 2)
                    if low_steps_if_win else None
                ),
                "avg_high_steps_if_win": (
                    round(sum(high_steps_if_win) / len(high_steps_if_win), 2)
                    if high_steps_if_win else None
                ),
            }
        )

    return {
        "phase": phase,
        "episodes": total_episodes,
        "wins": total_wins,
        "win_rate": total_wins / max(total_episodes, 1),
        "maps": map_rows,
        "episodes_detail": episode_rows,
    }


def map_hardness_key(row: Dict[str, object]) -> Tuple[float, float, float, str]:
    win_rate = float(row["win_rate"])
    low_steps = row["avg_low_steps_if_win"]
    high_steps = row["avg_high_steps_if_win"]
    low_rank = -(float(low_steps) if low_steps is not None else 1e9)
    high_rank = -(float(high_steps) if high_steps is not None else 1e9)
    return win_rate, low_rank, high_rank, str(row["map"])


def select_hard_maps(eval_summary: Dict[str, object], top_maps: int) -> List[str]:
    rows = list(eval_summary["maps"])
    rows.sort(key=map_hardness_key)
    return [str(row["map"]) for row in rows[:top_maps]]


def merge_dataset_pair(base_data: Dict[str, np.ndarray],
                       improved_data: Dict[str, np.ndarray],
                       replace_keys: Sequence[Tuple[str, int]]) -> Dict[str, np.ndarray]:
    replace_set = set(replace_keys)
    base_map_names = base_data["map_names"].astype(object)
    base_seeds = base_data["seeds"].astype(np.int64)
    keep_mask = np.array(
        [(str(map_name), int(seed)) not in replace_set
         for map_name, seed in zip(base_map_names, base_seeds)],
        dtype=np.bool_,
    )

    merged = {
        "obs": np.concatenate([base_data["obs"][keep_mask], improved_data["obs"]], axis=0),
        "masks": np.concatenate([base_data["masks"][keep_mask], improved_data["masks"]], axis=0),
        "actions": np.concatenate([base_data["actions"][keep_mask], improved_data["actions"]], axis=0),
        "map_names": np.concatenate([base_map_names[keep_mask], improved_data["map_names"]], axis=0),
        "seeds": np.concatenate([base_seeds[keep_mask], improved_data["seeds"]], axis=0),
        "step_indices": np.concatenate([base_data["step_indices"][keep_mask], improved_data["step_indices"]], axis=0),
    }
    return merged


def save_dataset(path: str, merged: Dict[str, np.ndarray]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        path,
        obs=np.asarray(merged["obs"], dtype=np.float32),
        masks=np.asarray(merged["masks"], dtype=np.bool_),
        actions=np.asarray(merged["actions"], dtype=np.int64),
        map_names=np.asarray(merged["map_names"], dtype=object),
        seeds=np.asarray(merged["seeds"], dtype=np.int64),
        step_indices=np.asarray(merged["step_indices"], dtype=np.int64),
    )


def train_from_dataset(dataset_path: str, output_dir: str, epochs: int, batch_size: int,
                       lr: float, hidden_dim: int, device: str, seed: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    cmd = [
        sys.executable,
        os.path.join(ROOT, "experiments", "solver_bc", "train_bc.py"),
        "--dataset", dataset_path,
        "--output-dir", output_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--hidden-dim", str(hidden_dim),
        "--val-ratio", "0",
        "--device", device,
        "--seed", str(seed),
    ]
    with open(log_path, "w", encoding="utf-8") as log_fh:
        subprocess.run(
            cmd,
            cwd=ROOT,
            check=True,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    return os.path.join(output_dir, "best.pt")


def save_json(path: str, payload: Dict[str, object]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--top-maps", type=int, default=1)
    parser.add_argument("--seeds-per-map", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--rollout-backend", default="auto",
                        choices=["auto", "cpu", "gpu"])
    parser.add_argument("--branch-budget", type=int, default=128)
    parser.add_argument("--branches-per-rollout", type=int, default=8)
    parser.add_argument("--branch-top-k", type=int, default=2)
    parser.add_argument("--frontier-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.phase not in CURRICULUM:
        raise SystemExit(f"unknown phase: {args.phase}")
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)
    current_checkpoint = os.path.abspath(args.checkpoint)
    current_dataset = os.path.abspath(args.dataset)
    loop_rows: List[Dict[str, object]] = []

    for iteration in range(1, args.iterations + 1):
        iter_dir = os.path.join(args.output_dir, f"iter_{iteration:02d}")
        os.makedirs(iter_dir, exist_ok=True)

        payload, model = load_policy(current_checkpoint, device)
        pre_eval = evaluate_policy(
            model,
            payload,
            args.phase,
            seeds_per_map=args.seeds_per_map,
            base_seed=args.base_seed,
            device=device,
            rollout_batch_size=args.rollout_batch_size,
            rollout_backend=args.rollout_backend,
        )
        save_json(os.path.join(iter_dir, "pre_eval.json"), pre_eval)

        selected_maps = select_hard_maps(pre_eval, args.top_maps)
        improved_rollouts: List[Tuple[str, int, RolloutResult]] = []
        branch_rows: List[Dict[str, object]] = []
        total_extra_rollouts = 0
        seed_manifest = load_seed_manifest()
        phase_cfg = CURRICULUM[args.phase]
        search_rows: List[RolloutRequest] = []
        search_meta: List[Tuple[str, int]] = []

        for map_path in get_map_pool(phase_cfg):
            map_name = os.path.basename(map_path)
            if map_name not in selected_maps:
                continue
            seeds_to_try = default_seeds_for_map(
                map_name,
                seed_manifest,
                args.seeds_per_map,
                args.base_seed,
            )
            for episode_seed in seeds_to_try:
                search_rows.append(
                    RolloutRequest(
                        map_path=map_path,
                        episode_seed=episode_seed,
                        prefix=(),
                    )
                )
                search_meta.append((map_name, episode_seed))

        batched_results = search_requests(
            model,
            payload,
            args.phase,
            search_rows,
            device=device,
            branch_budget=args.branch_budget,
            branches_per_rollout=args.branches_per_rollout,
            branch_top_k=args.branch_top_k,
            rollout_batch_size=args.rollout_batch_size,
            rollout_backend=args.rollout_backend,
            frontier_batch_size=args.frontier_batch_size,
        )
        for (map_name, episode_seed), (_, baseline, best, extra_rollouts, unique_paths) in zip(
            search_meta,
            batched_results,
        ):
                total_extra_rollouts += extra_rollouts
                improved = best is not None and (
                    not baseline.won or best.low_steps < baseline.low_steps
                )
                row = {
                    "map": map_name,
                    "seed": episode_seed,
                    "baseline_win": baseline.won,
                    "baseline_low_steps": (baseline.low_steps if baseline.won else None),
                    "baseline_high_steps": baseline.high_steps,
                    "best_win": bool(best and best.won),
                    "best_low_steps": (best.low_steps if best and best.won else None),
                    "best_high_steps": (best.high_steps if best and best.won else None),
                    "improved": improved,
                    "improvement_low_steps": (
                        baseline.low_steps - best.low_steps
                        if improved and baseline.won and best is not None
                        else None
                    ),
                    "extra_rollouts": extra_rollouts,
                    "unique_paths": unique_paths,
                }
                branch_rows.append(row)
                if improved and best is not None:
                    improved_rollouts.append((map_name, episode_seed, best))

        branch_summary = {
            "phase": args.phase,
            "selected_maps": selected_maps,
            "episodes": len(branch_rows),
            "improved_episodes": len(improved_rollouts),
            "avg_extra_rollouts": total_extra_rollouts / max(len(branch_rows), 1),
            "summaries": branch_rows,
        }
        branch_json = os.path.join(iter_dir, "branch_search.json")
        save_json(branch_json, branch_summary)

        improved_npz = os.path.join(iter_dir, "improved_samples.npz")
        if improved_rollouts:
            save_improved_dataset(improved_npz, improved_rollouts)

        merged_dataset = current_dataset
        next_checkpoint = current_checkpoint
        post_eval: Optional[Dict[str, object]] = None

        if improved_rollouts:
            base_data = dict(np.load(current_dataset, allow_pickle=True))
            improved_data = dict(np.load(improved_npz, allow_pickle=True))
            replace_keys = sorted({
                (str(map_name), int(seed))
                for map_name, seed, _ in improved_rollouts
            })
            merged = merge_dataset_pair(base_data, improved_data, replace_keys)
            merged_dataset = os.path.join(iter_dir, "merged_dataset.npz")
            save_dataset(merged_dataset, merged)
            save_json(
                os.path.join(iter_dir, "merge_summary.json"),
                {
                    "base_dataset": current_dataset,
                    "improved_dataset": improved_npz,
                    "merged_dataset": merged_dataset,
                    "replace_keys": replace_keys,
                    "base_samples": int(base_data["actions"].shape[0]),
                    "improved_samples": int(improved_data["actions"].shape[0]),
                    "merged_samples": int(merged["actions"].shape[0]),
                },
            )

            train_dir = os.path.join(iter_dir, "train")
            next_checkpoint = train_from_dataset(
                merged_dataset,
                train_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                device=str(device),
                seed=args.seed,
            )
            payload, model = load_policy(next_checkpoint, device)
            post_eval = evaluate_policy(
                model,
                payload,
                args.phase,
                seeds_per_map=args.seeds_per_map,
                base_seed=args.base_seed,
                device=device,
                rollout_batch_size=args.rollout_batch_size,
                rollout_backend=args.rollout_backend,
            )
            save_json(os.path.join(iter_dir, "post_eval.json"), post_eval)

        loop_rows.append(
            {
                "iteration": iteration,
                "checkpoint_in": current_checkpoint,
                "dataset_in": current_dataset,
                "selected_maps": selected_maps,
                "branch_json": branch_json,
                "improved_episodes": len(improved_rollouts),
                "checkpoint_out": next_checkpoint,
                "dataset_out": merged_dataset,
                "pre_eval_win_rate": pre_eval["win_rate"],
                "post_eval_win_rate": (post_eval["win_rate"] if post_eval else pre_eval["win_rate"]),
            }
        )

        current_checkpoint = next_checkpoint
        current_dataset = merged_dataset

        if not improved_rollouts:
            break

    final_summary = {
        "phase": args.phase,
        "iterations_requested": args.iterations,
        "iterations_completed": len(loop_rows),
        "final_checkpoint": current_checkpoint,
        "final_dataset": current_dataset,
        "rows": loop_rows,
    }
    save_json(os.path.join(args.output_dir, "summary.json"), final_summary)
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
