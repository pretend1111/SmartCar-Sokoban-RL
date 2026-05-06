from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.solver_bc.oracle_features import build_oracle_obs
from experiments.solver_bc.train_bc import MaskedBCPolicy
from smartcar_sokoban.rl.high_level_env import SokobanHLEnv
from smartcar_sokoban.rl.train import CURRICULUM, get_map_pool, load_seed_manifest


def default_seeds_for_map(map_name: str, seed_manifest: Dict[str, List[int]],
                          seeds_per_map: int, base_seed: int) -> List[int]:
    verified = list(seed_manifest.get(map_name, []))
    if verified:
        return verified[:seeds_per_map]
    return [base_seed + 9973 * i for i in range(seeds_per_map)]


@dataclass
class StepDecision:
    step: int
    action: int
    action_prob: float
    top_actions: List[int]
    top_probs: List[float]
    margin: float
    entropy: float
    forced: bool


@dataclass
class RolloutResult:
    prefix: Tuple[int, ...]
    actions: List[int]
    decisions: List[StepDecision]
    obs: List[np.ndarray]
    masks: List[np.ndarray]
    won: bool
    low_steps: int
    high_steps: int
    truncated_reason: str
    invalid_prefix_at: int = -1


@dataclass(frozen=True)
class RolloutRequest:
    map_path: str
    episode_seed: int
    prefix: Tuple[int, ...] = ()


@dataclass
class _ActiveRollout:
    request: RolloutRequest
    env: SokobanHLEnv
    actions: List[int]
    decisions: List[StepDecision]
    obs_trace: List[np.ndarray]
    mask_trace: List[np.ndarray]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def rank_valid_actions_batch(model: MaskedBCPolicy, obs_batch: Sequence[np.ndarray],
                             mask_batch: Sequence[np.ndarray], device: torch.device,
                             top_k: int) -> List[Tuple[List[int], List[float], float, float]]:
    if not obs_batch:
        return []

    obs_np = np.asarray(obs_batch, dtype=np.float32)
    mask_np = np.asarray(mask_batch, dtype=np.bool_)
    obs_t = torch.from_numpy(obs_np).to(device=device, non_blocking=True)
    mask_t = torch.from_numpy(mask_np).to(device=device, non_blocking=True)

    with torch.no_grad():
        logits = model(obs_t)
        masked_logits = logits.masked_fill(~mask_t, -1e9)
        probs = torch.softmax(masked_logits, dim=1)
        probs = probs * mask_t.to(dtype=probs.dtype)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        top_count = min(top_k, int(probs.shape[1]))
        top_probs_t, top_actions_t = torch.topk(probs, k=top_count, dim=1)

    results: List[Tuple[List[int], List[float], float, float]] = []
    entropy_np = entropy.detach().cpu().numpy()
    top_actions_np = top_actions_t.detach().cpu().numpy()
    top_probs_np = top_probs_t.detach().cpu().numpy()
    valid_counts = mask_np.sum(axis=1).astype(np.int64)

    for row_idx in range(obs_np.shape[0]):
        valid_count = min(int(valid_counts[row_idx]), top_count)
        row_actions = [int(x) for x in top_actions_np[row_idx, :valid_count]]
        row_probs = [float(x) for x in top_probs_np[row_idx, :valid_count]]
        margin = 1.0 if len(row_probs) < 2 else float(row_probs[0] - row_probs[1])
        results.append((row_actions, row_probs, margin, float(entropy_np[row_idx])))
    return results


def _resolve_rollout_backend(
    rollout_backend: str,
    device: torch.device,
    include_map_layout: bool,
) -> str:
    backend = str(rollout_backend).strip().lower()
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"unknown rollout backend: {rollout_backend}")
    if backend == "auto":
        return "gpu" if device.type == "cuda" else "cpu"
    return backend


def _checkpoint_obs_mode(checkpoint_payload: Dict[str, object]) -> str:
    mode = str(checkpoint_payload.get("obs_mode", "oracle")).strip().lower()
    if mode not in {"oracle", "state"}:
        raise ValueError(f"unknown checkpoint obs_mode: {mode}")
    return mode


def _build_model_obs(env: SokobanHLEnv, checkpoint_payload: Dict[str, object]) -> np.ndarray:
    include_map_layout = bool(checkpoint_payload["include_map_layout"])
    obs_mode = _checkpoint_obs_mode(checkpoint_payload)
    if obs_mode == "oracle":
        state = env.engine.get_state()
        return build_oracle_obs(
            state,
            step_count=env._step_count,
            max_steps=env.max_steps,
            include_map_layout=include_map_layout,
        )
    return env._build_state_vector()


def rollout_with_prefixes_cpu(model: MaskedBCPolicy, checkpoint_payload: Dict[str, object],
                              phase: int, requests: Sequence[RolloutRequest],
                              device: torch.device, branch_top_k: int,
                              inference_batch_size: int,
                              capture_traces: bool = True) -> List[RolloutResult]:
    if not requests:
        return []
    inference_batch_size = max(int(inference_batch_size), 1)
    include_map_layout = bool(checkpoint_payload["include_map_layout"])
    phase_cfg = CURRICULUM[phase]

    active: List[_ActiveRollout] = []
    for request in requests:
        random.seed(request.episode_seed)
        env = SokobanHLEnv(
            map_file=request.map_path,
            base_dir=ROOT,
            max_steps=phase_cfg["max_steps"],
            include_map_layout=include_map_layout,
        )
        env.reset(seed=request.episode_seed)
        active.append(
            _ActiveRollout(
                request=request,
                env=env,
                actions=[],
                decisions=[],
                obs_trace=[],
                mask_trace=[],
            )
        )

    results: List[Optional[RolloutResult]] = [None] * len(active)
    active_indices = list(range(len(active)))

    while active_indices:
        next_active: List[int] = []
        for start in range(0, len(active_indices), inference_batch_size):
            chunk_indices = active_indices[start:start + inference_batch_size]
            obs_batch: List[np.ndarray] = []
            mask_batch: List[np.ndarray] = []

            for idx in chunk_indices:
                tracker = active[idx]
                obs_batch.append(_build_model_obs(tracker.env, checkpoint_payload))
                mask_batch.append(tracker.env.action_masks())

            ranked_batch = rank_valid_actions_batch(
                model,
                obs_batch,
                mask_batch,
                device,
                top_k=branch_top_k,
            )
            ready_steps: List[Tuple[int, int]] = []

            for local_idx, idx in enumerate(chunk_indices):
                tracker = active[idx]
                obs = obs_batch[local_idx]
                mask = mask_batch[local_idx]
                top_actions, top_probs, margin, entropy = ranked_batch[local_idx]
                step_idx = len(tracker.actions)
                forced = step_idx < len(tracker.request.prefix)

                if forced:
                    action = int(tracker.request.prefix[step_idx])
                    if action < 0 or action >= len(mask) or not mask[action]:
                        results[idx] = RolloutResult(
                            prefix=tracker.request.prefix,
                            actions=list(tracker.actions),
                            decisions=list(tracker.decisions),
                            obs=list(tracker.obs_trace),
                            masks=list(tracker.mask_trace),
                            won=False,
                            low_steps=int(10**9),
                            high_steps=len(tracker.actions),
                            truncated_reason="invalid_prefix",
                            invalid_prefix_at=step_idx,
                        )
                        continue
                    action_prob = 0.0
                    for candidate_idx, candidate in enumerate(top_actions):
                        if candidate == action:
                            action_prob = top_probs[candidate_idx]
                            break
                else:
                    if not top_actions:
                        results[idx] = RolloutResult(
                            prefix=tracker.request.prefix,
                            actions=list(tracker.actions),
                            decisions=list(tracker.decisions),
                            obs=list(tracker.obs_trace),
                            masks=list(tracker.mask_trace),
                            won=False,
                            low_steps=int(10**9),
                            high_steps=len(tracker.actions),
                            truncated_reason="no_valid_actions",
                            invalid_prefix_at=-1,
                        )
                        continue
                    action = top_actions[0]
                    action_prob = top_probs[0]

                if capture_traces:
                    tracker.obs_trace.append(obs.copy())
                    tracker.mask_trace.append(mask.copy())
                tracker.decisions.append(
                    StepDecision(
                        step=step_idx,
                        action=action,
                        action_prob=float(action_prob),
                        top_actions=top_actions,
                        top_probs=top_probs,
                        margin=margin,
                        entropy=entropy,
                        forced=forced,
                    )
                )
                tracker.actions.append(action)
                ready_steps.append((idx, action))

            for idx, action in ready_steps:
                tracker = active[idx]
                _, _, terminated, truncated, info = tracker.env.step(action)
                if terminated or truncated:
                    results[idx] = RolloutResult(
                        prefix=tracker.request.prefix,
                        actions=list(tracker.actions),
                        decisions=list(tracker.decisions),
                        obs=list(tracker.obs_trace),
                        masks=list(tracker.mask_trace),
                        won=bool(tracker.env.engine.get_state().won),
                        low_steps=int(tracker.env._total_low_steps),
                        high_steps=len(tracker.actions),
                        truncated_reason=str(info.get("truncated_reason", "")),
                        invalid_prefix_at=-1,
                    )
                else:
                    next_active.append(idx)
        active_indices = next_active

    if any(result is None for result in results):
        raise RuntimeError("rollout_with_prefixes finished with incomplete results")
    return [result for result in results if result is not None]


def rollout_with_prefixes(model: MaskedBCPolicy, checkpoint_payload: Dict[str, object],
                          phase: int, requests: Sequence[RolloutRequest],
                          device: torch.device, branch_top_k: int,
                          inference_batch_size: int,
                          rollout_backend: str = "auto",
                          capture_traces: bool = True) -> List[RolloutResult]:
    include_map_layout = bool(checkpoint_payload["include_map_layout"])
    backend = _resolve_rollout_backend(
        rollout_backend=rollout_backend,
        device=device,
        include_map_layout=include_map_layout,
    )
    if backend == "gpu":
        from experiments.gpu_sim.gpu_rollout import rollout_with_prefixes_gpu

        return rollout_with_prefixes_gpu(
            model,
            checkpoint_payload,
            phase,
            requests,
            device=device,
            branch_top_k=branch_top_k,
            capture_traces=capture_traces,
        )
    return rollout_with_prefixes_cpu(
        model,
        checkpoint_payload,
        phase,
        requests,
        device=device,
        branch_top_k=branch_top_k,
        inference_batch_size=inference_batch_size,
        capture_traces=capture_traces,
    )


def rollout_with_prefix(model: MaskedBCPolicy, checkpoint_payload: Dict[str, object],
                        phase: int, map_path: str, episode_seed: int,
                        prefix: Sequence[int], device: torch.device,
                        branch_top_k: int, inference_batch_size: int = 64,
                        rollout_backend: str = "auto",
                        capture_traces: bool = True) -> RolloutResult:
    return rollout_with_prefixes(
        model,
        checkpoint_payload,
        phase,
        [RolloutRequest(map_path=map_path, episode_seed=episode_seed, prefix=tuple(prefix))],
        device=device,
        branch_top_k=branch_top_k,
        inference_batch_size=inference_batch_size,
        rollout_backend=rollout_backend,
        capture_traces=capture_traces,
    )[0]


def better_rollout(candidate: RolloutResult,
                   incumbent: Optional[RolloutResult]) -> bool:
    if not candidate.won:
        return False
    if incumbent is None or not incumbent.won:
        return True
    if candidate.low_steps != incumbent.low_steps:
        return candidate.low_steps < incumbent.low_steps
    return candidate.high_steps < incumbent.high_steps


def branch_candidates(result: RolloutResult) -> List[Tuple[int, int, float, float]]:
    ranked: List[Tuple[int, int, float, float]] = []
    for step in result.decisions:
        if len(step.top_actions) < 2:
            continue
        ranked.append((step.step, step.top_actions[1], step.margin, step.entropy))
    ranked.sort(key=lambda item: (item[2], -item[3], item[0]))
    return ranked


def frontier_key(result: RolloutResult) -> Tuple[int, int, int, float]:
    won_rank = 0 if result.won else 1
    low_steps = result.low_steps if result.won else 10**9
    min_margin = min((step.margin for step in result.decisions if len(step.top_actions) >= 2),
                     default=1.0)
    return won_rank, low_steps, result.high_steps, min_margin


def search_requests(model: MaskedBCPolicy, checkpoint_payload: Dict[str, object],
                    phase: int, requests: Sequence[RolloutRequest],
                    device: torch.device, branch_budget: int,
                    branches_per_rollout: int,
                    branch_top_k: int,
                    rollout_batch_size: int = 64,
                    rollout_backend: str = "auto",
                    frontier_batch_size: int = 1
                    ) -> List[Tuple[RolloutRequest, RolloutResult, Optional[RolloutResult], int, int]]:
    base_requests = list(requests)
    if not base_requests:
        return []

    baseline_results = rollout_with_prefixes(
        model,
        checkpoint_payload,
        phase,
        base_requests,
        device=device,
        branch_top_k=branch_top_k,
        inference_batch_size=rollout_batch_size,
        rollout_backend=rollout_backend,
        capture_traces=False,
    )

    best_by_episode: List[Optional[RolloutResult]] = [
        baseline if baseline.won else None
        for baseline in baseline_results
    ]
    frontier_by_episode: List[List[RolloutResult]] = [[baseline] for baseline in baseline_results]
    tried_prefixes = [{tuple(request.prefix)} for request in base_requests]
    seen_rollouts = [{tuple(baseline.actions)} for baseline in baseline_results]
    extra_rollouts = [0 for _ in base_requests]
    frontier_batch_size = max(int(frontier_batch_size), 1)

    while True:
        batch_requests: List[RolloutRequest] = []
        batch_owner: List[int] = []

        for episode_idx, base_request in enumerate(base_requests):
            frontier = frontier_by_episode[episode_idx]
            if not frontier or extra_rollouts[episode_idx] >= branch_budget:
                continue

            frontier.sort(key=frontier_key)
            current_batch = frontier[:frontier_batch_size]
            frontier_by_episode[episode_idx] = frontier[frontier_batch_size:]

            remaining_budget = branch_budget - extra_rollouts[episode_idx]
            planned_for_episode = 0
            for current in current_batch:
                branches_from_current = 0
                for step_idx, alt_action, _, _ in branch_candidates(current):
                    if branches_from_current >= branches_per_rollout:
                        break
                    if planned_for_episode >= remaining_budget:
                        break

                    new_prefix = tuple(current.actions[:step_idx] + [alt_action])
                    if new_prefix in tried_prefixes[episode_idx]:
                        continue
                    tried_prefixes[episode_idx].add(new_prefix)
                    batch_requests.append(
                        RolloutRequest(
                            map_path=base_request.map_path,
                            episode_seed=base_request.episode_seed,
                            prefix=new_prefix,
                        )
                    )
                    batch_owner.append(episode_idx)
                    branches_from_current += 1
                    planned_for_episode += 1

        if not batch_requests:
            break

        candidates = rollout_with_prefixes(
            model,
            checkpoint_payload,
            phase,
            batch_requests,
            device=device,
            branch_top_k=branch_top_k,
            inference_batch_size=rollout_batch_size,
            rollout_backend=rollout_backend,
            capture_traces=False,
        )

        for episode_idx, candidate in zip(batch_owner, candidates):
            extra_rollouts[episode_idx] += 1
            rollout_sig = tuple(candidate.actions)
            if rollout_sig in seen_rollouts[episode_idx]:
                continue
            seen_rollouts[episode_idx].add(rollout_sig)
            frontier_by_episode[episode_idx].append(candidate)
            if better_rollout(candidate, best_by_episode[episode_idx]):
                best_by_episode[episode_idx] = candidate

    materialize_requests: List[RolloutRequest] = []
    materialize_owner: List[int] = []
    for episode_idx, best in enumerate(best_by_episode):
        if best is None:
            continue
        materialize_requests.append(
            RolloutRequest(
                map_path=base_requests[episode_idx].map_path,
                episode_seed=base_requests[episode_idx].episode_seed,
                prefix=best.prefix,
            )
        )
        materialize_owner.append(episode_idx)

    if materialize_requests:
        materialized = rollout_with_prefixes(
            model,
            checkpoint_payload,
            phase,
            materialize_requests,
            device=device,
            branch_top_k=branch_top_k,
            inference_batch_size=rollout_batch_size,
            rollout_backend=rollout_backend,
            capture_traces=True,
        )
        for episode_idx, result in zip(materialize_owner, materialized):
            best_by_episode[episode_idx] = result

    return [
        (
            base_requests[episode_idx],
            baseline_results[episode_idx],
            best_by_episode[episode_idx],
            extra_rollouts[episode_idx],
            len(seen_rollouts[episode_idx]),
        )
        for episode_idx in range(len(base_requests))
    ]


def search_episodes(model: MaskedBCPolicy, checkpoint_payload: Dict[str, object],
                    phase: int, map_path: str, episode_seeds: Sequence[int],
                    device: torch.device, branch_budget: int,
                    branches_per_rollout: int,
                    branch_top_k: int,
                    rollout_batch_size: int = 64,
                    rollout_backend: str = "auto",
                    frontier_batch_size: int = 1
                    ) -> List[Tuple[RolloutResult, Optional[RolloutResult], int, int]]:
    seeds = [int(seed) for seed in episode_seeds]
    request_rows = [
        RolloutRequest(
            map_path=map_path,
            episode_seed=episode_seed,
            prefix=(),
        )
        for episode_seed in seeds
    ]
    rows = search_requests(
        model,
        checkpoint_payload,
        phase,
        request_rows,
        device=device,
        branch_budget=branch_budget,
        branches_per_rollout=branches_per_rollout,
        branch_top_k=branch_top_k,
        rollout_batch_size=rollout_batch_size,
        rollout_backend=rollout_backend,
        frontier_batch_size=frontier_batch_size,
    )
    return [
        (baseline, best, extra_rollouts, unique_paths)
        for _, baseline, best, extra_rollouts, unique_paths in rows
    ]


def search_episode(model: MaskedBCPolicy, checkpoint_payload: Dict[str, object],
                   phase: int, map_path: str, episode_seed: int,
                   device: torch.device, branch_budget: int,
                   branches_per_rollout: int,
                   branch_top_k: int,
                   rollout_batch_size: int = 64,
                   rollout_backend: str = "auto",
                   frontier_batch_size: int = 1) -> Tuple[RolloutResult, Optional[RolloutResult], int, int]:
    return search_episodes(
        model,
        checkpoint_payload,
        phase,
        map_path,
        [episode_seed],
        device=device,
        branch_budget=branch_budget,
        branches_per_rollout=branches_per_rollout,
        branch_top_k=branch_top_k,
        rollout_batch_size=rollout_batch_size,
        rollout_backend=rollout_backend,
        frontier_batch_size=frontier_batch_size,
    )[0]


def save_improved_dataset(path: str, improved: List[Tuple[str, int, RolloutResult]]) -> None:
    obs_rows: List[np.ndarray] = []
    mask_rows: List[np.ndarray] = []
    action_rows: List[int] = []
    map_rows: List[str] = []
    seed_rows: List[int] = []
    step_rows: List[int] = []
    episode_rows: List[int] = []

    for episode_idx, (map_name, seed, rollout) in enumerate(improved):
        for step_idx, action in enumerate(rollout.actions):
            obs_rows.append(rollout.obs[step_idx])
            mask_rows.append(rollout.masks[step_idx])
            action_rows.append(action)
            map_rows.append(map_name)
            seed_rows.append(seed)
            step_rows.append(step_idx)
            episode_rows.append(episode_idx)

    if not obs_rows:
        return

    np.savez_compressed(
        path,
        obs=np.asarray(obs_rows, dtype=np.float32),
        masks=np.asarray(mask_rows, dtype=np.bool_),
        actions=np.asarray(action_rows, dtype=np.int64),
        map_names=np.asarray(map_rows, dtype=object),
        seeds=np.asarray(seed_rows, dtype=np.int64),
        step_indices=np.asarray(step_rows, dtype=np.int64),
        episode_indices=np.asarray(episode_rows, dtype=np.int64),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--max-maps", type=int, default=0)
    parser.add_argument("--map-filter", default="",
                        help="only keep maps whose basename contains this substring")
    parser.add_argument("--seeds-per-map", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rollout-backend", default="auto",
                        choices=["auto", "cpu", "gpu"])
    parser.add_argument("--branch-budget", type=int, default=16,
                        help="max extra rollouts per map/seed")
    parser.add_argument("--branches-per-rollout", type=int, default=4,
                        help="how many low-confidence steps to branch from each rollout")
    parser.add_argument("--branch-top-k", type=int, default=2)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--frontier-batch-size", type=int, default=1,
                        help="expand this many frontier nodes per search wave")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-npz", default="")
    args = parser.parse_args()

    if args.phase not in CURRICULUM:
        raise SystemExit(f"unknown phase: {args.phase}")
    if args.branch_top_k < 2:
        raise SystemExit("--branch-top-k must be at least 2")

    device = resolve_device(args.device)

    payload = torch.load(args.checkpoint, map_location="cpu")
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

    phase_cfg = CURRICULUM[args.phase]
    map_pool = get_map_pool(phase_cfg)
    if args.map_filter:
        map_pool = [
            map_path for map_path in map_pool
            if args.map_filter in os.path.basename(map_path)
        ]
    if args.max_maps > 0:
        map_pool = map_pool[:args.max_maps]
    seed_manifest = load_seed_manifest()

    summaries = []
    improved_samples: List[Tuple[str, int, RolloutResult]] = []
    total_episodes = 0
    improved_episodes = 0
    total_branches = 0
    search_requests_rows: List[RolloutRequest] = []
    search_meta: List[Tuple[str, int]] = []

    for map_path in map_pool:
        map_name = os.path.basename(map_path)
        seeds_to_try = default_seeds_for_map(
            map_name,
            seed_manifest,
            args.seeds_per_map,
            args.base_seed,
        )
        for episode_seed in seeds_to_try:
            search_requests_rows.append(
                RolloutRequest(
                    map_path=map_path,
                    episode_seed=episode_seed,
                    prefix=(),
                )
            )
            search_meta.append((map_name, episode_seed))

    search_results = search_requests(
        model,
        payload,
        args.phase,
        search_requests_rows,
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
        search_results,
    ):

            total_episodes += 1
            total_branches += extra_rollouts
            improved = best is not None and (
                not baseline.won or best.low_steps < baseline.low_steps
            )

            summary = {
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
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False))

            if improved and best is not None:
                improved_episodes += 1
                improved_samples.append((map_name, episode_seed, best))

    aggregate = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "phase": args.phase,
        "episodes": total_episodes,
        "improved_episodes": improved_episodes,
        "improved_ratio": improved_episodes / max(total_episodes, 1),
        "avg_extra_rollouts": total_branches / max(total_episodes, 1),
        "summaries": summaries,
    }

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(aggregate, fh, indent=2, ensure_ascii=False)

    if args.output_npz:
        out_dir = os.path.dirname(args.output_npz)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_improved_dataset(args.output_npz, improved_samples)

    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
