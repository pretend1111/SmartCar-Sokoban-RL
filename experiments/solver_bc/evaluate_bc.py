from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.solver_bc.branch_search import (
    RolloutRequest,
    default_seeds_for_map,
    resolve_device,
    rollout_with_prefixes,
)
from experiments.solver_bc.train_bc import MaskedBCPolicy
from smartcar_sokoban.rl.train import CURRICULUM, get_map_pool, load_seed_manifest


def _build_model_from_payload(payload: dict):
    policy = str(payload.get("policy", "mlp")).lower()
    if policy == "conv":
        from experiments.solver_bc.policy_conv import MaskedConvBCPolicy
        return MaskedConvBCPolicy(
            n_actions=int(payload["n_actions"]),
            hidden_dim=int(payload["hidden_dim"]),
            wall_emb_dim=int(payload.get("wall_emb_dim", 64)),
        )
    return MaskedBCPolicy(payload["obs_dim"], payload["n_actions"],
                          payload["hidden_dim"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--max-maps", type=int, default=0)
    parser.add_argument("--seeds-per-map", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--rollout-backend", default="auto",
                        choices=["auto", "cpu", "gpu"])
    args = parser.parse_args()

    if args.phase not in CURRICULUM:
        raise SystemExit(f"unknown phase: {args.phase}")

    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = _build_model_from_payload(payload)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    phase_cfg = CURRICULUM[args.phase]
    map_pool = get_map_pool(phase_cfg)
    if args.max_maps > 0:
        map_pool = map_pool[:args.max_maps]
    seed_manifest = load_seed_manifest()

    requests: List[RolloutRequest] = []
    request_meta: List[Tuple[str, int]] = []
    for map_path in map_pool:
        map_name = os.path.basename(map_path)
        seeds_to_try = default_seeds_for_map(
            map_name,
            seed_manifest,
            args.seeds_per_map,
            args.base_seed,
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

    with torch.no_grad():
        rollout_results = rollout_with_prefixes(
            model,
            payload,
            args.phase,
            requests,
            device=device,
            branch_top_k=1,
            inference_batch_size=args.rollout_batch_size,
            rollout_backend=args.rollout_backend,
        )

    per_map: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    total = 0
    wins = 0
    for (map_name, episode_seed), result in zip(request_meta, rollout_results):
        del episode_seed
        total += 1
        if result.won:
            wins += 1
        per_map[map_name].append(
            {
                "won": result.won,
                "low_steps": result.low_steps if result.won else None,
            }
        )

    results = []
    for map_path in map_pool:
        map_name = os.path.basename(map_path)
        rows = per_map.get(map_name, [])
        map_wins = sum(1 for row in rows if row["won"])
        steps_if_win = [int(row["low_steps"]) for row in rows if row["low_steps"] is not None]
        result = {
            "map": map_name,
            "win_rate": map_wins / max(len(rows), 1),
            "wins": map_wins,
            "episodes": len(rows),
            "avg_low_steps_if_win": (
                round(sum(steps_if_win) / len(steps_if_win), 2)
                if steps_if_win else None
            ),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "phase": args.phase,
        "episodes": total,
        "wins": wins,
        "win_rate": wins / max(total, 1),
        "maps": results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
