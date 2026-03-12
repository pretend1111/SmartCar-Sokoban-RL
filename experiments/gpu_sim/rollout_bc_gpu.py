from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gpu_sim.gpu_push_env import GpuPushBatchEnv, resolve_device
from experiments.solver_bc.train_bc import MaskedBCPolicy


def _checkpoint_obs_mode(payload: Dict[str, object]) -> str:
    mode = str(payload.get("obs_mode", "oracle")).strip().lower()
    if mode not in {"oracle", "state"}:
        raise ValueError(f"unknown checkpoint obs_mode: {mode}")
    return mode


def _build_model_obs(env: GpuPushBatchEnv, payload: Dict[str, object]) -> torch.Tensor:
    include_map_layout = bool(payload["include_map_layout"])
    if _checkpoint_obs_mode(payload) == "oracle":
        return env.build_oracle_obs()
    return env.build_state_obs(include_map_layout=include_map_layout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 9980, 19953])
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = MaskedBCPolicy(payload["obs_dim"], payload["n_actions"], payload["hidden_dim"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    env = GpuPushBatchEnv.from_map_and_seeds(
        map_path=args.map,
        episode_seeds=[int(seed) for seed in args.seeds],
        max_steps=args.max_steps,
        base_dir=str(ROOT),
        device=device,
    )

    batch = len(args.seeds)
    active = torch.ones(batch, dtype=torch.bool, device=device)
    won = torch.zeros(batch, dtype=torch.bool, device=device)
    no_valid = torch.zeros(batch, dtype=torch.bool, device=device)
    action_trace: List[List[int]] = [[] for _ in range(batch)]

    for _ in range(args.max_steps):
        if not torch.any(active):
            break
        obs = _build_model_obs(env, payload)
        masks = env.action_masks()
        valid_any = masks.any(dim=1)
        no_valid |= active & ~valid_any
        active &= valid_any
        if not torch.any(active):
            break

        with torch.no_grad():
            logits = model(obs)
            masked_logits = logits.masked_fill(~masks, -1e9)
            actions = masked_logits.argmax(dim=1)

        result = env.step(actions)
        won |= result.won
        active &= ~result.won

        actions_cpu = actions.detach().cpu().tolist()
        for idx, action in enumerate(actions_cpu):
            if valid_any[idx].item():
                action_trace[idx].append(int(action))

    summary_rows: List[Dict[str, object]] = []
    for idx, seed in enumerate(args.seeds):
        summary_rows.append(
            {
                "seed": int(seed),
                "won": bool(won[idx].item()),
                "no_valid_action": bool(no_valid[idx].item()),
                "high_steps": int(env.step_count[idx].item()),
                "low_steps": int(env.total_low_steps[idx].item()),
                "actions": action_trace[idx],
            }
        )

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "map": args.map,
        "device": str(device),
        "episodes": summary_rows,
        "wins": int(won.sum().item()),
        "total": len(summary_rows),
        "win_rate": float(won.float().mean().item()) if summary_rows else 0.0,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
