from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gpu_sim.gpu_push_env import GpuPushBatchEnv, resolve_device
from smartcar_sokoban.rl.high_level_env import N_ACTIONS, SokobanHLEnv
from smartcar_sokoban.solver.pathfinder import pos_to_grid


def build_envs(map_file: str, seeds: List[int]) -> List[SokobanHLEnv]:
    envs: List[SokobanHLEnv] = []
    for seed in seeds:
        env = SokobanHLEnv(map_file=map_file, include_map_layout=True)
        env.reset(seed=seed)
        envs.append(env)
    return envs


def cpu_action_mask(env: SokobanHLEnv) -> torch.Tensor:
    return torch.from_numpy(env.action_masks()).bool()


def compare_state(env: SokobanHLEnv, gpu_env: GpuPushBatchEnv, row: int) -> Dict[str, object]:
    state = env.engine.get_state()
    cpu_boxes = sorted((int(box.x - 0.5), int(box.y - 0.5), int(box.class_id)) for box in state.boxes)
    cpu_targets = sorted((int(t.x - 0.5), int(t.y - 0.5), int(t.num_id)) for t in state.targets)
    cpu_bombs = sorted((int(b.x - 0.5), int(b.y - 0.5)) for b in state.bombs)
    gpu_state = gpu_env.debug_state(row)
    gpu_boxes = sorted((item["col"], item["row"], item["id"]) for item in gpu_state["boxes"])
    gpu_targets = sorted((item["col"], item["row"], item["id"]) for item in gpu_state["targets"])
    gpu_bombs = sorted((item["col"], item["row"]) for item in gpu_state["bombs"])
    return {
        "car_match": gpu_state["car"] == pos_to_grid(state.car_x, state.car_y),
        "boxes_match": gpu_boxes == cpu_boxes,
        "targets_match": gpu_targets == cpu_targets,
        "bombs_match": gpu_bombs == cpu_bombs,
        "seen_box_match": sorted(gpu_state["seen_box"]) == sorted(state.seen_box_ids),
        "seen_target_match": sorted(gpu_state["seen_target"]) == sorted(state.seen_target_ids),
        "walls_match": torch.equal(
            gpu_env.walls[row].cpu(),
            torch.as_tensor(state.grid, dtype=torch.bool),
        ),
        "won_match": gpu_state["won"] == bool(state.won),
        "low_steps_match": int(gpu_env.total_low_steps[row].item()) == int(env._total_low_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 9980, 19953])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    envs = build_envs(args.map, args.seeds)
    gpu_env = GpuPushBatchEnv.from_envs(envs, device=resolve_device(args.device))

    report: Dict[str, object] = {
        "map": args.map,
        "seeds": args.seeds,
        "device": str(gpu_env.device),
        "steps": [],
    }

    for step_idx in range(args.steps):
        gpu_mask = gpu_env.action_masks().cpu()
        cpu_masks = torch.stack([cpu_action_mask(env) for env in envs], dim=0)
        missing = cpu_masks & ~gpu_mask
        extra = gpu_mask & ~cpu_masks
        missing_count = int(missing.sum().item())
        extra_count = int(extra.sum().item())

        actions = []
        for row, env in enumerate(envs):
            valid = torch.nonzero(cpu_masks[row], as_tuple=False).squeeze(1)
            if valid.numel() == 0:
                break
            actions.append(int(valid[0].item()))
        if len(actions) != len(envs):
            report["steps"].append({
                "step": step_idx,
                "stopped": True,
                "reason": "cpu_env_has_no_action",
                "missing_cpu_actions": missing_count,
                "extra_gpu_actions": extra_count,
            })
            break

        gpu_result = gpu_env.step(torch.tensor(actions, dtype=torch.long, device=gpu_env.device))
        for env, action in zip(envs, actions):
            env.step(action)

        matches = [compare_state(env, gpu_env, row) for row, env in enumerate(envs)]
        all_ok = all(all(item.values()) for item in matches)
        report["steps"].append({
            "step": step_idx,
            "actions": actions,
            "missing_cpu_actions": missing_count,
            "extra_gpu_actions": extra_count,
            "gpu_valid": [bool(v) for v in gpu_result.valid.cpu().tolist()],
            "gpu_low_steps": [int(v) for v in gpu_result.low_steps.cpu().tolist()],
            "matches": matches,
            "all_ok": all_ok,
        })
        if not all_ok:
            break

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
