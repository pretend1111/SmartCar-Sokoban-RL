from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gpu_sim.gpu_push_env import GpuPushBatchEnv, resolve_device
from smartcar_sokoban.rl.high_level_env import SokobanHLEnv


def build_envs(map_file: str, seeds: List[int]) -> List[SokobanHLEnv]:
    envs: List[SokobanHLEnv] = []
    for seed in seeds:
        env = SokobanHLEnv(map_file=map_file, include_map_layout=True)
        env.reset(seed=seed)
        envs.append(env)
    return envs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    seeds = list(range(args.batch_size))
    envs = build_envs(args.map, seeds)
    gpu_env = GpuPushBatchEnv.from_envs(envs, device=resolve_device(args.device))

    for _ in range(5):
        _ = gpu_env.push_action_masks()
    if gpu_env.device.type == "cuda":
        torch.cuda.synchronize(gpu_env.device)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        for env in envs:
            _ = env.action_masks()
    cpu_mask_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(args.iters):
        _ = gpu_env.push_action_masks()
    if gpu_env.device.type == "cuda":
        torch.cuda.synchronize(gpu_env.device)
    gpu_mask_sec = time.perf_counter() - t0

    report = {
        "map": args.map,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "device": str(gpu_env.device),
        "cpu_mask_sec": cpu_mask_sec,
        "gpu_push_mask_sec": gpu_mask_sec,
        "speedup": cpu_mask_sec / max(gpu_mask_sec, 1e-9),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
