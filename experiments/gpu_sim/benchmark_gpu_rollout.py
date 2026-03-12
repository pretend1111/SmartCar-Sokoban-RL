from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gpu_sim.gpu_rollout import rollout_with_prefixes_gpu
from experiments.solver_bc.branch_search import RolloutRequest, rollout_with_prefixes, resolve_device
from experiments.solver_bc.train_bc import MaskedBCPolicy


def compare_summaries(cpu_results, gpu_results):
    report = {
        "same_length": len(cpu_results) == len(gpu_results),
        "matched_outcomes": True,
        "matched_low_steps": True,
        "matched_high_steps": True,
        "matched_actions": True,
    }
    if len(cpu_results) != len(gpu_results):
        return report
    for cpu, gpu in zip(cpu_results, gpu_results):
        if cpu.won != gpu.won or cpu.truncated_reason != gpu.truncated_reason:
            report["matched_outcomes"] = False
        if cpu.low_steps != gpu.low_steps:
            report["matched_low_steps"] = False
        if cpu.high_steps != gpu.high_steps:
            report["matched_high_steps"] = False
        if list(cpu.actions) != list(gpu.actions):
            report["matched_actions"] = False
    report["matched"] = all(report.values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--num-rollouts", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--branch-top-k", type=int, default=2)
    parser.add_argument("--cpu-batch-size", type=int, default=64)
    args = parser.parse_args()

    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = MaskedBCPolicy(payload["obs_dim"], payload["n_actions"], payload["hidden_dim"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    requests: List[RolloutRequest] = []
    for idx in range(args.num_rollouts):
        requests.append(
            RolloutRequest(
                map_path=args.map,
                episode_seed=7 + idx * 9973,
                prefix=(),
            )
        )

    t0 = time.perf_counter()
    cpu_results = rollout_with_prefixes(
        model,
        payload,
        args.phase,
        requests,
        device=device,
        branch_top_k=args.branch_top_k,
        inference_batch_size=args.cpu_batch_size,
    )
    cpu_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu_results = rollout_with_prefixes_gpu(
        model,
        payload,
        args.phase,
        requests,
        device=device,
        branch_top_k=args.branch_top_k,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gpu_sec = time.perf_counter() - t0

    compare_report = compare_summaries(cpu_results, gpu_results)
    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "phase": args.phase,
        "map": args.map,
        "num_rollouts": args.num_rollouts,
        "device": str(device),
        "cpu_rollout_sec": cpu_sec,
        "gpu_rollout_sec": gpu_sec,
        "speedup": cpu_sec / max(gpu_sec, 1e-9),
        **compare_report,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
