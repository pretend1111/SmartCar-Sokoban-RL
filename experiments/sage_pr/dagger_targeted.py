"""Targeted DAgger — runs DAgger on a specific list of map paths.

Use case: identify maps where the model fails, then sample MANY seeds + actions
on those specific maps to generate teacher labels.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.sage_pr.dagger_v1 import (
    collect_v1_dagger_episode, _MODEL_GLOBAL,
)
from experiments.sage_pr.build_dataset_v3 import save_dataset, Sample
from experiments.sage_pr.model import build_model_from_ckpt


def _worker_init(ckpt_path: str):
    device = torch.device("cpu")
    model = build_model_from_ckpt(ckpt_path, device=device)
    model.eval()
    _MODEL_GLOBAL["model"] = model
    _MODEL_GLOBAL["device"] = device


def _worker_episode(args):
    map_path, seed, phase, step_limit, top_k, teacher_tl = args
    model = _MODEL_GLOBAL["model"]
    device = _MODEL_GLOBAL["device"]
    samples, info = collect_v1_dagger_episode(
        model, device, map_path, seed, phase,
        step_limit=step_limit, top_k=top_k,
        teacher_time_limit=teacher_tl,
    )
    return samples, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--maps-file", required=True,
                        help="JSON file with list of {map_path, phase} or text file with one map per line")
    parser.add_argument("--seeds-per-map", type=int, default=20,
                        help="how many seeds to try per map")
    parser.add_argument("--step-limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--teacher-time-limit", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Load map list
    maps_list = []
    if args.maps_file.endswith(".json"):
        with open(args.maps_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    maps_list.append(item)
                elif isinstance(item, dict):
                    maps_list.append(item["map_path"])
                elif isinstance(item, (list, tuple)):
                    maps_list.append(item[0])
    else:
        with open(args.maps_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    maps_list.append(line)

    print(f"loaded {len(maps_list)} maps")

    # Determine phase from path
    def get_phase(path: str) -> int:
        for p in range(1, 7):
            if f"phase{p}/" in path or f"phase{p}\\" in path:
                return p
        return -1

    all_tasks = []
    for mp_str in maps_list:
        phase = get_phase(mp_str)
        for seed in range(args.seeds_per_map):
            all_tasks.append((mp_str, seed, phase, args.step_limit,
                              args.top_k, args.teacher_time_limit))

    print(f"total tasks: {len(all_tasks)}, workers: {args.workers}")
    all_samples: List[Sample] = []
    n_won = 0
    n_total = 0
    n_disagree = 0

    t0 = time.perf_counter()
    if args.workers <= 1:
        _worker_init(args.ckpt)
        for tsk in all_tasks:
            samples, info = _worker_episode(tsk)
            n_total += 1
            if info["won"]:
                n_won += 1
            n_disagree += info["n_disagree"]
            all_samples.extend(samples)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_worker_init,
                      initargs=(args.ckpt,)) as pool:
            for i, (samples, info) in enumerate(
                    pool.imap_unordered(_worker_episode, all_tasks, chunksize=2)):
                n_total += 1
                if info["won"]:
                    n_won += 1
                n_disagree += info["n_disagree"]
                all_samples.extend(samples)
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(all_tasks)} ({time.perf_counter()-t0:.0f}s) "
                          f"won={n_won}/{n_total} samples={len(all_samples)}")

    print(f"\nDone in {time.perf_counter()-t0:.0f}s")
    print(f"won={n_won}/{n_total} samples={len(all_samples)} disagree={n_disagree}")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    save_dataset(all_samples, out_path)


if __name__ == "__main__":
    main()
