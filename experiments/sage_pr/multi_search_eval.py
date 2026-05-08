"""Multi-config rollout search: 对每张图试多种 search 参数, 取首个 win.

策略:
    每个 (map, seed) 用 N 种不同 (beam, lookahead) 配置跑 rollout search.
    任一 win → 标记成功. 否则失败.

这是 "any-of-N test-time augmentation": 用算力换 win rate.
不同配置在不同失败模式下擅长 (lookahead 浅 vs 深, beam 宽 vs 窄), 互补.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.sage_pr.model import build_default_model
from experiments.sage_pr.build_dataset_v3 import (
    parse_phase456_seeds, list_phase_maps,
)
from experiments.sage_pr.rollout_search_eval import rollout_search_episode


def evaluate_phase_multi(model, device, phase, seeds_per_map,
                          *, configs, step_limit=60, max_maps=None,
                          verified_seeds_map=None):
    """每张图跑多个 search 配置, 任一 win 算成功."""
    maps = list_phase_maps(phase)
    if max_maps is not None:
        maps = maps[:max_maps]
    n_total = 0
    n_won = 0
    n_won_per_config = [0] * len(configs)
    total_inf_ms = 0.0

    for map_path in maps:
        if verified_seeds_map is not None and map_path in verified_seeds_map:
            ms = verified_seeds_map[map_path][:max(1, len(seeds_per_map))]
        else:
            ms = seeds_per_map
        for seed in ms:
            n_total += 1
            won_any = False
            for k, (b, la) in enumerate(configs):
                won, steps, avg_inf = rollout_search_episode(
                    model, device, map_path, seed,
                    step_limit=step_limit,
                    beam_width=b, lookahead=la,
                )
                total_inf_ms += avg_inf * 1000
                if won:
                    n_won_per_config[k] += 1
                    won_any = True
                    break  # 任一 config win 即可
            if won_any:
                n_won += 1
    return {
        "phase": phase,
        "n_total": n_total,
        "n_won": n_won,
        "win_rate": n_won / max(n_total, 1),
        "n_won_per_config": n_won_per_config,
        "config_names": [f"b={b}_l={la}" for b, la in configs],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--max-maps", type=int, default=100)
    parser.add_argument("--use-verified-seeds", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = build_default_model().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded {args.ckpt}")

    verified_map = None
    if args.use_verified_seeds:
        verified_map = parse_phase456_seeds(
            os.path.join(ROOT, "assets/maps/phase456_seed_manifest.json")
        )

    # 4 个互补配置
    configs = [
        (4, 12),
        (5, 20),
        (6, 25),
        (4, 50),
    ]

    results = []
    for ph in args.phases:
        print(f"\n=== Phase {ph} ===")
        t0 = time.perf_counter()
        r = evaluate_phase_multi(
            model, device, ph, seeds, configs=configs,
            max_maps=args.max_maps, verified_seeds_map=verified_map,
        )
        elapsed = time.perf_counter() - t0
        print(f"  any-of-{len(configs)} win_rate = {r['win_rate']*100:.1f}% "
              f"({r['n_won']}/{r['n_total']}); elapsed={elapsed:.0f}s")
        for cfg, won in zip(r["config_names"], r["n_won_per_config"]):
            print(f"    {cfg}: +{won} wins")
        results.append(r)

    print("\n=== Summary ===")
    for r in results:
        print(f"phase {r['phase']}: any={r['win_rate']*100:.1f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
