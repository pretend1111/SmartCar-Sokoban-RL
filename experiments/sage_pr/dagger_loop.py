"""DAgger 多轮自动迭代脚本.

每轮:
    1. 用上一轮 best ckpt rollout, 收集 dagger samples.
    2. fine-tune 当前 ckpt 在 base + 累积 DAgger samples 上.
    3. eval, 记录最佳, 进入下一轮.

输出:
    .agent/sage_pr/runs/<tag>_round{N}/best.pt
    .agent/sage_pr/dagger_round{N}.npz
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(cmd: str, log_tag: str = "") -> int:
    print(f"\n[{log_tag}] {cmd[:200]}")
    return subprocess.run(cmd, shell=True, cwd=ROOT).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-ckpt", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--maps-per-round", type=int, default=100)
    parser.add_argument("--epochs-per-round", type=int, default=15)
    parser.add_argument("--tag-prefix", default="dagger_loop")
    parser.add_argument("--base-data", nargs="+",
                        default=[".agent/sage_pr/phase1.npz",
                                 ".agent/sage_pr/phase2.npz",
                                 ".agent/sage_pr/phase3.npz",
                                 ".agent/sage_pr/phase4_v3.npz",
                                 ".agent/sage_pr/phase5_v3.npz",
                                 ".agent/sage_pr/phase6_v3.npz"])
    args = parser.parse_args()

    cur_ckpt = args.start_ckpt
    accumulated_data = list(args.base_data)
    py = "D:/anaconda3/envs/rl/python.exe"

    for r in range(1, args.rounds + 1):
        tag = f"{args.tag_prefix}_r{r}"
        dagger_npz = f".agent/sage_pr/dagger_{tag}.npz"

        # 1. 收集 DAgger
        print(f"\n=== Round {r}: collect DAgger ===")
        cmd_dagger = (
            f'{py} experiments/sage_pr/dagger_lite.py '
            f'--ckpt {cur_ckpt} --phases 3 4 5 6 '
            f'--use-verified-seeds --max-maps {args.maps_per_round} '
            f'--seeds 0 --step-limit 25 --solver-time-limit 1.5 '
            f'--out {dagger_npz}'
        )
        rc = run(cmd_dagger, f"r{r}_dagger")
        if rc != 0:
            print(f"DAgger failed at round {r}")
            sys.exit(1)

        accumulated_data.append(dagger_npz)

        # 2. Fine-tune
        print(f"\n=== Round {r}: fine-tune ===")
        train_tag = f"{args.tag_prefix}_r{r}_train"
        data_args = " ".join(accumulated_data)
        cmd_train = (
            f'{py} experiments/sage_pr/train_sage_pr.py '
            f'--data {data_args} --tag {train_tag} '
            f'--batch-size 256 --lr 1e-4 --epochs {args.epochs_per_round}'
        )
        rc = run(cmd_train, f"r{r}_train")
        if rc != 0:
            print(f"Train failed at round {r}")
            sys.exit(1)

        cur_ckpt = f".agent/sage_pr/runs/{train_tag}/best.pt"

        # 3. Quick eval
        print(f"\n=== Round {r}: eval ===")
        eval_out = f".agent/sage_pr/runs/{train_tag}/eval.json"
        cmd_eval = (
            f'{py} experiments/sage_pr/evaluate_sage_pr.py '
            f'--ckpt {cur_ckpt} --phases 1 2 3 4 5 6 --max-maps 100 '
            f'--seeds 0 --top-k 4 --out {eval_out}'
        )
        run(cmd_eval, f"r{r}_eval")

    print(f"\nDONE. Final ckpt: {cur_ckpt}")


if __name__ == "__main__":
    main()
