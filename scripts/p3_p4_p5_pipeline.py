"""P3 → P4 → P5 一键流水线.

输入: assets/maps/phase{N}_verified.json (P1.2 输出)
流程:
  1) build_dataset_v2 → .agent/data/p{N}_v2.npz   (extra_seeds 默认 2)
  2) train_bc --policy conv --hidden-dim 512      (epochs 默认 80)
  3) evaluate_bc 在所有 verified 图上 (3 seed)
  4) 输出 summary 到 .agent/runs/<tag>/pipeline_summary.json

用法:
  python scripts/p3_p4_p5_pipeline.py --phase 6 --tag p6_conv_h512 \
    [--extra-seeds 2] [--epochs 80] [--hidden-dim 512] [--time-limit 60]

若中间步失败立即返回非零, 不掩盖错误.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
RL_PY = "D:/anaconda3/envs/rl/python.exe"  # 显式指定 rl env (兼容 ralph 后台)


def run(cmd, log_path, label):
    print(f"\n[pipeline] >>> {label}", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        ret = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"[pipeline] <<< {label} done in {elapsed:.1f}s (rc={ret.returncode})",
          flush=True)
    return ret.returncode, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, required=True)
    p.add_argument("--tag", default="conv_h512")
    p.add_argument("--verified-path", default="",
                   help="default: assets/maps/phase{N}_verified.json")
    p.add_argument("--extra-seeds", type=int, default=2)
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--max-cost", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--num-workers", type=int, default=18)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--policy", choices=["mlp", "conv"], default="conv")
    p.add_argument("--eval-seeds", type=int, default=3)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    verified = (args.verified_path
                or str(ROOT / "assets" / "maps" / f"phase{args.phase}_verified.json"))
    if not os.path.exists(verified):
        raise SystemExit(f"verified not found: {verified}")

    work_dir = ROOT / ".agent" / "runs" / args.tag
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = work_dir / f"phase{args.phase}_v2.npz"
    train_dir = work_dir / f"phase{args.phase}_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_log = work_dir / f"phase{args.phase}_eval.log"
    summary_path = work_dir / "pipeline_summary.json"

    summary = {
        "phase": args.phase, "tag": args.tag,
        "verified": verified,
        "args": vars(args),
        "steps": [],
    }

    # 1) build dataset
    if not args.skip_build:
        cmd = [
            RL_PY, "-m", "experiments.solver_bc.build_dataset_v2",
            "--verified", verified,
            "--output", str(dataset_path),
            "--teacher", "solver_ida",
            "--time-limit", str(args.time_limit),
            "--max-cost", str(args.max_cost),
            "--max-steps", str(args.max_steps),
            "--extra-seeds", str(args.extra_seeds),
            "--num-workers", str(args.num_workers),
        ]
        rc, dt = run(cmd, work_dir / f"phase{args.phase}_build.log",
                     f"build_dataset_v2 phase {args.phase}")
        summary["steps"].append({"name": "build", "rc": rc, "elapsed_s": dt})
        if rc != 0:
            json.dump(summary, open(summary_path, "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False)
            sys.exit(rc)

    # 2) train
    if not args.skip_train:
        cmd = [
            RL_PY, "-m", "experiments.solver_bc.train_bc",
            "--dataset", str(dataset_path),
            "--output-dir", str(train_dir),
            "--policy", args.policy,
            "--device", "auto",
            "--gpu-resident", "on",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--hidden-dim", str(args.hidden_dim),
            "--lr", str(args.lr),
            "--log-every", "10",
        ]
        rc, dt = run(cmd, work_dir / f"phase{args.phase}_train.log",
                     f"train_bc {args.policy} h={args.hidden_dim}")
        summary["steps"].append({"name": "train", "rc": rc, "elapsed_s": dt})
        if rc != 0:
            json.dump(summary, open(summary_path, "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False)
            sys.exit(rc)

    # 3) eval — 用 max-maps 0 跑完所有 phase{N}/*.txt (而不仅是 verified)
    if not args.skip_eval:
        ckpt = train_dir / "best.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        cmd = [
            RL_PY, "-m", "experiments.solver_bc.evaluate_bc",
            "--checkpoint", str(ckpt),
            "--phase", str(args.phase),
            "--device", "auto",
            "--seeds-per-map", str(args.eval_seeds),
            "--rollout-batch-size", "128",
            "--rollout-backend", "cpu",
        ]
        rc, dt = run(cmd, eval_log,
                     f"evaluate_bc phase {args.phase}")
        summary["steps"].append({"name": "eval", "rc": rc, "elapsed_s": dt})
        # 提取 win_rate
        if eval_log.exists():
            try:
                import re
                t = eval_log.read_text(encoding="utf-8")
                m = re.search(r'"win_rate":\s*([0-9.]+),\s*"maps"', t[-5000:])
                if m:
                    summary["overall_win_rate"] = float(m.group(1))
            except Exception as e:
                summary["eval_parse_error"] = str(e)

    json.dump(summary, open(summary_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\n[pipeline] summary → {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
