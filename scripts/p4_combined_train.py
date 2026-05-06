"""P4 合并训练 — 多 phase npz 拼成大数据集训一个共享模型.

理论:
  - 共享 conv 能学到通用空间特征 (墙体/通道/死角)
  - 不同 phase 数据互补: phase 1-3 的简单状态帮助稳定特征提取,
    phase 4-6 的难状态训练高层决策
  - 单一模型避免 phase 切换的检查点管理

用法:
  python scripts/p4_combined_train.py \
    --datasets .agent/runs/p1_conv_h512/phase1_v2.npz \
               .agent/runs/p2_conv_h512/phase2_v2.npz \
               .agent/runs/p3_conv_h512/phase3_v2.npz \
    --output-dir .agent/runs/combined_p123 \
    --epochs 80 --batch-size 1024 --hidden-dim 512
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True,
                   help="多个 build_dataset_v2 输出的 .npz 文件")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--policy", choices=["mlp", "conv"], default="conv")
    p.add_argument("--combined-output", default="",
                   help="合并后的 npz 路径; 留空 = 不持久化, 仅训练")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    obs_list, mask_list, action_list = [], [], []
    map_list, seeds_list, step_list = [], [], []
    obs_mode_first = None
    samples_per_dataset = []

    for ds_path in args.datasets:
        if not os.path.exists(ds_path):
            raise SystemExit(f"missing dataset: {ds_path}")
        d = np.load(ds_path, allow_pickle=True)
        obs = d["obs"]
        masks = d["masks"]
        actions = d["actions"]
        if obs_mode_first is None:
            obs_mode_first = str(d["obs_mode"]) if "obs_mode" in d else "oracle"
        else:
            cur = str(d["obs_mode"]) if "obs_mode" in d else "oracle"
            if cur != obs_mode_first:
                raise SystemExit(f"mismatched obs_mode: {cur} vs {obs_mode_first}")
        n = obs.shape[0]
        samples_per_dataset.append((ds_path, n, obs.shape[1]))
        obs_list.append(obs)
        mask_list.append(masks)
        action_list.append(actions)
        if "map_names" in d:
            map_list.append(d["map_names"])
        if "seeds" in d:
            seeds_list.append(d["seeds"])
        if "step_indices" in d:
            step_list.append(d["step_indices"])

    obs_concat = np.concatenate(obs_list, axis=0)
    mask_concat = np.concatenate(mask_list, axis=0)
    action_concat = np.concatenate(action_list, axis=0)
    print(f"[combined] sources:")
    for path, n, dim in samples_per_dataset:
        print(f"  {path}: {n} samples, dim={dim}")
    print(f"[combined] total: {obs_concat.shape[0]} samples, dim={obs_concat.shape[1]}",
          flush=True)

    if args.combined_output:
        os.makedirs(os.path.dirname(args.combined_output) or ".", exist_ok=True)
        save_kwargs = {
            "obs": obs_concat,
            "masks": mask_concat,
            "actions": action_concat,
            "obs_mode": np.asarray(obs_mode_first),
        }
        if map_list:
            save_kwargs["map_names"] = np.concatenate(map_list, axis=0)
        if seeds_list:
            save_kwargs["seeds"] = np.concatenate(seeds_list, axis=0)
        if step_list:
            save_kwargs["step_indices"] = np.concatenate(step_list, axis=0)
        np.savez_compressed(args.combined_output, **save_kwargs)
        print(f"[combined] saved → {args.combined_output}", flush=True)
        dataset_for_train = args.combined_output
    else:
        # 写到 output_dir 下临时 npz
        tmp = os.path.join(args.output_dir, "_combined_temp.npz")
        np.savez_compressed(
            tmp,
            obs=obs_concat,
            masks=mask_concat,
            actions=action_concat,
            obs_mode=np.asarray(obs_mode_first),
        )
        dataset_for_train = tmp

    # 直接调用 train_bc.main 风格的入口太复杂, 改为 subprocess
    import subprocess
    cmd = [
        sys.executable, "-m", "experiments.solver_bc.train_bc",
        "--dataset", str(dataset_for_train),
        "--output-dir", str(args.output_dir),
        "--policy", args.policy,
        "--device", "auto",
        "--gpu-resident", "on",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--hidden-dim", str(args.hidden_dim),
        "--lr", str(args.lr),
        "--log-every", "10",
    ]
    print("[combined] >>> train", " ".join(cmd), flush=True)
    ret = subprocess.run(cmd, cwd=str(ROOT))
    if ret.returncode != 0:
        sys.exit(ret.returncode)
    print("[combined] training done", flush=True)


if __name__ == "__main__":
    main()
