"""V1 vs V2 深度对比.

V1 = pure exact + plan_exploration (push samples only, fully_observed=True)
V2 = god-mode A + suppression + insert inspect (push + inspect, partial-obs)

对比维度:
  1. 总样本数 / 每图样本数
  2. push vs inspect 分布
  3. mask 合法 candidate 数分布 (V2 应该更稀疏 — 抑制场)
  4. 每个 sample 的 unidentified entity 计数 (V1=0 always, V2 各种)
  5. label 类型分布 (V2 含 inspect)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def analyze(path: str, name: str):
    if not os.path.exists(path):
        print(f"{name}: file missing")
        return None
    d = np.load(path)
    n = len(d["label"])
    label = d["label"]
    mask = d["mask"]
    X_grid = d["X_grid"]
    X_cand = d["X_cand"]
    u_global = d["u_global"]
    source = d["source"]

    n_legal_per_sample = mask.sum(axis=1)

    # X_cand[..., :3] 是 type one-hot (push_box, push_bomb, inspect, pad?) — 看 cand_features.py
    # 简化: X_cand 的某些通道指示 type. 但我们没法直接知道, 用 label 处的 cand 推断
    # 改用: 检查 label 处的 type 通道. 先看 X_cand 维度分布
    chosen_cand = X_cand[np.arange(n), label]   # [n, 128]

    # 大概率前几个通道是 type one-hot. 看 cand_features.py:
    # type=push_box, push_bomb, inspect, pad — 假设 4 个 one-hot
    # Type one-hot is 8-dim at index 0:8 (pad, push_box, push_box_macro2, push_box_macro3,
    # push_bomb, push_bomb_diag, inspect, return_garage)
    type_oh = chosen_cand[:, :8]
    type_argmax = type_oh.argmax(axis=1)
    type_names = ["pad", "push_box", "push_box_m2", "push_box_m3",
                   "push_bomb", "push_bomb_diag", "inspect", "return_garage"]
    type_dist = {n: int((type_argmax == i).sum()) for i, n in enumerate(type_names)}

    # u_global[4] = unidentified boxes / 5; u_global[5] = unidentified targets / 5
    n_unid = (u_global[:, 4] * 5 + u_global[:, 5] * 5).round().astype(int)
    unid_dist = {}
    for i in range(0, 11):
        cnt = int((n_unid == i).sum())
        if cnt > 0:
            unid_dist[i] = cnt

    print(f"\n=== {name} ({n} samples) ===")
    print(f"  source codes: {dict(zip(*np.unique(source, return_counts=True)))}")
    print(f"  legal per sample: mean={n_legal_per_sample.mean():.2f} "
          f"min={int(n_legal_per_sample.min())} max={int(n_legal_per_sample.max())}")
    print(f"  label type distribution:")
    for k, v in type_dist.items():
        print(f"    {k:>10}: {v:>6} ({100*v/n:.1f}%)")
    print(f"  unidentified entities at sample state:")
    for k in sorted(unid_dist):
        print(f"    n_unid={k}: {unid_dist[k]:>6} ({100*unid_dist[k]/n:.1f}%)")
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-dir", default="runs/sage_pr/full_v5")
    parser.add_argument("--v2-dir", default="runs/sage_pr/full_v6")
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    args = parser.parse_args()

    v1_dir = os.path.join(ROOT, args.v1_dir)
    v2_dir = os.path.join(ROOT, args.v2_dir)

    summary = []
    for p in args.phases:
        print(f"\n{'='*70}")
        print(f"Phase {p}")
        print('='*70)
        v1 = analyze(os.path.join(v1_dir, f"phase{p}_exact.npz"), f"V1-P{p}")
        v2 = analyze(os.path.join(v2_dir, f"phase{p}_v2.npz"), f"V2-P{p}")
        if v1 is not None and v2 is not None:
            summary.append({
                "phase": p,
                "v1_n": len(v1["label"]),
                "v2_n": len(v2["label"]),
            })

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"{'phase':>6} {'V1 samples':>14} {'V2 samples':>14} {'ratio V2/V1':>14}")
    for s in summary:
        ratio = s["v2_n"] / max(1, s["v1_n"])
        print(f"{s['phase']:>6} {s['v1_n']:>14} {s['v2_n']:>14} {ratio:>14.3f}")


if __name__ == "__main__":
    main()
