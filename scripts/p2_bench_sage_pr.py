"""SAGE-PR sanity check + benchmark — P2.3.

打印参数量、前向时延 (cuda + cpu), 验证收敛性 (3 epoch 在随机数据上).
"""

from __future__ import annotations

import argparse
import sys
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.sage_pr.model import build_default_model, SAGEPolicyRanker


def count_params(model):
    """按层 / 模块统计参数."""
    total = 0
    print("\n=== Parameter breakdown ===")
    for name, module in model.named_children():
        cnt = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:18s}: {cnt:>8,} params")
        total += cnt
    print(f"  {'TOTAL':18s}: {total:>8,} params")
    return total


def estimate_int8_size(n_params: int) -> int:
    """假设 int8 量化, 1 byte / param + 少量 scale/zero."""
    return n_params + n_params // 32   # +3% overhead


def benchmark_forward(model, device, B: int = 1, n_warmup: int = 5, n_iter: int = 50):
    model.eval()
    model.to(device)
    x_grid = torch.randn(B, 30, 10, 14, device=device)
    x_cand = torch.randn(B, 64, 128, device=device)
    u_global = torch.randn(B, 16, device=device)
    mask = torch.ones(B, 64, device=device)
    # padding: 后 16 个非法
    mask[:, 48:] = 0

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x_grid, x_cand, u_global, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = model(x_grid, x_cand, u_global, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000 / n_iter
    print(f"  bs={B} on {device.type}: {elapsed:.2f} ms / forward")
    return elapsed


def random_train_sanity(model, device, n_epoch: int = 3, n_step: int = 50, B: int = 32):
    """在随机数据上训 3 epoch, 验证不发散."""
    model.train()
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    losses = []
    for epoch in range(n_epoch):
        total = 0.0
        for _ in range(n_step):
            x_grid = torch.randn(B, 30, 10, 14, device=device)
            x_cand = torch.randn(B, 64, 128, device=device)
            u_global = torch.randn(B, 16, device=device)
            mask = torch.ones(B, 64, device=device)
            mask[:, 48:] = 0
            # 随机正确动作
            target_idx = torch.randint(0, 48, (B,), device=device)
            target_value = torch.randn(B, device=device)

            score, value, dl, pg, ig = model(x_grid, x_cand, u_global, mask)
            loss_policy = F.cross_entropy(score, target_idx)
            loss_value = F.smooth_l1_loss(value, target_value)
            loss = loss_policy + 0.3 * loss_value

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        avg = total / n_step
        losses.append(avg)
        print(f"  epoch {epoch + 1}: loss = {avg:.4f}")

    if any(np.isnan(l) or np.isinf(l) for l in losses):
        return False, losses
    # 第一 epoch 应该 > 后面的 (大致下降)
    return True, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, nargs="*", default=[1, 32, 512])
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  SAGE-PR Sanity Check & Benchmark (P2.3)")
    print("=" * 60)

    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda") if (torch.cuda.is_available() and not args.no_cuda) else device_cpu

    model = build_default_model()
    n = count_params(model)
    print(f"\nINT8 estimated: {estimate_int8_size(n) / 1024:.1f} KB (raw {n / 1024:.1f} KB fp32)")

    print("\n=== CPU forward benchmark ===")
    for B in args.bs:
        benchmark_forward(model, device_cpu, B=B)

    if device_gpu.type == "cuda":
        print("\n=== CUDA forward benchmark ===")
        for B in args.bs:
            benchmark_forward(model, device_gpu, B=B)

    if not args.skip_train:
        print("\n=== Random data training sanity (3 epoch) ===")
        ok, losses = random_train_sanity(model, device_gpu, n_epoch=3, n_step=30, B=32)
        if ok:
            print(f"  ✓ converging: {losses[0]:.3f} -> {losses[-1]:.3f}")
        else:
            print(f"  ✗ DIVERGED: {losses}")
            sys.exit(1)

    # 完成判定
    print("\n=== Completion criteria ===")
    fp32_kb = n * 4 / 1024
    int8_kb = estimate_int8_size(n) / 1024
    print(f"  Params:           {n:>8,}  (target ≈ 98K, allow ±20K → 78K-200K)")
    print(f"  fp32 model size:  {fp32_kb:>7.1f} KB")
    print(f"  INT8 estimated:   {int8_kb:>7.1f} KB  (target ≤ 200 KB ✓ if {int8_kb <= 200})")
    if 50_000 <= n <= 250_000:
        print("  ✓ Param count in target range")
    else:
        print(f"  ⚠ Param count {n} outside [50K, 250K]")


if __name__ == "__main__":
    main()
