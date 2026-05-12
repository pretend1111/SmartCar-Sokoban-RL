"""SAGE-PR Stage A — BC 预训练.

输入: 多个 npz (build_dataset_v3 / v5 / v6 输出), 合并成 phase-stratified 大数据集.
损失 (FINAL_ARCH_DESIGN.md §5.2):
    L = L_policy + 0.3·L_value + 0.2·L_deadlock + 0.2·L_progress + 0.1·L_info
    L_info GT = X_cand[:, :, 108]  (cand_features 段 [108:118] 信息增益 = viewpoint
                  的 info_gain_heatmap, 非 inspect 候选自然为 0)
    (L_ranking 需 P3.4 hard negative, 暂未实现)

优化:
    AdamW lr=3e-4 cosine -> 3e-5
    batch=256, weight_decay=1e-4, grad_clip=1.0
    80 epoch (默认), phase-stratified 采样 5/10/15/25/20/25%

输出:
    .agent/sage_pr/runs/<tag>/best.pt
    .agent/sage_pr/runs/<tag>/train.log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.sage_pr.model import (
    build_default_model, build_large_model,
    build_push_only_model, build_push_only_large,
)
from smartcar_sokoban.symbolic.cand_features import slice_push_only_cand
from smartcar_sokoban.symbolic.grid_tensor import (
    slice_push_only_grid, slice_push_only_global,
)


# ── 数据集 ────────────────────────────────────────────────

class SagePrDataset(Dataset):
    """从多个 npz 加载样本, 合并 + phase index.

    Args:
        push_only: True 时把 X_grid 30→27ch, X_cand 128→118, u_global 16→12, 跟
            push_only model 输入对齐 (新架构默认).
    """
    def __init__(self, npz_paths: List[str], push_only: bool = True):
        self.X_grid: List[np.ndarray] = []
        self.X_cand: List[np.ndarray] = []
        self.u_global: List[np.ndarray] = []
        self.mask: List[np.ndarray] = []
        self.label: List[np.ndarray] = []
        self.phase: List[np.ndarray] = []
        self.source: List[np.ndarray] = []

        for path in npz_paths:
            print(f"  load {path}")
            d = np.load(path)
            self.X_grid.append(d["X_grid"])
            self.X_cand.append(d["X_cand"])
            self.u_global.append(d["u_global"])
            self.mask.append(d["mask"])
            self.label.append(d["label"])
            self.phase.append(d["phase"])
            self.source.append(d["source"])

        self.X_grid = np.concatenate(self.X_grid, axis=0)
        self.X_cand = np.concatenate(self.X_cand, axis=0)
        self.u_global = np.concatenate(self.u_global, axis=0)
        self.mask = np.concatenate(self.mask, axis=0)
        self.label = np.concatenate(self.label, axis=0)
        self.phase = np.concatenate(self.phase, axis=0)
        self.source = np.concatenate(self.source, axis=0)

        if push_only:
            print(f"  slice push_only: X_grid {self.X_grid.shape[-1]}→27ch, "
                  f"X_cand {self.X_cand.shape[-1]}→118, u_global {self.u_global.shape[-1]}→12")
            self.X_grid = slice_push_only_grid(self.X_grid)
            self.X_cand = slice_push_only_cand(self.X_cand)
            self.u_global = slice_push_only_global(self.u_global)
        print(f"  total: {len(self.label)} samples; phases unique = "
              f"{np.unique(self.phase).tolist()}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx: int):
        # X_grid: stored as [10, 14, 30] -> PyTorch [30, 10, 14]
        xg = self.X_grid[idx].transpose(2, 0, 1)
        return {
            "X_grid": xg.astype(np.float32),
            "X_cand": self.X_cand[idx].astype(np.float32),
            "u_global": self.u_global[idx].astype(np.float32),
            "mask": self.mask[idx].astype(np.float32),
            "label": int(self.label[idx]),
            "phase": int(self.phase[idx]),
        }


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in ["X_grid", "X_cand", "u_global", "mask"]:
        out[k] = torch.from_numpy(np.stack([b[k] for b in batch], axis=0))
    out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    out["phase"] = torch.tensor([b["phase"] for b in batch], dtype=torch.long)
    return out


# ── Phase-stratified sampler (简化版: 加权随机) ─────────────

def make_phase_weights(phase: np.ndarray, target: Dict[int, float]) -> np.ndarray:
    """每样本权重: 让 phase 1-6 出现频率匹配 target dict.

    target: {phase_id (int 1-6): freq (float, 总和 = 1.0)}.
    """
    n = len(phase)
    actual = {p: int((phase == p).sum()) for p in np.unique(phase)}
    weights = np.ones(n, dtype=np.float64)
    total_target = sum(target.values()) or 1.0
    for p, freq in target.items():
        cnt = actual.get(p, 0)
        if cnt == 0:
            continue
        weights[phase == p] = freq / total_target / cnt * n
    return weights


# ── 损失 ──────────────────────────────────────────────────

def _model_forward_score_value(model, X_grid, X_cand, u_global, mask):
    """适配 push_only (2 输出) 和 full (5 输出) 两种模型."""
    out = model(X_grid, X_cand, u_global, mask)
    if len(out) == 2:
        return out[0], out[1]  # push_only
    return out[0], out[1]  # full — 后 3 个 aux 弃用


def compute_losses_direct(model, X_grid, X_cand, u_global, mask, label):
    """直接从张量计算损失. 新架构仅 L_policy + L_value."""
    score, value = _model_forward_score_value(model, X_grid, X_cand, u_global, mask)
    loss_policy = F.cross_entropy(score, label)
    target_value = torch.ones_like(value)
    loss_value = F.smooth_l1_loss(value, target_value)
    total = loss_policy + 0.3 * loss_value
    with torch.no_grad():
        pred = score.argmax(dim=-1)
        acc = (pred == label).float().mean().item()
    return total, {"policy": loss_policy.item(), "value": loss_value.item(), "acc": acc}


def compute_losses(model, batch, device):
    X_grid = batch["X_grid"].to(device)
    X_cand = batch["X_cand"].to(device)
    u_global = batch["u_global"].to(device)
    mask = batch["mask"].to(device)
    label = batch["label"].to(device)

    score, value = _model_forward_score_value(model, X_grid, X_cand, u_global, mask)
    loss_policy = F.cross_entropy(score, label)
    target_value = torch.ones_like(value)
    loss_value = F.smooth_l1_loss(value, target_value)
    total = loss_policy + 0.3 * loss_value
    with torch.no_grad():
        pred = score.argmax(dim=-1)
        acc = (pred == label).float().mean().item()
    return total, {"policy": loss_policy.item(), "value": loss_value.item(), "acc": acc}


# ── 训练循环 ──────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    npz_paths = []
    for p in args.data:
        if not os.path.isabs(p):
            p = os.path.join(ROOT, p)
        npz_paths.append(p)
    dataset = SagePrDataset(npz_paths, push_only=(args.arch == "push_only"))
    n = len(dataset)

    # GPU-resident: 一次性搬到 GPU (慎用, ~9GB for X_cand); 推荐 cpu-tensor 模式
    if args.gpu_resident and device.type == "cuda":
        # 优先放 CPU pinned tensor + transposed grid, 训练时按 batch 异步搬 GPU
        print("  preparing CPU pinned tensors (X_cand ~9GB stays on CPU)...")
        X_grid_t = torch.from_numpy(dataset.X_grid.transpose(0, 3, 1, 2).copy()).float().pin_memory()
        X_cand_t = torch.from_numpy(dataset.X_cand.copy()).float().pin_memory()
        u_global_t = torch.from_numpy(dataset.u_global.copy()).float().pin_memory()
        mask_t = torch.from_numpy(dataset.mask.copy()).float().pin_memory()
        label_t = torch.from_numpy(dataset.label.astype(np.int64)).pin_memory()
        phase_t = torch.from_numpy(dataset.phase.astype(np.int64))
        print(f"  CPU tensors pinned, X_grid {X_grid_t.shape} X_cand {X_cand_t.shape}")
    else:
        X_grid_t = X_cand_t = u_global_t = mask_t = label_t = phase_t = None

    # 划分 train / val 80:20
    np.random.seed(42)
    idx = np.random.permutation(n)
    n_train = int(n * 0.8)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_phases = dataset.phase[train_idx]
    if args.phase_dist == "default":
        target_dist = {1: 0.05, 2: 0.10, 3: 0.15, 4: 0.25, 5: 0.20, 6: 0.25}
    elif args.phase_dist == "hard":
        # 上采样 phase 5/6 (含炸弹), 下调 phase 1-2
        target_dist = {1: 0.03, 2: 0.05, 3: 0.10, 4: 0.20, 5: 0.32, 6: 0.30}
    else:
        raise ValueError(f"unknown phase_dist {args.phase_dist}")
    weights_full = make_phase_weights(train_phases, target_dist)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights_full.tolist(), num_samples=len(train_idx), replacement=True
    )

    train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_subset = torch.utils.data.Subset(dataset, val_idx.tolist())

    if args.gpu_resident and X_grid_t is not None:
        # CPU-pinned: 直接索引 CPU 张量, 用 .to(device, non_blocking=True) 异步搬
        train_idx_t = torch.from_numpy(train_idx)
        val_idx_t = torch.from_numpy(val_idx)
        train_weights_t = torch.from_numpy(weights_full).float()
        train_loader = None
        val_loader = None
    else:
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, sampler=sampler,
            collate_fn=collate, num_workers=args.num_workers,
            pin_memory=True, persistent_workers=(args.num_workers > 0),
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate, num_workers=args.num_workers,
            pin_memory=True, persistent_workers=(args.num_workers > 0),
        )
        train_idx_t = val_idx_t = train_weights_t = None

    if args.arch == "push_only":
        model_builder = build_push_only_large if args.model == "large" else build_push_only_model
    else:
        model_builder = build_large_model if args.model == "large" else build_default_model
    model = model_builder().to(device)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        print(f"loaded init ckpt {args.init_ckpt}, prev val_acc={ck.get('val_acc', '?'):.3f}")
    print(f"model params: {model.num_parameters():,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_epoch = args.epochs
    if args.gpu_resident and X_grid_t is not None:
        n_iter_per_epoch = max(1, len(train_idx) // args.batch_size)
    else:
        n_iter_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epoch * n_iter_per_epoch, eta_min=args.lr * 0.1,
    )

    out_dir = os.path.join(ROOT, ".agent/sage_pr/runs", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    best_val_acc = 0.0
    log_path = os.path.join(out_dir, "train.log")
    log_f = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"epochs={n_epoch}, batch={args.batch_size}, lr={args.lr}")
    log(f"train n={len(train_idx)}, val n={len(val_idx)}")

    t0 = time.perf_counter()
    for epoch in range(n_epoch):
        model.train()
        accum: Dict[str, float] = {}
        n_batches = 0

        if args.gpu_resident and X_grid_t is not None:
            samples_per_epoch = len(train_idx_t)
            # train_weights_t 已经是 train_idx 对齐的 (make_phase_weights(train_phases))
            sampled_pos = torch.multinomial(
                train_weights_t, samples_per_epoch, replacement=True
            )
            sampled = train_idx_t[sampled_pos]
            for b_start in range(0, samples_per_epoch, args.batch_size):
                b_end = min(b_start + args.batch_size, samples_per_epoch)
                bi = sampled[b_start:b_end]
                # pinned tensor index → async copy to GPU
                xg = X_grid_t[bi].to(device, non_blocking=True)
                xc = X_cand_t[bi].to(device, non_blocking=True)
                ug = u_global_t[bi].to(device, non_blocking=True)
                mk = mask_t[bi].to(device, non_blocking=True)
                lb = label_t[bi].to(device, non_blocking=True)
                optimizer.zero_grad()
                loss, metrics = compute_losses_direct(model, xg, xc, ug, mk, lb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                for k, v in metrics.items():
                    accum[k] = accum.get(k, 0.0) + v
                accum["loss"] = accum.get("loss", 0.0) + loss.item()
                n_batches += 1
        else:
            for batch in train_loader:
                optimizer.zero_grad()
                loss, metrics = compute_losses(model, batch, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                for k, v in metrics.items():
                    accum[k] = accum.get(k, 0.0) + v
                accum["loss"] = accum.get("loss", 0.0) + loss.item()
                n_batches += 1

        train_metrics = {k: v / max(n_batches, 1) for k, v in accum.items()}

        # Val
        model.eval()
        val_acc_total = 0.0
        val_loss_total = 0.0
        val_n = 0
        per_phase_acc: Dict[int, Tuple[int, int]] = {}
        with torch.no_grad():
            if args.gpu_resident and X_grid_t is not None:
                for b_start in range(0, len(val_idx_t), args.batch_size):
                    bi = val_idx_t[b_start:b_start + args.batch_size]
                    xg = X_grid_t[bi].to(device, non_blocking=True)
                    xc = X_cand_t[bi].to(device, non_blocking=True)
                    ug = u_global_t[bi].to(device, non_blocking=True)
                    mk = mask_t[bi].to(device, non_blocking=True)
                    lb = label_t[bi].to(device, non_blocking=True)
                    ph_t = phase_t[bi]
                    loss, metrics = compute_losses_direct(model, xg, xc, ug, mk, lb)
                    B = lb.size(0)
                    val_acc_total += metrics["acc"] * B
                    val_loss_total += loss.item() * B
                    val_n += B
                    pred = model(xg, xc, ug, mk)[0].argmax(dim=-1).cpu().numpy()
                    lbl = lb.cpu().numpy()
                    ph = ph_t.cpu().numpy()
                    for i in range(B):
                        p = int(ph[i])
                        correct = int(pred[i] == lbl[i])
                        cnt, ok = per_phase_acc.get(p, (0, 0))
                        per_phase_acc[p] = (cnt + 1, ok + correct)
            else:
                for batch in val_loader:
                    loss, metrics = compute_losses(model, batch, device)
                    B = batch["label"].size(0)
                    val_acc_total += metrics["acc"] * B
                    val_loss_total += loss.item() * B
                    val_n += B
                    pred = (model(batch["X_grid"].to(device),
                                  batch["X_cand"].to(device),
                                  batch["u_global"].to(device),
                                  batch["mask"].to(device))[0]).argmax(dim=-1).cpu().numpy()
                    lbl = batch["label"].numpy()
                    ph = batch["phase"].numpy()
                    for i in range(B):
                        p = int(ph[i])
                        correct = int(pred[i] == lbl[i])
                        cnt, ok = per_phase_acc.get(p, (0, 0))
                        per_phase_acc[p] = (cnt + 1, ok + correct)

        val_acc = val_acc_total / max(val_n, 1)
        val_loss = val_loss_total / max(val_n, 1)
        ph_acc_str = ", ".join([
            f"p{p}={ok / cnt:.3f} ({cnt})"
            for p, (cnt, ok) in sorted(per_phase_acc.items())
        ])

        elapsed = time.perf_counter() - t0
        log(f"epoch {epoch + 1:3d}/{n_epoch} | "
            f"train loss={train_metrics['loss']:.3f} acc={train_metrics['acc']:.3f} | "
            f"val loss={val_loss:.3f} acc={val_acc:.3f} | "
            f"per-phase: {ph_acc_str} | {elapsed:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(out_dir, "best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "args": vars(args),
                "model_size": args.model,
                "model_arch": args.arch,
            }, ckpt_path)
            log(f"  saved best (val_acc={val_acc:.3f}) → {ckpt_path}")

    log_f.close()
    print("done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True,
                        help="npz paths (one or more)")
    parser.add_argument("--tag", required=True, help="run tag")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--init-ckpt", default=None,
                        help="path to initial ckpt (for fine-tune / DAgger).")
    parser.add_argument("--phase-dist", default="default", choices=["default", "hard"],
                        help="phase 采样权重: default=5/10/15/25/20/25, hard=3/5/10/20/32/30")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (0 = main thread). >0 大幅减 GPU 等待.")
    parser.add_argument("--gpu-resident", action="store_true",
                        help="把整数据集一次性搬 GPU (~5GB), 跳过 DataLoader, GPU 满载.")
    parser.add_argument("--model", default="default", choices=["default", "large"],
                        help="model size: default=105K, large=194K.")
    parser.add_argument("--arch", default="push_only", choices=["push_only", "full"],
                        help="push_only (新架构, 默认, 配 explorer 算法 + NN 推箱); "
                             "full (旧架构, 含 inspect / info_gain 等 aux heads).")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
