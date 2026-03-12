from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.solver_bc.oracle_features import checkpoint_payload


class MaskedBCPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


@dataclass
class EvalStats:
    loss: float
    acc: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def masked_loss_and_acc(model: nn.Module, obs: torch.Tensor, masks: torch.Tensor,
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = model(obs)
    masked_logits = logits.masked_fill(~masks, -1e9)
    loss = F.cross_entropy(masked_logits, actions)
    pred = masked_logits.argmax(dim=1)
    acc = (pred == actions).float().mean()
    return loss, acc


def should_use_gpu_resident(device: torch.device, mode: str, obs: torch.Tensor,
                            masks: torch.Tensor, actions: torch.Tensor) -> bool:
    if mode == "off":
        return False
    if device.type != "cuda":
        return False
    if mode == "on":
        return True

    total_bytes = (
        obs.numel() * obs.element_size()
        + masks.numel() * masks.element_size()
        + actions.numel() * actions.element_size()
    )
    return total_bytes <= 512 * 1024 * 1024


def iterate_gpu_batches(obs: torch.Tensor, masks: torch.Tensor, actions: torch.Tensor,
                        indices: torch.Tensor, batch_size: int,
                        shuffle: bool) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if indices.numel() == 0:
        return

    if shuffle:
        order = indices[torch.randperm(indices.numel(), device=indices.device)]
    else:
        order = indices

    for start in range(0, int(order.numel()), batch_size):
        batch_idx = order[start:start + batch_size]
        yield obs[batch_idx], masks[batch_idx], actions[batch_idx]


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalStats:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_items = 0
    use_non_blocking = device.type == "cuda"
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else contextlib.nullcontext()
    )

    with torch.no_grad():
        for obs, masks, actions in loader:
            obs = obs.to(device, non_blocking=use_non_blocking)
            masks = masks.to(device, non_blocking=use_non_blocking)
            actions = actions.to(device, non_blocking=use_non_blocking)
            with amp_ctx:
                loss, acc = masked_loss_and_acc(model, obs, masks, actions)
            batch = obs.shape[0]
            total_loss += float(loss.item()) * batch
            total_acc += float(acc.item()) * batch
            total_items += batch

    if total_items == 0:
        return EvalStats(loss=float("nan"), acc=float("nan"))
    return EvalStats(
        loss=total_loss / max(total_items, 1),
        acc=total_acc / max(total_items, 1),
    )


def evaluate_gpu_dataset(model: nn.Module, obs: torch.Tensor, masks: torch.Tensor,
                         actions: torch.Tensor, indices: torch.Tensor,
                         batch_size: int, use_amp: bool) -> EvalStats:
    if indices.numel() == 0:
        return EvalStats(loss=float("nan"), acc=float("nan"))

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_items = 0
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp else contextlib.nullcontext()
    )

    with torch.no_grad():
        for batch_obs, batch_masks, batch_actions in iterate_gpu_batches(
            obs,
            masks,
            actions,
            indices,
            batch_size,
            shuffle=False,
        ):
            with amp_ctx:
                loss, acc = masked_loss_and_acc(model, batch_obs, batch_masks, batch_actions)
            batch = batch_obs.shape[0]
            total_loss += float(loss.item()) * batch
            total_acc += float(acc.item()) * batch
            total_items += batch

    return EvalStats(
        loss=total_loss / max(total_items, 1),
        acc=total_acc / max(total_items, 1),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-resident", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.dataset, allow_pickle=True)
    obs = torch.from_numpy(data["obs"]).float()
    masks = torch.from_numpy(data["masks"]).bool()
    actions = torch.from_numpy(data["actions"]).long()

    include_map_layout = obs.shape[1] > 62
    obs_mode = str(data["obs_mode"]).strip().lower() if "obs_mode" in data else "oracle"
    n_items = obs.shape[0]
    perm = np.random.permutation(n_items)
    if args.val_ratio <= 0.0:
        val_idx = perm[:0]
        train_idx = perm
    else:
        val_size = max(1, int(n_items * args.val_ratio))
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]
        if len(train_idx) == 0:
            train_idx = val_idx

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model = MaskedBCPolicy(obs.shape[1], masks.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_gpu_resident = should_use_gpu_resident(device, args.gpu_resident, obs, masks, actions)

    if use_gpu_resident:
        obs = obs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        train_indices = torch.as_tensor(train_idx, device=device, dtype=torch.long)
        val_indices = torch.as_tensor(val_idx, device=device, dtype=torch.long)
        train_loader = None
        val_loader = None
    else:
        train_ds = TensorDataset(obs[train_idx], masks[train_idx], actions[train_idx])
        val_ds = TensorDataset(obs[val_idx], masks[val_idx], actions[val_idx])
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )
        train_indices = None
        val_indices = None

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_items = 0
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp else contextlib.nullcontext()
        )

        if use_gpu_resident:
            assert train_indices is not None
            batch_iter = iterate_gpu_batches(
                obs,
                masks,
                actions,
                train_indices,
                args.batch_size,
                shuffle=True,
            )
        else:
            assert train_loader is not None
            batch_iter = train_loader

        for batch_obs, batch_masks, batch_actions in batch_iter:
            if not use_gpu_resident:
                batch_obs = batch_obs.to(device, non_blocking=pin_memory)
                batch_masks = batch_masks.to(device, non_blocking=pin_memory)
                batch_actions = batch_actions.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                loss, acc = masked_loss_and_acc(model, batch_obs, batch_masks, batch_actions)
            loss.backward()
            optimizer.step()

            batch = batch_obs.shape[0]
            train_loss += float(loss.item()) * batch
            train_acc += float(acc.item()) * batch
            train_items += batch

        train_stats = EvalStats(
            loss=train_loss / max(train_items, 1),
            acc=train_acc / max(train_items, 1),
        )

        if use_gpu_resident:
            assert val_indices is not None
            val_stats = evaluate_gpu_dataset(
                model,
                obs,
                masks,
                actions,
                val_indices,
                args.batch_size,
                use_amp=use_amp,
            )
        else:
            assert val_loader is not None
            val_stats = evaluate_loader(model, val_loader, device)

        row: Dict[str, float] = {
            "epoch": epoch,
            "train_loss": round(train_stats.loss, 6),
            "train_acc": round(train_stats.acc, 6),
            "val_loss": round(val_stats.loss, 6),
            "val_acc": round(val_stats.acc, 6),
        }
        history.append(row)
        if args.log_every > 0 and (epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs):
            print(json.dumps(row, ensure_ascii=False))

        monitor_loss = val_stats.loss if len(val_idx) > 0 else train_stats.loss
        if monitor_loss < best_val:
            best_val = monitor_loss
            torch.save(
                checkpoint_payload(
                    model.state_dict(),
                    hidden_dim=args.hidden_dim,
                    include_map_layout=include_map_layout,
                    obs_mode=obs_mode,
                ),
                os.path.join(args.output_dir, "best.pt"),
            )

    torch.save(
        checkpoint_payload(
            model.state_dict(),
            hidden_dim=args.hidden_dim,
            include_map_layout=include_map_layout,
            obs_mode=obs_mode,
        ),
        os.path.join(args.output_dir, "last.pt"),
    )

    summary = {
        "dataset": os.path.abspath(args.dataset),
        "output_dir": os.path.abspath(args.output_dir),
        "samples": int(n_items),
        "train_items": int(len(train_idx)),
        "val_items": int(len(val_idx)),
        "epochs": args.epochs,
        "best_val_loss": round(best_val, 6),
        "include_map_layout": include_map_layout,
        "obs_mode": obs_mode,
        "device": str(device),
        "gpu_resident": use_gpu_resident,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
