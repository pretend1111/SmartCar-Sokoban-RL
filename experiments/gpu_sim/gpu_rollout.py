from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.gpu_sim.gpu_push_env import GpuPushBatchEnv
from experiments.solver_bc.branch_search import (
    RolloutRequest,
    RolloutResult,
    StepDecision,
)
from experiments.solver_bc.train_bc import MaskedBCPolicy
from smartcar_sokoban.rl.high_level_env import (
    DIR_DELTAS,
    N_DIRS,
    NO_PROGRESS_LIMIT_MAX,
    NO_PROGRESS_LIMIT_MIN,
    OSCILLATION_LIMIT,
    OSCILLATION_LOOKBACK,
    PUSH_BOMB_START,
    PUSH_BOX_START,
)
from smartcar_sokoban.rl.train import CURRICULUM


@dataclass
class _GpuActiveRollout:
    request: RolloutRequest
    actions: List[int]
    decisions: List[StepDecision]
    obs_trace: List[np.ndarray]
    mask_trace: List[np.ndarray]


def _parse_box_push(action: int) -> Optional[Tuple[int, int]]:
    if not (PUSH_BOX_START <= action < PUSH_BOMB_START):
        return None
    offset = action - PUSH_BOX_START
    return offset // N_DIRS, offset % N_DIRS


def _is_reverse_box_push(prev_action: int, action: int) -> bool:
    prev_push = _parse_box_push(prev_action)
    cur_push = _parse_box_push(action)
    if prev_push is None or cur_push is None:
        return False
    if prev_push[0] != cur_push[0]:
        return False
    prev_dx, prev_dy = DIR_DELTAS[prev_push[1]]
    cur_dx, cur_dy = DIR_DELTAS[cur_push[1]]
    return prev_dx + cur_dx == 0 and prev_dy + cur_dy == 0


def _reverse_box_push_mask(prev_actions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    prev_is_box = (prev_actions >= PUSH_BOX_START) & (prev_actions < PUSH_BOMB_START)
    cur_is_box = (actions >= PUSH_BOX_START) & (actions < PUSH_BOMB_START)
    both_box = prev_is_box & cur_is_box
    prev_offset = prev_actions - PUSH_BOX_START
    cur_offset = actions - PUSH_BOX_START
    prev_slot = torch.div(prev_offset, N_DIRS, rounding_mode="floor")
    cur_slot = torch.div(cur_offset, N_DIRS, rounding_mode="floor")
    prev_dir = torch.remainder(prev_offset, N_DIRS)
    cur_dir = torch.remainder(cur_offset, N_DIRS)
    return both_box & (prev_slot == cur_slot) & ((prev_dir ^ 1) == cur_dir)


def _rank_tensor(
    model: MaskedBCPolicy,
    obs: torch.Tensor,
    masks: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        logits = model(obs)
        masked_logits = logits.masked_fill(~masks, -1e9)
        probs = torch.softmax(masked_logits, dim=1)
        probs = probs * masks.to(dtype=probs.dtype)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        top_count = min(int(top_k), int(probs.shape[1]))
        top_probs, top_actions = torch.topk(probs, k=top_count, dim=1)
    return probs, top_actions, top_probs, entropy


def _checkpoint_obs_mode(checkpoint_payload: Dict[str, object]) -> str:
    mode = str(checkpoint_payload.get("obs_mode", "oracle")).strip().lower()
    if mode not in {"oracle", "state"}:
        raise ValueError(f"unknown checkpoint obs_mode: {mode}")
    return mode


def _build_model_obs(env: GpuPushBatchEnv, checkpoint_payload: Dict[str, object]) -> torch.Tensor:
    include_map_layout = bool(checkpoint_payload["include_map_layout"])
    obs_mode = _checkpoint_obs_mode(checkpoint_payload)
    if obs_mode == "oracle":
        return env.build_oracle_obs()
    return env.build_state_obs(include_map_layout=include_map_layout)


def rollout_with_prefixes_gpu(
    model: MaskedBCPolicy,
    checkpoint_payload: Dict[str, object],
    phase: int,
    requests: Sequence[RolloutRequest],
    device: torch.device,
    branch_top_k: int,
    capture_traces: bool = True,
) -> List[RolloutResult]:
    if not requests:
        return []
    if phase not in CURRICULUM:
        raise ValueError(f"unknown phase: {phase}")

    phase_cfg = CURRICULUM[phase]
    include_map_layout = bool(checkpoint_payload["include_map_layout"])

    env = GpuPushBatchEnv.from_requests(
        list(requests),
        max_steps=phase_cfg["max_steps"],
        base_dir=ROOT,
        device=device,
    )
    trackers = [
        _GpuActiveRollout(
            request=request,
            actions=[],
            decisions=[],
            obs_trace=[],
            mask_trace=[],
        )
        for request in requests
    ]
    final_results: List[Optional[RolloutResult]] = [None] * len(requests)
    active = torch.ones(len(requests), dtype=torch.bool, device=device)
    high_steps = torch.zeros(len(requests), dtype=torch.long, device=device)
    oscillation_streak = torch.zeros(len(requests), dtype=torch.long, device=device)
    no_progress_streak = torch.zeros(len(requests), dtype=torch.long, device=device)
    recent_hashes = torch.full(
        (len(requests), OSCILLATION_LOOKBACK),
        -1,
        dtype=torch.long,
        device=device,
    )
    recent_hashes[:, -1] = env.state_hash()
    last_actions = torch.full((len(requests),), -1, dtype=torch.long, device=device)
    prefix_lengths = torch.tensor(
        [len(request.prefix) for request in requests],
        dtype=torch.long,
        device=device,
    )
    max_prefix_len = max((len(request.prefix) for request in requests), default=0)
    prefix_actions = torch.full(
        (len(requests), max(max_prefix_len, 1)),
        -1,
        dtype=torch.long,
        device=device,
    )
    for row, request in enumerate(requests):
        if request.prefix:
            prefix_actions[row, :len(request.prefix)] = torch.tensor(
                request.prefix,
                dtype=torch.long,
                device=device,
            )
    batch_idx = torch.arange(len(requests), device=device)

    def _finalize_rows(
        rows: List[int],
        *,
        won: bool,
        truncated_reason: str,
        low_steps_by_row: Optional[np.ndarray] = None,
        high_steps_by_row: Optional[np.ndarray] = None,
        invalid_prefix_at: int = -1,
    ) -> None:
        for row in rows:
            tracker = trackers[row]
            final_results[row] = RolloutResult(
                prefix=tracker.request.prefix,
                actions=list(tracker.actions),
                decisions=list(tracker.decisions),
                obs=list(tracker.obs_trace),
                masks=list(tracker.mask_trace),
                won=won,
                low_steps=(int(low_steps_by_row[row]) if low_steps_by_row is not None else int(10**9)),
                high_steps=(int(high_steps_by_row[row]) if high_steps_by_row is not None else len(tracker.actions)),
                truncated_reason=truncated_reason,
                invalid_prefix_at=invalid_prefix_at,
            )

    for _ in range(phase_cfg["max_steps"]):
        if not torch.any(active):
            break

        obs = _build_model_obs(env, checkpoint_payload)
        masks = env.action_masks()
        masks = masks & active.unsqueeze(1)
        valid_any = masks.any(dim=1)

        no_valid_rows = torch.nonzero(active & ~valid_any, as_tuple=False).squeeze(1)
        if no_valid_rows.numel() > 0:
            high_steps_cpu = high_steps.detach().cpu().numpy()
            _finalize_rows(
                no_valid_rows.detach().cpu().tolist(),
                won=False,
                truncated_reason="no_valid_actions",
                high_steps_by_row=high_steps_cpu,
            )
        active &= valid_any
        if not torch.any(active):
            break

        probs, top_actions, top_probs, entropy = _rank_tensor(
            model=model,
            obs=obs,
            masks=masks,
            top_k=branch_top_k,
        )

        prev_hash = env.state_hash()
        prev_box_count = env.box_alive.sum(dim=1)
        prev_dist = env.box_distance_sums()
        prev_seen = env.seen_box.sum(dim=1) + env.seen_target.sum(dim=1)
        valid_counts = masks.sum(dim=1).clamp(max=int(top_actions.shape[1]))
        margin = torch.ones_like(entropy)
        if top_probs.shape[1] > 1:
            margin = torch.where(
                valid_counts >= 2,
                top_probs[:, 0] - top_probs[:, 1],
                margin,
            )

        forced = active & (high_steps < prefix_lengths)
        safe_step_index = high_steps.clamp(max=prefix_actions.shape[1] - 1)
        forced_actions = prefix_actions[batch_idx, safe_step_index]
        safe_forced_actions = forced_actions.clamp(0, masks.shape[1] - 1)
        forced_valid = forced & (forced_actions >= 0) & masks[batch_idx, safe_forced_actions]
        invalid_forced = forced & ~forced_valid

        if invalid_forced.any():
            invalid_rows = torch.nonzero(invalid_forced, as_tuple=False).squeeze(1)
            high_steps_cpu = high_steps.detach().cpu().numpy()
            invalid_step_cpu = high_steps.detach().cpu().numpy()
            for row in invalid_rows.detach().cpu().tolist():
                _finalize_rows(
                    [row],
                    won=False,
                    truncated_reason="invalid_prefix",
                    high_steps_by_row=high_steps_cpu,
                    invalid_prefix_at=int(invalid_step_cpu[row]),
                )
            active &= ~invalid_forced

        if not torch.any(active):
            break

        chosen_actions = top_actions[:, 0].clone()
        chosen_actions = torch.where(forced_valid, forced_actions, chosen_actions)
        chosen_probs = top_probs[:, 0].clone()
        forced_probs = probs[batch_idx, safe_forced_actions]
        chosen_probs = torch.where(forced_valid, forced_probs, chosen_probs)
        step_mask = active.clone()

        active_rows = torch.nonzero(step_mask, as_tuple=False).squeeze(1)
        active_rows_cpu = active_rows.detach().cpu().tolist()
        top_actions_cpu = top_actions.detach().cpu().numpy()
        top_probs_cpu = top_probs.detach().cpu().numpy()
        entropy_cpu = entropy.detach().cpu().numpy()
        margin_cpu = margin.detach().cpu().numpy()
        valid_counts_cpu = valid_counts.detach().cpu().numpy()
        chosen_actions_cpu = chosen_actions.detach().cpu().numpy()
        chosen_probs_cpu = chosen_probs.detach().cpu().numpy()
        forced_cpu = forced.detach().cpu().numpy()
        high_steps_cpu = high_steps.detach().cpu().numpy()
        if capture_traces:
            obs_cpu = obs.detach().cpu().numpy()
            masks_cpu = masks.detach().cpu().numpy()

        for row in active_rows_cpu:
            tracker = trackers[row]
            top_count = int(valid_counts_cpu[row])
            row_top_actions = [int(x) for x in top_actions_cpu[row, :top_count]]
            row_top_probs = [float(x) for x in top_probs_cpu[row, :top_count]]
            if capture_traces:
                tracker.obs_trace.append(obs_cpu[row].copy())
                tracker.mask_trace.append(masks_cpu[row].copy())
            tracker.decisions.append(
                StepDecision(
                    step=int(high_steps_cpu[row]),
                    action=int(chosen_actions_cpu[row]),
                    action_prob=float(chosen_probs_cpu[row]),
                    top_actions=row_top_actions,
                    top_probs=row_top_probs,
                    margin=float(margin_cpu[row]),
                    entropy=float(entropy_cpu[row]),
                    forced=bool(forced_cpu[row]),
                )
            )
            tracker.actions.append(int(chosen_actions_cpu[row]))
        high_steps[step_mask] += 1

        if not torch.any(step_mask):
            break

        step_result = env.step(chosen_actions, active_mask=step_mask)
        dead_boxes = env.dead_box_counts()
        new_hash = env.state_hash()
        box_eliminated = (prev_box_count - env.box_alive.sum(dim=1)) > 0
        distance_delta = prev_dist - env.box_distance_sums()
        newly_seen = (env.seen_box.sum(dim=1) + env.seen_target.sum(dim=1)) > prev_seen
        revisiting = (
            step_mask
            & (recent_hashes[:, :-1] == new_hash.unsqueeze(1)).any(dim=1)
        )
        reverse_rows = _reverse_box_push_mask(last_actions, chosen_actions) & step_mask
        oscillation_streak = torch.where(
            revisiting,
            oscillation_streak + 1,
            torch.where(step_mask, torch.zeros_like(oscillation_streak), oscillation_streak),
        )
        progress_made = (
            box_eliminated
            | newly_seen
            | ((distance_delta > 0) & ~revisiting)
            | ((new_hash != prev_hash) & ~revisiting)
        )
        no_progress_streak = torch.where(
            step_mask,
            torch.where(progress_made, torch.zeros_like(no_progress_streak), no_progress_streak + 1),
            no_progress_streak,
        )
        recent_hashes[step_mask] = torch.roll(recent_hashes[step_mask], shifts=-1, dims=1)
        recent_hashes[step_mask, -1] = new_hash[step_mask]
        last_actions[step_mask] = chosen_actions[step_mask]

        no_progress_limit = max(
            NO_PROGRESS_LIMIT_MIN,
            min(NO_PROGRESS_LIMIT_MAX, phase_cfg["max_steps"] // 3),
        )
        terminated = step_mask & step_result.won
        truncated_candidates = step_mask & ~step_result.won
        trunc_code = torch.zeros(len(requests), dtype=torch.long, device=device)
        trunc_code = torch.where(truncated_candidates & (dead_boxes > 0), torch.ones_like(trunc_code), trunc_code)
        trunc_code = torch.where(
            truncated_candidates & (trunc_code == 0) & (oscillation_streak >= OSCILLATION_LIMIT),
            torch.full_like(trunc_code, 2),
            trunc_code,
        )
        trunc_code = torch.where(
            truncated_candidates & (trunc_code == 0) & (no_progress_streak >= no_progress_limit),
            torch.full_like(trunc_code, 3),
            trunc_code,
        )

        finalize_mask = terminated | (trunc_code > 0)
        if finalize_mask.any():
            total_low_steps_cpu = env.total_low_steps.detach().cpu().numpy()
            high_steps_after_cpu = high_steps.detach().cpu().numpy()
            won_rows = torch.nonzero(terminated, as_tuple=False).squeeze(1).detach().cpu().tolist()
            dead_rows = torch.nonzero(trunc_code == 1, as_tuple=False).squeeze(1).detach().cpu().tolist()
            osc_rows = torch.nonzero(trunc_code == 2, as_tuple=False).squeeze(1).detach().cpu().tolist()
            nop_rows = torch.nonzero(trunc_code == 3, as_tuple=False).squeeze(1).detach().cpu().tolist()
            if won_rows:
                _finalize_rows(
                    won_rows,
                    won=True,
                    truncated_reason="",
                    low_steps_by_row=total_low_steps_cpu,
                    high_steps_by_row=high_steps_after_cpu,
                )
            if dead_rows:
                _finalize_rows(
                    dead_rows,
                    won=False,
                    truncated_reason="dead_box",
                    low_steps_by_row=total_low_steps_cpu,
                    high_steps_by_row=high_steps_after_cpu,
                )
            if osc_rows:
                _finalize_rows(
                    osc_rows,
                    won=False,
                    truncated_reason="oscillation",
                    low_steps_by_row=total_low_steps_cpu,
                    high_steps_by_row=high_steps_after_cpu,
                )
            if nop_rows:
                _finalize_rows(
                    nop_rows,
                    won=False,
                    truncated_reason="no_progress",
                    low_steps_by_row=total_low_steps_cpu,
                    high_steps_by_row=high_steps_after_cpu,
                )
        active &= ~finalize_mask

    remaining = torch.nonzero(active, as_tuple=False).squeeze(1)
    if remaining.numel() > 0:
        total_low_steps_cpu = env.total_low_steps.detach().cpu().numpy()
        high_steps_cpu = high_steps.detach().cpu().numpy()
        _finalize_rows(
            remaining.detach().cpu().tolist(),
            won=False,
            truncated_reason="max_steps",
            low_steps_by_row=total_low_steps_cpu,
            high_steps_by_row=high_steps_cpu,
        )

    if any(result is None for result in final_results):
        raise RuntimeError("gpu rollout finished with incomplete results")
    return [result for result in final_results if result is not None]
