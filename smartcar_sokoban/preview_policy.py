"""Visualize a trained MaskablePPO policy on a curriculum phase.

Exports one GIF and one JSON trace per map, plus a Markdown summary.

Usage:
    python preview_policy.py --phase 3
    python preview_policy.py --phase 3 --model C:/.../phase3_final.zip
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from smartcar_sokoban.config import GameConfig
from smartcar_sokoban.paths import PROJECT_ROOT, RUNS_ROOT
from smartcar_sokoban.renderer import Renderer
from smartcar_sokoban.rl.high_level_env import (
    DIR_UP,
    DIR_DOWN,
    DIR_LEFT,
    DIR_RIGHT,
    EXPLORE_BOX_START,
    EXPLORE_TGT_START,
    MAX_BOXES,
    MAX_TARGETS,
    N_DIRS,
    N_BOMB_DIRS,
    PUSH_BOX_START,
    PUSH_BOMB_START,
    SokobanHLEnv,
    STATE_DIM_WITH_MAP,
)
from smartcar_sokoban.rl.train import CURRICULUM, load_seed_manifest
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


LOW_LEVEL_ACTION_NAMES = {
    0: "MOVE_FWD",
    1: "MOVE_BACK",
    2: "STRAFE_LEFT",
    3: "STRAFE_RIGHT",
    4: "TURN_LEFT",
    5: "TURN_RIGHT",
    6: "SNAP/NOOP",
}

DIR_NAMES = {
    DIR_UP: "UP",
    DIR_DOWN: "DOWN",
    DIR_LEFT: "LEFT",
    DIR_RIGHT: "RIGHT",
}

BOMB_DIR_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "UP_LEFT",
    5: "UP_RIGHT",
    6: "DOWN_LEFT",
    7: "DOWN_RIGHT",
}


@dataclass
class StepTrace:
    step: int
    action_id: int
    action_name: str
    reward: float
    low_level_steps: int
    low_level_actions: List[str]
    valid_actions: int
    remaining_boxes: int
    remaining_bombs: int
    dead_boxes: int
    no_progress_streak: int
    oscillation_streak: int
    state_revisit_count: int
    won: bool
    terminated: bool
    truncated: bool
    truncated_reason: str


def detect_include_map_layout(model: MaskablePPO) -> bool:
    obs_shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if not obs_shape:
        return False
    obs_dim = int(np.prod(obs_shape))
    return obs_dim == STATE_DIM_WITH_MAP


def action_name(action: int) -> str:
    if EXPLORE_BOX_START <= action < EXPLORE_TGT_START:
        return f"EXPLORE_BOX[{action - EXPLORE_BOX_START}]"
    if EXPLORE_TGT_START <= action < PUSH_BOX_START:
        return f"EXPLORE_TGT[{action - EXPLORE_TGT_START}]"
    if PUSH_BOX_START <= action < PUSH_BOMB_START:
        offset = action - PUSH_BOX_START
        box_idx = offset // N_DIRS
        dir_idx = offset % N_DIRS
        return f"PUSH_BOX[{box_idx}]_{DIR_NAMES.get(dir_idx, dir_idx)}"
    offset = action - PUSH_BOMB_START
    bomb_idx = offset // N_BOMB_DIRS
    dir_idx = offset % N_BOMB_DIRS
    return f"PUSH_BOMB[{bomb_idx}]_{BOMB_DIR_NAMES.get(dir_idx, dir_idx)}"


def choose_seed(map_name: str, seed_manifest: Dict[str, List[int]], seed: Optional[int]) -> int:
    if seed is not None:
        return seed
    verified = seed_manifest.get(map_name, [])
    if verified:
        return int(verified[0])
    return 7


def get_phase_maps(phase: int) -> List[str]:
    phase_dir = os.path.join(PROJECT_ROOT, CURRICULUM[phase]["map_dir"])
    return [
        os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")
        for path in sorted(glob.glob(os.path.join(phase_dir, "*.txt")))
    ]


def annotate_frame(image: Image.Image, lines: List[str]) -> Image.Image:
    map_img = image.convert("RGB")
    font = ImageFont.load_default()
    pad = 8
    line_h = 14
    panel_h = pad * 2 + line_h * len(lines)

    img = Image.new(
        "RGB",
        (map_img.width, map_img.height + panel_h),
        color=(18, 18, 18),
    )
    img.paste(map_img, (0, 0))

    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle(
        (0, map_img.height, img.width, img.height),
        fill=(18, 18, 18, 255),
    )
    y = map_img.height + pad
    for line in lines:
        draw.text((pad, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_h
    return img


def render_state(renderer: Renderer, state, scale: float, lines: List[str]) -> Image.Image:
    pixels = renderer.render(state, None)
    img = Image.fromarray(pixels)
    if scale != 1.0:
        w = max(1, int(img.width * scale))
        h = max(1, int(img.height * scale))
        img = img.resize((w, h), Image.Resampling.BILINEAR)
    return annotate_frame(img, lines)


def gif_safe_frames(frames: List[Image.Image]) -> List[Image.Image]:
    return [frame.convert("P", palette=Image.Palette.ADAPTIVE) for frame in frames]


def model_default_path(phase: int) -> str:
    return os.path.join(RUNS_ROOT, "rl", "models", f"phase{phase}_final.zip")


def run_one_map(
    model: MaskablePPO,
    map_path: str,
    max_steps: int,
    include_map_layout: bool,
    seed_manifest: Dict[str, List[int]],
    seed: Optional[int],
    renderer: Renderer,
    scale: float,
) -> Dict[str, Any]:
    basename = os.path.basename(map_path)
    chosen_seed = choose_seed(basename, seed_manifest, seed)
    env = SokobanHLEnv(
        map_file=map_path,
        base_dir=str(PROJECT_ROOT),
        max_steps=max_steps,
        baseline_steps=100,
        seed_manifest=seed_manifest,
        include_map_layout=include_map_layout,
    )
    masked = ActionMasker(env, lambda e: e.action_masks())
    obs, info = masked.reset(seed=chosen_seed)

    original_discrete_step = env.engine.discrete_step
    recorded_low_states: List[Tuple[int, Any]] = []

    def wrapped_discrete_step(low_action: int):
        state = original_discrete_step(low_action)
        recorded_low_states.append((int(low_action), copy.deepcopy(state)))
        return state

    env.engine.discrete_step = wrapped_discrete_step  # type: ignore[method-assign]

    frames: List[Image.Image] = []
    traces: List[StepTrace] = []
    current_state = copy.deepcopy(env.engine.get_state())
    frames.append(
        render_state(
            renderer,
            current_state,
            scale,
            [
                f"{basename} | seed={chosen_seed}",
                "step=START",
                f"boxes={len(current_state.boxes)} bombs={len(current_state.bombs)}",
            ],
        )
    )

    done = False
    high_step = 0
    final_info = dict(info)

    while not done:
        masks = env.action_masks()
        valid_actions = int(np.sum(masks))
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)
        action = int(action)
        before = len(recorded_low_states)
        obs, reward, terminated, truncated, info = masked.step(action)
        final_info = dict(info)
        low_segment = recorded_low_states[before:]
        low_names = [LOW_LEVEL_ACTION_NAMES.get(a, str(a)) for a, _ in low_segment]

        step_trace = StepTrace(
            step=high_step,
            action_id=action,
            action_name=action_name(action),
            reward=float(reward),
            low_level_steps=int(info.get("low_level_steps", 0)),
            low_level_actions=low_names,
            valid_actions=valid_actions,
            remaining_boxes=int(info.get("remaining_boxes", 0)),
            remaining_bombs=int(info.get("remaining_bombs", 0)),
            dead_boxes=int(info.get("dead_boxes", 0)),
            no_progress_streak=int(info.get("no_progress_streak", 0)),
            oscillation_streak=int(info.get("oscillation_streak", 0)),
            state_revisit_count=int(info.get("state_revisit_count", 0)),
            won=bool(info.get("won", False)),
            terminated=bool(terminated),
            truncated=bool(truncated),
            truncated_reason=str(info.get("truncated_reason", "")),
        )
        traces.append(step_trace)

        if low_segment:
            for low_idx, (low_action, low_state) in enumerate(low_segment, start=1):
                frames.append(
                    render_state(
                        renderer,
                        low_state,
                        scale,
                        [
                            f"{basename} | seed={chosen_seed}",
                            f"high_step={high_step} {step_trace.action_name}",
                            (
                                f"low={low_idx}/{len(low_segment)} "
                                f"{LOW_LEVEL_ACTION_NAMES.get(low_action, low_action)}"
                            ),
                            (
                                f"boxes={len(low_state.boxes)} "
                                f"dead={step_trace.dead_boxes} "
                                f"noprog={step_trace.no_progress_streak}"
                            ),
                        ],
                    )
                )
        else:
            frames.append(
                render_state(
                    renderer,
                    copy.deepcopy(env.engine.get_state()),
                    scale,
                    [
                        f"{basename} | seed={chosen_seed}",
                        f"high_step={high_step} {step_trace.action_name}",
                        "low=0",
                        (
                            f"boxes={step_trace.remaining_boxes} "
                            f"dead={step_trace.dead_boxes} "
                            f"noprog={step_trace.no_progress_streak}"
                        ),
                    ],
                )
            )

        done = bool(terminated or truncated)
        high_step += 1

    summary = {
        "map": map_path,
        "seed": chosen_seed,
        "won": bool(final_info.get("won", False)),
        "remaining_boxes": int(final_info.get("remaining_boxes", 0)),
        "remaining_bombs": int(final_info.get("remaining_bombs", 0)),
        "high_steps": len(traces),
        "total_low_steps": int(final_info.get("total_low_steps", 0)),
        "dead_boxes": int(final_info.get("dead_boxes", 0)),
        "truncated_reason": str(final_info.get("truncated_reason", "")),
        "trace": [asdict(step) for step in traces],
        "frames": len(frames),
    }

    masked.close()
    return {"summary": summary, "frames": frames}


def write_report(output_dir: str, model_path: str, phase: int, rows: List[Dict[str, Any]]) -> str:
    report_path = os.path.join(output_dir, "report.md")
    lines = [
        f"# Phase {phase} Policy Visualization",
        "",
        f"- Model: `{model_path}`",
        f"- Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "| Map | Result | High Steps | Low Steps | Truncation | GIF | Trace |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        result = "WIN" if row["won"] else "FAIL"
        trunc = row["truncated_reason"] or "-"
        gif_name = os.path.basename(row["gif_path"])
        trace_name = os.path.basename(row["trace_path"])
        lines.append(
            f"| {row['map_name']} | {result} | {row['high_steps']} | "
            f"{row['total_low_steps']} | {trunc} | [{gif_name}]({gif_name}) | "
            f"[{trace_name}]({trace_name}) |"
        )
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize RL policy on curriculum maps.")
    parser.add_argument("--phase", type=int, default=3, help="Curriculum phase to replay.")
    parser.add_argument("--model", type=str, default=None, help="Path to model zip.")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for all maps.")
    parser.add_argument("--fps", type=int, default=8, help="GIF playback FPS.")
    parser.add_argument("--scale", type=float, default=1.0, help="Frame scale, e.g. 0.75.")
    parser.add_argument("--limit", type=int, default=None, help="Only export first N maps.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to write outputs.")
    args = parser.parse_args()

    if args.phase not in CURRICULUM:
        print(f"Invalid phase: {args.phase}")
        return 1

    model_path = args.model or model_default_path(args.phase)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return 1

    maps = get_phase_maps(args.phase)
    if args.limit:
        maps = maps[: args.limit]
    if not maps:
        print(f"No maps found for phase {args.phase}")
        return 1

    run_name = (
        f"phase{args.phase}_{os.path.splitext(os.path.basename(model_path))[0]}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or os.path.join(RUNS_ROOT, "policy_preview", run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = MaskablePPO.load(model_path, device="cpu")
    include_map_layout = detect_include_map_layout(model)
    seed_manifest = load_seed_manifest()

    cfg = GameConfig()
    cfg.render_mode = "simple"
    cfg.control_mode = "discrete"
    renderer = Renderer(cfg, str(PROJECT_ROOT))

    rows: List[Dict[str, Any]] = []
    duration_ms = max(40, int(round(1000 / max(args.fps, 1))))
    max_steps = CURRICULUM[args.phase]["max_steps"]

    for idx, map_path in enumerate(maps, start=1):
        map_name = os.path.basename(map_path)
        print(f"[{idx}/{len(maps)}] {map_name}")
        result = run_one_map(
            model=model,
            map_path=map_path,
            max_steps=max_steps,
            include_map_layout=include_map_layout,
            seed_manifest=seed_manifest,
            seed=args.seed,
            renderer=renderer,
            scale=args.scale,
        )
        summary = result["summary"]
        frames = gif_safe_frames(result["frames"])

        stem = os.path.splitext(map_name)[0]
        gif_path = os.path.join(output_dir, f"{stem}.gif")
        trace_path = os.path.join(output_dir, f"{stem}.json")

        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        with open(trace_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

        rows.append(
            {
                "map_name": map_name,
                "won": summary["won"],
                "high_steps": summary["high_steps"],
                "total_low_steps": summary["total_low_steps"],
                "truncated_reason": summary["truncated_reason"],
                "gif_path": gif_path,
                "trace_path": trace_path,
            }
        )
        status = "WIN" if summary["won"] else f"FAIL:{summary['truncated_reason'] or 'unknown'}"
        print(
            f"  {status} | high={summary['high_steps']} "
            f"| low={summary['total_low_steps']} | frames={summary['frames']}"
        )

    report_path = write_report(output_dir, model_path, args.phase, rows)
    renderer.close()

    wins = sum(1 for row in rows if row["won"])
    print(f"\nDone. {wins}/{len(rows)} maps won.")
    print(f"Output: {output_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
