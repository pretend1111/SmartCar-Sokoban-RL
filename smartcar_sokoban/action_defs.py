"""Shared action definitions used by engine and solver helpers."""

from __future__ import annotations

from typing import Dict, Tuple

Dir = Tuple[int, int]

# Legacy discrete controls used by manual play / wrappers.
LEGACY_TRANSLATION_ACTIONS = frozenset({0, 1, 2, 3})

# Absolute world-space translations reserved for solver execution/replay.
ABS_WORLD_MOVE_TO_ACTION: Dict[Dir, int] = {
    (0, -1): 7,    # up
    (0, 1): 8,     # down
    (-1, 0): 9,    # left
    (1, 0): 10,    # right
    (-1, -1): 11,  # up-left
    (1, -1): 12,   # up-right
    (-1, 1): 13,   # down-left
    (1, 1): 14,    # down-right
}

ACTION_TO_ABS_WORLD_MOVE: Dict[int, Dir] = {
    action: direction for direction, action in ABS_WORLD_MOVE_TO_ACTION.items()
}

ABS_TRANSLATION_ACTIONS = frozenset(ACTION_TO_ABS_WORLD_MOVE)


def direction_to_abs_action(dx: int, dy: int) -> int:
    """Convert a world-space grid direction into a solver move action."""
    return ABS_WORLD_MOVE_TO_ACTION.get((dx, dy), 6)


def is_translation_action(action: int) -> bool:
    """Whether an action is any kind of movement (legacy or absolute)."""
    return action in LEGACY_TRANSLATION_ACTIONS or action in ABS_TRANSLATION_ACTIONS
