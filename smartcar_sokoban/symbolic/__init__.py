"""SAGE-PR 符号层 — Belief State / 领域特征 / 候选生成器.

参考:
    docs/FINAL_ARCH_DESIGN.md §2-3
"""

from smartcar_sokoban.symbolic.belief import (
    BeliefBox, BeliefTarget, BeliefBomb, BeliefState,
    infer_remaining_ids,
)

__all__ = [
    "BeliefBox", "BeliefTarget", "BeliefBomb", "BeliefState",
    "infer_remaining_ids",
]
