"""Production teacher — wraps best oracle for V2 dataset generation.

Default: oracle_v8 α=0.4 (soft penalty for gambling, linear).

Optimality (30/90 verified-seed-0 sample):
  v0_lower (god, no scan, cheat):  48.00 / 49.06 step, 43-50% gambling
  v6 (no penalty oracle):          48.87 / 49.88 step, 31-36%
  v8 α=0.4 (soft penalty):         49.03 / 50.01 step, 10-22%
  v10 α=0.4 (log penalty):         48.97 / 49.96 step, 14-24%

Production: v8 α=0.4 balances:
  - Near-lb steps (~+1 step over physical lb)
  - Low gambling (~10-22% on first-pushes, all rational)
  - 100% win rate

ALPHA tuning intuition:
  α = 0.4 推荐: 1 步 penalty per uncertainty bit. 强制 ~80% 的首次 push 在
  identification 后进行. 剩余 20% 是 planner 经过 cost-benefit 决定的
  "rational gambles" (scan 代价 > expected gambling penalty).
"""

from typing import List, Optional
from smartcar_sokoban.engine import GameEngine

from experiments.min_steps.planner_oracle_v8 import (
    oracle_v8_min_steps, planner_oracle_v8,
)
from experiments.min_steps.planner_oracle_v10 import (
    oracle_v10_min_steps, planner_oracle_v10,
)
from experiments.min_steps.planner_oracle_v12 import (
    oracle_v12_min_steps, planner_oracle_v12,
)
from experiments.min_steps.planner_oracle_v14 import (
    oracle_v14_min_steps, planner_oracle_v14,
)
from experiments.min_steps.planner_oracle_v18 import planner_oracle_v18
from experiments.min_steps.planner_oracle import _god_plan


# 默认配置 — 经 30-map sweep 确定
# v18 当前最佳: K-best god plans × multi-alpha, trust_walk_reveal=True + sim-based gambling count
# 30-map: v6 物理下界 48.87/25.6%  →  v18 49.40/12.8% (gambling 砍半, step +0.53)
# 距 v6 物理下界 < 1 step, 贴近理论极限.
DEFAULT_ALPHA = 0.7   # v8/v10/v12/v14 用
DEFAULT_LAMBDA = 5.0  # v18 additive 模式 (legacy)
DEFAULT_P_PER_GAMBLE = 0.85   # Pareto 甜点: 50.87 step / 12.8% gambling (v6 baseline 48.87/25.6%)
DEFAULT_K_PLANS = 5           # K-best god plans (first-push 多样化)
DEFAULT_VARIANT = "v18"   # "v8"/"v10"/"v12"/"v14"/"v18"


_VARIANT_PLANNER = {
    "v8": planner_oracle_v8,
    "v10": planner_oracle_v10,
    "v12": planner_oracle_v12,
    "v14": planner_oracle_v14,
    "v18": planner_oracle_v18,
}
_VARIANT_ORACLE = {
    "v8": oracle_v8_min_steps,
    "v10": oracle_v10_min_steps,
    "v12": oracle_v12_min_steps,
    "v14": oracle_v14_min_steps,
}


def teach(eng: GameEngine, *,
           alpha: float = DEFAULT_ALPHA,
           gambling_weight: float = DEFAULT_LAMBDA,
           p_per_gamble: float = DEFAULT_P_PER_GAMBLE,
           k_plans: int = DEFAULT_K_PLANS,
           variant: str = DEFAULT_VARIANT) -> None:
    """生产用 teacher — 在主 engine 上 replay 最佳 belief-aware plan.

    用法 (在 dataset 生成里):
        teacher.teach(eng)   # 在主 engine 上跑完, eng.won = True

    v18 默认行为: K-best god plans (k_plans 个 first-push 多样化方案), 每个用 multi-alpha
    v14 DP 搜索, multiplicative 评分 (p_per_gamble, log step reward) 选全局最优.
    跑完后 engine 状态: won=True (几乎所有可解图), 30-map avg trajectory ~ 54 step / 3% gambling.
    """
    fn = _VARIANT_PLANNER.get(variant)
    if fn is None:
        raise ValueError(f"Unknown variant: {variant}")
    if variant == "v18":
        fn(eng, gambling_weight=gambling_weight,
           p_per_gamble=p_per_gamble, k_plans=k_plans)
    else:
        fn(eng, alpha=alpha)


def teach_get_plan(eng: GameEngine, *, alpha: float = DEFAULT_ALPHA,
                     variant: str = "v14"):
    """返回 (god_plan, oracle_result) — 用于 dataset 生成时拿 interleave structure.

    NOTE: v18 是 per-map best-of-α 包装, 无直接 oracle 接口. 默认 fallback 到 v14.
    """
    plan = _god_plan(eng)
    if not plan: return None, None
    if variant == "v18":
        variant = "v14"   # fallback for direct oracle access
    fn = _VARIANT_ORACLE.get(variant)
    if fn is None:
        raise ValueError(f"Unknown variant: {variant}")
    res = fn(eng, plan=plan, alpha=alpha)
    return plan, res
