"""跑 baseline 测量."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.min_steps.harness import (
    benchmark, planner_v0_godmode_lowerbound, planner_v1_explore_first,
    print_table, summary,
)
from experiments.min_steps.planner_v1 import planner_v1_opportunistic
from experiments.min_steps.planner_v2 import planner_v2_walk_first
from experiments.min_steps.planner_v3 import planner_v3_jit_explore
from experiments.min_steps.planner_v4 import planner_v4_detour_aware
from experiments.min_steps.planner_v5 import planner_v5_walk_then_explore
from experiments.min_steps.planner_v6 import planner_v6_tsp_explore
from experiments.min_steps.planner_best import planner_best_of_three, set_best_context
from experiments.min_steps.planner_v7 import planner_v7_dp_interleave
from experiments.min_steps.planner_oracle import planner_oracle
from experiments.min_steps.planner_oracle_v2 import planner_oracle_v2
from experiments.min_steps.planner_oracle_v3b import planner_oracle_v3b
from experiments.min_steps.planner_oracle_v4 import planner_oracle_v4

# 回到 8 张 verified 图 (seed=0 都 god 可解), 跑得快
maps = [
    ("assets/maps/phase4/phase4_0001.txt", 0),
    ("assets/maps/phase4/phase4_0005.txt", 0),
    ("assets/maps/phase5/phase5_0001.txt", 0),
    ("assets/maps/phase5/phase5_0005.txt", 0),
    ("assets/maps/phase5/phase5_0010.txt", 0),
    ("assets/maps/phase5/phase5_0020.txt", 0),
    ("assets/maps/phase6/phase6_0001.txt", 0),
    ("assets/maps/phase6/phase6_0010.txt", 0),
]
maps = [(m, s) for m, s in maps if os.path.exists(m)]

planners = {
    "v0_godmode_lower": planner_v0_godmode_lowerbound,
    "v1_explore_first": planner_v1_explore_first,
    "v1_opportunistic": planner_v1_opportunistic,
    "v2_walk_first": planner_v2_walk_first,
    "v3_jit_explore": planner_v3_jit_explore,
    "v4_detour_aware": planner_v4_detour_aware,
    "v5_walk_then_explore": planner_v5_walk_then_explore,
    "v6_tsp_explore": planner_v6_tsp_explore,
    "v7_dp_interleave": planner_v7_dp_interleave,
    "best_of_three": planner_best_of_three,
    "oracle": planner_oracle,
    "oracle_v2": planner_oracle_v2,
}

from experiments.min_steps.harness import run_planner
results = []
for mp, sd in maps:
    set_best_context(mp, sd)   # for best_of_three
    for name, fn in planners.items():
        results.append(run_planner(mp, sd, name, fn))
print_table(results)
summary(results)
