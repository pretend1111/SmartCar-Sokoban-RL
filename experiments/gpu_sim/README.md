# GPU High-Level Env Prototype

This folder contains a staged GPU port of the high-level environment.

Current scope:
- full high-level action space (`explore + push`)
- batched GPU reachability
- batched push action masks
- batched one-step transition
- bomb explosion and box-target pairing
- CPU-parity explore execution and visibility tracking
- failed-push history heuristic
- oracle observation export
- partial-observation state export
- direct GPU reset from `map + seeds`
- pure GPU greedy rollout for BC checkpoints
- rollout support for both `oracle` and partial `state` observations
- static geometry template cache for neighbors / blast stencils
- dead-box / oscillation / no-progress truncation parity in GPU rollout
- direct rollout backend integration for `branch_search.py`, `evaluate_bc.py`,
  and `self_improve_loop.py`
- unified mixed-map / mixed-seed search waves through `search_requests()`
- cached initial GPU state templates for repeated `map + seed` rollouts

Not ported yet:
- full reward shaping
- fully batched branch-search scheduler across large frontier batches

Validation:
- `compare_gpu_push_env.py` runs CPU and GPU envs in lockstep
- it checks the full CPU/GPU action mask parity
- it compares the full high-level state transition after the same CPU-chosen actions

Example:

```bash
python -m experiments.gpu_sim.compare_gpu_push_env \
  --map maps/phase6/phase6_11.txt \
  --seeds 7 9980 19953 \
  --steps 20 \
  --device auto
```

Pure GPU rollout example:

```bash
python experiments/gpu_sim/rollout_bc_gpu.py \
  --checkpoint .agent/solver_bc/phase6_full3_run_tuned/best.pt \
  --map maps/phase6/phase6_11.txt \
  --seeds 7 9980 19953 \
  --max-steps 100 \
  --device auto
```

CPU vs GPU rollout benchmark:

```bash
python experiments/gpu_sim/benchmark_gpu_rollout.py \
  --checkpoint .agent/solver_bc/phase6_full3_run_tuned/best.pt \
  --phase 6 \
  --map maps/phase6/phase6_11.txt \
  --num-rollouts 512 \
  --device auto
```

Branch search can now switch rollout backends directly:

```bash
python experiments/solver_bc/branch_search.py \
  --checkpoint .agent/solver_bc/phase6_full3_run_tuned/best.pt \
  --phase 6 \
  --map-filter phase6_11 \
  --seeds-per-map 12 \
  --branch-budget 64 \
  --branches-per-rollout 8 \
  --frontier-batch-size 16 \
  --rollout-backend gpu \
  --device auto
```

Note:
- GPU rollout is behavior-identical to the CPU rollout on the validated test
  cases once `frontier_batch_size` is held fixed
- raw rollout is already faster at larger batch sizes
- for branch search, `frontier_batch_size=16` is currently the most useful
  starting point on this repo
- branch search still needs larger cross-episode / cross-frontier batches before
  GPU becomes consistently faster than the CPU scheduler
