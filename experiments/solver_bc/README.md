# Solver BC Prototype

This folder contains an isolated prototype for a solver-driven imitation
learning pipeline.

Scope of this prototype:

- Use `MultiBoxSolver` as an expert.
- Export high-level push actions aligned with `SokobanHLEnv`'s 42-way action
  space.
- Train a small masked behavior cloning model with PyTorch.
- Evaluate the cloned policy with masked greedy rollout.

Important limitation:

- This prototype uses a fully observable "oracle" feature encoder for
  box/target IDs. It is meant to validate the imitation pipeline first.
- It does not replace the current RL observation yet.

Typical workflow:

```powershell
python experiments/solver_bc/build_dataset.py --phase 1 --output .agent/solver_bc/phase1_demo.npz
python experiments/solver_bc/train_bc.py --dataset .agent/solver_bc/phase1_demo.npz --output-dir .agent/solver_bc/phase1_demo_run --device auto
python experiments/solver_bc/evaluate_bc.py --checkpoint .agent/solver_bc/phase1_demo_run/best.pt --phase 1 --device auto --rollout-batch-size 64
```

Performance note:

- `train_bc.py` now defaults to `--device auto`, so it will use CUDA when available.
- `evaluate_bc.py`, `branch_search.py`, and `self_improve_loop.py` batch policy
  inference across many active rollouts with `--rollout-batch-size`, which is the
  main speed lever for this pipeline.

Low-confidence top-2 branching:

```powershell
python experiments/solver_bc/branch_search.py --checkpoint .agent/solver_bc/phase6_full3_run_tuned/best.pt --phase 6 --map-filter phase6_11 --seeds-per-map 3 --device auto --branch-budget 128 --branches-per-rollout 8 --rollout-batch-size 32 --output-json .agent/solver_bc/phase6_11_branch_search.json --output-npz .agent/solver_bc/phase6_11_branch_improved.npz
```

Self-improvement loop:

```powershell
python experiments/solver_bc/self_improve_loop.py --checkpoint .agent/solver_bc/phase6_full3_run_tuned/best.pt --dataset .agent/solver_bc/phase6_full3_hybrid.npz --phase 6 --output-dir .agent/solver_bc/phase6_self_improve_run --iterations 1 --top-maps 1 --seeds-per-map 3 --device auto --rollout-batch-size 32 --branch-budget 128 --branches-per-rollout 8 --epochs 3000 --batch-size 64 --lr 0.0005 --hidden-dim 2048
```
