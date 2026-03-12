"""RL 训练脚本 v2 — 单步推箱 + 效率奖励 + 弱图重训.

用法:
    python -m rl.train                      # 从 Phase 1 开始
    python -m rl.train --phase 4            # 从 Phase 4 开始
    python -m rl.train --resume model.zip   # 继续训练
    python -m rl.train --eval model.zip     # 评估

改进:
    1. 预计算 AutoPlayer 基线步数
    2. 弱图重训 (30% 概率训练表现最差的地图)
    3. 更大训练量
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import json
import numpy as np

from smartcar_sokoban.rl.high_level_env import SokobanHLEnv, STATE_DIM_WITH_MAP
from smartcar_sokoban.paths import MAPS_ROOT, PROJECT_ROOT, RUNS_ROOT


# ── 种子清单 ──────────────────────────────────────────

def load_seed_manifest() -> Dict[str, List[int]]:
    """加载种子清单, 返回 {文件名: [可用种子列表]}."""
    manifest_path = MAPS_ROOT / "phase456_seed_manifest.json"
    if not os.path.exists(manifest_path):
        return {}

    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {}
    for phase_name, phase_maps in data.get('phases', {}).items():
        for map_name, map_info in phase_maps.items():
            seeds = map_info.get('verified_seeds', [])
            if seeds:
                result[map_name] = seeds
    return result

# ── AutoPlayer 基线计算 ───────────────────────────────────

def compute_baseline(map_path: str, seed_manifest: Dict[str, List[int]],
                     n_seeds: int = 5) -> int:
    """用 AutoPlayer 计算地图的平均步数 (静默模式)."""
    from smartcar_sokoban.config import GameConfig
    from smartcar_sokoban.engine import GameEngine
    from smartcar_sokoban.solver.auto_player import AutoPlayer

    cfg = GameConfig()
    cfg.control_mode = "discrete"
    engine = GameEngine(cfg, str(PROJECT_ROOT))

    basename = os.path.basename(map_path)
    verified = seed_manifest.get(basename, [])
    seeds_to_try = verified[:n_seeds] if verified else [i * 7 + 42 for i in range(n_seeds)]

    steps_list = []
    # 静默 AutoPlayer 的 print 输出
    devnull = open(os.devnull, 'w')
    for seed in seeds_to_try:
        try:
            random.seed(seed)
            engine.reset(map_path)
            player = AutoPlayer(engine)
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                actions = player.solve()
            finally:
                sys.stdout = old_stdout
            if engine.get_state().won:
                steps_list.append(len(actions))
        except Exception:
            pass
    devnull.close()

    return int(np.mean(steps_list)) if steps_list else 100


def compute_all_baselines(map_pool: List[str],
                          seed_manifest: Dict[str, List[int]]
                          ) -> Dict[str, int]:
    """预计算所有地图的 AutoPlayer 基线."""
    baselines = {}
    total = len(map_pool)
    for idx, mp in enumerate(map_pool):
        bl = compute_baseline(mp, seed_manifest)
        baselines[mp] = bl
        # 单行进度
        print(f"\r    基线计算: [{idx+1}/{total}] {os.path.basename(mp)} = {bl}步",
              end='', flush=True)
    print()  # 换行
    return baselines


# ── 课程配置 ──────────────────────────────────────────────

CURRICULUM = {
    1: {
        'name': 'Phase 1: 1箱 空旷',
        'map_dir': 'assets/maps/phase1',
        'total_timesteps': 1_500_000,
        'max_steps': 30,
    },
    2: {
        'name': 'Phase 2: 1箱 有墙',
        'map_dir': 'assets/maps/phase2',
        'total_timesteps': 2_500_000,
        'max_steps': 40,
    },
    3: {
        'name': 'Phase 3: 2箱',
        'map_dir': 'assets/maps/phase3',
        'total_timesteps': 50_000_000,
        'max_steps': 60,
    },
    4: {
        'name': 'Phase 4: 3箱',
        'map_dir': 'assets/maps/phase4',
        'total_timesteps': 20_000_000,
        'max_steps': 80,
    },
    5: {
        'name': 'Phase 5: 3箱 + TNT',
        'map_dir': 'assets/maps/phase5',
        'total_timesteps': 30_000_000,
        'max_steps': 100,
    },
    6: {
        'name': 'Phase 6: 混合',
        'map_dir': 'assets/maps/phase6',
        'total_timesteps': 35_000_000,
        'max_steps': 100,
    },
}


def get_map_pool(phase_cfg: dict) -> List[str]:
    """获取某阶段的地图池."""
    map_dir = os.path.join(PROJECT_ROOT, phase_cfg['map_dir'])
    if not os.path.isdir(map_dir):
        return []
    maps = sorted(glob.glob(os.path.join(map_dir, '*.txt')))
    return [os.path.relpath(m, PROJECT_ROOT).replace('\\', '/') for m in maps]


# ── 带弱图重训的环境 ─────────────────────────────────────

class WeightedMapEnv(SokobanHLEnv):
    """带弱图重训权重的环境.

    70% 概率随机选图, 30% 概率选表现最差的图.
    """

    def __init__(self, map_pool, baselines, seed_manifest=None,
                 weak_maps=None, base_dir="", max_steps=60,
                 include_map_layout=False):
        avg_baseline = int(np.mean(list(baselines.values()))) if baselines else 100
        super().__init__(
            map_pool=map_pool,
            base_dir=base_dir,
            max_steps=max_steps,
            baseline_steps=avg_baseline,
            seed_manifest=seed_manifest or {},
            include_map_layout=include_map_layout,
        )
        self.baselines = baselines
        self.weak_maps = weak_maps or []
        self._all_maps = map_pool

    def _pick_map(self) -> str:
        if self.weak_maps and random.random() < 0.3:
            chosen = random.choice(self.weak_maps)
        else:
            chosen = random.choice(self._all_maps)
        # 更新 baseline_steps 为当前选择地图的基线
        self.baseline_steps = self.baselines.get(chosen, 100)
        return chosen


# ── 训练 ──────────────────────────────────────────────────

# ── 进度回调 ──────────────────────────────────────────────

class ProgressCallback:
    """显示训练进度: 速度、通关率、平均步数."""

    def __init__(self, total_timesteps, phase_name, log_interval=5000):
        from stable_baselines3.common.callbacks import BaseCallback

        self.total = total_timesteps
        self.phase_name = phase_name
        self.log_interval = log_interval
        self._last_log = 0
        self._t0 = time.time()
        self._wins = 0
        self._episodes = 0
        self._step_sums = 0

        class _CB(BaseCallback):
            def __init__(cb_self, progress_ref):
                super().__init__(verbose=0)
                cb_self._progress = progress_ref

            def _on_step(cb_self):
                # 从 info 中收集统计
                infos = cb_self.locals.get('infos', [])
                for info in infos:
                    if 'won' in info and ('terminal_observation' in info
                                          or info.get('won', False)
                                          or info.get('TimeLimit.truncated', False)):
                        cb_self._progress._episodes += 1
                        if info.get('won', False):
                            cb_self._progress._wins += 1
                            cb_self._progress._step_sums += info.get(
                                'total_low_steps', 0)

                n = cb_self.num_timesteps
                if n - cb_self._progress._last_log >= cb_self._progress.log_interval:
                    cb_self._progress._print_progress(n)
                    cb_self._progress._last_log = n
                return True

        self.callback = _CB(self)

    def _print_progress(self, n):
        elapsed = time.time() - self._t0
        fps = n / max(elapsed, 0.01)
        pct = min(n / max(self.total, 1) * 100, 100.0)

        wr = (self._wins / self._episodes * 100) if self._episodes > 0 else 0
        avg_s = (self._step_sums / self._wins) if self._wins > 0 else 0

        eta = max(0, (self.total - n) / max(fps, 1))
        eta_min = int(eta // 60)
        eta_sec = int(eta % 60)

        # 进度条 (20字符宽)
        bar_len = 20
        filled = int(bar_len * min(pct, 100) / 100)
        bar = '█' * filled + '░' * (bar_len - filled)

        line = (f"  {bar} {pct:5.1f}% | "
                f"FPS:{fps:5.0f} | "
                f"Win:{wr:4.0f}% ({self._wins}/{self._episodes}) | "
                f"Steps:{avg_s:4.0f} | "
                f"ETA:{eta_min}m{eta_sec:02d}s")
        print(f"\r{line}", end='', flush=True)


class PeriodicEvalCallback:
    """Run lightweight deterministic evals during training and save the best."""

    def __init__(self, phase: int, total_timesteps: int, model_dir: str,
                 map_pool: List[str], baselines: Dict[str, int],
                 seed_manifest: Dict[str, List[int]], max_steps: int,
                 include_map_layout: bool, n_seeds: int = 3):
        from stable_baselines3.common.callbacks import BaseCallback

        self.phase = phase
        self.map_pool = list(map_pool)
        self.baselines = dict(baselines)
        self.seed_manifest = dict(seed_manifest)
        self.max_steps = max_steps
        self.include_map_layout = include_map_layout
        self.n_seeds = n_seeds
        self.eval_interval = max(50_000, total_timesteps // 6)
        self._last_eval = 0
        self.best_path = os.path.join(model_dir, f'phase{phase}_best.zip')
        self.saved_any = False
        self.best_win_rate = -1.0
        self.best_solved_maps = -1
        self.best_avg_steps = float('inf')

        class _CB(BaseCallback):
            def __init__(cb_self, eval_ref):
                super().__init__(verbose=0)
                cb_self._eval_ref = eval_ref

            def _on_step(cb_self):
                n = cb_self.num_timesteps
                if n - cb_self._eval_ref._last_eval < cb_self._eval_ref.eval_interval:
                    return True

                cb_self._eval_ref._last_eval = n
                print()
                results = evaluate_per_map(
                    cb_self.model,
                    cb_self._eval_ref.map_pool,
                    cb_self._eval_ref.baselines,
                    cb_self._eval_ref.seed_manifest,
                    n_seeds=cb_self._eval_ref.n_seeds,
                    max_steps=cb_self._eval_ref.max_steps,
                    include_map_layout=cb_self._eval_ref.include_map_layout,
                )
                avg_wr = float(np.mean([r['win_rate'] for r in results.values()]))
                solved_maps = sum(1 for r in results.values() if r['win_rate'] > 0)
                win_steps = [r['avg_steps'] for r in results.values()
                             if r['avg_steps'] > 0]
                avg_steps = (float(np.mean(win_steps))
                             if win_steps else float('inf'))

                print(
                    f"  Eval@{n:,}: win={avg_wr:.0%} | "
                    f"solved={solved_maps}/{len(results)} | "
                    f"steps={(0 if not win_steps else avg_steps):.0f}"
                )

                better = (
                    avg_wr > cb_self._eval_ref.best_win_rate or
                    (avg_wr == cb_self._eval_ref.best_win_rate and
                     solved_maps > cb_self._eval_ref.best_solved_maps) or
                    (avg_wr == cb_self._eval_ref.best_win_rate and
                     solved_maps == cb_self._eval_ref.best_solved_maps and
                     avg_steps < cb_self._eval_ref.best_avg_steps)
                )
                if better:
                    cb_self.model.save(cb_self._eval_ref.best_path)
                    cb_self._eval_ref.saved_any = True
                    cb_self._eval_ref.best_win_rate = avg_wr
                    cb_self._eval_ref.best_solved_maps = solved_maps
                    cb_self._eval_ref.best_avg_steps = avg_steps
                    print(f"  Saved best: {cb_self._eval_ref.best_path}")
                return True

        self.callback = _CB(self)


# ── 训练 ──────────────────────────────────────────────────

def train(start_phase: int = 1, resume_path: Optional[str] = None,
          end_phase: Optional[int] = None):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        from stable_baselines3.common.callbacks import (
            CheckpointCallback, CallbackList,
        )
        import torch
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("pip install sb3-contrib tensorboard")
        return

    safe_base = RUNS_ROOT / "rl"
    log_dir = safe_base / "logs"
    model_dir = safe_base / "models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print(f"  日志: {log_dir}")
    print(f"  模型: {model_dir}")

    model = None

    # 加载种子清单
    seed_manifest = load_seed_manifest()
    print(f"  种子清单: {len(seed_manifest)} 张地图有限制")

    last_phase = end_phase if end_phase is not None else max(CURRICULUM)
    last_phase = min(last_phase, max(CURRICULUM))
    if last_phase < start_phase:
        print(f"  end_phase={last_phase} < start_phase={start_phase}, nothing to do.")
        return

    for phase in range(start_phase, last_phase + 1):
        cfg = CURRICULUM[phase]
        print(f"\n{'='*60}")
        print(f"  {cfg['name']}")
        print(f"{'='*60}")

        map_pool = get_map_pool(cfg)
        if not map_pool:
            print(f"  ⚠️ {cfg['map_dir']} 无地图, 跳过")
            continue
        print(f"  地图数: {len(map_pool)}")

        # 计算 AutoPlayer 基线
        print("  预计算 AutoPlayer 基线...")
        baselines = compute_all_baselines(map_pool, seed_manifest)
        avg_bl = int(np.mean(list(baselines.values())))
        print(f"  平均基线: {avg_bl} 步")

        # ── 环境工厂 (SubprocVecEnv 需要可 pickle 的工厂) ──
        # 用闭包捕获当前 phase 的变量
        _mp = list(map_pool)
        _bl = dict(baselines)
        _sm = dict(seed_manifest)
        _ms = cfg['max_steps']

        def _make_env_fn(rank):
            """返回一个环境工厂函数 (闭包)."""
            def _init():
                env = WeightedMapEnv(
                    map_pool=_mp,
                    baselines=_bl,
                    seed_manifest=_sm,
                    base_dir=str(PROJECT_ROOT),
                    max_steps=_ms,
                    include_map_layout=True,
                )
                return ActionMasker(env, lambda e: e.action_masks())
            return _init

        # 并行环境: 尽量多开, BFS 是 CPU 密集型
        n_envs = min(16, max(os.cpu_count() or 4, 8))
        print(f"  并行环境: {n_envs}")

        try:
            vec_env = SubprocVecEnv(
                [_make_env_fn(i) for i in range(n_envs)],
                start_method='spawn',
            )
            print("  ✅ SubprocVecEnv (真并行)")
        except Exception as e:
            print(f"  ⚠️ SubprocVecEnv 失败 ({e}), 用 DummyVecEnv")
            vec_env = DummyVecEnv([_make_env_fn(i) for i in range(n_envs)])

        # 创建/加载模型
        # 网络很小 (~33KB), 用 CPU 更快 (省去 GPU 数据传输)
        device = 'cpu'
        if model is None:
            if resume_path:
                print(f"  加载模型: {resume_path}")
                loaded_model = MaskablePPO.load(resume_path, device=device)
                obs_shape = getattr(getattr(loaded_model, 'observation_space', None),
                                    'shape', None)
                if obs_shape:
                    obs_dim = int(np.prod(obs_shape))
                    if obs_dim != STATE_DIM_WITH_MAP:
                        vec_env.close()
                        print("  ❌ 旧模型的 observation 维度与当前训练配置不兼容。")
                        print("     当前训练默认启用地图布局观测，请从头训练新模型。")
                        return
                loaded_model.set_env(vec_env)
                model = loaded_model
            else:
                model = MaskablePPO(
                    "MlpPolicy",
                    vec_env,
                    policy_kwargs={
                        "net_arch": [128, 128, 64],
                        "activation_fn": torch.nn.ReLU,
                    },
                    learning_rate=3e-4,
                    n_steps=1024,       # 更频繁更新
                    batch_size=512,     # 更大 batch
                    n_epochs=8,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.005,
                    verbose=0,          # 关掉默认输出, 用自定义进度
                    tensorboard_log=log_dir,
                    device=device,
                )
        else:
            model.set_env(vec_env)

        # 回调
        progress = ProgressCallback(cfg['total_timesteps'], cfg['name'])
        best_eval = PeriodicEvalCallback(
            phase=phase,
            total_timesteps=cfg['total_timesteps'],
            model_dir=model_dir,
            map_pool=map_pool,
            baselines=baselines,
            seed_manifest=seed_manifest,
            max_steps=cfg['max_steps'],
            include_map_layout=True,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=max(20000 // n_envs, 500),
            save_path=os.path.join(model_dir, f'ckpt_p{phase}'),
            name_prefix=f'p{phase}',
        )
        callbacks = CallbackList([progress.callback, ckpt_cb, best_eval.callback])

        # 训练
        print(f"  开始训练: {cfg['total_timesteps']:,} 步 (device={device})")
        t0 = time.time()
        model.learn(
            total_timesteps=cfg['total_timesteps'],
            callback=callbacks,
            tb_log_name=f'phase{phase}',
            reset_num_timesteps=True,
        )
        elapsed = time.time() - t0
        print()  # 进度条换行
        print(f"  训练完成! 耗时 {elapsed:.0f}s "
              f"({cfg['total_timesteps']/elapsed:.0f} steps/s)")

        if best_eval.saved_any and os.path.exists(best_eval.best_path):
            print(f"  Restoring best checkpoint: {best_eval.best_path}")
            model = MaskablePPO.load(best_eval.best_path, device=device)
            model.set_env(vec_env)

        # 保存
        save_path = os.path.join(model_dir, f'phase{phase}_final.zip')
        model.save(save_path)
        print(f"  已保存: {save_path}")

        # 评估
        print("  评估中...")
        results = evaluate_per_map(
            model,
            map_pool,
            baselines,
            seed_manifest,
            max_steps=cfg['max_steps'],
            include_map_layout=True,
        )
        total_wins = sum(1 for r in results.values() if r['win_rate'] > 0)
        avg_wr = np.mean([r['win_rate'] for r in results.values()])
        avg_steps = np.mean([r['avg_steps'] for r in results.values()
                             if r['avg_steps'] > 0])
        print(f"  通关率: {avg_wr:.0%} | 平均步数: {avg_steps:.0f}")

        for mp, r in results.items():
            bl = baselines.get(mp, 0)
            status = "✅" if r['avg_steps'] < bl else "⚠️"
            print(f"    {status} {os.path.basename(mp)}: "
                  f"win={r['win_rate']:.0%} "
                  f"steps={r['avg_steps']:.0f} "
                  f"(baseline={bl})")

        vec_env.close()

    print("\n✅ 训练完成!")


# ── 评估 ──────────────────────────────────────────────────

def evaluate_per_map(model, map_pool, baselines,
                     seed_manifest=None,
                     n_seeds=10,
                     max_steps=60,
                     include_map_layout=False) -> Dict[str, Dict]:
    """逐地图评估, 返回每张图的通关率和平均步数."""
    from sb3_contrib.common.wrappers import ActionMasker

    seed_manifest = seed_manifest or {}
    results = {}
    for mp in map_pool:
        wins = 0
        steps_list = []

        # 获取合法种子
        basename = os.path.basename(mp)
        verified = seed_manifest.get(basename, [])
        if verified:
            eval_seeds = verified[:n_seeds]
        else:
            eval_seeds = [i * 31 + 7 for i in range(n_seeds)]

        for seed in eval_seeds:
            env_max_steps = (max_steps.get(mp, 60) if isinstance(max_steps, dict)
                             else max_steps)
            env = SokobanHLEnv(
                map_file=mp,
                base_dir=str(PROJECT_ROOT),
                max_steps=env_max_steps,
                baseline_steps=baselines.get(mp, 100),
                seed_manifest=seed_manifest,
                include_map_layout=include_map_layout,
            )
            masked = ActionMasker(env, lambda e: e.action_masks())

            obs, info = masked.reset(seed=seed)
            done = False

            while not done:
                masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=True,
                                          action_masks=masks)
                obs, r, term, trunc, info = masked.step(action)
                done = term or trunc

            if info.get('won', False):
                wins += 1
                steps_list.append(info.get('total_low_steps', 0))

            masked.close()

        results[mp] = {
            'win_rate': wins / len(eval_seeds),
            'avg_steps': np.mean(steps_list) if steps_list else 0,
        }

    return results


# ── CLI ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='推箱子 RL 训练 v2')
    parser.add_argument('--phase', type=int, default=1)
    parser.add_argument('--end-phase', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval', type=str, default=None)
    args = parser.parse_args()

    if args.eval:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(args.eval)
        include_map_layout = False
        obs_shape = getattr(getattr(model, 'observation_space', None),
                            'shape', None)
        if obs_shape:
            obs_dim = int(np.prod(obs_shape))
            include_map_layout = (obs_dim == STATE_DIM_WITH_MAP)

        seed_manifest = load_seed_manifest()
        all_maps = []
        map_max_steps = {}
        for phase in range(1, 7):
            pool = get_map_pool(CURRICULUM[phase])
            all_maps.extend(pool)
            for mp in pool:
                map_max_steps[mp] = CURRICULUM[phase]['max_steps']

        baselines = compute_all_baselines(all_maps, seed_manifest)
        results = evaluate_per_map(model, all_maps, baselines,
                                   seed_manifest, n_seeds=20,
                                   max_steps=map_max_steps,
                                   include_map_layout=include_map_layout)

        print("\n=== 评估结果 ===")
        for mp, r in results.items():
            bl = baselines.get(mp, 0)
            diff = r['avg_steps'] - bl if r['avg_steps'] > 0 else 0
            print(f"  {os.path.basename(mp):20s} "
                  f"win={r['win_rate']:5.0%}  "
                  f"steps={r['avg_steps']:5.0f}  "
                  f"baseline={bl:3d}  "
                  f"diff={diff:+.0f}")
    else:
        train(start_phase=args.phase,
              resume_path=args.resume,
              end_phase=args.end_phase)


if __name__ == '__main__':
    main()
