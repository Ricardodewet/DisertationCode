"""Run baseline policies for adaptive assessment.

This script executes several fixed strategies (random, static, heuristic and
staircase) across multiple seeds and episodes.  The results are logged to
CSV files in ``outputs/runs`` for later analysis.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

import pandas as pd

from src.utils.config import load_config
from src.utils.seeding import set_seed
from src.envs.learner_env import LearnerEnv
from src.agents.baseline_policies import (
    random_policy,
    static_medium_policy,
    heuristic_zpd_policy,
    StaircasePolicy,
)


def run_baseline(method_name: str, policy_callable: Any, config: Dict[str, Any], seed: int, run_dir: str) -> None:
    """Execute a baseline policy for all episodes and save logs."""
    env_cfg = config.get('environment', {})
    train_cfg = config.get('training', {})
    max_steps = env_cfg.get('max_steps', 20)
    num_episodes = train_cfg.get('episodes', 200)
    set_seed(seed)
    env = LearnerEnv(max_steps=max_steps, seed=seed)
    # For staircase maintain policy state per episode
    staircase = None
    if method_name == 'staircase':
        staircase = StaircasePolicy()
    # Logging
    log_rows: List[Dict[str, Any]] = []
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        step = 0
        # Reset staircase at start of episode
        if staircase is not None:
            staircase.current_action = 1
        while not done:
            if method_name == 'random':
                action = random_policy(env)
            elif method_name == 'static':
                action = static_medium_policy(env)
            elif method_name == 'heuristic':
                action = heuristic_zpd_policy(env)
            elif method_name == 'staircase':
                action = staircase(env)
            else:
                raise ValueError(f"Unknown baseline method {method_name}")
            next_state, reward, done, info = env.step(action)
            # Update staircase state based on correctness
            if staircase is not None:
                staircase.update(info['correct'])
            log_rows.append({
                'episode': ep,
                'step': step,
                'action': action,
                'difficulty': info['difficulty'],
                'ability': info['ability'],
                'engagement': info['engagement'],
                'response_time': info['response_time'],
                'correct': info['correct'],
                'reward': reward,
                'prob_correct': info['prob_correct'],
                'seed': seed,
                'method': method_name,
            })
            state = next_state
            step += 1
    # Save log
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f'run_{method_name}_seed{seed}.csv')
    pd.DataFrame(log_rows).to_csv(out_path, index=False)


def main(args: List[str]) -> None:
    parser = argparse.ArgumentParser(description='Run baseline policies for adaptive assessment')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--out', type=str, default='outputs/runs', help='Directory for run logs')
    parsed = parser.parse_args(args)
    config = load_config(parsed.config)
    seeds = config.get('training', {}).get('seeds', [0])
    # Copy config to output directory once
    os.makedirs(parsed.out, exist_ok=True)
    import yaml
    with open(os.path.join(parsed.out, 'config_used.yaml'), 'w') as f:
        yaml.safe_dump(config, f)
    methods = ['random', 'static', 'heuristic', 'staircase']
    for method in methods:
        for seed in seeds:
            print(f'Running {method} baseline with seed {seed}')
            run_baseline(method, None, config, seed, parsed.out)
            print(f'Completed {method} seed {seed}')


if __name__ == '__main__':
    main(sys.argv[1:])
