"""Entry point for training the PPO agent.

This script reads a YAML configuration specifying environment and training
parameters, runs multiple seeds of the PPO algorithm, logs per‑step data to
CSV files and writes them to ``outputs/runs``.  A copy of the config is
stored alongside the logs for reproducibility.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..src.utils.config import load_config
from ..src.utils.seeding import set_seed
from ..src.envs.learner_env import LearnerEnv
from ..src.agents.ppo_agent import PPOAgent


def train_single_run(config: Dict[str, Any], seed: int, run_dir: str) -> None:
    """Train the PPO agent for one seed and save logs.

    Parameters
    ----------
    config:
        The loaded configuration dictionary.
    seed:
        Random seed for environment and agent.
    run_dir:
        Directory where the run CSV will be saved.
    """
    env_cfg = config.get('environment', {})
    train_cfg = config.get('training', {})
    # Set seeds
    set_seed(seed)
    # Initialise environment
    env = LearnerEnv(max_steps=env_cfg.get('max_steps', 20), seed=seed)
    state_dim = len(env.reset())
    action_dim = len(env.ACTIONS)
    # Instantiate PPO agent with hyperparameters
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=train_cfg.get('learning_rate', 0.01),
        gamma=train_cfg.get('gamma', 0.99),
        lam=train_cfg.get('lam', 0.95),
        eps_clip=train_cfg.get('eps_clip', 0.2),
        value_coef=train_cfg.get('value_coef', 0.5),
        epochs=train_cfg.get('epochs', 3),
        batch_size=train_cfg.get('batch_size', 64),
    )
    num_episodes = train_cfg.get('episodes', 200)
    # Storage for per‑step logs
    log_rows: List[Dict[str, Any]] = []
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        # Per‑episode memory for PPO
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        step = 0
        while not done:
            action, log_prob, value = agent.act(state)
            next_state, reward, done, info = env.step(action)
            # Record memory
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(float(done))
            # Log row
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
                'method': 'ppo',
            })
            state = next_state
            step += 1
        # Convert lists to arrays
        states_arr = np.array(states)
        actions_arr = np.array(actions)
        log_probs_arr = np.array(log_probs)
        values_arr = np.array(values)
        rewards_arr = np.array(rewards)
        dones_arr = np.array(dones)
        # Compute returns and advantages
        returns, advantages = agent.compute_returns_and_advantages(
            rewards_arr, values_arr, dones_arr
        )
        # Update agent
        agent.update(states_arr, actions_arr, log_probs_arr, returns, advantages)
    # Save logs to CSV
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f'run_ppo_seed{seed}.csv')
    pd.DataFrame(log_rows).to_csv(out_path, index=False)


def main(args: List[str]) -> None:
    parser = argparse.ArgumentParser(description='Train PPO agent for adaptive assessment')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--out', type=str, default='outputs/runs', help='Directory for run logs')
    parsed = parser.parse_args(args)
    config = load_config(parsed.config)
    seeds = config.get('training', {}).get('seeds', [0])
    # Copy config to output directory for reproducibility
    os.makedirs(parsed.out, exist_ok=True)
    config_copy_path = os.path.join(parsed.out, 'config_used.yaml')
    import yaml
    with open(config_copy_path, 'w') as f:
        yaml.safe_dump(config, f)
    # Train for each seed
    for seed in seeds:
        print(f'Training PPO with seed {seed}')
        train_single_run(config, seed, parsed.out)
        print(f'Completed seed {seed}')


if __name__ == '__main__':
    main(sys.argv[1:])