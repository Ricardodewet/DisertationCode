"""Integration test for end‑to‑end pipeline (short run)."""

from __future__ import annotations

import os
import shutil

from ..src.utils.config import load_config
from ..scripts.train import train_single_run
from ..scripts.baselines import run_baseline
from ..src.evaluation.metrics import summarise_runs


def test_end_to_end(tmp_path):
    # Create a temporary config with very few episodes for speed
    config = {
        'environment': {'max_steps': 3},
        'training': {
            'episodes': 2,
            'seeds': [0],
            'learning_rate': 0.01,
            'gamma': 0.99,
            'lam': 0.95,
            'eps_clip': 0.2,
            'value_coef': 0.5,
            'epochs': 1,
            'batch_size': 8,
        },
    }
    run_dir = tmp_path / 'runs'
    os.makedirs(run_dir, exist_ok=True)
    # Train PPO
    train_single_run(config, seed=0, run_dir=str(run_dir))
    # Run one baseline (random) for speed
    run_baseline('random', None, config, seed=0, run_dir=str(run_dir))
    # There should be two CSV files
    files = os.listdir(run_dir)
    assert any('ppo' in f for f in files)
    assert any('random' in f for f in files)
    # Summarise runs without error
    summary = summarise_runs(str(run_dir))
    assert not summary.empty
