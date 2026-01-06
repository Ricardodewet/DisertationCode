"""Metrics computation for RL adaptive assessment experiments.

This module provides functions to load episode logs, compute per‑run
statistics and aggregate results across multiple runs.  It relies on
Pandas for data manipulation and SciPy for inferential statistics in
conjunction with the `stats` module.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import pandas as pd


def load_run(file_path: str) -> pd.DataFrame:
    """Load a single run log CSV into a DataFrame.

    The CSV is expected to contain one row per step with at least the
    following columns: 'episode', 'step', 'action', 'difficulty', 'ability',
    'engagement', 'response_time', 'correct', 'reward', 'seed', 'method'.
    """
    return pd.read_csv(file_path)


def compute_metrics_for_run(df: pd.DataFrame) -> Dict[str, float]:
    """Compute summary metrics for a single run.

    Parameters
    ----------
    df:
        DataFrame containing one run of episode logs.

    Returns
    -------
    dict
        Dictionary with mean reward, accuracy, learning gain, mean
        engagement, mean response time and number of episodes.
    """
    # Compute mean across all steps
    mean_reward = df['reward'].mean()
    accuracy = df['correct'].mean()
    mean_engagement = df['engagement'].mean()
    mean_resp_time = df['response_time'].mean()
    # Learning gain: ability at last step minus ability at first step
    if len(df) > 0:
        ability_start = df.iloc[0]['ability']
        ability_end = df.iloc[-1]['ability']
        learning_gain = ability_end - ability_start
    else:
        learning_gain = 0.0
    # Count episodes – assume episodes numbered from 1
    n_episodes = df['episode'].nunique() if 'episode' in df.columns else 0
    return {
        'mean_reward': float(mean_reward),
        'accuracy': float(accuracy),
        'learning_gain': float(learning_gain),
        'mean_engagement': float(mean_engagement),
        'mean_resp_time': float(mean_resp_time),
        'episodes': int(n_episodes),
    }


def summarise_runs(run_dir: str) -> pd.DataFrame:
    """Summarise all run CSV files in a directory.

    Parameters
    ----------
    run_dir:
        Directory containing run CSV files produced by training or
        baseline scripts.

    Returns
    -------
    DataFrame
        Table with one row per run including method, seed and computed
        metrics.
    """
    rows: List[Dict[str, float]] = []
    for file in os.listdir(run_dir):
        if not file.endswith('.csv'):
            continue
        path = os.path.join(run_dir, file)
        df = load_run(path)
        metrics = compute_metrics_for_run(df)
        # Extract method and seed from filename (assumes pattern)
        match = re.match(r"run_(?P<method>.+)_seed(?P<seed>\d+)\.csv", file)
        method = match.group('method') if match else df.get('method', 'unknown')
        seed = int(match.group('seed')) if match else int(df.get('seed', 0))
        metrics['method'] = method
        metrics['seed'] = seed
        rows.append(metrics)
    return pd.DataFrame(rows)


def extract_time_series(run_dir: str, metric: str) -> Dict[str, List[pd.DataFrame]]:
    """Extract per‑episode time series for a given metric across runs.

    This is used for plotting learning curves.  The output is a mapping from
    method names to a list of 1‑D arrays (one per seed) where each array has
    length equal to the number of episodes and contains the average value of
    the specified metric per episode.

    Parameters
    ----------
    run_dir:
        Directory containing run CSV files.
    metric:
        Column name in the run logs ('reward', 'correct', 'engagement', etc.).

    Returns
    -------
    dict
        Mapping method -> list of arrays (seeds) -> episodes -> metric values.
    """
    series: Dict[str, List[np.ndarray]] = {}
    for file in os.listdir(run_dir):
        if not file.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(run_dir, file))
        method = df.loc[0, 'method'] if 'method' in df.columns else 'unknown'
        # Group by episode and compute mean metric per episode
        if metric == 'reward':
            ep_means = df.groupby('episode')['reward'].mean().values
        elif metric == 'accuracy':
            ep_means = df.groupby('episode')['correct'].mean().values
        elif metric == 'engagement':
            ep_means = df.groupby('episode')['engagement'].mean().values
        elif metric == 'response_time':
            ep_means = df.groupby('episode')['response_time'].mean().values
        else:
            raise ValueError(f"Unknown metric: {metric}")
        series.setdefault(method, []).append(ep_means)
    return series


def compute_difficulty_histograms(run_dir: str, n_bins: int = 10) -> Dict[str, np.ndarray]:
    """Compute histograms of difficulty choices binned by ability.

    For each run the learner abilities are normalised to [0,1] and binned
    into ``n_bins`` intervals.  Within each bin the number of times each
    difficulty level (0,1,2) was chosen is counted.  The histograms are
    averaged over seeds for each method.

    Parameters
    ----------
    run_dir:
        Directory containing run CSV files.
    n_bins:
        Number of bins for ability.

    Returns
    -------
    dict
        Mapping method -> histogram matrix of shape (n_bins, 3).
    """
    import numpy as np  # local import to avoid global dependency
    hist_data: Dict[str, List[np.ndarray]] = {}
    for file in os.listdir(run_dir):
        if not file.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(run_dir, file))
        method = df.loc[0, 'method'] if 'method' in df.columns else 'unknown'
        # Normalise ability to [0,1]
        abilities = df['ability'].values / 2.0
        # Discretise into bins
        bin_indices = np.clip((abilities * n_bins).astype(int), 0, n_bins - 1)
        hist = np.zeros((n_bins, 3), dtype=int)
        for bin_idx, action in zip(bin_indices, df['action'].values.astype(int)):
            hist[bin_idx, action] += 1
        hist_data.setdefault(method, []).append(hist)
    # Average histograms across seeds
    averaged: Dict[str, np.ndarray] = {}
    for method, hists in hist_data.items():
        avg = np.mean(hists, axis=0)
        averaged[method] = avg
    return averaged
