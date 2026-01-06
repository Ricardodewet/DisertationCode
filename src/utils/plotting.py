"""Plotting helpers for RL experiment outputs.

All plotting functions use Matplotlib directly to produce publication‑style
figures.  The functions take pre‑processed data and save figures to the
specified directory.  They avoid interactive back‑ends and rely solely on
Python's headless capabilities.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib

# Use a non‑interactive backend suitable for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def plot_reward_curves(curves: Dict[str, np.ndarray], out_dir: str, title: str = "Reward Curve") -> None:
    """Plot reward curves with confidence intervals.

    Parameters
    ----------
    curves:
        Mapping from run identifier to a 2‑D array of shape (n_seeds, n_episodes)
        containing per‑episode rewards.  The mean and standard deviation are
        computed across seeds.
    out_dir:
        Directory where the figure will be saved.  A PNG file named
        ``reward_curve.png`` is created.
    title:
        Title of the plot.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, data in curves.items():
        data = np.asarray(data)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        episodes = np.arange(1, mean.shape[0] + 1)
        ax.plot(episodes, mean, label=label)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'reward_curve.png'))
    plt.close(fig)


def plot_metric_over_time(metric_data: Dict[str, np.ndarray], metric_name: str, out_dir: str) -> None:
    """Plot a metric over episodes for each method.

    Parameters
    ----------
    metric_data:
        Mapping from method name to a 2‑D array of shape (n_seeds, n_episodes)
        containing the metric values per episode.
    metric_name:
        Human‑readable name for the metric (used in labels).
    out_dir:
        Directory where the figure will be saved.  A PNG file named
        ``{metric_name.lower()}_curve.png`` is created.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, data in metric_data.items():
        data = np.asarray(data)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        episodes = np.arange(1, mean.shape[0] + 1)
        ax.plot(episodes, mean, label=label)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Over Time')
    ax.legend()
    fig.tight_layout()
    filename = metric_name.lower().replace(' ', '_') + '_curve.png'
    fig.savefig(os.path.join(out_dir, filename))
    plt.close(fig)


def plot_difficulty_distribution(distributions: Dict[str, np.ndarray], out_dir: str) -> None:
    """Plot the distribution of chosen difficulties across ability bins.

    Parameters
    ----------
    distributions:
        Mapping from method name to a 2‑D histogram array of shape
        (n_bins_ability, n_actions).  Rows correspond to ability bins and
        columns to actions (Easy, Medium, Hard).  Values are counts.
    out_dir:
        Directory where the figure will be saved.  The file is named
        ``difficulty_distribution.png``.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(distributions), figsize=(5 * len(distributions), 4), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for ax, (label, hist) in zip(axes, distributions.items()):
        hist = np.asarray(hist)
        bins = np.arange(hist.shape[0] + 1)
        for action_idx, action_name in enumerate(['Easy', 'Medium', 'Hard']):
            ax.bar(bins[:-1] + action_idx * 0.25, hist[:, action_idx], width=0.25, label=action_name)
        ax.set_xticks(bins[:-1] + 0.25)
        ax.set_xticklabels([f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(hist.shape[0])], rotation=45)
        ax.set_xlabel('Normalised Ability Bin')
        ax.set_ylabel('Count')
        ax.set_title(f'Difficulty Distribution – {label}')
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'difficulty_distribution.png'))
    plt.close(fig)
