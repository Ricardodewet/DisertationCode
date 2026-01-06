"""Generate plots for RL adaptive assessment experiments.

This script reads run logs from a directory, extracts per‑episode reward,
accuracy and engagement curves and difficulty distributions, and writes
publication‑quality figures into the output directory using Matplotlib.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from src.evaluation.metrics import extract_time_series, compute_difficulty_histograms
from src.utils.plotting import (
    plot_reward_curves,
    plot_metric_over_time,
    plot_difficulty_distribution,
)


def main(args: List[str]) -> None:
    parser = argparse.ArgumentParser(description='Plot results for RL assessment experiments')
    parser.add_argument('--input', type=str, required=True, help='Directory containing run CSV files')
    parser.add_argument('--out', type=str, default='outputs/figures', help='Directory to save plots')
    parsed = parser.parse_args(args)
    run_dir = parsed.input
    out_dir = parsed.out
    # Extract curves
    reward_data = extract_time_series(run_dir, 'reward')
    accuracy_data = extract_time_series(run_dir, 'accuracy')
    engagement_data = extract_time_series(run_dir, 'engagement')
    # Plot reward
    plot_reward_curves(reward_data, out_dir, title='Training Reward')
    # Plot accuracy and engagement
    plot_metric_over_time(accuracy_data, 'Accuracy', out_dir)
    plot_metric_over_time(engagement_data, 'Engagement', out_dir)
    # Difficulty distribution
    difficulty_hist = compute_difficulty_histograms(run_dir, n_bins=10)
    plot_difficulty_distribution(difficulty_hist, out_dir)
    print(f'Saved plots to {out_dir}')


if __name__ == '__main__':
    main(sys.argv[1:])
