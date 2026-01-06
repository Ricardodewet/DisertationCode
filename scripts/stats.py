"""Compute inferential statistics for the adaptive assessment experiment.

This script reads the aggregated run metrics produced by ``evaluate.py``,
computes descriptive statistics per method and runs Welch’s t‑tests and
effect size calculations comparing the PPO agent to each baseline.  The
results are written as CSV files in ``outputs/tables``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd

from src.evaluation.stats import (
    compute_descriptive_stats,
    run_comparisons,
    export_tables,
)


def main(args: List[str]) -> None:
    parser = argparse.ArgumentParser(description='Compute statistics for RL assessment')
    parser.add_argument('--input', type=str, required=True, help='Directory containing run CSV files')
    parser.add_argument('--out', type=str, default='outputs/tables', help='Directory to save statistical tables')
    parser.add_argument('--metrics', type=str, nargs='+', default=['mean_reward', 'accuracy', 'learning_gain', 'mean_engagement'],
                        help='Metrics to analyse')
    parser.add_argument('--reference', type=str, default='ppo', help='Reference method for comparisons')
    parsed = parser.parse_args(args)
    # Aggregate run metrics if necessary
    # If run_metrics.csv exists, load it; otherwise summarise runs in input
    metrics_path = os.path.join(parsed.input, 'run_metrics.csv')
    if os.path.isfile(metrics_path):
        df = pd.read_csv(metrics_path)
    else:
        from src.evaluation.metrics import summarise_runs
        df = summarise_runs(parsed.input)
    # Compute descriptive stats
    descriptive = compute_descriptive_stats(df, parsed.metrics)
    inferential = run_comparisons(df, parsed.metrics, reference=parsed.reference)
    # Save tables
    os.makedirs(parsed.out, exist_ok=True)
    descriptive.to_csv(os.path.join(parsed.out, 'descriptive_stats.csv'))
    inferential.to_csv(os.path.join(parsed.out, 'inferential_stats.csv'))
    print(f'Saved descriptive and inferential tables to {parsed.out}')


if __name__ == '__main__':
    main(sys.argv[1:])
