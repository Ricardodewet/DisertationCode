"""Aggregate run logs into summary metrics.

This script reads all CSV files in a run directory, computes summary
statistics per run (mean reward, accuracy, learning gain, engagement,
response time) and writes the results to ``outputs/tables/run_metrics.csv``.
It is a preparatory step before statistical testing and plotting.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from src.evaluation.metrics import summarise_runs


def main(args: List[str]) -> None:
    parser = argparse.ArgumentParser(description='Aggregate run logs into summary metrics')
    parser.add_argument('--run_dir', type=str, required=True, help='Directory containing run CSV files')
    parser.add_argument('--out', type=str, default='outputs/tables', help='Directory to save summary table')
    parsed = parser.parse_args(args)
    summary = summarise_runs(parsed.run_dir)
    # Save summary
    os.makedirs(parsed.out, exist_ok=True)
    out_path = os.path.join(parsed.out, 'run_metrics.csv')
    summary.to_csv(out_path, index=False)
    print(f'Saved aggregated metrics to {out_path}')


if __name__ == '__main__':
    main(sys.argv[1:])
