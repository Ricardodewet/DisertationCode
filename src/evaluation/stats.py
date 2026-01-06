"""Statistical analysis utilities.

This module provides functions to compute descriptive statistics and
run inferential comparisons between the PPO agent and baseline policies.
Welch’s t‑tests are used by default.  Effect sizes (Cohen’s *d*) are
reported to quantify the magnitude of differences.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from scipy import stats


def compute_descriptive_stats(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Compute mean and standard deviation for each method and metric.

    Parameters
    ----------
    df:
        DataFrame returned by ``summarise_runs``.
    metrics:
        List of metric column names to summarise.

    Returns
    -------
    DataFrame
        Table with index = method and columns = metric_mean/metric_std.
    """
    summary = []
    for method, group in df.groupby('method'):
        row = {'method': method}
        for m in metrics:
            row[f'{m}_mean'] = group[m].mean()
            row[f'{m}_std'] = group[m].std(ddof=1)
        summary.append(row)
    return pd.DataFrame(summary).set_index('method')


def welchs_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Perform Welch’s t‑test between two independent samples.

    Returns the t statistic and p‑value.
    """
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)
    return float(t_stat), float(p_val)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen’s *d* effect size between two samples.

    The pooled standard deviation uses the unbiased estimator.
    """
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    s_pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / s_pooled if s_pooled > 0 else 0.0


def run_comparisons(df: pd.DataFrame, metrics: List[str], reference: str = 'ppo') -> pd.DataFrame:
    """Compare each baseline method against a reference method.

    Parameters
    ----------
    df:
        Table with per‑run metrics including 'method' and the metric columns.
    metrics:
        List of metric names to compare.
    reference:
        Name of the reference method (usually 'ppo').

    Returns
    -------
    DataFrame
        Table summarising t statistics, p values and effect sizes for each
        comparison and metric.  Each row corresponds to a method other than
        the reference; columns are multi‑indexed by metric.
    """
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    ref_df = df[df['method'] == reference]
    if ref_df.empty:
        raise ValueError(f"Reference method '{reference}' not found in data")
    for method, group in df.groupby('method'):
        if method == reference:
            continue
        method_results: Dict[str, float] = {}
        for m in metrics:
            t_stat, p_val = welchs_ttest(ref_df[m].values, group[m].values)
            d = cohen_d(ref_df[m].values, group[m].values)
            method_results[f'{m}_t'] = t_stat
            method_results[f'{m}_p'] = p_val
            method_results[f'{m}_d'] = d
        results[method] = method_results
    return pd.DataFrame.from_dict(results, orient='index')


def export_tables(descriptive: pd.DataFrame, inferential: pd.DataFrame, out_dir: str) -> None:
    """Export descriptive and inferential statistics to CSV files.

    Files are named ``descriptive_stats.csv`` and ``inferential_stats.csv``.
    """
    os.makedirs(out_dir, exist_ok=True)
    descriptive.to_csv(os.path.join(out_dir, 'descriptive_stats.csv'))
    inferential.to_csv(os.path.join(out_dir, 'inferential_stats.csv'))
