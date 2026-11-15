"""Summarise result CSVs with pandas (an optional dependency: ``pip install psoann[analysis]``)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

DEFAULT_METRICS = ("train_accuracy", "test_accuracy", "train_mse", "test_mse", "evals_to_final_best")


def load_results(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def summarise(
    results: pd.DataFrame,
    by: Sequence[str] | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Mean / std / min / max of each metric per group, plus the number of runs.

    ``by`` are column names such as ``"cfg.pso.n_informants"``; default groups by ``experiment``.
    The std is the sample std (ddof=1), so a single run gives NaN.
    """
    by = list(by) if by else ["experiment"]
    missing = [c for c in [*by, *metrics] if c not in results.columns]
    if missing:
        raise KeyError(f"columns not in results: {missing}; available: {list(results.columns)}")
    grouped = results.groupby(by, sort=False, dropna=False)
    table = grouped[list(metrics)].agg(["mean", "std", "min", "max"])
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
    table.insert(0, "n_runs", grouped.size())
    return table.reset_index()
