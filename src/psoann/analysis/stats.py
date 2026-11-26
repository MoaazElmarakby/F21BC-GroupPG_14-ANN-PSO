"""Statistical comparison of settings: Kruskal-Wallis, Holm-corrected Mann-Whitney U and the A12 effect size.

Run results are not normally distributed (accuracy saturates near 1.0 and ties are common), so every
test here is non-parametric. Runs use the same seeds across settings, but the tests treat settings as
independent samples, which is conservative.

    kruskal_wallis(results, by, metric)            one omnibus test over all groups
    pairwise_tests(results, by, metric, control)   Mann-Whitney U per pair, Holm-adjusted p, A12 effect size

A12 (Vargha and Delaney, 2000) is the probability that a random run of group A beats a random run of group
B (ties count half): 0.5 means no effect; |A12 - 0.5| >= 0.06, 0.14, 0.21 is a small, medium, large effect.
Requires scipy (``pip install psoann[analysis]``).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def a12(x: Sequence[float], y: Sequence[float]) -> float:
    """P(X > Y) + 0.5 P(X = Y) for a random draw from each sample."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError("A12 needs two non-empty samples")
    greater = (x[:, None] > y[None, :]).sum()
    equal = (x[:, None] == y[None, :]).sum()
    return float((greater + 0.5 * equal) / (x.size * y.size))


def effect_label(a: float) -> str:
    """Conventional magnitude of an A12 value (Vargha and Delaney thresholds)."""
    distance = abs(a - 0.5)
    if distance >= 0.21:
        return "large"
    if distance >= 0.14:
        return "medium"
    if distance >= 0.06:
        return "small"
    return "negligible"


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _groups(results: pd.DataFrame, by: str, metric: str) -> dict[str, np.ndarray]:
    for column in (by, metric):
        if column not in results.columns:
            raise KeyError(f"column {column!r} not in results; available: {list(results.columns)}")
    clean = results.dropna(subset=[metric])
    return {str(key): frame[metric].to_numpy(dtype=float) for key, frame in clean.groupby(by, sort=False)}


def kruskal_wallis(results: pd.DataFrame, by: str, metric: str) -> dict[str, float]:
    """Omnibus test: do the groups defined by column ``by`` differ in ``metric``? (p = 1 if all values are equal)"""
    groups = _groups(results, by, metric)
    if len(groups) < 2:
        raise ValueError("Kruskal-Wallis needs at least two groups")
    values = np.concatenate(list(groups.values()))
    if np.all(values == values[0]):
        return {"groups": float(len(groups)), "n": float(values.size), "H": 0.0, "p": 1.0}
    h, p = stats.kruskal(*groups.values())
    return {"groups": float(len(groups)), "n": float(values.size), "H": float(h), "p": float(p)}


def pairwise_tests(results: pd.DataFrame, by: str, metric: str, control: str | None = None) -> pd.DataFrame:
    """Two-sided Mann-Whitney U per pair of groups (or ``control`` against every other group).

    Columns: group_a, group_b, n_a, n_b, mean_a, mean_b, U, p, p_holm, a12 (A beats B), effect.
    """
    groups = _groups(results, by, metric)
    if len(groups) < 2:
        raise ValueError("pairwise tests need at least two groups")
    if control is not None:
        if control not in groups:
            raise KeyError(f"control group {control!r} not found; groups: {list(groups)}")
        pairs = [(control, other) for other in groups if other != control]
    else:
        pairs = list(combinations(groups, 2))
    rows = []
    for a, b in pairs:
        x, y = groups[a], groups[b]
        if np.all(x == x[0]) and np.all(y == x[0]):
            u, p = float(x.size * y.size / 2), 1.0  # identical constant samples: no evidence of a difference
        else:
            result = stats.mannwhitneyu(x, y, alternative="two-sided")
            u, p = float(result.statistic), float(result.pvalue)
        effect = a12(x, y)
        rows.append(
            {
                "group_a": a,
                "group_b": b,
                "n_a": x.size,
                "n_b": y.size,
                "mean_a": x.mean(),
                "mean_b": y.mean(),
                "U": u,
                "p": p,
                "a12": effect,
                "effect": effect_label(effect),
            }
        )
    table = pd.DataFrame(rows)
    table.insert(table.columns.get_loc("p") + 1, "p_holm", holm(table["p"].to_numpy()))
    return table
