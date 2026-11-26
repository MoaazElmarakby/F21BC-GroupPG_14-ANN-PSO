"""Plots (matplotlib, headless backend). Optional dependency: ``pip install psoann[analysis]``."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_convergence(
    histories: Sequence[Mapping[str, Any]],
    out_path: str | Path,
    key: str = "best",
    group_by: str | None = None,
) -> Path:
    """Mean +/- std of ``key`` against evaluations, one curve per group (default: per experiment/point)."""
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    evaluations: dict[str, np.ndarray] = {}
    for record in histories:
        label = (
            str(record.get(group_by, "")) if group_by else str(record.get("point") or record.get("experiment", "run"))
        )
        values = np.asarray(record["history"][key], dtype=float)
        groups[label].append(values)
        evaluations.setdefault(label, np.asarray(record["history"]["evaluations"], dtype=float))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, curves in groups.items():
        stack = np.vstack(curves)
        mean, std = stack.mean(axis=0), stack.std(axis=0)
        ax.plot(evaluations[label], mean, label=f"{label} (n={len(curves)})")
        ax.fill_between(evaluations[label], mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("fitness evaluations")
    ax.set_ylabel(
        {
            "diversity": "swarm diversity (mean per-dimension std)",
            "boundary_hits": "coordinates outside the bounds",
        }.get(key, f"{key} fitness")
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_sweep(results: pd.DataFrame, x: str, y: str, out_path: str | Path, hue: str | None = None) -> Path:
    """Mean of ``y`` against ``x`` with +/- std error bars (optionally one line per ``hue`` value)."""
    for column in (x, y, *([hue] if hue else [])):
        if column not in results.columns:
            raise KeyError(f"column {column!r} not in results; available: {list(results.columns)}")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    numeric_x = pd.api.types.is_numeric_dtype(results[x])
    for label, frame in results.groupby(hue, sort=False) if hue else [(None, results)]:
        stats = frame.groupby(x, sort=numeric_x)[y].agg(["mean", "std"]).fillna(0.0)
        positions = stats.index.to_numpy() if numeric_x else np.arange(len(stats))
        ax.errorbar(
            positions,
            stats["mean"],
            yerr=stats["std"],
            marker="o",
            capsize=3,
            label=None if label is None else f"{hue}={label}",
        )
        if not numeric_x:
            ax.set_xticks(positions, [str(v) for v in stats.index])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.3)
    if hue:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_boxplot(
    results: pd.DataFrame, by: str, metric: str, out_path: str | Path, order: Sequence[str] | None = None
) -> Path:
    """One box per group (e.g. ``by="point"``) showing the spread of ``metric`` over the repeated runs."""
    for column in (by, metric):
        if column not in results.columns:
            raise KeyError(f"column {column!r} not in results; available: {list(results.columns)}")
    clean = results.dropna(subset=[metric])
    labels = list(order) if order else [str(k) for k in clean[by].drop_duplicates()]
    data = [clean.loc[clean[by].astype(str) == label, metric].to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(labels) + 2.0), 4.5))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_xlabel(by)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelrotation=60)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_heatmap(results: pd.DataFrame, x: str, y: str, value: str, out_path: str | Path) -> Path:
    """Mean of ``value`` over a two-factor grid (rows ``y``, columns ``x``), annotated in each cell."""
    for column in (x, y, value):
        if column not in results.columns:
            raise KeyError(f"column {column!r} not in results; available: {list(results.columns)}")
    table = results.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * table.shape[1] + 2.5), max(4.0, 0.6 * table.shape[0] + 1.5)))
    image = ax.imshow(table.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns])
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index])
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if not np.isnan(table.iat[i, j]):
                ax.text(j, i, f"{table.iat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.colorbar(image, ax=ax, label=f"mean {value}")
    fig.tight_layout()
    return _save(fig, out_path)


def _save(fig: plt.Figure, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
