"""Load a headerless CSV whose last column is an integer class label."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    """Resolve ``path`` as given, then relative to the working directory, then to the repo root."""
    p = Path(path)
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, _REPO_ROOT / p]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"data file {str(path)!r} not found (tried: {tried})")


def load_csv(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return ``(X, y)``: float features of shape (n, d) and labels in ``0..K-1`` of shape (n,)."""
    raw = np.loadtxt(resolve_path(path), delimiter=",", ndmin=2, dtype=np.float64)
    if raw.shape[1] < 2:
        raise ValueError("expected at least one feature column and one label column")
    if not np.all(np.isfinite(raw)):
        raise ValueError("data contains NaN or infinite values")
    X = np.ascontiguousarray(raw[:, :-1])
    labels = raw[:, -1]
    if not np.all(labels == np.round(labels)) or labels.min() < 0:
        raise ValueError("labels (last column) must be non-negative integers")
    y = labels.astype(np.int64)
    if set(np.unique(y)) != set(range(int(y.max()) + 1)):
        raise ValueError("labels must be contiguous integers starting at 0")
    return X, y
