"""Stratified train/test splitting. All randomness comes from the Generator passed in."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from psoann.config import DataConfig
from psoann.data.scaler import Standardizer


@dataclass(frozen=True)
class DataSplit:
    X_train: NDArray[np.float64]
    y_train: NDArray[np.int64]
    X_test: NDArray[np.float64]
    y_test: NDArray[np.int64]
    train_index: NDArray[np.int64]
    test_index: NDArray[np.int64]


def stratified_split(
    y: NDArray[np.int64], train_fraction: float, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split indices so each class keeps (about) its proportion in both parts."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    train_parts, test_parts = [], []
    for label in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == label))
        n_train = round(train_fraction * len(members))
        if len(members) >= 2:
            n_train = min(max(n_train, 1), len(members) - 1)  # keep >= 1 sample on each side
        train_parts.append(members[:n_train])
        test_parts.append(members[n_train:])
    train = rng.permutation(np.concatenate(train_parts))
    test = rng.permutation(np.concatenate(test_parts))
    return train.astype(np.int64), test.astype(np.int64)


def make_split(X: NDArray[np.float64], y: NDArray[np.int64], config: DataConfig, rng: np.random.Generator) -> DataSplit:
    """Stratified split, then (optionally) standardise using the training rows only."""
    train_idx, test_idx = stratified_split(y, config.train_fraction, rng)
    X_train, X_test = X[train_idx], X[test_idx]
    if config.standardise:
        scaler = Standardizer().fit(X_train)
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    return DataSplit(X_train, y[train_idx], X_test, y[test_idx], train_idx, test_idx)
