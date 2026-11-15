"""Z-score standardisation fitted on training data only (decision D6: no test-set leakage)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Standardizer:
    def __init__(self) -> None:
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None

    def fit(self, X: NDArray[np.float64]) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std > 0.0, std, 1.0)  # constant columns map to 0 instead of NaN
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fitted before transform")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.fit(X).transform(X)
