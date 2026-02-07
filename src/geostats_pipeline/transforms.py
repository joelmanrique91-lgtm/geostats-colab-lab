from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NormalScoreTransform:
    values: np.ndarray
    scores: np.ndarray

    def transform(self, data: Iterable[float]) -> np.ndarray:
        data_arr = np.asarray(list(data), dtype=float)
        return np.interp(data_arr, self.values, self.scores)

    def back_transform(self, scores: Iterable[float]) -> np.ndarray:
        scores_arr = np.asarray(list(scores), dtype=float)
        return np.interp(scores_arr, self.scores, self.values)


def normal_score_transform(series: pd.Series) -> NormalScoreTransform:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("Normal-score transform requires at least one finite value.")
    sorted_vals = np.sort(values)
    ranks = np.arange(1, len(sorted_vals) + 1, dtype=float)
    probs = (ranks - 0.5) / len(sorted_vals)
    scores = np.sqrt(2.0) * np.erfinv(2.0 * probs - 1.0)
    return NormalScoreTransform(values=sorted_vals, scores=scores)


def indicator_transform(series: pd.Series, thresholds: Iterable[float]) -> Tuple[pd.DataFrame, list[float]]:
    values = pd.to_numeric(series, errors="coerce")
    thresholds_sorted = sorted(float(t) for t in thresholds)
    data = {}
    for thr in thresholds_sorted:
        data[f"indicator_{thr:g}"] = (values > thr).astype(int)
    return pd.DataFrame(data, index=series.index), thresholds_sorted
