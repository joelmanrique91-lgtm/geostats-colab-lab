from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrendModel:
    order: int
    coef: np.ndarray
    columns: Tuple[str, ...]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        design = build_design_matrix(df, self.columns, self.order)
        return design @ self.coef


def build_design_matrix(df: pd.DataFrame, columns: Iterable[str], order: int) -> np.ndarray:
    cols = tuple(columns)
    coords = [pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in cols]
    valid = np.all(np.isfinite(np.column_stack(coords)), axis=1)
    x = np.column_stack(coords)
    ones = np.ones((x.shape[0], 1))
    if order <= 1:
        design = np.column_stack([ones, x])
    else:
        x1, x2, x3 = x.T if x.shape[1] == 3 else (x.T[0], x.T[1], np.zeros_like(x.T[0]))
        design = np.column_stack(
            [
                ones[:, 0],
                x1,
                x2,
                x3,
                x1**2,
                x2**2,
                x3**2,
                x1 * x2,
                x1 * x3,
                x2 * x3,
            ]
        )
    design[~valid] = np.nan
    return design


def fit_trend(
    df: pd.DataFrame,
    value_col: str,
    columns: Iterable[str],
    order: int = 1,
) -> Tuple[TrendModel, float]:
    cols = tuple(columns)
    design = build_design_matrix(df, cols, order)
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values) & np.all(np.isfinite(design), axis=1)
    if not np.any(mask):
        raise ValueError("No valid samples to fit trend.")
    design_valid = design[mask]
    values_valid = values[mask]
    coef, *_ = np.linalg.lstsq(design_valid, values_valid, rcond=None)
    pred = design_valid @ coef
    ss_res = float(np.sum((values_valid - pred) ** 2))
    ss_tot = float(np.sum((values_valid - values_valid.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return TrendModel(order=order, coef=coef, columns=cols), r2
