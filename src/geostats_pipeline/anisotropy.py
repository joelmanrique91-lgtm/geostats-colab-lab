from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .variography import VariogramExperimental, experimental_variogram


def directional_variogram(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    vcol: str,
    azimuth: float,
    n_lags: int,
    lag_size: float,
    max_pairs: int,
) -> VariogramExperimental:
    coords = df[[xcol, ycol]].to_numpy(dtype=float)
    values = pd.to_numeric(df[vcol], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    coords = coords[valid]
    values = values[valid]

    if len(values) < 2:
        return VariogramExperimental(np.array([]), np.array([]), np.array([]))

    theta = np.deg2rad(azimuth)
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    rotated = coords @ rot.T

    df_rot = pd.DataFrame({"x": rotated[:, 0], "y": rotated[:, 1], "value": values})
    return experimental_variogram(df_rot, "x", "y", None, "value", n_lags, lag_size, max_pairs)


def anisotropy_sweep(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    vcol: str,
    azimuths: Iterable[float],
    n_lags: int,
    lag_size: float,
    max_pairs: int,
) -> List[Dict[str, float]]:
    results = []
    for azm in azimuths:
        exp = directional_variogram(df, xcol, ycol, vcol, azm, n_lags, lag_size, max_pairs)
        if exp.gamma.size:
            sill_proxy = float(np.nanmean(exp.gamma))
        else:
            sill_proxy = float("nan")
        results.append({"azimuth": float(azm), "sill_proxy": sill_proxy})
    return results
