from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def simple_cross_validation(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    vcol: str,
    radius: float,
    max_samples: int = 12,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Simple spatial CV using inverse-distance weighting (IDW)."""
    coords = df[[xcol, ycol]].values
    values = df[vcol].values
    tree = cKDTree(coords)
    preds = []

    for i, (x, y) in enumerate(coords):
        dists, idxs = tree.query([x, y], k=max_samples + 1)
        dists = dists[1:]
        idxs = idxs[1:]
        mask = dists <= radius
        if not np.any(mask):
            preds.append(float(np.nanmean(values)))
            continue
        dists = dists[mask]
        idxs = idxs[mask]
        weights = 1.0 / np.maximum(dists, 1e-6)
        preds.append(float(np.sum(weights * values[idxs]) / np.sum(weights)))

    out = df.copy()
    out["pred"] = preds
    errors = out["pred"] - out[vcol]

    metrics = {
        "ME": float(np.mean(errors)),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
    }
    return out, metrics
