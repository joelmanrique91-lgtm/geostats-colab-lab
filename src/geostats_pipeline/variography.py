from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class VariogramExperimental:
    lags: np.ndarray
    gamma: np.ndarray
    pairs: np.ndarray


def _pairwise_distances(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = coords.shape[0]
    dists = []
    pairs = []
    for i in range(n - 1):
        diff = coords[i + 1 :] - coords[i]
        dist = np.sqrt(np.sum(diff**2, axis=1))
        dists.append(dist)
        pairs.append(np.full(dist.shape, i))
    return np.concatenate(dists), np.concatenate(pairs)


def experimental_variogram(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    zcol: str | None,
    vcol: str,
    n_lags: int,
    lag_size: float,
    max_pairs: int,
) -> VariogramExperimental:
    coords = df[[xcol, ycol]].to_numpy(dtype=float)
    if zcol and zcol in df.columns:
        coords = df[[xcol, ycol, zcol]].to_numpy(dtype=float)
    values = pd.to_numeric(df[vcol], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(values)
    coords = coords[valid]
    values = values[valid]

    if len(values) < 2:
        return VariogramExperimental(np.array([]), np.array([]), np.array([]))

    dists, _ = _pairwise_distances(coords)
    diffs = []
    for i in range(len(values) - 1):
        diff = (values[i + 1 :] - values[i]) ** 2
        diffs.append(diff)
    diffs = np.concatenate(diffs)

    if len(dists) > max_pairs:
        idx = np.random.default_rng(42).choice(len(dists), size=max_pairs, replace=False)
        dists = dists[idx]
        diffs = diffs[idx]

    bins = np.arange(0, n_lags + 1) * lag_size
    gamma = np.zeros(n_lags)
    pairs = np.zeros(n_lags, dtype=int)

    for i in range(n_lags):
        mask = (dists >= bins[i]) & (dists < bins[i + 1])
        pairs[i] = int(mask.sum())
        gamma[i] = 0.5 * np.mean(diffs[mask]) if pairs[i] > 0 else np.nan

    lags = bins[:-1] + 0.5 * lag_size
    return VariogramExperimental(lags=lags, gamma=gamma, pairs=pairs)


def fit_variogram_model(exp: VariogramExperimental, model_type: str, sill: float, rng: float, nugget: float) -> Dict[str, float]:
    return {
        "type": model_type,
        "nug": float(nugget),
        "cc1": float(max(sill - nugget, 0.0)),
        "hmaj1": float(rng),
        "it1": 1 if model_type == "spherical" else 2,
        "cc2": 0.0,
        "hmaj2": 0.0,
        "it2": 1,
    }


def plot_variogram(exp: VariogramExperimental, model: Dict[str, float], path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(exp.lags, exp.gamma, "o", label="experimental")
    if exp.lags.size:
        h = np.linspace(0, exp.lags.max(), 100)
        nug = model.get("nug", 0.0)
        cc1 = model.get("cc1", 0.0)
        hmaj1 = model.get("hmaj1", 1.0)
        hr = np.clip(h / hmaj1, 0.0, 1.0)
        gamma = nug + cc1 * (1.5 * hr - 0.5 * hr**3)
        ax.plot(h, gamma, "-", label="model")
    ax.set_xlabel("lag")
    ax.set_ylabel("semivariance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
