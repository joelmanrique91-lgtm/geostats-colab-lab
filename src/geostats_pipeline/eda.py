from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def basic_stats(series: pd.Series, weights: pd.Series | None = None) -> Dict[str, float]:
    values = pd.to_numeric(series, errors="coerce")
    if weights is None:
        return {
            "count": int(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = values.notna() & weights.gt(0)
    values = values[mask]
    weights = weights[mask]
    if values.empty:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    wsum = weights.sum()
    mean = float((values * weights).sum() / wsum) if wsum > 0 else float(values.mean())
    variance = float(((values - mean) ** 2 * weights).sum() / wsum) if wsum > 0 else float(values.var(ddof=1))
    return {
        "count": int(values.count()),
        "mean": mean,
        "std": float(np.sqrt(variance)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def plot_histogram(series: pd.Series, path: str, title: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(series.dropna(), bins=30, color="#4c78a8", alpha=0.8)
    ax.set_xlabel(series.name or "value")
    ax.set_ylabel("count")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_xy_scatter(df: pd.DataFrame, xcol: str, ycol: str, vcol: str, path: str, color_by: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if color_by and color_by in df.columns:
        sc = ax.scatter(df[xcol], df[ycol], c=df[color_by], cmap="viridis", s=12, alpha=0.8)
        fig.colorbar(sc, ax=ax, label=color_by)
    else:
        sc = ax.scatter(df[xcol], df[ycol], c=df[vcol], cmap="viridis", s=12, alpha=0.8)
        fig.colorbar(sc, ax=ax, label=vcol)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title("Spatial scatter")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
