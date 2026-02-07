from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_topcut(df: pd.DataFrame, col: str, high: float) -> pd.DataFrame:
    """Clip values above a high threshold."""
    if col not in df.columns:
        raise KeyError(f"Column not found: {col}")
    df = df.copy()
    df[col] = df[col].clip(upper=high)
    return df


def basic_stats(series: pd.Series) -> Dict[str, float]:
    """Basic descriptive statistics."""
    s = series.dropna()
    return {
        "count": float(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        "min": float(s.min()) if s.shape[0] > 0 else 0.0,
        "p10": float(np.percentile(s, 10)) if s.shape[0] > 0 else 0.0,
        "p50": float(np.percentile(s, 50)) if s.shape[0] > 0 else 0.0,
        "p90": float(np.percentile(s, 90)) if s.shape[0] > 0 else 0.0,
        "max": float(s.max()) if s.shape[0] > 0 else 0.0,
    }


def plot_hist(series: pd.Series, outpath: str) -> None:
    """Histogram plot."""
    plt.figure(figsize=(6, 4))
    series.dropna().hist(bins=30, edgecolor="black")
    plt.xlabel(series.name or "value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_qq(series: pd.Series, outpath: str) -> None:
    """QQ plot against normal distribution."""
    from scipy import stats

    plt.figure(figsize=(5, 5))
    stats.probplot(series.dropna(), dist="norm", plot=plt)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_xy_scatter(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    vcol: str,
    outpath: str,
    color_by: str | None = None,
) -> None:
    """Scatter plot of XY colored by a variable or category."""
    plt.figure(figsize=(6, 5))
    if color_by and color_by in df.columns:
        c = df[color_by]
        if pd.api.types.is_numeric_dtype(c):
            plt.scatter(df[xcol], df[ycol], c=c, s=15, cmap="viridis", alpha=0.8)
            plt.colorbar(label=color_by)
        else:
            cats = pd.Categorical(c)
            sc = plt.scatter(df[xcol], df[ycol], c=cats.codes, s=15, cmap="tab20", alpha=0.8)
            cbar = plt.colorbar(sc, ticks=range(len(cats.categories)))
            cbar.ax.set_yticklabels([str(v) for v in cats.categories])
    else:
        plt.scatter(df[xcol], df[ycol], c=df[vcol], s=15, cmap="viridis", alpha=0.8)
        plt.colorbar(label=vcol)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
